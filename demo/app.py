import gradio as gr
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
#from conditional_mnist_diffusion_flow import DiffusionModel, FlowModel
# conditional_mnist_diffusion_flow.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import gradio as gr
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import cv2


def resize(image):

    stretch_near = cv2.resize(image, (200, 200), 
               interpolation = cv2.INTER_LINEAR)
    return stretch_near

# --- Modelos ---

class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_dim = 28 * 28  # 784
        self.t_dim = 1          # scalar
        self.label_dim = 10     # one-hot

        self.net = nn.Sequential(
            nn.Linear(28*28 + 1 + 10, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 28*28)
        )

    def forward(self, x, t, y):
        B = x.size(0)
        x_flat = x.view(B, -1)  # (B, 784)
        t_embed = t.view(B, 1).float()  # (B, 1)
        y_onehot = F.one_hot(y, num_classes=10).float()  # (B, 10)

        x_cat = torch.cat([x_flat, t_embed, y_onehot], dim=1)  # (B, 795)
        out = self.net(x_cat)  # (B, 784)
        return out.view(B, 1, 28, 28)  # reshape back to image


class SimpleConditionalUnet(nn.Module):
    def __init__(self, time_emb_dim=64, label_emb_dim=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.label_embed = nn.Embedding(10, label_emb_dim)

        self.enc1 = nn.Conv2d(1, 32, 3, padding=1)  # 28x28 → 28x28
        self.enc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 28x28 → 14x14
        self.enc3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 14x14 → 7x7

        self.middle = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1)
        )

        self.dec3 = nn.ConvTranspose2d(128 + time_emb_dim, 64, 4, stride=2, padding=1)  # 7x7 → 14x14
        self.dec2 = nn.ConvTranspose2d(64 + 64, 32, 4, stride=2, padding=1)  # 14x14 → 28x28
        self.dec1 = nn.Conv2d(32 + 32, 1, 3, padding=1)  # 28x28

    def forward(self, x, t, y):
        t_embed = self.time_embed(t.view(-1, 1).float())  # (B, 64)
        y_embed = self.label_embed(y)  # (B, 64)
        cond = t_embed+y_embed
        #cond = torch.cat([t_embed, y_embed], dim=1)  # (B, 128)
        cond = cond.view(x.size(0), -1, 1, 1).expand(-1, -1, 7, 7)

        # encoder
        e1 = F.relu(self.enc1(x))     # 28x28
        e2 = F.relu(self.enc2(e1))    # 14x14
        e3 = F.relu(self.enc3(e2))    # 7x7

        # middle
        m = self.middle(e3)

        # decoder
        d3 = F.relu(self.dec3(torch.cat([m, cond], dim=1)))  # 14x14
        d2 = F.relu(self.dec2(torch.cat([d3, e2], dim=1)))   # 28x28
        d1 = self.dec1(torch.cat([d2, e1], dim=1))           # 28x28

        return d1


class ConditionalUNet(nn.Module):
    def __init__(self, cond_dim=11):
        super().__init__()
        self.cond_dim = cond_dim

        # Encoder
        self.enc1 = nn.Conv2d(1, 32, 3, padding=1)     # input: x only
        self.enc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        self.middle = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1)
        )

        # Decoder (recibe condición aquí)
        self.dec3 = nn.ConvTranspose2d(128 + cond_dim, 64, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(64 + 64, 32, 4, stride=2, padding=1)
        self.dec1 = nn.Conv2d(32 + 32, 1, 3, padding=1)

    def forward(self, x, t, y):
        B = x.size(0)

        # Encode solo imagen
        e1 = F.relu(self.enc1(x))       # (B, 32, 28, 28)
        e2 = F.relu(self.enc2(e1))      # (B, 64, 14, 14)
        e3 = F.relu(self.enc3(e2))      # (B, 128, 7, 7)

        m = self.middle(e3)             # (B, 128, 7, 7)

        # Condición expandida a spatial map
        t = t.view(B, 1).float()
        y_onehot = F.one_hot(y, num_classes=10).float()
        cond = torch.cat([t, y_onehot], dim=1)               # (B, 11)
        cond_map = cond.view(B, self.cond_dim, 1, 1).expand(-1, -1, 7, 7)  # (B, 11, 7, 7)

        # Decoder usa condición
        d3 = F.relu(self.dec3(torch.cat([m, cond_map], dim=1)))      # (B, 64, 14, 14)
        d2 = F.relu(self.dec2(torch.cat([d3, e2], dim=1)))           # (B, 32, 28, 28)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))                   # (B, 1, 28, 28)

        return d1


class FlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28 + 1 + 10, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 28*28)
        )

    def forward(self, x, t, y):
        x_flat = x.view(x.size(0), -1)
        t_embed = t.unsqueeze(1).float() if t.ndim == 1 else t
        y_onehot = F.one_hot(y, num_classes=10).float()
        x_cat = torch.cat([x_flat, t_embed, y_onehot], dim=1)
        return self.net(x_cat).view(x.size())
class FlowUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
        )
        self.label_embed = nn.Sequential(
            nn.Linear(1, 32),
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(32, 64, 3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x, t, y):
        B = x.size(0)
        t_embed = self.time_embed(t.view(B, 1).float())
        y_embed = self.label_embed(y.view(B, 1).float())
        cond = t_embed + y_embed
        cond_map = cond.view(B, 32, 1, 1)#.expand(-1, 128, 1, 1)
        h = self.encoder(x)  # -> (B, 128, 14, 14)
        h_map = h+ cond_map
        out =   self.decoder(h_map) #[128, 1, 28, 28])
        return out

class FlowModelTrans(nn.Module):
    def __init__(self, num_classes=10, dim=128, depth=4, heads=4):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, dim)
        self.t_embed = nn.Linear(1, dim)
        self.input_proj = nn.Linear(28 * 28, dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads), num_layers=depth
        )
        self.output_proj = nn.Linear(dim, 28 * 28)

    def forward(self, x, noise_level, labels):
        B = x.size(0)
        x_flat = x.view(B, -1)  # (B, 784)
        x_embed = self.input_proj(x_flat).unsqueeze(0)  # (1, B, dim)

        label_embed = self.label_embedding(labels)      # (B, dim)
        t_embed = self.t_embed(noise_level.view(B, 1))  # (B, dim)
        cond = label_embed + t_embed
        cond = cond.unsqueeze(0)  # (1, B, dim)

        x_cond = x_embed + cond  # broadcasting (1, B, dim)
        transformed = self.transformer(x_cond)  # (1, B, dim)
        out = self.output_proj(transformed.squeeze(0))  # (B, 784)
        return out.view(B, 1, 28, 28)
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# --- Utilidades ---
def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, dtype=torch.float32) * torch.log(torch.tensor(10000.0)) / half)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# --- Bloque residual con condición ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.emb_proj = nn.Linear(emb_channels, out_channels)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h += self.emb_proj(F.silu(emb))[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


# --- UNet condicional ---
class ConditionalUNet(nn.Module):
    def __init__(self, num_classes=10, base_channels=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, base_channels),
        )
        self.label_emb = nn.Embedding(num_classes, base_channels)

        self.enc1 = ResidualBlock(1, base_channels, base_channels)
        self.enc2 = ResidualBlock(base_channels, base_channels * 2, base_channels)
        self.down = nn.Conv2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1)

        self.mid = ResidualBlock(base_channels * 2, base_channels * 2, base_channels)

        self.up = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1)
        self.dec2 = ResidualBlock(base_channels * 4, base_channels, base_channels)
        self.dec1 = ResidualBlock(base_channels * 2, base_channels, base_channels)

        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, x, t, y):
        emb_t = self.time_mlp(t.view(-1, 1))
        emb_y = self.label_emb(y)
        emb = emb_t + emb_y

        x1 = self.enc1(x, emb)
        x2 = self.enc2(x1, emb)
        x3 = self.down(x2)
        m = self.mid(x3, emb)
        u = self.up(m)

        d2 = self.dec2(torch.cat([u, x2], dim=1), emb)
        d1 = self.dec1(torch.cat([d2, x1], dim=1), emb)

        out = self.out_conv(F.silu(self.out_norm(d1)))
        return out
        
# --- Cargar modelos ---
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
timesteps = 100
img_shape = (1, 28, 28)

@torch.no_grad()
def generate_diffusion_intermediates(label):
    model = ConditionalUNet().to(device)
    #model = SimpleConditionalUnet().to(device)
    model.load_state_dict(torch.load("outputs/diffusion/diffusion_model2.pth", map_location=device))
    model.eval()

    x = torch.randn(1, *img_shape).to(device)
    y = torch.full((1,), label, dtype=torch.long, device=device)
    betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alphas = 1 - betas
    alpha_hat = torch.cumprod(alphas, dim=0)

    #x = (x + 1) / 2.0  # Map from [-1,1] to [0,1]
    #x = x.clamp(0, 1)

    images = [(x + 1) / 2.0]  # initial noise
    print("diffusion:")
    print("x:",x.min(),x.max())
    print("images:",images[0].min(),images[0].max())
    for t in reversed(range(timesteps)):
        if t in [25,50,75]:
            print("append","x:",x.min(),x.max())
            images.append((x + 1) / 2.0)
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)
        eps_pred = model(x, t_batch.float(), y)
        alpha_t = alphas[t]
        alpha_bar_t = alpha_hat[t]
        x0_pred = (x - (1 - alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = alpha_t.sqrt() * x0_pred + (1 - alpha_t).sqrt() * noise
        x = x.clamp(-1, 1)

        
    images.append((x + 1) / 2.0)
    print(images[0].shape)
    print(images[0][0].shape)
    print(images[0][0,:,:].shape)
    print(images[0][0].clamp(0, 1).cpu().numpy().shape)
    print(images[0][0,:,:].clamp(0, 1).cpu().numpy().shape)
    print(images[0][0][0].clamp(0, 1).cpu().numpy().shape)
    
    return [resize(images[0][0][0].clamp(0, 1).cpu().numpy())]+[resize(img[0][0].clamp(0, 1).cpu().numpy()) for img in images[-5:]]


def generate_localized_noise(shape, radius=5):
    """Genera una imagen con ruido solo en un círculo en el centro."""
    B, C, H, W = shape
    assert C == 1, "Solo imágenes en escala de grises."

    # Crear máscara circular
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    center_y, center_x = H // 2, W // 2
    mask = ((yy - center_y)**2 + (xx - center_x)**2) >= radius**2
    mask = mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Aplicar máscara a ruido
    noise = torch.randn(B, C, H, W)
    localized_noise = noise * mask + -1*(1-mask)  # solo hay ruido dentro del círculo
    return localized_noise


@torch.no_grad()
def generate_flow_intermediates(label):
    model = ConditionalUNet().to(device)
    #model.load_state_dict(torch.load("outputs/flow_matching/flow_model.pth", map_location=device))
    #model = FlowUNet().to(device)
    model.load_state_dict(torch.load("outputs/flow_matching/flow_model2_unet.pth", map_location=device))
    
    model.eval()

    x = torch.randn(1, *img_shape).to(device)
    #x = generate_localized_noise((1, 1, 28, 28), radius=12).to(device)
    y = torch.full((1,), label, dtype=torch.long, device=device)
    steps = 50
    dt = 1.0 / steps

    #x = (x + 1) / 2.0  # Map from [-1,1] to [0,1]
    #x = x.clamp(0, 1)
    
    images = [(x + 1) / 2.0]  # initial noise

    print("flow:")
    print("x:",x.min(),x.max())
    print("images:",images[0].min(),images[0].max())
    
    for i in range(steps):
        if i in [12,24,36]:
            print("append","x:",x.min(),x.max())
            images.append((x + 1) / 2.0)
        t = torch.full((1,), i * dt, device=device)
        v = model(x, t, y)
        x = x + v * dt
    images.append((x + 1) / 2.0)
    return [resize(images[0][0][0].clamp(0, 1).cpu().numpy())]+[resize(img[0][0].clamp(0, 1).cpu().numpy()) for img in images[-5:]]

with gr.Blocks() as demo:
    gr.Markdown("# Conditional MNIST Generation: Diffusion vs Flow Matching")

    with gr.Tab("Diffusion"):
        label_d = gr.Slider(0, 9, step=1, label="Digit Label")
        btn_d = gr.Button("Generate")
        with gr.Row():
            outs_d = [
                gr.Image(label="Noise"),
                gr.Image(label="Step 1"),
                gr.Image(label="Step 2"),
                gr.Image(label="Step 3"),
                gr.Image(label="Step 4"),
                gr.Image(label="Final")
            ]

        btn_d.click(fn=generate_diffusion_intermediates, inputs=label_d, outputs=outs_d)

    with gr.Tab("Flow Matching"):
        label_f = gr.Slider(0, 9, step=1, label="Digit Label")
        btn_f = gr.Button("Generate")
        with gr.Row():
            outs_f = [
                gr.Image(label="Noise"),
                gr.Image(label="Step 1"),
                gr.Image(label="Step 2"),
                gr.Image(label="Step 3"),
                gr.Image(label="Step 4"),
                gr.Image(label="Final")
            ]
    
        btn_f.click(fn=generate_flow_intermediates, inputs=label_f, outputs=outs_f)

#demo.launch()
demo.launch(share=True, server_port=9070)
