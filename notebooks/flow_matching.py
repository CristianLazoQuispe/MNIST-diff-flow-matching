# conditional_mnist_diffusion_flow.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import os
print(os.getcwd())
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.conditional_flow_matching import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial



# --- Configuración ---
device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
batch_size = 128
timesteps = 100
img_shape = (1, 28, 28)
os.makedirs("outputs/diffusion", exist_ok=True)
os.makedirs("outputs/flow_matching", exist_ok=True)
os.makedirs("outputs/diffusion/images/", exist_ok=True)
os.makedirs("outputs/flow_matching/images/", exist_ok=True)

# --- Dataset ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)
])
dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



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


# --- Entrenamiento Flow Matching ---
def train_flow(epochs=1000,save_imgs=True,model_name="flow_model"):
    #model = nn.DataParallel(FlowModel(), device_ids=[0,1,2,3,4,5]).to(device)
    
    model = ConditionalUNet().to(device)
    #model = FlowUNet().to(device)
    FM = ConditionalFlowMatcher(sigma=0.0)

    opt = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"[Flow {epoch}]", leave=True, ncols=80)
        for x_real, y in pbar:
            x_real = x_real.to(device)
            y = y.to(device)
            x_noise = torch.randn_like(x_real)
            #t = torch.rand(x_real.size(0), device=device)
            #x_t = (1 - t.view(-1, 1, 1, 1)) * x_noise + t.view(-1, 1, 1, 1) * x_real
            #v_target = x_real - x_noise
            t, x_t, v_target = FM.sample_location_and_conditional_flow(x_noise, x_real)
            v_pred = model(x_t, t, y)
            mse = torch.mean((v_pred - v_target) ** 2)
            norm_pred = F.log_softmax(v_pred.view(v_pred.size(0), -1), dim=1)
            norm_true = F.softmax(v_target.view(v_target.size(0), -1), dim=1)
            kl = F.kl_div(norm_pred, norm_true, reduction='batchmean')
            loss = mse# + 0.1 * kl

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        torch.save(model.state_dict(), f"outputs/flow_matching/{model_name}.pth")
        if (epoch + 1) % 50 == 0:
            if save_imgs:
                generate_flow(9, model=model, save_path=f"outputs/flow_matching/images/sample_epoch{epoch+1}.png")

    torch.save(model.state_dict(), f"outputs/flow_matching/{model_name}.pth")

# --- Generación Flow Condicional ---
@torch.no_grad()
def generate_flow(label, model=None, save_path=None, show=False):
    if model is None:
        #model = nn.DataParallel(FlowModel(), device_ids=[0,1,2,3,4,5])
        #model = FlowModel().to(device)
        model = FlowUNet().to(device)
        model.load_state_dict(torch.load("outputs/flow_matching/flow_model.pth"))
        model.eval()

    x = torch.randn(64, *img_shape).to(device)
    y = torch.full((64,), label, dtype=torch.long, device=device)
    steps = 50
    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full((x.size(0),), i * dt, device=device)
        v = model(x, t, y)
        x = x + v * dt

    img = (x + 1) / 2
    utils.save_image(img, save_path or f"outputs/flow_matching/images/flow_gen_{label}.png", nrow=8)
    if show:
        plt.imshow(img[0].cpu().squeeze().numpy(), cmap='gray')
        plt.title(f'Generated {label}')
        plt.axis('off')
        plt.show()

# train_flow()
# generate_flow(9)


train_flow(epochs=10000,save_imgs=True,model_name="flow_model2_unet")


