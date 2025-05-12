# conditional_mnist_diffusion_flow.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
print(os.getcwd())
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.conditional_flow_matching import *



# --- Configuración ---
device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
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
    #transforms.Lambda(lambda x: x * 2 - 1)
    transforms.Normalize([0.5],[0.5])
])



dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



# --- Modelos ---
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# --- Entrenamiento Diffusion ---
def train_diffusion(epochs=1000, save_imgs=True, model_name="diffusion_model"):
    model = ConditionalUNet().to(device)
    #model = ConditionalUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    FM = ConditionalFlowMatcher(sigma=0.0)

    betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alphas = 1 - betas
    alpha_hat = torch.cumprod(alphas, dim=0)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"[Diffusion {epoch}]", leave=True, ncols=80)
        for x_real, y in pbar:
            x_real = x_real.to(device)
            y = y.to(device)
            x_noise = torch.randn_like(x_real)
            t = torch.rand(x_real.size(0), device=device)
            x_t = (1 - t.view(-1, 1, 1, 1)) * x_noise + t.view(-1, 1, 1, 1) * x_real
            v_target = x_real - x_noise
            v_pred = model(x_t, t, y)
            loss = F.mse_loss(v_pred, v_target)
            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        torch.save(model.state_dict(), f"outputs/diffusion/{model_name}.pth")
        if (epoch + 1) % 50 == 0:
            if save_imgs:
                generate_diffusion(9, model=model, save_path=f"outputs/diffusion/images/sample_epoch{epoch+1}.png")

    torch.save(model.state_dict(), f"outputs/diffusion/{model_name}.pth")


# --- Generación Diffusion Condicional ---
@torch.no_grad()
def generate_diffusion(label, model=None, save_path=None, show=False):
    if model is None:
        #model = nn.DataParallel(DiffusionModel(), device_ids=[0,1,2,3,4,5]).to(device)
        #model = ConditionalUNet().to(device)
        model = ConditionalUNet().to(device)
        #model = SimpleConditionalUnet().to(device)
        model.load_state_dict(torch.load("outputs/diffusion/diffusion_model.pth"))
        model.eval()

    x = torch.randn(64, *img_shape).to(device)
    y = torch.full((64,), label, dtype=torch.long, device=device)

    betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alphas = 1 - betas
    alpha_hat = torch.cumprod(alphas, dim=0)

    for t in reversed(range(timesteps)):

        t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)
        eps_pred = model(x, t_batch, y)
        alpha_t = alphas[t]
        alpha_bar_t = alpha_hat[t]
        x0_pred = (x - (1 - alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = alpha_t.sqrt() * x0_pred + (1 - alpha_t).sqrt() * noise
        x = x.clamp(-1, 1)

    img = (x + 1) / 2
    utils.save_image(img, save_path or f"outputs/diffusion/images/diffusion_gen_{label}.png", nrow=8)
    if show:
        plt.imshow(img[0].cpu().squeeze().numpy(), cmap='gray')
        plt.title(f'Generated {label}')
        plt.axis('off')
        plt.show()

# --- Ejecutar ---
# train_diffusion()
# generate_diffusion(9)

train_diffusion(epochs=10000,save_imgs=True,model_name="diffusion_model2")


