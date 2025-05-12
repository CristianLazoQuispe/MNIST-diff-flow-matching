# conditional_mnist_diffusion_flow.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- Configuración ---
device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
batch_size = 128
timesteps = 100
img_shape = (1, 28, 28)
betas = torch.linspace(1e-4, 0.02, timesteps)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

os.makedirs("outputs/diffusion/images", exist_ok=True)

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
        
# --- Entrenamiento con DDPM ---
def train_diffusion(epochs=100,save_imgs=False, model_name="diffusion_model"):
    model = ConditionalUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"[Epoch {epoch}]", leave=True, ncols=80)
        for x0, labels in pbar:
            x0, labels = x0.to(device), labels.to(device)
            t = torch.randint(0, timesteps, (x0.size(0),), device=device)
            at = alphas_cumprod[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x0)
            xt = (at.sqrt() * x0 + (1 - at).sqrt() * noise).clamp(-1, 1)
            noise_pred = model(xt, t.float(), labels)
            loss = F.mse_loss(noise_pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if (epoch + 1) % 10 == 0:
            if save_imgs:
                generate_diffusion(9, model, save_path=f"outputs/diffusion/images/sample_epoch{epoch+1}.png")
        torch.save(model.state_dict(), f"outputs/diffusion/{model_name}.pth")

@torch.no_grad()
def generate_diffusion(label, model=None, save_path=None, show=False):
    if model is None:
        model = ConditionalUNet().to(device)
        model.load_state_dict(torch.load("outputs/diffusion/diffusion_model.pth"))
        model.eval()

    x = torch.randn(64, *img_shape).to(device)
    y = torch.full((64,), label, dtype=torch.long, device=device)

    for t in reversed(range(timesteps)):
        t_tensor = torch.full((x.size(0),), t, device=device, dtype=torch.float)
        eps_pred = model(x, t_tensor, y)
        alpha_t = alphas[t]
        alpha_bar = alphas_cumprod[t]
        x0_pred = (x - (1 - alpha_bar).sqrt() * eps_pred) / alpha_bar.sqrt()
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = alpha_t.sqrt() * x0_pred + (1 - alpha_t).sqrt() * noise
        x = x.clamp(-1, 1)

    img = (x + 1) / 2
    utils.save_image(img, save_path or f"outputs/diffusion/images/generated_label{label}.png", nrow=8)
    if show:
        plt.imshow(img[0].cpu().squeeze().numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f'Generated {label}')
        plt.show()

# --- Ejecutar ---
# train_diffusion()
# generate_diffusion(9)
train_diffusion(epochs=100,save_imgs=True, model_name="diffusion_model")