import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import  utils
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import ConditionalUNet
from src.utils import set_seed
from src.dataset import get_data
set_seed(42)


# --- Configuration ---
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
batch_size = 128
timesteps = 500
img_shape = (1, 28, 28)
betas = torch.linspace(1e-4, 0.02, timesteps)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

os.makedirs("outputs/diffusion", exist_ok=True)
os.makedirs("outputs/diffusion/images/", exist_ok=True)

# --- Dataset ---
train_loader, val_loader = get_data(batch_size=batch_size,img_shape=img_shape)


# --- DDPM Training ---
def train_diffusion(epochs=100, save_imgs=False, model_name="diffusion_model"):
    model = ConditionalUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    min_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_bar = tqdm(train_loader, desc=f"[Train Epoch {epoch}]", leave=True, ncols=100)
        for x0, labels in train_bar:
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
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        model.eval()
        val_losses = []
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"[Val   Epoch {epoch}]", leave=True, ncols=100)
            for x0, labels in val_bar:
                x0, labels = x0.to(device), labels.to(device)
                t = torch.randint(0, timesteps, (x0.size(0),), device=device)
                at = alphas_cumprod[t].view(-1, 1, 1, 1)
                noise = torch.randn_like(x0)
                xt = (at.sqrt() * x0 + (1 - at).sqrt() * noise).clamp(-1, 1)
                noise_pred = model(xt, t.float(), labels)
                val_loss = F.mse_loss(noise_pred, noise)
                val_losses.append(val_loss.item())
                val_bar.set_postfix({"loss": f"{np.mean(val_losses):.4f}"})

        avg_val_loss = np.mean(val_losses)
        #print(f"Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            print("Saving best model!")
            torch.save(model.state_dict(), f"outputs/diffusion/{model_name}.pth")

        if (epoch + 1) % 1 == 0 and save_imgs:
            generate_diffusion(9, model, save_path=f"outputs/diffusion/images/sample_epoch{epoch+1}.png")

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
        noise_pred = model(x, t_tensor, y)
        x = (1 / alphas[t].sqrt()) * (x - noise_pred * betas[t] / (1 - alphas_cumprod[t]).sqrt())
        if t > 0:
            noise = torch.randn(64, *img_shape).to(device)
            v = (1 - alphas_cumprod[t - 1]) / (1 - alphas_cumprod[t]) * betas[t]
            x += v.sqrt() * noise
        x = x.clamp(-1, 1)

    img = (x + 1) / 2
    utils.save_image(img, save_path or f"outputs/diffusion/images/generated_label{label}.png", nrow=8)
    if show:
        plt.imshow(img[0].cpu().squeeze().numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f'Generated {label}')
        plt.show()

train_diffusion(epochs=500, save_imgs=True, model_name="diffusion_model_aux")
