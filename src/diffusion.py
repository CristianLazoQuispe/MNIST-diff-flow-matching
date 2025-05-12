# conditional_mnist_diffusion_flow.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# --- Configuración ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
timesteps = 100
img_shape = (1, 28, 28)
os.makedirs("outputs/diffusion", exist_ok=True)
os.makedirs("outputs/flow_matching", exist_ok=True)


# --- Dataset ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)
])
dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

os.makedirs("outputs/diffusion", exist_ok=True)
os.makedirs("outputs/flow_matching", exist_ok=True)

# --- Modelo Condicional ---
class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28 + 1 + 10, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 28*28)
        )

    def forward(self, x, t, y):
        x_flat = x.view(x.size(0), -1)
        t_embed = t.unsqueeze(1).float() / timesteps
        y_onehot = F.one_hot(y, num_classes=10).float()
        x_cat = torch.cat([x_flat, t_embed, y_onehot], dim=1)
        return self.net(x_cat).view(x.size())

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

# --- Entrenamiento Diffusion ---
def train_diffusion():
    model = DiffusionModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alphas = 1 - betas
    alpha_hat = torch.cumprod(alphas, dim=0)

    for epoch in range(100):
        pbar = tqdm(dataloader, desc=f"[Diffusion {epoch}]", leave=False, ncols=80)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            t = torch.randint(0, timesteps, (x.size(0),), device=device)
            a_hat = alpha_hat[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            x_t = a_hat.sqrt() * x + (1 - a_hat).sqrt() * noise

            noise_pred = model(x_t, t, y)
            loss = F.mse_loss(noise_pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if (epoch + 1) % 10 == 0:
            generate_diffusion(9, model=model, save_path=f"outputs/diffusion/sample_epoch{epoch+1}.png")

    torch.save(model.state_dict(), "outputs/diffusion_model.pth")

# --- Generación Diffusion Condicional ---
@torch.no_grad()
def generate_diffusion(label, model=None, save_path=None, show=False):
    if model is None:
        model = DiffusionModel().to(device)
        model.load_state_dict(torch.load("outputs/diffusion_model.pth"))
        model.eval()

    x = torch.randn(64, *img_shape).to(device)
    y = torch.full((64,), label, dtype=torch.long, device=device)

    betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alphas = 1 - betas
    alpha_hat = torch.cumprod(alphas, dim=0)

    for t in reversed(range(timesteps)):
        t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)
        eps_pred = model(x, t_batch, y)
        beta = betas[t]
        alpha = alphas[t]
        x = (1 / alpha.sqrt()) * (x - beta / (1 - alpha_hat[t]).sqrt() * eps_pred)
        if t > 0:
            x += torch.randn_like(x) * beta.sqrt()

    if save_path:
        utils.save_image((x + 1) / 2, save_path, nrow=8)
    else:        
        utils.save_image((x + 1) / 2, f"outputs/diffusion/diffusion_gen_{label}.png", nrow=8)
    if show:
        plt.imshow(((x[0].cpu().squeeze() + 1) / 2).numpy(), cmap='gray')
        plt.title(f'Generated {label}')
        plt.axis('off')
        plt.show()
        
# --- Entrenamiento Flow Matching ---
def train_flow():
    model = FlowModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(100):
        pbar = tqdm(dataloader, desc=f"[Flow {epoch}]", leave=False, ncols=80)
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

        if (epoch + 1) % 10 == 0:
            generate_flow(9, model=model, save_path=f"outputs/flow_matching/sample_epoch{epoch+1}.png")

    torch.save(model.state_dict(), "outputs/flow_model.pth")

# --- Generación Flow Condicional ---
@torch.no_grad()
def generate_flow(label, model=None, save_path=None, show=False):
    if model is None:
        model = FlowModel().to(device)
        model.load_state_dict(torch.load("outputs/flow_model.pth"))
        model.eval()

    x = torch.randn(64, *img_shape).to(device)
    y = torch.full((64,), label, dtype=torch.long, device=device)
    steps = 50
    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full((x.size(0),), i * dt, device=device)
        v = model(x, t, y)
        x = x + v * dt

    if save_path:
        utils.save_image((x + 1) / 2, save_path, nrow=8)
    else:
        utils.save_image((x + 1) / 2, f"outputs/flow_gen_{label}.png", nrow=8)
    if show:
        import matplotlib.pyplot as plt
        plt.imshow(((x[0].cpu().squeeze() + 1) / 2).numpy(), cmap='gray')
        plt.title(f'Generated {label}')
        plt.axis('off')
        plt.show()