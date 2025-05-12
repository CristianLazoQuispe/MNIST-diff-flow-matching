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