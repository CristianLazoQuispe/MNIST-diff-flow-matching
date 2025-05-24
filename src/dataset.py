
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from tqdm import tqdm



def get_data(batch_size,img_shape):
    # --- Dataset ---
    transform = transforms.Compose([
        transforms.Resize((img_shape[-2], img_shape[-1])),            # <-- resize
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)  # scale to [-1, 1]
    ])
    
    full_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
