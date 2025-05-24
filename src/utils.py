import os
import torch
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def generate_centered_gaussian_noise(shape=(1, 1, 28, 28), sigma=5.0, mu=0):
    B, C, H, W = shape
    assert C == 1, "only image gray"

    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    yy = yy.to(torch.float32)
    xx = xx.to(torch.float32)

    center_y, center_x = H / 2, W / 2
    gauss = torch.exp(-((yy - center_y)**2 + (xx - center_x)**2) / (2 * sigma**2))
    gauss = gauss / gauss.max()  # Normalization to [0, 1]
    gauss = gauss.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)

    noise = mu + torch.randn(B, C, H, W)  # Noise with mean mu
    localized_noise = noise * gauss + mu * (1 - gauss)

    return localized_noise

