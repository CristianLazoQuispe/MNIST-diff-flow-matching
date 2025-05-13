import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import cv2
import sys
import torch
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from src.model import ConditionalUNet
from huggingface_hub import hf_hub_download
import time
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
img_shape = (1, 28, 28)


def resize(image,size=(200,200)):
    stretch_near = cv2.resize(image, size, interpolation = cv2.INTER_LINEAR)
    return stretch_near
        

model_diff = ConditionalUNet().to(device)
model_path = hf_hub_download(repo_id="CristianLazoQuispe/MNIST_Diff_Flow_matching", filename="outputs/diffusion/diffusion_model.pth",
                        cache_dir="models")
#model_path = "outputs/diffusion/diffusion_model.pth"
print("Diff Downloaded!")
model_diff.load_state_dict(torch.load(model_path, map_location=device))
model_diff.eval()


model_flow = ConditionalUNet().to(device)
model_path = hf_hub_download(repo_id="CristianLazoQuispe/MNIST_Diff_Flow_matching", filename="outputs/flow_matching/flow_model.pth",
                        cache_dir="models")
#model_path = "outputs/flow_matching/flow_model.pth"
print("Flow Downloaded!")
model_flow.load_state_dict(torch.load(model_path, map_location=device))
model_flow.eval()

@torch.no_grad()
def generate_diffusion_intermediates_streaming(label):
    timesteps = 500
    betas = torch.linspace(1e-4, 0.02, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    x = torch.randn(1, *img_shape).to(device)
    y = torch.tensor([label], dtype=torch.long, device=device)

    # Inicial
    img_np = ((x + 1) / 2.0)[0, 0].clamp(0, 1).cpu().numpy()

    # Para mantener la posición de cada imagen
    outputs = [None] * 13
    yield tuple(outputs)
    outputs[0] = resize(img_np)
    yield tuple(outputs)
    time.sleep(0.2)

    for t in reversed(range(timesteps)):
        t_tensor = torch.full((x.size(0),), t, device=device, dtype=torch.float)
        noise_pred = model_diff(x, t_tensor, y)
        x = (1 / alphas[t].sqrt()) * (x - noise_pred * betas[t] / (1 - alphas_cumprod[t]).sqrt() )
        if t > 0:
            noise = torch.randn(1, *img_shape).to(device)
            v = (1 - alphas_cumprod[t - 1]) / (1 - alphas_cumprod[t]) * betas[t]
            x += v.sqrt() * noise
        x = x.clamp(-1, 1)


        if t in [499, 399, 299, 199, 99, 0]:
            step_idx = {499: 6, 399: 7, 299: 8, 199: 9, 99: 10, 0: 11}[t]
            v_mag = noise_pred[0, 0].abs().clamp(0, 3).cpu().numpy()
            v_mag = (v_mag - v_mag.min()) / (v_mag.max() - v_mag.min() + 1e-5)
            vel_colored = plt.get_cmap("coolwarm")(v_mag)[:, :, :3]
            vel_colored = (vel_colored * 255).astype(np.uint8)
            outputs[step_idx] = resize(vel_colored)
            yield tuple(outputs)
            #time.sleep(0.5)

        if t in [400, 300, 200, 100, 1, 0]:
            step_idx = {400: 1, 300: 2, 200: 3, 100: 4, 1: 5, 0 :12}[t]
            if t==0:
                outputs[step_idx] = resize(((x + 1) / 2.0)[0, 0].cpu().numpy(),(300,300))
            else:
                outputs[step_idx] = resize(((x + 1) / 2.0)[0, 0].cpu().numpy())
            yield tuple(outputs)


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
def generate_flow_intermediates_streaming(label):
    x = torch.randn(1, *img_shape).to(device)
    #x = generate_localized_noise((1, 1, 28, 28), radius=12).to(device)
    y = torch.full((1,), label, dtype=torch.long, device=device)
    steps = 50
    dt = 1.0 / steps
    
    images = [(x + 1) / 2.0]  # initial noise
    vel_magnitudes = []

    # Inicial
    img_np = ((x + 1) / 2.0)[0, 0].clamp(0, 1).cpu().numpy()

    # Para mantener la posición de cada imagen
    outputs = [None] * 13
    yield tuple(outputs)
    outputs[0] = resize(img_np)
    yield tuple(outputs)
    time.sleep(0.2)


    for i in range(steps):
            
        t = torch.full((1,), i * dt, device=device)
        v = model_flow(x, t, y)
        x = x + v * dt

        if i in [10,20,30,40,48,49]:
            #images.append((x + 1) / 2.0)
            step_idx = {10: 1, 20: 2, 30: 3, 40: 4, 48: 5, 49:12}[i]
            if i==49:
                outputs[step_idx] = resize(((x + 1) / 2.0)[0, 0].clamp(0, 1).cpu().numpy(),(300,300))
            else:
                outputs[step_idx] = resize(((x + 1) / 2.0)[0, 0].clamp(0, 1).cpu().numpy())
            yield tuple(outputs)

            # Compute velocity magnitude and convert to numpy for visualization
        if i in [0,11,21,31,41,49]:
            v_mag = dt*v[0, 0].abs().clamp(0, 3).cpu().numpy()  # Clamp to max value for better contrast
            v_mag = (v_mag - v_mag.min()) / (v_mag.max() - v_mag.min() + 1e-5)
            vel_colored = plt.get_cmap("coolwarm")(v_mag)[:, :, :3]  # (H,W,3)
            vel_colored = (vel_colored * 255).astype(np.uint8)
            step_idx = {0: 6, 11: 7, 21: 8, 31: 9, 41: 10, 49:11}[i]
            outputs[step_idx] = resize(vel_colored)
            yield tuple(outputs)
            #time.sleep(0.5)

with gr.Blocks() as demo:
    gr.Markdown("# Conditional MNIST Generation: Diffusion vs Flow Matching")

    with gr.Tab("Diffusion"):
        label_d = gr.Slider(0, 9, step=1, label="Digit Label")
        btn_d = gr.Button("Generate")
        with gr.Row():
            outs_d = [
                gr.Image(label="Noise",streaming=True),
                gr.Image(label="Diffusion t=400",streaming=True),
                gr.Image(label="Diffusion t=300",streaming=True),
                gr.Image(label="Diffusion t=200",streaming=True),
                gr.Image(label="Diffusion t=100",streaming=True),
                gr.Image(label="Diffusion t=1",streaming=True),
            ]
        with gr.Row():
            #400, 300, 200, 100,0
            diff_noise_imgs = [
                gr.Image(label="Noise pred t=500",streaming=True),
                gr.Image(label="Noise pred t=400",streaming=True),
                gr.Image(label="Noise pred t=300",streaming=True),
                gr.Image(label="Noise pred t=200",streaming=True),
                gr.Image(label="Noise pred t=100",streaming=True),
                gr.Image(label="Noise pred t=1",streaming=True),
            ]
        with gr.Row():
            diff_result_imgs = [
                gr.Image(label="Diffusion t=0",streaming=True),
            ]
        btn_d.click(fn=generate_diffusion_intermediates_streaming, inputs=label_d, outputs=outs_d+diff_noise_imgs+diff_result_imgs)

    with gr.Tab("Flow Matching"):
        label_f = gr.Slider(0, 9, step=1, label="Digit Label")
        btn_f = gr.Button("Generate")
        with gr.Row():
            outs_f = [
                gr.Image(label="Noise"),
                gr.Image(label="Flow step=10"),
                gr.Image(label="Flow step=20"),
                gr.Image(label="Flow step=30"),
                gr.Image(label="Flow step=40"),
                gr.Image(label="Flow step=48"),
            ]
        with gr.Row():
            #100,200,300,400,499
            flow_vel_imgs = [
                gr.Image(label="Velocity step=0"),
                gr.Image(label="Velocity step=10"),
                gr.Image(label="Velocity step=20"),
                gr.Image(label="Velocity step=30"),
                gr.Image(label="Velocity step=40"),
                gr.Image(label="Velocity step=48")
            ]
        with gr.Row():
            flow_result_imgs = [
                gr.Image(label="Flow step=49",streaming=True),
            ]
        btn_f.click(fn=generate_flow_intermediates_streaming, inputs=label_f, outputs=outs_f+flow_vel_imgs+flow_result_imgs)

#demo.launch()
demo.launch(share=False, server_port=9071)
