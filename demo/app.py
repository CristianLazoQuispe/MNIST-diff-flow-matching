import gradio as gr
import torch
import os
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.conditional_unet import ConditionalUNet


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
img_shape = (1, 28, 28)


def resize(image):
    stretch_near = cv2.resize(image, (200, 200), interpolation = cv2.INTER_LINEAR)
    return stretch_near
        


@torch.no_grad()
def generate_diffusion_intermediates(label):
    timesteps = 500
    img_shape = (1, 28, 28)
    model = ConditionalUNet().to(device)
    model.load_state_dict(torch.load("outputs/diffusion/diffusion_model.pth", map_location=device))
    model.eval()
    betas = torch.linspace(1e-4, 0.02, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

    x = torch.randn(1, *img_shape).to(device)
    y = torch.tensor([label], dtype=torch.long, device=device)

    intermediates = [resize(((x + 1) / 2.0)[0][0].clamp(0, 1).cpu().numpy())]

    for t in reversed(range(timesteps)):
        t_tensor = torch.full((x.size(0),), t, device=device, dtype=torch.float)
        noise_pred = model(x, t_tensor, y)
        x = (1 / alphas[t].sqrt()) * (x - noise_pred * betas[t] / (1 - alphas_cumprod[t]).sqrt() )
        if t > 0:
            noise = torch.randn(1, *img_shape).to(device)
            v = (1 - alphas_cumprod[t - 1]) / (1 - alphas_cumprod[t]) * betas[t]
            x += v.sqrt() * noise
            
        x = x.clamp(-1, 1)
        if t in [99*5, 75*5, 50*5, 25*5, 0]:
            print("t:",t)
            img_np = ((x + 1) / 2)[0, 0].cpu().numpy()
            intermediates.append(resize(img_np))

    return intermediates


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
