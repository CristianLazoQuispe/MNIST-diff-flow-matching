import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import time
import gradio as gr
from src.utils import generate_centered_gaussian_noise
from src.demo import resize,plot_flow,load_models,plot_diff

device = 'cuda' if torch.cuda.is_available() else 'cpu'
img_shape = (1, 28, 28)
ENV = "DEPLOY"
TIME_SLEEP = 0.05



model_diff_standard,model_flow_standard,model_flow_localized = load_models(ENV,device=device)


@torch.no_grad()
def generate_diffusion_intermediates_streaming(label):
    timesteps = 500
    betas = torch.linspace(1e-4, 0.02, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
    model_diff = model_diff_standard

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

        outputs = plot_diff(outputs,x,t,noise_pred)

        if t % 10 == 0:
            yield tuple(outputs)
            time.sleep(0.06)

        if ENV=="LOCAL":
            time.sleep(TIME_SLEEP)

    yield tuple(outputs)




@torch.no_grad()
def generate_flow_intermediates_streaming(label,noise_type):
    if noise_type=="Localized":
        x = generate_centered_gaussian_noise((1, *img_shape)).to(device)        
        model_flow = model_flow_localized
    else:
        x = torch.randn(1, *img_shape).to(device)
        model_flow = model_flow_standard

    y = torch.full((1,), label, dtype=torch.long, device=device)
    steps = 50
    dt = 1.0 / steps
    
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
        outputs = plot_flow(outputs,i,x,dt,v)
        if i % 2 == 0:
            yield tuple(outputs)
            time.sleep(0.2) # sleep to render properly in gradio
        if ENV=="LOCAL":
            time.sleep(TIME_SLEEP)
    yield tuple(outputs)


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
        with gr.Row():
            noise_selector_f = gr.Radio(
                ["Standard", "Localized"],
                label="Noise Type:",
                value="Standard"  # o "Standard", según quieras el valor por defecto
            )
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
        btn_f.click(fn=generate_flow_intermediates_streaming, inputs=[label_f,noise_selector_f], outputs=outs_f+flow_vel_imgs+flow_result_imgs)


if ENV=="DEPLOY":
    demo.launch()
    #demo.launch(share=True, server_port=9071)
else:
    demo.launch(share=True, server_port=9071)
