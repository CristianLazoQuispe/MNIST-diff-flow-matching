import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from .model import ConditionalUNet
from huggingface_hub import hf_hub_download


def load_models(ENV,device):
    if ENV=="DEPLOY":
        model_path = hf_hub_download(repo_id="CristianLazoQuispe/MNIST_Diff_Flow_matching", filename="outputs/diffusion/diffusion_model.pth",cache_dir="models")
    else:
        model_path  = "outputs/diffusion/diffusion_model.pth"
    print("Diff Downloaded!")
    model_diff_standard  = ConditionalUNet().to(device)
    model_diff_standard.load_state_dict(torch.load(model_path, map_location=device))
    model_diff_standard.eval()

    if ENV=="DEPLOY":
        model_path_standard  = hf_hub_download(repo_id="CristianLazoQuispe/MNIST_Diff_Flow_matching", filename="outputs/flow_matching/flow_model.pth",cache_dir="models")
        model_path_localized = hf_hub_download(repo_id="CristianLazoQuispe/MNIST_Diff_Flow_matching", filename="outputs/flow_matching/flow_model_localized_noise.pth",cache_dir="models")
    else:
        model_path_standard  = "outputs/flow_matching/flow_model.pth"
        model_path_localized = "outputs/flow_matching/flow_model_localized_noise.pth"
    print("Flow Downloaded!")
    model_flow_standard  = ConditionalUNet().to(device)
    model_flow_standard.load_state_dict(torch.load(model_path_standard, map_location=device))
    model_flow_standard.eval()
    model_flow_localized = ConditionalUNet().to(device)
    model_flow_localized.load_state_dict(torch.load(model_path_localized, map_location=device))
    model_flow_localized.eval()

    return model_diff_standard,model_flow_standard,model_flow_localized
def resize(image,size=(200,200)):
    stretch_near = cv2.resize(image, size, interpolation = cv2.INTER_LINEAR)
    return stretch_near
        


def plot_diff(outputs,x,t,noise_pred):

    if t in [499, 399, 299, 199, 99, 0]:
        step_idx = {499: 6, 399: 7, 299: 8, 199: 9, 99: 10, 0: 11}[t]
        v_mag = noise_pred[0, 0].abs().clamp(0, 3).cpu().numpy()
        v_mag = (v_mag - v_mag.min()) / (v_mag.max() - v_mag.min() + 1e-5)
        vel_colored = plt.get_cmap("coolwarm")(v_mag)[:, :, :3]
        vel_colored = (vel_colored * 255).astype(np.uint8)
        outputs[step_idx] = resize(vel_colored)

    outputs[12] = resize(((x + 1) / 2.0)[0, 0].cpu().numpy(),(300,300))

    if t in [400, 300, 200, 100, 1, 0]:
        step_idx = {400: 1, 300: 2, 200: 3, 100: 4, 1: 5, 0 :12}[t]
        if t==0:
            outputs[step_idx] = resize(((x + 1) / 2.0)[0, 0].cpu().numpy(),(300,300))
        else:
            outputs[step_idx] = resize(((x + 1) / 2.0)[0, 0].cpu().numpy())
        
    return outputs

def plot_flow(outputs,i,x,dt,v):
    # Compute velocity magnitude and convert to numpy for visualization
    outputs[12] =  resize(((x + 1) / 2.0)[0, 0].clamp(0, 1).cpu().numpy(),(300,300))
    if i in [10,20,30,40,48,49]: #
        step_idx = {10: 1, 20: 2, 30: 3, 40: 4, 48: 5,49:12}[i] #, 
        if i==49:
            outputs[step_idx] = resize(((x + 1) / 2.0)[0, 0].clamp(0, 1).cpu().numpy(),(300,300))
        else:
            outputs[step_idx] = resize(((x + 1) / 2.0)[0, 0].clamp(0, 1).cpu().numpy())

    if i in [0,11,21,31,41,49]:
        v_mag = dt*v[0, 0].abs().clamp(0, 3).cpu().numpy()  # Clamp to max value for better contrast
        v_mag = (v_mag - v_mag.min()) / (v_mag.max() - v_mag.min() + 1e-5)
        vel_colored = plt.get_cmap("coolwarm")(v_mag)[:, :, :3]  # (H,W,3)
        vel_colored = (vel_colored * 255).astype(np.uint8)
        step_idx = {0: 6, 11: 7, 21: 8, 31: 9, 41: 10, 49:11}[i]
        outputs[step_idx] = resize(vel_colored)
    return outputs