# 🧠 MNIST: Diffusion vs Flow Matching

<div align="center">
  <img src="demo/demo.gif" alt="MNIST Diffusion vs Flow Matching" width="800"/>
</div>

An interactive demo comparing **Diffusion Models** and **Flow Matching** for conditional image generation on MNIST. Understand their dynamics step by step — and visually explore the difference between denoising and velocity-guided synthesis. 

---

## 🌍 Live Demo

▶️ [Try it on Hugging Face Spaces](https://huggingface.co/spaces/CristianLazoQuispe/mnist-diffusion-flow)

---

## 📦 Pretrained Models

- 🤗 Hugging Face: [`CristianLazoQuispe/MNIST_Diff_Flow_matching`](https://huggingface.co/CristianLazoQuispe/MNIST_Diff_Flow_matching)

---

## 🚀 Features

### 🌀 Diffusion
- Iterative denoising based on noise prediction.
- Visualization of noise magnitude across time steps.

### ⚡ Flow Matching
- Direct velocity-based generation in continuous time.
- Visualization of predicted velocities via color maps.

---

## 📁 Project Structure

```bash
.
├── codes/           # Training notebooks and scripts
│   ├── Diffusion.ipynb        # Diffusion training notebook
│   ├── diffusion.py           # Diffusion training script
│   ├── FlowMatching.ipynb     # Flow Matching training notebook
│   └── flow_matching.py       # Flow Matching training script
├── demo/            # Gradio app and animation
│   ├── app.py
│   └── demo.gif
├── src/             # Dataset, model and utilities
├── data/            # MNIST dataset (auto-downloaded)
└── requirements.txt # Dependencies
└── Understabding Flow Matching by Cristian Lazo Quispe # Notes of Flow Matching
```

## ⚙️ Installation

Clone the repo:

```bash
git clone https://github.com/CristianLazoQuispe/MNIST-diff-flow-matching.git
cd MNIST-diff-flow-matching
```

Create a virtual environment (optional but recommended):

```bash
python -m venv env
source env/bin/activate  # or `.\env\Scripts\activate` on Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🏋️‍♂️ Training

You can train the models from scratch using the provided notebooks or scripts:

### 🔧 Diffusion

```bash
python codes/diffusion.py
```

Or use `codes/Diffusion.ipynb` to train interactively.

### 🔧 Flow Matching

```bash
python codes/flow_matching.py
```

Or use `codes/FlowMatching.ipynb` to explore in notebook format.

---

## 🧪 Launch the Demo Locally

```bash
cd demo
python app.py
```

---

## 📄 License

This project is licensed under the MIT License.

---

Created by Cristian Lazo Quispe
