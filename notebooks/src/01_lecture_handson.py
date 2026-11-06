# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Deep Learning, Hands-On: from tensors to a shipped model
# **Guest lecture · The Roux Institute at Northeastern University · November 19, 2026**
# **Dr. Sushanth Manakhari** — host: Dr. Umesh Hodeghatta Rao
#
# Today you are a new ML engineer with **two tickets** on your sprint board:
#
# | Ticket | Modality | Business problem | Model |
# |---|---|---|---|
# | 🌱 **LEAF-101** | images | Farmers photograph a bean leaf → instant diagnosis (healthy / angular leaf spot / bean rust) | ResNet-50, transfer learning |
# | 💬 **SUPPORT-202** | text | Route customer messages to the right team (77 intents) | DistilBERT, fine-tuning |
#
# You will solve both with **the same recipe**, break things on purpose, debug them like a
# professional, and finish with a web app you can open **on your phone**.
#
# ### How this notebook works (the "Don't Panic" rules)
# | Icon | Meaning |
# |---|---|
# | ⚙️ | Setup — run it, don't edit it |
# | 🛟 | **Catch-up cell.** Lost, restarted, or got an error? Run the 🛟 cell at the top of the section and you are back in sync |
# | 🎛️ | **Try it** — change the values and re-run; nothing breaks |
# | ⏩ | Optional — skip if we are short on time |
# | 🧮 | The math behind what the code does |
# | 🏭 | How this is used in industry |
#
# **Every cell works top to bottom.** If you mistype something, fix it or just run the next
# 🛟 cell. On Colab: `Runtime → Change runtime type → T4 GPU` before running anything.

# %% [markdown]
# ## ⚙️ 0. Setup  *(run now — it downloads data and models while we talk)*

# %%
# ⚙️ SETUP — finds (or downloads) the course code, installs only what is missing.
# Safe to re-run. Takes ~1-2 minutes on a fresh Colab machine.
import importlib.util, os, pathlib, shutil, subprocess, sys

REPO_URL = "https://github.com/Sushanth-Manakhari/deep-learning-handson-lecture.git"
os.environ.setdefault("HF_HUB_VERBOSITY", "error")  # public models need no token: hide the "unauthenticated" notice
REPO_BRANCH = "main"   # instructor dry run: set to "release"

def find_repo():
    """Walk up from the current folder looking for src/dlhandson (local clone)."""
    here = pathlib.Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "src" / "dlhandson").is_dir():
            return p
    return None

ROOT = find_repo()
if ROOT is None:  # Colab / Kaggle: fetch code + small data (~2 MB) from GitHub
    base = pathlib.Path("/content") if pathlib.Path("/content").is_dir() else pathlib.Path.cwd()
    ROOT = base / "deep-learning-handson-lecture"
    if ROOT.exists():  # an earlier run cloned something: reuse it only if it is the right, complete branch
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        if head != REPO_BRANCH or not (ROOT / "src" / "dlhandson").is_dir():
            print(f"re-cloning: found branch '{head or '?'}', need '{REPO_BRANCH}'")
            shutil.rmtree(ROOT)
    if not ROOT.exists():
        subprocess.run(["git", "clone", "--depth", "1", "-b", REPO_BRANCH, REPO_URL, str(ROOT)], check=True)
if not (ROOT / "src" / "dlhandson").is_dir():
    raise RuntimeError(f"Branch '{REPO_BRANCH}' does not contain the course code (src/dlhandson). "
                       "Instructor dry run: set REPO_BRANCH = \"release\" above and run this cell again.")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
if not (pathlib.Path("/content").is_dir() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ):
    # local laptop: keep downloaded models inside the project folder, not in ~/.cache
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

# Colab/Kaggle already have PyTorch & Transformers; add the few extras if missing.
missing = [pkg for mod, pkg in [("gradio", "gradio"), ("qrcode", "qrcode"), ("safetensors", "safetensors")]
           if importlib.util.find_spec(mod) is None]
if missing:
    print("installing:", missing)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
print("✅ code ready at", ROOT)

# %%
# ⚙️ Imports + hardware check
import warnings; warnings.filterwarnings("ignore", category=UserWarning)
import json, math, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

import dlhandson
from dlhandson import viz, CLASS_NAMES_BEANS
from dlhandson.device import get_device, describe_device, set_seed, gpu_memory_gb, smoke_mode
from dlhandson.config import vision_config_for, text_config_for, hardware_profile
from dlhandson.timing import SectionTimer
from dlhandson.notebook_utils import headless

device = get_device()
set_seed(42)
info = describe_device(device)
for k, v in info.items():
    print(f"{k:>20}: {v}")
PROFILE = hardware_profile(device)
print(f"\nHardware profile: '{PROFILE}'")
if device.type == "cpu" and not smoke_mode():
    print("⚠️  No GPU found. On Colab: Runtime → Change runtime type → T4 GPU → Save, then Runtime → Run all.\n"
          "   No GPU available at all? Keep going — the notebook switches to FAST mode (smaller model, less data).")

RESULTS = {"device": info}          # key numbers collected for the slides / verification report
timer = SectionTimer(ROOT / "runs" / "timings.json")
timer.start("0. Setup")

# %%
# ⚙️ Pre-load data and pretrained weights NOW so later cells start instantly.
from dlhandson.data_vision import ensure_beans
from dlhandson.data_text import load_banking77, build_tokenizer
from dlhandson.models import build_vision_model, build_text_model

t0 = time.time()
BEANS_DIR = ensure_beans()                                 # images (Release → Hugging Face fallback)
TEXT_DATA = load_banking77()                               # ~1 MB CSV shipped in the repo
vcfg0, tcfg0 = vision_config_for(device), text_config_for(device)
_ = build_vision_model(vcfg0.model_name, 3, vcfg0.pretrained)  # caches ImageNet weights
_ = build_tokenizer(tcfg0.model_name, texts=TEXT_DATA["train"]["text"])
if not smoke_mode():
    _ = build_text_model(tcfg0.model_name, 77, TEXT_DATA["id2label"], TEXT_DATA["label2id"])  # caches DistilBERT
del _
print(f"✅ data + pretrained weights ready in {time.time() - t0:.0f}s")

# %% [markdown]
# ---
# ## 1. Why deep learning? *(framing — 15 min)*
#
# ### The paradigm shift
# | | Traditional ML (≈ until 2012) | Deep learning |
# |---|---|---|
# | Features | **Designed by humans** (colour histograms, edge detectors, n-grams) | **Learned from data**, layer by layer |
# | Pipeline | pixels → *hand-crafted features* → classifier | pixels → *learned features* → classifier, trained end-to-end |
# | Scales with data? | Plateaus | Keeps improving with more data + compute |
# | Example | SIFT + SVM | ResNet, Vision Transformer, BERT, GPT |
#
# 🧮 A deep network is a composition of simple functions,
# $$f(x) = f_L\big(f_{L-1}(\cdots f_1(x))\big),\qquad f_l(h) = \sigma(W_l h + b_l)$$
# Each layer re-represents the input a little more abstractly: **edges → textures → lesions → "bean rust"**.
# All parameters $\theta = \{W_l, b_l\}$ are fitted by gradient descent on a loss $L(\theta)$.
#
# ### The modern stack you will touch today
# | Layer | Tool | Role |
# |---|---|---|
# | Compute | **GPU** (NVIDIA T4 on Colab), TPU, Apple MPS | thousands of parallel multiply-adds |
# | Framework | **PyTorch** | tensors, automatic differentiation, `nn.Module` |
# | Model hub | **Hugging Face** | download pretrained models and datasets in one line |
# | Tracking | **TensorBoard** / Weights & Biases | log every experiment |
# | Serving | **Gradio** | turn a model into a web app |
#
# 🧮 **Why GPUs?** Multiplying two $n\times n$ matrices costs $\approx 2n^3$ floating-point operations.
# Let's measure it.

# %%
timer.start("1. Framing")
def matmul_tflops(dev, n=2048, repeats=10, dtype=torch.float32):
    """Time n x n matrix multiplications and convert to TFLOP/s (2n^3 FLOPs each)."""
    a = torch.randn(n, n, device=dev, dtype=dtype); b = torch.randn(n, n, device=dev, dtype=dtype)
    def sync():  # GPU calls are ASYNCHRONOUS: wait for the GPU before reading the clock
        if dev.type == "cuda": torch.cuda.synchronize()
        if dev.type == "mps": torch.mps.synchronize()
    a @ b; sync()  # warm-up (first call initialises kernels)
    t = time.time()
    for _ in range(repeats): a @ b
    sync()
    secs = (time.time() - t) / repeats
    return 2 * n**3 / secs / 1e12, secs

n = 256 if smoke_mode() else 2048
cpu_tf, cpu_s = matmul_tflops(torch.device("cpu"), n, repeats=3)
print(f"CPU            : {cpu_s*1000:8.1f} ms per {n}x{n} matmul  ≈ {cpu_tf:6.2f} TFLOP/s")
if device.type != "cpu":
    gpu_tf, gpu_s = matmul_tflops(device, n)
    print(f"{device.type.upper():<15}: {gpu_s*1000:8.1f} ms per matmul  ≈ {gpu_tf:6.2f} TFLOP/s  → {cpu_s/gpu_s:.0f}x faster")
    if device.type == "cuda":
        h_tf, h_s = matmul_tflops(device, n, dtype=torch.float16)
        print(f"CUDA float16   : {h_s*1000:8.1f} ms per matmul  ≈ {h_tf:6.2f} TFLOP/s  (Tensor Cores → mixed precision)")

# %% [markdown]
# 🏭 **Industry note.** Training cost is dominated by these matrix multiplications, which is why
# companies budget AI projects in *GPU-hours*, and why tricks like mixed precision (float16) matter:
# same model, roughly half the memory and up to ~2x the speed.
#
# ---
# ## 2. The PyTorch mental model *(20 min)*
# Four ideas cover 90% of PyTorch: **tensors**, **autograd**, **`nn.Module`**, and **the training loop**.
#
# ### 2.1 Tensors — n-dimensional arrays that can live on a GPU

# %%
timer.start("2. PyTorch mental model")
x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
print("shape", x.shape, "| dtype", x.dtype, "| device", x.device)

# Real data are just tensors with agreed-upon axis meanings:
images = torch.rand(32, 3, 224, 224)   # [batch, channels(RGB), height, width]
tokens = torch.randint(0, 30522, (32, 64))   # [batch, sequence length] of token ids
print("image batch:", tuple(images.shape), "→", images.numel() * 4 / 1e6, "MB in float32")
print("text batch :", tuple(tokens.shape))

# Broadcasting: per-channel normalisation (x - mean) / std without loops
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
print("normalised batch:", tuple(((images - mean) / std).shape))

# Moving to the accelerator is one call
print("on device:", images.to(device).device)

# %% [markdown]
# ### 2.2 Gradient descent — by hand first
# 🏭 Toy problem: an insurer estimates **repair cost** ($k) from the **damaged area** (m²).
# Model: $\hat y = w x + b$. Loss (mean squared error):
# $$L(w,b) = \frac{1}{N}\sum_{i=1}^{N} (w x_i + b - y_i)^2$$
# 🧮 Partial derivatives (chain rule):
# $$\frac{\partial L}{\partial w} = \frac{2}{N}\sum_i (\hat y_i - y_i)\,x_i \qquad \frac{\partial L}{\partial b} = \frac{2}{N}\sum_i (\hat y_i - y_i)$$
# Update rule with learning rate $\eta$:  $\; w \leftarrow w - \eta\,\partial L/\partial w,\;\; b \leftarrow b - \eta\,\partial L/\partial b$

# %%
torch.manual_seed(0)
N = 200
area = torch.rand(N) * 5                              # x: damaged area, 0-5 m^2
cost = 1.8 * area + 0.5 + 0.4 * torch.randn(N)        # y: true rule + noise (unknown to the model)

def gradient_descent_by_hand(lr, steps=60):
    w, b, losses = 0.0, 0.0, []
    for _ in range(steps):
        pred = w * area + b
        losses.append(((pred - cost) ** 2).mean().item())
        grad_w = (2 * (pred - cost) * area).mean()      # dL/dw  (formula above)
        grad_b = (2 * (pred - cost)).mean()             # dL/db
        w, b = w - lr * grad_w.item(), b - lr * grad_b.item()
    return w, b, losses

w, b, losses = gradient_descent_by_hand(lr=0.05)
print(f"learned: cost ≈ {w:.2f} · area + {b:.2f}   (truth: 1.80 · area + 0.50)")
viz.plot_curves({"MSE loss": losses}, "Gradient descent by hand", xlabel="step", logy=True)

# %% [markdown]
# ### 2.3 Autograd — PyTorch computes those derivatives for you
# Mark tensors with `requires_grad=True`; PyTorch records every operation in a **computational graph**
# and `loss.backward()` walks it backwards applying the chain rule (reverse-mode automatic
# differentiation). Cost: about one extra forward pass — for *all* parameters at once, whether you
# have 2 or 2 billion.

# %%
w = torch.tensor(0.7, requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)
loss = ((w * area + b - cost) ** 2).mean()
loss.backward()                                   # ← the 1-line backprop
print("loss.grad_fn:", loss.grad_fn)              # the graph remembers how loss was computed

with torch.no_grad():                             # compare against our hand-derived formulas
    pred = w * area + b
    manual_w = (2 * (pred - cost) * area).mean()
    manual_b = (2 * (pred - cost)).mean()
print(f"autograd dL/dw = {w.grad.item():.5f} | by hand = {manual_w.item():.5f}")
print(f"autograd dL/db = {b.grad.item():.5f} | by hand = {manual_b.item():.5f}")
print("match:", torch.allclose(w.grad, manual_w) and torch.allclose(b.grad, manual_b))

# GOTCHA: gradients ACCUMULATE (+=). Call backward again without zeroing:
((w * area + b - cost) ** 2).mean().backward()
print(f"after a 2nd backward (no zero_grad) dL/dw = {w.grad.item():.5f}  ← doubled! → always optimizer.zero_grad()")

# %% [markdown]
# ### 2.4 🎛️ Try it: the learning rate
# 🧮 For this quadratic loss, gradient descent is stable only if $\eta < 2/\lambda_{\max}$, where
# $\lambda_{\max}$ is the largest eigenvalue of the Hessian (the loss "curvature").
# Too small → crawls. Too large → each step overshoots further → **divergence**.

# %%
H = torch.tensor([[2 * (area**2).mean(), 2 * area.mean()], [2 * area.mean(), 2.0]])   # Hessian of the MSE
lam_max = torch.linalg.eigvalsh(H).max().item()
print(f"λ_max = {lam_max:.1f}  →  largest stable learning rate ≈ 2/λ_max = {2/lam_max:.3f}")

LEARNING_RATES = [0.001, 0.01, 0.1, 0.13]        # 🎛️ edit me
curves = {f"lr={lr}": gradient_descent_by_hand(lr)[2] for lr in LEARNING_RATES}
viz.plot_curves(curves, "Same problem, different learning rates", logy=True)

# %% [markdown]
# ### 2.5 Anatomy of a model: `nn.Module`
# Every PyTorch model — from this 3-layer net to GPT — is a class with
# **`__init__`** (declare layers = parameters) and **`forward`** (how data flows).

# %%
class TinyNet(nn.Module):
    def __init__(self, in_dim=2, hidden=32, n_classes=2):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden)      # W1: 2x32, b1: 32
        self.layer2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, n_classes)     # outputs LOGITS (unnormalised scores)

    def forward(self, x):
        h = F.relu(self.layer1(x))                   # ReLU(z) = max(0, z): the non-linearity
        h = F.relu(self.layer2(h))
        return self.head(h)

model = TinyNet()
print(model)
print("parameters:", sum(p.numel() for p in model.parameters()))

# %% [markdown]
# ### 2.6 `Dataset` + `DataLoader`
# A `Dataset` answers two questions — *how many items?* (`__len__`) and *give me item i*
# (`__getitem__`). A `DataLoader` batches, shuffles and (optionally) loads in parallel.

# %%
from sklearn.datasets import make_moons
from torch.utils.data import Dataset, DataLoader

class MoonsDataset(Dataset):
    def __init__(self, n=1000, noise=0.2, seed=0):
        X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
        self.X, self.y = torch.tensor(X, dtype=torch.float32), torch.tensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

moons = MoonsDataset()
loader = DataLoader(moons, batch_size=64, shuffle=True)
xb, yb = next(iter(loader))
print("one batch:", tuple(xb.shape), tuple(yb.shape))

# %% [markdown]
# ### 2.7 The training loop — five lines you will write for the rest of your career
# 🧮 Loss for classification = **cross-entropy**: softmax turns logits $z$ into probabilities
# $p_k = e^{z_k} / \sum_j e^{z_j}$ and $L = -\log p_{y}$. Its gradient is beautifully simple:
# $\partial L/\partial z_k = p_k - \mathbb{1}[k=y]$ — push the right class up, the rest down.

# %%
set_seed(0)
model = TinyNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

for epoch in range(30):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()                 # 1. reset gradients
        logits = model(xb)                    # 2. forward pass
        loss = F.cross_entropy(logits, yb)    # 3. compute loss
        loss.backward()                       # 4. backprop: dL/dθ for every parameter
        optimizer.step()                      # 5. update: θ ← θ − η · (Adam step)
    if epoch % 10 == 9:
        model.eval()
        with torch.no_grad():
            acc = (model(moons.X.to(device)).argmax(1).cpu() == moons.y).float().mean()
        print(f"epoch {epoch+1:>2}: loss {loss.item():.3f}  accuracy {acc:.1%}")

viz.plot_decision_boundary(model.cpu(), moons.X.numpy(), moons.y.numpy(), title="TinyNet learned a curved boundary")
RESULTS["moons_acc"] = float(acc)

# %% [markdown]
# 🏭 That loop is **identical** for a ResNet on images or a Transformer on text. The course package
# keeps a production version in `dlhandson.train.fit` (adds validation, mixed precision, gradient
# clipping, a NaN guard and checkpointing) so we don't copy-paste it — exactly what teams do.

# %% [markdown]
# ---
# ## 3. 🌱 Ticket LEAF-101 — leaf-disease detection with transfer learning *(40 min)*
#
# > **LEAF-101** · *As a farmer co-op, we want growers to photograph a bean leaf with their phone and
# > get an instant diagnosis, so that disease is treated before it spreads.*
# > **Acceptance criteria:** ≥ 90% accuracy on held-out photos · trains on a free GPU in minutes ·
# > says **"not sure — ask an agronomist"** instead of guessing on unusual photos.
#
# **Data:** *Beans* (Makerere University AI Lab, Uganda; MIT licence) — smartphone photos taken in real
# fields: 1,034 train / 133 validation / 128 test, 3 classes.
#
# 🏭 Same pattern elsewhere: crop scouting apps, drone crop monitoring, medical-image triage,
# insurance claim photos, manufacturing defect detection.

# %%
# 🛟 CATCH-UP (vision): rebuilds device, config, data loaders. Safe to run any time.
timer.start("3. Vision transfer learning")
from dlhandson.catchup import vision_state, trained_vision_model
S = vision_state(device)
vcfg, loaders, CLASSES = S["cfg"], S["loaders"], S["class_names"]
print(vcfg)

# %% [markdown]
# ### 3.1 Look at the data first (always)

# %%
from dlhandson.data_vision import image_folder, class_counts
raw_train = image_folder(S["beans_dir"], "train")          # no transform → PIL images
viz.plot_bar(class_counts(raw_train), "Training images per class (balanced → accuracy is a fair metric)")
viz.show_class_examples(raw_train, per_class=4)
print("image size on disk:", raw_train[0][0].size)

# %% [markdown]
# ### 3.2 The "2015 way": hand-crafted features + logistic regression
# A human decides that **colour distribution** (rust is orange, spots are brown) and **edge texture**
# matter, computes them by hand, and fits a classic model. This is our baseline to beat.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from dlhandson.classic import featurize_folder

t0 = time.time()
Xtr, ytr = featurize_folder(raw_train, max_items=vcfg.max_train_samples)
Xva, yva = featurize_folder(image_folder(S["beans_dir"], "validation"))
baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(Xtr, ytr)
baseline_acc = baseline.score(Xva, yva)
print(f"Hand-crafted features: {Xtr.shape[1]} numbers per image → validation accuracy {baseline_acc:.1%} "
      f"({time.time()-t0:.0f}s)")
RESULTS["vision_baseline_val_acc"] = baseline_acc

# %% [markdown]
# ### 3.3 Pre-processing: what the network actually sees
# Pretrained ImageNet models expect 224×224 RGB, standardised per channel with ImageNet statistics:
# $$x' = \frac{x - \mu}{\sigma},\quad \mu = (0.485, 0.456, 0.406),\ \sigma = (0.229, 0.224, 0.225)$$
# 🧮 Standardised inputs keep activations and gradients in a healthy range — and they must match what
# the model saw during pre-training, or its learned features no longer line up.

# %%
batch = next(iter(loaders["train"]))
print("pixel_values:", tuple(batch["pixel_values"].shape), "| labels:", batch["labels"][:8].tolist())
print(f"value range after normalisation: [{batch['pixel_values'].min():.2f}, {batch['pixel_values'].max():.2f}]")

# %% [markdown]
# ### 3.4 Transfer learning step 1-2: load a pretrained ResNet, freeze it, replace the head
# 🧮 Think of the classifier as $f(x) = h_\phi(g_\theta(x))$:
# - $g_\theta$ = **backbone** (ResNet-50: 23.5M parameters pretrained on 1.28M ImageNet photos)
# - $h_\phi$ = **head**, a single linear layer $\mathbb{R}^{2048} \to \mathbb{R}^{3}$ for our classes
#
# We **freeze** $\theta$ (`requires_grad=False`: no gradients, no updates) and train only $\phi$.
# ResNet's key idea, the **residual connection** $y = F(x) + x$, gives gradients an identity
# "highway" ($\partial y/\partial x = \partial F/\partial x + I$) so 50+ layers stay trainable.

# %%
from dlhandson.models import build_vision_model, count_parameters, model_size_mb, get_head

model = build_vision_model(vcfg.model_name, num_classes=3, pretrained=vcfg.pretrained, freeze_backbone=True)
p = count_parameters(model)
print(f"{vcfg.model_name}: {p['total']:,} parameters, {model_size_mb(model)} MB of weights")
print(f"trainable after freezing: {p['trainable']:,}  ({p['trainable_pct']}%)  ← just the new head")
print("new head:", get_head(model))

# %% [markdown]
# ### 3.5 Train the head (no augmentation yet)
# - **Optimizer:** AdamW, $\eta = 10^{-3}$ — a random head tolerates a fairly large learning rate.
# - **Mixed precision** on CUDA: float16 matmuls on Tensor Cores; a `GradScaler` multiplies the loss by
#   a large $S$ before `backward()` so tiny gradients don't underflow to 0, then divides by $S$.
# - Every run is logged to **TensorBoard** (`runs/`).

# %%
from dlhandson.train import build_optimizer, fit, evaluate
from dlhandson.models import trainable_params
from dlhandson.tracking import ExperimentLogger
from dlhandson.data_vision import make_vision_loaders

set_seed(42)
plain_loaders = make_vision_loaders(vcfg, S["beans_dir"], device, augment=False)
model = build_vision_model(vcfg.model_name, 3, vcfg.pretrained, freeze_backbone=True)
opt = build_optimizer(vcfg.optimizer, trainable_params(model), vcfg.lr, vcfg.weight_decay)
logger = ExperimentLogger("vision-head-noaug", config=vcfg.to_dict())
hist_plain = fit(model, plain_loaders["train"], plain_loaders["validation"], epochs=vcfg.epochs, optimizer=opt,
                 device=device, amp=vcfg.amp, logger=logger, name="head, no augmentation")
logger.close()
print(f"\nbest validation accuracy: {hist_plain.best_val_acc:.1%}  vs hand-crafted baseline {baseline_acc:.1%}")
RESULTS["vision_head_noaug_val_acc"] = hist_plain.best_val_acc

# %% [markdown]
# ### 3.6 Data augmentation: more data for free
# Each epoch the network sees a *slightly different* version of every photo. We encode facts we know
# about the problem (**invariances**): leaves can be photographed at any angle (flips, rotation),
# framed differently (random crop), in different light (brightness/contrast).
#
# ⚠️ Augmentation must **preserve the label**: rust is orange-brown, so we keep the hue shift tiny —
# a big hue change could make a diseased leaf look healthy. (For digits, a vertical flip turns a 6
# into a 9 — domain knowledge matters.)

# %%
from dlhandson.data_vision import train_transform
aug = train_transform(vcfg.image_size, augment=True)
leaf = raw_train[5][0]
viz.show_images([leaf] + [aug(leaf) for _ in range(7)], ["original"] + [f"augmented #{i}" for i in range(1, 8)],
                ncols=8, size=1.8)

# %%
set_seed(42)
model = build_vision_model(vcfg.model_name, 3, vcfg.pretrained, freeze_backbone=True)
opt = build_optimizer(vcfg.optimizer, trainable_params(model), vcfg.lr, vcfg.weight_decay)
logger = ExperimentLogger("vision-head-aug", config=vcfg.to_dict())
from dlhandson.catchup import CKPT_DIR
hist_aug = fit(model, loaders["train"], loaders["validation"], epochs=vcfg.epochs, optimizer=opt, device=device,
               amp=vcfg.amp, logger=logger, name="head + augmentation",   # best epoch is cached for the 🛟 cells:
               ckpt_path=CKPT_DIR / f"vision_{vcfg.model_name}_{vcfg.image_size}_head.safetensors")
logger.close()
viz.plot_history({"no augmentation": hist_plain, "with augmentation": hist_aug}, "Effect of data augmentation")
RESULTS["vision_head_aug_val_acc"] = hist_aug.best_val_acc

# %% [markdown]
# **Read the curves like an engineer:** the gap between *train* (dashed) and *validation* (solid) is
# **overfitting**. Augmentation narrows that gap because every epoch is *harder* (new crops, angles,
# lighting). The flip side: with only 5 epochs it can score the same or even a little lower — it pays
# off with more epochs and when fine-tuning deeper layers (try `epochs=10` in the challenge). It also
# costs extra CPU time per image. Trade-offs, not magic.
#
# ### 3.7 🎛️ 5-MINUTE CHALLENGE — break it, then beat it
# Change the knobs, run the cell, watch validation accuracy. Ideas and what to look for:
#
# | Try | What you should see | Why |
# |---|---|---|
# | `lr=1.0` | loss jumps around or 💥 NaN | steps overshoot: $\eta > 2/\lambda_{\max}$ |
# | `lr=1e-5` | barely learns in 5 epochs | steps too tiny |
# | `batch_size=4` vs `128` | time per epoch & curve noise change | small batches = noisy gradients, poor GPU use; big batches need more memory |
# | `optimizer="sgd"` | slower start than AdamW at the same lr | Adam adapts a per-parameter step size |
# | `augment=False, epochs=15` | train → 100%, validation stalls | overfitting |
# | `label_smoothing=0.1` | slightly lower confidence | softer targets, better calibration |
#
# Post your best **validation accuracy + settings** on the class leaderboard (link on the slide).

# %%
# 🎛️ CHALLENGE KNOBS — edit and re-run this cell as often as you like
KNOBS = dict(
    lr=1e-3,              # try 1.0, 1e-2, 1e-4, 1e-5
    batch_size=32,        # try 4, 16, 64, 128
    optimizer="adamw",    # "adamw" | "adam" | "sgd"
    epochs=5,             # try 2 or 10
    augment=True,
    label_smoothing=0.0,  # try 0.1
)

set_seed(42)
ccfg = vision_config_for(device, **KNOBS)
c_loaders = make_vision_loaders(ccfg, S["beans_dir"], device)
c_model = build_vision_model(ccfg.model_name, 3, ccfg.pretrained, freeze_backbone=True)
c_opt = build_optimizer(ccfg.optimizer, trainable_params(c_model), ccfg.lr, ccfg.weight_decay)
t0 = time.time()
c_hist = fit(c_model, c_loaders["train"], c_loaders["validation"], epochs=ccfg.epochs, optimizer=c_opt,
             device=device, amp=ccfg.amp, label_smoothing=ccfg.label_smoothing, name="challenge")
print(f"\n📋 LEADERBOARD ENTRY → best val acc {c_hist.best_val_acc:.1%} | {KNOBS} | {time.time()-t0:.0f}s on {device.type}")

# %% [markdown]
# ### 3.8 Transfer learning step 3: fine-tune the top of the backbone
# Early layers learn generic features (edges, colours) useful for any image; the **last block**
# (`layer4`) learns combinations specific to ImageNet objects. We unfreeze it and use
# **discriminative learning rates**: head $10^{-3}$, backbone $10^{-4}$ — big steps for the random head,
# gentle steps for weights that are already good (avoids *catastrophic forgetting*).
#
# Trade-off: more trainable parameters → better fit, but more GPU memory, slower epochs, and more risk
# of overfitting a small dataset.

# %%
from dlhandson.models import unfreeze_top, param_groups
from dlhandson.train import build_scheduler, save_checkpoint
from dlhandson.catchup import CKPT_DIR

model = trained_vision_model(S)                     # head trained (cached) — our starting point
unfreeze_top(model, n_blocks=1)
print(f"trainable now: {count_parameters(model)['trainable']:,} parameters")
ft_epochs = max(1, vcfg.epochs - 1)
opt = torch.optim.AdamW(param_groups(model, lr=1e-3, backbone_lr_mult=0.1, weight_decay=vcfg.weight_decay))
sched = build_scheduler(opt, total_steps=ft_epochs * len(loaders["train"]), kind="cosine")
hist_ft = fit(model, loaders["train"], loaders["validation"], epochs=ft_epochs, optimizer=opt, scheduler=sched,
              device=device, amp=vcfg.amp, name="fine-tune layer4",
              ckpt_path=CKPT_DIR / f"vision_{vcfg.model_name}_{vcfg.image_size}_finetuned.safetensors")
vision_model = model
RESULTS["vision_finetune_val_acc"] = hist_ft.best_val_acc
print(f"\nhead only {hist_aug.best_val_acc:.1%}  →  + fine-tuned layer4 {hist_ft.best_val_acc:.1%}")

# %% [markdown]
# ### 3.9 Batch size vs GPU memory — *watch it crash (safely)*
# 🧮 Memory for one training step ≈ weights + gradients + optimizer state (Adam keeps 2 extra copies)
# **+ activations saved for backprop**. Activations grow **linearly with batch size** and with the
# number of pixels (quadratically with image side). Bigger batches train faster per image… until the
# GPU runs out of memory. We fully unfreeze ResNet-50 (worst case) and double the batch until it breaks.
# The error is **caught**, so the notebook survives.

# %%
from dlhandson.debug_tools import find_max_batch_size
if device.type == "cuda" and not smoke_mode():
    big = build_vision_model(vcfg.model_name, 3, pretrained=False, freeze_backbone=False)
    rows = find_max_batch_size(big, vcfg.image_size, device, start=32, limit=1024)
    del big
    ok = [r["batch_size"] for r in rows if r["ok"]]
    RESULTS["max_batch_full_finetune_fp32"] = max(ok) if ok else None
    print("\nFixes when you hit the ceiling: mixed precision (≈ half the activation memory), smaller images,\n"
          "gradient accumulation (several small batches, one optimizer step), or freeze more layers.")
else:
    find_max_batch_size(nn.Linear(1, 1), vcfg.image_size, device)

# %% [markdown]
# ### 3.10 Final exam: the test set (touch it once)
# 🧮 Per-class metrics, with TP/FP/FN counted for that class:
# $\text{precision} = \frac{TP}{TP+FP}$ ("when it says rust, is it rust?"),
# $\text{recall} = \frac{TP}{TP+FN}$ ("of all rust leaves, how many did it catch?"),
# $F_1 = \frac{2PR}{P+R}$. For a disease scanner, **recall on diseased leaves** is what farmers care about.

# %%
from sklearn.metrics import classification_report
test_out = evaluate(vision_model, loaders["test"], device, return_outputs=True)
print(f"TEST accuracy: {test_out['acc']:.1%}\n")
print(classification_report(test_out["labels"], test_out["preds"], target_names=CLASSES, digits=3))
viz.plot_confusion_matrix(test_out["labels"], test_out["preds"], CLASSES, title="Test set — leaf disease")
RESULTS["vision_test_acc"] = test_out["acc"]

# %%
# Look at the mistakes — the fastest way to learn what the model gets wrong.
test_raw = image_folder(S["beans_dir"], "test")
wrong = np.where(test_out["preds"] != test_out["labels"])[0][:10]
if len(wrong):
    viz.show_images([test_raw[i][0] for i in wrong],
                    [f"true {CLASSES[test_out['labels'][i]]}\npred {CLASSES[test_out['preds'][i]]} "
                     f"({test_out['probs'][i].max():.0%})" for i in wrong], ncols=5, size=2.6)
else:
    print("No mistakes on the test set!")

# %% [markdown]
# ### 3.11 Stress test: what happens with photos it has never seen?
# Production inputs are messy. We test four kinds of "unusual" images:
# 1. **Bean leaves in harder conditions** — blur, dusk, glare, heavy compression, zoomed into a corner
# 2. **Not a leaf at all** — a coffee cup, a cat
# 3. **The instructor's photos** — a house-plant leaf, a tree leaf, something green that isn't a leaf
# 4. **Synthetic nonsense** — random noise, a flat green square
#
# The model has only three outputs, and softmax probabilities **must sum to 1** — so it *will* pick one.

# %%
from dlhandson.ood import corrupt_images, load_folder_images, synthetic_ood_images, predict_images
from dlhandson.data_vision import eval_transform

STRESS = ROOT / "data" / "stress_test"
picks = [0, len(test_raw) // 2, len(test_raw) - 1]
stress = corrupt_images([test_raw[i][0] for i in picks], [CLASSES[test_raw[i][1]] for i in picks])[:10]
stress += load_folder_images(STRESS / "non_leaf", "not a leaf")
stress += load_folder_images(STRESS / "instructor_photos", "instructor photo")
stress += synthetic_ood_images(vcfg.image_size)
tfm = eval_transform(vcfg.image_size)
stress = predict_images(vision_model, stress, tfm, device, CLASSES)
viz.show_predictions(stress, ncols=5)
pd.DataFrame([{k: it[k] for k in ("group", "name", "pred", "confidence")} for it in stress]).round(3)

# %% [markdown]
# Notice the **confident nonsense**: a cat or a coffee cup can get a disease label at high confidence.
# That is not a bug in our code — it is how softmax classifiers behave outside their training
# distribution, and it is one of the most common ways ML fails in production.
#
# ### 3.12 The fix: confidence threshold → "ask a human" (selective prediction)
# 🧮 Confidence = maximum softmax probability $\max_k p_k$. Uncertainty = entropy
# $H(p) = -\sum_k p_k \log p_k$ (0 = certain, $\log 3 \approx 1.10$ = clueless).
# Rule: act automatically only if $\max_k p_k \ge \tau$, otherwise route to a human.
# Raising $\tau$ trades **coverage** (share automated) for **accuracy** on what is automated.
# We pick $\tau$ on the **validation** set to hit a business target (here: 98% accuracy on automatic
# decisions), then apply it everywhere.

# %%
from dlhandson.ood import selective_curve, pick_threshold, route
val_out = evaluate(vision_model, loaders["validation"], device, return_outputs=True)
curve = selective_curve(val_out["probs"], val_out["labels"])
TAU = pick_threshold(curve, target_accuracy=0.98)
viz.plot_selective(curve, TAU, "Leaf model: confidence threshold trade-off (validation set)")

test_conf = test_out["probs"].max(1)
auto = test_conf >= TAU
auto_acc = (test_out['preds'][auto] == test_out['labels'][auto]).mean() if auto.any() else None
print(f"τ = {TAU:.2f} → on the TEST set: {auto.mean():.0%} decided automatically, accuracy on those = "
      f"{'n/a' if auto_acc is None else f'{auto_acc:.1%}'}; {(~auto).sum()} photos sent to an agronomist.")
stress_routes = pd.Series([route(it["confidence"], TAU) for it in stress]).value_counts()
print("stress-test images:", stress_routes.to_dict())
viz.show_predictions(stress, ncols=5, threshold=TAU)
RESULTS.update(vision_tau=TAU, vision_test_coverage=float(auto.mean()))

# %% [markdown]
# **Honest limits:** a threshold catches many odd inputs but not all (some nonsense still scores high).
# Production systems add more defences: an explicit **"not a leaf"** class trained on negative examples,
# dedicated OOD detectors, **calibration** (temperature scaling: $p = \mathrm{softmax}(z/T)$ with $T$ fit
# on validation), input checks (blur/brightness), and **monitoring** of the confidence distribution over
# time (data drift).
#
# ### 3.13 📱 Try your own photo *(optional — off by default so "Run all" never stops)*
# Set `TRY_MY_PHOTO = True` and run: a file picker opens (photos from your phone's gallery work if
# they are on this computer — AirDrop / Google Photos / email them to yourself). Easier on a phone?
# In **section 5.7** we launch a web app with a QR code — open it on your phone and upload straight
# from your gallery or camera.

# %%
TRY_MY_PHOTO = False   # 🎛️ set to True, then run this cell
from dlhandson.notebook_utils import upload_photos
if TRY_MY_PHOTO:
    upload_photos(STRESS / "my_photos")
    mine = predict_images(vision_model, load_folder_images(STRESS / "my_photos", "my photo"), tfm, device, CLASSES)
    viz.show_predictions(mine, ncols=4, threshold=TAU)

# %% [markdown]
# ### 3.14 ⏩ Instructor demo: swap in a Vision Transformer (two-line change)
# A **ViT** cuts the image into 16×16 patches, treats them as a sequence of "tokens", and applies
# self-attention — the same architecture family as BERT and GPT. In our code it is a config change.
# ViT-B/16 has **86M** parameters (3.7× ResNet-50) and a ~330 MB download, which is why we demo it
# rather than everyone running it.

# %%
RUN_VIT_DEMO = device.type == "cuda" and not smoke_mode()
if RUN_VIT_DEMO:
    set_seed(42)
    vit_cfg = vision_config_for(device, model_name="vit_b_16", epochs=3)             # change 1
    vit = build_vision_model("vit_b_16", 3, pretrained=True, freeze_backbone=True)  # change 2
    vit_opt = build_optimizer("adamw", trainable_params(vit), 1e-3, vit_cfg.weight_decay)
    hist_vit = fit(vit, loaders["train"], loaders["validation"], epochs=vit_cfg.epochs, optimizer=vit_opt,
                   device=device, amp=True, name="ViT-B/16 head")
    print(f"ViT-B/16 head-only: {hist_vit.best_val_acc:.1%}  vs  ResNet-50 head-only: {hist_aug.best_val_acc:.1%}")
    RESULTS["vit_head_val_acc"] = hist_vit.best_val_acc
    del vit
else:
    print("⏩ Skipped (needs a CUDA GPU). The instructor shows this result on screen.")

# %% [markdown]
# ---
# ## 4. 💬 Ticket SUPPORT-202 — routing customer messages with a Transformer *(20 min)*
#
# > **SUPPORT-202** · *As a digital bank, we receive thousands of chat messages a day. Route each one to
# > one of 77 intents (card arrival, refund, top-up…) so customers get the right answer fast.*
# > **Acceptance:** ≥ 90% test accuracy · low-confidence or off-topic messages go to a human agent.
#
# **Data:** *Banking77* (PolyAI, CC BY 4.0) — 13,083 real-style customer queries, 77 fine-grained intents.
# 🏭 Same pattern: insurance claim intake, IT help-desk ticket triage, e-mail routing, content moderation.
#
# **Same recipe as the leaves:** pretrained backbone (DistilBERT) → new head (768 → 77) → fine-tune.

# %%
# 🛟 CATCH-UP (text): rebuilds config, data, tokenizer and loaders. Safe to run any time.
timer.start("4. NLP fine-tuning")
from dlhandson.catchup import text_state, trained_text_model, CKPT_DIR
from dlhandson.models import build_text_model, count_parameters, trainable_params
from dlhandson.train import build_optimizer, build_scheduler, fit, evaluate
from dlhandson.tracking import ExperimentLogger
from dlhandson.ood import selective_curve, pick_threshold
T = text_state(device)
tcfg, tdata, tok, tloaders = T["cfg"], T["data"], T["tokenizer"], T["loaders"]
print(tcfg)

# %% [markdown]
# ### 4.1 Look at the data

# %%
print(f"{len(tdata['train']):,} train | {len(tdata['validation']):,} validation | {len(tdata['test']):,} test messages, "
      f"{len(tdata['label_names'])} intents")
counts = tdata["train"]["category"].value_counts()
print(f"messages per intent: min {counts.min()}, median {int(counts.median())}, max {counts.max()}")
tdata["train"].sample(8, random_state=1)[["text", "category"]]

# %% [markdown]
# ### 4.2 Tokenisation: text → integers
# Networks only eat numbers. DistilBERT's **WordPiece** tokenizer splits text into sub-words from a
# 30,522-entry vocabulary, so rare words become known pieces (`##` marks a continuation) and nothing is
# ever "out of vocabulary". Special tokens: `[CLS]` (its final vector summarises the sentence) and `[SEP]`.

# %%
example = "My contactless payment isn't working at the supermarket"
enc = tok(example)
print("tokens        :", tok.convert_ids_to_tokens(enc["input_ids"]))
print("input_ids     :", enc["input_ids"])
print("attention_mask:", enc["attention_mask"])

from dlhandson.data_text import token_lengths
lengths = token_lengths(tok, tdata["train"]["text"])
viz.plot_histogram(lengths, "Tokens per message", "tokens", marker=tcfg.max_length,
                   marker_label=f"max_length = {tcfg.max_length}")
print(f"95th percentile: {np.percentile(lengths, 95):.0f} tokens → max_length={tcfg.max_length} truncates "
      f"{(lengths > tcfg.max_length).mean():.1%} of messages")

# %% [markdown]
# 🧮 **Why not pad everything to 512?** Self-attention compares every token with every other token:
# cost $\propto n^2$ in sequence length $n$. Padding a 20-token message to 512 multiplies the work by
# ~650× for zeros. We use **dynamic padding**: each batch is padded only to its own longest message.
#
# ### 4.3 The Transformer in one formula
# Each token's vector is projected to a **query** $q$, **key** $k$ and **value** $v$. Every token
# attends to every other token:
# $$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
# The softmax weights say *"how much should this word look at that word"* — e.g. "isn't" attends to
# "working". Stack this (multi-head, 6 layers in DistilBERT, 768 dimensions) and you get contextual
# meaning. **GPT and other LLMs are the same building block, scaled up.**
#
# ### 4.4 Experiment A — freeze the encoder, train only the head (like the leaves)

# %%
from dlhandson.models import build_text_model
from dlhandson.train import build_scheduler

set_seed(42)
frozen = build_text_model(tcfg.model_name, 77, tdata["id2label"], tdata["label2id"], freeze_encoder=True,
                          vocab_size=len(tok))
print(f"trainable: {count_parameters(frozen)['trainable']:,} of {count_parameters(frozen)['total']:,} parameters")
opt = build_optimizer("adamw", trainable_params(frozen), 1e-3, 0.0)
hist_frozen = fit(frozen, tloaders["train"], tloaders["validation"], epochs=max(1, tcfg.epochs - 2), optimizer=opt,
                  device=device, amp=tcfg.amp, name="frozen encoder")
RESULTS["text_frozen_val_acc"] = hist_frozen.best_val_acc
del frozen

# %% [markdown]
# ### 4.5 Experiment B — full fine-tuning
# Now every one of the 66M weights moves, but **gently**:
# - small learning rate $\eta = 5\times10^{-5}$ (pretrained weights are already good; big steps cause
#   catastrophic forgetting),
# - **warm-up** for the first 10% of steps, then linear decay to 0,
# - **gradient clipping** at $\lVert g\rVert_2 \le 1$ and mixed precision on CUDA.

# %%
set_seed(42)
text_model = build_text_model(tcfg.model_name, 77, tdata["id2label"], tdata["label2id"], vocab_size=len(tok))
opt = build_optimizer("adamw", trainable_params(text_model), tcfg.lr, tcfg.weight_decay)
total = tcfg.epochs * len(tloaders["train"])
sched = build_scheduler(opt, total, warmup_steps=int(tcfg.warmup_ratio * total), kind="linear")
viz.plot_lr_schedule([sched.lr_lambdas[0](s) * tcfg.lr for s in range(total)], "Warm-up + linear decay")
logger = ExperimentLogger("text-finetune", config=tcfg.to_dict())
hist_ft_text = fit(text_model, tloaders["train"], tloaders["validation"], epochs=tcfg.epochs, optimizer=opt,
                   scheduler=sched, device=device, amp=tcfg.amp, grad_clip=tcfg.grad_clip, logger=logger,
                   name="full fine-tune",
                   ckpt_path=CKPT_DIR / f"text_{tcfg.model_name.split('/')[-1]}_{tcfg.profile}.safetensors")
logger.close()
viz.plot_history({"frozen encoder": hist_frozen, "full fine-tune": hist_ft_text}, "Freeze vs fine-tune (text)")
RESULTS["text_finetune_val_acc"] = hist_ft_text.best_val_acc

# %% [markdown]
# **Why did freezing work for leaves but not as well here?** ImageNet features (textures, colours,
# shapes) already separate leaf diseases. A sentence vector pretrained for *general* language doesn't
# yet separate 77 very similar banking intents ("card arrival" vs "card delivery estimate"). Rule of thumb:
#
# | | Similar domain to pre-training | Different domain |
# |---|---|---|
# | **Little data** | freeze, train head | freeze most, fine-tune top layers |
# | **Lots of data** | fine-tune all (small LR) | fine-tune all, or pre-train further |
#
# ### 4.6 Evaluate on the test set
# 🧮 With 77 classes we also report **macro-F1** = the plain average of per-class F1 scores, so rare
# intents count as much as common ones.

# %%
from sklearn.metrics import f1_score
t_out = evaluate(text_model, tloaders["test"], device, return_outputs=True)
macro_f1 = f1_score(t_out["labels"], t_out["preds"], average="macro")
print(f"TEST accuracy {t_out['acc']:.1%} | macro-F1 {macro_f1:.3f}")
RESULTS.update(text_test_acc=t_out["acc"], text_test_macro_f1=float(macro_f1))
viz.most_confused(t_out["labels"], t_out["preds"], tdata["label_names"], k=8)

# %% [markdown]
# Read the confused pairs: many are **genuinely ambiguous even for humans**. That is a data/label-design
# conversation with the business, not just a modelling problem.
#
# ### 4.7 🎛️ Try your own messages + "route to a human"

# %%
from dlhandson.demo_app import make_text_predictor
t_val = evaluate(text_model, tloaders["validation"], device, return_outputs=True)
TAU_TEXT = pick_threshold(selective_curve(t_val["probs"], t_val["labels"]), target_accuracy=0.97)
predict_text = make_text_predictor(text_model, tok, tdata["id2label"], device, threshold=TAU_TEXT,
                                   max_length=tcfg.max_length)
MY_MESSAGES = [                                   # 🎛️ add your own
    "I still haven't received my new card, it's been two weeks",
    "Why was I charged twice for the same coffee?",
    "Can I add money using Apple Pay?",
    "My card got swallowed by the ATM!!",
    "What's the weather like in Portland today?",   # off-topic → should be unsure
]
rows = []
for m in MY_MESSAGES:
    probs, decision = predict_text(m)
    top, p = max(probs.items(), key=lambda kv: kv[1])
    rows.append({"message": m, "top intent": top, "confidence": round(p, 3),
                 "decision": "auto" if p >= TAU_TEXT else "human agent"})
print(f"threshold τ = {TAU_TEXT:.2f} (chosen on validation for 97% accuracy on automatic decisions)")
pd.DataFrame(rows)

# %% [markdown]
# ---
# ## 5. Deep learning in the real world: debugging & MLOps *(20 min)*
# **"Why is my model not learning?"** — the checklist professionals run, in order:
# 1. **Overfit a single batch** (the golden rule)
# 2. Check the **optimizer actually holds your parameters**
# 3. Watch for **exploding** (→ NaN) and **vanishing** (→ nothing learns) gradients
# 4. Then — and only then — tune hyperparameters, add data, or change architecture

# %%
# 🛟 CATCH-UP (debugging): only the leaf data is needed here.
timer.start("5. Pitfalls & MLOps")
from dlhandson.catchup import vision_state, trained_vision_model, CKPT_DIR
from dlhandson.data_vision import make_vision_loaders, eval_transform
from dlhandson.models import build_vision_model, trainable_params
from dlhandson.train import save_checkpoint
from dlhandson.debug_tools import overfit_one_batch, optimizer_covers_model
if "S" not in globals():
    S = vision_state(device)
vcfg, loaders, CLASSES = S["cfg"], S["loaders"], S["class_names"]
tfm = eval_transform(vcfg.image_size)
if "vision_model" not in globals():          # skipped section 3? load/train the leaf model
    vision_model = trained_vision_model(S)
TAU = globals().get("TAU", 0.8)
one_batch_cfg = vision_config_for(device, batch_size=16, augment=False)
small_batch = next(iter(make_vision_loaders(one_batch_cfg, S["beans_dir"], device)["train"]))
print("batch for the test:", tuple(small_batch["pixel_values"].shape))

# %% [markdown]
# ### 5.1 The golden rule: overfit one batch
# If a model **cannot reach 100% training accuracy on 16 images**, more data or more epochs will not
# help — your code has a bug. ✅ First the correct pipeline:

# %%
set_seed(0)
m = build_vision_model(vcfg.model_name, 3, vcfg.pretrained, freeze_backbone=True)
ok = overfit_one_batch(m, small_batch, device, lr=1e-2, steps=60)

# %% [markdown]
# 🐞 Now a **real bug** that costs teams days: the optimizer is created, *then* someone swaps the head
# ("let me just add dropout…"). The optimizer still points at the **old** head's tensors.

# %%
set_seed(0)
buggy = build_vision_model(vcfg.model_name, 3, vcfg.pretrained, freeze_backbone=True)
buggy_opt = torch.optim.AdamW(trainable_params(buggy), lr=1e-2)        # optimizer built here…
buggy.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(buggy.fc.in_features, 3))   # …head replaced AFTER
bad = overfit_one_batch(buggy, small_batch, device, optimizer=buggy_opt, steps=60)
optimizer_covers_model(buggy_opt, buggy)

# %%
# 🔧 The fix: (re)build the optimizer AFTER the model is final, then re-run the golden-rule test.
fixed_opt = torch.optim.AdamW(trainable_params(buggy), lr=1e-2)
optimizer_covers_model(fixed_opt, buggy)
_ = overfit_one_batch(buggy, small_batch, device, optimizer=fixed_opt, steps=60)
RESULTS["overfit_one_batch"] = {"correct": ok["passed"], "bug_detected": not bad["passed"]}

# %% [markdown]
# 🏭 Teams turn this into an **automated test** (`tests/test_debugging.py`) that runs on every code
# change, so a broken training loop is caught in CI, not after a week of GPU time.
#
# ### 5.2 Exploding gradients → `inf` → `NaN`
# When the learning rate is too large (or weights/activations blow up), each update overshoots,
# the loss grows geometrically, overflows to `inf`, and then `inf − inf = NaN` poisons every weight.

# %%
print("exp(1000) =", torch.exp(torch.tensor(1000.)).item(), "  |  inf - inf =", (torch.tensor(float("inf")) - float("inf")).item())
from dlhandson.debug_tools import exploding_experiment
runs = exploding_experiment(lrs=(0.001, 0.01, 0.05, 0.1), steps=40)
viz.plot_curves({f"lr={lr}" + (" 💥 NaN" if any(math.isnan(v) for v in ls) else ""): ls for lr, ls in runs.items()},
                "Same network, four learning rates (gaps = loss became NaN)", logy=True)
clipped = exploding_experiment(lrs=(0.1,), steps=40, clip=1.0)[0.1]
print(f"lr=0.1 without clipping: {'NaN' if math.isnan(runs[0.1][-1]) else runs[0.1][-1]:}  |  "
      f"with gradient clipping (‖g‖ ≤ 1): final loss {clipped[-1]:.3f}")

# %% [markdown]
# 🧮 **Gradient clipping:** if $\lVert g\rVert_2 > \tau$ then $g \leftarrow g\,\tau/\lVert g\rVert_2$ — same
# direction, capped length. Other fixes: lower LR, LR warm-up, normalisation layers, check inputs for
# NaN/inf. `dlhandson.train.fit` has a **NaN guard**: it stops cleanly and tells you why, instead of
# silently training a broken model.
#
# ### 5.3 Vanishing gradients — why depth used to be hard
# 🧮 Backprop multiplies one local derivative per layer. The sigmoid's derivative is at most 0.25, so
# after 20 layers: $0.25^{20} \approx 10^{-12}$ → the first layers receive (almost) no signal.
# Fixes that made deep learning work: **ReLU**, **He initialisation**, **normalisation**, and
# **residual connections** (ResNet).

# %%
from dlhandson.debug_tools import vanishing_gradient_experiment
norms = vanishing_gradient_experiment(depth=20)
viz.plot_grad_norms(norms, "Gradient size per layer in a 20-layer network")
for name, g in norms.items():
    print(f"{name:>16}: first layer {g[0]:.1e} vs last layer {g[-1]:.1e}  (ratio {g[0]/g[-1]:.0e})")

# %% [markdown]
# ### 5.4 Experiment tracking
# Every `fit` above was logged to `runs/` (TensorBoard + CSV + the exact config). Open the dashboard:
# compare runs, spot divergence, share results. Prefer hosted dashboards? Weights & Biases works the
# same way — set `DLH_WANDB=1` and store your key as a **Colab Secret** (🔑 icon) named `WANDB_API_KEY`.
# **Never paste API keys into a notebook.**

# %%
if not headless():
    get_ipython().run_line_magic("load_ext", "tensorboard")
    get_ipython().run_line_magic("tensorboard", "--logdir runs")
else:
    print("(headless run: TensorBoard UI skipped)")

# %% [markdown]
# ### 5.5 Save, reload, document
# - **safetensors** instead of pickle: `torch.save` uses Python pickle, and *loading a malicious pickle
#   can run code on your machine*. safetensors stores raw tensors only.
# - Always **round-trip test**: reload the file into a fresh model and check predictions match.
# - Ship a **model card** (Mitchell et al., 2019): what it does, data, metrics, limits, intended use.

# %%
from dlhandson.train import load_checkpoint
EXPORT = ROOT / "runs" / "export"
path = save_checkpoint(vision_model, EXPORT / f"leaf_{vcfg.model_name}.safetensors")
fresh = build_vision_model(vcfg.model_name, 3, pretrained=False, freeze_backbone=True)
load_checkpoint(fresh, path, device)
x = next(iter(loaders["test"]))["pixel_values"][:8].to(device)
vision_model.eval(); fresh.eval()
with torch.no_grad():
    same = torch.allclose(vision_model(x).float(), fresh.to(device)(x).float(), atol=1e-4)
print(f"saved {path.name} ({path.stat().st_size/1e6:.0f} MB) | reloaded predictions identical: {same}")

card = f"""# Model card — LEAF-101 bean leaf disease classifier
- **Model:** {vcfg.model_name} (ImageNet-pretrained) + linear head, top block fine-tuned
- **Data:** Beans (Makerere AI Lab, Uganda; MIT). Field photos, 3 classes.
- **Test accuracy:** {RESULTS.get('vision_test_acc', float('nan')):.1%} (128 held-out photos)
- **Decision rule:** auto-diagnose only if confidence ≥ {RESULTS.get('vision_tau', float('nan')):.2f}; otherwise refer to an agronomist
- **Intended use:** decision support for bean growers; NOT a replacement for an agronomist.
- **Known limits:** trained on one region, one crop, 2 diseases. Unknown behaviour on other crops, regions,
  cameras, or diseases. Confident mistakes on non-leaf images are possible (see stress test).
"""
(EXPORT / "MODEL_CARD.md").write_text(card)
print(card)

# %% [markdown]
# ### 5.6 From notebook to production code
# The notebook is for learning. The **same logic** lives in a tested Python package — that is what a
# team reviews, versions, and deploys:

# %%
from dlhandson.notebook_utils import show_tree
show_tree(ROOT, max_depth=2)
print("\nRunning the automated test suite (tiny models, ~30 s)…")
r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-o", "addopts=", "-q"], capture_output=True, text=True,
                   env={**os.environ, "DLH_SMOKE": "1", "DLH_DEVICE": "cpu"}, cwd=ROOT)
print(r.stdout[-600:] or r.stderr[-600:])

# %% [markdown]
# ### 5.7 📱 Ship it: a web app on your phone
# A model nobody can use has zero business value. **Gradio** wraps both models in a web UI;
# `share=True` creates a temporary **public HTTPS link** — scan the QR code with your phone, then
# **upload from your photo gallery or take a picture**, or type a support message.
#
# 🔒 The link is public while this notebook runs and expires automatically (about a week); photos go to
# *your* Colab machine only. Don't upload photos of people or documents. Stop it with `demo.close()`.

# %%
from dlhandson.demo_app import build_demo, launch, qr_code, make_vision_predictor, make_text_predictor
from IPython.display import display
if "predict_text" not in globals():           # skipped section 4? load/train the text model
    T = text_state(device)
    predict_text = make_text_predictor(trained_text_model(T), T["tokenizer"], T["data"]["id2label"], device,
                                       threshold=0.7, max_length=T["cfg"].max_length)
vision_predict = make_vision_predictor(vision_model, tfm, CLASSES, device, threshold=TAU)
demo = build_demo(vision_predict, predict_text)
if headless():
    print("(headless run: app built but not launched)")
else:
    url = launch(demo, share=True)
    if url:
        print("📱 Open on your phone:", url)
        display(qr_code(url).resize((260, 260)))

# %%
# When you are done with the app (frees the public link):
# demo.close()

# %% [markdown]
# ### 5.8 Security & responsible-AI checklist (what reviewers will ask you)
# | Area | Practice used in this project |
# |---|---|
# | Secrets | Tokens live in **Colab Secrets** / environment variables; a git **pre-commit hook** blocks token-like strings |
# | Model files | **safetensors** (no pickle code execution); `trust_remote_code` never enabled |
# | Downloads | Dataset pack verified by **SHA-256**; zip extraction blocks path traversal ("zip-slip") |
# | Privacy | Instructor photos have **EXIF/GPS stripped**; raw phone photos are git-ignored; demo link expires |
# | Licences | Beans (MIT), Banking77 (CC BY 4.0), sample photos (CC0) — documented in the README |
# | Human oversight | Low-confidence predictions → **human review** (agronomist / support agent) |
# | Transparency | Model card with intended use and known limits |
# | Bias & coverage | Leaf photos come from one region; text is UK/US banking English — test before using elsewhere |
#
# ---
# ## 6. Horizons — where this goes next *(5 min)*
# | Today you used | Scaled up, it becomes |
# |---|---|
# | DistilBERT (66M, encoder) | **LLMs** (decoder-only Transformers, billions of parameters) trained to predict the next token: $L = -\sum_t \log p_\theta(x_t \mid x_{<t})$ |
# | Fine-tuning all weights | **Parameter-efficient fine-tuning (LoRA):** keep $W$ frozen, learn a low-rank update $W' = W + BA$ with rank $r \ll d$ — fine-tune big models on one GPU |
# | A labelled dataset | **Prompting** and **retrieval-augmented generation (RAG)**: give an LLM your documents at question time |
# | Classifiers ($p(y\mid x)$) | **Generative models** ($p(x)$): **GANs** — a generator $G$ and discriminator $D$ play $\min_G \max_D\ \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$; **diffusion models** learn to remove noise step by step (image and video generation) |
# | ResNet / ViT | **Multimodal** models that read images and text together (vision-language models) |
#
# **What to learn next:** the PyTorch tutorials, the Hugging Face course (free), *Dive into Deep
# Learning* (d2l.ai, free), fast.ai, and — most of all — **build a project with data you care about.**

# %%
timer.start("6. Wrap-up")
print("⏱️ Time per section (instructor uses this to tune the schedule):\n")
timer.print_summary()
RESULTS["timings_minutes"] = timer.summary()
(ROOT / "runs").mkdir(exist_ok=True)
(ROOT / "runs" / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
print("\nKey results:")
for k, v in RESULTS.items():
    if isinstance(v, float):
        print(f"  {k:<32} {v:.3f}")

# %% [markdown]
# ## 🎉 What you built today
# - The **PyTorch mental model**: tensors, autograd (checked against hand-derived gradients), `nn.Module`, the 5-step loop
# - A **leaf-disease classifier** via transfer learning: frozen backbone → new head → fine-tuning, with augmentation
# - A **customer-intent router** with a fine-tuned Transformer — the same recipe on text
# - **Engineering judgement:** learning rate, batch size vs memory, overfitting, freeze vs fine-tune
# - **Debugging:** overfit-one-batch, optimizer bug, exploding and vanishing gradients
# - **Production habits:** tracking, safetensors, model cards, tests, a confidence threshold with a human in the loop, and a live app
#
# Questions → now. Code → `github.com/Sushanth-Manakhari/deep-learning-handson-lecture`.

# %% [markdown]
# ---
# ## Appendix — data, models and references
# **Datasets**
# | Dataset | Licence | Link | How this notebook gets it |
# |---|---|---|---|
# | Beans — bean leaf photos (Makerere University AI Lab) | MIT | [Hugging Face](https://huggingface.co/datasets/AI-Lab-Makerere/beans) · [original GitHub](https://github.com/AI-Lab-Makerere/ibean/) | GitHub Release `beans_224.zip` (SHA-256 checked) → fallback Hugging Face parquet |
# | Banking77 — customer intents (PolyAI) | CC BY 4.0 | [GitHub](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data) · [paper](https://arxiv.org/abs/2003.04807) | committed in `data/banking77/` |
# | Coffee cup & cat photos (scikit-image samples) | CC0 | [scikit-image data](https://github.com/scikit-image/scikit-image/tree/v0.25.2/skimage/data) | committed in `data/stress_test/non_leaf/` |
#
# **Pretrained models**
# | Model | Params | Licence | Link | Weights file |
# |---|---|---|---|---|
# | ResNet-50 (ImageNet-1K V2) | 25.6M | BSD-3 (torchvision) | [docs](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html) · [paper](https://arxiv.org/abs/1512.03385) | `download.pytorch.org/models/resnet50-11ad3fa6.pth` |
# | ResNet-18 (ImageNet-1K V1, CPU mode) | 11.7M | BSD-3 (torchvision) | [docs](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) | `download.pytorch.org/models/resnet18-f37072fd.pth` |
# | ViT-B/16 (ImageNet-1K V1, demo) | 86.6M | BSD-3 (torchvision) | [docs](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html) · [paper](https://arxiv.org/abs/2010.11929) | `download.pytorch.org/models/vit_b_16-c867db91.pth` |
# | DistilBERT base uncased | 66M | Apache-2.0 | [Hugging Face](https://huggingface.co/distilbert/distilbert-base-uncased) · [paper](https://arxiv.org/abs/1910.01108) | `model.safetensors` on the Hub |
#
# **Key papers:** [ResNet](https://arxiv.org/abs/1512.03385) · [He init](https://arxiv.org/abs/1502.01852) ·
# [Transformer](https://arxiv.org/abs/1706.03762) · [BERT](https://arxiv.org/abs/1810.04805) ·
# [ViT](https://arxiv.org/abs/2010.11929) · [AdamW](https://arxiv.org/abs/1711.05101) ·
# [Mixed precision](https://arxiv.org/abs/1710.03740) · [MSP baseline for OOD](https://arxiv.org/abs/1610.02136) ·
# [Calibration](https://arxiv.org/abs/1706.04599) · [Model cards](https://arxiv.org/abs/1810.03993) ·
# [LoRA](https://arxiv.org/abs/2106.09685) · [GANs](https://arxiv.org/abs/1406.2661) · [Diffusion (DDPM)](https://arxiv.org/abs/2006.11239)
#
# **Free learning resources:** [PyTorch tutorials](https://docs.pytorch.org/tutorials/) ·
# [Hugging Face course](https://huggingface.co/learn) · [Dive into Deep Learning](https://d2l.ai) · [fast.ai](https://course.fast.ai)
