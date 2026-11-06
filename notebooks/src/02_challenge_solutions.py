# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🎛️ Challenge solutions + take-home extensions
# **Deep Learning, Hands-On · The Roux Institute at Northeastern University**
#
# Part A re-runs the in-class challenge as a small **hyperparameter sweep** and explains every result.
# Part B gives two take-home projects that reuse the same recipe on new domains.

# %%
# ⚙️ SETUP
import importlib.util, os, pathlib, shutil, subprocess, sys, time
REPO_URL = "https://github.com/Sushanth-Manakhari/deep-learning-handson-lecture.git"
os.environ.setdefault("HF_HUB_VERBOSITY", "error")  # public models need no token: hide the "unauthenticated" notice
REPO_BRANCH = "main"
def find_repo():
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
os.chdir(ROOT); sys.path.insert(0, str(ROOT / "src"))
if not (pathlib.Path("/content").is_dir() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ):
    # local laptop: keep downloaded models inside the project folder, not in ~/.cache
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

import pandas as pd, torch
from dlhandson import viz
from dlhandson.catchup import vision_state
from dlhandson.config import vision_config_for
from dlhandson.data_vision import make_vision_loaders
from dlhandson.device import get_device, set_seed, smoke_mode
from dlhandson.models import build_vision_model, trainable_params
from dlhandson.train import build_optimizer, fit
device = get_device()
S = vision_state(device)

# %% [markdown]
# ## Part A — the challenge as a sweep
# Each row changes **one** knob from the baseline (lr 1e-3, batch 32, AdamW, augmentation on), so the
# effect is attributable to that knob — the core habit of good experimentation.

# %%
EPOCHS = 1 if smoke_mode() else 4
SWEEP = {
    "baseline":                 {},
    "lr = 1.0 (too high)":      {"lr": 1.0},
    "lr = 1e-5 (too low)":      {"lr": 1e-5},
    "batch 4":                  {"batch_size": 4},
    "batch 128":                {"batch_size": 128},
    "SGD (same lr)":            {"optimizer": "sgd"},
    "SGD, lr = 1e-2":           {"optimizer": "sgd", "lr": 1e-2},
    "no augmentation":          {"augment": False},
    "label smoothing 0.1":      {"label_smoothing": 0.1},
}
rows = []
for name, knobs in SWEEP.items():
    set_seed(42)
    cfg = vision_config_for(device, epochs=EPOCHS, **knobs)
    L = make_vision_loaders(cfg, S["beans_dir"], device)
    m = build_vision_model(cfg.model_name, 3, cfg.pretrained, freeze_backbone=True)
    opt = build_optimizer(cfg.optimizer, trainable_params(m), cfg.lr, cfg.weight_decay)
    t0 = time.time()
    h = fit(m, L["train"], L["validation"], epochs=cfg.epochs, optimizer=opt, device=device, amp=cfg.amp,
            label_smoothing=cfg.label_smoothing, verbose=False, name=name)
    rows.append({"run": name, "best val acc": round(h.best_val_acc, 3),
                 "final train acc": round(h.last.get("train_acc", float("nan")), 3),
                 "sec/epoch": round((time.time() - t0) / max(1, len(h.rows)), 1), "diverged": h.diverged})
    print(f"{name:<24} val {h.best_val_acc:.3f}")
results = pd.DataFrame(rows)
results

# %% [markdown]
# ### What each result teaches
# | Run | Typical outcome | Engineering lesson |
# |---|---|---|
# | **lr = 1.0** | accuracy jumps around or collapses; may diverge | Step size exceeds the stable range ($\eta > 2/\lambda_{\max}$). First suspect when the loss explodes. |
# | **lr = 1e-5** | barely better than chance after a few epochs | Too-small steps waste compute; use an LR finder or warm-up + decay. |
# | **batch 4** | slow epochs, noisy curve | Small batches underuse the GPU (many tiny kernel launches) and give noisy gradients — though the noise can act as a regulariser. |
# | **batch 128** | fast epochs, fewer updates per epoch → may need more epochs or a larger LR | Memory grows linearly with batch size; the *linear scaling rule* (scale LR with batch size) is a common starting point. |
# | **SGD, same lr** | slower than AdamW at 1e-3 | Adam adapts a per-parameter step ($\hat m/\sqrt{\hat v}$); plain SGD needs its own LR tuning (try 1e-2). |
# | **no augmentation** | higher train accuracy, similar or lower validation | Classic overfitting: the train/validation gap widens. |
# | **label smoothing** | similar accuracy, lower confidence | Better-calibrated probabilities — useful when you threshold on confidence. |

# %%
viz.plot_bar(dict(zip(results["run"], results["best val acc"])), "Best validation accuracy per run",
             ylabel="accuracy", horizontal=True)

# %% [markdown]
# ## Part B — take-home extensions (same recipe, new domains)
#
# ### B1. Health sciences: pneumonia from chest X-rays (MedMNIST)
# ```python
# # pip install medmnist          (dataset: MedMNIST v2, CC BY 4.0; 5,856 pediatric chest X-rays)
# from medmnist import PneumoniaMNIST
# train = PneumoniaMNIST(split="train", download=True, size=128)   # 2 classes: normal / pneumonia
# # X-rays are grayscale → repeat to 3 channels so ImageNet backbones accept them:
# #   transforms: T.Grayscale(num_output_channels=3), T.Resize(224), T.ToTensor(), T.Normalize(...)
# model = build_vision_model("resnet50", num_classes=2, pretrained=True, freeze_backbone=True)
# ```
# Things to think about: **class imbalance** (report recall/sensitivity, not just accuracy), the cost of a
# false negative vs a false positive, and that a research model is **not a medical device**.
#
# ### B2. Cybersecurity: SMS spam / phishing detection
# ```python
# # SMS Spam Collection (UCI, CC BY 4.0): 5,574 messages, ~13% spam → imbalanced
# from datasets import load_dataset
# ds = load_dataset("ucirvine/sms_spam")      # columns: sms, label (0 = ham, 1 = spam)
# model = build_text_model("distilbert/distilbert-base-uncased", num_labels=2,
#                          id2label={0: "ham", 1: "spam"}, label2id={"ham": 0, "spam": 1})
# ```
# Things to think about: choose the threshold for **high precision** (don't block real messages),
# adversarial drift (spammers adapt), and privacy of message data.
