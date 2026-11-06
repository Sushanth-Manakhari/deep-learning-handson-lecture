# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ✈️ Pre-flight check — run this BEFORE the lecture (≈ 5 minutes)
# **Deep Learning, Hands-On · The Roux Institute at Northeastern University · Nov 19, 2026**
#
# This notebook checks that your environment can run the hands-on lab, so that on the day you can start
# immediately.
#
# **On Google Colab (recommended):**
# 1. `Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save`
# 2. `Runtime → Run all`  (Colab may warn *"This notebook was not authored by Google"* → **Run anyway**)
# 3. Scroll to the bottom: every line should show ✅
#
# **On your own laptop (optional):** open this file in Jupyter / VS Code from the repository folder with
# the project's `.venv` selected, then *Run All*. Running it at home also **downloads and caches ~600 MB
# of models**, so you don't need to download them on campus Wi-Fi.
#
# Seeing ❌? The message next to it tells you what to do. See also `docs/STUDENT_PREREQUISITES.md`.

# %%
# ⚙️ SETUP — same cell as in the lecture notebook
import importlib.util, os, pathlib, shutil, subprocess, sys, time

REPO_URL = "https://github.com/Sushanth-Manakhari/deep-learning-handson-lecture.git"
os.environ.setdefault("HF_HUB_VERBOSITY", "error")  # public models need no token: hide the "unauthenticated" notice
REPO_BRANCH = "main"   # instructor dry run: set to "release"

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
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
if not (pathlib.Path("/content").is_dir() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ):
    # local laptop: keep downloaded models inside the project folder, not in ~/.cache
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
missing = [pkg for mod, pkg in [("gradio", "gradio"), ("qrcode", "qrcode"), ("safetensors", "safetensors")]
           if importlib.util.find_spec(mod) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

CHECKS = []   # (name, ok, detail)
def check(name):
    """Decorator: run a check, record ✅/❌ with a helpful message, never crash the notebook."""
    def wrap(fn):
        t0 = time.time()
        try:
            detail = fn() or ""
            CHECKS.append((name, True, f"{detail} ({time.time()-t0:.0f}s)"))
            print(f"✅ {name}: {detail}")
        except Exception as exc:
            CHECKS.append((name, False, f"{type(exc).__name__}: {exc}"))
            print(f"❌ {name}: {type(exc).__name__}: {exc}")
        return fn
    return wrap
print("✅ code ready at", ROOT)

# %%
@check("Python packages")
def _():
    import torch, torchvision, transformers, sklearn, matplotlib, gradio
    return (f"python {sys.version.split()[0]}, torch {torch.__version__}, torchvision {torchvision.__version__}, "
            f"transformers {transformers.__version__}, gradio {gradio.__version__}")

# %%
@check("GPU / accelerator")
def _():
    import torch
    from dlhandson.device import get_device, describe_device
    dev = get_device()
    info = describe_device(dev)
    a = torch.randn(1024, 1024, device=dev)
    (a @ a).sum().item()                      # one real computation on the device
    if dev.type == "cpu":
        raise RuntimeError("No GPU. Colab: Runtime → Change runtime type → T4 GPU → Save, then Run all. "
                           "(No GPU at all is OK — the lab runs in FAST mode, just slower.)")
    return f"{info['gpu_name']} on {info['environment']}"

# %%
@check("Leaf images (Beans dataset)")
def _():
    from dlhandson.data_vision import ensure_beans, image_folder, class_counts
    d = ensure_beans(verbose=False)
    return f"{sum(class_counts(image_folder(d, 'train')).values())} training photos ready"

# %%
@check("Customer messages (Banking77)")
def _():
    from dlhandson.data_text import load_banking77
    data = load_banking77()
    return f"{len(data['train'])} train messages, {len(data['label_names'])} intents"

# %%
@check("Pretrained vision model (ResNet-50, ~98 MB)")
def _():
    import torch
    from dlhandson.models import build_vision_model
    from dlhandson.device import get_device, smoke_mode
    m = build_vision_model("resnet18" if smoke_mode() else "resnet50", 3, pretrained=not smoke_mode()).to(get_device()).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 224, 224, device=get_device()))
    return f"forward pass OK, output shape {tuple(out.shape)}"

# %%
@check("Pretrained language model (DistilBERT, ~260 MB)")
def _():
    import torch
    from dlhandson.data_text import build_tokenizer
    from dlhandson.models import build_text_model
    from dlhandson.device import get_device
    tok = build_tokenizer("distilbert/distilbert-base-uncased", texts=["my card has not arrived"])
    m = build_text_model("distilbert/distilbert-base-uncased", 77, {i: str(i) for i in range(77)},
                         {str(i): i for i in range(77)}, vocab_size=len(tok)).to(get_device()).eval()
    enc = tok(["my card has not arrived"], return_tensors="pt").to(get_device())
    with torch.no_grad():
        logits = m(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
    return f"tokenised + forward pass OK, logits {tuple(logits.shape)}"

# %%
@check("One training step (autograd on the device)")
def _():
    import torch, torch.nn.functional as F
    from dlhandson.device import get_device
    dev = get_device()
    model = torch.nn.Sequential(torch.nn.Linear(10, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x, y = torch.randn(64, 10, device=dev), torch.randint(0, 2, (64,), device=dev)
    l0 = F.cross_entropy(model(x), y).item()
    for _ in range(20):
        opt.zero_grad(); loss = F.cross_entropy(model(x), y); loss.backward(); opt.step()
    assert loss.item() < l0, "loss did not decrease"
    return f"loss {l0:.3f} → {loss.item():.3f}"

# %%
@check("Web demo library (Gradio)")
def _():
    from dlhandson.demo_app import build_demo
    build_demo(lambda im: ({}, ""), lambda t: ({}, ""))
    return "app builds"

# %%
print("=" * 70)
for name, ok, detail in CHECKS:
    print(f"{'✅' if ok else '❌'} {name:<45} {detail}")
print("=" * 70)
if all(ok for _, ok, _ in CHECKS):
    print("🎉 You're ready for the lecture! (You can close this tab: Runtime → Disconnect and delete runtime.)")
else:
    print("Some checks failed — read the messages above, or see docs/STUDENT_PREREQUISITES.md → Troubleshooting.\n"
          "Only the GPU check failing? You can still follow along in FAST (CPU) mode or use Kaggle.")
