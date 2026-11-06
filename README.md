# Deep Learning, Hands-On: from tensors to a shipped model

**Guest lecture kit · The Roux Institute at Northeastern University · November 19, 2026**
Dr. Sushanth Manakhari · hosted by Dr. Umesh Hodeghatta Rao

| Notebook | What it is | Open |
|---|---|---|
| `00_preflight_check` | 5-minute environment check — **run before the lecture** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sushanth-Manakhari/deep-learning-handson-lecture/blob/main/notebooks/00_preflight_check.ipynb) |
| `01_lecture_handson` | The lab: PyTorch → vision → NLP → debugging & MLOps → phone demo | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sushanth-Manakhari/deep-learning-handson-lecture/blob/main/notebooks/01_lecture_handson.ipynb) |
| `02_challenge_solutions` | Hyperparameter sweep with explanations + take-home projects | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sushanth-Manakhari/deep-learning-handson-lecture/blob/main/notebooks/02_challenge_solutions.ipynb) |

**Students:** start with [docs/STUDENT_PREREQUISITES.md](docs/STUDENT_PREREQUISITES.md).

---

## What you build

You are a new ML engineer with two tickets on your sprint board. Both are solved with **the same
transfer-learning recipe** — pretrained backbone → new head → fine-tune — then debugged and shipped.

| Ticket | Modality | Business problem | Data | Model |
|---|---|---|---|---|
| 🌱 **LEAF-101** | images | Farmers photograph a bean leaf → instant diagnosis (healthy / angular leaf spot / bean rust) | Beans, 1,295 field photos | ResNet-50 (ImageNet) |
| 💬 **SUPPORT-202** | text | Route customer messages to one of 77 intents; unsure → human agent | Banking77, 13,083 messages | DistilBERT |

Along the way: autograd checked against hand-derived gradients, the learning-rate stability limit,
data augmentation, freezing vs fine-tuning, batch size vs GPU memory (a *caught* out-of-memory crash),
the "overfit one batch" debugging rule with a planted bug, exploding and vanishing gradients,
confident mistakes on out-of-distribution photos and a **"route to a human" confidence threshold**,
experiment tracking, safe checkpoints, a model card, tests — and a **web app you open on your phone**
to upload a photo from your gallery.

## Quick start

### Google Colab (recommended — nothing to install)
1. Click an **Open in Colab** badge above.
2. **Runtime → Change runtime type → T4 GPU → Save**.
3. **Runtime → Run all** (accept "Run anyway"). The first cell clones this repository and installs the few
   missing packages; data and models download automatically.

### Kaggle Notebooks (backup)
Phone-verify your Kaggle account → **Create → New Notebook → File → Import Notebook → GitHub** →
`Sushanth-Manakhari/deep-learning-handson-lecture` → Accelerator **GPU T4**, **Internet On**.

### Your own laptop (optional)
Everything installs into a project-local virtual environment (`.venv/`); caches stay in `.cache/`.

```bash
# macOS / Linux  (Python 3.10+; macOS: Homebrew python, e.g. `brew install python`)
git clone https://github.com/Sushanth-Manakhari/deep-learning-handson-lecture.git
cd deep-learning-handson-lecture
bash scripts/setup_mac.sh                  # creates .venv, installs requirements
source .venv/bin/activate                  # later: deactivate
```
```powershell
# Windows 10/11 (PowerShell, Python 3.10+ from python.org with the py launcher)
git clone https://github.com/Sushanth-Manakhari/deep-learning-handson-lecture.git
cd deep-learning-handson-lecture
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
```
Then open the folder in VS Code (Python + Jupyter extensions), pick the `.venv` kernel and run
`notebooks/00_preflight_check.ipynb`.

## Hardware profiles (automatic)

| Detected device | Profile | Vision | Text |
|---|---|---|---|
| NVIDIA GPU (Colab/Kaggle T4, CUDA laptops) | `gpu` | ResNet-50 @ 224 px, mixed precision | full Banking77, batch 32 × 4 epochs, mixed precision |
| Apple Silicon GPU | `mps` | ResNet-50 @ 224 px | full Banking77 (eager attention) |
| CPU only | `cpu` (FAST) | ResNet-18 @ 160 px, 450 images, 3 epochs | 2,000 messages, 1 epoch |

## Lecture agenda (2 h + 30 min Q&A)

| Time | Section | Notebook |
|---|---|---|
| 4:00 | Setup + why deep learning (paradigm shift, GPUs, the stack) | §0–1 |
| 4:15 | The PyTorch mental model: tensors, gradient descent by hand, autograd, `nn.Module`, the loop | §2 |
| 4:35 | 🌱 LEAF-101: transfer learning, augmentation, **5-min challenge**, fine-tuning, OOM demo, test, stress test, threshold, ViT | §3 |
| 5:15 | 💬 SUPPORT-202: tokenisation, attention, frozen vs fine-tuned DistilBERT, confused intents, routing | §4 |
| 5:35 | Real-world debugging & MLOps: overfit one batch, NaN, vanishing gradients, tracking, safetensors, model card, tests, **phone demo** | §5 |
| 5:55 | Horizons: LLMs, LoRA, RAG, GANs, diffusion | §6 |
| 6:00 | Q&A | — |

## Results (verified run)

Measured with fixed seeds on Nov 5, 2026, on a **free Google Colab Tesla T4** (classroom conditions: Python 3.13,
PyTorch 2.11, Transformers 5.22) and on the instructor's **MacBook Pro M1 Pro** (Apple GPU via MPS: Python 3.15.2,
PyTorch 2.15, Transformers 5.22). The T4 run is saved with all outputs in `notebooks/01_lecture_handson_INSTRUCTOR.ipynb`.

| Ticket | Step | Colab T4 | MacBook M1 Pro |
|---|---|---|---|
| 🌱 LEAF-101 | Hand-crafted colour/edge features + logistic regression (validation) | 70.7% | 71.4% |
| 🌱 LEAF-101 | ResNet-50 frozen + new head, no augmentation (validation) | 88.7% | 88.7% |
| 🌱 LEAF-101 | Same + data augmentation, 5 epochs (validation) | 87.2% | 85.7% |
| 🌱 LEAF-101 | + fine-tune `layer4`, discriminative LR (validation / **test**) | 91.0% / **90.6%** | 93.2% / **91.4%** |
| 🌱 LEAF-101 | Confidence threshold τ: share decided automatically, accuracy on those | τ 0.92 · 68% · 98.9% | τ 0.89 · 75% |
| 🌱 LEAF-101 | Stress test (corrupted / non-leaf / synthetic images sent to a human) | 14 of 15 | 14 of 15 |
| 🌱 LEAF-101 | ViT-B/16, head only (instructor demo, validation) | 88.0% | needs CUDA |
| 💬 SUPPORT-202 | DistilBERT, frozen encoder (validation) | 55.5% | 58.7% |
| 💬 SUPPORT-202 | DistilBERT, full fine-tune (validation / **test** · macro-F1) | 91.2% / **92.1%** · 0.921 | 91.3% / **91.8%** · 0.917 |
| ⏱️ Compute | Matrix-multiply speed (float32 · float16) | 3.8 · 21.7 TFLOP/s | 4.0 TFLOP/s |
| ⏱️ Compute | Epoch time: ResNet-50 head · DistilBERT fine-tune | 6–11 s · 17 s | 7–13 s · 51 s |
| ⏱️ Compute | Out-of-memory demo (full fine-tune, 224 px) | 7.6 / 10.5 / 13.5 GB at batch 32 / 64 / 128; batch 256 caught | needs CUDA |

Debugging demos behave identically on both machines: overfit-one-batch ✅, planted optimizer bug detected, lr 0.1 → NaN
(fixed by gradient clipping), sigmoid first-layer gradient ≈ 10⁻¹⁸ vs ReLU/residual ≈ 10⁻¹, the exported 94 MB
safetensors checkpoint reloads with identical predictions, and all 19 tests pass inside Colab.

## Repository map

```
deep-learning-handson-lecture/
├── notebooks/            00_preflight_check · 01_lecture_handson · 02_challenge_solutions (.ipynb)
│   └── src/              the same notebooks as readable .py sources (edit these, then build)
├── src/dlhandson/        the tested package behind the notebooks
│   ├── device.py, config.py          hardware detection, seeds, per-hardware configs
│   ├── data_vision.py, data_text.py  datasets with download fallbacks, transforms, loaders
│   ├── models.py, train.py           transfer-learning models, the training loop
│   ├── debug_tools.py, ood.py        debugging demos, stress tests, selective prediction
│   ├── tracking.py, demo_app.py      TensorBoard/W&B logging, Gradio phone app
│   ├── catchup.py, classic.py        🛟 catch-up helpers, the "2015" hand-crafted baseline
│   └── viz.py, timing.py, sources.py plots, section timer, all data/model links
├── tests/                19 pytest tests (smoke mode: tiny models, no downloads)
├── scripts/              setup_mac.sh · setup_windows.ps1 · verify.py · run_notebooks.py · build_notebooks.py
│                         build_beans_pack.py · prepare_stress_images.py · train_vision.py · train_text.py
├── data/                 banking77/ (committed) · stress_test/ · beans_pack.json (Beans is downloaded)
├── docs/                 TECHNICAL_DESIGN.md · STUDENT_PREREQUISITES.md · diagrams/
└── tools/git-hooks/      pre-commit (blocks slides, large files and secrets)
```

## Data and models

| | Licence | Link | How it is obtained |
|---|---|---|---|
| **Beans** (Makerere University AI Lab) | MIT | https://huggingface.co/datasets/AI-Lab-Makerere/beans · original https://github.com/AI-Lab-Makerere/ibean/ | GitHub Release `data-v1/beans_224.zip` (26 MB, SHA-256 checked) → fallback Hugging Face |
| **Banking77** (PolyAI) | CC BY 4.0 | https://github.com/PolyAI-LDN/task-specific-datasets · paper https://arxiv.org/abs/2003.04807 | committed in `data/banking77/` |
| Coffee cup & cat photos (scikit-image) | CC0 | https://github.com/scikit-image/scikit-image/tree/v0.25.2/skimage/data | committed in `data/stress_test/non_leaf/` |
| **ResNet-50** ImageNet-1K V2 (25.6M) | BSD-3 (torchvision) | https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html · weights https://download.pytorch.org/models/resnet50-11ad3fa6.pth | downloaded on first use |
| ResNet-18 ImageNet-1K V1 (11.7M, CPU mode) | BSD-3 | https://download.pytorch.org/models/resnet18-f37072fd.pth | downloaded on first use |
| ViT-B/16 ImageNet-1K V1 (86.6M, demo) | BSD-3 | https://download.pytorch.org/models/vit_b_16-c867db91.pth | downloaded on first use |
| **DistilBERT** base uncased (66M) | Apache-2.0 | https://huggingface.co/distilbert/distilbert-base-uncased · paper https://arxiv.org/abs/1910.01108 | downloaded on first use |
| Take-home: MedMNIST v2, SMS Spam Collection | CC BY 4.0 | https://medmnist.com/ · https://archive.ics.uci.edu/dataset/228/sms+spam+collection | not used in class |

Every link also lives in `src/dlhandson/sources.py` and in the notebook appendix.

## For developers

```bash
source .venv/bin/activate
pytest -q                                              # 19 tests, ~15 s, no downloads
DLH_SMOKE=1 DLH_HEADLESS=1 python scripts/run_notebooks.py   # every notebook top to bottom, tiny models
python scripts/verify.py                               # real data + models -> reports/verify_report.md
python scripts/build_notebooks.py                      # after editing notebooks/src/*.py
python scripts/train_vision.py --epochs 5 --unfreeze 1 # the notebook as a CLI
python scripts/train_text.py --epochs 4
```
Environment variables: `DLH_DEVICE=cpu|mps|cuda` (force a device) · `DLH_SMOKE=1` (tiny models) ·
`DLH_HEADLESS=1` (skip interactive cells) · `DLH_WANDB=1` (log to Weights & Biases; key in Colab Secrets
or `WANDB_API_KEY`).

Students always use the **`main`** branch. Updates arrive through a reviewed pull request after a Colab dry run.

## Security and privacy
- Checkpoints use **safetensors** (no pickle code execution); remote code is never trusted.
- The dataset pack is **SHA-256 verified**; zip extraction blocks path traversal.
- No secrets in code: Colab Secrets / environment variables; a **pre-commit hook** blocks token-like strings,
  slide files and files over 20 MB.
- Instructor photos are stripped of **EXIF/GPS** metadata before commit.
- The phone demo's public link is temporary and runs on your own Colab machine — don't upload sensitive photos.

## Documentation
[Technical design](docs/TECHNICAL_DESIGN.md) · [Student prerequisites](docs/STUDENT_PREREQUISITES.md) ·
[Diagrams](docs/diagrams/)

## Licence and credits
Code: MIT (see `LICENSE`). Datasets and pretrained models keep their own licences (table above).
Built with PyTorch, torchvision, Hugging Face Transformers, Gradio, scikit-learn and matplotlib.
