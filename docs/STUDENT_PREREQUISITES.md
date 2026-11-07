# Before the lecture: student preparation checklist

**Deep Learning, Hands-On: from tensors to a shipped model**
Guest lecture by Dr. Sushanth Manakhari · hosted by Dr. Umesh Hodeghatta Rao
The Roux Institute at Northeastern University, Portland, Maine · **Thursday, November 19, 2026, 4:00–5:30 PM** (80 minutes hands-on + 10 minutes of questions)

We will train two real models in class — a **leaf-disease detector** (images) and a **customer-message
router** (text) — break them on purpose, debug them, and open the result on your phone.
Doing the steps below takes **about 15 minutes** and means you can start coding the moment we begin.

| Part | Who | Time |
|---|---|---|
| A. Google Colab check | **Everyone (required)** | 10 min |
| B. Kaggle backup | Recommended | 5 min |
| C. Run it on your own laptop | Optional (Mac or Windows) | 20–30 min |
| D. Knowledge refresh | Optional | 20 min |
| E. On the day | Everyone | — |

Course repository: **https://github.com/Sushanth-Manakhari/deep-learning-handson-lecture**

---

## A. Required — check that Google Colab works for you (10 min)

Google Colab is a free Jupyter notebook in your browser with a free NVIDIA T4 GPU. Nothing to install,
and it works the same on Mac, Windows, Linux or a Chromebook.

1. **Google account.** Any Google account works. A *personal* Gmail account is the safest choice:
   some university Google accounts restrict Colab.
2. **Browser.** An up-to-date Chrome, Edge, Firefox or Safari (Chrome is the most tested).
3. **Open the pre-flight notebook:**
   **https://colab.research.google.com/github/Sushanth-Manakhari/deep-learning-handson-lecture/blob/main/notebooks/00_preflight_check.ipynb**
4. Sign in if asked. Colab shows *"Warning: This notebook was not authored by Google"* — that is normal
   for any notebook opened from GitHub. Click **Run anyway**.
5. Turn on the GPU: **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.**
6. Run everything: **Runtime → Run all** (or Ctrl+F9 / ⌘+F9). It takes 3–5 minutes.
7. Scroll to the last cell. Every line should show ✅ and the GPU line should say **Tesla T4**.
8. Free the GPU for others: **Runtime → Disconnect and delete runtime.**

> **No GPU available?** Colab's free GPUs are shared and occasionally run out at busy times. Try again
> later — and if it still fails, you can follow the whole lab in **FAST mode** (it switches to a smaller
> model automatically) or use the Kaggle backup below.

## B. Recommended — a Kaggle backup (5 min)

Kaggle Notebooks also give free GPUs (about 30 hours per week) and are a good plan B on the day.

1. Create a free account at **https://www.kaggle.com**.
2. **Verify your phone number** (Settings → Phone verification). Kaggle only allows GPU *and internet*
   for phone-verified accounts — and our notebook needs both.
3. To open the lab later: **Create → New Notebook → File → Import Notebook → GitHub**, paste
   `Sushanth-Manakhari/deep-learning-handson-lecture` and pick a notebook. Then, in the right-hand
   panel: **Session options → Accelerator → GPU T4** and **Internet → On**.
   *(Menu names on Kaggle change from time to time; look for "Import", "Accelerator" and "Internet".)*

## C. Optional — run it on your own laptop

Not needed for class (we use Colab), but useful if you want to keep working afterwards.
Everything installs into a **virtual environment inside the project folder** (`.venv`), so nothing
touches the rest of your computer. Needs about **3 GB** of free disk space.

### Mac (Apple Silicon or Intel)
```bash
# 1. Tools (skip what you already have). Homebrew: https://brew.sh
xcode-select --install
brew install python git
# 2. Get the code and set it up (creates .venv inside the folder)
git clone https://github.com/Sushanth-Manakhari/deep-learning-handson-lecture.git
cd deep-learning-handson-lecture
bash scripts/setup_mac.sh
```
Apple Silicon Macs (M1–M4) use their built-in GPU automatically ("MPS").

### Windows 10/11
1. Install **Python 3.12 or newer** from https://www.python.org/downloads/windows/ — tick **"Add python.exe to PATH"**.
2. Install **Git for Windows** from https://git-scm.com/download/win.
3. In **PowerShell**:
   ```powershell
   git clone https://github.com/Sushanth-Manakhari/deep-learning-handson-lecture.git
   cd deep-learning-handson-lecture
   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
   ```
4. Laptop with an **NVIDIA GPU**? The script prints how to install the GPU build of PyTorch (from the
   selector at https://pytorch.org/get-started/locally/). Without it, everything runs on the CPU in FAST mode.

### Linux
Same as the Mac commands (`sudo apt install python3-venv git` on Ubuntu instead of Homebrew).

### Then, on any OS
1. Install **VS Code** (https://code.visualstudio.com) with the **Python** and **Jupyter** extensions.
2. **File → Open Folder…** → the `deep-learning-handson-lecture` folder.
3. Open `notebooks/00_preflight_check.ipynb`, click **Select Kernel → Python Environments → .venv**, then **Run All**.
   Doing this at home also downloads the pretrained models (~400 MB) so you don't need campus Wi-Fi for them.

To use the environment from a terminal later: `source .venv/bin/activate` (Mac/Linux) or
`.\.venv\Scripts\Activate.ps1` (Windows); `deactivate` to leave it.

## D. Optional — 20-minute refresher

We assume Python and basic machine learning (train/validation/test split, overfitting, loss).
If you'd like a refresher:
- PyTorch "Learn the Basics": https://docs.pytorch.org/tutorials/beginner/basics/intro.html
- Neural networks, visually (3Blue1Brown): https://www.3blue1brown.com/topics/neural-networks
- Google Machine Learning Crash Course: https://developers.google.com/machine-learning/crash-course

## E. On the day

- **Laptop, fully charged, plus charger.** Training uses the cloud GPU, but the browser stays busy for the whole 90 minutes.
- **Campus Wi-Fi login** working before 4:00 PM.
- **Your phone**, and — fun bonus — **take a photo of any plant leaf before you come** (a house plant,
  a tree, a salad leaf). Near the end you'll upload it from your phone's gallery to the model you trained.
- **Arrive by 3:55 if you can.** The lecture notebook link is on the first slide (QR code). We run the three
  setup cells in the first 5 minutes, so data downloads while we talk.
- Lost or got an error? Every section starts with a 🛟 **catch-up cell** — run it and you're back in sync.
  Setup completely broken? Pair up with a neighbour; it's encouraged.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Cannot connect to GPU backend" / T4 greyed out | Free GPUs are temporarily used up. Wait and retry, use Kaggle, or continue in FAST (CPU) mode. |
| "This notebook was not authored by Google" | Expected for notebooks from GitHub → **Run anyway**. |
| Colab: "You are not allowed to use Colab" (university account) | Use a personal Gmail account. |
| Kaggle: no GPU option, or downloads fail | Verify your phone number, then turn **Internet → On** in the notebook settings. |
| Windows: "running scripts is disabled on this system" | Run once in PowerShell: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Mac: `CERTIFICATE_VERIFY_FAILED` | Use the setup script (it fixes certificates), or run *Install Certificates.command* in your Python folder. |
| `CUDA out of memory` | Lower `batch_size` in the cell you are running, or *Runtime → Restart session* and run the 🛟 cell. |
| Anything else | Bring it to class — or open an issue on the GitHub repository. |

**Privacy:** the demo app on the day creates a temporary public link. Only upload photos you are happy
to share (leaves, objects) — not photos of people or documents.
