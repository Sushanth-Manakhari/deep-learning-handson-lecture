"""Build every architecture diagram: docs/diagrams/architecture.drawio (9 pages) + one PNG per page.

    python tools/diagrams/build_diagrams.py            # build + run layout checks (fails on problems)

Open architecture.drawio at https://app.diagrams.net (File -> Open from -> Device) or in the
draw.io desktop app / VS Code "Draw.io Integration" extension. Each diagram is one page (tab).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drawio_engine import Diagram, write_drawio  # noqa: E402


# ---------------------------------------------------------------- 1. high-level design
def high_level() -> Diagram:
    d = Diagram("01_high_level_design", "1 · High-level design", 1280, 850)
    d.title("High-level design", "Who uses what, where the code runs, and what it downloads")
    d.box("students", 40, 130, 200, 76, "Students (45–60)", "laptop browser", "actor")
    d.box("instructor", 40, 380, 200, 76, "Instructor", "MacBook · Apple GPU", "actor")
    d.box("phone", 40, 520, 200, 76, "Phones in the room", "open the demo link or QR", "actor")
    d.group("gB", 290, 90, 280, 420, "Where the code runs")
    d.box("colab", 320, 130, 220, 76, "Google Colab (primary)", "free NVIDIA T4 GPU · no install", "env")
    d.box("kaggle", 320, 250, 220, 76, "Kaggle Notebooks (backup)", "T4 / P100 · phone-verified", "env")
    d.box("local", 320, 380, 220, 76, "Local laptop (optional)", "project .venv · CUDA / MPS / CPU", "env")
    d.text(300, 470, 260, 34, "All three run the same notebooks.", 11, color="#52514e")
    d.group("gC", 605, 90, 290, 600, "Course code (this repository)")
    d.box("nb", 635, 130, 240, 76, "Notebooks", "preflight · lecture · solutions", "code")
    d.box("pkg", 635, 250, 240, 76, "dlhandson package", "data · models · train · debug · OOD", "code")
    d.box("gradio", 635, 520, 240, 76, "Gradio demo app", "two tabs · public link + QR", "code")
    d.group("gD", 945, 90, 315, 420, "Downloaded at run time")
    d.box("gh", 970, 130, 270, 64, "GitHub repository", "code + small data (git clone)", "external")
    d.box("rel", 970, 214, 270, 64, "GitHub Release data-v1", "beans_224.zip · 26 MB · SHA-256", "external")
    d.box("hf", 970, 298, 270, 64, "Hugging Face Hub", "DistilBERT · Beans fallback", "external")
    d.box("pt", 970, 382, 270, 64, "PyTorch model server", "ResNet-50 / ResNet-18 / ViT weights", "external")
    d.box("runs", 970, 540, 270, 76, "runs/ (on that machine)", "TensorBoard logs · safetensors checkpoints · model card", "output")
    d.edge("students", "colab", "open\nnotebook")
    d.edge("instructor", "local", "verify\nrun")
    d.edge("colab", "nb", "runs")
    d.edge("nb", "gh", "git clone", exit=("R", 32 / 76), entry=("L", 0.5))
    d.edge("nb", "pkg", "import", exit=("B", 0.5), entry=("T", 0.5))
    d.edge("pkg", "rel", "", exit=("R", 14 / 76), entry=("L", 0.5), via=[(931, 264), (931, 246)])
    d.edge("pkg", "hf", "", exit=("R", 38 / 76), entry=("L", 0.5), via=[(917, 288), (917, 330)])
    d.edge("pkg", "pt", "", exit=("R", 62 / 76), entry=("L", 0.5), via=[(903, 312), (903, 414)])
    d.edge("pkg", "gradio", "trained models", exit=("B", 0.3), entry=("T", 0.3))
    d.edge("pkg", "runs", "logs · checkpoints", exit=("B", 0.7), entry=("T", 130 / 270), via=[(803, 470), (1100, 470)])
    d.edge("gradio", "phone", "public HTTPS link + QR code", exit=("L", 0.5), entry=("R", 0.5))
    d.legend(290, 600, [("actor", "People"), ("env", "Runtime environment"), ("code", "Course code"),
                        ("external", "Downloaded from the internet"), ("output", "Output on the machine"),
                        ("edge", "Data / control flow")], w=280)
    return d


# ---------------------------------------------------------------- 2. data flow
def data_flow() -> Diagram:
    d = Diagram("02_data_flow", "2 · Data flow", 1180, 1310)
    d.title("Data flow", "From public sources to batches, through training, to a decision")
    VX, MX, TX, W, H = 90, 450, 810, 300, 64
    d.box("vh", VX, 88, W, 34, "IMAGES · Beans (LEAF-101)", "", "header", align="left", size=14)
    d.box("th", TX, 88, W, 34, "TEXT · Banking77 (SUPPORT-202)", "", "header", align="left", size=14)
    rows = [130, 230, 330, 430, 530, 630]
    vis = [("v1", "Hugging Face Hub (original)", "AI-Lab-Makerere/beans · parquet · ~180 MB · MIT", "store"),
           ("v2", "build_beans_pack.py (once)", "decode · resize 224 px · JPEG q90 · zip", "code"),
           ("v3", "GitHub Release data-v1", "beans_224.zip · 26 MB · SHA-256 in beans_pack.json", "store"),
           ("v4", "ensure_beans()", "download · verify SHA-256 · block zip-slip · unzip", "code"),
           ("v5", "Transforms", "train: crop, flip, rotate, colour jitter · all: x' = (x − μ) / σ", "code"),
           ("v6", "DataLoader", "batch 32 · pixel_values [B,3,224,224] + labels", "code")]
    txt = [("t1", "PolyAI GitHub repository", "banking_data/train.csv, test.csv · CC BY 4.0", "external"),
           ("t2", "data/banking77/ (committed)", "13,083 messages · 77 intents · ~1 MB", "store"),
           ("t3", "load_banking77()", "stratified split 9,002 / 1,001 / 3,080", "code"),
           ("t4", "DistilBERT tokenizer", "WordPiece · [CLS] … [SEP] · max 64 tokens", "code"),
           ("t5", "TextClassificationDataset", "pre-tokenised once (fast tokenizer)", "code"),
           ("t6", "DataLoader + dynamic padding", "batch 32 · input_ids, attention_mask + labels", "code")]
    for (i, t, s, k), y in zip(vis, rows):
        d.box(i, VX, y, W, H, t, s, k)
    for (i, t, s, k), y in zip(txt, rows):
        d.box(i, TX, y, W, H, t, s, k)
    for a, b, lab in [("v1", "v2", "export"), ("v2", "v3", "upload once"), ("v3", "v4", "download 26 MB"),
                      ("v4", "v5", "ImageFolder 1,034 / 133 / 128"), ("v5", "v6", "tensors"),
                      ("t1", "t2", "copied once"), ("t2", "t3", "read CSV"), ("t3", "t4", "text"),
                      ("t4", "t5", "token ids"), ("t5", "t6", "")]:
        d.edge(a, b, lab, exit=("B", 0.5), entry=("T", 0.5))
    d.edge("v4", "v1", "fallback", exit=("L", 0.5), entry=("L", 0.5), via=[(45, 462), (45, 162)], dashed=True)
    d.box("n1", MX, 130, 300, 190, "One batch format for both tickets",
          "Images → pixel_values [B, 3, 224, 224]\nText → input_ids + attention_mask [B, T]\nBoth → labels [B]\n"
          "So ONE training loop serves both.", "note", align="left")
    d.box("n2", MX, 360, 300, 150, "Data rules",
          "Tune on validation, report on test once.\nDownloads are verified before use.\nNothing is uploaded anywhere.",
          "note", align="left")
    # shared training pipeline
    d.box("m1", MX, 780, W, H, "Model forward", "ResNet-50 or DistilBERT → logits z", "code")
    d.box("m2", MX, 880, W, H, "Loss + backward", "cross-entropy L = −log p_y · autograd", "code")
    d.box("m3", MX, 980, W, H, "Optimizer step", "AdamW · warm-up / cosine LR · clip ‖g‖", "code")
    d.box("m4", MX, 1080, W, H, "Evaluate (each epoch)", "accuracy · F1 · confusion · best epoch", "code")
    d.box("m5", MX + 30, 1180, 240, 100, "Confidence ≥ τ ?", "", "decision", size=13)
    d.box("o1", TX, 1080, W, H, "runs/checkpoints (safetensors)", "best epoch · reused by catch-up cells", "store")
    d.box("o2", VX, 1080, W, H, "TensorBoard + metrics.csv", "every epoch of every run", "output")
    d.box("o3", TX, 1198, W, H, "Automatic decision", "e.g. bean_rust · card_arrival", "output")
    d.box("o4", VX, 1198, W, H, "Route to a human", "agronomist / support agent", "risk")
    d.box("n3", VX, 800, W, 110, "Fallback chain (images)",
          "local disk → GitHub Release → Hugging Face Hub → clear error message", "note", align="left")
    d.edge("v6", "m1", "image batches", exit=("B", 0.5), entry=("T", 0.25), via=[(240, 740), (525, 740)])
    d.edge("t6", "m1", "text batches", exit=("B", 0.5), entry=("T", 0.75), via=[(960, 750), (675, 750)])
    d.edge("m1", "m2", "logits", exit=("B", 0.5), entry=("T", 0.5))
    d.edge("m2", "m3", "gradients", exit=("B", 0.5), entry=("T", 0.5))
    d.edge("m3", "m1", "next batch", exit=("R", 0.5), entry=("R", 0.5), via=[(790, 1012), (790, 812)], dashed=True)
    d.edge("m3", "m4", "end of epoch", exit=("B", 0.5), entry=("T", 0.5))
    d.edge("m4", "o1", "save\nbest", exit=("R", 0.5), entry=("L", 0.5))
    d.edge("m4", "o2", "log", exit=("L", 0.5), entry=("R", 0.5))
    d.edge("m4", "m5", "softmax p", exit=("B", 0.5), entry=("T", 0.5))
    d.edge("m5", "o3", "yes", exit=("R", 0.5), entry=("L", 0.5))
    d.edge("m5", "o4", "no", exit=("L", 0.5), entry=("R", 0.5))
    d.legend(840, 790, [("store", "Data store / file"), ("external", "External source"), ("code", "Processing step"),
                        ("output", "Result"), ("decision", "Decision"), ("edge-dashed", "Fallback / loop")], w=270)
    return d


# ---------------------------------------------------------------- 3. sequence
def sequence() -> Diagram:
    d = Diagram("03_sequence", "3 · Sequence", 1400, 1100)
    d.title("Sequence: a student's session", "From opening the link to using the trained model on a phone")
    parts = [("S", "Student", "laptop + phone", "actor"), ("N", "Colab notebook", "T4 runtime", "env"),
             ("G", "GitHub", "repository + Release", "external"), ("H", "HF Hub / PyTorch", "pretrained weights",
                                                                       "external"),
             ("D", "dlhandson + GPU", "fit() on CUDA", "code"), ("R", "Gradio app", "public share link", "output")]
    cx = {}
    for i, (k, t, s, kind) in enumerate(parts):
        c = 130 + i * 230
        cx[k] = c
        d.box(k, c - 95, 88, 190, 58, t, s, kind)
        d.line((c, 146), (c, 960), arrow=False, dashed=True, color="#b8b7b1", width=1.2)
    msgs = [("S", "N", "1. open link · T4 · run setup", 0), ("N", "G", "2. git clone (code + small data)", 0),
            ("G", "N", "3. repository (~2 MB)", 1), ("N", "G", "4. download beans_224.zip", 0),
            ("G", "N", "5. zip (26 MB) → verify SHA-256", 1), ("N", "H", "6. request pretrained weights", 0),
            ("H", "N", "7. ResNet-50 · DistilBERT", 1), ("S", "N", "8. run training cell (knobs)", 0),
            ("N", "D", "9. fit(model, loaders, config)", 0), None,
            ("D", "N", "11. history + best checkpoint", 1), ("N", "S", "12. curves · metrics · confusion", 1),
            ("S", "N", "13. run the demo cell", 0), ("N", "R", "14. build_demo() · launch(share=True)", 0),
            ("R", "S", "15. public link + QR code", 1), ("S", "R", "16. upload leaf photo / type a message", 0),
            ("R", "S", "17. label + confidence, or 'ask a human'", 1)]
    y = 195
    for m in msgs:
        if m is None:
            d.box("note10", cx["D"] + 14, y - 22, 200, 44, "10. forward → loss → backward → step",
                  "", "note", size=11)
        else:
            a, b, lab, ret = m
            off = 1 if cx[b] > cx[a] else -1
            d.line((cx[a] + 2 * off, y), (cx[b] - 2 * off, y), lab, dashed=bool(ret), label_above=True)
        y += 45
    d.legend(30, 975, [("edge", "request / call"), ("edge-dashed", "response")], w=220, title="Legend")
    return d


# ---------------------------------------------------------------- 4. error handling
def error_handling() -> Diagram:
    rows = [("No GPU available", "Colab GPUs used up / laptop without GPU", "device is CPU?",
             "FAST profile", "ResNet-18 @ 160 px · data subset · fewer epochs",
             "Banner: how to switch to T4", "the lab still finishes, just slower"),
            ("Dataset Release unreachable or altered", "network error / tampered file",
             "download fails or SHA-256 differs?", "Refuse file · fall back to Hugging Face",
             "zip-slip guard on every extraction", "Same data, slower download", "no action needed"),
            ("Hugging Face also unreachable", "no internet / Kaggle Internet off", "both sources failed?",
             "Stop with a clear message", "\"check the connection (Kaggle: Internet On)\"",
             "Actionable message, not a stack trace", "fix and re-run the cell"),
            ("Loss explodes", "learning rate too high · bad inputs", "loss is inf or NaN?",
             "NaN guard stops training", "fp16: GradScaler skips the bad step",
             "Explanation and fixes", "lower LR · clip gradients · warm-up"),
            ("GPU out of memory", "batch or images too large", "out-of-memory error?",
             "Catch it, free memory, continue", "report the largest batch that fits",
             "Batch size vs memory table", "notebook stays alive"),
            ("Optimizer misses the new head", "head replaced after the optimizer was built",
             "overfit-one-batch fails?", "optimizer_covers_model()", "lists the missing tensors",
             "FAIL → fix → PASS", "the bug is found in seconds"),
            ("Lost notebook state", "restart · skipped cells · typo", "variables missing?",
             "Catch-up cell rebuilds state", "loads cached checkpoints instead of retraining",
             "Back in sync", "continue with the class"),
            ("Public demo link fails", "share service blocked", "launch() raises?",
             "Keep the inline app", "in-notebook photo upload as fallback",
             "Demo still works", "on the laptop instead of the phone")]
    top, step, h = 140, 116, 96
    d = Diagram("04_error_handling", "4 · Error handling", 1400, top + step * len(rows) + 50)
    d.title("Error handling and fallbacks", "Every failure we expect in a classroom, how it is detected, and what happens")
    for x, w, t in [(40, 250, "What can go wrong"), (350, 230, "How it is detected"),
                    (640, 330, "Automatic handling"), (1030, 330, "What the student sees")]:
        d.box(f"h{x}", x, 90, w, 34, t, "", "header", size=14, align="left" if x == 40 else "center")
    for i, (r, rs, q, c, cs, o, os_) in enumerate(rows):
        y = top + i * step
        d.box(f"r{i}", 40, y + 10, 250, h - 20, r, rs, "risk")
        d.box(f"q{i}", 350, y, 230, h, q, "", "decision", size=12)
        d.box(f"c{i}", 640, y + 10, 330, h - 20, c, cs, "control")
        d.box(f"o{i}", 1030, y + 10, 330, h - 20, o, os_, "output")
        d.edge(f"r{i}", f"q{i}")
        d.edge(f"q{i}", f"c{i}", "yes")
        d.edge(f"c{i}", f"o{i}")
    return d


# ---------------------------------------------------------------- 5. model architecture
def model_arch() -> Diagram:
    d = Diagram("05_model_architecture", "5 · Model architecture", 1300, 930)
    d.title("Model architecture: transfer learning on both tickets",
            "Keep the pretrained backbone, add a new head, then fine-tune the top with a small learning rate")
    d.box("hv", 60, 88, 380, 34, "Vision · ResNet-50 (25.6M parameters)", "", "header", size=14, align="left")
    d.box("ht", 860, 88, 380, 34, "Text · DistilBERT (66M parameters)", "", "header", size=14, align="left")
    V = [("a1", 130, 56, "Leaf photo", "224 × 224 × 3 RGB, normalised (x − μ) / σ", "note"),
         ("a2", 216, 64, "conv1 + layer1 – layer3", "frozen in both stages · edges → textures → parts", "frozen"),
         ("a3", 310, 64, "layer4 (3 residual blocks)", "frozen in stage A · fine-tuned in stage B (LR 1e-4)",
          "trainable"),
         ("a4", 404, 56, "Global average pooling", "7 × 7 × 2048 → 2048 features", "frozen"),
         ("a5", 490, 56, "New head: Linear 2048 → 3", "6,147 trainable parameters (LR 1e-3)", "newhead"),
         ("a6", 576, 56, "softmax → 3 probabilities", "angular_leaf_spot · bean_rust · healthy", "note")]
    T = [("b1", 130, 56, "Customer message", "\"My card got swallowed by the ATM!!\"", "note"),
         ("b2", 216, 64, "WordPiece tokenizer", "[CLS] my card got swallowed … [SEP] → ids", "note"),
         ("b3", 310, 64, "Embeddings (768-d)", "token + position · frozen in A, trained in B", "trainable"),
         ("b4", 404, 70, "6 × Transformer layer", "self-attention + feed-forward · trained in B (LR 5e-5)",
          "trainable"),
         ("b5", 504, 56, "New head: 768 → 768 → 77", "pre-classifier + ReLU + dropout + classifier", "newhead"),
         ("b6", 590, 56, "softmax → 77 intent probabilities", "e.g. card_swallowed 0.89", "note")]
    for x, items in ((60, V), (860, T)):
        for i, (k, y, h, t, s, kind) in enumerate(items):
            d.box(k, x, y, 380, h, t, s, kind)
            if i:
                d.edge(items[i - 1][0], k, "", exit=("B", 0.5), entry=("T", 0.5))
    d.box("a7", 130, 662, 240, 96, "max p ≥ τ = 0.89 ?", "", "decision", size=12)
    d.box("a8", 60, 800, 170, 56, "Auto-diagnosis", "", "output")
    d.box("a9", 270, 800, 170, 56, "Agronomist review", "", "risk")
    d.box("b7", 930, 676, 240, 96, "max p ≥ τ = 0.79 ?", "", "decision", size=12)
    d.box("b8", 860, 814, 170, 56, "Auto-route intent", "", "output")
    d.box("b9", 1070, 814, 170, 56, "Human support agent", "", "risk")
    d.edge("a6", "a7", "", exit=("B", 0.5), entry=("T", 0.5))
    d.edge("b6", "b7", "", exit=("B", 0.5), entry=("T", 0.5))
    d.edge("a7", "a8", "yes", exit=("L", 0.5), entry=("T", 40 / 170), via=[(100, 710)])
    d.edge("a7", "a9", "no", exit=("R", 0.5), entry=("T", 130 / 170), via=[(400, 710)])
    d.edge("b7", "b8", "yes", exit=("L", 0.5), entry=("T", 40 / 170), via=[(900, 724)])
    d.edge("b7", "b9", "no", exit=("R", 0.5), entry=("T", 130 / 170), via=[(1200, 724)])
    d.box("n1", 480, 130, 340, 130, "Two training stages",
          "Stage A: train only the new head (green).\nStage B: also fine-tune the orange blocks with a "
          "smaller learning rate.", "note", align="left")
    d.box("n2", 480, 290, 340, 110, "Residual connection (ResNet)",
          "y = F(x) + x  →  ∂y/∂x = ∂F/∂x + I\nan identity path keeps gradients alive through 50 layers",
          "note", align="left")
    d.box("n3", 480, 430, 340, 110, "Self-attention (Transformer)",
          "softmax(Q·K^T / √d) · V\nevery token weighs every other token", "note", align="left")
    d.legend(480, 580, [("frozen", "Frozen pretrained weights"), ("trainable", "Fine-tuned in stage B"),
                        ("newhead", "New head (always trained)"), ("note", "Data / tokens / outputs"),
                        ("decision", "Confidence threshold")], w=340)
    return d


# ---------------------------------------------------------------- 6. security
def security() -> Diagram:
    rows = [("Malicious model files", "pickle can run code when loaded", "safetensors for every checkpoint",
             "raw tensors only · trust_remote_code never used", "train.py", "save_checkpoint / load_checkpoint"),
            ("Tampered or corrupted dataset", "download altered in transit", "SHA-256 check before use",
             "hash recorded in data/beans_pack.json", "data_vision.py", "_from_release()"),
            ("Path traversal in archives", "'zip-slip' writes outside the folder", "Reject absolute or '..' paths",
             "checked for every zip member", "data_vision.py", "zip extraction"),
            ("Leaked API tokens", "keys pasted into notebooks", "Colab Secrets / environment variables",
             "pre-commit hook blocks hf_, AWS and private keys", "tracking.py · tools/git-hooks", "get_secret()"),
            ("Private slides published", "deck and speaker notes", ".gitignore + hook block *.pptx and slides/",
             "verified with git check-ignore before push", ".gitignore · pre-commit", "git check-ignore"),
            ("Location data in photos", "phone EXIF metadata holds GPS", "Strip all metadata, resize ≤ 800 px",
             "raw photos stay in git-ignored _incoming/", "scripts/", "prepare_stress_images.py"),
            ("Public demo link misuse", "anyone with the URL can use it", "Temporary link on the student's own VM",
             "nothing stored · demo.close() · no sensitive photos", "demo_app.py", "launch(share=True)"),
            ("Large files in git history", "datasets / checkpoints committed", "Hook blocks files > 20 MB",
             "datasets ship as GitHub Releases", "tools/git-hooks/pre-commit", "size check"),
            ("Over-trusting predictions", "confident mistakes on odd inputs", "Threshold τ → human review",
             "model card: intended use, limits, licences", "ood.py", "runs/export/MODEL_CARD.md")]
    top, step = 130, 82
    d = Diagram("06_security", "6 · Security & responsible AI", 1300, top + step * len(rows) + 30)
    d.title("Security, privacy and responsible AI", "Each risk, the control that addresses it, and where it lives in the code")
    for x, w, t in [(40, 330, "Risk"), (440, 400, "Control"), (910, 350, "Implemented in")]:
        d.box(f"h{x}", x, 88, w, 34, t, "", "header", size=14, align="left")
    for i, (r, rs, c, cs, wh, ws) in enumerate(rows):
        y = top + i * step
        d.box(f"r{i}", 40, y, 330, 64, r, rs, "risk")
        d.box(f"c{i}", 440, y, 400, 64, c, cs, "control")
        d.box(f"w{i}", 910, y, 350, 64, wh, ws, "where")
        d.edge(f"r{i}", f"c{i}")
        d.edge(f"c{i}", f"w{i}")
    return d


# ---------------------------------------------------------------- 7. environment decision
def environment() -> Diagram:
    d = Diagram("07_environment_decision", "7 · Environment decision", 1200, 830)
    d.title("Where should I run the lab?", "Every path runs the same notebooks; the code adapts to the hardware it finds")
    d.box("start", 430, 100, 300, 56, "Start: open the lecture notebook", "", "terminal")
    d.box("d1", 455, 196, 250, 110, "Colab gives you a T4 GPU?", "", "decision", size=12)
    d.box("d2", 455, 360, 250, 110, "Kaggle account phone-verified?", "", "decision", size=12)
    d.box("d3", 455, 524, 250, 110, "Laptop with NVIDIA GPU or Apple Silicon?", "", "decision", size=12)
    d.box("o1", 820, 215, 330, 72, "Google Colab (primary)", "GPU profile: ResNet-50, full data", "output")
    d.box("o2", 820, 379, 330, 72, "Kaggle Notebooks (backup)", "Accelerator GPU T4 · Internet On", "output")
    d.box("o3", 820, 543, 330, 72, "Local .venv (CUDA or MPS)", "setup_mac.sh · setup_windows.ps1", "output")
    d.box("o4", 430, 690, 300, 72, "FAST mode on CPU, or pair up", "ResNet-18 @ 160 px · data subset", "output")
    d.edge("start", "d1", "", exit=("B", 0.5), entry=("T", 0.5))
    d.edge("d1", "o1", "yes")
    d.edge("d1", "d2", "no", exit=("B", 0.5), entry=("T", 0.5))
    d.edge("d2", "o2", "yes")
    d.edge("d2", "d3", "no", exit=("B", 0.5), entry=("T", 0.5))
    d.edge("d3", "o3", "yes")
    d.edge("d3", "o4", "no", exit=("B", 0.5), entry=("T", 0.5))
    d.box("n1", 60, 196, 320, 150, "Before class",
          "Run 00_preflight_check once. It checks packages, GPU, data and models and prints OK or the fix.",
          "note", align="left")
    d.box("n2", 60, 380, 320, 130, "In class",
          "Every section starts with a catch-up cell, so you can switch environment and continue.",
          "note", align="left")
    d.legend(60, 600, [("terminal", "Start"), ("decision", "Question"), ("output", "Where you run it"),
                       ("note", "Tip")], w=320)
    return d


# ---------------------------------------------------------------- 8. lecture flow
def lecture_flow() -> Diagram:
    segs = [("4:00", "Setup", "§0 · 5 min", "Open the notebook from the QR code · turn on the T4 GPU · run the 3 setup cells (they download while we talk)"),
            ("4:05", "Why deep learning", "§1 · 5 min", "Computers learn what to look for: hand-made features 70.7% vs a pretrained model 88.7% · why a GPU (28× faster)"),
            ("4:10", "How a model learns", "§2 · 12 min", "Small steps downhill, by hand · PyTorch does the calculus · how big each step should be · the five-line training loop"),
            ("4:22", "Project 1: leaf disease", "§3 · 28 min", "Look at the data · the score to beat · reuse ResNet · more photos for free · 3-min challenge · fine-tuning · test set 90.6% · unsure → ask a human"),
            ("4:50", "Project 2: messages", "§4 · 13 min", "Word pieces · attention · last layer or whole model · test set 92.1% · unsure messages go to a person"),
            ("5:03", "When training goes wrong", "§5 · 12 min", "Learn 16 examples by heart · numbers blow up (NaN) or fade away · ship it: a web app on your phone · share a model safely"),
            ("5:15", "Wrap-up", "§6 · 5 min", "What you built today · where this leads: LLMs, LoRA, RAG · take-home projects"),
            ("5:20", "Q&A", "10 min", "Careers, research, anything from today · hard stop at 5:30")]
    d = Diagram("08_lecture_flow", "8 · Lecture flow", 1500, 620)
    d.title("Lecture flow · Thursday, November 19, 2026 · 4:00–5:30 PM",
            "80 minutes hands-on + 10 minutes Q&A; each block maps to a notebook section")
    x0, w, gap = 40, 160, 20
    for i, (t, title, sub, concepts) in enumerate(segs):
        x = x0 + i * (w + gap)
        d.box(f"t{i}", x + 30, 96, w - 60, 34, t, "", "time", font_color="#ffffff", size=14)
        d.box(f"s{i}", x, 166, w, 70, title, sub, "code")
        d.box(f"c{i}", x, 272, w, 250, "", concepts, "note", align="left", sub_size=12)
        d.edge(f"t{i}", f"s{i}", "", exit=("B", 0.5), entry=("T", 0.5), arrow=False)
        d.edge(f"s{i}", f"c{i}", "", exit=("B", 0.5), entry=("T", 0.5), arrow=False, dashed=True)
        if i:
            d.edge(f"s{i - 1}", f"s{i}", "")
    d.text(40, 548, 1420, 40, "Both projects use the same recipe: pretrained model → new last layer → gentle fine-tuning. "
           "Cells 3.9, 3.14, 5.4 and 5.6 are skipped in class (marked in the notebook); try them at home.", 13, color="#52514e")
    return d


# ---------------------------------------------------------------- 9. git + release
def git_flow() -> Diagram:
    d = Diagram("09_git_release_flow", "9 · Git & release flow", 1400, 940)
    d.title("Git branching and data release",
            "Private work on non-prod · public main changes only by pull request · students use main")
    d.box("p1", 40, 130, 220, 80, "Private repository", "branch non-prod · all work", "external")
    d.box("g1", 40, 290, 220, 80, "Local repository", "~/Projects/deep_learning · non-prod", "code")
    d.box("g2", 330, 290, 200, 80, "Release script", "student-safe snapshot", "code")
    d.box("g3", 600, 290, 200, 80, "Colab dry run", "REPO_BRANCH = \"release\"", "env")
    d.box("g4", 870, 290, 200, 80, "Pull request", "release → main", "code")
    d.box("g5", 1140, 290, 220, 80, "GitHub: main (protected)", "PR required · no force-push", "external")
    d.box("s1", 1140, 470, 220, 80, "Students", "Colab badges and links point to main", "actor")
    d.edge("g1", "p1", "push", exit=("T", 0.5), entry=("B", 0.5))
    d.edge("g1", "g2", "publish")
    d.edge("g2", "g3", "public\nrelease")
    d.edge("g3", "g4", "checks\npass")
    d.edge("g4", "g5", "merge")
    d.edge("g5", "s1", "links", exit=("B", 0.5), entry=("T", 0.5))
    d.box("c1", 40, 470, 220, 96, "pre-commit hook", "blocks slides (*.pptx), files > 20 MB and token-like secrets", "control")
    d.box("c2", 300, 470, 220, 96, ".gitignore", "slides/, .venv/, .cache/, reports/, runs/, dist/", "control")
    d.edge("c1", "g1", "every commit", exit=("T", 66 / 220), entry=("B", 66 / 220), dashed=True)
    d.edge("c2", "g1", "", exit=("T", 0.5), entry=("B", 154 / 220), via=[(410, 430), (194, 430)], dashed=True)
    d.box("r1", 40, 700, 220, 90, "build_beans_pack.py (once)", "Hugging Face → 224 px JPEG → zip + SHA-256", "code")
    d.box("r2", 330, 700, 200, 90, "dist/beans_224.zip", "26 MB · git-ignored", "store")
    d.box("r3", 600, 700, 200, 90, "Release data-v1", "uploaded once (web or gh CLI)", "external")
    d.box("r4", 870, 700, 200, 90, "Notebooks download it", "verify SHA-256 → unzip", "code")
    d.box("c3", 40, 840, 220, 70, "data/beans_pack.json", "URL + SHA-256 (committed)", "store")
    d.edge("r1", "r2", "zip")
    d.edge("r2", "r3", "upload")
    d.edge("r3", "r4", "download")
    d.edge("r4", "s1", "fast, verified data", exit=("R", 0.5), entry=("B", 110 / 220), via=[(1250, 745)])
    d.edge("r1", "c3", "writes", exit=("B", 0.5), entry=("T", 0.5))
    d.legend(870, 430, [("code", "Code / repository"), ("external", "GitHub"), ("env", "Colab"),
                        ("control", "Safety guard"), ("store", "File / artifact"), ("actor", "People")], w=230)
    return d


DIAGRAMS = [high_level, data_flow, sequence, error_handling, model_arch, security, environment, lecture_flow,
            git_flow]

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "docs" / "diagrams"
    out.mkdir(parents=True, exist_ok=True)
    diagrams = [f() for f in DIAGRAMS]
    problems = [p for dgm in diagrams for p in dgm.check()]
    for dgm in diagrams:
        print(f"{dgm.slug:<26} boxes {len([b for b in dgm.boxes if not b.group]):>3} · edges {len(dgm.edges):>3} · "
              f"edge crossings {dgm.crossings()}")
    if problems:
        print(f"\n{len(problems)} layout problem(s):")
        print("\n".join("  " + p for p in problems))
        sys.exit(1)
    write_drawio(diagrams, out / "architecture.drawio")
    for dgm in diagrams:
        dgm.to_png(str(out / f"{dgm.slug}.png"))
    print(f"\n✅ all layout checks passed → {out / 'architecture.drawio'} + {len(diagrams)} PNGs")
