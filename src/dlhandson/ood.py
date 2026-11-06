"""Stress tests, confidence, and "route to a human" (selective prediction).

THE PROBLEM
-----------
A softmax classifier MUST spread 100% probability across the classes it knows.
Show a 3-class leaf model a coffee cup and it still answers "bean_rust, 97%".
Real inputs drift from training data (new phones, lighting, crops, users
uploading the wrong photo) — out-of-distribution (OOD) inputs are the norm in
production, not the exception.

SIMPLE, EFFECTIVE DEFENCES
--------------------------
* Confidence = max softmax probability (MSP), a strong OOD baseline
  (Hendrycks & Gimpel, 2017).
* Uncertainty = entropy  H(p) = - sum_k p_k log p_k   (0 = certain, log K = clueless)
* SELECTIVE PREDICTION: act automatically only if confidence >= tau, otherwise
  send to a human. Raising tau trades COVERAGE (share handled automatically)
  for ACCURACY on what is handled. Insurers, banks and hospitals tune tau to a
  business target, e.g. "98% accuracy on auto-decisions".
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic"}


# ---------------------------------------------------------------------------
# Building stress-test images
# ---------------------------------------------------------------------------
def _jpeg(im: Image.Image, quality: int = 5) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


CORRUPTIONS = {
    "blurry (shaky hand)": lambda im: im.filter(ImageFilter.GaussianBlur(radius=im.width / 40)),
    "very dark (dusk)": lambda im: ImageEnhance.Brightness(im).enhance(0.25),
    "over-exposed (sun glare)": lambda im: ImageEnhance.Brightness(im).enhance(2.4),
    "heavy compression": lambda im: _jpeg(im, 4),
    "zoomed into a corner": lambda im: im.crop((0, 0, im.width // 3, im.height // 3)).resize(im.size),
}


def corrupt_images(images: list[Image.Image], labels: list[str]) -> list[dict]:
    """Apply each corruption to the given in-distribution images (labels known)."""
    out = []
    for im, lab in zip(images, labels):
        for cname, fn in CORRUPTIONS.items():
            out.append({"image": fn(im.convert("RGB")), "name": cname, "group": "bean leaf, harder conditions",
                        "true_label": lab})
    return out


def synthetic_ood_images(size: int = 224, seed: int = 0) -> list[dict]:
    """Images that look like NOTHING in the training set."""
    rng = np.random.default_rng(seed)
    noise = Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
    green = Image.new("RGB", (size, size), (40, 140, 60))
    grad = np.linspace(0, 255, size, dtype=np.uint8)
    stripes = Image.fromarray(np.stack([np.tile(grad, (size, 1))] * 3, axis=-1))
    return [{"image": noise, "name": "random noise", "group": "synthetic", "true_label": None},
            {"image": green, "name": "flat green square", "group": "synthetic", "true_label": None},
            {"image": stripes, "name": "grey gradient", "group": "synthetic", "true_label": None}]


def load_folder_images(folder: str | Path, group: str, max_side: int = 640) -> list[dict]:
    """Load every image in a folder (instructor photos, student uploads, ...)."""
    items = []
    folder = Path(folder)
    if not folder.exists():
        return items
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")  # respect phone rotation
        except Exception:  # HEIC without a plugin, corrupt file, ...
            print(f"   skipped {p.name} (could not open; try JPG/PNG)")
            continue
        im.thumbnail((max_side, max_side))
        items.append({"image": im, "name": p.stem.replace("_", " "), "group": group, "true_label": None})
    return items


# ---------------------------------------------------------------------------
# Prediction with confidence
# ---------------------------------------------------------------------------
def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


@torch.no_grad()
def predict_images(model, items: list[dict], transform, device: torch.device, class_names: list[str]) -> list[dict]:
    """Add top-1 prediction, confidence (MSP) and entropy to each item."""
    model.eval()
    if not items:
        return []
    x = torch.stack([transform(it["image"]) for it in items]).to(device)
    probs = torch.softmax(model(x).float(), dim=1).cpu().numpy()
    for it, p in zip(items, probs):
        k = int(p.argmax())
        it.update(pred=class_names[k], confidence=float(p[k]), entropy=float(entropy(p)), probs=p)
    return items


# ---------------------------------------------------------------------------
# Selective prediction
# ---------------------------------------------------------------------------
def selective_curve(probs: np.ndarray, labels: np.ndarray, thresholds=None):
    """Coverage and accuracy-on-accepted for each confidence threshold tau."""
    import pandas as pd

    thresholds = np.linspace(0.0, 0.99, 100) if thresholds is None else np.asarray(thresholds)
    conf, pred = probs.max(1), probs.argmax(1)
    rows = []
    for t in thresholds:
        keep = conf >= t
        cov = keep.mean()
        acc = (pred[keep] == labels[keep]).mean() if keep.any() else np.nan
        rows.append({"threshold": float(t), "coverage": float(cov), "accuracy": float(acc), "n_auto": int(keep.sum())})
    return pd.DataFrame(rows)


def pick_threshold(curve, target_accuracy: float = 0.98) -> float:
    """Smallest threshold whose accepted predictions reach the target accuracy."""
    ok = curve[(curve["accuracy"] >= target_accuracy) & (curve["n_auto"] > 0)]
    return float(ok["threshold"].min()) if len(ok) else float(curve["threshold"].max())


def route(confidence: float, threshold: float) -> str:
    return "auto" if confidence >= threshold else "human review"
