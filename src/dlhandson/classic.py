"""The "2015 way": hand-crafted features + a classic ML model.

PARADIGM SHIFT IN ONE FILE
--------------------------
Traditional ML:  pixels -> [features a HUMAN designs] -> classifier
Deep learning:   pixels -> [features the NETWORK learns] -> classifier

Here a human decides that "colour distribution" and "edge texture" matter,
computes them by hand, and fits logistic regression. It works a little — and
it is exactly the ceiling deep learning broke through, because a CNN learns
thousands of better features (edges -> textures -> lesions -> leaf) from data.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def handcrafted_features(img: Image.Image, bins: int = 8, size: int = 128) -> np.ndarray:
    """HSV colour histogram (bins^3) + gradient-magnitude histogram (16)."""
    img = img.convert("RGB").resize((size, size))
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    hist, _ = np.histogramdd(hsv.reshape(-1, 3), bins=(bins, bins, bins), range=((0, 1),) * 3)
    hist = hist.ravel() / hist.sum()
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    gy, gx = np.gradient(gray)                      # finite-difference derivatives
    mag = np.sqrt(gx**2 + gy**2)                    # |grad I| = edge strength
    ghist, _ = np.histogram(mag, bins=16, range=(0, 0.5))
    ghist = ghist / max(ghist.sum(), 1)
    return np.concatenate([hist, ghist]).astype(np.float32)


def featurize_folder(ds, max_items: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Turn an ImageFolder (transform=None) into a feature matrix X and labels y."""
    n = len(ds) if max_items is None else min(max_items, len(ds))
    X = np.stack([handcrafted_features(ds[i][0]) for i in range(n)])
    y = np.array([ds[i][1] for i in range(n)])
    return X, y
