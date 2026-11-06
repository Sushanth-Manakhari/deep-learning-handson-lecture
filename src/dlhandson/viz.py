"""Plots for the notebooks and slides (matplotlib, projector-friendly).

Style rules used everywhere: one y-axis per chart, thin 2 px lines, light grid,
a legend whenever there are 2+ series, labels in neutral ink (never coloured
text), a fixed colour order so "train" / "validation" keep the same colour in
every chart.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import torch

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 13,
        "axes.titleweight": "semibold", "axes.labelsize": 11, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2, "font.size": 11, "lines.linewidth": 2.0,
        "legend.frameon": False, "figure.dpi": 110,
    })


set_style()


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def _to_numpy_image(img):
    if isinstance(img, torch.Tensor):
        from .data_vision import denormalize

        return denormalize(img.detach().cpu()).permute(1, 2, 0).numpy()
    return np.asarray(img)


def show_images(images, titles=None, ncols: int = 6, size: float = 2.2, suptitle: str | None = None):
    n = len(images)
    nrows = max(1, math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows + (0.4 if suptitle else 0)))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(_to_numpy_image(images[i]))
            if titles:
                ax.set_title(titles[i], fontsize=9, color=INK)
    if suptitle:
        fig.suptitle(suptitle, color=INK, fontsize=13)
    fig.tight_layout()
    plt.show()


def show_class_examples(ds, per_class: int = 4, seed: int = 0):
    """Grid of raw training photos, `per_class` per class (ds: ImageFolder w/o transform)."""
    rng = np.random.default_rng(seed)
    images, titles = [], []
    targets = np.array([y for _, y in ds.samples])
    for c, name in enumerate(ds.classes):
        for i in rng.choice(np.where(targets == c)[0], size=per_class, replace=False):
            images.append(ds[int(i)][0])
            titles.append(name)
    show_images(images, titles, ncols=per_class * 1 if per_class >= 4 else 6)


def show_predictions(items: list[dict], ncols: int = 5, threshold: float | None = None):
    """Image gallery titled with prediction + confidence (+ routing decision)."""
    titles = []
    for it in items:
        t = f"{it['name']}\n→ {it['pred']} ({it['confidence']:.0%})"
        if threshold is not None:
            t += "\n✅ auto" if it["confidence"] >= threshold else "\n🧑 human review"
        titles.append(t)
    show_images([it["image"] for it in items], titles, ncols=ncols, size=2.6)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def plot_history(histories: dict, title: str = "Training curves"):
    """Loss and accuracy side by side (two charts, never a dual axis)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, (name, h) in enumerate(histories.items()):
        c = SERIES[i % len(SERIES)]
        ep = h.column("epoch")
        a1.plot(ep, h.column("train_loss"), color=c, linestyle="--", alpha=0.8, label=f"{name} · train")
        a1.plot(ep, h.column("val_loss"), color=c, label=f"{name} · validation")
        a2.plot(ep, h.column("train_acc"), color=c, linestyle="--", alpha=0.8, label=f"{name} · train")
        a2.plot(ep, h.column("val_acc"), color=c, marker="o", markersize=5, label=f"{name} · validation")
    a1.set(title="Loss (lower is better)", xlabel="epoch", ylabel="cross-entropy")
    a2.set(title="Accuracy (higher is better)", xlabel="epoch", ylabel="accuracy", ylim=(0, 1.02))
    a2.legend(loc="lower right", fontsize=9)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    plt.show()


def plot_curves(curves: dict, title: str, xlabel: str = "step", ylabel: str = "loss", logy: bool = False):
    """Generic multi-series line chart (gradient descent, explosion demos, ...)."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, (name, ys) in enumerate(curves.items()):
        ax.plot(range(len(ys)), ys, color=SERIES[i % len(SERIES)], label=str(name))
    if logy:
        ax.set_yscale("log")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if len(curves) > 1:
        ax.legend(fontsize=9)
    fig.tight_layout()
    plt.show()


def plot_grad_norms(norms: dict, title: str = "Gradient size per layer (input → output)"):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, (name, ys) in enumerate(norms.items()):
        ax.plot(range(1, len(ys) + 1), np.maximum(ys, 1e-30), color=SERIES[i], marker="o", markersize=4, label=name)
    ax.set_yscale("log")
    ax.set(title=title, xlabel="hidden layer (1 = closest to the input)", ylabel="‖∂L/∂W‖ (log scale)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.show()


def plot_lr_schedule(lrs: list, title: str = "Learning-rate schedule"):
    plot_curves({"learning rate": lrs}, title, xlabel="optimizer step", ylabel="lr")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, normalize: bool = True, title: str = "Confusion matrix"):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    shown = cm / cm.sum(1, keepdims=True).clip(min=1) if normalize else cm
    fig, ax = plt.subplots(figsize=(1.5 + 1.3 * len(class_names), 1.2 + 1.1 * len(class_names)))
    ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.grid(False)
    ax.set_xticks(range(len(class_names)), class_names, rotation=20, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            v = shown[i, j]
            ax.text(j, i, f"{v:.0%}\n({cm[i, j]})" if normalize else str(v), ha="center", va="center",
                    color="white" if (v > 0.6 if normalize else v > shown.max() * 0.6) else INK, fontsize=10)
    ax.set(xlabel="predicted", ylabel="actual", title=title)
    fig.tight_layout()
    plt.show()


def most_confused(y_true, y_pred, names, k: int = 10):
    """Top-k (actual -> predicted) mistakes; readable even with 77 classes."""
    import pandas as pd

    pairs = pd.DataFrame({"actual": [names[i] for i in y_true], "predicted": [names[i] for i in y_pred]})
    wrong = pairs[pairs.actual != pairs.predicted]
    return wrong.value_counts().head(k).rename("count").reset_index()


def plot_selective(curve, chosen: float | None = None, title: str = "Confidence threshold trade-off"):
    """Coverage and accuracy vs threshold — two small charts sharing the x-axis."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    a1.plot(curve.threshold, curve.coverage, color=SERIES[0])
    a1.set(title="Coverage: share decided automatically", xlabel="confidence threshold τ", ylabel="coverage",
           ylim=(0, 1.02))
    a2.plot(curve.threshold, curve.accuracy, color=SERIES[1])
    a2.set(title="Accuracy on automatic decisions", xlabel="confidence threshold τ", ylabel="accuracy")
    if chosen is not None:
        for ax in (a1, a2):
            ax.axvline(chosen, color=INK2, linestyle=":", linewidth=1.5)
            ax.annotate(f"τ = {chosen:.2f}", (chosen, ax.get_ylim()[0]), xytext=(4, 6), textcoords="offset points",
                        color=INK2, fontsize=9)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    plt.show()


def plot_bar(counts: dict, title: str, xlabel: str = "", ylabel: str = "count", horizontal: bool = False):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    names, vals = list(counts.keys()), list(counts.values())
    if horizontal:
        ax.barh(names, vals, color=SERIES[0], height=0.6)
        ax.invert_yaxis()
    else:
        ax.bar(names, vals, color=SERIES[0], width=0.6)
    for i, v in enumerate(vals):
        if horizontal:
            ax.text(v, i, f" {v}", va="center", color=INK2, fontsize=9)
        else:
            ax.text(i, v, f"{v}", ha="center", va="bottom", color=INK2, fontsize=9)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout()
    plt.show()


def plot_histogram(values, title: str, xlabel: str, marker: float | None = None, marker_label: str = ""):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(values, bins=30, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    if marker is not None:
        ax.axvline(marker, color=INK2, linestyle=":", linewidth=1.5)
        ax.annotate(marker_label, (marker, ax.get_ylim()[1] * 0.9), xytext=(5, 0), textcoords="offset points",
                    color=INK2, fontsize=9)
    ax.set(title=title, xlabel=xlabel, ylabel="count")
    fig.tight_layout()
    plt.show()


def plot_decision_boundary(model, X, y, device=None, title: str = "Decision boundary"):
    """Colour the 2-D plane by the model's predicted class (toy datasets)."""
    device = device or torch.device("cpu")
    x0, x1 = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y0, y1 = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        zz = torch.softmax(model(grid), 1)[:, 1].cpu().numpy().reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.35)
    for c in (0, 1):
        ax.scatter(X[y == c, 0], X[y == c, 1], s=14, color=SERIES[c], label=f"class {c}",
                   edgecolor=SURFACE, linewidth=0.5)
    ax.legend(fontsize=9)
    ax.set(title=title, xlabel="x₁", ylabel="x₂")
    ax.grid(False)
    fig.tight_layout()
    plt.show()
