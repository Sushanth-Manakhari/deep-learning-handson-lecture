"""The canonical training loop — the same five steps for every network.

    for each batch (x, y):
        1. optimizer.zero_grad()      # gradients ACCUMULATE in PyTorch -> reset
        2. z = model(x)               # forward pass  -> logits
        3. L = loss(z, y)             # how wrong are we?
        4. L.backward()               # backprop: dL/dtheta for every parameter
        5. optimizer.step()           # theta <- theta - lr * update(dL/dtheta)

LOSS: CROSS-ENTROPY
-------------------
softmax  p_k = exp(z_k) / sum_j exp(z_j)
loss     L   = -log p_y            (y = correct class; average over the batch)
gradient dL/dz_k = p_k - 1[k = y]   -> push the right class up, the others down.
Label smoothing replaces the one-hot target with (1 - eps) on y and eps/K
elsewhere, which discourages over-confident predictions.

OPTIMISERS (g = gradient, eta = learning rate)
-----------------------------------------------
SGD + momentum : v <- mu*v + g ;  theta <- theta - eta*v
Adam           : m <- b1*m + (1-b1)*g ;  v <- b2*v + (1-b2)*g^2
                 theta <- theta - eta * m_hat / (sqrt(v_hat) + eps)
AdamW          : Adam + DECOUPLED weight decay  theta <- theta - eta*lambda*theta

MIXED PRECISION (AMP, CUDA only)
--------------------------------
Run matmuls in float16 on Tensor Cores (~2x faster on a T4, ~half the
activation memory). float16 can underflow tiny gradients to 0, so GradScaler
multiplies the loss by S (e.g. 65,536) before backward and divides the
gradients by S before the update; it skips a step if any gradient is inf/NaN.

GRADIENT CLIPPING
-----------------
if ||g||_2 > tau:  g <- g * tau / ||g||_2   (caps the step size; tames explosions)
"""
from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .models import forward_logits, trainable_params


class NonFiniteLossError(RuntimeError):
    """Raised when the loss becomes inf or NaN (training has diverged)."""


def move_batch(batch: dict, device: torch.device) -> dict:
    nb = device.type == "cuda"
    return {k: (v.to(device, non_blocking=nb) if torch.is_tensor(v) else v) for k, v in batch.items()}


def autocast_ctx(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Optimiser and learning-rate schedule
# ---------------------------------------------------------------------------
def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0, momentum: float = 0.9):
    """Create an optimiser by name: 'adamw' (default), 'adam' or 'sgd'."""
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=momentum > 0)
    raise ValueError(f"unknown optimizer '{name}' (adamw | adam | sgd)")


def build_scheduler(optimizer, total_steps: int, warmup_steps: int = 0, kind: str = "cosine"):
    """Linear WARMUP then cosine (or linear) DECAY, stepped once per batch.

    warmup: lr_t = lr * t / W            (t < W) — avoids huge early updates while
            the new head is random and Adam's statistics are still unreliable.
    cosine: lr_t = lr * 0.5 * (1 + cos(pi * progress))   -> smooth decay to 0
    """
    total_steps = max(1, total_steps)

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress)) if kind == "cosine" else 1.0 - progress

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# ---------------------------------------------------------------------------
# History container
# ---------------------------------------------------------------------------
@dataclass
class History:
    name: str = "run"
    rows: list = field(default_factory=list)
    diverged: bool = False
    best_val_acc: float = 0.0
    best_epoch: int = -1

    def add(self, **row) -> None:
        self.rows.append(row)

    def column(self, key: str) -> list:
        return [r.get(key) for r in self.rows]

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(self.rows)

    @property
    def last(self) -> dict:
        return self.rows[-1] if self.rows else {}


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def train_one_epoch(model: nn.Module, loader, optimizer, device: torch.device, *, scaler=None, scheduler=None,
                    grad_clip: float | None = None, label_smoothing: float = 0.0, max_steps: int | None = None) -> dict:
    model.train()  # enables dropout + batch-norm batch statistics
    params = trainable_params(model)
    tot_loss, correct, n, gnorms = 0.0, 0, 0, []
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        batch = move_batch(batch, device)
        labels = batch["labels"]
        optimizer.zero_grad(set_to_none=True)                       # 1
        with autocast_ctx(device, scaler is not None):
            logits = forward_logits(model, batch)                   # 2
            loss = F.cross_entropy(logits.float(), labels, label_smoothing=label_smoothing)  # 3
        if not torch.isfinite(loss):
            raise NonFiniteLossError(f"loss became {loss.item()} at step {step}")
        if scaler is not None:
            scaler.scale(loss).backward()                           # 4 (scaled)
            scaler.unscale_(optimizer)                              # true gradients for clipping
        else:
            loss.backward()                                         # 4
        # clip_grad_norm_ returns the total norm ||g||_2 BEFORE clipping, so it
        # doubles as a cheap health monitor (exploding -> huge, vanishing -> ~0).
        gnorm = torch.nn.utils.clip_grad_norm_(params, grad_clip if grad_clip else float("inf"))
        gnorms.append(float(gnorm))
        if scaler is not None:
            scaler.step(optimizer)                                  # 5 (skips if inf/NaN)
            scaler.update()
        else:
            optimizer.step()                                        # 5
        if scheduler is not None:
            scheduler.step()
        bs = labels.size(0)
        tot_loss += loss.item() * bs
        correct += (logits.argmax(1) == labels).sum().item()
        n += bs
    finite = [g for g in gnorms if math.isfinite(g)]
    return {"loss": tot_loss / max(n, 1), "acc": correct / max(n, 1),
            "grad_norm": float(np.mean(finite)) if finite else float("nan"),
            "lr": optimizer.param_groups[0]["lr"], "steps": len(gnorms)}


@torch.no_grad()  # no graph is built -> faster, less memory
def evaluate(model: nn.Module, loader, device: torch.device, amp: bool = False, return_outputs: bool = False) -> dict:
    model.eval()  # dropout off, batch-norm uses running statistics
    tot_loss, n = 0.0, 0
    all_probs, all_labels = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast_ctx(device, amp):
            logits = forward_logits(model, batch).float()
        loss = F.cross_entropy(logits, batch["labels"], reduction="sum")
        tot_loss += loss.item()
        n += batch["labels"].size(0)
        all_probs.append(torch.softmax(logits, dim=1).cpu())
        all_labels.append(batch["labels"].cpu())
    probs = torch.cat(all_probs).numpy() if all_probs else np.zeros((0, 1))
    labels = torch.cat(all_labels).numpy() if all_labels else np.zeros((0,), dtype=int)
    preds = probs.argmax(1) if len(probs) else labels
    out = {"loss": tot_loss / max(n, 1), "acc": float((preds == labels).mean()) if n else 0.0}
    if return_outputs:
        out.update(probs=probs, labels=labels, preds=preds)
    return out


def save_checkpoint(model: nn.Module, path: str | Path) -> Path:
    """Save weights with safetensors.

    SECURITY: classic torch.save() uses Python pickle, and unpickling a
    malicious file can execute code. safetensors stores raw tensors only.
    """
    from safetensors.torch import save_model

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(path))
    return path


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device | str = "cpu") -> nn.Module:
    from safetensors.torch import load_model

    load_model(model, str(path), device=str(device))
    return model


def fit(model: nn.Module, train_loader, val_loader, *, epochs: int, optimizer, device: torch.device,
        scheduler=None, amp: bool = False, grad_clip: float | None = None, label_smoothing: float = 0.0,
        logger=None, ckpt_path: str | Path | None = None, name: str = "run", verbose: bool = True,
        max_steps_per_epoch: int | None = None) -> History:
    """Train for `epochs`, validate after each, keep the best checkpoint.

    Never crashes on divergence: a NaN/inf loss stops training with an
    explanation and returns the history so far (history.diverged = True).
    """
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    hist = History(name=name)
    model.to(device)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        try:
            tr = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler, scheduler=scheduler,
                                 grad_clip=grad_clip, label_smoothing=label_smoothing, max_steps=max_steps_per_epoch)
        except NonFiniteLossError as exc:
            hist.diverged = True
            print(f"💥 Epoch {epoch}: training DIVERGED ({exc}).\n"
                  "   Typical causes: learning rate too high, no normalisation, bad data (NaN/inf inputs).\n"
                  "   Typical fixes : lower the LR, gradient clipping, warmup, check inputs.")
            break
        va = evaluate(model, val_loader, device, amp=use_amp)
        dt = time.time() - t0
        hist.add(epoch=epoch, train_loss=tr["loss"], train_acc=tr["acc"], val_loss=va["loss"], val_acc=va["acc"],
                 grad_norm=tr["grad_norm"], lr=tr["lr"], seconds=round(dt, 1))
        improved = va["acc"] > hist.best_val_acc
        if improved:
            hist.best_val_acc, hist.best_epoch = va["acc"], epoch
            if ckpt_path:
                save_checkpoint(model, ckpt_path)
        if logger is not None:
            logger.log({"train/loss": tr["loss"], "train/acc": tr["acc"], "val/loss": va["loss"],
                        "val/acc": va["acc"], "grad_norm": tr["grad_norm"], "lr": tr["lr"]}, step=epoch)
        if verbose:
            star = " *" if improved else ""
            print(f"Epoch {epoch}/{epochs} | train loss {tr['loss']:.3f} acc {tr['acc']:.3f} | "
                  f"val loss {va['loss']:.3f} acc {va['acc']:.3f} | {dt:5.1f}s | lr {tr['lr']:.1e}{star}")
    return hist
