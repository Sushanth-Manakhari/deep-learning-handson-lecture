"""Debugging a network that "won't learn" — the practitioner's checklist.

 1. OVERFIT ONE BATCH (the golden rule). A healthy model + pipeline can drive
    training accuracy on ~5-16 examples to 100%. If it can't, the bug is in
    your code (optimizer, labels, loss, data), not in "not enough data".
 2. CHECK THE OPTIMIZER SEES YOUR PARAMETERS. Classic transfer-learning bug:
    the optimizer is built, THEN the head is replaced -> the optimizer keeps
    updating the old, detached head; the new one never moves.
 3. WATCH GRADIENT NORMS. ||g|| -> 0 in early layers = VANISHING;
    ||g|| -> huge / inf / NaN = EXPLODING.
 4. FIND THE BATCH-SIZE CEILING without crashing (catch out-of-memory).

VANISHING GRADIENTS — the math
------------------------------
Backprop multiplies one local derivative per layer (chain rule):
    dL/dh_1 = dL/dh_L * prod_{l=2..L} W_l^T diag(sigma'(a_l))
The sigmoid's derivative is at most 0.25, so after 20 layers the factor can be
~0.25^20 ~ 1e-12: early layers stop learning. Fixes: ReLU (derivative 1 for
positive inputs), careful initialisation (He/Kaiming), normalisation layers,
and residual connections h_{l+1} = h_l + F(h_l), whose Jacobian I + dF/dh
keeps an identity path for the gradient.

EXPLODING GRADIENTS — the mirror image: factors > 1 multiply up, updates
overshoot, the loss jumps to inf, then NaN (inf - inf). Fixes: lower LR,
gradient clipping, normalisation, warmup.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .device import free_memory
from .models import forward_logits, trainable_params
from .train import move_batch


# ---------------------------------------------------------------------------
# 1 + 2: overfit one batch, optimizer sanity check
# ---------------------------------------------------------------------------
def overfit_one_batch(model: nn.Module, batch: dict, device: torch.device, optimizer=None, lr: float = 1e-3,
                      steps: int = 60, verbose: bool = True) -> dict:
    """Train repeatedly on ONE batch. Expect loss -> ~0 and accuracy -> 100%."""
    model.to(device).train()
    batch = move_batch(batch, device)
    opt = optimizer or torch.optim.AdamW(trainable_params(model), lr=lr)
    losses, accs = [], []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = forward_logits(model, batch)
        loss = F.cross_entropy(logits, batch["labels"])
        loss.backward()
        opt.step()
        losses.append(loss.item())
        accs.append((logits.argmax(1) == batch["labels"]).float().mean().item())
    ok = accs[-1] == 1.0 and losses[-1] < 0.1
    if verbose:
        n = batch["labels"].numel()
        verdict = ("✅ PASS — the model can memorise this batch, so the training code is wired correctly."
                   if ok else
                   "❌ FAIL — it cannot even memorise a handful of examples. That is a BUG, not a data problem.\n"
                   "   Check: does the optimizer hold the right parameters? labels aligned? loss on logits?")
        print(f"Overfit {n} examples for {steps} steps: loss {losses[0]:.3f} -> {losses[-1]:.4f}, "
              f"train acc {accs[0]:.0%} -> {accs[-1]:.0%}\n{verdict}")
    return {"losses": losses, "accs": accs, "passed": ok}


def optimizer_covers_model(optimizer, model: nn.Module, verbose: bool = True) -> bool:
    """True if every trainable parameter of `model` is managed by `optimizer`."""
    in_opt = {id(p) for g in optimizer.param_groups for p in g["params"]}
    missing = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in in_opt]
    if verbose:
        if missing:
            print(f"❌ {len(missing)} trainable tensors are NOT in the optimizer, e.g. {missing[:3]}\n"
                  "   -> rebuild the optimizer AFTER changing the model.")
        else:
            print("✅ optimizer manages every trainable parameter.")
    return not missing


# ---------------------------------------------------------------------------
# 3: gradient health — vanishing / exploding demos
# ---------------------------------------------------------------------------
class DeepMLP(nn.Module):
    """A deliberately deep MLP for gradient experiments (2-D inputs)."""

    def __init__(self, depth: int = 20, width: int = 32, activation: str = "sigmoid", residual: bool = False,
                 in_dim: int = 2, out_dim: int = 2):
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.out = nn.Linear(width, out_dim)
        self.act = {"sigmoid": torch.sigmoid, "tanh": torch.tanh, "relu": F.relu}[activation]
        self.residual = residual
        if activation == "relu":  # He/Kaiming init keeps activation variance ~constant for ReLU
            for layer in self.hidden:
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)
                if residual:  # scale residual branches down so the sum stays stable
                    layer.weight.data.mul_(1 / math.sqrt(depth))

    def forward(self, x):
        h = self.act(self.inp(x))
        for layer in self.hidden:
            h = h + self.act(layer(h)) if self.residual else self.act(layer(h))
        return self.out(h)


def layer_grad_norms(model: DeepMLP, X: torch.Tensor, y: torch.Tensor) -> list[float]:
    """One forward/backward pass; return ||dL/dW_l|| for each hidden layer (input -> output)."""
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(X), y).backward()
    return [layer.weight.grad.norm().item() for layer in model.hidden]


def vanishing_gradient_experiment(depth: int = 20, seed: int = 0) -> dict:
    """Compare per-layer gradient norms: sigmoid vs ReLU vs ReLU + residual."""
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=256, noise=0.15, random_state=seed)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y)
    out = {}
    for label, act, res in (("sigmoid", "sigmoid", False), ("relu", "relu", False), ("relu + residual", "relu", True)):
        torch.manual_seed(seed)
        out[label] = layer_grad_norms(DeepMLP(depth, activation=act, residual=res), X, y)
    return out


def exploding_experiment(lrs=(0.01, 0.1, 1.0), steps: int = 40, clip: float | None = None, seed: int = 0) -> dict:
    """Train a small regression MLP with SGD at different learning rates.

    Gradient descent on a quadratic-like loss is stable only if lr < 2/lambda_max
    (lambda_max = largest curvature/Hessian eigenvalue). Above that, each step
    overshoots further than the last: loss -> inf -> NaN.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(256, 8, generator=g) * 3
    y = (X @ torch.randn(8, 1, generator=g)) + 0.1 * torch.randn(256, 1, generator=g)
    results = {}
    for lr in lrs:
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        losses = []
        for _ in range(steps):
            opt.zero_grad()
            loss = F.mse_loss(model(X), y)
            if not torch.isfinite(loss):
                losses.append(float("nan"))
                break
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            losses.append(loss.item())
        results[lr] = losses
    return results


def find_nonfinite(model: nn.Module) -> list[str]:
    """Names of parameters (or their gradients) that contain inf/NaN."""
    bad = []
    for n, p in model.named_parameters():
        if not torch.isfinite(p).all() or (p.grad is not None and not torch.isfinite(p.grad).all()):
            bad.append(n)
    return bad


# ---------------------------------------------------------------------------
# 4: batch size vs GPU memory, without crashing the notebook
# ---------------------------------------------------------------------------
def _is_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    return isinstance(exc, oom_type) or "out of memory" in str(exc).lower()


def try_batch_size(model: nn.Module, batch_size: int, image_size: int, device: torch.device,
                   num_classes: int = 3, amp: bool = False) -> dict:
    """One training step with random data. Returns ok / peak memory / error.

    MEMORY BUDGET of a training step:
      weights + gradients + optimizer state (Adam keeps 2 extra copies)
      + ACTIVATIONS saved for backprop  ~ batch_size x (pixels) x (depth x width)
    Activations dominate and grow LINEARLY with batch size (and quadratically
    with image side length) — that is why a big batch crashes with OOM.
    """
    model.to(device).train()
    free_memory(device)
    x = y = loss = None
    try:
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        y = torch.randint(0, num_classes, (batch_size,), device=device)
        ctx = torch.autocast("cuda", dtype=torch.float16) if (amp and device.type == "cuda") else torch.enable_grad()
        with ctx:
            loss = F.cross_entropy(model(x).float(), y)
        loss.backward()
        peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else float("nan")
        result = {"batch_size": batch_size, "ok": True, "peak_gb": round(peak, 2), "error": ""}
    except Exception as exc:  # noqa: BLE001 — we WANT to catch the OOM and keep going
        if not _is_oom(exc):
            raise
        result = {"batch_size": batch_size, "ok": False, "peak_gb": float("nan"), "error": "CUDA out of memory"}
    finally:
        model.zero_grad(set_to_none=True)
        x = y = loss = None  # drop references so the memory can be freed
        free_memory(device)
    return result


def find_max_batch_size(model: nn.Module, image_size: int, device: torch.device, start: int = 16,
                        limit: int = 1024, amp: bool = False, num_classes: int = 3) -> list[dict]:
    """Double the batch size until we run out of GPU memory (CUDA only)."""
    if device.type != "cuda":
        print("ℹ️  Skipped: this demo needs an NVIDIA GPU. On Apple Silicon, GPU memory is shared with the\n"
              "   whole Mac, so instead of a clean error the computer slows to a crawl — not a demo for class.")
        return []
    rows, bs = [], start
    while bs <= limit:
        r = try_batch_size(model, bs, image_size, device, num_classes=num_classes, amp=amp)
        rows.append(r)
        status = f"peak {r['peak_gb']:.2f} GB" if r["ok"] else "💥 OUT OF MEMORY (caught — notebook still alive)"
        print(f"batch {bs:>5}: {status}")
        if not r["ok"]:
            break
        bs *= 2
    return rows
