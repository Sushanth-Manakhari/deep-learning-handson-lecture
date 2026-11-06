"""Hardware detection, reproducibility and environment helpers.

WHY THIS MATTERS
----------------
Training a network is mostly matrix multiplication. For a linear layer
y = W x + b with W in R^{m x n}, one forward pass on a batch of B examples costs
about 2 * B * m * n floating-point operations (FLOPs). A free Colab T4 GPU does
roughly 65 TFLOP/s in half precision; a laptop CPU does well under 1 TFLOP/s.
Same code, ~50-100x faster -> that is the whole reason GPUs/TPUs power modern AI.

PyTorch exposes accelerators as "devices":
    cuda -> NVIDIA GPUs (Colab T4, Kaggle T4/P100, gaming laptops)
    mps  -> Apple Silicon GPU (M1/M2/M3/M4 Macs) via Metal Performance Shaders
    cpu  -> always available, slowest
"""
from __future__ import annotations

import os
import platform
import random
import sys

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
def is_colab() -> bool:
    """True when running inside Google Colab."""
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def is_kaggle() -> bool:
    """True when running inside a Kaggle Notebook."""
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def smoke_mode() -> bool:
    """Smoke mode = tiny models + tiny data, no downloads.

    Used by automated tests (CI) and by the sandbox that verifies the notebooks
    run top to bottom. Never used in class. Enable with:  DLH_SMOKE=1
    """
    return os.environ.get("DLH_SMOKE", "0") == "1"


def environment_name() -> str:
    if is_colab():
        return "Google Colab"
    if is_kaggle():
        return "Kaggle Notebooks"
    return f"Local ({platform.system()} {platform.machine()})"


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def get_device(prefer: str | None = None) -> torch.device:
    """Return the fastest available device: CUDA -> MPS -> CPU.

    `prefer` (or the env var DLH_DEVICE) forces a choice, e.g. "cpu" to
    compare speeds or to reproduce a bug without the GPU.
    """
    choice = (prefer or os.environ.get("DLH_DEVICE", "")).lower().strip()
    if choice:
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> dict:
    """Human-readable facts about the device (printed in the setup cell)."""
    info = {"device": str(device), "torch": torch.__version__, "python": platform.python_version(),
            "environment": environment_name()}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info["gpu_name"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
        # Compute capability 7.5 = Turing (T4) -> has Tensor Cores for fp16.
        info["compute_capability"] = f"{props.major}.{props.minor}"
    elif device.type == "mps":
        info["gpu_name"] = "Apple Silicon GPU (MPS)"
    else:
        info["gpu_name"] = "none (CPU only) -> notebook switches to FAST mode"
        info["cpu_threads"] = torch.get_num_threads()
    return info


def gpu_memory_gb(device: torch.device) -> dict:
    """Allocated / peak / total GPU memory in GB (CUDA only; zeros elsewhere)."""
    if device.type != "cuda":
        return {"allocated": 0.0, "peak": 0.0, "total": 0.0}
    gb = 1024**3
    return {
        "allocated": round(torch.cuda.memory_allocated(device) / gb, 2),
        "peak": round(torch.cuda.max_memory_allocated(device) / gb, 2),
        "total": round(torch.cuda.get_device_properties(device).total_memory / gb, 2),
    }


def free_memory(device: torch.device | None = None) -> None:
    """Release cached GPU memory (call after an out-of-memory error)."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:  # older torch builds
            pass


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed every random number generator we use.

    Deep learning has randomness in weight initialisation, data shuffling,
    augmentation and dropout. Fixing seeds makes runs repeatable so that
    a change in accuracy is caused by YOUR change, not by luck.

    deterministic=True also forces deterministic GPU kernels: fully repeatable
    but slower (an engineering trade-off you will meet in production).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        # benchmark=True lets cuDNN pick the fastest convolution algorithm.
        torch.backends.cudnn.benchmark = True


def default_num_workers(device: torch.device) -> int:
    """DataLoader worker processes.

    Workers decode/augment images in parallel while the GPU trains.
    - Colab/Kaggle (Linux + CUDA): 2 workers keeps a T4 busy on this dataset.
    - macOS/Windows notebooks use "spawn" multiprocessing, which is slow to start
      and can hang inside Jupyter -> 0 workers (load in the main process).
    """
    if device.type == "cuda" and platform.system() == "Linux":
        return min(2, os.cpu_count() or 1)
    return 0
