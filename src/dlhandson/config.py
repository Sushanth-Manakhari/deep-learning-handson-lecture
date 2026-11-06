"""Configuration objects — every "knob" of an experiment in one place.

Industry habit: never scatter hyperparameters through the code. Put them in a
config, log the config with every run, and you can always answer
"which settings produced this model?".

HARDWARE PROFILES
-----------------
The same notebook must finish in class on very different machines, so each
config is adjusted to the hardware it finds:

    profile  | when                     | vision model, image | text subset
    ---------+--------------------------+---------------------+------------
    gpu      | CUDA (Colab/Kaggle T4)   | ResNet-50, 224 px   | full data
    mps      | Apple Silicon Mac        | ResNet-50, 224 px   | full data
    cpu      | no GPU at all ("FAST")   | ResNet-18, 160 px   | 2,000 msgs
    smoke    | automated tests only     | tiny, random init   | 64 msgs
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch

from .device import smoke_mode


def hardware_profile(device: torch.device) -> str:
    if smoke_mode():
        return "smoke"
    return {"cuda": "gpu", "mps": "mps"}.get(device.type, "cpu")


@dataclass
class VisionConfig:
    """Ticket 1 — leaf-disease classifier (transfer learning on images)."""

    model_name: str = "resnet50"     # torchvision backbone: resnet18 | resnet50 | vit_b_16
    pretrained: bool = True          # ImageNet weights = "knowledge" we transfer
    freeze_backbone: bool = True     # train only the new head first
    image_size: int = 224            # ImageNet models were trained at 224x224
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-3                 # head-only training tolerates a larger LR
    weight_decay: float = 1e-4       # L2-style regularisation (AdamW: decoupled)
    optimizer: str = "adamw"         # adamw | sgd | adam
    momentum: float = 0.9            # only used by SGD
    augment: bool = True
    label_smoothing: float = 0.0
    grad_clip: float | None = None   # e.g. 1.0 clips ||g||_2 to 1.0
    amp: bool = True                 # mixed precision (CUDA only)
    max_train_samples: int | None = None
    num_workers: int | None = None   # None -> decided from the device
    seed: int = 42
    profile: str = "gpu"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TextConfig:
    """Ticket 2 — customer-message intent router (fine-tuning a Transformer)."""

    model_name: str = "distilbert/distilbert-base-uncased"
    max_length: int = 64             # Banking77 messages are short (95th pct ~ 40 tokens)
    batch_size: int = 32             # 32 -> ~280 optimizer steps per epoch on 9k messages
    epochs: int = 4                  # verified run: 3 epochs at batch 64 was still improving (underfit)
    lr: float = 5e-5                 # small LR: nudge, don't overwrite, pretrained weights
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1        # LR ramps up for the first 10% of steps
    freeze_encoder: bool = False
    amp: bool = True
    grad_clip: float | None = 1.0
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    seed: int = 42
    profile: str = "gpu"

    def to_dict(self) -> dict:
        return asdict(self)


def vision_config_for(device: torch.device, **overrides) -> VisionConfig:
    """Build a VisionConfig tuned for the hardware, then apply overrides."""
    profile = hardware_profile(device)
    cfg = VisionConfig(profile=profile)
    if profile == "cpu":
        cfg.model_name, cfg.image_size, cfg.epochs, cfg.max_train_samples = "resnet18", 160, 3, 450
    elif profile == "smoke":
        cfg.model_name, cfg.pretrained, cfg.image_size = "resnet18", False, 64
        cfg.epochs, cfg.batch_size, cfg.max_train_samples = 1, 8, 24
    if device.type != "cuda":
        cfg.amp = False  # fp16 autocast + GradScaler is a CUDA feature here
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"VisionConfig has no field '{k}'")
        setattr(cfg, k, v)
    return cfg


def text_config_for(device: torch.device, **overrides) -> TextConfig:
    """Build a TextConfig tuned for the hardware, then apply overrides."""
    profile = hardware_profile(device)
    cfg = TextConfig(profile=profile)
    if profile == "cpu":
        cfg.max_train_samples, cfg.max_eval_samples, cfg.epochs, cfg.max_length = 2000, 600, 1, 48
        cfg.batch_size = 32
    elif profile == "smoke":
        cfg.max_train_samples, cfg.max_eval_samples, cfg.epochs, cfg.max_length = 64, 32, 1, 24
        cfg.batch_size = 16
    if device.type != "cuda":
        cfg.amp = False
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"TextConfig has no field '{k}'")
        setattr(cfg, k, v)
    return cfg


@dataclass
class Paths:
    """Where things live inside the repository."""

    root: str = "."
    data: str = "data"
    beans: str = "data/beans"
    banking77: str = "data/banking77"
    stress_test: str = "data/stress_test"
    runs: str = "runs"
    extra: dict = field(default_factory=dict)
