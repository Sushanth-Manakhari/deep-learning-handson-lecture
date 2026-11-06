"""🛟 Catch-up helpers — the "Don't Panic" safety net.

Fell behind? Restarted the runtime? Mistyped a cell? Each notebook section
starts with ONE catch-up cell that calls these functions. They rebuild
everything that section needs (device, config, data, a trained model) and are
safe to run any number of times: trained weights are cached in
runs/checkpoints/ so a second call loads instead of retraining.
"""
from __future__ import annotations

import torch

from . import CLASS_NAMES_BEANS
from .config import text_config_for, vision_config_for
from .data_text import build_tokenizer, load_banking77, make_text_loaders
from .data_vision import ensure_beans, make_vision_loaders, repo_root
from .device import get_device, set_seed
from .models import build_text_model, build_vision_model, trainable_params
from .train import build_optimizer, build_scheduler, fit, load_checkpoint

CKPT_DIR = repo_root() / "runs" / "checkpoints"


def vision_state(device: torch.device | None = None, verbose: bool = True, **overrides) -> dict:
    device = device or get_device()
    cfg = vision_config_for(device, **overrides)
    set_seed(cfg.seed)
    beans_dir = ensure_beans(verbose=verbose)
    loaders = make_vision_loaders(cfg, beans_dir, device)
    if verbose:
        print(f"🛟 vision ready: {cfg.model_name} @ {cfg.image_size}px on {device} (profile '{cfg.profile}')")
    return {"device": device, "cfg": cfg, "beans_dir": beans_dir, "loaders": loaders, "class_names": CLASS_NAMES_BEANS}


def trained_vision_model(state: dict, retrain: bool = False, verbose: bool = True):
    """A ResNet with a trained head: load from cache or train it now."""
    cfg, device, loaders = state["cfg"], state["device"], state["loaders"]
    model = build_vision_model(cfg.model_name, len(CLASS_NAMES_BEANS), cfg.pretrained, freeze_backbone=True)
    path = CKPT_DIR / f"vision_{cfg.model_name}_{cfg.image_size}_head.safetensors"
    if path.exists() and not retrain:
        try:
            load_checkpoint(model, path, device)
            if verbose:
                print(f"🛟 loaded trained head from {path.relative_to(repo_root())}")
            return model.to(device)
        except Exception:  # config changed -> shapes differ -> retrain
            pass
    if verbose:
        print("🛟 training the classification head (takes a moment)...")
    opt = build_optimizer(cfg.optimizer, trainable_params(model), cfg.lr, cfg.weight_decay)
    fit(model, loaders["train"], loaders["validation"], epochs=cfg.epochs, optimizer=opt, device=device,
        amp=cfg.amp, ckpt_path=path, verbose=verbose, name="vision-head")
    load_checkpoint(model, path, device)  # best epoch, not last
    return model.to(device)


def text_state(device: torch.device | None = None, verbose: bool = True, **overrides) -> dict:
    device = device or get_device()
    cfg = text_config_for(device, **overrides)
    set_seed(cfg.seed)
    data = load_banking77()
    tokenizer = build_tokenizer(cfg.model_name, texts=data["train"]["text"])
    loaders = make_text_loaders(cfg, data, tokenizer, device)
    if verbose:
        print(f"🛟 text ready: {len(loaders['train'].dataset)} train msgs, {len(data['label_names'])} intents, "
              f"device {device} (profile '{cfg.profile}')")
    return {"device": device, "cfg": cfg, "data": data, "tokenizer": tokenizer, "loaders": loaders}


def trained_text_model(state: dict, retrain: bool = False, verbose: bool = True):
    """DistilBERT fully fine-tuned on Banking77: load from cache or train now."""
    cfg, device, data, tok, loaders = (state[k] for k in ("cfg", "device", "data", "tokenizer", "loaders"))
    model = build_text_model(cfg.model_name, len(data["label_names"]), data["id2label"], data["label2id"],
                             vocab_size=len(tok))
    path = CKPT_DIR / f"text_{cfg.model_name.split('/')[-1]}_{cfg.profile}.safetensors"
    if path.exists() and not retrain:
        try:
            load_checkpoint(model, path, device)
            if verbose:
                print(f"🛟 loaded fine-tuned text model from {path.relative_to(repo_root())}")
            return model.to(device)
        except Exception:
            pass
    if verbose:
        print("🛟 fine-tuning DistilBERT (takes a moment)...")
    opt = build_optimizer("adamw", trainable_params(model), cfg.lr, cfg.weight_decay)
    steps = cfg.epochs * len(loaders["train"])
    sched = build_scheduler(opt, steps, int(cfg.warmup_ratio * steps), kind="linear")
    fit(model, loaders["train"], loaders["validation"], epochs=cfg.epochs, optimizer=opt, scheduler=sched,
        device=device, amp=cfg.amp, grad_clip=cfg.grad_clip, ckpt_path=path, verbose=verbose, name="text-ft")
    load_checkpoint(model, path, device)
    return model.to(device)
