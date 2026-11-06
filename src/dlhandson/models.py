"""Models for both tickets — and the core idea of TRANSFER LEARNING.

THE IDEA IN ONE EQUATION
------------------------
Write a classifier as a composition   f(x) = h_phi( g_theta(x) )

    g_theta : the BACKBONE / feature extractor (millions of parameters,
              pretrained on ImageNet-1k (1.28M images) or on billions of words)
    h_phi   : the HEAD — a small layer mapping features -> our K classes

Transfer learning keeps g_theta (the "knowledge" of edges, textures, shapes,
or of grammar and meaning) and learns only what is new:

    step 1  freeze theta  (requires_grad = False -> no gradients, no updates)
    step 2  replace h with a fresh layer for OUR classes
    step 3  train phi (and later, optionally, the top of theta with a small LR)

Why it works with ~1,000 images: we optimise ~6 thousand parameters instead of
~23 million, so far less data is needed to avoid overfitting, and it trains
in seconds instead of days.

ResNet's key trick (He et al., 2015): RESIDUAL connections  y = F(x) + x.
The identity path lets gradients flow straight through 50+ layers
(dy/dx = dF/dx + I), which is what made very deep networks trainable.
"""
from __future__ import annotations

import torch
from torch import nn

from .device import get_device, smoke_mode


# ---------------------------------------------------------------------------
# Freezing utilities
# ---------------------------------------------------------------------------
def freeze(module: nn.Module) -> None:
    """Stop gradient computation for every parameter in `module`."""
    for p in module.parameters():
        p.requires_grad = False


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_pct": round(100 * trainable / max(total, 1), 3)}


def model_size_mb(model: nn.Module) -> float:
    """Memory taken by the weights alone (fp32 = 4 bytes per parameter)."""
    return round(sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6, 1)


def trainable_params(model: nn.Module) -> list:
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------
def build_vision_model(name: str = "resnet50", num_classes: int = 3, pretrained: bool = True,
                       freeze_backbone: bool = True) -> nn.Module:
    """A torchvision ImageNet model with a NEW head for `num_classes`.

    resnet18 : 11.7M params, ~45 MB download  (CPU-friendly)
    resnet50 : 25.6M params, ~98 MB download  (default on GPU)
    vit_b_16 : 86.6M params, ~330 MB download (Vision Transformer; demo)
    """
    from torchvision import models as tvm

    if name in ("resnet18", "resnet50"):
        weights = None
        if pretrained:
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if name == "resnet18" else tvm.ResNet50_Weights.IMAGENET1K_V2
        model = getattr(tvm, name)(weights=weights)
        if freeze_backbone:
            freeze(model)
        # The ImageNet head maps 512/2048 features -> 1000 classes. Replace it.
        # A freshly created layer has requires_grad=True, so ONLY it will train.
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "vit_b_16":
        weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.vit_b_16(weights=weights)
        if freeze_backbone:
            freeze(model)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"unknown vision model '{name}' (use resnet18 | resnet50 | vit_b_16)")
    model.dlh_name = name
    return model


def get_head(model: nn.Module) -> nn.Module:
    """Return the classification head of any model built in this module."""
    if hasattr(model, "fc"):
        return model.fc
    if hasattr(model, "heads"):
        return model.heads.head
    if hasattr(model, "classifier"):
        return model.classifier
    raise ValueError("model has no recognised head")


def unfreeze_top(model: nn.Module, n_blocks: int = 1) -> list[str]:
    """Unfreeze the top `n_blocks` of the backbone for FINE-TUNING.

    Early layers learn generic features (edges, colours) that transfer to any
    task; later layers learn task-specific combinations. So we unfreeze from the
    TOP down, and give those layers a smaller learning rate (see param_groups).
    """
    names: list[str] = []
    if hasattr(model, "layer4"):  # ResNet
        blocks = ["layer4", "layer3", "layer2", "layer1"][:n_blocks]
        for b in blocks:
            for p in getattr(model, b).parameters():
                p.requires_grad = True
        names = blocks
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):  # torchvision ViT
        layers = list(model.encoder.layers)[-n_blocks:]
        for layer in layers:
            for p in layer.parameters():
                p.requires_grad = True
        for p in model.encoder.ln.parameters():
            p.requires_grad = True
        names = [f"encoder.layers[-{n_blocks}:]", "encoder.ln"]
    elif hasattr(model, "base_model_prefix"):  # Hugging Face transformer
        base = getattr(model, model.base_model_prefix)
        layers = base.transformer.layer if hasattr(base, "transformer") else base.encoder.layer
        for layer in list(layers)[-n_blocks:]:
            for p in layer.parameters():
                p.requires_grad = True
        names = [f"transformer.layer[-{n_blocks}:]"]
    return names


def param_groups(model: nn.Module, lr: float, backbone_lr_mult: float = 0.1, weight_decay: float = 0.0) -> list[dict]:
    """DISCRIMINATIVE LEARNING RATES: head gets lr, unfrozen backbone gets lr * mult.

    Pretrained weights are already good; a large update would destroy them
    ("catastrophic forgetting"). The randomly initialised head needs big steps.
    """
    head_ids = {id(p) for p in get_head(model).parameters()}
    head = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
    body = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head, "lr": lr, "weight_decay": weight_decay, "name": "head"}]
    if body:
        groups.append({"params": body, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay, "name": "backbone"})
    return groups


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
def build_text_model(model_name: str, num_labels: int, id2label: dict, label2id: dict,
                     freeze_encoder: bool = False, vocab_size: int | None = None) -> nn.Module:
    """DistilBERT (66M params, 6 Transformer layers) + a new classification head.

    DistilBERT is BERT compressed by knowledge distillation: 40% smaller,
    60% faster, ~97% of BERT's accuracy — a common production choice.

    Loading prints "Some weights ... were newly initialized: classifier..."
    That message is EXPECTED: it is literally our new head.
    """
    if smoke_mode():
        from transformers import DistilBertConfig, DistilBertForSequenceClassification

        cfg = DistilBertConfig(vocab_size=vocab_size or 3100, dim=32, n_layers=1, n_heads=2, hidden_dim=64,
                               max_position_embeddings=128, num_labels=num_labels,
                               id2label=id2label, label2id=label2id)
        model = DistilBertForSequenceClassification(cfg)
    else:
        from transformers import AutoModelForSequenceClassification

        kwargs = {}
        if get_device().type == "mps":
            # PyTorch's fused attention kernel (SDPA) cannot apply dropout on Apple GPUs during
            # training. The "eager" implementation computes the same softmax(QK^T/sqrt(d))V with
            # plain matmuls, so it supports dropout everywhere (slightly slower).
            kwargs["attn_implementation"] = "eager"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, id2label=id2label, label2id=label2id, **kwargs)
    if freeze_encoder:
        freeze(getattr(model, model.base_model_prefix))
    return model


# ---------------------------------------------------------------------------
# One forward function for both modalities
# ---------------------------------------------------------------------------
def forward_logits(model: nn.Module, batch: dict) -> torch.Tensor:
    """Return raw class scores (logits) z for a batch of images OR text.

    Logits are unnormalised; softmax turns them into probabilities
        p_k = exp(z_k) / sum_j exp(z_j)
    """
    if "pixel_values" in batch:
        return model(batch["pixel_values"])
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return out.logits
