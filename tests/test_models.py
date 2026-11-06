import torch

from dlhandson.models import build_text_model, build_vision_model, count_parameters, param_groups, unfreeze_top


def test_frozen_backbone_only_trains_head():
    m = build_vision_model("resnet18", 3, pretrained=False, freeze_backbone=True)
    stats = count_parameters(m)
    assert stats["trainable"] == 512 * 3 + 3  # just the new Linear(512 -> 3)
    assert m(torch.randn(2, 3, 64, 64)).shape == (2, 3)


def test_unfreeze_top_and_discriminative_lr():
    m = build_vision_model("resnet18", 3, pretrained=False, freeze_backbone=True)
    before = count_parameters(m)["trainable"]
    unfreeze_top(m, 1)
    assert count_parameters(m)["trainable"] > before
    groups = param_groups(m, lr=1e-3, backbone_lr_mult=0.1)
    assert {g["name"]: g["lr"] for g in groups} == {"head": 1e-3, "backbone": 1e-4}


def test_tiny_text_model_forward():
    m = build_text_model("unused", num_labels=77, id2label={i: str(i) for i in range(77)},
                         label2id={str(i): i for i in range(77)}, vocab_size=200)
    out = m(input_ids=torch.randint(0, 200, (2, 10)), attention_mask=torch.ones(2, 10, dtype=torch.long))
    assert out.logits.shape == (2, 77)
