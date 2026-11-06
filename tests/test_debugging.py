"""The 'overfit one batch' rule as an automated test — the industry pattern.

If a refactor breaks the training loop (wrong loss, optimizer not updating,
labels shuffled), this test fails in CI before anyone wastes GPU hours.
"""
import math

import torch

from dlhandson.debug_tools import (exploding_experiment, optimizer_covers_model, overfit_one_batch,
                                   vanishing_gradient_experiment)
from dlhandson.models import build_vision_model, trainable_params
from dlhandson.train import NonFiniteLossError, train_one_epoch


def _batch(n=8, size=32):
    torch.manual_seed(0)
    return {"pixel_values": torch.randn(n, 3, size, size), "labels": torch.arange(n) % 3}


def test_overfit_one_batch_passes_for_correct_code():
    m = build_vision_model("resnet18", 3, pretrained=False, freeze_backbone=False)
    result = overfit_one_batch(m, _batch(), torch.device("cpu"), lr=1e-3, steps=40, verbose=False)
    assert result["passed"], result["losses"][-5:]


def test_stale_optimizer_bug_is_detected():
    m = build_vision_model("resnet18", 3, pretrained=False, freeze_backbone=True)
    opt = torch.optim.AdamW(trainable_params(m), lr=1e-3)  # built for the OLD head
    m.fc = torch.nn.Linear(512, 3)                           # head replaced afterwards -> bug
    assert not optimizer_covers_model(opt, m, verbose=False)
    result = overfit_one_batch(m, _batch(), torch.device("cpu"), optimizer=opt, steps=20, verbose=False)
    assert not result["passed"]


def test_vanishing_gradients_sigmoid_vs_residual():
    norms = vanishing_gradient_experiment(depth=20)
    assert norms["sigmoid"][0] < 1e-3 * norms["sigmoid"][-1]       # early layers starve
    assert norms["relu + residual"][0] > 1e-2 * norms["relu + residual"][-1]  # residual keeps flow


def test_large_lr_explodes_small_lr_does_not():
    res = exploding_experiment(lrs=(0.001, 0.5), steps=40)
    assert all(math.isfinite(v) for v in res[0.001])
    assert any(not math.isfinite(v) for v in res[0.5]) or res[0.5][-1] > 1e6


def test_nan_guard_raises():
    m = torch.nn.Linear(4, 3)

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = m

        def forward(self, x):
            return self.fc(x) * float("nan")

    loader = [{"pixel_values": torch.randn(4, 4), "labels": torch.tensor([0, 1, 2, 0])}]
    w = Wrap()
    try:
        train_one_epoch(w, loader, torch.optim.SGD(w.parameters(), lr=0.1), torch.device("cpu"))
        raise AssertionError("expected NonFiniteLossError")
    except NonFiniteLossError:
        pass
