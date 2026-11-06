import torch

from dlhandson.config import text_config_for, vision_config_for
from dlhandson.device import get_device, set_seed


def test_device_override_cpu():
    assert get_device().type == "cpu"  # DLH_DEVICE=cpu in conftest


def test_seed_makes_runs_repeatable():
    set_seed(7)
    a = torch.randn(3)
    set_seed(7)
    b = torch.randn(3)
    assert torch.equal(a, b)


def test_smoke_profiles_are_tiny():
    v = vision_config_for(torch.device("cpu"))
    t = text_config_for(torch.device("cpu"))
    assert v.profile == "smoke" and not v.pretrained and v.image_size <= 64
    assert t.profile == "smoke" and t.max_train_samples <= 64


def test_unknown_override_is_rejected():
    import pytest

    with pytest.raises(AttributeError):
        vision_config_for(torch.device("cpu"), learning_rate=0.1)  # field is called `lr`
