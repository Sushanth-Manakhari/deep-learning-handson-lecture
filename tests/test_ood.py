import numpy as np

from dlhandson.ood import entropy, pick_threshold, selective_curve, synthetic_ood_images


def test_entropy_bounds():
    assert abs(entropy(np.array([1.0, 0, 0]))) < 1e-6
    assert abs(entropy(np.array([1 / 3] * 3)) - np.log(3)) < 1e-6


def test_selective_curve_tradeoff():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([1, 1, 1], size=500)
    labels = probs.argmax(1).copy()
    labels[:100] = (labels[:100] + 1) % 3  # some mistakes
    curve = selective_curve(probs, labels)
    assert curve.coverage.is_monotonic_decreasing
    assert curve.coverage.iloc[0] == 1.0
    t = pick_threshold(curve, 0.9)
    assert 0.0 <= t <= 0.99


def test_synthetic_ood_images():
    items = synthetic_ood_images(32)
    assert len(items) == 3 and all(it["image"].size == (32, 32) for it in items)
