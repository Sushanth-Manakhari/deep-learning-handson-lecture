"""Command-line training for Ticket LEAF-101 — the notebook logic as a reusable tool.

Example:  .venv/bin/python scripts/train_vision.py --epochs 5 --lr 1e-3 --unfreeze 1
Writes runs/<name>-<time>/ (config, metrics, TensorBoard) and a safetensors checkpoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dlhandson.config import vision_config_for  # noqa: E402
from dlhandson.data_vision import ensure_beans, make_vision_loaders  # noqa: E402
from dlhandson.device import get_device, set_seed  # noqa: E402
from dlhandson.models import build_vision_model, param_groups, trainable_params, unfreeze_top  # noqa: E402
from dlhandson.tracking import ExperimentLogger  # noqa: E402
from dlhandson.train import build_optimizer, evaluate, fit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="resnet18 | resnet50 | vit_b_16 (default: by hardware)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--optimizer", default=None, choices=["adamw", "adam", "sgd"])
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--unfreeze", type=int, default=0, help="number of top backbone blocks to fine-tune")
    args = ap.parse_args()

    device = get_device()
    overrides = {k: v for k, v in {"model_name": args.model, "epochs": args.epochs, "lr": args.lr,
                                   "batch_size": args.batch_size, "optimizer": args.optimizer}.items() if v is not None}
    if args.no_augment:
        overrides["augment"] = False
    cfg = vision_config_for(device, **overrides)
    set_seed(cfg.seed)
    loaders = make_vision_loaders(cfg, ensure_beans(), device)
    model = build_vision_model(cfg.model_name, 3, cfg.pretrained, freeze_backbone=True)
    if args.unfreeze:
        unfreeze_top(model, args.unfreeze)
        opt = build_optimizer(cfg.optimizer, param_groups(model, cfg.lr, 0.1, cfg.weight_decay), cfg.lr)
    else:
        opt = build_optimizer(cfg.optimizer, trainable_params(model), cfg.lr, cfg.weight_decay)
    logger = ExperimentLogger(f"cli-vision-{cfg.model_name}", root=ROOT / "runs", config=cfg.to_dict())
    ckpt = logger.dir / "best.safetensors"
    hist = fit(model, loaders["train"], loaders["validation"], epochs=cfg.epochs, optimizer=opt, device=device,
               amp=cfg.amp, logger=logger, ckpt_path=ckpt)
    logger.close()
    test = evaluate(model, loaders["test"], device)
    print(f"best val acc {hist.best_val_acc:.3f} | test acc (last epoch) {test['acc']:.3f} | checkpoint {ckpt}")


if __name__ == "__main__":
    main()
