"""Command-line fine-tuning for Ticket SUPPORT-202 (DistilBERT on Banking77).

Example:  .venv/bin/python scripts/train_text.py --epochs 3 --lr 5e-5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dlhandson.config import text_config_for  # noqa: E402
from dlhandson.data_text import build_tokenizer, load_banking77, make_text_loaders  # noqa: E402
from dlhandson.device import get_device, set_seed  # noqa: E402
from dlhandson.models import build_text_model, trainable_params  # noqa: E402
from dlhandson.tracking import ExperimentLogger  # noqa: E402
from dlhandson.train import build_optimizer, build_scheduler, evaluate, fit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--freeze-encoder", action="store_true")
    args = ap.parse_args()

    device = get_device()
    overrides = {k: v for k, v in {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size}.items()
                 if v is not None}
    cfg = text_config_for(device, freeze_encoder=args.freeze_encoder, **overrides)
    set_seed(cfg.seed)
    data = load_banking77()
    tok = build_tokenizer(cfg.model_name, texts=data["train"]["text"])
    loaders = make_text_loaders(cfg, data, tok, device)
    model = build_text_model(cfg.model_name, len(data["label_names"]), data["id2label"], data["label2id"],
                             freeze_encoder=cfg.freeze_encoder, vocab_size=len(tok))
    opt = build_optimizer("adamw", trainable_params(model), cfg.lr, cfg.weight_decay)
    steps = cfg.epochs * len(loaders["train"])
    sched = build_scheduler(opt, steps, int(cfg.warmup_ratio * steps), kind="linear")
    logger = ExperimentLogger("cli-text-distilbert", root=ROOT / "runs", config=cfg.to_dict())
    hist = fit(model, loaders["train"], loaders["validation"], epochs=cfg.epochs, optimizer=opt, scheduler=sched,
               device=device, amp=cfg.amp, grad_clip=cfg.grad_clip, logger=logger,
               ckpt_path=logger.dir / "best.safetensors")
    logger.close()
    print(f"best val acc {hist.best_val_acc:.3f} | test acc {evaluate(model, loaders['test'], device)['acc']:.3f}")


if __name__ == "__main__":
    main()
