"""Experiment tracking: TensorBoard by default, Weights & Biases optional.

WHY TRACK EXPERIMENTS
---------------------
After 30 runs nobody remembers which learning rate produced the 94% model.
A tracker records config + metrics for every run so results are comparable,
reproducible and shareable — standard practice in any ML team.

* TensorBoard  : free, local, no account; renders inside Colab/Jupyter.
* W&B          : hosted dashboards & team sharing; needs a free account + API key.
  Opt in with DLH_WANDB=1 and store the key as a Colab Secret named
  WANDB_API_KEY (never paste keys into a notebook you might share).
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path


def get_secret(name: str) -> str | None:
    """Read a secret from the environment, or from Colab Secrets (key icon)."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        from google.colab import userdata  # type: ignore

        return userdata.get(name)
    except Exception:
        return None


class ExperimentLogger:
    def __init__(self, run_name: str, root: str | Path = "runs", config: dict | None = None,
                 use_tensorboard: bool = True, use_wandb: bool | None = None):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / f"{run_name}-{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        (self.dir / "config.json").write_text(json.dumps(self.config, indent=2, default=str))
        self._csv = self.dir / "metrics.csv"
        self.tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(log_dir=str(self.dir))
            except Exception as exc:  # tensorboard not installed
                print(f"(TensorBoard unavailable: {exc}; logging to CSV only)")
        self.wandb = None
        use_wandb = os.environ.get("DLH_WANDB") == "1" if use_wandb is None else use_wandb
        if use_wandb:
            key = get_secret("WANDB_API_KEY")
            try:
                import wandb

                if key:
                    wandb.login(key=key, relogin=False)
                self.wandb = wandb.init(project="dl-handson-lecture", name=run_name, config=self.config,
                                        dir=str(self.dir), reinit=True)
            except Exception as exc:
                print(f"(W&B disabled: {exc}) — continuing with TensorBoard/CSV.")

    def log(self, metrics: dict, step: int) -> None:
        if self.tb:
            for k, v in metrics.items():
                self.tb.add_scalar(k, v, step)
        if self.wandb:
            self.wandb.log(metrics, step=step)
        new = not self._csv.exists()
        with open(self._csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step", *metrics.keys()])
            if new:
                w.writeheader()
            w.writerow({"step": step, **metrics})

    def close(self) -> None:
        if self.tb:
            self.tb.flush()
            self.tb.close()
        if self.wandb:
            self.wandb.finish()
