"""Small helpers that only make sense inside a notebook."""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def headless() -> bool:
    """True when the notebook is executed automatically (tests / verification).

    Interactive-only cells (public demo links, file pickers, TensorBoard UI)
    check this and skip themselves, so "Run all" never blocks waiting for a click.
    """
    return os.environ.get("DLH_HEADLESS", "0") == "1"


def upload_photos(dest: str | Path) -> list[Path]:
    """Let the user pick photos from their computer and copy them to `dest`.

    Colab: opens a file picker (photos synced/AirDropped from a phone work too).
    Local Jupyter: copy files into `dest` yourself, then re-run the next cell.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if headless():
        print("(headless run: upload skipped)")
        return []
    try:
        from google.colab import files  # type: ignore
    except ImportError:
        print(f"Running locally: copy your photos into  {dest.resolve()}  and run the next cell.")
        return sorted(dest.iterdir())
    uploaded = files.upload()
    saved = []
    for name, data in uploaded.items():
        out = dest / Path(name).name
        out.write_bytes(data)
        saved.append(out)
    print(f"saved {len(saved)} file(s) to {dest}")
    return saved


def show_tree(root: str | Path, max_depth: int = 2, skip=(".git", ".venv", "__pycache__", "runs", "beans",
                                                          "cache", ".ipynb_checkpoints", "reports", "slides")) -> None:
    """Print a compact directory tree (used in the 'notebook -> package' section)."""
    root = Path(root)

    def walk(p: Path, depth: int, prefix: str = ""):
        if depth > max_depth:
            return
        items = sorted([c for c in p.iterdir() if c.name not in skip and not c.name.startswith(".")],
                       key=lambda c: (c.is_file(), c.name))
        for i, c in enumerate(items):
            last = i == len(items) - 1
            print(f"{prefix}{'└── ' if last else '├── '}{c.name}{'/' if c.is_dir() else ''}")
            if c.is_dir():
                walk(c, depth + 1, prefix + ("    " if last else "│   "))

    print(f"{root.name}/")
    walk(root, 1)


def clear_folder(p: str | Path) -> None:
    shutil.rmtree(p, ignore_errors=True)
