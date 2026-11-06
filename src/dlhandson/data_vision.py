"""Ticket 1 data: the Beans leaf-disease dataset.

REAL-WORLD CONTEXT
------------------
Beans (Makerere University AI Lab, Uganda; MIT licence) are smartphone photos
taken in farmers' fields: 1,034 train / 133 validation / 128 test images,
3 classes -> angular_leaf_spot, bean_rust, healthy. The same pattern powers
crop-scouting apps, drone crop monitoring, and insurance damage triage:
"photo in -> label + confidence out -> act or ask a human".

GETTING THE DATA FAST (the "pre-load datasets" rule)
----------------------------------------------------
University Wi-Fi + 60 laptops + a 180 MB download = a stalled lecture. So the
loader tries the cheapest source first and falls back automatically:

  1. data/beans/ already on disk                 -> 0 s
  2. GitHub Release asset beans_224.zip (~25 MB,  -> seconds; pre-resized,
     SHA-256 verified)                               integrity-checked
  3. Hugging Face Hub parquet files (~180 MB)    -> original source
  4. Synthetic "leaf" images                     -> smoke tests only (never in class)

Folder layout produced (torchvision ImageFolder convention):
  data/beans/{train,validation,test}/{angular_leaf_spot,bean_rust,healthy}/*.jpg
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import v2 as T

from . import CLASS_NAMES_BEANS
from .device import default_num_workers, smoke_mode

SPLITS = ("train", "validation", "test")
HF_REPO = "AI-Lab-Makerere/beans"
HF_FILES = {s: f"data/{s}-00000-of-00001.parquet" for s in SPLITS}

# ImageNet statistics. Pretrained models expect inputs standardised the same way
# they saw during pre-training:  x' = (x - mu) / sigma  (per RGB channel).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Locating / downloading the data
# ---------------------------------------------------------------------------
def repo_root() -> Path:
    """The repository root (two levels above this file: src/dlhandson/..)."""
    return Path(__file__).resolve().parents[2]


def _has_beans(root: Path) -> bool:
    return all((root / s).is_dir() and any((root / s).rglob("*.jpg")) for s in SPLITS)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _from_release(target: Path, manifest_path: Path) -> bool:
    """Download + verify + unzip the pre-resized pack from the GitHub Release."""
    if not manifest_path.exists():
        print("   (no data/beans_pack.json manifest yet -> skipping GitHub Release source)")
        return False
    manifest = json.loads(manifest_path.read_text())
    url, expected = manifest["url"], manifest.get("sha256")
    tmp = target.parent / "cache" / "beans_224.zip"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    print(f"   downloading {url}")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as exc:  # network blocked, release not published yet, ...
        print(f"   GitHub Release download failed: {exc}")
        return False
    # SECURITY: verify integrity before trusting downloaded bytes.
    if expected and _sha256(tmp) != expected:
        print("   SHA-256 mismatch -> refusing to use this file (corrupted or tampered).")
        return False
    with zipfile.ZipFile(tmp) as zf:
        for member in zf.namelist():  # SECURITY: block "zip-slip" path traversal
            if member.startswith("/") or ".." in Path(member).parts:
                raise ValueError(f"unsafe path in zip: {member}")
        zf.extractall(target.parent)
    return _has_beans(target)


def export_hf_split(parquet_path: Path, out_dir: Path, size: int = 224, quality: int = 90) -> int:
    """Decode one parquet split from the Hub and write resized JPEGs (ImageFolder)."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path).to_pydict()
    n = 0
    for i, (img, label) in enumerate(zip(table["image"], table["labels"])):
        cls = CLASS_NAMES_BEANS[int(label)]
        d = out_dir / cls
        d.mkdir(parents=True, exist_ok=True)
        pil = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        # The originals are 500x500. 224x224 is what the models consume anyway,
        # so resizing once here saves download size AND JPEG decode time per epoch.
        pil = pil.resize((size, size), Image.Resampling.BICUBIC)
        pil.save(d / f"{i:04d}.jpg", quality=quality, optimize=True)
        n += 1
    return n


def _from_hub(target: Path, size: int = 224) -> bool:
    """Download the original parquet files from the Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False
    try:
        for split in SPLITS:
            local = hf_hub_download(HF_REPO, HF_FILES[split], repo_type="dataset")
            n = export_hf_split(Path(local), target / split, size=size)
            print(f"   {split:<10} {n:>5} images")
    except Exception as exc:
        print(f"   Hugging Face download failed: {exc}")
        return False
    return _has_beans(target)


def make_synthetic_beans(target: Path, per_class: int = 12, size: int = 96, seed: int = 0) -> Path:
    """Draw fake 'leaves' so tests run with no internet. NOT real data.

    healthy -> clean green leaf; angular_leaf_spot -> brown angular patches;
    bean_rust -> small orange-brown pustules. Good enough to exercise code paths.
    """
    rng = np.random.default_rng(seed)
    for split, n in (("train", per_class), ("validation", max(2, per_class // 3)), ("test", max(2, per_class // 3))):
        for ci, cls in enumerate(CLASS_NAMES_BEANS):
            d = target / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for k in range(n):
                bg = tuple(int(v) for v in rng.integers(60, 120, 3))
                im = Image.new("RGB", (size, size), bg)
                dr = ImageDraw.Draw(im)
                g = int(rng.integers(110, 180))
                m = int(size * 0.12)
                dr.ellipse([m, m, size - m, size - m], fill=(30, g, 40))
                if cls != "healthy":
                    for _ in range(int(rng.integers(5, 14))):
                        x, y = rng.integers(m + 5, size - m - 5, 2)
                        r = int(rng.integers(2, 6))
                        if cls == "angular_leaf_spot":
                            dr.polygon([(x - r, y), (x, y - r), (x + r, y + 1), (x, y + r)], fill=(110, 80, 30))
                        else:
                            dr.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(200, 110, 30))
                im.filter(ImageFilter.GaussianBlur(0.6)).save(d / f"{split}_{ci}_{k:03d}.jpg", quality=92)
    return target


def ensure_beans(root: str | Path | None = None, verbose: bool = True) -> Path:
    """Return a folder with the Beans dataset, fetching it if needed (see module doc)."""
    root = Path(root) if root else repo_root() / "data" / "beans"
    if smoke_mode():
        synth = root.parent / "cache" / "beans_synthetic"
        if not _has_beans(synth):
            make_synthetic_beans(synth)
        if verbose:
            print("🧪 SMOKE MODE: using synthetic leaf images (tests only).")
        return synth
    if _has_beans(root):
        if verbose:
            print(f"✅ Beans dataset found at {root}")
        return root
    root.mkdir(parents=True, exist_ok=True)
    manifest = root.parent / "beans_pack.json"
    if verbose:
        print("📦 Beans not on disk yet. Source 1/2: GitHub Release (pre-resized, ~25 MB)")
    if _from_release(root, manifest):
        print("✅ Beans ready (GitHub Release).")
        return root
    if verbose:
        print("📦 Source 2/2: Hugging Face Hub (original parquet, ~180 MB)")
    shutil.rmtree(root, ignore_errors=True)
    if _from_hub(root):
        print("✅ Beans ready (Hugging Face Hub).")
        return root
    raise RuntimeError(
        "Could not download the Beans dataset. Check the internet connection "
        "(Kaggle: turn on Settings -> Internet) and re-run this cell."
    )


# ---------------------------------------------------------------------------
# Transforms (pre-processing + data augmentation)
# ---------------------------------------------------------------------------
def eval_transform(size: int = 224) -> T.Compose:
    """Deterministic pre-processing for validation/test/inference."""
    return T.Compose([
        T.ToImage(),
        T.Resize(size, antialias=True),
        T.CenterCrop(size),
        T.ToDtype(torch.float32, scale=True),     # uint8 [0,255] -> float [0,1]
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # x' = (x - mu) / sigma
    ])


def train_transform(size: int = 224, augment: bool = True) -> T.Compose:
    """Training pre-processing, optionally with data augmentation.

    DATA AUGMENTATION = free extra data. Each epoch the model sees a slightly
    different version of every photo, which teaches INVARIANCES we know are
    true for this problem:
      * leaves can be photographed at any angle      -> flips, rotation
      * farmers frame photos differently             -> random resized crop
      * sunlight / cloud / phone camera differences  -> brightness, contrast
    and it acts as a regulariser (harder to memorise the training set).

    Augmentations must PRESERVE THE LABEL. Rust is orange-brown and leaf spot is
    brown, so hue jitter is kept tiny (0.02): a big hue shift could turn a
    "rust" leaf into something that looks healthy -> label noise.
    """
    if not augment:
        return eval_transform(size)
    return T.Compose([
        T.ToImage(),
        T.RandomResizedCrop(size, scale=(0.6, 1.0), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Undo Normalize for display: x = x' * sigma + mu, clipped to [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)


# ---------------------------------------------------------------------------
# Datasets and loaders
# ---------------------------------------------------------------------------
class DictDataset(Dataset):
    """Wrap an (image, label) dataset so it yields {'pixel_values', 'labels'}.

    Using dict batches for BOTH images and text lets one training loop serve
    both tickets — the "transfer learning is modality-agnostic" lesson.
    """

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict:
        x, y = self.base[i]
        return {"pixel_values": x, "labels": y}


def image_folder(beans_dir: Path, split: str, transform=None) -> datasets.ImageFolder:
    ds = datasets.ImageFolder(str(Path(beans_dir) / split), transform=transform)
    assert ds.classes == CLASS_NAMES_BEANS, f"unexpected class folders: {ds.classes}"
    return ds


def class_counts(ds: datasets.ImageFolder) -> dict:
    counts = np.bincount([y for _, y in ds.samples], minlength=len(ds.classes))
    return dict(zip(ds.classes, counts.tolist()))


def _subsample(ds: Dataset, n: int | None, seed: int) -> Dataset:
    if n is None or n >= len(ds):
        return ds
    idx = np.random.default_rng(seed).permutation(len(ds))[:n]
    return Subset(ds, idx.tolist())


def make_vision_loaders(cfg, beans_dir: Path, device: torch.device, augment: bool | None = None) -> dict:
    """Build train/validation/test DataLoaders for the given config.

    batch_size trade-off (you will tune it in the challenge):
      bigger batch -> fewer, smoother gradient steps, better GPU utilisation,
                      but MORE GPU MEMORY (activations scale ~linearly with B)
      smaller batch -> noisier gradients (a mild regulariser), less memory
    """
    augment = cfg.augment if augment is None else augment
    nw = cfg.num_workers if cfg.num_workers is not None else default_num_workers(device)
    pin = device.type == "cuda"  # page-locked host memory -> faster host->GPU copies
    train_ds = DictDataset(_subsample(image_folder(beans_dir, "train", train_transform(cfg.image_size, augment)),
                                      cfg.max_train_samples, cfg.seed))
    val_ds = DictDataset(image_folder(beans_dir, "validation", eval_transform(cfg.image_size)))
    test_ds = DictDataset(image_folder(beans_dir, "test", eval_transform(cfg.image_size)))
    g = torch.Generator().manual_seed(cfg.seed)
    common = dict(num_workers=nw, pin_memory=pin)
    return {
        "train": DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, generator=g, **common),
        "validation": DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False, **common),
        "test": DataLoader(test_ds, batch_size=cfg.batch_size * 2, shuffle=False, **common),
        "class_names": CLASS_NAMES_BEANS,
    }
