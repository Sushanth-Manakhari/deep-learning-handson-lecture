"""Build the pre-resized Beans pack that the classroom downloads (instructor, once).

1. Download the original Beans parquet files from the Hugging Face Hub
   (https://huggingface.co/datasets/AI-Lab-Makerere/beans, MIT licence, ~180 MB).
2. Decode and resize every photo to 224x224 JPEG -> data/beans/{split}/{class}/
3. Zip it -> dist/beans_224.zip (~25 MB) and compute its SHA-256.
4. Write data/beans_pack.json (committed). The notebooks read it to download the
   pack from the GitHub Release and VERIFY the checksum before using it.

Then publish dist/beans_224.zip as a GitHub Release asset with tag `data-v1`
on the repository's Releases page. Usage:  .venv/bin/python scripts/build_beans_pack.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dlhandson.data_vision import HF_FILES, HF_REPO, SPLITS, export_hf_split  # noqa: E402

TAG = "data-v1"
URL = f"https://github.com/Sushanth-Manakhari/deep-learning-handson-lecture/releases/download/{TAG}/beans_224.zip"


def main() -> None:
    from huggingface_hub import hf_hub_download

    manifest_path, zpath_existing = ROOT / "data" / "beans_pack.json", ROOT / "dist" / "beans_224.zip"
    if manifest_path.exists() and zpath_existing.exists() and "--force" not in sys.argv:
        print(f"Beans pack already built (sha256 {json.loads(manifest_path.read_text())['sha256'][:12]}…) — "
              "skipping. Use --force to rebuild (this changes the checksum!).")
        return
    beans = ROOT / "data" / "beans"
    shutil.rmtree(beans, ignore_errors=True)
    counts = {}
    for split in SPLITS:
        print(f"downloading {HF_REPO}/{HF_FILES[split]}")
        local = hf_hub_download(HF_REPO, HF_FILES[split], repo_type="dataset")
        counts[split] = export_hf_split(Path(local), beans / split, size=224)
        print(f"  {split:<10} {counts[split]:>5} images -> {beans / split}")

    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    zpath = dist / "beans_224.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:  # JPEGs are already compressed
        for f in sorted(beans.rglob("*.jpg")):
            zf.write(f, f.relative_to(beans.parent))              # paths start with "beans/"
    sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    manifest = {"version": TAG, "url": URL, "sha256": sha, "bytes": zpath.stat().st_size, "counts": counts,
                "source": f"https://huggingface.co/datasets/{HF_REPO}", "licence": "MIT",
                "note": "Images resized to 224x224 JPEG (quality 90) by scripts/build_beans_pack.py"}
    (ROOT / "data" / "beans_pack.json").write_text(json.dumps(manifest, indent=2))
    print(f"\npack: {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)\nsha256: {sha}\nmanifest: data/beans_pack.json")
    print(f"NEXT: publish {zpath.name} as a GitHub Release asset, tag '{TAG}'")


if __name__ == "__main__":
    main()
