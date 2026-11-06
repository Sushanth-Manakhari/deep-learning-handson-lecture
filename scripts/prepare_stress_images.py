"""Prepare the instructor's phone photos for the public repository.

Phone photos carry EXIF metadata — often the GPS location where they were taken,
the phone model and the time. Before anything is committed to a public repo:
  * read every image in data/stress_test/_incoming/   (git-ignored)
  * apply the EXIF rotation, then DROP all metadata
  * resize to at most 800 px and save as JPEG in data/stress_test/instructor_photos/

iPhone HEIC files: converted with macOS `sips` (built in), or pillow-heif if installed.
Usage:  .venv/bin/python scripts/prepare_stress_images.py
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "stress_test" / "_incoming"
DST = ROOT / "data" / "stress_test" / "instructor_photos"
EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}


def open_any(path: Path) -> Image.Image:
    if path.suffix.lower() in {".heic", ".heif"}:
        try:
            import pillow_heif  # type: ignore

            pillow_heif.register_heif_opener()
        except ImportError:
            if shutil.which("sips"):  # macOS built-in converter
                tmp = Path(tempfile.mkdtemp()) / (path.stem + ".jpg")
                subprocess.run(["sips", "-s", "format", "jpeg", str(path), "--out", str(tmp)],
                               check=True, capture_output=True)
                return Image.open(tmp)
            raise RuntimeError("HEIC needs macOS 'sips' or `pip install pillow-heif` (inside .venv)")
    return Image.open(path)


def main() -> None:
    SRC.mkdir(parents=True, exist_ok=True)
    DST.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(SRC.iterdir()) if p.suffix.lower() in EXTS]
    if not files:
        print(f"No photos in {SRC.relative_to(ROOT)} — nothing to do.")
        return
    for p in files:
        im = open_any(p)
        had_exif = bool(im.getexif())
        im = ImageOps.exif_transpose(im).convert("RGB")   # keep the correct orientation…
        im.thumbnail((800, 800))
        clean = Image.new("RGB", im.size)
        clean.paste(im)                                   # …then copy pixels only (no metadata)
        out = DST / (p.stem.lower().replace(" ", "_") + ".jpg")
        clean.save(out, quality=88, optimize=True)
        print(f"{p.name:<30} -> {out.relative_to(ROOT)}  {clean.size}  "
              f"{'metadata removed' if had_exif else 'no metadata found'}")
    print("\nCheck the photos, then commit data/stress_test/instructor_photos/. "
          "Originals in _incoming/ are git-ignored (you may delete them).")


if __name__ == "__main__":
    main()
