"""Test configuration: tests run in SMOKE mode (tiny models, synthetic images,
no downloads) so they finish in seconds on any laptop or CI runner."""
import os
import sys
from pathlib import Path

os.environ["DLH_SMOKE"] = "1"
os.environ.setdefault("DLH_DEVICE", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
