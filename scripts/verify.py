"""Instructor verification — real data, real pretrained models, full notebooks.

Runs on the instructor's machine (called by `bash scripts/setup_mac.sh --verify`)
and writes reports/verify_report.md: environment, unit tests, data checks,
notebook execution (top to bottom, headless), key metrics and section timings.

Usage:  .venv/bin/python scripts/verify.py [--all] [--skip-notebooks]
  --all             also run 02_challenge_solutions (a 9-run sweep, +5-10 min)
  --skip-notebooks  environment + tests + data only
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
STEPS: list[dict] = []


def record(name: str, fn):
    t0 = time.time()
    print(f"\n==> {name}")
    try:
        detail = fn() or ""
        STEPS.append({"step": name, "ok": True, "seconds": round(time.time() - t0), "detail": detail})
        print(f"✅ {name}: {detail}")
    except Exception as exc:  # keep going; the report shows what failed
        STEPS.append({"step": name, "ok": False, "seconds": round(time.time() - t0),
                      "detail": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-2500:]})
        print(f"❌ {name}: {type(exc).__name__}: {exc}")


def env_check() -> str:
    import torch
    import transformers
    import torchvision
    from dlhandson.device import describe_device, get_device

    in_venv = Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()
    info = describe_device(get_device())
    return (f"{platform.platform()} | python {platform.python_version()} | torch {torch.__version__} | "
            f"torchvision {torchvision.__version__} | transformers {transformers.__version__} | "
            f"device {info['device']} ({info['gpu_name']}) | project .venv: {'yes' if in_venv else 'NO'}")


def unit_tests() -> str:
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-o", "addopts=", "-q"], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "DLH_SMOKE": "1", "DLH_DEVICE": "cpu"})
    tail = (r.stdout or r.stderr).strip().splitlines()[-1]
    if r.returncode != 0:
        raise RuntimeError(f"pytest failed: {tail}\n{r.stdout[-1500:]}")
    return tail


def data_check() -> str:
    from dlhandson.data_text import load_banking77
    from dlhandson.data_vision import class_counts, ensure_beans, image_folder

    beans = ensure_beans(verbose=False)
    counts = {s: sum(class_counts(image_folder(beans, s)).values()) for s in ("train", "validation", "test")}
    manifest = ROOT / "data" / "beans_pack.json"
    sha = json.loads(manifest.read_text())["sha256"][:12] + "…" if manifest.exists() else "no manifest yet"
    text = load_banking77()
    return (f"Beans {counts} (pack sha256 {sha}) | Banking77 train/val/test "
            f"{len(text['train'])}/{len(text['validation'])}/{len(text['test'])}")


def run_nb(name: str):
    from run_notebooks import run

    def _():
        r = run(ROOT / "notebooks" / name)
        if r["status"] != "passed":
            raise RuntimeError(f"{name} failed after {r['minutes']} min:\n{r['error'][-2000:]}")
        return f"passed in {r['minutes']} min → {r['executed_copy']}"
    return _


def write_report() -> Path:
    results_path = ROOT / "runs" / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    lines = ["# Verification report", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} on {platform.node()}", "",
             "| Step | Result | Time | Detail |", "|---|---|---|---|"]
    for s in STEPS:
        detail = s["detail"].replace("|", "/").replace("\n", " ")[:400]
        lines.append(f"| {s['step']} | {'✅' if s['ok'] else '❌'} | {s['seconds']}s | {detail} |")
    if results:
        lines += ["", "## Key metrics (from the lecture notebook run)", "", "| Metric | Value |", "|---|---|"]
        for k, v in results.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
            elif isinstance(v, (int, str)) or v is None:
                lines.append(f"| {k} | {v} |")
        if results.get("timings_minutes"):
            lines += ["", "## Minutes per notebook section (compute time on this machine)", "",
                      "| Section | Minutes |", "|---|---|"]
            lines += [f"| {r['section']} | {r['minutes']} |" for r in results["timings_minutes"]]
    failures = [s for s in STEPS if not s["ok"]]
    if failures:
        lines += ["", "## Failure details"]
        for s in failures:
            lines += ["", f"### {s['step']}", "```", s.get("trace", s["detail"]), "```"]
    out = ROOT / "reports" / "verify_report.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--skip-notebooks", action="store_true")
    args = ap.parse_args()
    os.environ["DLH_HEADLESS"] = "1"
    record("Environment", env_check)
    record("Unit tests (pytest)", unit_tests)
    record("Datasets", data_check)
    if not args.skip_notebooks:
        shutil.rmtree(ROOT / "runs", ignore_errors=True)  # fresh run: real timings, no cached checkpoints
        record("Notebook 00_preflight_check", run_nb("00_preflight_check.ipynb"))
        record("Notebook 01_lecture_handson", run_nb("01_lecture_handson.ipynb"))
        if args.all:
            record("Notebook 02_challenge_solutions", run_nb("02_challenge_solutions.ipynb"))
    out = write_report()
    ok = all(s["ok"] for s in STEPS)
    print(f"\n{'✅ ALL CHECKS PASSED' if ok else '❌ SOME CHECKS FAILED'} — report: {out.relative_to(ROOT)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
