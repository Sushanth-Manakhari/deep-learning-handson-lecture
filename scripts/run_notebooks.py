"""Execute notebooks top to bottom without a browser (the "Don't Panic" guarantee).

Saves executed copies (with outputs) to reports/executed/ and prints a pass/fail
line per notebook. Used by scripts/verify.py and by CI-style smoke tests:

    DLH_SMOKE=1 DLH_HEADLESS=1 python scripts/run_notebooks.py          # seconds, tiny models
    DLH_HEADLESS=1 python scripts/run_notebooks.py notebooks/01_*.ipynb  # real run (GPU/MPS)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "executed"


def run(path: Path, timeout: int = 3600) -> dict:
    nb = nbformat.read(path, as_version=4)
    os.environ.setdefault("DLH_HEADLESS", "1")
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}})
    t0 = time.time()
    status, error = "passed", ""
    try:
        client.execute()
    except CellExecutionError as exc:
        status, error = "FAILED", str(exc)[-3000:]
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / path.name
    nbformat.write(nb, out)
    return {"notebook": path.name, "status": status, "minutes": round((time.time() - t0) / 60, 1),
            "executed_copy": str(out.relative_to(ROOT)), "error": error}


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "notebooks").glob("0*.ipynb"))
    results = []
    for p in paths:
        r = run(p.resolve())
        results.append(r)
        print(f"{'✅' if r['status'] == 'passed' else '❌'} {r['notebook']:<34} {r['status']} in {r['minutes']} min")
        if r["error"]:
            print(r["error"])
    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    sys.exit(0 if all(r["status"] == "passed" for r in results) else 1)
