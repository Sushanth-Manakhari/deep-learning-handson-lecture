"""Turn an executed Colab run into the public INSTRUCTOR copy of the lecture notebook.

    python scripts/make_instructor_copy.py reports/colab_dryrun/01_lecture_handson.ipynb

* takes cell sources from the current notebooks/01_lecture_handson.ipynb and outputs from the run
* PRIVACY: drops all cell metadata (Colab stores the runner's Google display name / user id there)
  and notebook metadata other than the standard Colab keys
* removes interactive outputs that cannot work in a static copy (TensorBoard, the public Gradio link,
  widgets, scripts) and blanks any *.gradio.live URL
* writes notebooks/01_lecture_handson_INSTRUCTOR.ipynb
"""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDENT = ROOT / "notebooks" / "01_lecture_handson.ipynb"
OUT = ROOT / "notebooks" / "01_lecture_handson_INSTRUCTOR.ipynb"
NOTE = ("> 🧑‍🏫 **Instructor copy.** Executed top to bottom on a free Google Colab **Tesla T4** on Nov 5, 2026. "
        "All outputs (accuracies, timings, plots) come from that run. Interactive outputs (TensorBoard, the "
        "public phone-demo link) are removed — run those cells yourself to see them live.\n\n")
DROP_MIME = {"application/vnd.jupyter.widget-view+json", "application/javascript"}
LINK = re.compile(r"https://[a-z0-9]+\.gradio\.live\S*")
# Printed labels renamed in the code since the dry run (old text -> current text)
RENAMED = {"📋 LEADERBOARD ENTRY": "📋 YOUR RESULT",
           "1. Framing": "1. Why deep learning",
           "2. PyTorch mental model": "2. How a model learns",
           "3. Vision transfer learning": "3. Project 1: leaf disease",
           "4. NLP fine-tuning": "4. Project 2: customer messages",
           "5. Pitfalls & MLOps": "5. When training goes wrong",
}


def realign_table(text: str) -> str:
    """Re-pad "name | value" rows after renaming, so the timing table stays aligned."""
    lines = text.splitlines(keepends=True)
    rows = [i for i, l in enumerate(lines) if " | " in l and l.count("|") == 1]
    if len(rows) < 2:
        return text
    width = max(len(lines[i].split("|")[0].rstrip()) for i in rows)
    for i in rows:
        left, right = lines[i].split("|")
        lines[i] = f"{left.rstrip().ljust(width)} |{right}"
    for i, l in enumerate(lines):                      # separator lines follow the new width
        if l.strip() and set(l.strip()) == {"-"}:
            lines[i] = "-" * (width + 10) + ("\n" if l.endswith("\n") else "")
    return "".join(lines)
# Files that are never published (listed in the private repository's .public-exclude). The dry run was
# executed on that branch, so its folder listing (section 5.6) names them; those lines are dropped.
PRIVATE_LIST = ROOT / ".public-exclude"
PRIVATE_NAMES = sorted({Path(l.split("#")[0].strip()).name for l in PRIVATE_LIST.read_text().splitlines()
                        if l.split("#")[0].strip()}) if PRIVATE_LIST.exists() else []


def hide_private_names(text: str) -> str:
    """Drop folder-tree lines ("├── NAME", "└── NAME") that name an unpublished file."""
    keep = [line for line in text.splitlines(keepends=True)
            if not any(line.rstrip().endswith("── " + name) for name in PRIVATE_NAMES)]
    return "".join(keep)


def clean_outputs(cell: dict) -> list:
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if "run_line_magic(\"tensorboard\"" in src or "launch(demo" in src:
        return [{"output_type": "stream", "name": "stdout",
                 "text": ["(interactive output removed in the instructor copy — run this cell to see it live)\n"]}]
    out = []
    for o in cell.get("outputs", []):
        o = copy.deepcopy(o)
        o.pop("metadata", None) if o.get("output_type") == "stream" else None
        if "data" in o:
            data = {k: v for k, v in o["data"].items() if k not in DROP_MIME}
            html = data.get("text/html")
            if html and "<script" in ("".join(html) if isinstance(html, list) else html):
                data.pop("text/html")
            if not data:
                continue
            o["data"] = data
            o["metadata"] = {}
        if "text" in o:
            t = "".join(o["text"]) if isinstance(o["text"], list) else o["text"]
            for old, new in RENAMED.items():
                t = t.replace(old, new)
            o["text"] = realign_table(hide_private_names(LINK.sub("<expired demo link>", t)))
        out.append(o)
    return out


def main(executed_path: str) -> None:
    run = json.loads(Path(executed_path).read_text())
    student = json.loads(STUDENT.read_text())
    same_shape = [c["cell_type"] for c in run["cells"]] == [c["cell_type"] for c in student["cells"]]
    base = student if same_shape else run
    nb = {"nbformat": 4, "nbformat_minor": base.get("nbformat_minor", 5), "metadata": student["metadata"], "cells": []}
    for i, cell in enumerate(base["cells"]):
        c = {"cell_type": cell["cell_type"], "id": cell.get("id") or f"cell-{i}", "metadata": {}, "source": cell["source"]}
        if cell["cell_type"] == "code":
            c["execution_count"] = run["cells"][i].get("execution_count")
            c["outputs"] = clean_outputs({**run["cells"][i], "source": cell["source"]})
        nb["cells"].append(c)
    first = nb["cells"][0]
    src = "".join(first["source"]) if isinstance(first["source"], list) else first["source"]
    first["source"] = NOTE + src.replace("01_lecture_handson.ipynb", OUT.name)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"{'aligned to current sources' if same_shape else 'run sources kept (cell layout differs)'} → "
          f"{OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "reports" / "colab_dryrun" / "01_lecture_handson.ipynb"))
