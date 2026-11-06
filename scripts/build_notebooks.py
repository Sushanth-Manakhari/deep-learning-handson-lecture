"""Build the .ipynb notebooks from their readable sources in notebooks/src/*.py.

Why sources in .py? Notebook JSON is hard to review in a pull request (every
output and execution count shows up as a diff). The percent-format .py files
("# %%" cells, via jupytext) diff cleanly; the .ipynb files are generated.

Usage:  python scripts/build_notebooks.py
"""
from __future__ import annotations

from pathlib import Path

import jupytext
import nbformat

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "notebooks" / "src", ROOT / "notebooks"
COLAB = "https://colab.research.google.com/github/Sushanth-Manakhari/deep-learning-handson-lecture/blob/main/notebooks/{}"
BADGE = "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({})\n\n"


def build(src: Path) -> Path:
    nb = jupytext.read(src)
    # Colab reads these keys: open the notebook on a T4 GPU runtime by default.
    nb.metadata = {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    out = OUT / (src.stem + ".ipynb")
    first = nb.cells[0]
    if first.cell_type == "markdown" and "colab-badge" not in first.source:
        first.source = BADGE.format(COLAB.format(out.name)) + first.source
    for cell in nb.cells:  # clean, output-free student copies
        cell.metadata = {}
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    nbformat.validate(nb)
    nbformat.write(nb, out)
    return out


if __name__ == "__main__":
    for src in sorted(SRC.glob("*.py")):
        print("built", build(src).relative_to(ROOT))
