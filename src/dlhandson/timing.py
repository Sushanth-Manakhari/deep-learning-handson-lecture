"""SectionTimer — measure how long each lecture section really takes.

The instructor's Colab dry run prints this table at the end of the notebook;
those numbers calibrate the slide timings and the instructor run-sheet.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class SectionTimer:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.marks: list[tuple[str, float]] = []
        if self.path and self.path.exists():  # survive a kernel restart
            try:
                self.marks = [tuple(m) for m in json.loads(self.path.read_text())]
            except Exception:
                self.marks = []

    def start(self, name: str) -> None:
        """Mark the start of a section (the previous section ends here)."""
        self.marks.append((name, time.time()))
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.marks))
        print(f"⏱️  {name}")

    def summary(self) -> list[dict]:
        rows = []
        for i, (name, t0) in enumerate(self.marks):
            t1 = self.marks[i + 1][1] if i + 1 < len(self.marks) else time.time()
            rows.append({"section": name, "minutes": round((t1 - t0) / 60, 1)})
        return rows

    def print_summary(self) -> None:
        rows = self.summary()
        width = max((len(r["section"]) for r in rows), default=10)
        print(f"{'Section'.ljust(width)} | minutes")
        print("-" * (width + 10))
        for r in rows:
            print(f"{r['section'].ljust(width)} | {r['minutes']:>6}")
        total = sum(r["minutes"] for r in rows)
        print("-" * (width + 10))
        print(f"{'TOTAL'.ljust(width)} | {total:>6.1f}")
