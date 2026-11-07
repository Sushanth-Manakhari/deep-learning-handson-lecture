# Architecture diagrams

`architecture.drawio` holds all nine diagrams, one per page (tab). Open it with
**https://app.diagrams.net → File → Open from → Device**, the draw.io desktop app, or the
VS Code extension *Draw.io Integration* (hediet.vscode-drawio). Each page is also exported as a PNG.

| # | Page | PNG | Shows |
|---|---|---|---|
| 1 | High-level design | `01_high_level_design.png` | people, runtimes, course code, what is downloaded, outputs |
| 2 | Data flow | `02_data_flow.png` | Beans and Banking77 from source to batches, the shared training loop, the decision |
| 3 | Sequence | `03_sequence.png` | one student session from opening the link to the phone demo (17 steps) |
| 4 | Error handling | `04_error_handling.png` | 8 classroom failures: detection → automatic handling → what the student sees |
| 5 | Model architecture | `05_model_architecture.png` | ResNet-50 and DistilBERT: frozen vs fine-tuned vs new head, thresholds |
| 6 | Security & responsible AI | `06_security.png` | 9 risks → controls → where they live in the code |
| 7 | Environment decision | `07_environment_decision.png` | Colab → Kaggle → local GPU → CPU FAST mode |
| 8 | Lecture flow | `08_lecture_flow.png` | the 4:00–5:30 PM timeline (80 min hands-on + 10 min Q&A) mapped to notebook sections and concepts |
| 9 | Git & release flow | `09_git_release_flow.png` | private work → public release branch → PR → main, the data-v1 release, safety guards |

**Colour key (same on every page):** purple = people · yellow = runtime / decision · blue = code / frozen
weights · orange = external source / fine-tuned layers · green = outputs, controls, new heads ·
red = risks / human review · grey dashed boxes = grouping · dashed arrows = fallback, response or loop.

## Regenerating
The diagrams are generated from code so they stay consistent:
```bash
.venv/bin/python tools/diagrams/build_diagrams.py
```
The script runs layout checks first (text fits its box, no overlapping boxes, no arrow through a box,
legend or text, no two arrows on top of each other, no colliding labels) and refuses to write files if
any check fails. Edits made by hand in draw.io are fine too — but regenerating overwrites them.
