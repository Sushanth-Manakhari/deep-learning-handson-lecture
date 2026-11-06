# Banking77 (customer-message intent dataset)

| | |
|---|---|
| Source | PolyAI, `task-specific-datasets` repository on GitHub (`banking_data/`) |
| Paper | Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders*, NLP4ConvAI @ ACL 2020 |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Size | 10,003 training and 3,080 test messages, 77 intent labels |
| Files | `train.csv`, `test.csv` (columns `text`, `category`), `categories.json` (the 77 labels) |

Committed to this repository (about 1 MB) so the lecture never waits on a download.
If the files are missing, `dlhandson.data_text.ensure_banking77()` re-downloads them
from the original GitHub source.

Real-world use: routing customer messages to the right team or self-service flow
(banking, insurance, telecom, retail). Low-confidence predictions go to a human agent.
