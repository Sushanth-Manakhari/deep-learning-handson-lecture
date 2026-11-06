"""Ticket 2 data: Banking77 customer messages -> 77 intents.

REAL-WORLD CONTEXT
------------------
"I still haven't received my card", "Why was I charged twice?" — every bank,
insurer and telecom routes thousands of messages a day. An intent classifier
sends each one to the right queue or self-service flow; low-confidence ones
go to a human. Banking77 (PolyAI, CC BY 4.0) is a standard benchmark for it.

TEXT -> NUMBERS: TOKENISATION
-----------------------------
Networks consume numbers, so text is split into sub-word TOKENS and mapped to
integer ids from a fixed vocabulary (DistilBERT: WordPiece, ~30,522 tokens):

    "Why was I charged twice?"
    -> [CLS] why was i charged twice ? [SEP]
    -> [101, 2339, 2001, 1045, 5338, 3807, 1029, 102]

attention_mask marks real tokens (1) vs padding (0) so attention ignores pads.

DYNAMIC PADDING
---------------
A batch must be a rectangle, so short sequences are padded. Padding every
message to a global max (e.g. 512) wastes compute on zeros — attention costs
O(n^2) in sequence length n. We pad each batch only to ITS longest message.
"""
from __future__ import annotations

import json
import re
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .device import smoke_mode

SOURCE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{}"
FILES = ("train.csv", "test.csv", "categories.json")


def ensure_banking77(root: str | Path | None = None) -> Path:
    """Return the folder holding Banking77 CSVs (download only if missing)."""
    from .data_vision import repo_root

    root = Path(root) if root else repo_root() / "data" / "banking77"
    root.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if not (root / f).exists():
            print(f"📦 downloading {f} from the PolyAI GitHub repository")
            urllib.request.urlretrieve(SOURCE.format(f), root / f)
    return root


def load_banking77(root: str | Path | None = None, val_fraction: float = 0.1, seed: int = 42) -> dict:
    """Load train/validation/test DataFrames with integer labels.

    The official release has only train/test, so we carve a STRATIFIED
    validation split out of train (each intent keeps its proportion).
    Rule: tune on validation, report on test exactly once.
    """
    from sklearn.model_selection import train_test_split

    root = ensure_banking77(root)
    train_full = pd.read_csv(root / "train.csv")
    test = pd.read_csv(root / "test.csv")
    label_names = sorted(json.loads((root / "categories.json").read_text()))
    label2id = {n: i for i, n in enumerate(label_names)}
    for df in (train_full, test):
        df["label"] = df["category"].map(label2id).astype(int)
    train, val = train_test_split(train_full, test_size=val_fraction, random_state=seed,
                                  stratify=train_full["label"])
    return {
        "train": train.reset_index(drop=True),
        "validation": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
        "label_names": label_names,
        "label2id": label2id,
        "id2label": {i: n for n, i in label2id.items()},
    }


def subsample(df: pd.DataFrame, n: int | None, seed: int = 42) -> pd.DataFrame:
    """Random subset of n rows (keeps every intent when n is large enough)."""
    if n is None or n >= len(df):
        return df
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def _tiny_wordpiece_tokenizer(texts) -> object:
    """Offline tokenizer for smoke tests: a word-level vocab built from the data."""
    try:
        from transformers import BertTokenizerFast as Tok
    except ImportError:  # transformers v5 merged slow/fast tokenizers
        from transformers import BertTokenizer as Tok
    words = Counter(w for t in texts for w in re.findall(r"[a-z0-9']+|[^\sa-z0-9]", str(t).lower()))
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [w for w, _ in words.most_common(3000)]
    vocab_file = Path(tempfile.mkdtemp()) / "vocab.txt"
    vocab_file.write_text("\n".join(vocab))
    return Tok(vocab_file=str(vocab_file), do_lower_case=True)


def build_tokenizer(model_name: str, texts=None):
    """Load the tokenizer that matches the pretrained model (or a tiny offline one)."""
    if smoke_mode():
        return _tiny_wordpiece_tokenizer(texts if texts is not None else ["hello world"])
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def token_lengths(tokenizer, texts) -> np.ndarray:
    """Number of tokens per text (used to choose max_length)."""
    enc = tokenizer(list(texts), truncation=False, add_special_tokens=True)
    return np.array([len(ids) for ids in enc["input_ids"]])


# ---------------------------------------------------------------------------
# Dataset + dynamic-padding collate
# ---------------------------------------------------------------------------
class TextClassificationDataset(Dataset):
    """Pre-tokenises once (fast tokenizers do ~10k short texts in ~1 s)."""

    def __init__(self, texts, labels, tokenizer, max_length: int):
        enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        return {"input_ids": self.input_ids[i], "attention_mask": self.attention_mask[i], "labels": self.labels[i]}


def make_collate(pad_token_id: int):
    """Pad each batch only to its own longest sequence (dynamic padding)."""

    def collate(batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            mask[i, :L] = torch.tensor(b["attention_mask"])
        labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    return collate


def make_text_loaders(cfg, data: dict, tokenizer, device: torch.device) -> dict:
    """DataLoaders for the three splits, sub-sampled per the hardware profile."""
    train_df = subsample(data["train"], cfg.max_train_samples, cfg.seed)
    val_df = subsample(data["validation"], cfg.max_eval_samples, cfg.seed)
    test_df = subsample(data["test"], cfg.max_eval_samples, cfg.seed)
    collate = make_collate(tokenizer.pad_token_id)
    pin = device.type == "cuda"
    mk = lambda df: TextClassificationDataset(df["text"], df["label"], tokenizer, cfg.max_length)  # noqa: E731
    g = torch.Generator().manual_seed(cfg.seed)
    return {
        "train": DataLoader(mk(train_df), batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
                            pin_memory=pin, generator=g),
        "validation": DataLoader(mk(val_df), batch_size=cfg.batch_size * 2, collate_fn=collate, pin_memory=pin),
        "test": DataLoader(mk(test_df), batch_size=cfg.batch_size * 2, collate_fn=collate, pin_memory=pin),
        "frames": {"train": train_df, "validation": val_df, "test": test_df},
    }
