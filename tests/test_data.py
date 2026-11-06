import torch

from dlhandson.config import text_config_for, vision_config_for
from dlhandson.data_text import build_tokenizer, load_banking77, make_collate, make_text_loaders
from dlhandson.data_vision import IMAGENET_MEAN, ensure_beans, make_vision_loaders


def test_beans_loaders_shapes():
    dev = torch.device("cpu")
    cfg = vision_config_for(dev)
    loaders = make_vision_loaders(cfg, ensure_beans(verbose=False), dev)
    batch = next(iter(loaders["train"]))
    assert batch["pixel_values"].shape[1:] == (3, cfg.image_size, cfg.image_size)
    assert batch["labels"].max() < 3
    # normalisation: values are roughly zero-centred, not in [0, 255]
    assert batch["pixel_values"].abs().max() < 10


def test_banking77_has_77_intents_and_stratified_val():
    data = load_banking77()
    assert len(data["label_names"]) == 77
    assert len(data["train"]) + len(data["validation"]) == 10003
    assert data["validation"]["label"].nunique() == 77  # every intent present in validation


def test_dynamic_padding_pads_to_longest_in_batch():
    collate = make_collate(pad_token_id=0)
    batch = collate([{"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": 1},
                     {"input_ids": [5], "attention_mask": [1], "labels": 0}])
    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_text_loaders_batch_keys():
    dev = torch.device("cpu")
    cfg = text_config_for(dev)
    data = load_banking77()
    tok = build_tokenizer(cfg.model_name, texts=data["train"]["text"])
    batch = next(iter(make_text_loaders(cfg, data, tok, dev)["train"]))
    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape[1] <= cfg.max_length
