import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from dataset_utils import CFNDDataset, transform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EANN_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "CFND_dataset"
VOCAB_PATH = PROJECT_ROOT / "vocab.json"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vocab():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def make_loader(split, vocab, batch_size=32, shuffle=False, missing_rate=0.0):
    dataset = CFNDDataset(
        csv_file=str(DATASET_DIR / f"{split}_data.csv"),
        root_dir=str(DATASET_DIR),
        vocab=vocab,
        transform=transform,
        missing_rate=missing_rate,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def unpack_batch(batch):
    texts, images, labels, event_labels = batch[:4]
    flags = batch[4:] if len(batch) > 4 else ()
    return texts, images, labels, event_labels, flags


def detector_logits(model, texts, images, alpha=0.0):
    outputs = model(texts, images, alpha=alpha)
    return outputs[0] if isinstance(outputs, tuple) else outputs


def classification_metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def save_rows_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
