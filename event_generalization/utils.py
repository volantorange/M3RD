import json
import random
from pathlib import Path

import jieba
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
DATASET_ROOT = PROJECT_ROOT / "CFND_dataset"
VOCAB_PATH = PROJECT_ROOT / "vocab.json"

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class EventSplitDataset(Dataset):
    def __init__(self, csv_file, vocab, max_len=50):
        self.data_frame = pd.read_csv(csv_file)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data_frame)

    def text_to_sequence(self, text):
        words = jieba.lcut(str(text))
        seq = [self.vocab.get(word, self.vocab.get("<UNK>", 1)) for word in words]
        if len(seq) < self.max_len:
            seq += [self.vocab.get("<PAD>", 0)] * (self.max_len - len(seq))
        else:
            seq = seq[: self.max_len]
        return torch.tensor(seq, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        text = self.text_to_sequence(row["title"])

        image_path = DATASET_ROOT / str(row["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        image = IMAGE_TRANSFORM(image)

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        event = torch.tensor(int(row["event"]), dtype=torch.long)
        return text, image, label, event


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vocab():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def make_loader(split, vocab, batch_size, shuffle):
    path = DATA_DIR / f"{split}_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run split_by_event.py first.")
    dataset = EventSplitDataset(path, vocab)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def metrics_from_predictions(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
