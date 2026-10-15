import sys
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EANN_DIR = PROJECT_ROOT / "EANN"
if str(EANN_DIR) not in sys.path:
    sys.path.insert(0, str(EANN_DIR))

from models_EANN import EANN, ResNet18_EANN, TextCNN_EANN  # noqa: E402


class PlainMultimodalDetector(nn.Module):
    """Ordinary multimodal detector without event adversarial training."""

    def __init__(self, vocab_size, shared_dim=32):
        super().__init__()
        self.text_extractor = TextCNN_EANN(vocab_size=vocab_size, output_dim=shared_dim)
        self.visual_extractor = ResNet18_EANN(output_dim=shared_dim)
        self.detector = nn.Sequential(
            nn.Linear(shared_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, text, image):
        text_feat = self.text_extractor(text)
        image_feat = self.visual_extractor(image)
        fused_feat = torch.cat((text_feat, image_feat), dim=1)
        return self.detector(fused_feat)


def build_model(model_name, vocab_size, num_events=10):
    if model_name == "plain":
        return PlainMultimodalDetector(vocab_size=vocab_size)
    if model_name == "eann":
        return EANN(vocab_size=vocab_size, num_events=num_events)
    raise ValueError(f"Unknown model_name: {model_name}")
