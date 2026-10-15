import torch.nn as nn

from models_EANN import TextCNN_EANN, ResNet18_EANN


class TextTeacher(nn.Module):
    """Single-modality teacher that only uses the news text."""

    def __init__(self, vocab_size, shared_dim=32):
        super().__init__()
        self.extractor = TextCNN_EANN(vocab_size=vocab_size, output_dim=shared_dim)
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, text):
        feat = self.extractor(text)
        return self.classifier(feat), feat


class ImageTeacher(nn.Module):
    """Single-modality teacher that only uses the news image."""

    def __init__(self, shared_dim=32):
        super().__init__()
        self.extractor = ResNet18_EANN(output_dim=shared_dim)
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, image):
        feat = self.extractor(image)
        return self.classifier(feat), feat
