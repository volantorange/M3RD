# 文件名：teacher_models.py
import torch
import torch.nn as nn
from models_EANN import TextCNN_EANN, ResNet18_EANN

class TeacherText(nn.Module):
    """文本教师模型：仅依赖文本特征"""
    def __init__(self, vocab_size, hidden_dim=64, num_classes=2):
        super(TeacherText, self).__init__()
        self.extractor = TextCNN_EANN(vocab_size=vocab_size, output_dim=hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, text, image=None):
        # 教师模型只看文本，忽略图像输入
        feat = self.extractor(text)
        return self.classifier(feat)

class TeacherImage(nn.Module):
    """图像教师模型：仅依赖视觉特征"""
    def __init__(self, hidden_dim=64, num_classes=2):
        super(TeacherImage, self).__init__()
        self.extractor = ResNet18_EANN(output_dim=hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, text=None, image=None):
        # 教师模型只看图像，忽略文本输入
        feat = self.extractor(image)
        return self.classifier(feat)