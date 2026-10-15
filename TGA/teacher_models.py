import torch
import torch.nn as nn
import torch.nn.functional as F
# 确保你已经把上一轮给你的全新架构保存为了 models_TGA.py
from models_TGA import TextTransformer_Extractor, ViT_Extractor

class TextTeacher(nn.Module):
    """基于 Text-Transformer 的单模态文本教师模型，仅使用新闻文本"""

    def __init__(self, vocab_size, shared_dim=32):
        super().__init__()
        # 彻底替换为 TGA 同款的文本 Transformer 提取器
        self.extractor = TextTransformer_Extractor(vocab_size=vocab_size, output_dim=shared_dim)
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # 二分类：真/假新闻
        )

    def forward(self, text):
        feat = self.extractor(text)
        return self.classifier(feat), feat


class ImageTeacher(nn.Module):
    """基于 Vision Transformer (ViT) 的单模态图像教师模型，仅使用新闻图片"""

    def __init__(self, shared_dim=32):
        super().__init__()
        # 彻底替换为 TGA 同款的 ViT 图像提取器
        self.extractor = ViT_Extractor(output_dim=shared_dim)
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # 二分类：真/假新闻
        )

    def forward(self, image):
        feat = self.extractor(image)
        return self.classifier(feat), feat