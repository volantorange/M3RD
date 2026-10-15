"""
文件名：Test_Single_Modality.py
用途：测试极端缺失场景（纯文本 vs 纯图像）下的学生模型准确率
时间：2026-05-08
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import numpy as np
import pandas as pd
from dataset_utils import CFNDDataset, transform
from models_EANN import EANN, TextTeacher, ImageTeacher
import torch.nn.functional as F

# --- 1. 配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_path = "../CFND_dataset"

# 加载词表
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

num_events = 50

# --- 2. 加载模型 ---
print("🔄 正在加载模型...")

# 学生模型
student_model = EANN(vocab_size=len(vocab), num_events=num_events).to(device)
student_model.load_state_dict(torch.load('eann_student_kd_final.pth'))
student_model.eval()

# 教师模型 (用于对比)
text_teacher = TextTeacher(vocab_size=len(vocab)).to(device)
image_teacher = ImageTeacher().to(device)
text_teacher.load_state_dict(torch.load('text_teacher_best.pth'))
image_teacher.load_state_dict(torch.load('image_teacher_best.pth'))
text_teacher.eval()
image_teacher.eval()

print("✅ 模型加载完成")

# --- 3. 数据加载 (使用完整数据，我们在循环中手动抹除) ---
# 注意：这里 missing_rate=0.0，我们要拿到原始完整数据，然后手动破坏它
test_dataset = CFNDDataset(
    csv_file=os.path.join(root_path, "test_data.csv"),
    root_dir=root_path,
    vocab=vocab,
    transform=transform,
    missing_rate=0.0
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# --- 4. 定义测试函数 ---

def test_text_only():
    """
    模拟：文本缺失100% (只有图像)
    操作：将文本输入替换为全0张量
    """
    print("\n🔍 正在测试：纯图像模式 (文本缺失100%)...")
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)

            # 【核心操作】模拟文本缺失
            # 创建一个全0的文本张量，形状与原文本相同
            # 这相当于输入了 [PAD] 或空文本
            fake_texts = torch.zeros_like(texts).to(device)

            # 学生模型推理
            outputs, _ = student_model(fake_texts, images, alpha=0.0)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def test_image_only():
    """
    模拟：图像缺失100% (只有文本)
    操作：将图像输入替换为全0张量 (黑图)
    """
    print("\n🔍 正在测试：纯文本模式 (图像缺失100%)...")
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, images, labels, _ in test_loader:
            texts, labels = texts.to(device), labels.to(device)

            # 【核心操作】模拟图像缺失
            # 创建一个全0的图像张量，形状与图像通道、高、宽一致
            # 这相当于输入了全黑图片
            fake_images = torch.zeros_like(images).to(device)

            # 学生模型推理
            outputs, _ = student_model(texts, fake_images, alpha=0.0)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def test_full_multimodal():
    """
    基准测试：完整多模态数据
    """
    print("\n🔍 正在测试：完整多模态模式 (基准)...")
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, images, labels, _ in test_loader:
            texts, images, labels = texts.to(device), images.to(device), labels.to(device)

            outputs, _ = student_model(texts, images, alpha=0.0)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# --- 5. 执行测试并汇总 ---

# 1. 测基准
acc_full = test_full_multimodal()

# 2. 测纯图像 (文本缺失)
acc_image_only = test_text_only()

# 3. 测纯文本 (图像缺失)
acc_text_only = test_image_only()

# --- 6. 输出结果表格 ---
print("\n" + "=" * 70)
print("📊 极端缺失场景测试结果")
print("=" * 70)

data = {
    '测试场景': ['完整多模态', '纯文本 (图像缺失)', '纯图像 (文本缺失)'],
    '准确率 (%)': [acc_full, acc_text_only, acc_image_only]
}

df = pd.DataFrame(data)
print(df.to_string(index=False))
print("-" * 70)

# --- 7. 性能下降分析 ---
print(f"\n📉 性能下降分析：")
print(f"  仅使用文本导致准确率下降: {acc_full - acc_text_only:.2f}%")
print(f"  仅使用图像导致准确率下降: {acc_full - acc_image_only:.2f}%")

# 判断哪个模态更重要
if acc_text_only > acc_image_only:
    print(f"  💡 结论：文本模态包含更关键的谣言特征 (文本准确率 {acc_text_only:.2f}% > 图像准确率 {acc_image_only:.2f}%)")
else:
    print(f"  💡 结论：图像模态包含更关键的谣言特征")

print("\n✅ 单模态测试完成！")