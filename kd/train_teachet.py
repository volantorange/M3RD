"""
文件名：Train_Student_Distillation.py
用途：利用预训练的单模态教师模型，蒸馏训练多模态EANN学生模型（应对模态缺失）。
时间：2026-05-06
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import numpy as np
from dataset_utils import CFNDDataset, transform
from models_EANN import EANN, TextTeacher, ImageTeacher  # 确保模型定义已导入
import torch.nn.functional as F

# --- 1. 配置与参数 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_path = "../CFND_dataset"

# 加载词表
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

# 事件数量
num_events = 50
print(f"📚 词表大小: {len(vocab)}, 事件数: {num_events}")

# 蒸馏超参数 (参考论文常用设置)
T = 5.0  # 温度 Temperature
alpha = 0.7  # Hard Label Loss 的权重 (1-alpha 是 Distill Loss 的权重)

# --- 2. 数据加载 ---
# 关键点：在训练学生模型时，我们主动引入 missing_rate 来模拟真实场景的缺失
# 验证集保持完整 (missing_rate=0) 以观察泛化能力
train_dataset = CFNDDataset(
    csv_file=os.path.join(root_path, "train_data.csv"),
    root_dir=root_path,
    vocab=vocab,
    transform=transform,
    missing_rate=0.3  # 模拟30%的模态缺失，强迫学生模型去看教师模型
)
val_dataset = CFNDDataset(
    csv_file=os.path.join(root_path, "val_data.csv"),
    root_dir=root_path,
    vocab=vocab,
    transform=transform,
    missing_rate=0.0  # 验证时不模拟缺失
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --- 3. 初始化模型组件 ---

# A. 学生模型 (EANN)
student_model = EANN(vocab_size=len(vocab), num_events=num_events).to(device)
optimizer = optim.Adam(student_model.parameters(), lr=1e-3)
# 注意：EANN的Loss包含谣言检测和事件对抗，这里我们主要关注谣言检测的蒸馏

# B. 教师模型 (Text & Image)
text_teacher = TextTeacher(vocab_size=len(vocab)).to(device)
image_teacher = ImageTeacher().to(device)

# 加载预训练权重
print("🔄 正在加载教师模型预训练权重...")
text_teacher.load_state_dict(torch.load('text_teacher_best.pth'))
image_teacher.load_state_dict(torch.load('image_teacher_best.pth'))

# 冻结教师模型参数 (非常重要！)
text_teacher.eval()
image_teacher.eval()

for param in text_teacher.parameters():
    param.requires_grad = False
for param in image_teacher.parameters():
    param.requires_grad = False

print("✅ 教师模型已加载并冻结。")

# 损失函数
criterion_cls = nn.CrossEntropyLoss()  # 真实标签损失
criterion_kl = nn.KLDivLoss(reduction="batchmean")  # KL散度损失 (蒸馏)

# --- 4. 训练循环 ---
epochs = 20

for epoch in range(epochs):
    student_model.train()
    total_loss = 0
    distill_loss = 0

    for batch_idx, (texts, images, labels, event_labels) in enumerate(train_loader):
        texts, images, labels = texts.to(device), images.to(device), labels.to(device)

        optimizer.zero_grad()

        # --- 前向传播：教师模型 (提供监督信号) ---
        with torch.no_grad():
            # 教师模型不看缺失的数据，它们看的是“完整”的单模态信息
            # 注意：即使学生模型看到黑图，教师模型依然能看到正确的图像特征（在实际部署中，这是通过分支逻辑实现的）
            # 这里为了简化，我们假设教师模型输入是完整的（在蒸馏时，我们通常用完整数据喂给教师，或者教师只看自己有的模态）
            # 修正：教师模型只依赖自己的模态
            text_logits = text_teacher(texts, None)
            image_logits = image_teacher(None, images)

            # 软化标签
            text_logit_soft = text_logits / T
            image_logit_soft = image_logits / T

            # 教师平均概率 (Soft Target)
            # 使用 Log-Space 的 Softmax 用于 KL Loss 计算
            teacher_text_prob = F.softmax(text_logit_soft, dim=1)
            teacher_image_prob = F.softmax(image_logit_soft, dim=1)
            # 简单平均两个教师的知识
            teacher_prob = (teacher_text_prob + teacher_image_prob) / 2

        # --- 前向传播：学生模型 (EANN) ---
        # 学生模型接收可能包含缺失的输入
        fake_out, event_out = student_model(texts, images, alpha=1.0)

        # --- 计算损失 ---
        # 1. 学生模型对真实标签的预测 (Hard Loss)
        loss_cls = criterion_cls(fake_out, labels)

        # 2. 蒸馏损失 (Student 模仿 Teacher)
        # 学生模型的输出也需要经过 Log-Softmax
        student_log_prob = F.log_softmax(fake_out / T, dim=1)

        # KL 散度损失
        loss_distill = criterion_kl(student_log_prob, teacher_prob)

        # 3. 总损失 (结合 Hard 和 Soft)
        # 公式: Loss = alpha * Hard_Loss + (1-alpha) * Soft_Loss * T^2
        loss = alpha * loss_cls + (1 - alpha) * loss_distill * (T * T)

        # 反向传播 (只更新学生模型)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        distill_loss += loss_distill.item()

        if batch_idx % 20 == 0:
            print(
                f"Epoch: {epoch + 1} | Batch: {batch_idx} | Total Loss: {loss.item():.4f} | Distill Loss: {loss_distill.item():.4f}")

    # --- 验证阶段 (验证集不包含缺失) ---
    student_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, images, labels, _ in val_loader:
            texts, images, labels = texts.to(device), images.to(device), labels.to(device)
            fake_out, _ = student_model(texts, images, alpha=0.0)  # 推理时 alpha=0
            _, predicted = torch.max(fake_out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"🚀 Epoch {epoch + 1} 完成 | 平均训练损失: {total_loss / len(train_loader):.4f} | 验证准确率: {val_acc:.2f}%")

# --- 5. 保存最终模型 ---
torch.save(student_model.state_dict(), 'eann_student_kd_final.pth')
print("🎉 知识蒸馏训练完成！学生模型已保存。")