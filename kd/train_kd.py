# 文件名：train_kd.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from dataset_utils import CFNDDataset, transform
from models_EANN import EANN
from teacher_models import TeacherText, TeacherImage  # 导入教师

# --- 1. 配置与初始化 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_path = "../CFND_dataset"
vocab_path = "vocab.json"

with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)

num_events = 50
vocab_size = len(vocab)

# --- 2. 数据加载 ---
# 注意：dataset_utils 中的 missing_rate 应该在训练蒸馏时设为 0，或者我们在蒸馏 Loss 中处理
train_dataset = CFNDDataset(os.path.join(root_path, "train_data.csv"),
                            root_path, vocab, transform=transform, missing_rate=0.0)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = CFNDDataset(os.path.join(root_path, "val_data.csv"),
                          root_path, vocab, transform=transform, missing_rate=0.0)

# --- 3. 模型初始化 ---
# 学生模型
student_model = EANN(vocab_size=vocab_size, num_events=num_events).to(device)

# --- 关键：手动冻结学生模型中的视觉和文本提取器 ---
# 冻结视觉提取器
for param in student_model.visual_extractor.parameters():
    param.requires_grad = False
# 冻结文本提取器 (可选，取决于你是否想微调Embedding)
for param in student_model.text_extractor.parameters():
    param.requires_grad = False
# 教师模型 (文本 & 图像)
teacher_text = TeacherText(vocab_size=vocab_size).to(device)
teacher_image = TeacherImage().to(device)

# --- 4. 加载教师预训练权重 (关键步骤) ---
# 你需要先单独训练好 teacher_text 和 teacher_image，或者使用提供的预训练权重
# teacher_text.load_state_dict(torch.load("teacher_text_pretrained.pth"))
# teacher_image.load_state_dict(torch.load("teacher_image_pretrained.pth"))

# 冻结教师模型参数
# for param in teacher_text.parameters():
#     param.requires_grad = False
# for param in teacher_image.parameters():
#     param.requires_grad = False

teacher_text.eval()
teacher_image.eval()

# --- 5. 优化器与损失函数 ---
optimizer = optim.Adam(student_model.parameters(), lr=1e-4)
criterion_task = nn.CrossEntropyLoss()  # 真实标签损失
criterion_kd = nn.KLDivLoss(reduction="batchmean")  # 蒸馏损失 (KL散度)

# 蒸馏超参数
alpha = 0.7  # 真实任务损失权重
beta = 0.3  # 蒸馏损失权重
T = 5  # 温度 Temperature

# --- 6. 训练循环 ---
epochs = 20
for epoch in range(epochs):
    student_model.train()
    for batch_idx, (texts, images, labels, event_labels) in enumerate(train_loader):
        texts, images, labels = texts.to(device), images.to(device), labels.to(device)

        optimizer.zero_grad()

        # --- 前向传播：学生模型 ---
        # 注意：在蒸馏阶段，我们通常关闭对抗训练的 alpha (设为0)，或者保留。
        # 这里为了简单，先只关注谣言检测头 (fake_out)
        fake_out_student, _ = student_model(texts, images, alpha=0.0)

        # --- 获取教师输出 (Soft Targets) ---
        with torch.no_grad():
            teacher_logit_t = teacher_text(texts)  # 文本教师只看文本
            teacher_logit_v = teacher_image(images)  # 图像教师只看图像
            # 简单平均融合两位教师的知识
            teacher_logits = (teacher_logit_t + teacher_logit_v) / 2

            # --- 计算损失 ---
        # 1. 真实标签损失
        loss_task = criterion_task(fake_out_student, labels)

        # 2. 蒸馏损失 (学生模仿教师的 Soft Label)
        # 使用 log_softmax 处理学生输出，softmax 处理教师输出
        loss_kd = criterion_kd(
            nn.functional.log_softmax(fake_out_student / T, dim=1),
            nn.functional.softmax(teacher_logits / T, dim=1)
        ) * (T * T)  # 温度缩放

        # 3. 总损失
        loss = alpha * loss_task + beta * loss_kd

        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f} (Task: {loss_task:.4f}, KD: {loss_kd:.4f})")

    # --- 验证逻辑 (Validation) ---
    # 这里可以加入模态缺失的模拟测试，例如随机 mask 掉文本或图像
    student_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, images, labels, _ in val_loader:
            texts, images, labels = texts.to(device), images.to(device), labels.to(device)

            # 模拟模态缺失测试：例如这里只给图像，不给文本 (文本置零)
            # texts = torch.zeros_like(texts)

            fake_out, _ = student_model(texts, images, alpha=0)
            _, predicted = torch.max(fake_out, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Acc: {100 * correct / total:.2f}%")

torch.save(student_model.state_dict(), 'eann_kd_model.pth')