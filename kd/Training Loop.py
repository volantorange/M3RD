import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import numpy as np
from dataset_utils import CFNDDataset, transform
from models_EANN import EANN  # 导入刚刚拼装好的 EANN 模型
import pandas as pd

# 1. 配置与参数
device = torch.device("cpu")
root_path = "../CFND_dataset"
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

# 获取总事件数 (K)
temp_df = pd.read_csv(os.path.join(root_path, "train_data.csv"))
num_events = 50
print(f"📊 训练开始：词表大小 {len(vocab)}, 识别到 {num_events} 个事件")

# 2. 加载数据
train_dataset = CFNDDataset(os.path.join(root_path, "train_data.csv"), root_path, vocab, transform=transform, missing_rate=1)
val_dataset = CFNDDataset(os.path.join(root_path, "val_data.csv"), root_path, vocab, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 3. 实例化模型
model = EANN(vocab_size=len(vocab), num_events=num_events).to(device)

# 4. 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# 5. 训练循环
epochs = 20
for epoch in range(epochs):
    model.train()

    for batch_idx, (texts, images, labels, event_labels) in enumerate(train_loader):
        texts, images, labels, event_labels = texts.to(device), images.to(device), labels.to(device), event_labels.to(
            device)

        # 计算动态 alpha (对抗强度系数)，参考论文公式 15 [cite: 343]
        p = float(batch_idx + epoch * len(train_loader)) / epochs / len(train_loader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        optimizer.zero_grad()

        # 前向传播：得到谣言预测和事件预测
        fake_out, event_out = model(texts, images, alpha=alpha)

        # 计算损失 [cite: 326]
        loss_d = criterion(fake_out, labels)  # 谣言检测损失
        loss_e = criterion(event_out, event_labels)  # 事件鉴别损失

        # 总损失 = 检测损失 + 对抗损失
        # 注意：由于 GRL 的存在，反向传播时 loss_e 的梯度会自动变负，实现对抗 [cite: 337]
        loss = loss_d + loss_e

        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print(
                f"Epoch {epoch + 1} [{batch_idx}/{len(train_loader)}] | Loss D: {loss_d.item():.4f} | Loss E: {loss_e.item():.4f}")

    # 验证逻辑 (仅评估谣言检测的准确率)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, images, labels, _ in val_loader:
            texts, images, labels = texts.to(device), images.to(device), labels.to(device)
            fake_out, _ = model(texts, images, alpha=0)
            _, predicted = torch.max(fake_out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"==> Epoch {epoch + 1} Val Acc: {100 * correct / total:.2f}%")
torch.save(model.state_dict(), 'eann_best_model.pth')