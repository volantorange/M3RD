import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from dataset_utils import CFNDDataset, transform
from models import MultimodalDetector

# 模拟简单词表
# vocab = {'<PAD>': 0, '<UNK>': 1, '谣言': 2, '真实': 3}
# vocab_size = len(vocab) + 10000

# 读取词表
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

# 词典大小
vocab_size = len(vocab)

# 实例化Dataset和DataLoader
root_path = "CFND_dataset"
train_dataset = CFNDDataset(os.path.join(root_path, "train_data.csv"), root_path, vocab, transform=transform)
val_dataset = CFNDDataset(os.path.join(root_path, "val_data.csv"), root_path, vocab, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 初始化模型
device = torch.device("cpu")
model = MultimodalDetector(vocab_size=vocab_size).to(device)

# 定义损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
best_val_acc = 0.0
# 训练循环
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (texts, images, labels) in enumerate(train_loader):
        texts, images, labels = texts.to(device), images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts, images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    train_acc = 100. * correct / total

    # 验证
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for texts, images, labels in val_loader:
            texts, images, labels = texts.to(device), images.to(device), labels.to(device)
            outputs = model(texts, images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100. * val_correct / val_total
    print(f"==> Epoch {epoch + 1} Summary | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # 保存最优模型权重到本地
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        #
        torch.save(model.state_dict(), "best_model.pth")
    else:
        print("\n")
