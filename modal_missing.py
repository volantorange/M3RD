import os
import torch
import json
import random
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset_utils import CFNDDataset, transform
from models import MultimodalDetector

# 1. 基础配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_path = "CFND_dataset"  # 你的根目录，注意跟之前脚本里的相对路径保持一致
test_csv = "test_data.csv"  # 确保你有这个测试集切分
vocab_path = "vocab.json"
missing_rates = [0.0, 0.2, 0.5, 1.0]

print("⏳ 正在准备 Baseline (TextCNN+ResNet18) 模态缺失全指标测试环境...")

# 2. 加载词表和数据集
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)

test_dataset = CFNDDataset(os.path.join(root_path, test_csv), root_path, vocab, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. 加载已经训练好的 Baseline 模型
model = MultimodalDetector(vocab_size=len(vocab)).to(device)
weights_path = "best_model.pth"  # 对应你 Training Loop.py 中保存的名字

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    print(f"✅ 成功加载基准模型权重：{weights_path}")
else:
    print(f"❌ 找不到权重文件 {weights_path}！请确认模型已训练并保存。")
    exit()


# 4. 定义缺失评估函数
def evaluate_missing_baseline(model, dataloader, missing_rate, missing_modality):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # Baseline 数据集 __getitem__ 只返回 3 个值：text_seq, image, label
        for texts, images, labels in dataloader:
            texts = texts.clone().to(device)
            images = images.clone().to(device)
            labels = labels.to(device)

            batch_size = texts.size(0)

            # 遍历 batch，按概率执行丢弃 (将输入张量置零)
            for i in range(batch_size):
                if random.random() < missing_rate:
                    if missing_modality == 'text':
                        texts[i] = torch.zeros_like(texts[i])
                    elif missing_modality == 'image':
                        images[i] = torch.zeros_like(images[i])

            # Baseline 模型前向传播，仅需 text 和 image
            outputs = model(texts, images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算四大核心指标 (宏平均)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return acc, prec, rec, f1


# 5. 开始多场景遍历测试并输出精美表格
print("\n🚀 开始进行 Baseline 模型模态缺失场景全指标评测...\n")
print("=" * 95)
print(
    f"{'缺失率':<8} | {'缺失目标':<12} | {'准确率 (Acc)':<12} | {'精确率 (Pre)':<12} | {'召回率 (Rec)':<12} | {'F1 分数':<12}")
print("-" * 95)

# 黄金基准线：0% 缺失
acc, prec, rec, f1 = evaluate_missing_baseline(model, test_loader, 0.0, 'none')
print(
    f"{'0.0%':<8} | {'完整图文':<12} | {acc * 100:>10.2f}% | {prec * 100:>10.2f}% | {rec * 100:>10.2f}% | {f1 * 100:>10.2f}%")
print("-" * 95)

# 逐步增加缺失率进行测试
for rate in missing_rates:
    if rate == 0.0:
        continue  # 0% 已经测过

    # 模拟文本缺失
    acc, prec, rec, f1 = evaluate_missing_baseline(model, test_loader, rate, 'text')
    print(
        f"{rate * 100:>6.1f}%   | {'仅丢失文本':<12} | {acc * 100:>10.2f}% | {prec * 100:>10.2f}% | {rec * 100:>10.2f}% | {f1 * 100:>10.2f}%")

    # 模拟图像缺失
    acc, prec, rec, f1 = evaluate_missing_baseline(model, test_loader, rate, 'image')
    print(
        f"{rate * 100:>6.1f}%   | {'仅丢失图像':<12} | {acc * 100:>10.2f}% | {prec * 100:>10.2f}% | {rec * 100:>10.2f}% | {f1 * 100:>10.2f}%")
    print("-" * 95)

print("\n✅ Baseline 全指标测试完成！可将此数据作为消融实验的对照组，与未来的 TGA-KD 系统进行对比。")
# 模拟模态缺失
