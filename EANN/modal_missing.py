import os
import torch
import json
import random
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset_utils import CFNDDataset, transform
from models_EANN import EANN

# 1. 基础配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_path = "../CFND_dataset"
test_csv = "test_data.csv"
missing_rates = [0.0, 0.2, 0.5, 1.0]  # 对应 0%, 20%, 50%, 100% 的缺失率
num_events = 50

print("⏳ 正在准备 EANN 模态缺失全指标测试环境...")

# 2. 加载词表和数据集
vocab_path = "vocab.json"
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)

test_dataset = CFNDDataset(os.path.join(root_path, test_csv), root_path, vocab, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. 加载已经训练好的模型 (这里默认加载经过蒸馏的最佳模型，你也可以改成基线模型)
model = EANN(vocab_size=len(vocab), num_events=num_events).to(device)

# ⚠️ 注意：你可以修改这里，分别测试 eann_best_model.pth 和 eann_distilled_feature_best.pth
weights_path = "eann_distilled_feature_best.pth"

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    print(f"✅ 成功加载 EANN 模型权重：{weights_path}")
else:
    print(f"❌ 未找到权重文件 {weights_path}！请确认模型已保存。")
    exit()

# 4. 定义包含全面评估指标的缺失测试函数
def evaluate_missing_eann(model, dataloader, missing_rate, missing_modality):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # 使用 *_ 兼容不同版本的 Dataset 返回值 (无论有无 event_labels)
        for texts, images, labels, *_ in dataloader:
            texts = texts.clone().to(device)
            images = images.clone().to(device)
            labels = labels.to(device)

            batch_size = texts.size(0)

            # 遍历 batch，执行模态丢弃 (置零)
            for i in range(batch_size):
                if random.random() < missing_rate:
                    if missing_modality == 'text':
                        texts[i] = torch.zeros_like(texts[i])
                    elif missing_modality == 'image':
                        images[i] = torch.zeros_like(images[i])

            # 推理阶段关闭对抗 (alpha=0.0)
            # EANN 返回: fake_output, event_output, text_feat, vis_feat
            fake_out, _, _, _ = model(texts, images, alpha=0.0)
            _, preds = torch.max(fake_out, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算四大核心指标 (使用 macro 平均，公平对待真假新闻两类)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return acc, prec, rec, f1

# 5. 开始多场景遍历测试并输出表格
print("\n🚀 开始进行 EANN 模态缺失场景全指标评测...\n")
print("=" * 95)
print(f"{'缺失率':<8} | {'缺失目标':<12} | {'准确率 (Acc)':<12} | {'精确率 (Pre)':<12} | {'召回率 (Rec)':<12} | {'F1 分数':<12}")
print("-" * 95)

# 黄金基准线：0% 缺失
acc, prec, rec, f1 = evaluate_missing_eann(model, test_loader, 0.0, 'none')
print(f"{'0.0%':<8} | {'完整图文':<12} | {acc * 100:>10.2f}% | {prec * 100:>10.2f}% | {rec * 100:>10.2f}% | {f1 * 100:>10.2f}%")
print("-" * 95)

# 逐步增加缺失率进行测试
for rate in missing_rates:
    if rate == 0.0:
        continue

    # 测试文本缺失
    acc, prec, rec, f1 = evaluate_missing_eann(model, test_loader, rate, 'text')
    print(f"{rate * 100:>6.1f}%   | {'仅丢失文本':<12} | {acc * 100:>10.2f}% | {prec * 100:>10.2f}% | {rec * 100:>10.2f}% | {f1 * 100:>10.2f}%")

    # 测试图像缺失
    acc, prec, rec, f1 = evaluate_missing_eann(model, test_loader, rate, 'image')
    print(f"{rate * 100:>6.1f}%   | {'仅丢失图像':<12} | {acc * 100:>10.2f}% | {prec * 100:>10.2f}% | {rec * 100:>10.2f}% | {f1 * 100:>10.2f}%")
    print("-" * 95)

print("\n✅ 全指标测试完成！你可以通过修改脚本第 32 行的 weights_path，分别测试 Baseline 和 蒸馏版 的表现。")