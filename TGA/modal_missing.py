import os
import torch
import json
import random
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from dataset_utils import CFNDDataset, transform
# 修改点 1：导入 TGA 模型
from models_TGA import TGA

# 1. 基础配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_path = "../CFND_dataset"
test_csv = "test_data.csv"
missing_rates = [0.0, 0.2, 0.5, 1.0]  # 对应 0%, 20%, 50%, 100% 的缺失率
num_events = 50

print("⏳ 正在准备 TGA 模态缺失测试环境...")

# 2. 加载词表和数据集
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

test_dataset = CFNDDataset(os.path.join(root_path, test_csv), root_path, vocab, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. 加载已经训练好的 TGA 蒸馏模型
model = TGA(vocab_size=len(vocab), num_events=num_events).to(device)
# 修改点 2：指定 TGA 蒸馏后的最佳权重文件
weights_path = "tga_distilled_feature_best.pth"

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"✅ 成功加载权重文件：{weights_path}")
else:
    print(f"❌ 未找到权重文件 {weights_path}！请确认蒸馏模型已成功训练并保存。")
    exit()


# 4. 定义缺失评估函数
def evaluate_missing(model, dataloader, missing_rate, missing_modality):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, images, labels, _ in dataloader:
            texts = texts.clone().to(device)
            images = images.clone().to(device)
            labels = labels.to(device)

            batch_size = texts.size(0)

            # 修改点 3：在推理阶段实时构建缺失标记 Tensor，确保 TGA 门控安全触发
            text_missing_flag = torch.zeros(batch_size, dtype=torch.float32).to(device)
            image_missing_flag = torch.zeros(batch_size, dtype=torch.float32).to(device)

            # 遍历 batch 中的每个样本，按概率执行丢弃 (置零)
            for i in range(batch_size):
                if random.random() < missing_rate:
                    if missing_modality == 'text':
                        texts[i] = torch.zeros_like(texts[i])
                        text_missing_flag[i] = 1.0  # 标记当前样本缺失文本
                    elif missing_modality == 'image':
                        images[i] = torch.zeros_like(images[i])
                        image_missing_flag[i] = 1.0  # 标记当前样本缺失图像

            # 修改点 4：将动态计算出来的缺失标记送入 TGA 前向传播
            fake_out, *_ = model(
                texts, images,
                text_missing_flag=text_missing_flag,
                image_missing_flag=image_missing_flag,
                grl_alpha=0.0
            )
            _, preds = torch.max(fake_out, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds)


# 5. 开始多场景遍历测试
print("\n🚀 开始进行 TGA 模态缺失场景评测...\n")
print("-" * 60)
print(f"{'缺失率 (Missing Rate)':<20} | {'缺失文本 (Drop Text)':<15} | {'缺失图像 (Drop Image)':<15}")
print("-" * 60)

results_text = []
results_image = []

for rate in missing_rates:
    if rate == 0.0:
        acc = evaluate_missing(model, test_loader, rate, 'none')
        results_text.append(acc)
        results_image.append(acc)
        print(f"{rate * 100:>16.1f}% | {acc * 100:>13.2f}% | {acc * 100:>13.2f}% (TGA-KD Full)")
    else:
        acc_text = evaluate_missing(model, test_loader, rate, 'text')
        results_text.append(acc_text)

        acc_image = evaluate_missing(model, test_loader, rate, 'image')
        results_image.append(acc_image)

        print(f"{rate * 100:>16.1f}% | {acc_text * 100:>13.2f}% | {acc_image * 100:>13.2f}%")

print("-" * 60)
print("✅ 测试完成！请将此表格数据作为 TGA 模型性能记录在你的毕业论文中。")