# # fi指标等
# import os
# import torch
# import json
# import random
# from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# from dataset_utils import CFNDDataset, transform
# from models_EANN import EANN_Attention
#
# # 1. 基础配置
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# root_path = "../CFND_dataset"
# test_csv = "test_data.csv"
# missing_rates = [0.0, 0.2, 0.5, 1.0]  # 对应 0%, 20%, 50%, 100% 的缺失率
# num_events = 50
#
# print("⏳ 正在准备多维度模态缺失测试环境...")
#
# # 2. 加载词表和数据集
# with open("vocab.json", "r", encoding="utf-8") as f:
#     vocab = json.load(f)
#
# test_dataset = CFNDDataset(os.path.join(root_path, test_csv), root_path, vocab, transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#
# # 3. 加载已经训练好的 EANN 模型
# model = EANN_Attention(vocab_size=len(vocab), num_events=num_events).to(device)
# weights_path = "eann_best_model.pth"
# if os.path.exists(weights_path):
#     model.load_state_dict(torch.load(weights_path, map_location=device))
#     print(f"✅ 成功加载模型权重: {weights_path}")
# else:
#     print("❌ 未找到权重文件 eann_best_model.pth！请确认模型已保存。")
#     exit()
#
#
# # 4. 定义包含全面评估指标的缺失测试函数
# def evaluate_missing_full(model, dataloader, missing_rate, missing_modality):
#     model.eval()
#     all_preds = []
#     all_labels = []
#
#     with torch.no_grad():
#         for texts, images, labels, _ in dataloader:
#             texts = texts.clone().to(device)
#             images = images.clone().to(device)
#             labels = labels.to(device)
#
#             batch_size = texts.size(0)
#
#             # 执行模态丢弃 (置零)
#             for i in range(batch_size):
#                 if random.random() < missing_rate:
#                     if missing_modality == 'text':
#                         texts[i] = torch.zeros_like(texts[i])
#                     elif missing_modality == 'image':
#                         images[i] = torch.zeros_like(images[i])
#
#             # 推理阶段关闭对抗 (alpha=0.0)
#             fake_out, _, _, _ = model(texts, images, alpha=0.0)
#             _, preds = torch.max(fake_out, 1)
#
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#
#     # 计算四大核心指标 (使用 macro 平均，公平对待真假新闻两类)
#     acc = accuracy_score(all_labels, all_preds)
#     prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
#     rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
#     f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
#
#     return acc, prec, rec, f1, all_labels, all_preds
#
#
# # 5. 开始多场景遍历测试并输出精美表格
# print("\n🚀 开始进行模态缺失场景全指标评测...\n")
#
# # 测试文本缺失
# print("=" * 80)
# print("🎯 场景一：【文本缺失】测试报告")
# print("-" * 80)
# print(f"{'缺失率':<10} | {'Accuracy':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12}")
# print("-" * 80)
# for rate in missing_rates:
#     acc, prec, rec, f1, _, _ = evaluate_missing_full(model, test_loader, rate, 'text')
#     if rate == 0.0:
#         print(
#             f"{rate * 100:>8.1f}% | {acc * 100:>10.2f}% | {prec * 100:>10.2f}% | {rec * 100:>10.2f}% | {f1 * 100:>10.2f}%  (Baseline)")
#     else:
#         print(
#             f"{rate * 100:>8.1f}% | {acc * 100:>10.2f}% | {prec * 100:>10.2f}% | {rec * 100:>10.2f}% | {f1 * 100:>10.2f}%")
#
# # 测试图像缺失
# print("\n" + "=" * 80)
# print("🖼️ 场景二：【图像缺失】测试报告")
# print("-" * 80)
# print(f"{'缺失率':<10} | {'Accuracy':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12}")
# print("-" * 80)
# for rate in missing_rates:
#     # 0% 的情况已经在上面跑过了，这里略过以节省时间，直接从 20% 开始打印
#     if rate == 0.0:
#         continue
#     acc, prec, rec, f1, _, _ = evaluate_missing_full(model, test_loader, rate, 'image')
#     print(
#         f"{rate * 100:>8.1f}% | {acc * 100:>10.2f}% | {prec * 100:>10.2f}% | {rec * 100:>10.2f}% | {f1 * 100:>10.2f}%")
#
# print("=" * 80)
#
# # 6. 极限情况 (100%缺失) 的详细分类报告 (非常有助于论文分析)
# print("\n🔍 极限情况深度诊断 (100% 文本缺失时，模型到底怎么了？)")
# _, _, _, _, final_labels, final_preds = evaluate_missing_full(model, test_loader, 1.0, 'text')
# print(classification_report(final_labels, final_preds, target_names=['真实新闻 (0)', '虚假新闻 (1)'], zero_division=0))
#
# print("\n✅ 全指标测试完成！请将控制台表格直接复制到论文中。")
import os
import torch
import json
import random
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset_utils import CFNDDataset, transform
# 导入基础版的 EANN 模型
from models_EANN import EANN

# ================= 1. 基础配置 =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_path = "../CFND_dataset"
test_csv = "test_data.csv"  # 确保你有这个测试集文件
num_events = 50

# 我们要测试的缺失率阶梯
missing_rates = [0.0, 0.2, 0.5, 0.8, 1.0]

# 加载词表
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

print("⏳ 正在准备测试环境...")
# 实例化测试集 (注意：这里的 missing_rate 初始必须设为 0，缺失逻辑我们在测试循环里手动控制，以便精准对比)
test_dataset = CFNDDataset(os.path.join(root_path, test_csv), root_path, vocab, transform=transform, missing_rate=0.0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ================= 2. 加载最强王者：特征蒸馏模型 =================
model = EANN(vocab_size=len(vocab), num_events=num_events).to(device)

# ⚠️ 替换为你刚刚训练出来的最牛的权重文件名
weights_path = "eann_distilled_feature_best.pth"

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"✅ 成功加载知识蒸馏模型权重: {weights_path}")
else:
    print(f"❌ 找不到权重文件 {weights_path}！请检查路径。")
    exit()


# ================= 3. 核心缺失评估函数 =================
def evaluate_missing_scenario(model, dataloader, missing_rate, drop_target):
    """
    评估模型在特定缺失场景下的表现
    :param drop_target: 'text' (模拟文本缺失), 'image' (模拟图像缺失), 'none' (完整数据)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # 注意：这里我们接收了 6 个返回值，虽然最后两个 flag 我们测试时不用
        for texts, images, labels, _, _, _ in dataloader:

            # 使用 clone() 防止在原地修改张量影响下一个 epoch 的测试
            curr_texts = texts.clone().to(device)
            curr_images = images.clone().to(device)
            labels = labels.to(device)
            batch_size = curr_texts.size(0)

            # 💡 核心：手动且精准地模拟模态缺失
            if missing_rate > 0.0:
                for i in range(batch_size):
                    if random.random() < missing_rate:
                        if drop_target == 'text':
                            curr_texts[i] = torch.zeros_like(curr_texts[i])
                        elif drop_target == 'image':
                            curr_images[i] = torch.zeros_like(curr_images[i])

            # 前向推理 (测试阶段不需要对抗训练，alpha=0)
            # 基础 EANN 返回 4 个值：fake_out, event_out, text_feat, vis_feat
            fake_out, _, _, _ = model(curr_texts, curr_images, alpha=0.0)

            # 取概率最大的类别作为预测结果
            _, preds = torch.max(fake_out, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算详细指标
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return acc, prec, rec, f1


# ================= 4. 执行多场景拉网式排查 =================
print("\n🚀 开始【知识蒸馏网络】极限压力测试...")
print("=" * 90)
print(
    f"{'缺失率':<8} | {'缺失目标':<12} | {'准确率 (Acc)':<12} | {'精确率 (Pre)':<12} | {'召回率 (Rec)':<12} | {'F1 分数':<12}")
print("-" * 90)

# 1. 黄金基准线：0% 缺失
acc, prec, rec, f1 = evaluate_missing_scenario(model, test_loader, 0.0, 'none')
print(
    f"{'0%':<8} | {'完整图文':<12} | {acc * 100:>10.2f}% | {prec * 100:>10.2f}% | {rec * 100:>10.2f}% | {f1 * 100:>10.2f}%")
print("-" * 90)

results_summary = {"text": {}, "image": {}}

# 2. 逐步增加缺失率进行测试
for rate in missing_rates[1:]:  # 跳过 0%
    # 模拟文本缺失
    acc, prec, rec, f1 = evaluate_missing_scenario(model, test_loader, rate, 'text')
    results_summary["text"][rate] = acc
    print(
        f"{rate * 100:>6.0f}%   | {'仅丢失文本':<12} | {acc * 100:>10.2f}% | {prec * 100:>10.2f}% | {rec * 100:>10.2f}% | {f1 * 100:>10.2f}%")

    # 模拟图像缺失
    acc, prec, rec, f1 = evaluate_missing_scenario(model, test_loader, rate, 'image')
    results_summary["image"][rate] = acc
    print(
        f"{rate * 100:>6.0f}%   | {'仅丢失图像':<12} | {acc * 100:>10.2f}% | {prec * 100:>10.2f}% | {rec * 100:>10.2f}% | {f1 * 100:>10.2f}%")
    print("-" * 90)

print("\n🎉 测试圆满结束！请将上述数据填入你的毕业论文表格中。")