import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
import json
from torch.utils.data import DataLoader
from dataset_utils import CFNDDataset, transform
from models_EANN import EANN

# 1. 基础配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False

# 加载数据和模型
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
test_dataset = CFNDDataset("../CFND_dataset/test_data.csv", "../CFND_dataset", vocab, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)  # 拿一部分数据画图即可

model = EANN(vocab_size=len(vocab), num_events=50).to(device)
model.load_state_dict(torch.load("eann_best_model.pth", map_location=device))
model.eval()

# 2. 提取特征
features = []
event_labels_list = []
rumor_labels_list = []

print("正在提取高维特征...")
with torch.no_grad():
    for i, (texts, images, labels, event_labels) in enumerate(test_loader):
        texts, images = texts.to(device), images.to(device)

        # 提取文本和图像特征，这里对应 models_EANN.py 里的逻辑
        t_feat = model.text_extractor(texts)
        v_feat = model.visual_extractor(images)
        combined_feat = torch.cat((t_feat, v_feat), dim=1)  # 得到64维融合特征

        features.append(combined_feat.cpu().numpy())
        event_labels_list.append(event_labels.numpy())
        rumor_labels_list.append(labels.numpy())

        if i > 10: break  # 画几百个点就够了，太多了看不清

features = np.concatenate(features, axis=0)
events = np.concatenate(event_labels_list, axis=0)
rumors = np.concatenate(rumor_labels_list, axis=0)

# 3. 使用 t-SNE 进行降维
print("正在进行 t-SNE 降维计算 (可能需要几十秒)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
features_2d = tsne.fit_transform(features)

# 4. 绘制特征分布散点图
plt.figure(figsize=(10, 8))
# 为了清晰，我们只挑出前 5 个事件的数据来画图
selected_events = [0, 1, 2, 3, 4]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6']

for idx, event_id in enumerate(selected_events):
    mask = (events == event_id)
    plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                c=colors[idx], label=f'新闻事件 {event_id}', alpha=0.7, edgecolors='w', s=60)

plt.title('EANN 多模态特征空间 t-SNE 可视化\n(不同颜色代表不同新闻事件)', fontsize=16, pad=15)
plt.xlabel('t-SNE 维度 1', fontsize=12)
plt.ylabel('t-SNE 维度 2', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig('eann_tsne_visualization.png', dpi=300, bbox_inches='tight')
print("✅ 可视化图表已保存为 eann_tsne_visualization.png")