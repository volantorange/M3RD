'''聚类算法：处理原始数据，对数据进行划分，用于实现EANN'''
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os

# 1. 配置文件路径
root_path = "../CFND_dataset"
csv_files = [ "val_data.csv", "test_data.csv"]  # 假设你有这三个文件
num_events = 10  # 假设我们将新闻聚类为50个大事件
all_texts = []
data_frames = {}

# 读取所有数据并进行分词
for file in csv_files:
    file_path = os.path.join(root_path, file)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        data_frames[file] = df
        # 对标题进行分词，用空格拼成字符串供 TF-IDF 使用
        processed_texts = [" ".join(jieba.lcut(str(title))) for title in df['title']]
        all_texts.extend(processed_texts)

# 提取 TF-IDF 特征
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(all_texts)

# 使用 KMeans 进行聚类
kmeans = KMeans(n_clusters=num_events, random_state=42)
all_labels = kmeans.fit_predict(X)

# 将聚类结果分配回CSV并保存
start_idx = 0
for file, df in data_frames.items():
    end_idx = start_idx + len(df)
    # 截取对应的聚类标签
    df['event'] = all_labels[start_idx:end_idx]
    start_idx = end_idx

    # 保存回原文件（或者你可以加个后缀比如 train_data_with_event.csv）
    save_path = os.path.join(root_path, file)
    df.to_csv(save_path, index=False)
    print(f"更新: {save_path}")
