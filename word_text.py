import pandas as pd
import jieba
import json
from collections import Counter

# 加载训练集
TRAIN_CSV = "CFND_dataset/train_data.csv"
df = pd.read_csv(TRAIN_CSV)

# 分词
words_count = Counter()
for t in df['title'].astype(str):
    words = jieba.lcut(t)
    words_count.update(words)

# 构建字典
vocab = {'<PAD>': 0, '<UNK>': 1}  # 0用于补充长度，1表示未知词
idx = 2  # 编号
# 过滤低频词并编号
for word, count in words_count.items():
    if count >= 2:   # 低频率词不要
        vocab[word] = idx
        idx += 1

# 保存文件
with open("vocab1.json", 'w', encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=4)
