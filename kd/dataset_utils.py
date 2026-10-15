import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import jieba
from torchvision import transforms
import random

class CFNDDataset(Dataset):
    def __init__(self, csv_file, root_dir, vocab, max_len=50, transform=None, missing_rate=0.0):
        """
        :param csv_file: CSV文件路径 (num, title, image, label, [event])
        :param root_dir: 数据集根目录 (CFND_dataset)
        :param vocab: 词表字典
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.vocab = vocab
        self.max_len = max_len
        self.transform = transform
        self.missing_rate = missing_rate

        # 没有even列默认为0
        if 'event' not in self.data_frame.columns:
            self.data_frame['event'] = 0

    def __len__(self):
        return len(self.data_frame)

    def text_to_sequence(self, text):
        words = jieba.lcut(str(text))
        seq = [self.vocab.get(w, self.vocab.get('<UNK>')) for w in words]
        if len(seq) < self.max_len:
            seq += [self.vocab.get('<PAD>')] * (self.max_len - len(seq))
        else:
            seq = seq[:self.max_len]
        return torch.tensor(seq, dtype=torch.long)

    def __getitem__(self, idx):
        # 文本
        text = self.data_frame.iloc[idx]['title']
        text_seq = self.text_to_sequence(text)

        # 图像
        img_relative_path = self.data_frame.iloc[idx]['image']
        img_path = os.path.join(self.root_dir, img_relative_path)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))  # 占位图

        if self.transform:
            image = self.transform(image)

        # ================= 核心：模拟模态缺失 =================
        if self.missing_rate > 0.0:
            if random.random() < self.missing_rate:
                # 随机选择丢弃文本还是丢弃图像 (各 50% 概率)
                if random.random() < 0.5:
                    # 丢弃文本：将序列全部替换为 <PAD> (通常为 0)
                    text_seq = torch.zeros_like(text_seq)
                else:
                    # 丢弃图像：替换为全黑的零张量
                    image = torch.zeros_like(image)
        # ======================================================

        # 3. 标签
        label = int(self.data_frame.iloc[idx]['label'])
        event_label = int(self.data_frame.iloc[idx]['event'])  # 获取事件标签

        return text_seq, image, torch.tensor(label, dtype=torch.long), torch.tensor(event_label, dtype=torch.long)

    # 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
