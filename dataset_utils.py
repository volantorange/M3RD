import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import jieba

# 数据集类
class CFNDDataset(Dataset):
    def __init__(self, csv_file, root_dir, vocab, max_len=50, transform=None):
        """
        :param csv_file: csv文件路径 (例如 train_data.csv)
        :param root_dir: 根目录路径 (即 CFND_dataset)
        :param vocab: 词表字典 {word: index}
        :param max_len: 文本最大长度
        :param transform: 图像的预处理操作
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.vocab = vocab
        self.max_len = max_len
        self.transform = transform

    # 总样本量
    def __len__(self):
        return len(self.data_frame)

    # 文本转数字序列
    def text_to_numsequence(self, text):
        words = jieba.lcut(text)   # 分词

        seq = []
        # 遍历分词后的每一个词语
        for w in words:
            # 查词w，如果找不到用1替代
            number = self.vocab.get(w, self.vocab.get('<UNK>'))
            seq.append(number)

        # 截断或填充
        if len(seq) < self.max_len:   # 补0
            seq.extend([self.vocab.get('<PAD>')] * (self.max_len - len(seq)))
        else:   # 截断
            seq = seq[:self.max_len]
        # 返回长整型张量
        return torch.tensor(seq, dtype=torch.long)

    def __getitem__(self, idx):
        # 读取文本并转化为数字序列
        text = self.data_frame.iloc[idx]['title']
        text_seq = self.text_to_numsequence(text)

        # 读取image
        img_name = self.data_frame.iloc[idx]['image']
        # 拼接绝对路径: CFND_dataset/images/fake_news_image/xxx.jpg
        img_path = os.path.join(self.root_dir, img_name)

        # 确保转换为RGB模式，若出现问题，用黑色展位图
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # 预处理
        if self.transform:
            image = self.transform(image)

        # 真假标签
        label = int(self.data_frame.iloc[idx]['label'])
        l = torch.tensor(label, dtype=torch.long)

        # 返回
        return text_seq, image, l

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])






# class CFNDDataset(Dataset):
#     def __init__(self, csv_file, root_dir, vocab, max_len=50, transform=None):
#         """
#         :param csv_file: csv文件路径 (例如 train_data.csv)
#         :param root_dir: 根目录路径 (即 CFND_dataset)
#         :param vocab: 词表字典 {word: index}
#         :param max_len: 文本最大长度
#         :param transform: 图像的预处理操作
#         """
#         self.data_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.vocab = vocab
#         self.max_len = max_len
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.data_frame)
#
#     def text_to_sequence(self, text):
#         # 【修改点】：使用 jieba 进行真正的中文分词
#         words = jieba.lcut(str(text))
#
#         seq = [self.vocab.get(w, self.vocab.get('<UNK>')) for w in words]
#         # 截断或填充
#         if len(seq) < self.max_len:
#             seq += [self.vocab.get('<PAD>')] * (self.max_len - len(seq))
#         else:
#             seq = seq[:self.max_len]
#         return torch.tensor(seq, dtype=torch.long)
#
#     def __getitem__(self, idx):
#         # 读取文本
#         text = self.data_frame.iloc[idx]['title']
#         text_seq = self.text_to_sequence(text)
#
#         # 读取图像
#         img_name = self.data_frame.iloc[idx]['image']
#         # 拼接绝对路径: CFND_dataset/images/fake_news_image/xxx.jpg
#         img_path = os.path.join(self.root_dir, img_name)
#
#         try:
#             image = Image.open(img_path).convert('RGB')
#         except Exception as e:
#             # 如果图片损坏或丢失，使用黑色占位图防止训练崩溃
#             image = Image.new('RGB', (224, 224), (0, 0, 0))
#
#         if self.transform:
#             image = self.transform(image)
#
#         # 读取标签
#         label = int(self.data_frame.iloc[idx]['label'])
#
#         return text_seq, image, torch.tensor(label, dtype=torch.long)
#
#
# # VGG19 标准的图像预处理
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])