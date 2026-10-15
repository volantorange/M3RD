import torch
import torch.nn as nn
import torch.nn.functional as fun
import torchvision.models as models


# TextCNN
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=100, filter_sizes=[3, 4, 5]):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 卷积核尺度
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)  # 丢弃

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)

        # 卷积 + ReLU + 最大池化
        x = [fun.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [fun.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        # 拼接多个特征
        x = torch.cat(x, 1)  # shape: (batch_size, num_filters * len(filter_sizes))
        return self.dropout(x)


class ResNet18Extractor(nn.Module):
    def __init__(self, image_out_dim=256):
        super(ResNet18Extractor, self).__init__()
        # 预训练的 ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 冻结参数
        for param in resnet.parameters():
            param.requires_grad = False

        # 剥离 ResNet 原本用于1000分类的最后一层，只保留特征提取骨干
        # resnet.children() 包含了网络的所有层，[:-1] 表示去掉最后的全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()

        # ResNet18 提取出的特征固定是 512 维，我们将其映射到你设置的 256 维
        self.fc = nn.Linear(512, image_out_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x shape: (batch_size, 3, 224, 224)
        x = self.features(x)  # 输出 shape: (batch_size, 512, 1, 1)
        x = self.flatten(x)  # 输出 shape: (batch_size, 512)
        x = fun.relu(self.fc(x))
        return self.dropout(x)




class MultimodalDetector(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, text_filters=100, image_out_dim=256, hidden_dim=128):
        super(MultimodalDetector, self).__init__()

        self.text_net = TextCNN(vocab_size, embed_dim, text_filters)
        # self.image_net = VGG19Extractor(image_out_dim)
        self.image_net = ResNet18Extractor(image_out_dim)

        # TextCNN 输出维度为 text_filters * 3 (因为有3个不同大小的卷积核)
        text_out_dim = text_filters * 3

        # 融合层
        self.fc1 = nn.Linear(text_out_dim + image_out_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # 二分类: 真假
        self.dropout = nn.Dropout(0.5)

    def forward(self, text, image):
        text_feat = self.text_net(text)
        image_feat = self.image_net(image)

        # 特征拼接 (Concat)
        combined_feat = torch.cat((text_feat, image_feat), dim=1)

        # 分类器
        x = fun.relu(self.fc1(combined_feat))
        x = self.dropout(x)
        out = self.fc2(x)
        return out