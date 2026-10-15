import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Function


# ----------------- 1. 核心：梯度反转层 (GRL) -----------------
# 论文提到使用 GRL 来实现 minimax 游戏 [cite: 336, 339]
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时将梯度乘以 -alpha [cite: 337]
        output = grad_output.neg() * ctx.alpha
        return output, None


# ----------------- 2. 文本特征提取器 (Text-CNN) -----------------
# 采用论文参数：filters=20, sizes=[1,2,3,4], hidden=32 [cite: 423, 424]
class TextCNN_EANN(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, num_filters=20, filter_sizes=[1, 2, 3, 4], output_dim=32):
        super(TextCNN_EANN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), output_dim)   # 输出32维

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        conv_outs = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        pool_outs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_outs]
        x_cat = torch.cat(pool_outs, dim=1)
        return F.relu(self.fc(x_cat))


# ----------------- 3. 视觉特征提取器 (ResNet-18 优化版) -----------------
class ResNet18_EANN(nn.Module):
    def __init__(self, output_dim=32):
        super(ResNet18_EANN, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False  # 冻结参数 [cite: 257]
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return F.relu(self.fc(x))


# ----------------- 4. 完整的 EANN 模型 -----------------
class EANN(nn.Module):
    def __init__(self, vocab_size, num_events, shared_dim=32):
        super(EANN, self).__init__()

        # A. 多模态特征提取器 (Gf) [cite: 85, 137, 218]
        self.text_extractor = TextCNN_EANN(vocab_size=vocab_size, output_dim=shared_dim)
        self.visual_extractor = ResNet18_EANN(output_dim=shared_dim)

        # B. 假新闻检测器 (Gd) [cite: 85, 137, 268]
        # 论文设定：隐藏层 64 [cite: 424]
        self.detector = nn.Sequential(
            nn.Linear(shared_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 二分类：真/假
        )

        # C. 事件鉴别器 (Ge) [cite: 85, 137, 305]
        # 论文设定：两层全连接，隐藏层分别为 64 和 32 [cite: 424]
        self.event_discriminator = nn.Sequential(
            nn.Linear(shared_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_events)  # K分类：事件标签
        )

    def forward(self, text, image, alpha=1.0):
        # 1. 提取特征并拼接 [cite: 219, 262]
        text_feat = self.text_extractor(text)
        vis_feat = self.visual_extractor(image)
        combined_feat = torch.cat((text_feat, vis_feat), dim=1)  # 64维

        # 2. 预测谣言标签 (Gd 路) [cite: 221]
        fake_output = self.detector(combined_feat)

        # 3. 对抗训练预测事件标签 (Ge 路) [cite: 222, 338]
        # 通过 GRL 翻转梯度
        reverse_feat = ReverseLayerF.apply(combined_feat, alpha)
        event_output = self.event_discriminator(reverse_feat)

        return fake_output, event_output

# ----------------- 5. 教师模型：纯文本教师 -----------------
class TextTeacher(nn.Module):
    def __init__(self, vocab_size, num_classes=2, embed_dim=128, hidden_dim=64):
        super(TextTeacher, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes) # 双向LSTM

    def forward(self, text, image=None): # image参数是为了接口统一，教师模型不使用image
        x = self.embedding(text)
        x, _ = self.lstm(x)
        # 简单的池化
        x = torch.mean(x, dim=1)
        return self.classifier(x)

# ----------------- 6. 教师模型：纯视觉教师 -----------------
class ImageTeacher(nn.Module):
    def __init__(self, num_classes=2, output_dim=32):
        super(ImageTeacher, self).__init__()
        # 使用预训练ResNet提取特征，但这里我们简化，直接用EANN里的视觉提取器逻辑
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # 移除最后的fc层
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(512, num_classes) # ResNet18最后特征是512

    def forward(self, text=None, image=None):
        x = self.features(image)
        x = self.flatten(x)
        return self.classifier(x)