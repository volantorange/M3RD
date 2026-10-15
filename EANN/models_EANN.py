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
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
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
        # combined_feat = torch.cat((text_feat, vis_feat), dim=1)  # 64维
        #
        # # 2. 预测谣言标签 (Gd 路) [cite: 221]
        # fake_output = self.detector(combined_feat)
        #
        # # 3. 对抗训练预测事件标签 (Ge 路) [cite: 222, 338]
        # # 通过 GRL 翻转梯度
        # reverse_feat = ReverseLayerF.apply(combined_feat, alpha)
        # event_output = self.event_discriminator(reverse_feat)
        fused_feat = torch.cat((text_feat, vis_feat), dim=1)

        fake_output = self.detector(fused_feat)

        reverse_feat = ReverseLayerF.apply(fused_feat, alpha)
        event_output = self.event_discriminator(reverse_feat)

        return fake_output, event_output, text_feat, vis_feat


# class ModalityAttention(nn.Module):
#     def __init__(self, shared_dim=32):
#         """
#         模态注意力融合模块
#         :param shared_dim: 单个模态的特征维度 (EANN中对齐为 32)
#         """
#         super(ModalityAttention, self).__init__()
#         # 注意力计算网络：输入拼接后的特征 (64维)，输出2维 (文本和图像的权重)
#         self.attention_layer = nn.Sequential(
#             nn.Linear(shared_dim * 2, shared_dim),
#             nn.Tanh(),
#             nn.Linear(shared_dim, 2)
#         )
#
#     def forward(self, text_feat, vis_feat):
#         # 1. 拼接特征用于计算注意力分布
#         # combined shape: (batch_size, 64)
#         combined = torch.cat((text_feat, vis_feat), dim=1)
#
#         # 2. 计算 softmax 权重
#         # attn_weights shape: (batch_size, 2)，保证两个权重相加为 1
#         attn_weights = torch.softmax(self.attention_layer(combined), dim=1)
#
#         # 3. 分离出文本权重和图像权重并扩展维度以匹配特征
#         weight_t = attn_weights[:, 0].unsqueeze(1)  # (batch_size, 1)
#         weight_v = attn_weights[:, 1].unsqueeze(1)  # (batch_size, 1)
#
#         # 4. 特征加权
#         attended_text = text_feat * weight_t
#         attended_vis = vis_feat * weight_v
#
#         # 5. 加权后再次拼接，保持 64 维输出，无缝接入后续网络
#         fused_feat = torch.cat((attended_text, attended_vis), dim=1)
#
#         # 返回融合后的特征，以及权重（方便后续做可视化分析）
#         return fused_feat, weight_t, weight_v

# class EANN_Attention(nn.Module):
#     def __init__(self, vocab_size, num_events, shared_dim=32):
#         super(EANN_Attention, self).__init__()
#
#         # A. 模态特征提取器 (保持不变)
#         self.text_extractor = TextCNN_EANN(vocab_size=vocab_size, output_dim=shared_dim)
#         self.visual_extractor = ResNet18_EANN(output_dim=shared_dim)
#
#         # B. 新增：模态注意力融合层
#         self.attention_fusion = ModalityAttention(shared_dim=shared_dim)
#
#         # C. 假新闻检测器 (Gd) (保持 64 -> 64 -> 2)
#         self.detector = nn.Sequential(
#             nn.Linear(shared_dim * 2, 64),
#             nn.ReLU(),
#             nn.Linear(64, 2)
#         )
#
#         # D. 事件鉴别器 (Ge) (保持 64 -> 64 -> 32 -> num_events)
#         self.event_discriminator = nn.Sequential(
#             nn.Linear(shared_dim * 2, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, num_events)
#         )
#
#     def forward(self, text, image, alpha=1.0):
#         # 1. 单模态特征提取
#         text_feat = self.text_extractor(text)
#         vis_feat = self.visual_extractor(image)
#
#         # 2. 注意力机制动态融合 (取代原来的 torch.cat)
#         # fused_feat 依然是 64 维，但内部信息已经被重新分配了权重
#         fused_feat, weight_t, weight_v = self.attention_fusion(text_feat, vis_feat)
#
#         # 3. 预测谣言标签 (Gd 路)
#         fake_output = self.detector(fused_feat)
#
#         # 4. 对抗训练预测事件标签 (Ge 路)
#         reverse_feat = ReverseLayerF.apply(fused_feat, alpha)
#         event_output = self.event_discriminator(reverse_feat)
#
#         # 这里除了返回损失计算需要的输出，也可以把权重返回，后续写论文时可以画图！
#         return fake_output, event_output, weight_t, weight_v
