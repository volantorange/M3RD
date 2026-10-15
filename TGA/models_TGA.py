import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.autograd import Function


# ----------------- 1. 梯度反转层 (GRL) -----------------
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# ----------------- 2. 文本特征提取器 -----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class TextTransformer_Extractor(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=8, num_layers=4, output_dim=32, max_len=50):
        super(TextTransformer_Extractor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len)
        encoder_layers = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.pos_encoder(emb)
        out = self.transformer(emb)
        out_pooled = out.mean(dim=1)
        return F.relu(self.fc(out_pooled))


# ----------------- 3. 视觉特征提取器 -----------------
class ViT_Extractor(nn.Module):
    def __init__(self, output_dim=32):
        super(ViT_Extractor, self).__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.heads = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return F.relu(self.vit(x))


# ----------------- 4. TGA 注意力融合模块 -----------------
class TGAModalityAttention(nn.Module):
    def __init__(self, shared_dim=32):
        super(TGAModalityAttention, self).__init__()
        # TGA 论文中的双层前馈网络计算注意力权重 [cite: 220, 221]
        self.w1 = nn.Linear(shared_dim, shared_dim)
        self.w2 = nn.Linear(shared_dim, 1)

    def forward(self, text_feat, vis_feat):
        # 1. 计算各自的注意力得分 (tanh 激活)
        score_t = self.w2(torch.tanh(self.w1(text_feat)))
        score_v = self.w2(torch.tanh(self.w1(vis_feat)))

        # 2. Softmax 归一化
        scores = torch.cat([score_t, score_v], dim=1)
        weights = torch.softmax(scores, dim=1)

        weight_t = weights[:, 0].unsqueeze(1)
        weight_v = weights[:, 1].unsqueeze(1)

        # 3. 加权特征
        attended_text = text_feat * weight_t
        attended_vis = vis_feat * weight_v

        # 为兼容后续的 64 维检测器，采用拼接输出
        fused_feat = torch.cat((attended_text, attended_vis), dim=1)

        return fused_feat, weight_t, weight_v


# ----------------- 5. 完整的 TGA 学生模型 -----------------
class TGA(nn.Module):
    def __init__(self, vocab_size, num_events, shared_dim=32, mapping_dim=128):
        super(TGA, self).__init__()

        # A. 特征提取器
        self.text_extractor = TextTransformer_Extractor(vocab_size=vocab_size, output_dim=shared_dim)
        self.visual_extractor = ViT_Extractor(output_dim=shared_dim)

        # B. 晚期融合机制 (Late Fusion) [cite: 216]
        self.attention_fusion = TGAModalityAttention(shared_dim=shared_dim)

        # C. 映射网络：用于计算特征相似度 [cite: 247, 288]
        # 论文设定映射后的维度为 128
        self.mapping_t = nn.Sequential(
            nn.Linear(shared_dim, 64),
            nn.ReLU(),
            nn.Linear(64, mapping_dim)
        )
        self.mapping_v = nn.Sequential(
            nn.Linear(shared_dim, 64),
            nn.ReLU(),
            nn.Linear(64, mapping_dim)
        )

        # D. 假新闻检测器 (Gd)
        self.detector = nn.Sequential(
            nn.Linear(shared_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        # E. 事件鉴别器 (Ge) - 保持多任务对抗特性
        self.event_discriminator = nn.Sequential(
            nn.Linear(shared_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_events)
        )

    # 关键修改：前向传播增加 text_missing_flag 和 image_missing_flag
    def forward(self, text, image, text_missing_flag=None, image_missing_flag=None, grl_alpha=1.0):
        # 1. 单模态特征提取
        text_feat = self.text_extractor(text)
        vis_feat = self.visual_extractor(image)

        # 2. 注意力机制动态融合
        fused_feat, weight_t, weight_v = self.attention_fusion(text_feat, vis_feat)

        # 3. 基础判别分支
        fake_logits = self.detector(fused_feat)

        # 4. 对抗事件分支
        reverse_feat = ReverseLayerF.apply(fused_feat, grl_alpha)
        event_output = self.event_discriminator(reverse_feat)

        # ---------------- 核心创新点：缺失感知的相似度门控 ----------------
        if text_missing_flag is not None and image_missing_flag is not None:
            # 找到那些图文都完整的样本（缺失标记为 0）
            complete_idx = (text_missing_flag == 0) & (image_missing_flag == 0)

            if complete_idx.any():
                # 仅对完整的样本计算特征相似度
                mapped_t = self.mapping_t(text_feat[complete_idx])
                mapped_v = self.mapping_v(vis_feat[complete_idx])

                # 计算欧氏距离 [cite: 245]
                distance = torch.norm(mapped_t - mapped_v, dim=1)

                # TGA 论文超参数设定 [cite: 254]
                sim_beta = 0.65
                sim_alpha = 0.1

                # 如果距离大于阈值 β，激活惩罚项 [cite: 253]
                penalty_mask = (distance > sim_beta).float()
                penalty = sim_alpha * distance * penalty_mask

                # 假设假新闻标签为 1，我们将惩罚值加到 fake_logits 的假新闻概率上，提高模型预测为假的倾向
                fake_logits[complete_idx, 1] += penalty
        # -----------------------------------------------------------------

        return fake_logits, event_output, text_feat, vis_feat