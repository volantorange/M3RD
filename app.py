import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from models import MultimodalDetector  # 导入你写的模型
import json
import jieba

# 设备
device = torch.device("cpu")

# 读取真实词表
try:
    with open("vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
except Exception as e:
    print(f"❌ 加载 vocab.json 失败: {e}")
    vocab_size = 10000 # 占位防崩

# 实例化模型
model = MultimodalDetector(vocab_size=vocab_size).to(device)

# 加载的最优权重
model.load_state_dict(torch.load("best_model.pth", map_location=device))

model.eval() # 切换到评估模式

# 图像预处理 (与训练时保持一致)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def text_to_sequence(text, max_len=50):
    """文本转序列函数（已同步使用 jieba 分词）"""
    # 使用 jieba 进行中文分词
    words = jieba.lcut(str(text))
    seq = [vocab.get(w, vocab.get('<UNK>')) for w in words]
    if len(seq) < max_len:
        seq += [vocab.get('<PAD>')] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return torch.tensor(seq, dtype=torch.long).unsqueeze(0)  # 增加 batch 维度: (1, seq_len)


# ================= 3. 定义预测逻辑 =================
def predict_fake_news(text, image):
    if not text or image is None:
        return "⚠️ 请同时输入文本和上传图片！", None

    try:
        # 预处理输入
        text_tensor = text_to_sequence(text).to(device)
        image_tensor = transform(image).unsqueeze(0).to(device)  # 增加 batch 维度

        # 模型推理
        with torch.no_grad():
            outputs = model(text_tensor, image_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

        # 解析结果 (假设 0: 谣言, 1: 真实，请根据你的数据集实际标签调整)
        labels = {1: "🛑 谣言 (Fake News)", 0: "✅ 真实 (True News)"}
        result_text = labels.get(pred_class, "未知")

        # 返回结果文本和置信度供图表显示
        return f"检测结果：{result_text}\n置信度：{confidence * 100:.2f}%", {labels[0]: probs[0][0].item(),
                                                                           labels[1]: probs[0][1].item()}

    except Exception as e:
        return f"预测过程中发生错误: {str(e)}", None


# ================= 4. 构建 Gradio 界面 =================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕵️‍♂️ 多模态谣言检测系统")
    gr.Markdown("输入新闻的**标题/内容**并上传**相关配图**，AI 将综合分析图文特征以判断其真实性。")

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(lines=4, placeholder="在此输入新闻文本...", label="新闻文本")
            image_input = gr.Image(type="pil", label="新闻配图")
            submit_btn = gr.Button("开始检测", variant="primary")

        with gr.Column(scale=1):
            text_output = gr.Textbox(label="判定结果")
            label_output = gr.Label(label="模型置信度分布")

    # 绑定按钮点击事件
    submit_btn.click(
        fn=predict_fake_news,
        inputs=[text_input, image_input],
        outputs=[text_output, label_output]
    )

if __name__ == "__main__":
    # 启动界面
    demo.launch(share=False)

