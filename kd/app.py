import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import json
import jieba

# 导入你刚刚拼装好的 EANN 模型
from models_EANN import EANN

# ================= 1. 基础配置与模型加载 =================
device = torch.device("cpu")
num_events = 50  # 根据你 clustering.py 和 Training Loop.py 的设定

# 读取真实词表
try:
    with open("vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    print(f"✅ 成功加载 vocab.json, 词表大小: {vocab_size}")
except Exception as e:
    print(f"❌ 加载 vocab.json 失败: {e}")
    vocab_size = 10000  # 占位防崩

# 实例化 EANN 模型
model = EANN(vocab_size=vocab_size, num_events=num_events).to(device)

# 加载 EANN 的最优权重
weights_path = "eann_best_model.pth"
try:
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"✅ 成功加载 EANN 权重: {weights_path}")
except Exception as e:
    print(f"⚠️ 加载权重失败，请检查文件路径: {e}")

model.eval()  # 切换到评估模式

# ================= 2. 预处理函数 =================
# 图像预处理 (与训练时保持一致)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def text_to_sequence(text, max_len=50):
    """文本转序列函数（同步使用 jieba 分词）"""
    if not text.strip():  # 处理空文本情况
        return torch.zeros((1, max_len), dtype=torch.long)

    words = jieba.lcut(str(text))
    seq = [vocab.get(w, vocab.get('<UNK>')) for w in words]
    if len(seq) < max_len:
        seq += [vocab.get('<PAD>')] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return torch.tensor(seq, dtype=torch.long).unsqueeze(0)  # 增加 batch 维度: (1, seq_len)


# ================= 3. 定义预测逻辑 (包含模态缺失模拟) =================
def predict_fake_news(text, image, missing_mode):
    # 基础校验
    if missing_mode == "完整图文 (Complete)" and (not text or image is None):
        return "⚠️ '完整图文' 模式下，请同时输入文本和上传图片！", None
    if missing_mode == "仅有文本 (图片丢失)" and not text:
        return "⚠️ 请输入文本！", None
    if missing_mode == "仅有图片 (文本丢失)" and image is None:
        return "⚠️ 请上传图片！", None

    try:
        # 1. 文本处理
        if missing_mode == "仅有图片 (文本丢失)":
            text_tensor = torch.zeros((1, 50), dtype=torch.long).to(device)  # 模拟文本缺失置零
        else:
            text_tensor = text_to_sequence(text).to(device)

        # 2. 图像处理
        if missing_mode == "仅有文本 (图片丢失)" or image is None:
            image_tensor = torch.zeros((1, 3, 224, 224), dtype=torch.float32).to(device)  # 模拟图像缺失置零
        else:
            image_tensor = transform(image).unsqueeze(0).to(device)

        # 3. 模型推理
        with torch.no_grad():
            # EANN 的前向传播需要传入 alpha。推理阶段不需要对抗，因此 alpha=0.0
            outputs, _ = model(text_tensor, image_tensor, alpha=0.0)
            probs = F.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

        # 解析结果 (注意：这里的 0 和 1 要根据你数据集中真实的标签对应关系来写！)
        labels = {1: "🛑 谣言 (Fake News)", 0: "✅ 真实 (True News)"}
        result_text = labels.get(pred_class, "未知")

        return f"检测结果：{result_text}\n置信度：{confidence * 100:.2f}%", {labels[0]: probs[0][0].item(),
                                                                           labels[1]: probs[0][1].item()}

    except Exception as e:
        return f"预测过程中发生错误: {str(e)}", None


# ================= 4. 构建 Gradio 界面 =================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🕵️‍♂️ 面向模态缺失的多模态谣言检测系统 (EANN版)")
    gr.Markdown("""
    本系统用于演示毕业设计成果。系统基于 **EANN (事件对抗神经网络)** 构建。
    💡 **特色功能**：您可以利用左侧的 **“场景模拟”** 单选框，切断某一模态的数据输入，以此观察模型在“模态缺失”极端环境下的容错与鲁棒能力！
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # 新增：模态缺失模拟器
            missing_mode = gr.Radio(
                choices=["完整图文 (Complete)", "仅有文本 (图片丢失)", "仅有图片 (文本丢失)"],
                value="完整图文 (Complete)",
                label="⚙️ 测试场景模拟 (Modality Drop Simulator)"
            )

            text_input = gr.Textbox(lines=4, placeholder="在此输入新闻文本...", label="📝 新闻文本 (Text)")
            image_input = gr.Image(type="pil", label="🖼️ 新闻配图 (Image)")
            submit_btn = gr.Button("🚀 执行多模态检测", variant="primary")

        with gr.Column(scale=1):
            text_output = gr.Textbox(label="📊 判定结果", lines=2)
            label_output = gr.Label(label="🧠 模型置信度分布")

            gr.Markdown("""
            ### 📝 系统说明：
            1. **完整模式**：提取文本的 TextCNN 特征与图像的 ResNet18 特征进行拼接分类。
            2. **缺失模式**：当某模态丢失时，系统将动态用零张量（Zero Tensor）进行底层特征空间插补，模拟传感器故障或数据不全场景。
            """)

    # 绑定按钮点击事件
    submit_btn.click(
        fn=predict_fake_news,
        inputs=[text_input, image_input, missing_mode],
        outputs=[text_output, label_output]
    )

if __name__ == "__main__":
    # 启动界面
    demo.launch(share=False)