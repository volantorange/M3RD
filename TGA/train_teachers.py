import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from experiment_utils import EANN_DIR, get_device, load_vocab, make_loader, set_seed, unpack_batch
# 修改点 1：导入全新的 Transformer 教师架构
from teacher_models import ImageTeacher, TextTeacher


def train_teacher(model, name, train_loader, val_loader, device, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    # 过滤掉未激活梯度的参数（因为 ViT 的主干网络权重被我们冻结了，只训练分类头）
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    best_acc = 0.0
    best_path = EANN_DIR / f"{name}_best.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            texts, images, labels, _, _ = unpack_batch(batch)
            labels = labels.to(device)

            optimizer.zero_grad()
            if name == "Text_Teacher":
                logits, _ = model(texts.to(device))
            else:
                logits, _ = model(images.to(device))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        acc = evaluate_teacher(model, name, val_loader, device)
        print(f"✨ {name} epoch {epoch}/{epochs} | val_acc={acc * 100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)
            print(f"💾 成功保存更佳权重 -> {best_path.name} | best_val_acc={best_acc * 100:.2f}%")


def evaluate_teacher(model, name, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            texts, images, labels, _, _ = unpack_batch(batch)
            labels = labels.to(device)
            if name == "Text_Teacher":
                logits, _ = model(texts.to(device))
            else:
                logits, _ = model(images.to(device))
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Train Transformer-based text and image teachers for KD.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    vocab = load_vocab()

    train_loader = make_loader("train", vocab, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader("val", vocab, batch_size=args.batch_size, shuffle=False)

    print(f"🎬 开始训练 Transformer 教师模型群 | 硬件环境={device} 词表规模={len(vocab)}")
    print(f"设定参数：Epochs={args.epochs}, Learning Rate={args.lr}")

    # =======================================================================
    # 修改点 1：将文本教师的训练彻底注释掉（因为你已经跑出 89.56% 的最佳模型了）
    # =======================================================================
    # print("\n[Phase 1] 正在训练文本 Transformer 专家教师...")
    # train_teacher(TextTeacher(len(vocab)).to(device), "Text_Teacher", train_loader, val_loader, device, args.epochs, args.lr)

    # =======================================================================
    # 修改点 2：保持图像教师的训练激活，它会自动接收你从命令行传进来的 args.epochs 和 args.lr
    # =======================================================================
    print("\n[Phase 2] 正在训练图像 Vision Transformer (ViT) 专家教师...")
    train_teacher(ImageTeacher().to(device), "Image_Teacher", train_loader, val_loader, device, args.epochs, args.lr)

    print("\n🎉 图像 Transformer 专家教师模型训练并保存完毕！可以正式开启 train_distillation.py 了。")


if __name__ == "__main__":
    main()