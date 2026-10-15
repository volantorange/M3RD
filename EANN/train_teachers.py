import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from experiment_utils import EANN_DIR, get_device, load_vocab, make_loader, set_seed, unpack_batch
from teacher_models import ImageTeacher, TextTeacher


def train_teacher(model, name, train_loader, val_loader, device, epochs, lr):
    criterion = nn.CrossEntropyLoss()
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
        print(f"{name} epoch {epoch}/{epochs} | val_acc={acc * 100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)
            print(f"saved {best_path.name} | best_val_acc={best_acc * 100:.2f}%")

    return best_acc


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
    parser = argparse.ArgumentParser(description="Train text and image teachers for KD.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    vocab = load_vocab()
    train_loader = make_loader("train", vocab, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader("val", vocab, batch_size=args.batch_size, shuffle=False)

    print(f"device={device} vocab={len(vocab)}")
    train_teacher(TextTeacher(len(vocab)).to(device), "Text_Teacher", train_loader, val_loader, device, args.epochs, args.lr)
    train_teacher(ImageTeacher().to(device), "Image_Teacher", train_loader, val_loader, device, args.epochs, args.lr)


if __name__ == "__main__":
    main()
