import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from experiment_utils import EANN_DIR, detector_logits, get_device, load_vocab, make_loader, set_seed, unpack_batch
from models_EANN import EANN


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            texts, images, labels, _, _ = unpack_batch(batch)
            texts, images, labels = texts.to(device), images.to(device), labels.to(device)
            logits = detector_logits(model, texts, images, alpha=0.0)
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Train the EANN baseline without knowledge distillation.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-events", type=int, default=50)
    parser.add_argument("--missing-rate", type=float, default=0.0)
    parser.add_argument("--output", default="eann_best_model.pth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    vocab = load_vocab()
    train_loader = make_loader("train", vocab, batch_size=args.batch_size, shuffle=True, missing_rate=args.missing_rate)
    val_loader = make_loader("val", vocab, batch_size=args.batch_size, shuffle=False)

    model = EANN(vocab_size=len(vocab), num_events=args.num_events).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    best_acc = 0.0
    output_path = EANN_DIR / args.output

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            texts, images, labels, event_labels, _ = unpack_batch(batch)
            texts, images = texts.to(device), images.to(device)
            labels, event_labels = labels.to(device), event_labels.to(device)

            p = float(batch_idx + (epoch - 1) * len(train_loader)) / args.epochs / len(train_loader)
            alpha_adv = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

            optimizer.zero_grad()
            fake_out, event_out, *_ = model(texts, images, alpha=alpha_adv)
            loss = criterion(fake_out, labels) + criterion(event_out, event_labels)
            loss.backward()
            optimizer.step()

        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}/{args.epochs} | val_acc={val_acc * 100:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"saved {output_path.name} | best_val_acc={best_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
