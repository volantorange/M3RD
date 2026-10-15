import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import build_model
from utils import get_device, load_vocab, make_loader, metrics_from_predictions, set_seed


BASE_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def forward_logits(model_name, model, texts, images, alpha=0.0):
    if model_name == "eann":
        fake_out, _, _, _ = model(texts, images, alpha=alpha)
        return fake_out
    return model(texts, images)


def evaluate(model_name, model, loader, device, max_batches=None):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (texts, images, labels, _) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            texts, images = texts.to(device), images.to(device)
            logits = forward_logits(model_name, model, texts, images, alpha=0.0)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().tolist())
    return metrics_from_predictions(all_labels, all_preds)


def train_one_model(model_name, args, loaders, vocab_size, device, output_dir):
    model = build_model(model_name, vocab_size=vocab_size, num_events=args.num_events).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_val_acc = 0.0
    best_state = None

    train_loader, val_loader, test_loader = loaders
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0

        for batch_idx, (texts, images, labels, event_labels) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break

            texts, images = texts.to(device), images.to(device)
            labels, event_labels = labels.to(device), event_labels.to(device)

            optimizer.zero_grad()
            if model_name == "eann":
                p = float(batch_idx + (epoch - 1) * len(train_loader)) / args.epochs / len(train_loader)
                alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
                fake_out, event_out, _, _ = model(texts, images, alpha=alpha)
                loss = criterion(fake_out, labels) + args.event_loss_weight * criterion(event_out, event_labels)
            else:
                fake_out = model(texts, images)
                loss = criterion(fake_out, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batches += 1

        val_metrics = evaluate(model_name, model, val_loader, device, args.max_eval_batches)
        print(
            f"{model_name} epoch {epoch}/{args.epochs} | "
            f"loss={running_loss / max(batches, 1):.4f} "
            f"val_acc={val_metrics['accuracy'] * 100:.2f}% "
            f"val_f1={val_metrics['f1'] * 100:.2f}%"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model_name, model, test_loader, device, args.max_eval_batches)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"{model_name}_event_generalization_best.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(
        f"{model_name} test | acc={test_metrics['accuracy'] * 100:.2f}% "
        f"precision={test_metrics['precision'] * 100:.2f}% "
        f"recall={test_metrics['recall'] * 100:.2f}% "
        f"f1={test_metrics['f1'] * 100:.2f}% -> {ckpt_path}"
    )
    return test_metrics


def save_results(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "event_generalization_results.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "accuracy", "precision", "recall", "f1"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"results -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare EANN and plain multimodal model on held-out events.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-events", type=int, default=10)
    parser.add_argument("--event-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", default="plain,eann", help="Comma-separated: plain,eann")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    vocab = load_vocab()
    loaders = (
        make_loader("train", vocab, args.batch_size, shuffle=True),
        make_loader("val", vocab, args.batch_size, shuffle=False),
        make_loader("test", vocab, args.batch_size, shuffle=False),
    )
    is_smoke = args.max_train_batches is not None or args.max_eval_batches is not None
    output_dir = BASE_OUTPUT_DIR / "smoke" if is_smoke else BASE_OUTPUT_DIR

    print(f"device={device} vocab={len(vocab)} train_events=2,9,5,6 val_event=3 test_events=0,1,4,7,8")
    rows = []
    for model_name in [item.strip() for item in args.models.split(",") if item.strip()]:
        metrics = train_one_model(model_name, args, loaders, len(vocab), device, output_dir)
        rows.append({"model": model_name, **metrics})
    save_results(rows, output_dir)


if __name__ == "__main__":
    main()
