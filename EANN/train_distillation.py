import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from experiment_utils import EANN_DIR, get_device, load_vocab, make_loader, set_seed, unpack_batch
from models_EANN import EANN
from teacher_models import ImageTeacher, TextTeacher


def kd_loss(student_logits, teacher_logits, temperature):
    teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature ** 2)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            texts, images, labels, _, _ = unpack_batch(batch)
            texts, images, labels = texts.to(device), images.to(device), labels.to(device)
            fake_out, _, _, _ = model(texts, images, alpha=0.0)
            preds = fake_out.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total


def load_teacher_weights(text_teacher, image_teacher, device):
    text_path = EANN_DIR / "Text_Teacher_best.pth"
    image_path = EANN_DIR / "Image_Teacher_best.pth"
    if not text_path.exists() or not image_path.exists():
        missing = [p.name for p in (text_path, image_path) if not p.exists()]
        raise FileNotFoundError(
            f"Missing teacher weights: {', '.join(missing)}. Run train_teachers.py first."
        )
    text_teacher.load_state_dict(torch.load(text_path, map_location=device))
    image_teacher.load_state_dict(torch.load(image_path, map_location=device))
    text_teacher.eval()
    image_teacher.eval()
    for teacher in (text_teacher, image_teacher):
        for param in teacher.parameters():
            param.requires_grad = False


def main():
    parser = argparse.ArgumentParser(
        description="Train the retained KD version: EANN student + single-modality teachers."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-events", type=int, default=50)
    parser.add_argument("--missing-rate", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--kd-weight", type=float, default=0.5)
    parser.add_argument("--feature-weight", type=float, default=10.0)
    parser.add_argument("--output", default="eann_distilled_feature_best.pth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    vocab = load_vocab()
    train_loader = make_loader(
        "train",
        vocab,
        batch_size=args.batch_size,
        shuffle=True,
        missing_rate=args.missing_rate,
    )
    val_loader = make_loader("val", vocab, batch_size=args.batch_size, shuffle=False)

    student = EANN(vocab_size=len(vocab), num_events=args.num_events).to(device)
    text_teacher = TextTeacher(vocab_size=len(vocab)).to(device)
    image_teacher = ImageTeacher().to(device)
    load_teacher_weights(text_teacher, image_teacher, device)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=args.lr)
    best_acc = 0.0
    output_path = EANN_DIR / args.output

    print(
        f"device={device} vocab={len(vocab)} missing_rate={args.missing_rate} "
        f"T={args.temperature} kd_weight={args.kd_weight}"
    )

    for epoch in range(1, args.epochs + 1):
        student.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            texts, images, labels, event_labels, flags = unpack_batch(batch)
            texts, images = texts.to(device), images.to(device)
            labels, event_labels = labels.to(device), event_labels.to(device)

            if len(flags) >= 2:
                text_missing = flags[0].to(device).bool()
                image_missing = flags[1].to(device).bool()
            else:
                text_missing = texts.sum(dim=1).eq(0)
                image_missing = images.flatten(1).abs().sum(dim=1).eq(0)

            p = float(batch_idx + (epoch - 1) * len(train_loader)) / args.epochs / len(train_loader)
            alpha_adv = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

            optimizer.zero_grad()
            s_fake, s_event, s_text_feat, s_image_feat = student(texts, images, alpha=alpha_adv)

            loss_detect = criterion_ce(s_fake, labels)
            loss_event = criterion_ce(s_event, event_labels)
            loss_kd = torch.tensor(0.0, device=device)

            if text_missing.any():
                with torch.no_grad():
                    image_logits, image_feat = image_teacher(images[text_missing])
                loss_kd = loss_kd + kd_loss(
                    s_fake[text_missing],
                    image_logits,
                    args.temperature,
                )
                loss_kd = loss_kd + criterion_mse(
                    s_image_feat[text_missing],
                    image_feat,
                ) * args.feature_weight

            if image_missing.any():
                with torch.no_grad():
                    text_logits, text_feat = text_teacher(texts[image_missing])
                loss_kd = loss_kd + kd_loss(
                    s_fake[image_missing],
                    text_logits,
                    args.temperature,
                )
                loss_kd = loss_kd + criterion_mse(
                    s_text_feat[image_missing],
                    text_feat,
                ) * args.feature_weight

            loss = loss_detect + loss_event + args.kd_weight * loss_kd
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 50 == 0:
                print(
                    f"epoch {epoch}/{args.epochs} batch {batch_idx}/{len(train_loader)} | "
                    f"loss={loss.item():.4f} ce={loss_detect.item():.4f} "
                    f"event={loss_event.item():.4f} kd={loss_kd.item():.4f}"
                )

        val_acc = evaluate(student, val_loader, device)
        print(
            f"epoch {epoch}/{args.epochs} | "
            f"avg_loss={running_loss / len(train_loader):.4f} val_acc={val_acc * 100:.2f}%"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), output_path)
            print(f"saved {output_path.name} | best_val_acc={best_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
