import argparse
import random

import torch

from experiment_utils import (
    EANN_DIR,
    classification_metrics,
    get_device,
    load_vocab,
    make_loader,
    save_rows_csv,
    set_seed,
    unpack_batch,
)
from models_EANN import EANN


def parse_rates(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def apply_missing(texts, images, missing_rate, target, rng):
    if missing_rate <= 0 or target == "complete":
        return texts, images

    texts = texts.clone()
    images = images.clone()
    for i in range(texts.size(0)):
        if rng.random() < missing_rate:
            if target == "drop_text":
                texts[i] = torch.zeros_like(texts[i])
            elif target == "drop_image":
                images[i] = torch.zeros_like(images[i])
    return texts, images


def evaluate_model(model, loader, device, missing_rate, target, seed):
    model.eval()
    rng = random.Random(seed)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            texts, images, labels, _, _ = unpack_batch(batch)
            texts, images = apply_missing(texts, images, missing_rate, target, rng)
            texts, images = texts.to(device), images.to(device)

            fake_out, _, _, _ = model(texts, images, alpha=0.0)
            preds = fake_out.argmax(dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().tolist())

    return classification_metrics(all_labels, all_preds)


def load_eann(weight_path, vocab_size, num_events, device):
    model = EANN(vocab_size=vocab_size, num_events=num_events).to(device)
    incompatible = model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"loaded with compatibility mode | "
            f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
        )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EANN with and without KD on the same test set."
    )
    parser.add_argument("--baseline", default="eann_best_model.pth")
    parser.add_argument("--kd", default="eann_distilled_feature_best.pth")
    parser.add_argument("--num-events", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rates", default="0,0.2,0.5,0.8,1.0")
    parser.add_argument("--output", default="kd_ablation_test_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    vocab = load_vocab()
    test_loader = make_loader("test", vocab, batch_size=args.batch_size, shuffle=False)

    model_specs = [
        ("EANN_no_KD", False, EANN_DIR / args.baseline),
        ("EANN_with_KD", True, EANN_DIR / args.kd),
    ]
    rates = parse_rates(args.rates)
    rows = []

    print(f"device={device} test_set=CFND_dataset/test_data.csv vocab={len(vocab)}")
    for model_name, uses_kd, weight_path in model_specs:
        if not weight_path.exists():
            print(f"skip {model_name}: missing {weight_path}")
            continue

        model = load_eann(weight_path, len(vocab), args.num_events, device)
        print(f"\nmodel={model_name} weight={weight_path.name}")

        for rate in rates:
            targets = ["complete"] if rate == 0 else ["drop_text", "drop_image"]
            for target in targets:
                metrics = evaluate_model(
                    model,
                    test_loader,
                    device,
                    missing_rate=rate,
                    target=target,
                    seed=args.seed + int(rate * 1000) + (0 if target != "drop_image" else 7),
                )
                row = {
                    "model": model_name,
                    "uses_kd": uses_kd,
                    "missing_target": target,
                    "missing_rate": rate,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                }
                rows.append(row)
                print(
                    f"{target:10s} rate={rate:.1f} | "
                    f"acc={metrics['accuracy'] * 100:.2f}% "
                    f"precision={metrics['precision'] * 100:.2f}% "
                    f"recall={metrics['recall'] * 100:.2f}% "
                    f"f1={metrics['f1'] * 100:.2f}%"
                )

    output_path = EANN_DIR / args.output
    save_rows_csv(output_path, rows)
    print(f"\nsaved {output_path}")


if __name__ == "__main__":
    main()
