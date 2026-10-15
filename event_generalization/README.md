# Event generalization experiment

This folder contains the scripts for the event-held-out generalization test.
The split follows the design:

- Train events: `2, 9, 5, 6`
- Validation event: `3`
- Test events: `0, 1, 4, 7, 8`

The original `CFND_dataset/*.csv` files are not modified. New split files are
written under `event_generalization/data/`.

## Usage

Run from `D:\pyproject\Multimodal`:

```powershell
.\venv\Scripts\python.exe event_generalization\split_by_event.py
.\venv\Scripts\python.exe event_generalization\train_compare.py --epochs 20
```

For a quick smoke test:

```powershell
.\venv\Scripts\python.exe event_generalization\train_compare.py --epochs 1 --max-train-batches 2 --max-eval-batches 2
```

The comparison trains:

- `plain`: ordinary multimodal detector without event adversarial learning.
- `eann`: EANN with gradient reversal and event discriminator.

Both models use the same TextCNN and ResNet18 feature extractors, so the key
difference is whether event adversarial training is used.
