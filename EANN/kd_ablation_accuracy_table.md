# Knowledge distillation ablation accuracy table

Dataset: `CFND_dataset/test_data.csv`

| Test scenario | Missing rate | EANN without KD | EANN with KD | Accuracy gain |
|---|---:|---:|---:|---:|
| Complete text + image | 0% | 77.07% | 82.79% | +5.72% |
| Drop text | 20% | 75.38% | 79.70% | +4.32% |
| Drop image | 20% | 77.00% | 82.15% | +5.16% |
| Drop text | 50% | 72.52% | 75.57% | +3.06% |
| Drop image | 50% | 78.42% | 81.83% | +3.41% |
| Drop text | 100% | 67.51% | 67.77% | +0.26% |
| Drop image | 100% | 79.42% | 80.73% | +1.31% |

Conclusion for thesis writing: knowledge distillation improves the EANN
student on the complete test set and most missing-modality settings. The gain
is strongest at 20%-50% missing rates. When text is completely missing, the
improvement is small, which suggests that the text modality is the dominant
source of discriminative rumor cues in this dataset.
