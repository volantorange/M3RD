# EANN knowledge-distillation experiment

This folder is the retained research line for the graduation project:
**multi-modal rumor detection under missing modalities**.

The root-level baseline model is intentionally untouched. The recommended
paper experiments are in this folder:

1. `train_baseline.py`
   Trains the EANN student without knowledge distillation.

2. `train_teachers.py`
   Trains the text-only and image-only teacher models.

3. `train_distillation.py`
   Trains the retained KD version: EANN student + text/image teachers, with
   logit-level KL distillation and feature-level MSE alignment.

4. `evaluate_kd_ablation.py`
   Evaluates the no-KD and KD checkpoints on the same test set and writes
   `kd_ablation_test_results.csv` for the thesis table.

Typical commands from `D:\pyproject\Multimodal`:

```powershell
.\venv\Scripts\python.exe EANN\train_baseline.py --epochs 20 --output eann_best_model.pth
.\venv\Scripts\python.exe EANN\train_teachers.py --epochs 15
.\venv\Scripts\python.exe EANN\train_distillation.py --epochs 20 --missing-rate 1.0
.\venv\Scripts\python.exe EANN\evaluate_kd_ablation.py
```

The old `kd/` folder is kept only as historical scratch code. Use the scripts
above as the single reproducible KD version.
