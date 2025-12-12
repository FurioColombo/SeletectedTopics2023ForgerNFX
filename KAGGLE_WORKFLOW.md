# Quick Reference: Kaggle Training Workflow

## 🚀 Run Training on Kaggle

### First Time (Upload Dataset)
```bash
python kaggle_train.py --push-dataset --run
```

### Subsequent Runs (Skip Dataset Upload - FAST!)
```bash
python kaggle_train.py --run
```

### Force Dataset Re-upload (if dataset changed)
```bash
python kaggle_train.py --push-dataset --force-dataset --run
```

### Download Results
```bash
python kaggle_train.py --pull-metrics
```

## 📊 Evaluate Trained Models

```bash
python evaluate.py
```
- Select checkpoint interactively
- Choose effect to evaluate
- Get interactive HTML report

## 🧪 Run Multiple Experiments

```bash
python run_experiments.py
```
Guides you through running multiple trainings with different hyperparameters

---

## 💡 Tips

- **Save time**: After first upload, use `--run` only (skips heavy dataset upload)
- **Memory**: Evaluation saves NO wav files by default (configurable in interactive prompt)
- **Organization**: All evaluations saved to timestamped folders in `outputs/evaluations/`
