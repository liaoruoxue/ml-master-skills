# Global Wisdom - ML Best Practices
<!--
  L3 层：先验智慧
  生命周期：永久 - 跨任务存在
  内容：通用代码模板、常见错误解决方案、最佳实践
  权限：只读（除非任务结束时触发 L3 更新）
-->

## Data Preparation

### Best Practices
- Always check for data leakage between train/val/test splits
- Normalize/standardize features before training
- Handle missing values consistently across all splits
- Check for class imbalance and address if needed
- Verify data types and ranges match expectations

### Common Patterns
```python
# Robust train/val/test split
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
```

---

## Model Training

### Best Practices
- Start with simple baselines before complex models
- Use early stopping to prevent overfitting
- Log all hyperparameters for reproducibility
- Monitor both training and validation metrics
- Use learning rate schedulers for stable convergence

### Common Patterns
```python
# Early stopping callback
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]
```

---

## Debugging

### Best Practices
- Check tensor shapes at every layer
- Verify loss decreases on small data subset first
- Use gradient checking for custom layers
- Print intermediate outputs to verify data flow
- Start with known-good configurations

### Debugging Checklist
1. [ ] Data loaded correctly? (shapes, types, ranges)
2. [ ] Labels encoded properly? (one-hot vs integer)
3. [ ] Learning rate reasonable? (try 1e-3 to 1e-5)
4. [ ] Batch size appropriate? (start with 32)
5. [ ] Model outputs correct shape?

---

## Common Errors & Solutions

| Error | Typical Cause | Solution |
|-------|---------------|----------|
| NaN loss | Learning rate too high | Reduce LR by 10x, check for inf in data |
| Loss not decreasing | LR too low or wrong loss | Increase LR, verify loss function matches task |
| Overfitting | Model too complex | Add dropout, regularization, or reduce capacity |
| Underfitting | Model too simple | Increase capacity, train longer |
| OOM (Out of Memory) | Batch size too large | Reduce batch size, use gradient accumulation |
| Shape mismatch | Incorrect layer config | Print shapes, verify input/output dims |
| Vanishing gradients | Too deep network | Use residual connections, batch norm |

---

## Hyperparameter Starting Points

| Task | Learning Rate | Batch Size | Epochs |
|------|---------------|------------|--------|
| Image Classification | 1e-3 to 1e-4 | 32-128 | 50-200 |
| Text Classification | 2e-5 to 5e-5 | 16-32 | 3-10 |
| Tabular Data | 1e-3 to 1e-2 | 64-256 | 100-500 |
| Object Detection | 1e-4 to 1e-5 | 8-16 | 50-300 |

---

## Cross-Validation Template

```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

def cross_validate(model_fn, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        score = model.score(X[val_idx], y[val_idx])
        scores.append(score)
        print(f"Fold {fold+1}: {score:.4f}")

    print(f"Mean: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    return scores
```

---

<!--
  NOTE: This file is READ-ONLY during task execution.
  Update only at task completion if new universal patterns are discovered.
  Prefix updates with date and source task.
-->
