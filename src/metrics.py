"""Evaluation helpers without a separate machine-learning dependency."""
import numpy as np
import torch


def classification_metrics(y_true, y_pred, class_names):
    count = len(class_names)
    confusion = np.zeros((count, count), dtype=int)
    np.add.at(confusion, (y_true, y_pred), 1)
    tp = np.diag(confusion).astype(float)
    precision = tp / np.maximum(1, confusion.sum(axis=0))
    recall = tp / np.maximum(1, confusion.sum(axis=1))
    f1 = 2 * precision * recall / np.maximum(1e-9, precision + recall)
    return {"accuracy": float(tp.sum() / max(1, confusion.sum())),
            "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
            "per_class": [{"Activity": name, "Precision": float(precision[i]),
                           "Recall": float(recall[i]), "F1": float(f1[i]),
                           "Support": int(confusion[i].sum())} for i, name in enumerate(class_names)],
            "confusion": confusion}


def predict(model, X, batch_size=256):
    model.eval()
    device = next(model.parameters()).device
    probs = []
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            logits = model(torch.from_numpy(X[start:start + batch_size]).to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs)

