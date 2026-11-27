"""Train a forearm IMU CNN-LSTM for standard vs abrupt industrial movements."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.dasig_data import load_dasig, subject_split
from src.metrics import classification_metrics, predict
from src.model import ActivityCNNLSTM


def scale_signals(X_fit, X_other):
    mean = X_fit.mean(axis=(0, 1))
    scale = np.maximum(X_fit.std(axis=(0, 1)), 1e-6)
    return ((X_fit - mean) / scale).astype(np.float32), ((X_other - mean) / scale).astype(np.float32), mean, scale


def fit(model, X, y, X_val, y_val, class_names, epochs=50, batch_size=128, patience=8):
    counts = np.bincount(y, minlength=len(class_names))
    weights = len(y) / (len(counts) * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                        batch_size=batch_size, shuffle=True)
    best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            # Small standardized jitter and gain variation model sensor noise and user differences.
            gain = torch.empty((len(xb), 1, 1)).uniform_(0.93, 1.07)
            xb = xb * gain + torch.randn_like(xb) * 0.04
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_pred = predict(model, X_val).argmax(axis=1)
        val = classification_metrics(y_val, val_pred, class_names)
        score = val["macro_f1"]
        print(f"Epoch {epoch:02d}: val accuracy={val['accuracy']:.3f}, macro-F1={score:.3f}")
        if score > best_score + 1e-4:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), best_epoch


def measure_latency(model, sample, repeats=200):
    tensor = torch.from_numpy(sample[:1])
    with torch.inference_mode():
        for _ in range(10):
            model(tensor)
        start = time.perf_counter()
        for _ in range(repeats):
            model(tensor)
    return (time.perf_counter() - start) * 1000 / repeats


def without_confusion(metrics):
    return {key: value for key, value in metrics.items() if key != "confusion"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/dasig/DASIG")
    parser.add_argument("--out", default="models/dasig_forearm_cnn_lstm.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(17)
    np.random.seed(17)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    data = load_dasig(args.data)
    train_mask, valid_mask, test_mask, train_ids, valid_ids, test_ids = subject_split(data["subjects"])
    X_fit, X_valid, mean, scale = scale_signals(data["X"][train_mask], data["X"][valid_mask])
    X_test = ((data["X"][test_mask] - mean) / scale).astype(np.float32)
    y_fit, y_valid, y_test = data["y"][train_mask], data["y"][valid_mask], data["y"][test_mask]

    model = ActivityCNNLSTM(num_classes=len(data["class_names"]))
    model, best_epoch = fit(model, X_fit, y_fit, X_valid, y_valid, data["class_names"],
                            args.epochs, args.batch_size, args.patience)
    valid_metrics = classification_metrics(y_valid, predict(model, X_valid).argmax(axis=1), data["class_names"])
    test_metrics = classification_metrics(y_test, predict(model, X_test).argmax(axis=1), data["class_names"])
    latency = measure_latency(model, X_test)

    checkpoint = {
        "state_dict": model.state_dict(), "mean": torch.tensor(mean), "scale": torch.tensor(scale),
        "class_names": data["class_names"], "channels": data["channels"],
        "window_samples": data["window_samples"], "window_seconds": data["window_seconds"],
        "sample_rate_hz": data["sample_rate_hz"], "train_subjects": train_ids,
        "validation_subjects": valid_ids, "test_subjects": test_ids,
        "best_epoch": int(best_epoch), "validation_metrics": without_confusion(valid_metrics),
        "test_metrics": without_confusion(test_metrics), "latency_ms": float(latency),
        "task": "standard vs abrupt industrial pick-and-place movement",
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)

    report = {
        "dataset": "DASIG v3", "input_sensor": "active forearm IMU accelerometer + gyroscope",
        "raw_sample_rate_hz": 200, "model_sample_rate_hz": data["sample_rate_hz"],
        "window_seconds": data["window_seconds"], "window_samples": data["window_samples"],
        "total_windows": int(len(data["y"])), "class_names": data["class_names"],
        "class_counts": {name: int((data["y"] == i).sum()) for i, name in enumerate(data["class_names"])},
        "train_subjects": train_ids, "validation_subjects": valid_ids, "test_subjects": test_ids,
        "excluded_flagged_trials": data["dropped_trials"], "best_epoch": int(best_epoch),
        "latency_ms_per_window": float(latency), "validation": without_confusion(valid_metrics),
        "test": without_confusion(test_metrics),
    }
    Path("report").mkdir(exist_ok=True)
    Path("report/experiment_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSubject-held-out test: accuracy={test_metrics['accuracy']:.3f}, "
          f"macro-F1={test_metrics['macro_f1']:.3f}, balanced accuracy={test_metrics['balanced_accuracy']:.3f}")
    print(f"Model-only CPU latency: {latency:.2f} ms per 3-second window")
    print(f"Saved model: {output}")


if __name__ == "__main__":
    main()
