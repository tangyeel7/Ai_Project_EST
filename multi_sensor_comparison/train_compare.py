"""Train comparable 6-channel and 30-channel DASIG CNN-LSTM models."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dasig_data import (  # noqa: E402
    RAW_SAMPLE_RATE_HZ, SAMPLE_RATE_HZ, WINDOW_SECONDS, WINDOW_SAMPLES,
    _read_events, load_dasig, subject_split,
)
from src.metrics import classification_metrics, predict  # noqa: E402
from src.model import ActivityCNNLSTM  # noqa: E402

SENSOR_NAMES = ["right_forearm", "left_forearm", "sternum", "right_upper_arm", "left_upper_arm"]
CHANNEL_NAMES = [f"{sensor}_{axis}" for sensor in SENSOR_NAMES
                 for axis in ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")]


class MultiSensorCNNLSTM(nn.Module):
    """Same CNN-LSTM as the baseline, with a configurable sensor-channel count."""
    def __init__(self, input_channels, num_classes=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=48, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(96, num_classes))

    def forward(self, x):
        z = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        _, (hidden, _) = self.lstm(z)
        return self.classifier(torch.cat([hidden[-2], hidden[-1]], dim=1))


def read_all_sensors(path):
    """Read the five synchronized accel+gyro blocks from one trial CSV."""
    frame = pd.read_csv(path, sep=";", decimal=",", skiprows=3, header=None, low_memory=False)
    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 66:
        raise ValueError(f"Expected timestamp plus five 13-value IMU blocks in {path}; got {values.shape}.")
    time_s = values[:, 0]
    keep = np.isfinite(time_s)
    time_s = time_s[keep]
    if len(time_s) < 600 or np.any(np.diff(time_s) <= 0):
        raise ValueError(f"Invalid or too-short timestamp series in {path}.")

    blocks = []
    for sensor_index in range(5):
        first = 1 + 13 * sensor_index
        block = values[keep, first:first + 6]
        # Keep the comparison complete-case: don't synthesize an absent IMU stream.
        if np.isfinite(block).mean() < 0.95:
            return time_s, None
        for channel in range(block.shape[1]):
            valid = np.isfinite(block[:, channel])
            if valid.sum() < 2:
                return time_s, None
            if not valid.all():
                block[~valid, channel] = np.interp(time_s[~valid], time_s[valid], block[valid, channel])
        blocks.append(block)
    return time_s, np.concatenate(blocks, axis=1).astype(np.float32)


def resample_window(time_s, signal, start_s):
    bin_starts = start_s + np.arange(WINDOW_SAMPLES, dtype=np.float64) / SAMPLE_RATE_HZ
    offsets = (np.arange(RAW_SAMPLE_RATE_HZ // SAMPLE_RATE_HZ) + 0.5) / RAW_SAMPLE_RATE_HZ
    target = bin_starts[:, None] + offsets[None, :]
    if target[-1, -1] > time_s[-1] + 1e-8:
        return None
    return np.stack([
        np.interp(target.ravel(), time_s, signal[:, channel])
        .reshape(WINDOW_SAMPLES, -1).mean(axis=1)
        for channel in range(signal.shape[1])
    ], axis=1).astype(np.float32)


def load_all_sensor_windows(data_root, baseline=None):
    """Reuse baseline labels/order and replace its 6-channel windows with 30 channels."""
    baseline = baseline if baseline is not None else load_dasig(data_root)
    cache = {}
    windows, indices, excluded_trials = [], [], set()
    for index, item in enumerate(baseline["metadata"]):
        key = (item["subject"], item["trial"])
        if key not in cache:
            path = Path(data_root) / key[0] / f"{key[0]}_{key[1]}_MIMU.csv"
            cache[key] = read_all_sensors(path)
        time_s, signals = cache[key]
        if signals is None:
            excluded_trials.add(f"{key[0]}_{key[1]}")
            continue
        window = resample_window(time_s, signals, item["cue_time_s"])
        if window is None:
            raise ValueError(f"Could not align all five sensors for {key} at {item['cue_time_s']:.3f}s.")
        windows.append(window)
        indices.append(index)
    keep = np.asarray(indices, dtype=np.int64)
    return {**baseline, "X": np.asarray(windows, dtype=np.float32), "y": baseline["y"][keep],
            "subjects": baseline["subjects"][keep],
            "metadata": [baseline["metadata"][index] for index in keep], "channels": CHANNEL_NAMES,
            "indices": keep, "excluded_incomplete_sensor_trials": sorted(excluded_trials)}


def scale_data(X_fit, X_other):
    mean = X_fit.mean(axis=(0, 1))
    scale = np.maximum(X_fit.std(axis=(0, 1)), 1e-6)
    return ((X_fit - mean) / scale).astype(np.float32), ((X_other - mean) / scale).astype(np.float32), mean, scale


def fit(model, X, y, X_val, y_val, class_names, epochs, batch_size, patience, seed):
    torch.manual_seed(seed)
    counts = np.bincount(y, minlength=len(class_names))
    weights = len(y) / (len(counts) * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                        batch_size=batch_size, shuffle=True, generator=generator)
    best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            gain = torch.empty((len(xb), 1, 1)).uniform_(0.93, 1.07)
            xb = xb * gain + torch.randn_like(xb) * 0.04
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        pred = predict(model, X_val).argmax(axis=1)
        metrics = classification_metrics(y_val, pred, class_names)
        score = metrics["macro_f1"]
        print(f"  Epoch {epoch:02d}: val accuracy={metrics['accuracy']:.3f}, macro-F1={score:.3f}")
        if score > best_score + 1e-4:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), best_epoch


def run_model(name, X, data, split, output_dir, epochs, batch_size, patience, seed):
    train_mask, valid_mask, test_mask, train_ids, valid_ids, test_ids = split
    X_fit, X_valid, mean, scale = scale_data(X[train_mask], X[valid_mask])
    X_test = ((X[test_mask] - mean) / scale).astype(np.float32)
    y_fit, y_valid, y_test = data["y"][train_mask], data["y"][valid_mask], data["y"][test_mask]
    channels = X.shape[2]
    print(f"\nTraining {name} model ({channels} channels; train {len(y_fit)}, valid {len(y_valid)}, test {len(y_test)})")
    if channels == 6:
        torch.manual_seed(seed)
        model = ActivityCNNLSTM(num_classes=len(data["class_names"]))
    else:
        torch.manual_seed(seed)
        model = MultiSensorCNNLSTM(channels, num_classes=len(data["class_names"]))
    model, best_epoch = fit(model, X_fit, y_fit, X_valid, y_valid, data["class_names"],
                            epochs, batch_size, patience, seed)
    valid_metrics = classification_metrics(y_valid, predict(model, X_valid).argmax(axis=1), data["class_names"])
    test_metrics = classification_metrics(y_test, predict(model, X_test).argmax(axis=1), data["class_names"])
    sample = torch.from_numpy(X_test[:1])
    with torch.inference_mode():
        for _ in range(10):
            model(sample)
        started = time.perf_counter()
        for _ in range(200):
            model(sample)
    latency = (time.perf_counter() - started) * 1000 / 200
    checkpoint = {
        "state_dict": model.state_dict(), "mean": torch.tensor(mean), "scale": torch.tensor(scale),
        "class_names": data["class_names"], "channels": data["channels"],
        "window_samples": data["window_samples"], "window_seconds": data["window_seconds"],
        "sample_rate_hz": data["sample_rate_hz"], "train_subjects": train_ids,
        "validation_subjects": valid_ids, "test_subjects": test_ids, "best_epoch": best_epoch,
    }
    torch.save(checkpoint, output_dir / "models" / f"{name}.pt")
    clean = lambda m: {key: value for key, value in m.items() if key != "confusion"}
    return {"input_channels": channels, "best_epoch": int(best_epoch),
            "validation": clean(valid_metrics), "test": clean(test_metrics),
            "latency_ms_per_window": float(latency)}


def format_report(results):
    single = results["single_forearm"]["test"]
    fused = results["five_sensor_fusion"]["test"]
    counts = results["class_counts"]
    split = results["split"]
    lines = [
        "# DASIG single-sensor vs five-sensor comparison", "",
        "## Experimental setup", "",
        f"- Common complete-sensor windows: {results['total_windows']:,} (Standard {counts['Standard pick-and-place']:,}; Abrupt {counts['Abrupt alarm response']:,}).",
        f"- Incomplete-sensor trials excluded from both models: {len(results['excluded_incomplete_sensor_trials'])} ({', '.join(results['excluded_incomplete_sensor_trials'])}).",
        f"- Same participant split for both: {len(split['train_subjects'])} fit, {len(split['validation_subjects'])} validation, {len(split['test_subjects'])} held-out test participants.",
        "- Same window labels, 3-second duration, 50 Hz rate, optimizer, class weighting, augmentation, and validation macro-F1 early stopping.",
        "- Single-sensor model: 6 active-forearm accel/gyro channels.",
        "- Fused model: 30 accel/gyro channels from right forearm, left forearm, sternum, right upper arm, and left upper arm.",
        "- All channels are normalized with fit-participant statistics. Sensor blocks share the trial timestamps in each DASIG CSV.", "",
        "## Held-out test comparison", "",
        "| Model | Input channels | Accuracy | Macro F1 | Balanced accuracy | Model latency (ms/window) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Single active forearm | 6 | {single['accuracy']:.4f} | {single['macro_f1']:.4f} | {single['balanced_accuracy']:.4f} | {results['single_forearm']['latency_ms_per_window']:.3f} |",
        f"| Five-sensor early fusion | 30 | {fused['accuracy']:.4f} | {fused['macro_f1']:.4f} | {fused['balanced_accuracy']:.4f} | {results['five_sensor_fusion']['latency_ms_per_window']:.3f} |",
        f"| Difference (fusion - single) | +24 | {fused['accuracy']-single['accuracy']:+.4f} | {fused['macro_f1']-single['macro_f1']:+.4f} | {fused['balanced_accuracy']-single['balanced_accuracy']:+.4f} | {results['five_sensor_fusion']['latency_ms_per_window']-results['single_forearm']['latency_ms_per_window']:+.3f} |",
        "", "## Per-class held-out test metrics", "",
        "| Model | Class | Precision | Recall | F1 | Support |", "|---|---|---:|---:|---:|---:|",
    ]
    for name, title in (("single_forearm", "Single forearm"), ("five_sensor_fusion", "Five-sensor fusion")):
        for row in results[name]["test"]["per_class"]:
            lines.append(f"| {title} | {row['Activity']} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1']:.4f} | {row['Support']} |")
    lines += ["", "## Interpretation", "",
              "A positive difference means the fused model scored higher on this held-out DASIG split; a negative value means the single-sensor model scored higher. The 30-channel model uses the same number of windows, not five times more examples. This comparison measures one split and should not be treated as proof that sensor fusion always helps.",
              "", "Phone demo note: the existing phone supplies one IMU only. It is compatible with the 6-channel baseline, not this 30-channel fused checkpoint.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "raw" / "dasig" / "DASIG"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    torch.set_num_threads(min(4, torch.get_num_threads()))
    data = load_dasig(args.data)
    fused_data = load_all_sensor_windows(args.data, baseline=data)
    indices = fused_data["indices"]
    common_data = {**data, "X": data["X"][indices], "y": data["y"][indices],
                   "subjects": data["subjects"][indices],
                   "excluded_incomplete_sensor_trials": fused_data["excluded_incomplete_sensor_trials"]}
    if not (np.array_equal(common_data["y"], fused_data["y"]) and
            np.array_equal(common_data["subjects"], fused_data["subjects"])):
        raise RuntimeError("Single- and five-sensor windows do not align; refusing an unfair comparison.")
    split = subject_split(common_data["subjects"], seed=args.seed)
    output_dir = Path(__file__).resolve().parent
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)

    results = {
        "dataset": "DASIG v3",
        "split": {"train_subjects": split[3], "validation_subjects": split[4], "test_subjects": split[5]},
        "total_windows": int(len(common_data["y"])),
        "class_counts": {name: int((common_data["y"] == i).sum()) for i, name in enumerate(data["class_names"])},
        "excluded_incomplete_sensor_trials": common_data["excluded_incomplete_sensor_trials"],
        "single_forearm": run_model("single_forearm", common_data["X"], common_data, split, output_dir,
                                     args.epochs, args.batch_size, args.patience, args.seed),
        "five_sensor_fusion": run_model("five_sensor_fusion", fused_data["X"], common_data, split, output_dir,
                                         args.epochs, args.batch_size, args.patience, args.seed),
    }
    report_path = output_dir / "reports" / "comparison_results.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "reports" / "comparison_results.md").write_text(format_report(results), encoding="utf-8")
    print("\nHeld-out comparison:")
    for name in ("single_forearm", "five_sensor_fusion"):
        metrics = results[name]["test"]
        print(f"  {name}: accuracy={metrics['accuracy']:.3f}, macro-F1={metrics['macro_f1']:.3f}, "
              f"balanced-accuracy={metrics['balanced_accuracy']:.3f}")
    print(f"\nReports: {report_path.parent}")


if __name__ == "__main__":
    main()
