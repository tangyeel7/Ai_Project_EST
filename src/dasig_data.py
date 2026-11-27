"""Load DASIG events and build active-forearm, 3-second IMU windows."""
from pathlib import Path
import numpy as np
import pandas as pd

SAMPLE_RATE_HZ = 50
RAW_SAMPLE_RATE_HZ = 200
WINDOW_SECONDS = 3
WINDOW_SAMPLES = SAMPLE_RATE_HZ * WINDOW_SECONDS
CLASS_NAMES = ["Standard pick-and-place", "Abrupt alarm response"]
TRIALS = {"FR_R": "right", "FR_L": "left", "LA_L": "left"}
# The DASIG paper identifies these trials as missing a visual-alarm response.
MISSING_RESPONSE_TRIALS = {("sub027", "FR_R"), ("sub033", "LA_L"),
                           ("sub039", "FR_R"), ("sub056", "FR_R"),
                           ("sub060", "FR_R")}
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def _read_mimu(path, active_side):
    # DASIG uses semicolon fields, decimal commas, and three descriptive header rows.
    frame = pd.read_csv(path, sep=";", decimal=",", skiprows=3, header=None, low_memory=False)
    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 66:
        raise ValueError(f"Unexpected DASIG IMU columns in {path} ({values.shape}).")
    base = 1 if active_side == "right" else 14
    time_s = values[:, 0]
    signal = np.column_stack([values[:, base:base + 3], values[:, base + 3:base + 6]])
    valid = np.isfinite(time_s) & np.isfinite(signal).all(axis=1)
    time_s, signal = time_s[valid], signal[valid]
    if len(time_s) < 600 or np.any(np.diff(time_s) <= 0):
        raise ValueError(f"Invalid or too-short time series in {path}.")
    return time_s, signal.astype(np.float32)


def _read_events(path):
    frame = pd.read_csv(path, sep=";", decimal=",", header=0)
    if frame.shape[1] < 2:
        raise ValueError(f"Unexpected DASIG Arduino file: {path}")
    time_s = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float64)
    code = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(time_s) & np.isfinite(code)
    return time_s[valid], code[valid].astype(int)


def _resample_window(time_s, signal, start_s):
    bin_starts = start_s + np.arange(WINDOW_SAMPLES, dtype=np.float64) / SAMPLE_RATE_HZ
    # Average four interpolated 200 Hz readings into each 50 Hz model sample.
    offsets = (np.arange(RAW_SAMPLE_RATE_HZ // SAMPLE_RATE_HZ) + 0.5) / RAW_SAMPLE_RATE_HZ
    target = bin_starts[:, None] + offsets[None, :]
    if target[-1, -1] > time_s[-1] + 1e-8:
        return None
    return np.stack([np.interp(target.ravel(), time_s, signal[:, c]).reshape(WINDOW_SAMPLES, -1).mean(axis=1)
                     for c in range(6)], axis=1).astype(np.float32)


def load_dasig(root="data/raw/dasig/DASIG"):
    root = Path(root)
    X, y, metadata = [], [], []
    dropped_trials = []
    for subject_dir in sorted(root.glob("sub[0-9][0-9][0-9]")):
        subject = subject_dir.name
        for trial, active_side in TRIALS.items():
            if (subject, trial) in MISSING_RESPONSE_TRIALS:
                dropped_trials.append(f"{subject}_{trial}")
                continue
            stem = f"{subject}_{trial}"
            imu_path = subject_dir / f"{stem}_MIMU.csv"
            event_path = subject_dir / f"{stem}_Arduino.csv"
            if not imu_path.is_file() or not event_path.is_file():
                continue
            time_s, signal = _read_mimu(imu_path, active_side)
            event_time, event_code = _read_events(event_path)
            cue_indices = np.flatnonzero((event_code >= 2) & (event_code <= 5))
            alarm_times = event_time[(event_code >= 6) & (event_code <= 10)]
            for pos, cue_index in enumerate(cue_indices):
                cue_time = float(event_time[cue_index])
                next_cue = float(event_time[cue_indices[pos + 1]]) if pos + 1 < len(cue_indices) else cue_time + 3.0
                alarm_in_cycle = np.any((alarm_times >= cue_time) & (alarm_times < next_cue))
                window = _resample_window(time_s, signal, cue_time)
                if window is None:
                    continue
                X.append(window)
                y.append(int(alarm_in_cycle))
                metadata.append({"subject": subject, "trial": trial, "cue_time_s": cue_time,
                                 "station_code": int(event_code[cue_index]),
                                 "label": CLASS_NAMES[int(alarm_in_cycle)]})
    if not X:
        raise FileNotFoundError(f"No DASIG trial CSV files found under {root}. Run `python -m src.download_dasig`.")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    subject_ids = np.asarray([item["subject"] for item in metadata])
    return {"X": X, "y": y, "subjects": subject_ids, "metadata": metadata,
            "class_names": CLASS_NAMES, "channels": CHANNELS,
            "window_samples": WINDOW_SAMPLES, "sample_rate_hz": SAMPLE_RATE_HZ,
            "window_seconds": WINDOW_SECONDS, "dropped_trials": dropped_trials}


def subject_split(subjects, seed=17):
    unique = np.unique(subjects)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_train = round(0.60 * len(unique))
    n_valid = round(0.20 * len(unique))
    train_ids = sorted(shuffled[:n_train].tolist())
    valid_ids = sorted(shuffled[n_train:n_train + n_valid].tolist())
    test_ids = sorted(shuffled[n_train + n_valid:].tolist())
    return (np.isin(subjects, train_ids), np.isin(subjects, valid_ids),
            np.isin(subjects, test_ids), train_ids, valid_ids, test_ids)
