"""Convert labeled phone sensor CSV recordings into model windows."""
import numpy as np
import pandas as pd

from src.dasig_data import CHANNELS, SAMPLE_RATE_HZ, WINDOW_SAMPLES, WINDOW_SECONDS, CLASS_NAMES

LABEL_TO_ID = {"standard": 0, "abrupt": 1}


def phone_windows(frame, stride_seconds=1.5):
    required = ["time_s", "label", *CHANNELS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Phone CSV is missing columns: " + ", ".join(missing))
    clean = frame[required].copy()
    clean["time_s"] = pd.to_numeric(clean["time_s"], errors="coerce")
    for channel in CHANNELS:
        clean[channel] = pd.to_numeric(clean[channel], errors="coerce")
    clean = clean.dropna().sort_values("time_s").reset_index(drop=True)
    if len(clean) < WINDOW_SAMPLES:
        raise ValueError("Record at least 3 seconds of labeled phone motion before checking it.")

    times = clean["time_s"].to_numpy(dtype=np.float64)
    labels = clean["label"].astype(str).str.lower().to_numpy()
    signals = clean[CHANNELS].to_numpy(dtype=np.float32)
    boundary = np.r_[True, (labels[1:] != labels[:-1]) | (np.diff(times) > 0.15)]
    starts = np.flatnonzero(boundary)
    ends = np.r_[starts[1:], len(clean)]

    X, y, metadata = [], [], []
    ignored_rows = int(np.sum(~np.isin(labels, list(LABEL_TO_ID))))
    for start_i, end_i in zip(starts, ends):
        label = labels[start_i]
        if label not in LABEL_TO_ID or end_i - start_i < 2:
            continue
        t = times[start_i:end_i]
        x = signals[start_i:end_i]
        begin, end = float(t[0]), float(t[-1])
        cursor = begin
        while cursor + WINDOW_SECONDS - 1 / SAMPLE_RATE_HZ <= end + 1e-8:
            target = cursor + np.arange(WINDOW_SAMPLES) / SAMPLE_RATE_HZ
            window = np.column_stack([np.interp(target, t, x[:, c]) for c in range(6)])
            X.append(window.astype(np.float32))
            y.append(LABEL_TO_ID[label])
            metadata.append({"label": CLASS_NAMES[LABEL_TO_ID[label]], "start_s": cursor})
            cursor += stride_seconds
    if not X:
        raise ValueError("No complete 3-second standard/abrupt windows found. Keep each labeled action active for at least 3 seconds.")
    return {"X": np.asarray(X, dtype=np.float32), "y": np.asarray(y, dtype=np.int64),
            "metadata": metadata, "ignored_rows": ignored_rows,
            "class_names": CLASS_NAMES, "channels": CHANNELS}


def load_phone_csv(path):
    return phone_windows(pd.read_csv(path))
