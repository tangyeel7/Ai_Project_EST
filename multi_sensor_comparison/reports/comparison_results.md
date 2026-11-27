# DASIG single-sensor vs five-sensor comparison

## Experimental setup

- Common complete-sensor windows: 5,220 (Standard 4,524; Abrupt 696).
- Incomplete-sensor trials excluded from both models: 1 (sub013_LA_L).
- Same participant split for both: 36 fit, 12 validation, 12 held-out test participants.
- Same window labels, 3-second duration, 50 Hz rate, optimizer, class weighting, augmentation, and validation macro-F1 early stopping.
- Single-sensor model: 6 active-forearm accel/gyro channels.
- Fused model: 30 accel/gyro channels from right forearm, left forearm, sternum, right upper arm, and left upper arm.
- All channels are normalized with fit-participant statistics. Sensor blocks share the trial timestamps in each DASIG CSV.

## Held-out test comparison

| Model | Input channels | Accuracy | Macro F1 | Balanced accuracy | Model latency (ms/window) |
|---|---:|---:|---:|---:|---:|
| Single active forearm | 6 | 0.9676 | 0.9301 | 0.9314 | 0.661 |
| Five-sensor early fusion | 30 | 0.9380 | 0.8761 | 0.9084 | 0.752 |
| Difference (fusion - single) | +24 | -0.0296 | -0.0539 | -0.0230 | +0.091 |

## Per-class held-out test metrics

| Model | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| Single forearm | Standard pick-and-place | 0.9818 | 0.9808 | 0.9813 | 936 |
| Single forearm | Abrupt alarm response | 0.8759 | 0.8819 | 0.8789 | 144 |
| Five-sensor fusion | Standard pick-and-place | 0.9791 | 0.9487 | 0.9636 | 936 |
| Five-sensor fusion | Abrupt alarm response | 0.7225 | 0.8681 | 0.7886 | 144 |

## Interpretation

A positive difference means the fused model scored higher on this held-out DASIG split; a negative value means the single-sensor model scored higher. The 30-channel model uses the same number of windows, not five times more examples. This comparison measures one split and should not be treated as proof that sensor fusion always helps.

Phone demo note: the existing phone supplies one IMU only. It is compatible with the 6-channel baseline, not this 30-channel fused checkpoint.
