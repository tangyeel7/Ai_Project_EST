# DASIG single-sensor vs five-sensor comparison

This separate experiment compares the existing active-forearm input (6 channels) with early fusion of all five synchronized DASIG IMUs (30 channels). Both models use the same CNN-LSTM shape apart from the CNN input-channel count, the same labels, subject split, optimizer, augmentation, and model-selection rule.

## Run

Run this from the parent project directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe multi_sensor_comparison\train_compare.py
```

The script reads the existing dataset in `data/raw/dasig/DASIG/`. If that folder is missing, download it from the parent project first:

```powershell
.\.venv\Scripts\python.exe -m src.download_dasig
```

Training both models can take a few minutes on CPU. Results are written to `multi_sensor_comparison/reports/comparison_results.md` and `.json`; the two checkpoints are stored under `multi_sensor_comparison/models/`.

## Sensor/channel layout

Each per-trial DASIG MIMU CSV has one shared timestamp column and five sensor blocks: right forearm, left forearm, sternum, right upper arm, and left upper arm. Each block includes acceleration, gyroscope, magnetometer, and orientation. This comparison uses only the acceleration and gyroscope channels from every block: 5 × 6 = 30 channels. The current baseline uses only the active forearm block: 6 channels.

Every 3-second window is resampled to 150 time steps at 50 Hz. The trial labels and excluded-trial rule come from the parent project's `src.dasig_data` pipeline. Using the same windows and participant split makes the accuracy comparison meaningful.

## Fair comparison

- Participant split is identical for both models: 36 fit, 12 validation, 12 held-out test participants.
- Normalization is fitted separately for each input using only fit participants.
- Both models use class-weighted cross-entropy, jitter/gain augmentation, AdamW, and validation macro-F1 early stopping.
- The test set is evaluated only after each model is selected on validation data.
- More input channels do not create more training examples. Five-sensor fusion may help, but can also overfit; compare the measured results.

## Current result

On the matched complete-sensor set (5,220 windows; one incomplete-sensor trial excluded from both models), the 6-channel model scored **96.76% accuracy**, **93.01% macro F1**, and **93.14% balanced accuracy**. The 30-channel model scored **93.80% accuracy**, **87.61% macro F1**, and **90.84% balanced accuracy**. Here, early fusion lowered accuracy by 2.96 percentage points and macro F1 by 5.39 points. It did not improve this experiment's held-out score.

The comparison's single-sensor baseline was retrained on the matched windows using this folder's shared training routine, so it can differ slightly from the earlier main-project result. The five-sensor result is from one fixed participant split; it does not prove sensor fusion will always hurt.

This is early sensor fusion (concatenated channels), not an attention or separate-branch architecture. It is an offline comparison. The existing phone demo remains a one-sensor check and does not run the 30-channel model.
