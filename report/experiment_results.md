# DASIG experiment results

## Evaluation setup

- **Task:** classify a 3-second window as standard pick-and-place or abrupt alarm-response movement.
- **Dataset:** DASIG v3; 60 participants, 3 trials each. The five trials flagged by the dataset authors as missing a visual-alarm response were excluded.
- **Input:** active forearm accelerometer and gyroscope (6 channels), resampled from 200 Hz to 50 Hz; 150 samples per window.
- **Windows:** 5,250 total; Standard 4,550, Abrupt 700.
- **Split:** 36 fitting participants, 12 validation participants, 12 held-out test participants.
- **Best epoch:** 27.
- **Model-only CPU inference:** 0.62 ms per 3-second window on this computer. This excludes sensor acquisition and communication.

## Held-out test results

- Accuracy: **97.1%**
- Macro F1: **93.7%**
- Balanced accuracy: **93.3%**

| Class | Precision | Recall | F1 | Test windows |
|---|---:|---:|---:|---:|
| Standard pick-and-place | 98.2% | 98.5% | 98.3% | 936 |
| Abrupt alarm response | 90.1% | 88.2% | 89.1% | 144 |

Validation metrics: accuracy 99.1%, macro F1 98.2%, balanced accuracy 98.3%.

## How to interpret this score

The split keeps all windows from each participant together. However, labels are derived from alarm events and do not mark exact movement onset; this is a controlled-workstation benchmark, not factory validation. The reported score applies to DASIG's event-derived standard/abrupt task. The phone CSV score is a separate device/user check and must not be combined with this benchmark score.

Dataset: [DASIG on Zenodo](https://zenodo.org/records/17660014). Cite the dataset authors in submissions.
