# Code guide: what each file does

This guide follows one sample from the original dataset through training, prediction, and the phone demo. File names are linked to the project tree so the team can navigate while presenting.

## End-to-end flow

```text
DASIG IMU + event CSVs
        ↓
src/dasig_data.py → labeled (150 time steps × 6 channels) windows
        ↓
src/train.py → CNN-LSTM training and participant-held-out evaluation
        ↓
models/dasig_forearm_cnn_lstm.pt
        ↓
app/dashboard.py ─────────── app/phone.html (live phone sensors)
        ↑                              ↓ sensor windows / labeled rows
        └──── src/metrics.py ← src/live_server.py
```

The phone is a separate input/demo route. It uses the saved DASIG model; it does not retrain that model.

The primary saved model and phone route use one forearm IMU (six channels). The optional five-sensor comparison is isolated under `multi_sensor_comparison/` and has its own trainer, checkpoints, and reports.

## Main project files

| File/folder | Responsibility | Explain it in the viva |
|---|---|---|
| `src/download_dasig.py` | Downloads selected DASIG CSV/text files from the public archive, skipping large optional bundles. | Reproducible data acquisition. |
| `src/dasig_data.py` | Parses DASIG sensor and Arduino event files; chooses an active forearm sensor; creates event-derived 3-second windows; resamples to 50 Hz; builds participant splits. | Converts raw heterogeneous sensor/event data into model-ready labeled sequences. |
| `src/model.py` | Defines the neural network and checkpoint load/save helpers. | Architecture: 1D CNN for local patterns, bidirectional LSTM for temporal order, linear output for two classes. |
| `src/train.py` | Loads windows, fits normalization on training participants, trains with class-weighted loss and augmentation, selects a checkpoint using validation performance, and evaluates held-out participants. | The training/evaluation pipeline; test subjects never train the model. |
| `src/metrics.py` | Predicts model scores and calculates classification metrics. | Shared evaluation logic used in reports and app views. |
| `models/dasig_forearm_cnn_lstm.pt` | Trained parameters plus preprocessing/class/split metadata. | The checkpoint contains learned weights and information needed to process new windows consistently. |
| `app/dashboard.py` | Streamlit dashboard: DASIG benchmark and uploaded/local phone CSV check. | Visual interface for inspecting metrics, a sample signal, and phone-session results. |
| `src/live_server.py` | Standard-library HTTP server. Loads the checkpoint; serves the phone page; accepts phone windows for prediction; stores labeled sensor batches as CSV. | Laptop-side inference API and recording receiver. |
| `app/phone.html` | Browser UI. Requests motion permission, samples/resamples phone motion, sends windows to the laptop, shows scores, and uploads labeled recordings when stopped. | Phone sensor capture and live display. |
| `src/phone_data.py` | Reads saved/uploaded phone CSVs and converts stable labeled segments into 150×6 windows. | Prepares user recordings with the same sequence shape as the classifier input. |
| `report/experiment_results.json` | Machine-readable experiment result. | Source for stored test metrics and split information. |
| `report/experiment_results.md` | Human-readable metrics, setup, and limitations. | The numbers we report. |
| `docs/LEADERS_GUIDE.md` | Project overview, terms, demo, and limitations. | Group lead's project briefing. |
| `docs/LEADERS_VIVA_BANK.md` | Likely teacher/team questions with answer guidance. | Shared viva preparation. |
| `docs/CODE_EXPLANATION.md` | This file. | Code walk-through. |

## Important implementation snippets

### 1. Event cues become labels

In `src/dasig_data.py`, the event stream is inspected between consecutive green pick cues. The idea is:

```python
label = ABRUPT if red_or_buzzer_occurs_before_next_green else STANDARD
window = sensor_rows_starting_at_green_cue_for_three_seconds
```

The actual source handles DASIG's semicolon delimiters, decimal commas, sensor/event timestamp alignment, active forearm choice, interpolation/resampling, and excluded trials. The labels mark an event-derived window class; they are not exact human-labeled movement boundaries.

### 2. Subject split prevents overlap

In `src/dasig_data.py`, participants—not individual windows—are assigned to fitting, validation, and test groups:

```python
fit_subjects, validation_subjects, test_subjects = subject_split(all_subjects)
```

All trials/windows for a participant follow that participant's assignment. This guards against near-duplicate windows from one person appearing on both sides of evaluation.

### 3. CNN-LSTM input/output

In `src/model.py`, the network accepts a batch shaped like:

```text
(batch_size, 150 time_steps, 6 sensor_channels)
```

The convolution layer extracts short temporal patterns. The bidirectional LSTM processes the resulting sequence. A final linear layer produces two logits. During prediction, `softmax` turns the logits into two class scores that sum to 1.

### 4. Training stays separate from test

In `src/train.py`, normalization uses fitting participant data, and the optimizer updates weights only from fitting batches. Validation macro F1 guides checkpoint selection/early stopping. Held-out test data is used for the final reported metrics, not for weight updates.

Conceptually:

```python
mean, scale = fit_normalizer(X_fit)
X_fit = (X_fit - mean) / scale
model = train_on_fit_data(model, X_fit, y_fit)
checkpoint = choose_using_validation_macro_f1(model, validation_data)
test_metrics = evaluate_once_on_held_out_subjects(checkpoint, test_data)
```

### 5. Live phone inference

The browser in `app/phone.html` uses the DeviceMotion API. It assembles a six-value sensor row, interpolates readings onto a 50 Hz grid, and once a full 3-second window exists, sends it to `src/live_server.py`:

```javascript
fetch('/api/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({window: windowData})
});
```

The server validates the window shape and finite values, applies the checkpoint's training mean/scale, runs the model, and returns class scores. The laptop runs PyTorch inference; the browser captures and displays.

### 6. Phone recordings become check windows

While recording, the browser stores each sampled row as:

```text
time_s, label, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
```

On stop, it posts rows to `/api/phone/record`; the server writes them to `data/phone_records/<session>.csv`. `src/phone_data.py` groups contiguous same-label samples, ignores Unknown and gaps/label transitions, interpolates complete 3-second windows, then `app/dashboard.py` applies the saved model and compares predictions with the selected labels.

That score checks how this particular phone/user recording behaves. It is separate from the DASIG participant-held-out score because the device, placement, axis orientation, and sensor behavior differ.

## Local run commands

Run these from the project folder in PowerShell.

### Dataset benchmark/dashboard only

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
```

### Phone demo

Terminal 1:

```powershell
.\.venv\Scripts\python.exe -m src.live_server
```

Terminal 2:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
```

Then use Android Chrome on the laptop's printed Wi-Fi URL, grant motion access, record labeled movement, and review it in the phone tab. The Android Chrome local-insecure-origin setup is documented in `README.md`.

## Data and generated outputs

- Public raw DASIG CSVs: `data/raw/dasig/DASIG/`
- Trained model: `models/dasig_forearm_cnn_lstm.pt`
- Labeled phone sessions: `data/phone_records/`
- Benchmark report: `report/experiment_results.md` and `.json`

Keep raw published data credited to DASIG authors. Don't describe phone-session scores as part of DASIG's held-out test.
