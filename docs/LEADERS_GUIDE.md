# Project 8 leader's guide

## The project in one sentence

We train a CNN-LSTM on wearable inertial sensor time series from DASIG to distinguish a **Standard pick-and-place** window from an **Abrupt alarm-response** window, then demonstrate the trained model with live motion from a phone.

The submitted/main system is the single active-forearm model with six input channels. The five-sensor result is an additional comparison experiment, kept in `multi_sensor_comparison/`; it is not the model used by the main dashboard or phone demo.

## What problem are we solving?

The assigned idea is Bosch-style gesture recognition for factory automation. Our academic prototype studies movement patterns that could inform wearable human-machine interaction. The available dataset contains controlled pick-and-place trials and alarm-triggered abrupt responses. We use these as two movement classes because they are the industrial-style labels DASIG supports.

Be precise: this project does **not** predict equipment failure, recognize arbitrary commands such as Start/Stop/Left/Right, or control a robot. It is not a deployed factory safety system.

## What we made

1. A data pipeline that reads DASIG IMU and event CSV files, aligns events to sensor time, creates 3-second labeled windows, and holds out participants for evaluation.
2. A compact 1D CNN plus bidirectional LSTM classifier for six-axis acceleration and gyroscope sequences.
3. A trained model checkpoint and a report with per-class test results.
4. A Streamlit dashboard for the DASIG benchmark and a phone-recording check.
5. A browser page that reads a phone's accelerometer and gyroscope, shows rolling predictions, and saves a labeled CSV to the laptop.

## Vocabulary to explain

| Term | Plain explanation |
|---|---|
| IMU | Inertial measurement unit: sensors that measure acceleration and rotation. |
| Accelerometer | Measures linear acceleration along X, Y, and Z; readings may include gravity depending on the device/API. |
| Gyroscope | Measures angular speed around X, Y, and Z axes. |
| Time series | Measurements ordered by time rather than independent rows. |
| Window | A fixed-length slice of the signal. Here, 3 seconds represented by 150 samples. |
| Label | The target class assigned to a window: Standard or Abrupt. |
| CNN | Convolutional neural network; learns short/local patterns such as motion changes. |
| LSTM | A recurrent neural network that models how patterns change through a sequence. |
| CNN-LSTM | CNN extracts local signal patterns; LSTM uses their sequence and order. |
| Feature fusion | Combining information from multiple input channels/sensors. Here, the six axes from one active forearm IMU are fused by the network. |
| Class imbalance | One class has fewer examples. Abrupt windows are less common than Standard windows. |
| Class weighting | Gives more loss weight to examples from the less common class during training. |
| Augmentation | Small training-only changes (sensor jitter and gain) to help the model handle variation. |
| Subject-disjoint split | All windows from one person stay in one split, reducing leakage between train and test. |
| Accuracy | Fraction of all predictions that are correct. It can hide poor performance on a rare class. |
| Precision | Of windows predicted as a class, how many truly belong to it. |
| Recall | Of windows that truly belong to a class, how many the model found. |
| F1 | Balance of precision and recall. Macro F1 averages class F1 scores equally. |
| Balanced accuracy | Average recall across classes; gives each class equal importance. |
| Inference | Applying the trained model to new sensor windows to get predicted class scores. |
| Proxy label | A label inferred from a related signal—in this case, DASIG event cues—not a hand annotation of exact motion boundaries. |

## Dataset and labels

DASIG v3 contains 60 participants, each performing three trials at a simulated workstation. Its wearable setup includes five IMUs on the forearms, upper arms, and chest, with inertial signals captured at 200 Hz. Our pipeline selects the active forearm sensor for each trial and uses only its three acceleration and three gyroscope axes. The full project data contains 5,250 windows: 4,550 Standard and 700 Abrupt.

An Arduino/event log records visual/audio cue times, not a ground-truth label on every IMU row. We start a 3-second window at each green pick cue. If a red-light or buzzer cue occurs before the next green cue, the window is labeled Abrupt; otherwise it is Standard. The dataset authors identify five trials without an alarm-response movement; those trials are excluded. Since human reaction time is not zero, some windows can mix normal and response motion. State this limitation if asked.

We resample each window to 50 Hz, giving 150 time steps and six channels. We split by participant: 36 fitting, 12 validation, and 12 test participants. No participant's windows cross these groups.

## Model and training

- Input shape: `(batch, 150 time steps, 6 channels)`.
- A one-dimensional CNN learns local patterns in the sensor stream.
- A bidirectional LSTM reads the extracted sequence in both temporal directions.
- A final classifier returns two scores/probabilities: Standard and Abrupt.
- Training uses class-weighted cross-entropy, small sensor jitter and gain changes, and validation macro F1 for model selection/early stopping.
- Normalization statistics come from fitting participants, not the held-out test group.

## Results to quote accurately

The participant-held-out test has 1,080 windows from 12 participants. Results: **97.1% accuracy**, **93.7% macro F1**, and **93.3% balanced accuracy**. Standard F1 is 98.3%; Abrupt F1 is 89.1% (Abrupt precision 90.1%, recall 88.2%). Model-only CPU inference was approximately 0.62 ms per window on the development computer; that timing excludes sensor capture and communication.

These numbers are the DASIG benchmark results. They are not the accuracy of phone sensing. Phone placement, hardware, axes, gravity handling, and sampling behavior differ. Phone metrics should be described as a separate single-device demonstration/check.

## Phone demo: leader checklist

### On the laptop

1. From the project directory, start `.\.venv\Scripts\python.exe -m src.live_server`.
2. Note the local Wi-Fi address printed by the server, for example `http://192.168.1.20:8765`.
3. Allow the Python app through Windows Firewall on a private network if prompted.

### On Android

1. Put the laptop and phone on the same Wi-Fi.
2. In Android Chrome, open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, enable it, and enter the exact server origin (including scheme and port). Relaunch Chrome.
3. Open the server's Wi-Fi URL; allow motion access.
4. Select the expected movement, then tap **Allow motion access and start recording**.
5. Perform the movement for at least 3 seconds. Stop sensing to save the CSV on the laptop.

### Check results

Start Streamlit in a second laptop terminal with `.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py`. In **Check my phone recording**, choose the saved session. The model's predicted labels are compared with the labels selected by the person recording. Keep each movement label stable for several seconds; unknown and mixed-label windows are skipped.

This local Chrome flag is a development workaround for accessing motion sensors over a private HTTP address. Turn it off after the demo if it is no longer needed. Android Chrome is the documented path; iPhone Safari may require trusted HTTPS hosting for motion access.

## What the project does not establish

- General recognition of all factory gestures or commands.
- Production-line accuracy or safety readiness.
- Exact onset and end times for each movement.
- That the phone result transfers to other users or phone models.
- That DASIG's test score will hold on a different sensor location.

## Suggested group roles

- **Project lead/presenter:** explain scope, coordinate the demo, and keep benchmark results separate from phone results.
- **Data lead:** show one raw sensor trial, explain event-derived labeling, windowing, and participant split.
- **Model lead:** explain CNN, LSTM, class weighting, normalization, and metrics.
- **Demo lead:** run the laptop server, capture phone motion, and open the dashboard check.

Agree on one sentence all teammates can repeat: **“Our DASIG model classifies event-derived standard versus abrupt movement windows; phone sensing is a separate live demonstration.”**

## Useful project files

- `report/experiment_results.md`: exact experiment setup and metrics.
- `multi_sensor_comparison/reports/comparison_results.md`: separate single-versus-five-sensor experiment; use it only when discussing the comparison.
- `docs/LEADERS_VIVA_BANK.md`: teacher/team questions with answer guidance.
- `docs/CODE_EXPLANATION.md`: file-by-file explanation.
- `README.md`: setup and run commands.
