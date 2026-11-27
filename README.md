# Project 8: Industrial wearable movement recognition

This project uses a CNN-LSTM to classify 3-second movement windows as **Standard pick-and-place** or **Abrupt alarm-response movement**. It trains and evaluates on DASIG, then offers an optional live phone-sensor demo and a labeled phone recording check.

The primary project stays **single-sensor**: it uses six channels from one active forearm IMU. The five-sensor experiment is separate and retained only for comparison; it does not replace the primary model or phone demo.

## Run the DASIG project without a phone

From PowerShell in `E:\AI_For_Eng\EST_Project`, start the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
```

Open the local address printed by Streamlit, usually `http://localhost:8501`. The **DASIG benchmark** tab shows held-out participant metrics, a sample signal, and a prediction. If data or a trained model is missing, run:

```powershell
.\.venv\Scripts\python.exe -m src.download_dasig
.\.venv\Scripts\python.exe -m src.train
```

## Try live motion on an Android phone

The model server runs on your laptop; the phone browser reads the phone's motion sensors and sends 3-second windows to the laptop model.

1. Start the receiver in one PowerShell window:

   ```powershell
   .\.venv\Scripts\python.exe -m src.live_server
   ```

2. Note the Wi-Fi URL printed by the server, such as `http://192.168.1.20:8765`. Keep the laptop and phone on the same Wi-Fi.
3. On Android Chrome, open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`. Enable the flag, add the exact printed URL (including `http://` and `:8765`) to its origin list, and relaunch Chrome. This is a local development workaround that lets Chrome expose motion sensors on the laptop's plain-HTTP Wi-Fi page.
4. Open that same URL on the phone. Allow motion access. Choose the label matching the movement you are doing and tap **Allow motion access and start recording**.
5. Move for several seconds, then tap **Stop sensing and save recording**. Predictions appear live; the labeled CSV is saved in `data/phone_records/` on the laptop.
6. Start the Streamlit dashboard in another PowerShell window and open **Check my phone recording** to compare predictions with your selected labels.

Use Android Chrome for this setup. Browsers require a secure context to expose motion sensors; iPhone Safari on an ordinary local HTTP address may block them. Disable the Chrome flag when you finish the local demo if you do not need it.

The phone page is served by `src.live_server`; don't open `app/phone.html` directly as a file. If Windows Firewall asks, allow Python on your private network so the phone can reach the laptop.

## Dataset and model in brief

DASIG v3 contains industrial-style pick-and-place trials with forearm, upper-arm, and chest IMU data. This project uses one active forearm sensor: 3-axis acceleration and 3-axis gyroscope. Windows are 3 seconds (150 samples at 50 Hz). The split is by participant, preventing the same participant from appearing in both training and test groups.

DASIG supplies event cues rather than a dense label at every sensor timestamp. A window starting at a green pick cue is labeled Abrupt if a red-light or buzzer cue occurs before the next pick cue; otherwise it is Standard. The labels approximate movement type and do not mark exact movement onset.

The CNN learns local motion patterns, and the bidirectional LSTM models their order over time. On the held-out DASIG test group, the current model achieved 97.1% accuracy and 93.7% macro F1. These metrics describe the DASIG event-derived benchmark only. Phone placement, axes, sensors, and sampling differ, so phone results are a separate demo/check and may be lower.

## Documentation and results

- [Code explanations](docs/CODE_EXPLANATION.md): responsibilities and data flow for each project file.
- [Experiment results](report/experiment_results.md): detailed held-out metrics.
- [Five-sensor comparison project](multi_sensor_comparison/README.md): compares one forearm sensor with all five DASIG IMUs on matched windows.

## Scope

This is an academic proof of concept on a simulated workstation dataset. It recognizes two event-derived movement classes; it does not detect machine failure or classify general factory commands, and it has not been validated on a production line.
