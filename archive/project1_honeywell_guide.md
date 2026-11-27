# Honeywell Predictive Maintenance: AI for Engineers Project Guide

**Topic:** Neural-network-based predictive maintenance model for smart factories

**Problem statement:** Design a robust neural-network model to predict potential equipment failures in advance using multi-sensor heterogeneous data streams, incorporating anomaly detection, feature fusion, and real-time inference, so that maintenance can be scheduled optimally while minimizing false alarms and improving overall system efficiency.

---

## 1. What the Brief Is Really Asking For

| Phrase in the brief | What you must build |
|---|---|
| Neural-network-based predictive maintenance | A deep learning model that predicts failure, ideally Remaining Useful Life (RUL) or "failure within N cycles" |
| Multi-sensor heterogeneous data streams | Multiple sensor types (temperature, vibration, pressure, current) with different scales and sampling rates |
| Anomaly detection | A separate or combined unsupervised model (autoencoder) that flags abnormal behavior |
| Feature fusion | A way to combine sensor streams (early or late fusion, multi-branch network) |
| Real-time inference | A streaming simulation that scores new data window by window, with measured latency |
| Optimal scheduling, minimal false alarms | Threshold tuning, alarm logic, and a maintenance decision layer |

---

## 2. Dataset Options

- **NASA C-MAPSS turbofan (recommended).** 21 sensors, 3 operating settings, run-to-failure data, widely used and easy to cite. Start with FD001, then try FD003/FD004 for harder cases.
- **AI4I 2020 Predictive Maintenance (UCI).** Small, tabular, good as a quick baseline.
- **CWRU bearing dataset.** Vibration signals, good if you want a signal-processing angle (an ECE strength).
- **Best combination:** C-MAPSS as the main dataset, plus CWRU or AI4I for a "generalization" slide.

---

## 3. System Architecture

```
Raw sensor streams
      |
      v
Preprocessing (cleaning, normalization, sliding windows)
      |
      v
Feature engineering (rolling mean/std, FFT for vibration, health indicators)
      |
      v
+-----------------------+-------------------------+
| Anomaly branch        | Prognostic branch       |
| (Autoencoder)         | (CNN-LSTM / LSTM-GRU)   |
+-----------+-----------+------------+------------+
            |      Feature fusion    |
            +-----------+------------+
                        v
                Failure risk / RUL
                        |
                        v
     Alarm logic (threshold + persistence rule)
                        |
                        v
   Maintenance scheduler (risk vs. cost)
                        |
                        v
     Dashboard + real-time simulation
```

---

## 4. Models to Use

1. **Baselines:** Logistic Regression, Random Forest, XGBoost. You need these to show the neural network actually beats something.
2. **Anomaly detection:** LSTM or dense **Autoencoder** trained on healthy data only. Reconstruction error above a threshold means anomaly.
3. **Main predictor:** **1D-CNN + LSTM (or GRU)** on sliding windows (for example 30 timesteps). The CNN extracts local patterns per sensor and the LSTM captures degradation over time.
4. **Feature fusion:** A multi-branch network where each sensor group (thermal, pressure, vibration) has its own encoder and the outputs are concatenated, optionally with an **attention layer** to weight sensors. Compare against simple early fusion (all sensors stacked) in an ablation study.
5. **Optional extras:** Transformer encoder, Temporal Convolutional Network (TCN).

---

## 5. Tech Stack

| Purpose | Tools |
|---|---|
| Language | Python |
| Data and plotting | pandas, numpy, matplotlib, seaborn |
| Classical ML and metrics | scikit-learn, XGBoost |
| Deep learning | PyTorch (or TensorFlow/Keras) |
| Signal processing | scipy (FFT, filters) |
| Real-time simulation | Python loop feeding rows one at a time, or a Kafka-style queue simulation (optional) |
| Dashboard | Streamlit (fastest) or Gradio |
| Explainability | SHAP or attention weights |
| Training environment | Google Colab or Kaggle (free GPU) |
| Tracking and repo | GitHub, optionally MLflow or Weights & Biases |
| Export (optional) | ONNX or TorchScript to show deployment-readiness |

---

## 6. Specification Sheet

**Project title:** Multi-Sensor Neural Predictive Maintenance with Anomaly Detection and Real-Time Inference for Smart Factories

**Objective:** Predict equipment failure in advance with minimal false alarms and convert predictions into maintenance schedules.

| Item | Specification |
|---|---|
| Input | Multivariate time series, 14+ sensor channels, window length 30 to 50 |
| Output 1 | Anomaly score and anomaly flag |
| Output 2 | RUL estimate (regression) and/or failure within N cycles (classification) |
| Output 3 | Maintenance recommendation (Normal / Monitor / Schedule / Urgent) |
| Preprocessing | Drop constant sensors, per-condition normalization, sliding windows, RUL clipping (about 125) |
| Data split | Split by engine/unit, never by random rows (avoids leakage) |
| Anomaly model | Autoencoder trained on early-life healthy data |
| Prognostic model | CNN-LSTM with multi-branch fusion |
| Loss | MSE or Huber for RUL, weighted BCE for classification |
| Metrics (regression) | RMSE, MAE, NASA asymmetric score function |
| Metrics (classification) | Precision, recall, F1, PR-AUC, **false alarm rate** |
| Metrics (system) | Inference latency per window (target under 50 ms), model size |
| Real-time | Streaming simulator with rolling buffer |
| Alarm logic | Alarm only if risk exceeds the threshold for k consecutive windows (reduces false alarms) |
| Scheduler | Cost-based rule: schedule maintenance when expected failure cost exceeds maintenance cost |
| Deliverables | Code repo, trained model, dashboard, report, PPT, demo video |

**Targets to aim for (adjust to what you actually achieve):** RUL RMSE in the low-to-mid teens on FD001, false alarm rate under 5%, latency under 50 ms.

---

## 7. Project Checklist

### Phase 1: Setup
- [ ] Read the problem statement and list its 6 requirements
- [ ] Create a GitHub repo and folder structure (`data/`, `notebooks/`, `src/`, `models/`, `app/`, `report/`)
- [ ] Download the dataset and document its source

### Phase 2: Data
- [ ] Exploratory analysis: sensor trends, correlations, missing values
- [ ] Remove constant or useless sensors
- [ ] Normalize, create RUL labels, and build sliding windows
- [ ] Split by engine ID into train, validation and test

### Phase 3: Baselines
- [ ] Train Random Forest / XGBoost and record the metrics

### Phase 4: Anomaly Detection
- [ ] Train the autoencoder on healthy data
- [ ] Choose a threshold (percentile or validation-based)
- [ ] Plot anomaly score over an engine's lifetime, showing it rising before failure

### Phase 5: Main Neural Model
- [ ] Build CNN-LSTM
- [ ] Build the multi-branch fusion version (with attention if time permits)
- [ ] Ablation: no fusion vs early fusion vs multi-branch fusion
- [ ] Tune hyperparameters, use early stopping, save the best model

### Phase 6: Alarms and Scheduling
- [ ] Implement the k-consecutive-window rule
- [ ] Plot the false alarm vs missed detection trade-off (ROC / PR curve)
- [ ] Implement the maintenance decision logic with a cost example

### Phase 7: Real-Time
- [ ] Build the streaming simulator
- [ ] Measure and report latency
- [ ] Build the Streamlit dashboard (live sensor plot, risk gauge, recommendation)

### Phase 8: Evaluation and Explainability
- [ ] Final test-set results table (all models)
- [ ] SHAP or attention plots showing which sensors drive predictions
- [ ] Error analysis: where does the model fail?

### Phase 9: Submission
- [ ] Clean README with run instructions
- [ ] Report: problem, dataset, methodology, results, limitations, future work
- [ ] PPT and a 2-minute demo video
- [ ] Viva prep: be ready to explain every design choice

---

## 8. Likely Viva Questions

- Why did you split by engine and not randomly? (data leakage)
- Why LSTM or CNN instead of Random Forest? What did your results show?
- How does your fusion layer work and what did the ablation prove?
- How did you reduce false alarms?
- Why is the autoencoder trained only on healthy data?
- What would change when moving this to a real Honeywell factory? (sensor drift, missing data, edge deployment, retraining)

---

## 9. Suggested Timeline

| Days | Work |
|---|---|
| 1-2 | Setup, data, EDA, preprocessing |
| 3-4 | Baselines and autoencoder |
| 5-7 | CNN-LSTM, fusion, ablation |
| 8-9 | Alarm logic, scheduler, real-time simulator |
| 10 | Dashboard, explainability |
| 11-12 | Report, PPT, video, viva prep |

**If your deadline is shorter:** cut the Transformer, the CWRU second dataset, and ONNX export first. Keep the autoencoder, CNN-LSTM with fusion, alarm logic and the dashboard, since those map directly onto the brief.

---

## 10. Suggested Repo Structure

```
honeywell-predictive-maintenance/
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_baselines.ipynb
|   |-- 03_autoencoder.ipynb
|   `-- 04_cnn_lstm_fusion.ipynb
|-- src/
|   |-- data.py          # loading, windowing, normalization
|   |-- models.py        # autoencoder, CNN-LSTM, fusion
|   |-- train.py
|   |-- evaluate.py      # metrics, false alarm rate
|   |-- alarm.py         # k-consecutive-window logic
|   |-- scheduler.py     # cost-based maintenance decisions
|   `-- stream.py        # real-time simulator
|-- app/
|   `-- dashboard.py     # Streamlit app
|-- models/              # saved weights
|-- report/
`-- README.md
```
