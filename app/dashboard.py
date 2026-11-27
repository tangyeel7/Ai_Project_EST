from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.dasig_data import load_dasig
from src.metrics import classification_metrics, predict
from src.model import load_checkpoint
from src.phone_data import load_phone_csv, phone_windows

MODEL_PATH = ROOT / "models" / "dasig_forearm_cnn_lstm.pt"
DATA_PATH = ROOT / "data" / "raw" / "dasig" / "DASIG"
PHONE_PATH = ROOT / "data" / "phone_records"

st.set_page_config(page_title="Industrial wearable gesture recognition", page_icon="🏭", layout="wide")
st.title("Industrial wearable gesture recognition")
st.caption("CNN-LSTM · forearm accelerometer + gyroscope · standard vs abrupt industrial movement")


@st.cache_resource(max_entries=1, show_spinner="Loading DASIG windows and trained model…")
def load_system():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run `python -m src.train` first to create the DASIG model.")
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run `python -m src.download_dasig` first to get the DASIG CSV data.")
    data = load_dasig(DATA_PATH)
    model, state = load_checkpoint(MODEL_PATH)
    test_mask = np.isin(data["subjects"], state["test_subjects"])
    X_test = ((data["X"][test_mask] - state["mean"].numpy()) / state["scale"].numpy()).astype(np.float32)
    y_test = data["y"][test_mask]
    probabilities = predict(model, X_test)
    metrics = classification_metrics(y_test, probabilities.argmax(axis=1), data["class_names"])
    return data, model, state, test_mask, X_test, y_test, probabilities, metrics


try:
    data, model, state, test_mask, X_test, y_test, probabilities, metrics = load_system()
except Exception as error:
    st.error(str(error), icon="⚠️")
    st.stop()

st.info("DASIG records a pick-and-place task at a simulated workstation. Windows are labeled standard or abrupt from synchronized light/buzzer events; the alarm is a cue for the movement, not a predicted output.")
benchmark_tab, phone_tab = st.tabs(["DASIG benchmark", "Check my phone recording"])

with benchmark_tab:
    st.subheader("Held-out participant results")
    st.write(f"Accuracy **{metrics['accuracy']:.1%}** · Macro F1 **{metrics['macro_f1']:.1%}** · Balanced accuracy **{metrics['balanced_accuracy']:.1%}**")
    st.caption(f"Test subjects: {', '.join(state['test_subjects'])}. The split is by participant; no participant appears in both fitting and test data.")
    row_indices = np.flatnonzero(test_mask)
    choice = st.slider("Test window", 0, len(row_indices) - 1, 0)
    current = int(row_indices[choice])
    pred = int(probabilities[choice].argmax())
    with st.container(horizontal=True):
        st.metric("Predicted movement", data["class_names"][pred], border=True)
        st.metric("Event-derived label", data["class_names"][int(y_test[choice])], border=True)
        st.metric("Top model score", f"{probabilities[choice, pred]:.1%}", border=True)
        st.metric("Model-only latency", f"{state['latency_ms']:.2f} ms", border=True)
    signal = pd.DataFrame(data["X"][current], columns=data["channels"])
    signal.index = np.arange(len(signal)) / state["sample_rate_hz"]
    signal.index.name = "Time (seconds)"
    st.line_chart(signal, alt="Three-second forearm accelerometer and gyroscope test window")
    st.dataframe(pd.DataFrame(metrics["per_class"]), hide_index=True, width="stretch",
                 alt="Precision, recall, F1, and count for standard and abrupt movement classes")

with phone_tab:
    st.subheader("Evaluate a labeled phone recording")
    st.markdown("Start `python -m src.live_server`, then open the laptop's printed Wi-Fi address in Android Chrome on your phone. Start and stop sensing there; the labeled CSV is saved in `data/phone_records/`. Each complete three-second window with one consistent label is scored; Unknown rows and windows crossing label changes are skipped.")
    uploaded = st.file_uploader("Or upload a phone CSV", type=["csv"], key="phone_csv")
    recording_files = sorted(PHONE_PATH.glob("*.csv")) if PHONE_PATH.exists() else []
    local_path = None
    if recording_files:
        local_path = st.selectbox("Laptop recordings", recording_files,
                                  format_func=lambda path: path.name)
    chosen = uploaded if uploaded is not None else local_path
    if chosen is None:
        st.info("No phone recording found yet. Start `python -m src.live_server`, then record a labeled phone session from the browser page.")
    else:
        try:
            phone = phone_windows(pd.read_csv(uploaded)) if uploaded is not None else load_phone_csv(local_path)
            phone_X = ((phone["X"] - state["mean"].numpy()) / state["scale"].numpy()).astype(np.float32)
            phone_prob = predict(model, phone_X)
            phone_metrics = classification_metrics(phone["y"], phone_prob.argmax(axis=1), data["class_names"])
            with st.container(horizontal=True):
                st.metric("Labeled phone windows", len(phone["y"]), border=True)
                st.metric("Phone accuracy", f"{phone_metrics['accuracy']:.1%}", border=True)
                st.metric("Phone macro F1", f"{phone_metrics['macro_f1']:.1%}", border=True)
            st.dataframe(pd.DataFrame(phone_metrics["per_class"]), hide_index=True, width="stretch",
                         alt="Phone check precision, recall, F1, and class support")
            st.caption("This phone result is a separate device and placement check, not part of the DASIG test score. Phone/browser sensors differ from the dataset forearm IMUs.")
        except Exception as error:
            st.error(str(error), icon="⚠️")

with st.expander("What this model does and the limits"):
    st.markdown("""
- Each model window is 3 seconds of six-axis forearm IMU motion (accelerometer and gyroscope).
- The 1D CNN learns short movement patterns; the bidirectional LSTM learns their order through the window.
- The two labels are standard pick-and-place and an alarm-triggered abrupt response.
- DASIG uses 200 Hz sensors; this project averages/resamples windows to 50 Hz so the phone browser can provide comparable window lengths.
- Alarm event timing provides proxy labels. They are not precise hand annotations for the exact start and end of every movement.
- The dataset is from a simulated workstation, not a production factory. The phone check may shift by sensor placement, orientation, hardware, and participant.
""")
