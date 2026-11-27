"""
Generate publication-quality, visually pristine presentation figures for the EST Project PPT deck.
All figures are designed for 16:9 widescreen presentation slides with zero overlaps, zero clipping,
clean typography, generous whitespace, and professional color harmony.
Outputs high-resolution 300 DPI PNG graphics to both `figures/` and `ppt_figures/`.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Ensure output directories exist
for folder in ["figures", "ppt_figures"]:
    Path(folder).mkdir(parents=True, exist_ok=True)

# Styling palette
PALETTE = {
    "navy": "#1E3A8A",
    "blue": "#2563EB",
    "light_blue": "#DBEAFE",
    "teal": "#0D9488",
    "light_teal": "#CCFBF1",
    "emerald": "#059669",
    "light_green": "#DCFCE7",
    "crimson": "#DC2626",
    "light_red": "#FEE2E2",
    "amber": "#D97706",
    "light_amber": "#FEF3C7",
    "purple": "#7C3AED",
    "light_purple": "#F3E8FF",
    "slate_dark": "#0F172A",
    "slate_text": "#334155",
    "slate_light": "#F8FAFC",
    "card_bg": "#FFFFFF",
    "border": "#CBD5E1"
}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = PALETTE["border"]
plt.rcParams["axes.linewidth"] = 1.2


def save_figure(fig, name):
    for folder in ["figures", "ppt_figures"]:
        out_path = Path(folder) / f"{name}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Successfully rendered: {name}.png")
    plt.close(fig)


# ==============================================================================
# FIGURE 1: Dataset Architecture & Class Distribution
# ==============================================================================
def make_fig1_dataset_distribution():
    fig = plt.figure(figsize=(16, 8.5), facecolor=PALETTE["slate_light"])
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.25], wspace=0.28, top=0.86, bottom=0.12, left=0.08, right=0.92)

    # 1. Donut Chart for Class Balance
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(PALETTE["slate_light"])

    sizes = [4550, 700]
    labels = ["Standard Pick-and-Place\n4,550 windows (86.7%)", 
              "Abrupt Alarm Response\n700 windows (13.3%)"]
    colors = [PALETTE["blue"], PALETTE["crimson"]]
    explode = (0, 0.08)

    wedges, texts, autotexts = ax1.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct="%1.1f%%", pctdistance=0.76, startangle=140,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=3),
        textprops=dict(color=PALETTE["slate_dark"], fontsize=12, fontweight="bold")
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(13)
        at.set_fontweight("bold")

    ax1.text(0, 0.08, "5,250", ha="center", va="center", fontsize=22, fontweight="bold", color=PALETTE["navy"])
    ax1.text(0, -0.10, "Total Windows\n(3.0s @ 50 Hz)", ha="center", va="center", fontsize=10.5, fontweight="bold", color=PALETTE["slate_text"])
    
    ax1.set_title("Class Imbalance Ratio (~6.5 : 1)\nJustifies Class-Weighted Cross-Entropy Loss",
                  fontsize=13, fontweight="bold", color=PALETTE["navy"], pad=20)

    # 2. Participant & Split Distribution (Twin Y-Axis with clean separation)
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(PALETTE["card_bg"])

    categories = ["Train\n(Fitting Cohort)", "Validation\n(Model Selection)", "Held-Out Test\n(Final Benchmark)"]
    participants = [36, 12, 12]
    windows = [3135, 1035, 1080]

    x = np.arange(len(categories))
    width = 0.28

    # Left Y-Axis: Windows
    rects_win = ax2.bar(x - width/2 - 0.02, windows, width, label="3-Second Windows",
                        color=PALETTE["navy"], edgecolor="white", linewidth=1.5, zorder=3)
    ax2.set_ylabel("Number of Windows", fontsize=12, fontweight="bold", color=PALETTE["navy"])
    ax2.set_ylim(0, 4400)
    ax2.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    # Right Y-Axis: Participants
    ax2_twin = ax2.twinx()
    rects_sub = ax2_twin.bar(x + width/2 + 0.02, participants, width, label="Participants (Subjects)",
                             color=PALETTE["teal"], edgecolor="white", linewidth=1.5, zorder=3)
    ax2_twin.set_ylabel("Number of Participants", fontsize=12, fontweight="bold", color=PALETTE["teal"])
    ax2_twin.set_ylim(0, 55)

    # Staggered non-overlapping labels
    for rect in rects_win:
        h = rect.get_height()
        ax2.annotate(f"{int(h):,} win", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 7), textcoords="offset points", ha="center", va="bottom",
                     fontsize=10.5, fontweight="bold", color=PALETTE["navy"])

    for rect in rects_sub:
        h = rect.get_height()
        ax2_twin.annotate(f"{int(h)} subj", xy=(rect.get_x() + rect.get_width()/2, h),
                          xytext=(0, 7), textcoords="offset points", ha="center", va="bottom",
                          fontsize=10.5, fontweight="bold", color=PALETTE["teal"])

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=11, fontweight="bold", color=PALETTE["slate_dark"])
    ax2.set_title("Subject-Disjoint Split (60% Fit / 20% Val / 20% Test)\nStrict Zero-Leakage Evaluation Across 60 Unique Workers",
                  fontsize=13, fontweight="bold", color=PALETTE["navy"], pad=20)

    # Combined Legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True,
               facecolor="white", edgecolor=PALETTE["border"], fontsize=10.5)

    fig.suptitle("DASIG v3 Industrial Wearable Dataset Architecture",
                 fontsize=17, fontweight="bold", color=PALETTE["navy"], y=0.96)

    save_figure(fig, "01_dataset_class_distribution")


# ==============================================================================
# FIGURE 2: Subject-Disjoint Split vs Random Split (Data Leakage Defense)
# ==============================================================================
def make_fig2_split_leakage_defense():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8.5), facecolor=PALETTE["slate_light"])
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.05, right=0.95, wspace=0.15)

    for ax in [ax1, ax2]:
        ax.set_facecolor(PALETTE["card_bg"])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    # ------------------ LEFT: Flawed Random Split ------------------
    ax1.add_patch(patches.FancyBboxPatch((0.4, 0.4), 9.2, 9.2, boxstyle="round,pad=0.2",
                                         facecolor=PALETTE["card_bg"], edgecolor=PALETTE["crimson"], linewidth=2.5))
    
    # Header banner
    ax1.add_patch(patches.FancyBboxPatch((0.7, 8.3), 8.6, 1.0, boxstyle="round,pad=0.15",
                                         facecolor=PALETTE["light_red"], edgecolor=PALETTE["crimson"], linewidth=1.5))
    ax1.text(5.0, 8.95, "FLAWED: Naive Random Window Split", ha="center", va="center",
             fontsize=13.5, fontweight="bold", color=PALETTE["crimson"])
    ax1.text(5.0, 8.55, "[X] High Risk of Overfitting & Artificial Test Accuracy", ha="center", va="center",
             fontsize=10.5, fontweight="bold", color=PALETTE["slate_text"])

    # Worker stream rows
    worker_names = ["Operator #1 Stream", "Operator #2 Stream", "Operator #3 Stream", "Operator #4 Stream"]
    for i, wname in enumerate(worker_names):
        y = 6.8 - i * 1.15
        ax1.add_patch(patches.Rectangle((0.8, y), 8.4, 0.75, facecolor="#F1F5F9", edgecolor="#CBD5E1", lw=1.2))
        ax1.text(1.1, y + 0.38, wname, va="center", fontsize=9.5, fontweight="bold", color=PALETTE["slate_dark"])
        
        ax1.add_patch(patches.Rectangle((4.8, y + 0.1), 1.6, 0.55, facecolor=PALETTE["blue"], alpha=0.85))
        ax1.text(5.6, y + 0.38, "Train", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        
        ax1.add_patch(patches.Rectangle((6.8, y + 0.1), 1.6, 0.55, facecolor=PALETTE["crimson"], alpha=0.85))
        ax1.text(7.6, y + 0.38, "Test", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")

    # Bottom explanation box (with ample padding, clean margins)
    ax1.add_patch(patches.FancyBboxPatch((0.7, 0.6), 8.6, 1.9, boxstyle="round,pad=0.2",
                                         facecolor=PALETTE["light_red"], edgecolor=PALETTE["crimson"], lw=1.2))
    ax1.text(5.0, 2.15, "CRITICAL DATA LEAKAGE FLAW", ha="center", va="center",
             fontsize=10.5, fontweight="bold", color=PALETTE["crimson"])
    flaw_lines = [
        "• Consecutive windows from the same worker exist in BOTH splits.",
        "• The model memorizes personal anatomy, gait, and sensor orientation.",
        "• Results in artificially high scores that fail completely on new operators!"
    ]
    for idx, line in enumerate(flaw_lines):
        ax1.text(1.0, 1.70 - idx * 0.40, line, va="center", fontsize=9.2, fontweight="bold", color=PALETTE["slate_dark"])

    # ------------------ RIGHT: Subject-Disjoint Split (OUR RIGOROUS APPROACH) ------------------
    ax2.add_patch(patches.FancyBboxPatch((0.4, 0.4), 9.2, 9.2, boxstyle="round,pad=0.2",
                                         facecolor=PALETTE["card_bg"], edgecolor=PALETTE["emerald"], linewidth=2.5))
    
    # Header banner
    ax2.add_patch(patches.FancyBboxPatch((0.7, 8.3), 8.6, 1.0, boxstyle="round,pad=0.15",
                                         facecolor=PALETTE["light_green"], edgecolor=PALETTE["emerald"], linewidth=1.5))
    ax2.text(5.0, 8.95, "RIGOROUS: Subject-Disjoint Split (Our Method)", ha="center", va="center",
             fontsize=13.5, fontweight="bold", color=PALETTE["emerald"])
    ax2.text(5.0, 8.55, "[OK] Zero Information Leakage to Unseen Human Workers", ha="center", va="center",
             fontsize=10.5, fontweight="bold", color=PALETTE["slate_text"])

    # 3 distinct subject groups
    groups = [
        ("Fitting Cohort (36 Subjects)", "60% of Cohort", "sub003 ... sub060 (3,135 windows)", PALETTE["navy"], PALETTE["light_blue"], 6.6),
        ("Validation Cohort (12 Subjects)", "20% of Cohort", "sub005 ... sub056 (1,035 windows)", PALETTE["teal"], PALETTE["light_teal"], 5.0),
        ("Held-Out Test Cohort (12 Subjects)", "20% of Cohort", "sub001, sub002, sub009... (1,080 windows)", PALETTE["emerald"], PALETTE["light_green"], 3.4)
    ]
    for title, pct, members, col, bg_col, y_pos in groups:
        ax2.add_patch(patches.FancyBboxPatch((0.8, y_pos), 8.4, 1.25, boxstyle="round,pad=0.15",
                                             facecolor=bg_col, edgecolor=col, lw=1.8))
        ax2.text(1.2, y_pos + 0.85, title, fontsize=11, fontweight="bold", color=col)
        ax2.text(8.8, y_pos + 0.85, pct, ha="right", fontsize=10, fontweight="bold", color=col)
        ax2.text(1.2, y_pos + 0.35, members, fontsize=9.2, fontweight="bold", color=PALETTE["slate_text"])

    # Bottom explanation box (with ample padding, clean margins)
    ax2.add_patch(patches.FancyBboxPatch((0.7, 0.6), 8.6, 1.9, boxstyle="round,pad=0.2",
                                         facecolor=PALETTE["light_green"], edgecolor=PALETTE["emerald"], lw=1.2))
    ax2.text(5.0, 2.15, "PROVEN GENERALIZATION RIGOR", ha="center", va="center",
             fontsize=10.5, fontweight="bold", color=PALETTE["emerald"])
    rigor_lines = [
        "• All windows from any given person remain strictly within one partition.",
        "• Normalization statistics (mean, std) are computed ONLY on the 36 fit subjects.",
        "• Validates whether the AI accurately transfers to unseen human operators!"
    ]
    for idx, line in enumerate(rigor_lines):
        ax2.text(1.0, 1.70 - idx * 0.40, line, va="center", fontsize=9.2, fontweight="bold", color=PALETTE["slate_dark"])

    fig.suptitle("Data Integrity: Why Subject Holdout is Essential for Wearable AI",
                 fontsize=17, fontweight="bold", color=PALETTE["navy"], y=0.96)

    save_figure(fig, "02_subject_disjoint_split_pipeline")


# ==============================================================================
# FIGURE 3: Authentic Sensor Time-Series Waveforms (Standard vs Abrupt)
# ==============================================================================
def make_fig3_sensor_waveforms():
    from src.dasig_data import _read_mimu, _read_events, _resample_window
    
    mimu_path = Path("data/raw/dasig/DASIG/sub001/sub001_FR_R_MIMU.csv")
    ard_path = Path("data/raw/dasig/DASIG/sub001/sub001_FR_R_Arduino.csv")
    
    t_imu, signal = _read_mimu(mimu_path, "right")
    t_ev, ev = _read_events(ard_path)
    
    cues = np.flatnonzero((ev >= 2) & (ev <= 5))
    alarms = t_ev[(ev >= 6) & (ev <= 10)]
    
    std_window, abr_window = None, None
    for pos, cue_idx in enumerate(cues):
        cue_time = float(t_ev[cue_idx])
        next_cue = float(t_ev[cues[pos + 1]]) if pos + 1 < len(cues) else cue_time + 3.0
        has_alarm = np.any((alarms >= cue_time) & (alarms < next_cue))
        win = _resample_window(t_imu, signal, cue_time)
        if win is not None:
            if not has_alarm and std_window is None:
                std_window = win
            elif has_alarm and abr_window is None:
                abr_window = win
        if std_window is not None and abr_window is not None:
            break

    time_axis = np.linspace(0, 3.0, 150)

    fig, axes = plt.subplots(2, 2, figsize=(16, 8.5), facecolor=PALETTE["slate_light"], sharex=True)
    plt.subplots_adjust(top=0.88, bottom=0.10, left=0.08, right=0.95, hspace=0.32, wspace=0.18)

    c_x, c_y, c_z = "#EF4444", "#10B981", "#2563EB"

    # [0, 0]: Standard Accel
    ax = axes[0, 0]
    ax.set_facecolor(PALETTE["card_bg"])
    ax.plot(time_axis, std_window[:, 0], label="Acc X", color=c_x, lw=2.0)
    ax.plot(time_axis, std_window[:, 1], label="Acc Y", color=c_y, lw=2.0)
    ax.plot(time_axis, std_window[:, 2], label="Acc Z", color=c_z, lw=2.0)
    ax.set_title("Standard Pick-and-Place: Accelerometer (m/s²)\n[Smooth, Periodic Workstation Motion]",
                 fontsize=12, fontweight="bold", color=PALETTE["blue"], pad=10)
    ax.set_ylabel("Linear Acc (m/s²)", fontsize=11, fontweight="bold", color=PALETTE["slate_text"])
    ax.set_ylim(-12, 18)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", ncol=3, fontsize=9.5, frameon=True, facecolor="white")

    # [1, 0]: Standard Gyro
    ax = axes[1, 0]
    ax.set_facecolor(PALETTE["card_bg"])
    ax.plot(time_axis, std_window[:, 3], label="Gyro X", color=c_x, lw=2.0)
    ax.plot(time_axis, std_window[:, 4], label="Gyro Y", color=c_y, lw=2.0)
    ax.plot(time_axis, std_window[:, 5], label="Gyro Z", color=c_z, lw=2.0)
    ax.set_title("Standard Pick-and-Place: Gyroscope (rad/s)\n[Controlled Forearm Rotation]",
                 fontsize=12, fontweight="bold", color=PALETTE["blue"], pad=10)
    ax.set_xlabel("Time (seconds) [150 samples @ 50 Hz]", fontsize=11, fontweight="bold", color=PALETTE["slate_text"])
    ax.set_ylabel("Angular Vel (rad/s)", fontsize=11, fontweight="bold", color=PALETTE["slate_text"])
    ax.set_ylim(-2.5, 3.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", ncol=3, fontsize=9.5, frameon=True, facecolor="white")

    # [0, 1]: Abrupt Accel
    ax = axes[0, 1]
    ax.set_facecolor(PALETTE["card_bg"])
    ax.axvspan(1.0, 2.2, color="#FECDD3", alpha=0.45, label="Alarm Response Period")
    ax.plot(time_axis, abr_window[:, 0], label="Acc X", color=c_x, lw=2.0)
    ax.plot(time_axis, abr_window[:, 1], label="Acc Y", color=c_y, lw=2.0)
    ax.plot(time_axis, abr_window[:, 2], label="Acc Z", color=c_z, lw=2.0)
    ax.set_title("Abrupt Alarm Response: Accelerometer (m/s²)\n[Sharp Startle Jerk & Rapid Arm Retraction]",
                 fontsize=12, fontweight="bold", color=PALETTE["crimson"], pad=10)
    ax.set_ylim(-16, 22)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", ncol=4, fontsize=9.5, frameon=True, facecolor="white")

    # [1, 1]: Abrupt Gyro
    ax = axes[1, 1]
    ax.set_facecolor(PALETTE["card_bg"])
    ax.axvspan(1.0, 2.2, color="#FECDD3", alpha=0.45, label="Alarm Response Period")
    ax.plot(time_axis, abr_window[:, 3], label="Gyro X", color=c_x, lw=2.0)
    ax.plot(time_axis, abr_window[:, 4], label="Gyro Y", color=c_y, lw=2.0)
    ax.plot(time_axis, abr_window[:, 5], label="Gyro Z", color=c_z, lw=2.0)
    ax.set_title("Abrupt Alarm Response: Gyroscope (rad/s)\n[Violent Forearm Angular Deflection]",
                 fontsize=12, fontweight="bold", color=PALETTE["crimson"], pad=10)
    ax.set_xlabel("Time (seconds) [150 samples @ 50 Hz]", fontsize=11, fontweight="bold", color=PALETTE["slate_text"])
    ax.set_ylim(-5.5, 4.5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", ncol=4, fontsize=9.5, frameon=True, facecolor="white")

    fig.suptitle("Raw Kinematic Comparison: 6-Axis Forearm IMU Windows (DASIG sub001)",
                 fontsize=17, fontweight="bold", color=PALETTE["navy"], y=0.96)

    save_figure(fig, "03_sensor_signals_time_series")


# ==============================================================================
# FIGURE 4: Neural Network Architecture Diagram (Clean Layout)
# ==============================================================================
def make_fig4_model_architecture():
    fig, ax = plt.subplots(figsize=(16, 8.5), facecolor=PALETTE["slate_light"])
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.04, right=0.96)

    ax.set_facecolor(PALETTE["slate_light"])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # 5 Sequential Blocks
    blocks = [
        {"x": 0.5, "w": 2.5, "title": "1. Input Window", 
         "lines": ["Shape: (B, 150, 6)", "3-Axis Accel (X,Y,Z)", "3-Axis Gyro (X,Y,Z)", "Resampled to 50 Hz", "Duration: 3.0 sec"], 
         "col": PALETTE["navy"], "bg": PALETTE["light_blue"]},
        
        {"x": 3.4, "w": 2.7, "title": "2. Conv1D Block 1", 
         "lines": ["Conv1D(6 -> 32, k=5)", "BatchNorm1D(32)", "ReLU Activation", "MaxPool1D(kernel=2)", "Out: (B, 32, 75)"], 
         "col": PALETTE["blue"], "bg": "#EFF6FF"},
        
        {"x": 6.5, "w": 2.7, "title": "3. Conv1D Block 2", 
         "lines": ["Conv1D(32 -> 64, k=3)", "BatchNorm1D(64)", "ReLU Activation", "Local Feature Map", "Out: (B, 64, 75)"], 
         "col": PALETTE["teal"], "bg": PALETTE["light_teal"]},
        
        {"x": 9.6, "w": 2.8, "title": "4. Bi-LSTM Layer", 
         "lines": ["LSTM(in=64, hidden=48)", "Bidirectional=True", "Forward (48) + Back (48)", "Concat Last Hidden", "Out: (B, 96)"], 
         "col": PALETTE["purple"], "bg": PALETTE["light_purple"]},
        
        {"x": 12.8, "w": 2.7, "title": "5. Classifier Head", 
         "lines": ["Dropout(rate=0.3)", "Linear(96 -> 2)", "Softmax Probabilities", "Dual Class Logits", "Out: (B, 2)"], 
         "col": PALETTE["crimson"], "bg": PALETTE["light_red"]}
    ]

    for b in blocks:
        # Main card
        ax.add_patch(patches.FancyBboxPatch((b["x"], 2.2), b["w"], 4.5, boxstyle="round,pad=0.15",
                                             facecolor="white", edgecolor=b["col"], linewidth=2.5))
        # Header banner
        ax.add_patch(patches.FancyBboxPatch((b["x"]+0.05, 5.85), b["w"]-0.1, 0.75, boxstyle="round,pad=0.1",
                                             facecolor=b["col"], edgecolor=b["col"]))
        ax.text(b["x"] + b["w"]/2, 6.22, b["title"], ha="center", va="center",
                color="white", fontsize=11.5, fontweight="bold")
        
        # Text bullet points
        for idx, line in enumerate(b["lines"]):
            y_txt = 5.2 - idx * 0.65
            ax.text(b["x"] + b["w"]/2, y_txt, line, ha="center", va="center",
                    fontsize=10, fontweight="bold" if idx == len(b["lines"])-1 else "normal",
                    color=b["col"] if idx == len(b["lines"])-1 else PALETTE["slate_dark"])

    # Connecting directional arrows between cards
    arrow_pairs = [(0.5+2.5, 3.4), (3.4+2.7, 6.5), (6.5+2.7, 9.6), (9.6+2.8, 12.8)]
    for x_start, x_end in arrow_pairs:
        ax.annotate("", xy=(x_end - 0.08, 4.45), xytext=(x_start + 0.08, 4.45),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["navy"], lw=2.5, mutation_scale=18))

    # Bottom Stage Annotations
    ax.add_patch(patches.FancyBboxPatch((0.5, 0.6), 15.0, 1.1, boxstyle="round,pad=0.15",
                                         facecolor="white", edgecolor=PALETTE["border"], linewidth=1.5))
    ax.text(8.0, 1.15, "Total Model Parameters: 94,146 | Single Forearm IMU (6 Channels) | Ultra-Fast CPU Latency: 0.62 ms",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color=PALETTE["navy"])
    ax.text(8.0, 0.82, "1D CNN detects local kinematic jerks/rotations  -->  Bi-LSTM models full 3-second sequence trajectory  -->  Softmax outputs probabilities",
            ha="center", va="center", fontsize=10, color=PALETTE["slate_text"])

    fig.suptitle("ActivityCNNLSTM: Compact Temporal Deep Sequence Architecture",
                 fontsize=17, fontweight="bold", color=PALETTE["navy"], y=0.96)

    save_figure(fig, "04_system_architecture_diagram")


# ==============================================================================
# FIGURE 5: Test Performance Benchmark Metrics
# ==============================================================================
def make_fig5_benchmark_metrics():
    fig = plt.figure(figsize=(16, 8.5), facecolor=PALETTE["slate_light"])
    gs = GridSpec(1, 2, width_ratios=[1.3, 1.0], wspace=0.25, top=0.86, bottom=0.10, left=0.08, right=0.94)

    # 1. Main Metrics Bar Chart
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(PALETTE["card_bg"])

    metrics = [
        "Overall\nAccuracy",
        "Macro\nF1-Score",
        "Balanced\nAccuracy",
        "Standard\nClass F1",
        "Abrupt\nClass F1",
        "Abrupt\nRecall"
    ]
    values = [97.13, 93.73, 93.35, 98.35, 89.12, 88.19]
    colors = [PALETTE["navy"], PALETTE["blue"], PALETTE["teal"], PALETTE["emerald"], PALETTE["crimson"], PALETTE["amber"]]

    bars = ax1.bar(metrics, values, color=colors, width=0.55, edgecolor="white", linewidth=1.5, zorder=3)
    ax1.set_ylim(80, 105)
    ax1.set_ylabel("Score (%)", fontsize=12, fontweight="bold", color=PALETTE["slate_text"])
    ax1.set_title("Held-Out Test Benchmark (12 Unseen Subjects, 1,080 Windows)",
                  fontsize=13, fontweight="bold", color=PALETTE["navy"], pad=15)
    ax1.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    for bar, val in zip(bars, values):
        ax1.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width()/2, val),
                     xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                     fontsize=11.5, fontweight="bold", color=PALETTE["slate_dark"])

    # 2. Performance Summary Card & Latency
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(PALETTE["slate_light"])
    ax2.axis("off")

    # Explicit transAxes with clip_on=False ensures clean visible borders!
    ax2.add_patch(patches.FancyBboxPatch((0.05, 0.05), 0.90, 0.90, transform=ax2.transAxes,
                                         boxstyle="round,pad=0.02", facecolor="white",
                                         edgecolor=PALETTE["navy"], linewidth=2.5, clip_on=False))
    
    ax2.text(0.5, 0.90, "Quantitative Benchmark Highlights", transform=ax2.transAxes,
             ha="center", va="center", fontsize=14, fontweight="bold", color=PALETTE["navy"])
    
    items = [
        ("Test Cohort Size", "12 Participants (1,080 windows)", PALETTE["navy"]),
        ("Standard Precision / Recall", "98.2% / 98.5%", PALETTE["blue"]),
        ("Abrupt Precision / Recall", "90.1% / 88.2%", PALETTE["crimson"]),
        ("Trainable Parameters", "94,146 parameters", PALETTE["purple"]),
        ("CPU Inference Latency", "0.62 ms / window", PALETTE["emerald"]),
        ("Real-Time Throughput", "~1,610 windows / sec", PALETTE["emerald"])
    ]
    for idx, (title, val, col) in enumerate(items):
        y_p = 0.76 - idx * 0.12
        ax2.text(0.10, y_p, title, transform=ax2.transAxes, fontsize=10.5, fontweight="bold", color=PALETTE["slate_text"])
        ax2.text(0.10, y_p - 0.045, val, transform=ax2.transAxes, fontsize=12, fontweight="bold", color=col)

    fig.suptitle("Quantitative Test Benchmark on DASIG Subject Holdout",
                 fontsize=17, fontweight="bold", color=PALETTE["navy"], y=0.96)

    save_figure(fig, "05_test_performance_metrics")


# ==============================================================================
# FIGURE 6: Test Confusion Matrix Heatmap (Cleaned with Safe Margins)
# ==============================================================================
def make_fig6_confusion_matrix():
    fig, ax = plt.subplots(figsize=(11, 9.5), facecolor=PALETTE["slate_light"])
    plt.subplots_adjust(top=0.88, bottom=0.18, left=0.20, right=0.92)

    ax.set_facecolor(PALETTE["card_bg"])

    cm = np.array([
        [922, 14],   # Actual Standard
        [17, 127]    # Actual Abrupt
    ])

    # Plot base matrix
    im = ax.imshow([[0.95, 0.10], [0.10, 0.85]], cmap="Blues", alpha=0.35, vmin=0, vmax=1)

    labels = ["Standard\nPick-and-Place", "Abrupt\nAlarm Response"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Standard\n(Normal Assembly)", "Predicted Abrupt\n(Alarm Response)"], 
                       fontsize=12, fontweight="bold", color=PALETTE["slate_dark"])
    ax.set_yticklabels(["Actual Standard\n(936 Windows)", "Actual Abrupt\n(144 Windows)"], 
                       fontsize=12, fontweight="bold", color=PALETTE["slate_dark"])

    # Remove protruding tick marks
    ax.tick_params(length=0, pad=12)

    # Text annotations in cells
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            row_sum = cm[i, :].sum()
            pct = val / row_sum * 100
            
            # Diagonal: Correct predictions
            if i == j:
                col = PALETTE["navy"] if i == 0 else PALETTE["crimson"]
                rect = patches.FancyBboxPatch((j-0.42, i-0.42), 0.84, 0.84, boxstyle="round,pad=0.05",
                                              fill=True, color=col, alpha=0.88, zorder=2)
                ax.add_patch(rect)
                txt_col = "white"
            else:
                rect = patches.FancyBboxPatch((j-0.42, i-0.42), 0.84, 0.84, boxstyle="round,pad=0.05",
                                              fill=True, color="#FEE2E2", alpha=0.90, zorder=2)
                ax.add_patch(rect)
                txt_col = PALETTE["crimson"]

            ax.text(j, i - 0.08, f"{val:,}", ha="center", va="center",
                    color=txt_col, fontsize=22, fontweight="bold", zorder=3)
            ax.text(j, i + 0.14, f"({pct:.1f}% of row)", ha="center", va="center",
                    color=txt_col, fontsize=11.5, fontweight="bold", zorder=3)

    ax.set_title("Held-Out Test Set Confusion Matrix (N = 1,080 Windows)\nStandard Recall: 98.5% | Abrupt Recall: 88.2%",
                 fontsize=14, fontweight="bold", color=PALETTE["navy"], pad=20)
    
    # Bottom summary callout with safe margin (well below x-axis labels)
    plt.figtext(0.56, 0.05, "True Standard: 922  |  False Abrupt: 14  |  False Standard: 17  |  True Abrupt: 127",
                ha="center", fontsize=11, fontweight="bold", color=PALETTE["slate_text"],
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=PALETTE["border"], lw=1.2))

    save_figure(fig, "06_test_confusion_matrix")


# ==============================================================================
# FIGURE 7: Single Forearm vs 5-Sensor Fusion Comparison (Cleaned Layout)
# ==============================================================================
def make_fig7_sensor_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8.5), facecolor=PALETTE["slate_light"])
    plt.subplots_adjust(top=0.86, bottom=0.10, left=0.07, right=0.95, wspace=0.22)

    # 1. Bar chart comparison
    ax1.set_facecolor(PALETTE["card_bg"])

    metrics = ["Overall\nAccuracy", "Macro\nF1-Score", "Balanced\nAccuracy", "Abrupt\nPrecision", "Abrupt\nRecall"]
    single_vals = [96.76, 93.01, 93.14, 87.59, 88.19]
    fused_vals  = [93.80, 87.61, 90.84, 72.25, 86.81]

    x = np.arange(len(metrics))
    width = 0.35

    rects1 = ax1.bar(x - width/2, single_vals, width, label="Single Forearm (6 channels)",
                     color=PALETTE["blue"], edgecolor="white", linewidth=1.5, zorder=3)
    rects2 = ax1.bar(x + width/2, fused_vals, width, label="5-Sensor Fusion (30 channels)",
                     color=PALETTE["amber"], edgecolor="white", linewidth=1.5, zorder=3)

    ax1.set_ylim(60, 110)
    ax1.set_ylabel("Score (%)", fontsize=12, fontweight="bold", color=PALETTE["slate_text"])
    ax1.set_title("Single IMU vs. 5-Sensor Full Body Early Fusion\n(Same Held-Out Test Split, 1,080 Windows)",
                  fontsize=12.5, fontweight="bold", color=PALETTE["navy"], pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=10.5, fontweight="bold", color=PALETTE["slate_dark"])
    ax1.legend(loc="upper right", frameon=True, facecolor="white", fontsize=10.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    # Annotate differences safely above each pair with a clean pill badge
    for r1, r2, s, f in zip(rects1, rects2, single_vals, fused_vals):
        diff = f - s
        max_h = max(s, f)
        badge_col = PALETTE["crimson"] if diff < 0 else PALETTE["emerald"]
        ax1.annotate(f"{diff:+.1f}%", xy=(r2.get_x() + r2.get_width()/2, max_h),
                     xytext=(0, 10), textcoords="offset points", ha="center", va="bottom",
                     fontsize=9.5, fontweight="bold", color=badge_col,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=badge_col, lw=1.2))

    # 2. Insights card
    ax2.set_facecolor(PALETTE["slate_light"])
    ax2.axis("off")
    
    # Visible card border using transAxes & clip_on=False
    ax2.add_patch(patches.FancyBboxPatch((0.05, 0.05), 0.90, 0.90, transform=ax2.transAxes,
                                         boxstyle="round,pad=0.02", facecolor="white",
                                         edgecolor=PALETTE["teal"], linewidth=2.5, clip_on=False))

    ax2.text(0.5, 0.90, "Key Empirical Ablation Insights", transform=ax2.transAxes,
             ha="center", va="center", fontsize=14, fontweight="bold", color=PALETTE["navy"])

    insights = [
        ("1. Why does Single Sensor Win?", 
         "The active forearm IMU directly captures the manipulation actuation.\nUpper arms & sternum IMUs add extraneous torso motion noise."),
        
        ("2. Abrupt Precision Collapse in Fusion", 
         "5-Sensor fusion suffers a massive -15.3% drop in Abrupt Precision (72.3% vs 87.6%).\nIncidental body shifts trigger false alarm detections."),
        
        ("3. Computational & Hardware Efficiency", 
         "Single sensor runs in 0.66 ms latency (~14% faster than 30-ch fusion).\nRequires only 1 device on worker rather than 5 full-body harnesses."),
        
        ("4. Real-World Factory Feasibility", 
         "Single forearm wearable is ergonomic, lightweight, less invasive, and\nproves superior in generalizing across unseen human operators.")
    ]

    for idx, (heading, desc) in enumerate(insights):
        y_pos = 0.75 - idx * 0.18
        ax2.text(0.10, y_pos, heading, transform=ax2.transAxes, fontsize=11, fontweight="bold", color=PALETTE["teal"])
        ax2.text(0.10, y_pos - 0.065, desc, transform=ax2.transAxes, fontsize=9.8, color=PALETTE["slate_dark"], linespacing=1.35)

    fig.suptitle("Empirical Study: Does Adding More Wearable Sensors Improve Movement Classification?",
                 fontsize=17, fontweight="bold", color=PALETTE["navy"], y=0.96)

    save_figure(fig, "07_single_vs_five_sensor_comparison")


# ==============================================================================
# FIGURE 8: Live Demonstration & Real-Time Deployment Pipeline (Clean Layout)
# ==============================================================================
def make_fig8_live_demo_architecture():
    fig, ax = plt.subplots(figsize=(16, 8.5), facecolor=PALETTE["slate_light"])
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.04, right=0.96)

    ax.set_facecolor(PALETTE["slate_light"])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    stages = [
        {
            "x": 0.5, "w": 3.9, "title": "1. Mobile Edge Client",
            "body": [
                "• Android Chrome Webview",
                "• HTML5 DeviceMotionEvent API",
                "• 50 Hz Hardware IMU Sampling",
                "• 6 Channels: Linear Accel + Gyro",
                "• Ground-Truth Label Selector",
                "• Client-side rolling window buffer"
            ],
            "col": PALETTE["blue"]
        },
        {
            "x": 6.0, "w": 4.0, "title": "2. Local Wi-Fi Bridge",
            "body": [
                "• Python WebSocket / HTTP Server",
                "• Dedicated Port (e.g. :8765)",
                "• Chrome Secure Origin Flag enabled",
                "• 150-Sample FIFO Rolling Buffer",
                "• CSV Session Recording Logger",
                "• Multi-client thread-safe routing"
            ],
            "col": PALETTE["teal"]
        },
        {
            "x": 11.6, "w": 3.9, "title": "3. Laptop AI & Dashboard",
            "body": [
                "• PyTorch ActivityCNNLSTM Model",
                "• Fast CPU Inference (~0.62 ms)",
                "• Streamlit Interactive Dashboard",
                "• Live Probability Meters (0-100%)",
                "• Raw Waveform Real-Time Plotting",
                "• Recorded Session Verification Tab"
            ],
            "col": PALETTE["purple"]
        }
    ]

    for s in stages:
        # Box
        ax.add_patch(patches.FancyBboxPatch((s["x"], 2.2), s["w"], 4.5, boxstyle="round,pad=0.15",
                                             facecolor="white", edgecolor=s["col"], linewidth=2.5))
        # Header banner
        ax.add_patch(patches.FancyBboxPatch((s["x"]+0.05, 5.85), s["w"]-0.1, 0.75, boxstyle="round,pad=0.1",
                                             facecolor=s["col"], edgecolor=s["col"]))
        ax.text(s["x"] + s["w"]/2, 6.22, s["title"], ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")
        
        # Text lines
        for idx, line in enumerate(s["body"]):
            y_txt = 5.2 - idx * 0.52
            ax.text(s["x"] + 0.35, y_txt, line, va="center",
                    fontsize=10.2, color=PALETTE["slate_dark"])

    # Clear connecting arrows in the wide gaps with clean pill badges well above the arrows
    # Gap 1: between 4.4 and 6.0
    ax.annotate("", xy=(5.9, 3.8), xytext=(4.5, 3.8),
                arrowprops=dict(arrowstyle="->", color=PALETTE["navy"], lw=2.5, mutation_scale=18))
    ax.text(5.2, 5.0, "Wi-Fi Stream\n(50 Hz JSON)", ha="center", va="center",
            fontsize=10, fontweight="bold", color=PALETTE["navy"],
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=PALETTE["blue"], lw=1.5))

    # Gap 2: between 10.0 and 11.6
    ax.annotate("", xy=(11.5, 3.8), xytext=(10.1, 3.8),
                arrowprops=dict(arrowstyle="->", color=PALETTE["navy"], lw=2.5, mutation_scale=18))
    ax.text(10.8, 5.0, "Window Tensor\n(150x6 @ 0.6ms)", ha="center", va="center",
            fontsize=10, fontweight="bold", color=PALETTE["navy"],
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=PALETTE["purple"], lw=1.5))

    # Bottom disclaimer card
    ax.add_patch(patches.FancyBboxPatch((0.5, 0.7), 15.0, 1.0, boxstyle="round,pad=0.15",
                                         facecolor="white", edgecolor=PALETTE["crimson"], linewidth=1.5))
    ax.text(8.0, 1.20, "CRUCIAL VIVA DISTINCTION (Demonstration vs. Benchmark):",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color=PALETTE["crimson"])
    ax.text(8.0, 0.90, "The phone demo verifies hardware streaming and real-time execution mechanics. The 97.1% accuracy belongs strictly to DASIG wearable sensors.",
            ha="center", va="center", fontsize=9.8, color=PALETTE["slate_dark"])

    fig.suptitle("Hardware-in-the-Loop Real-Time Mobile Demonstration Architecture",
                 fontsize=17, fontweight="bold", color=PALETTE["navy"], y=0.96)

    save_figure(fig, "08_live_demo_architecture")


# ==============================================================================
# FIGURE 9: Executive Presentation Summary Dashboard (Master Slide)
# ==============================================================================
def make_fig9_executive_summary():
    fig = plt.figure(figsize=(16, 9.0), facecolor=PALETTE["slate_light"])
    gs = GridSpec(2, 4, height_ratios=[1.0, 1.5], wspace=0.20, hspace=0.35,
                  top=0.88, bottom=0.08, left=0.06, right=0.94)

    # 4 Top KPI Metric Cards (using transAxes and clip_on=False for crisp visible card borders)
    kpis = [
        ("97.1%", "Test Accuracy", "Held-out 12 participants", PALETTE["navy"]),
        ("93.7%", "Macro F1-Score", "Balanced class evaluation", PALETTE["blue"]),
        ("0.62 ms", "Inference Latency", "CPU (~1,610 win/sec)", PALETTE["emerald"]),
        ("60 Subj", "Cohort Size", "Zero data leakage split", PALETTE["purple"])
    ]
    for idx, (val, title, sub, col) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor("white")
        ax.axis("off")
        ax.add_patch(patches.FancyBboxPatch((0.05, 0.05), 0.90, 0.90, transform=ax.transAxes,
                                             boxstyle="round,pad=0.02", facecolor="white",
                                             edgecolor=col, linewidth=2.5, clip_on=False))
        ax.text(0.5, 0.68, val, transform=ax.transAxes, ha="center", va="center", fontsize=26, fontweight="bold", color=col)
        ax.text(0.5, 0.38, title, transform=ax.transAxes, ha="center", va="center", fontsize=11.5, fontweight="bold", color=PALETTE["slate_dark"])
        ax.text(0.5, 0.18, sub, transform=ax.transAxes, ha="center", va="center", fontsize=9.5, color=PALETTE["slate_text"])

    # Bottom Left: Methodology Card
    ax_left = fig.add_subplot(gs[1, :2])
    ax_left.set_facecolor("white")
    ax_left.axis("off")
    ax_left.add_patch(patches.FancyBboxPatch((0.03, 0.03), 0.94, 0.94, transform=ax_left.transAxes,
                                             boxstyle="round,pad=0.02", facecolor="white",
                                             edgecolor=PALETTE["navy"], linewidth=2.5, clip_on=False))
    ax_left.text(0.5, 0.90, "Technical Methodology & Scope", transform=ax_left.transAxes,
                 ha="center", va="center", fontsize=13, fontweight="bold", color=PALETTE["navy"])
    
    left_points = [
        ("Task Formulation", "Classify 3.0s window: Standard Pick-and-Place vs Abrupt Alarm Response."),
        ("Input Representation", "Active forearm IMU (3 Accel + 3 Gyro = 6 channels) resampled to 50 Hz."),
        ("Neural Architecture", "ActivityCNNLSTM: 1D CNN feature learning + Bi-LSTM temporal sequence modeling."),
        ("Imbalance Strategy", "Class-weighted cross-entropy loss (~6.5:1 ratio) + sensor jitter/gain augmentation."),
        ("Ablation Finding", "Single forearm IMU significantly outperforms 5-sensor full-body fusion.")
    ]
    for idx, (head, text) in enumerate(left_points):
        y_pos = 0.76 - idx * 0.145
        ax_left.text(0.08, y_pos, f"• {head}:", transform=ax_left.transAxes, fontsize=10.5, fontweight="bold", color=PALETTE["navy"])
        ax_left.text(0.08, y_pos - 0.055, text, transform=ax_left.transAxes, fontsize=9.5, color=PALETTE["slate_dark"])

    # Bottom Right: Viva & Defense Takeaways
    ax_right = fig.add_subplot(gs[1, 2:])
    ax_right.set_facecolor("white")
    ax_right.axis("off")
    ax_right.add_patch(patches.FancyBboxPatch((0.03, 0.03), 0.94, 0.94, transform=ax_right.transAxes,
                                              boxstyle="round,pad=0.02", facecolor="white",
                                              edgecolor=PALETTE["teal"], linewidth=2.5, clip_on=False))
    ax_right.text(0.5, 0.90, "Key Viva & Defense Takeaways", transform=ax_right.transAxes,
                  ha="center", va="center", fontsize=13, fontweight="bold", color=PALETTE["teal"])

    right_points = [
        ("Zero Data Leakage", "Subject-disjoint holdout guarantees true generalization to unseen workers."),
        ("Proxy Label Limitation", "Alarm cue timestamps serve as proxy boundaries; subject reaction delays exist."),
        ("Phone Domain Shift", "Mobile web demo validates real-time pipeline; DASIG is the empirical benchmark."),
        ("Hardware Readiness", "Ultra-low CPU latency (0.62 ms) makes model deployable on low-power edge MCUs."),
        ("Bosch HMI Relevance", "Proves feasibility of wearable anomaly/reaction monitoring in assembly lines.")
    ]
    for idx, (head, text) in enumerate(right_points):
        y_pos = 0.76 - idx * 0.145
        ax_right.text(0.08, y_pos, f"• {head}:", transform=ax_right.transAxes, fontsize=10.5, fontweight="bold", color=PALETTE["teal"])
        ax_right.text(0.08, y_pos - 0.055, text, transform=ax_right.transAxes, fontsize=9.5, color=PALETTE["slate_dark"])

    fig.suptitle("Project 8: Wearable Inertial Movement Recognition - Executive Summary",
                 fontsize=17, fontweight="bold", color=PALETTE["navy"], y=0.96)

    save_figure(fig, "09_executive_summary_dashboard")


if __name__ == "__main__":
    print("Generating PPT graphics & charts with polished 16:9 widescreen layout...")
    make_fig1_dataset_distribution()
    make_fig2_split_leakage_defense()
    make_fig3_sensor_waveforms()
    make_fig4_model_architecture()
    make_fig5_benchmark_metrics()
    make_fig6_confusion_matrix()
    make_fig7_sensor_comparison()
    make_fig8_live_demo_architecture()
    make_fig9_executive_summary()
    print("All 9 figures successfully re-generated in figures/ and ppt_figures/!")
