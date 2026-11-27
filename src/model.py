"""Compact 1D CNN-LSTM for wearable industrial movement windows."""
from pathlib import Path
import torch
from torch import nn


class ActivityCNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=48, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(96, num_classes))

    def forward(self, x):
        z = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        _, (hidden, _) = self.lstm(z)
        summary = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(summary)


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = ActivityCNNLSTM(num_classes=len(state["class_names"]))
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, state

