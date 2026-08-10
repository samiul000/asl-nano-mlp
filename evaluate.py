"""Evaluate a trained model on the test split.

Outputs:
    eval_report.json -> accuracy, precision, recall, F1, confusion matrix
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)

LABELS = ["A", "B", "C", "D", "E"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(42, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 5),
        )

    def forward(self, x):
        return self.net(x)


def load_split(path):
    X, y = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 43:
                continue
            X.append([float(v) for v in row[:42]])
            y.append(LABEL2IDX[row[42].strip().upper()])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def evaluate(args):
    X, y = load_split(Path(args.data) / "test.csv")
    print(f"Test samples: {len(X)}")

    model = MLP()
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        pred = model(torch.tensor(X)).argmax(1).numpy()

    acc = accuracy_score(y, pred)
    report = classification_report(y, pred, target_names=LABELS, output_dict=True)
    cm = confusion_matrix(y, pred).tolist()

    metrics = {"accuracy": acc, "classification_report": report, "confusion_matrix": cm}
    with open("eval_report.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Accuracy: {acc:.3f}")
    print("Confusion matrix:")
    print(np.array(cm))
    print("Saved eval_report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--data", default="dataset")
    evaluate(parser.parse_args())
