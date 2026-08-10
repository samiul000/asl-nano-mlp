"""Train a 42→32→16→5 MLP classifier on collected landmark data.

Outputs:
    model.pt          PyTorch state dict
    train_metrics.json  accuracy, per-class precision/recall/F1, confusion matrix
"""
import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)

LABELS = ["A", "B", "C", "D", "E"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}


def load_split(path):
    X, y = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 43:
                continue
            feats = [float(v) for v in row[:42]]
            label = row[42].strip().upper()
            X.append(feats)
            y.append(LABEL2IDX[label])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


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


def train(args):
    data_dir = Path(args.data)
    X_train, y_train = load_split(data_dir / "train.csv")
    X_val, y_val = load_split(data_dir / "val.csv")
    print(f"Train: {len(X_train)} samples  Val: {len(X_val)} samples")

    X_t = torch.tensor(X_train)
    y_t = torch.tensor(y_train)
    X_v = torch.tensor(X_val)
    y_v = torch.tensor(y_val)

    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    bs = args.batch

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X_t))
        total_loss = 0.0
        for i in range(0, len(X_t), bs):
            idx = perm[i:i+bs]
            out = model(X_t[idx])
            loss = loss_fn(out, y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        avg = total_loss / len(X_t)
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_pred = model(X_v).argmax(1).numpy()
            val_acc = accuracy_score(y_v.numpy(), val_pred)
            print(f"  epoch {epoch+1:3d}  loss={avg:.4f}  val_acc={val_acc:.3f}")

    # final eval
    model.eval()
    val_pred = model(X_v).argmax(1).numpy()
    y_true = y_v.numpy()
    acc = accuracy_score(y_true, val_pred)
    report = classification_report(y_true, val_pred, target_names=LABELS, output_dict=True)
    cm = confusion_matrix(y_true, val_pred).tolist()

    torch.save(model.state_dict(), "model.pt")
    metrics = {"accuracy": acc, "classification_report": report, "confusion_matrix": cm,
               "train_samples": len(X_train), "val_samples": len(X_val)}
    with open("train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved model.pt  accuracy={acc:.3f}")
    print("Confusion matrix:")
    print(cm)

    # confusion heatmap
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)
    cm_arr = confusion_matrix(y_true, val_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(len(LABELS)), yticks=range(len(LABELS)),
           xticklabels=LABELS, yticklabels=LABELS,
           xlabel="Predicted", ylabel="True", title="Confusion Matrix (Float)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(cm_arr[i, j]),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black")
    fig.tight_layout()
    heatmap_path = out_dir / "confusion_matrix.png"
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"Saved {heatmap_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--output", default="output", help="Output directory for plots")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    train(parser.parse_args())
