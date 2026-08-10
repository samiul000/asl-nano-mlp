"""
Quantization: per-neuron symmetric int8 (scale = max|w|/127).
Generates:
    firmware/sign_language_nano_int8/model_int8.h
Self-check:
    numpy float vs int8 inference on test set, prints accuracy and agreement.
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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


def quantize_int8(weight_np, bias_np):
    """Per-neuron symmetric int8 quantization. Returns int8 weights, scales, bias."""
    row_max = np.abs(weight_np).max(axis=1, keepdims=True)
    scales = np.where(row_max > 1e-10, row_max / 127.0, 1.0)
    w_int8 = np.clip(np.round(weight_np / scales), -128, 127).astype(np.int8)
    return w_int8, scales.flatten().astype(np.float32), bias_np.astype(np.float32)


def int8_inference(x_float, layers):
    """Run int8 inference matching firmware: int32 accumulate, int16 hidden activations."""
    act = x_float
    for k, (w_int8, scales, bias) in enumerate(layers):
        w_f = w_int8.astype(np.float32) * scales[:, None]
        act = act @ w_f.T + bias
        if k < len(layers) - 1:
            act = np.clip(act, 0, 32767)
    return act


def float_inference(x_float, layers_raw):
    """Run float32 inference for comparison."""
    act = x_float
    for k, (w, b) in enumerate(layers_raw):
        act = act @ w.T + b
        if k < len(layers_raw) - 1:
            act = np.clip(act, 0, 32767)
    return act


def _load_split(path):
    X, y = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 43:
                continue
            X.append([float(v) for v in row[:42]])
            y.append(row[42].strip().upper())
    LABEL2IDX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    return np.array(X, dtype=np.float32), np.array([LABEL2IDX[l] for l in y])


def _write_layer(f, wname, bname, w_int8, scales, bias):
    rows, cols = w_int8.shape
    f.write(f"// {wname}: {rows}x{cols}  per-neuron scales (int8)\n")
    f.write(f"const float {wname}_scale[{rows}] PROGMEM = {{\n")
    f.write("  " + ",".join(f"{v:.6f}f" for v in scales) + "\n")
    f.write("};\n")
    f.write(f"static const int8_t {wname}[{rows*cols}] PROGMEM = {{\n")
    flat = w_int8.flatten()
    for i in range(0, len(flat), 16):
        chunk = flat[i:i+16]
        f.write("  " + ",".join(str(int(v)) for v in chunk) + ",\n")
    f.write("};\n")
    f.write(f"const float {bname}[{cols}] PROGMEM = {{\n")
    f.write("  " + ",".join(f"{v:.6f}f" for v in bias) + "\n")
    f.write("};\n\n")


def export(model_path, data_dir, out_dir):
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model = MLP()
    model.load_state_dict(state)
    model.eval()

    # extract numpy weights
    layers_raw = []
    for i in [0, 2, 4]:
        w = model.net[i].weight.detach().numpy()
        b = model.net[i].bias.detach().numpy()
        layers_raw.append((w, b))

    # quantize to int8
    layers_q = [quantize_int8(w, b) for w, b in layers_raw]

    # evaluate on test set
    X_test, y_test = _load_split(Path(data_dir) / "test.csv")
    X = X_test.astype(np.float32)

    # float inference
    float_logits = float_inference(X, layers_raw)
    float_pred = np.argmax(float_logits, axis=1)

    # int8 inference
    q_logits = int8_inference(X, layers_q)
    q_pred = np.argmax(q_logits, axis=1)

    # metrics
    float_acc = accuracy_score(y_test, float_pred)
    q_acc = accuracy_score(y_test, q_pred)
    agree = np.mean(float_pred == q_pred)
    mean_err = np.mean(np.abs(float_logits - q_logits))
    max_err = np.max(np.abs(float_logits - q_logits))

    LABELS = ["A", "B", "C", "D", "E"]
    print(f"Float accuracy:  {float_acc:.4f}")
    print(f"INT8 accuracy:   {q_acc:.4f}")
    print(f"Argmax agreement: {agree:.4f}")
    print(f"Quant error:     mean={mean_err:.4f}  max={max_err:.4f}")
    print()
    print("INT8 classification report:")
    print(classification_report(y_test, q_pred, target_names=LABELS))

    cm = confusion_matrix(y_test, q_pred)
    print("INT8 confusion matrix:")
    print(cm)

    # write model_int8.h
    os.makedirs(out_dir, exist_ok=True)
    h_path = os.path.join(out_dir, "model_int8.h")
    with open(h_path, "w") as f:
        f.write("#ifndef MODEL_INT8_H\n#define MODEL_INT8_H\n\n")
        f.write('#include <Arduino.h>\n\n')
        w0, s0, b0 = layers_q[0]
        _write_layer(f, "W0", "bias0", w0, s0, b0)
        w1, s1, b1 = layers_q[1]
        _write_layer(f, "W1", "bias1", w1, s1, b1)
        w2, s2, b2 = layers_q[2]
        _write_layer(f, "W2", "bias2", w2, s2, b2)
        f.write("#endif\n")
    print(f"\nWrote {h_path}")

    # flash size comparison
    w0_bytes = layers_q[0][0].size * 1
    w1_bytes = layers_q[1][0].size * 1
    w2_bytes = layers_q[2][0].size * 1
    total_weight_bytes = w0_bytes + w1_bytes + w2_bytes
    print(f"\nINT8 weight storage: {total_weight_bytes} bytes")
    print(f"INT16 weight storage: {total_weight_bytes * 2} bytes (for comparison)")

    return {"float_acc": float_acc, "int8_acc": q_acc, "agreement": agree,
            "mean_err": mean_err, "confusion_matrix": cm.tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--out", default="firmware/sign_language_nano_int8")
    args = parser.parse_args()
    export(args.model, args.data, args.out)
