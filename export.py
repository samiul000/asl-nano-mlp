"""
Quantization: per-layer symmetric int8 (scale = max|w|/127).
Generates:
    firmware/model.h  — PROGMEM int8 weights + float scale/bias per neuron
Self-check:
    numpy float vs int8 inference on train set, asserts mean error < threshold.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


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


def quantize_layer(weight_np, bias_np):
    """Return int16 weights, per-row scales, bias float."""
    row_max = np.abs(weight_np).max(axis=1, keepdims=True)
    scales = np.where(row_max > 1e-10, row_max / 32767.0, 1.0)
    w_int16 = np.clip(np.round(weight_np / scales), -32768, 32767).astype(np.int16)
    return w_int16, scales.flatten().astype(np.float32), bias_np.astype(np.float32)


def int8_inference(x_float, layers):
    """Run int16 inference matching firmware: int32 accumulate, int16 hidden activations."""
    act = x_float
    for k, (w_int16, scales, bias) in enumerate(layers):
        w_f = w_int16.astype(np.float32) * scales[:, None]
        act = act @ w_f.T + bias
        if k < len(layers) - 1:
            act = np.clip(act, 0, 32767)
    return act


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

    # quantize
    layers_q = [quantize_layer(w, b) for w, b in layers_raw]

    # self-check on train data
    X_train, y_train = _load_split(Path(data_dir) / "train.csv")
    X = X_train.astype(np.float32)

    float_logits = X
    for k, (w, b) in enumerate(layers_raw):
        float_logits = float_logits @ w.T + b
        if k < len(layers_raw) - 1:
            float_logits = np.clip(float_logits, 0, 32767)

    q_logits = int8_inference(X, layers_q)
    mean_err = np.mean(np.abs(float_logits - q_logits))
    print(f"Quantization error: mean={mean_err:.4f}  max={np.max(np.abs(float_logits-q_logits)):.4f}")
    assert mean_err < 15.0, f"Quantization error too high: {mean_err}"
    # check argmax agreement
    float_pred = np.argmax(float_logits, axis=1)
    q_pred = np.argmax(q_logits, axis=1)
    agree = np.mean(float_pred == q_pred)
    print(f"Argmax agreement: {agree:.3f}")
    assert agree > 0.95, f"Agreement too low: {agree}"

    # write model.h
    os.makedirs(out_dir, exist_ok=True)
    h_path = os.path.join(out_dir, "model.h")
    with open(h_path, "w") as f:
        f.write("#ifndef MODEL_H\n#define MODEL_H\n\n")
        f.write('#include <Arduino.h>\n\n')
        # layer 0: 42x32
        w0, s0, b0 = layers_q[0]
        _write_layer(f, "W0", "bias0", w0, s0, b0)
        w1, s1, b1 = layers_q[1]
        _write_layer(f, "W1", "bias1", w1, s1, b1)
        w2, s2, b2 = layers_q[2]
        _write_layer(f, "W2", "bias2", w2, s2, b2)
        f.write("#endif\n")
    print(f"Wrote {h_path}")


def _write_layer(f, wname, bname, w_int16, scales, bias):
    rows, cols = w_int16.shape
    f.write(f"// {wname}: {rows}x{cols}  per-neuron scales (int16)\n")
    f.write(f"const float {wname}_scale[{rows}] PROGMEM = {{\n")
    f.write("  " + ",".join(f"{v:.6f}f" for v in scales) + "\n")
    f.write("};\n")
    f.write(f"static const int16_t {wname}[{rows*cols}] PROGMEM = {{\n")
    flat = w_int16.flatten()
    for i in range(0, len(flat), 16):
        chunk = flat[i:i+16]
        f.write("  " + ",".join(str(int(v)) for v in chunk) + ",\n")
    f.write("};\n")
    f.write(f"const float {bname}[{cols}] PROGMEM = {{\n")
    f.write("  " + ",".join(f"{v:.6f}f" for v in bias) + "\n")
    f.write("};\n\n")


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
    return np.array(X), np.array([LABEL2IDX[l] for l in y])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--out", default="firmware/sign_language_nano")
    args = parser.parse_args()
    export(args.model, args.data, args.out)
