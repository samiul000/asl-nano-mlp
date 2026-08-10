"""
raw float32 weights and biases in PROGMEM.
"""
import csv
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


def export(model_path, out_dir):
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model = MLP()
    model.load_state_dict(state)
    model.eval()

    # extract numpy weights
    layers = []
    for i in [0, 2, 4]:
        w = model.net[i].weight.detach().numpy().astype(np.float32)
        b = model.net[i].bias.detach().numpy().astype(np.float32)
        layers.append((w, b))

    os.makedirs(out_dir, exist_ok=True)
    h_path = os.path.join(out_dir, "model_float.h")

    with open(h_path, "w") as f:
        f.write("#ifndef MODEL_FLOAT_H\n#define MODEL_FLOAT_H\n\n")
        f.write('#include <Arduino.h>\n\n')

        # layer 0: 32×42
        _write_layer(f, "W0", "bias0", *layers[0])
        # layer 1: 16×32
        _write_layer(f, "W1", "bias1", *layers[1])
        # layer 2: 5×16
        _write_layer(f, "W2", "bias2", *layers[2])

        f.write("#endif\n")

    print(f"Wrote {h_path}")
    # report sizes
    total = 0
    for w, b in layers:
        total += w.nbytes + b.nbytes
    print(f"Model size: {total} bytes ({total/1024:.1f} KB)")


def _write_layer(f, wname, bname, weight, bias):
    rows, cols = weight.shape
    f.write(f"// {wname}: {rows}x{cols} float32\n")
    f.write(f"static const float {wname}[{rows*cols}] PROGMEM = {{\n")
    flat = weight.flatten()
    for i in range(0, len(flat), 8):
        chunk = flat[i:i+8]
        f.write("  " + ",".join(f"{v:.6f}f" for v in chunk) + ",\n")
    f.write("};\n")
    f.write(f"static const float {bname}[{cols}] PROGMEM = {{\n")
    f.write("  " + ",".join(f"{v:.6f}f" for v in bias) + "\n")
    f.write("};\n\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--out", default="firmware/sign_language_nano_float")
    args = parser.parse_args()
    export(args.model, args.out)
