# Training and Export

## Training

```bash
python train.py --epochs 100 --batch 32 --data dataset
```

Reads `dataset/train.csv` and `dataset/val.csv`. Saves `model.pt` (PyTorch state dict) and `train_metrics.json`.

## Export to Firmware

### Float32

```bash
python export_float.py --model model.pt --data dataset --out firmware/sign_language_nano_float
```

Generates `model_float.h` with float32 weights in PROGMEM.

### INT16 (Recommended)

```bash
python export.py --model model.pt --data dataset --out firmware/sign_language_nano
```

Generates `model.h` with:

- `W0`, `W1`, `W2` -> int16 weight arrays in PROGMEM
- `W0_scale`, `W1_scale`, `W2_scale` -> float scale factors
- `bias0`, `bias1`, `bias2` -> float bias arrays in PROGMEM

### INT8 (Most Compact)

```bash
python export_int8.py --model model.pt --data dataset --out firmware/sign_language_nano_int8
```

Generates `model_int8.h` with int8 weights in PROGMEM.

## Compile Firmware

```bash
# Float32
arduino-cli compile --fqbn arduino:avr:nano:cpu=atmega328 firmware/sign_language_nano_float

# INT16
arduino-cli compile --fqbn arduino:avr:nano:cpu=atmega328 firmware/sign_language_nano

# INT8
arduino-cli compile --fqbn arduino:avr:nano:cpu=atmega328 firmware/sign_language_nano_int8
```

## Upload to Nano

```bash
arduino-cli upload --fqbn arduino:avr:nano:cpu=atmega328 --port COM3 firmware/sign_language_nano_float
```

Or use Arduino IDE: select Board → Arduino Nano, Port → COM3, Upload.

## Evaluation

```bash
python evaluate.py --model model.pt --data dataset
```

Outputs `eval_report.json` with accuracy, precision, recall, F1, and confusion matrix.

## Comparison

```bash
python compare_models.py
```

Generates confusion matrices for all three variants and `model_comparison_all.csv`.

## Realtime Stream

```bash
python stream.py --variant float --port COM3    # float32 model
python stream.py --variant int16 --port COM3    # int16 model
python stream.py --variant int8 --port COM3     # int8 model
python stream.py --echo                         # test without hardware
```

Logs: `--log` flag, `--log-file output.csv`, `--conf 0.7`.
