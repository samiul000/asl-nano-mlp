# Architecture

## System Overview

```
Laptop Webcam → OpenCV → MediaPipe Hands → 42 features → Serial → Arduino Nano → OLED
```

The laptop handles image capture and hand landmark extraction. The Arduino
Nano runs all ML inference on-device using a layer-streaming MLP.

## MLP Design

```
Layer       Input    Output  Activation  Weights
─────────────────────────────────────────────────
Input       42       —       —           —
Hidden 1    42       32      ReLU        quantized
Hidden 2    32       16      ReLU        quantized
Output      16       5       Softmax     quantized
```

Total parameters: (42×32) + (32×16) + (16×5) = 1,344 + 512 + 80 = 1,936 weights + 53 biases = **1,989 parameters**.

## Feature Vector

42 int16 values (0–1000) from 21 hand landmarks:

- Wrist-centered normalization
- Scale normalization (hand size invariant)
- Left hand: x-coordinates mirrored to right-hand space
- MediaPipe landmark indices: wrist=0, thumb=1–4, index=5–8, middle=9–12, ring=13–16, pinky=17–20

## Quantization Variants

### Float32 (Baseline)

- Weights: float32 (32-bit IEEE 754)
- Inference: float32 arithmetic
- Model size: 7,956 B (1,936 weights × 4 B + 53 biases × 4 B)
- Accuracy: 89.58%, 100% argmax agreement with INT16

### INT16 (Recommended)

- Weights: int16 with per-neuron float scale
- Inference: int32 accumulation → float scale + bias
- Model size: 4,296 B (1,936 weights × 2 B + 53 biases × 4 B + scales × 4 B)
- Accuracy: 89.58%, 100% argmax agreement with Float32

### INT8 (Most Compact)

- Weights: int8 with per-neuron float scale
- Inference: int32 accumulation → float scale + bias
- Model size: 2,360 B (1,936 weights × 1 B + 53 biases × 4 B + scales × 4 B)
- Accuracy: 88.45%, 98.01% argmax agreement with Float32

## Memory Budget (INT16 Variant)

| Resource           | Usage        | Total       |
| ------------------ | ------------ | ----------- |
| Weights (PROGMEM)  | 3,872 B      | 32 KB flash |
| Biases (PROGMEM)   | 212 B        | —           |
| Scales (PROGMEM)   | 212 B        | —           |
| Activations (SRAM) | ~150 B       | 2 KB SRAM   |
| Serial buffer      | 240 B        | —           |
| OLED buffers       | ~200 B       | —           |
| **Total SRAM**     | **~1,098 B** | 2,048 B     |
| **Free SRAM**      | **950 B**    | —           |

## Layer-Streaming

Weights are read from PROGMEM on demand, only one layer's parameters
and activations reside in SRAM at any time:

```
Read W0 from flash → compute hidden1 → free W0
Read W1 from flash → compute hidden2 → free W1
Read W2 from flash → compute output  → free W2
```

## PROGMEM Read Safety

- `memcpy_P` used for all PROGMEM reads (safe on AVR)
- `pgm_read_float` avoided unsafe for aligned float reads on ATmega328P
- B0/B1 macros avoided (conflict with Arduino binary.h pin macros)
