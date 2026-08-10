# Hardware

## Components

| Component | Quantity |
|-----------|----------|
| Arduino Nano (ATmega328P) | 1 |
| SSD1306 OLED (128×64, I2C) | 1 |
| USB cable (Mini-B) | 1 |
| Breadboard | 1 |
| Jumper wires | 4 |

## Wiring

```
Arduino Nano          SSD1306 OLED
  A4 (SDA) ─────────── SDA
  A5 (SCL) ─────────── SCL
  5V       ─────────── VCC
  GND      ─────────── GND
```

## OLED Display Layout

```
┌──────────────────────────────┐
│ Sign : B                     │
│ Conf : 97%                   │
│ Lat  : 11.2ms                │
│ RAM  : 803B                  │
│ FPS  : 22                    │
└──────────────────────────────┘
```

Displayed fields:
- **Sign** — predicted alphabet (A–E)
- **Conf** — softmax confidence percentage
- **Lat** — inference latency in milliseconds
- **RAM** — free SRAM in bytes
- **FPS** — frames (predictions) per second

## Arduino Nano Specs

| Spec | Value |
|------|-------|
| MCU | ATmega328P |
| Flash | 32 KB |
| SRAM | 2 KB |
| Clock | 16 MHz |
| ADC | 10-bit |
| Digital I/O | 22 |

## Flash/RAM Usage (Real Measurements)

### Float32 Variant

| Resource | Used | Total | % |
|----------|------|-------|---|
| Flash | 27,326 B | 30,720 B | 88% |
| SRAM | 1,222 B | 2,048 B | 59% |
| Free SRAM | 826 B | — | — |

### INT16 Variant

| Resource | Used | Total | % |
|----------|------|-------|---|
| Flash | 23,778 B | 30,720 B | 77% |
| SRAM | 1,098 B | 2,048 B | 53% |
| Free SRAM | 950 B | — | — |

### INT8 Variant

| Resource | Used | Total | % |
|----------|------|-------|---|
| Flash | 21,884 B | 30,720 B | 71% |
| SRAM | 1,118 B | 2,048 B | 54% |
| Free SRAM | 930 B | — | — |

*Total flash after bootloader: 30,720 B (32,768 - 1,048 bootloader)*

## Inference Latency

| Variant | Latency | Speedup vs Float32 |
|---------|---------|---------------------|
| Float32 | 49.25 ms | 1× |
| INT16 | 10.70 ms | 4.6× |
| INT8 | 10.26 ms | 4.8× |
