# Data Collector

## Usage

```bash
python collect_data.py                    # default settings
python collect_data.py --target 2000      # 2000 samples/class target
python collect_data.py --rate 10          # 10 samples/sec
python collect_data.py --sim              # headless test (no camera)
```

## Controls

| Key   | Action                              |
| ----- | ----------------------------------- |
| A–E   | Select current label                |
| SPACE | Start / stop recording              |
| O     | Toggle sample count overlay         |
| Q     | Quit (writes train/val/test splits) |

## Workflow

1. Run `python collect_data.py`
2. Press `A` to select label A
3. Hold the A hand sign in front of the camera
4. Press `SPACE` to start recording
5. Samples save at 7/sec (configurable via `--rate`)
6. Press `SPACE` again to stop
7. Switch to next label, repeat
8. Press `Q` to quit : splits are written automatically

## Features

- **Fixed-rate sampling** : saves at N samples/sec, not every frame, to avoid near-duplicates
- **Confidence gate** : skips low-confidence detections (default 0.7)
- **Left-hand mirroring** : auto-mirrors x for left hands so both map to right-hand space
- **Progress bar** : live count toward target per class
- **Session delta** : `+N` counter shows new samples saved this session
- **Auto-split** : on quit, shuffles and writes `train.csv` (70%), `val.csv` (15%), `test.csv` (15%)
- **Accumulate across sessions** : raw CSVs append, splits regenerate from all data

## Output Files

```
dataset/
  raw/
    A.csv    # timestamp, x1,y1,...,x42,y42
    B.csv
    C.csv
    D.csv
    E.csv
  train.csv  # x1,y1,...,x42,y42,label
  val.csv
  test.csv
```

## Flags

| Flag       | Default | Description                     |
| ---------- | ------- | ------------------------------- |
| `--target` | 2000    | Samples per class target        |
| `--rate`   | 7       | Samples per second              |
| `--conf`   | 0.7     | Min handedness confidence       |
| `--data`   | dataset | Dataset directory               |
| `--sim`    | off     | Headless simulation (no camera) |
