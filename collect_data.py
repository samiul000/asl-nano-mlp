"""hand-landmark data collector.

Controls:
    A-E     Select current label
    SPACE   Start / Stop recording
    O       Toggle sample-count overlay
    Q       Quit (writes train/val/test splits)

Flags:
    --target  Samples per class target  (default 2000)
    --rate    Save rate per second       (default 7)
    --conf    Min handedness confidence  (default 0.7)
    --data    Dataset directory           (default dataset/)
    --sim     Headless simulation (no camera, synthetic landmarks)
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from features import normalize_and_encode

LABELS = ["A", "B", "C", "D", "E"]
TARGET_PER_CLASS = 2000
SAVE_INTERVAL = 1 / 7.0
CONF_THRESHOLD = 0.7
MODEL_PATH = str(Path(__file__).parent / "models" / "hand_landmarker.task")


def _synthetic_landmarks():
    """Fake a hand for --sim testing."""
    from types import SimpleNamespace
    base = np.random.rand(21, 3).astype(np.float32)
    return [SimpleNamespace(x=float(base[i, 0]),
                            y=float(base[i, 1]),
                            z=float(base[i, 2])) for i in range(21)]


def _create_detector(conf_threshold):
    from mediapipe.tasks.python import vision, BaseOptions
    base_options = BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=conf_threshold,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def _detect(detector, frame_bgr):
    """Run detection, return (landmarks_list, handedness_list) or (None, None)."""
    from mediapipe import Image, ImageFormat
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None, None
    return result.hand_landmarks[0], result.handedness[0]


def _write_splits(data_dir):
    raw_dir = Path(data_dir) / "raw"
    out_dir = Path(data_dir)
    all_rows = []
    for label in LABELS:
        p = raw_dir / f"{label}.csv"
        if not p.exists():
            continue
        with open(p, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                all_rows.append((label, row))
    rng = np.random.default_rng(42)
    rng.shuffle(all_rows)
    n = len(all_rows)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    splits = {
        "train.csv": all_rows[:n_train],
        "val.csv": all_rows[n_train:n_train + n_val],
        "test.csv": all_rows[n_train + n_val:],
    }
    for name, rows in splits.items():
        with open(out_dir / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"x{i}" for i in range(1, 43)] + ["label"])
            for label, row in rows:
                w.writerow(row[1:] + [label])  # skip timestamp, keep 42 features
    print(f"\nSplits written: {n} total -> "
          f"train {len(splits['train.csv'])}, "
          f"val {len(splits['val.csv'])}, "
          f"test {len(splits['test.csv'])}")


def run(args):
    data_dir = Path(args.data)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    counts = {l: sum(1 for _ in open(raw_dir / f"{l}.csv")) - 1
              if (raw_dir / f"{l}.csv").exists() else 0
              for l in LABELS}

    sim_mode = args.sim
    cap = None
    detector = None
    if not sim_mode:
        detector = _create_detector(args.conf)
        # try DirectShow first, then default backend
        for backend in [cv2.CAP_DSHOW, 0]:
            for idx in range(4):
                test_cap = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)
                if test_cap.isOpened():
                    time.sleep(0.5)
                    ret, _ = test_cap.read()
                    if ret:
                        cap = test_cap
                        backend_name = "DirectShow" if backend == cv2.CAP_DSHOW else "default"
                        print(f"[camera] Opened camera index {idx} ({backend_name})")
                        break
                    test_cap.release()
            if cap is not None:
                break
        if cap is None:
            print("[camera] No webcam found. Close other apps or check drivers.")
            sys.exit(1)

    current_label = LABELS[0]
    recording = False
    show_counts = False
    last_save = 0.0
    last_warn = 0.0
    last_reopen = 0.0
    frame_count = 0
    session_saved = {l: 0 for l in LABELS}
    init_counts = dict(counts)
    save_handles = {l: open(raw_dir / f"{l}.csv", "a", newline="") for l in LABELS}
    writers = {l: csv.writer(save_handles[l]) for l in LABELS}

    # write headers if files are empty
    for l in LABELS:
        p = raw_dir / f"{l}.csv"
        if p.stat().st_size == 0 if p.exists() else True:
            writers[l].writerow(["timestamp"] + [f"x{i}" for i in range(1, 43)])

    print("Collector ready.  A-E=label  SPACE=record  O=counts  Q=quit")

    try:
        while True:
            now = time.time()

            if sim_mode:
                landmarks = _synthetic_landmarks()
                is_left = False
                ok = True
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                score = 0.99
            else:
                ret, frame = cap.read()
                if not ret:
                    if frame_count > 0 and now - last_reopen > 3.0:
                        print("[camera] No frame — reopening camera...")
                        last_reopen = now
                        try: cap.release()
                        except: pass
                        time.sleep(0.3)
                        for backend in [cv2.CAP_DSHOW, 0]:
                            cap = cv2.VideoCapture(0, backend) if backend else cv2.VideoCapture(0)
                            if cap.isOpened():
                                ret2, _ = cap.read()
                                if ret2:
                                    print("[camera] Reopened OK")
                                    break
                                try: cap.release()
                                except: pass
                                cap = None
                    continue
                frame = cv2.flip(frame, 1)
                frame_count += 1
                lm_list, handed_list = _detect(detector, frame)
                ok = lm_list is not None and len(lm_list) == 21
                landmarks = lm_list if ok else []
                if ok and handed_list:
                    score = handed_list[0].score
                    is_left = handed_list[0].category_name.lower().startswith("left")
                else:
                    score = 0.0
                    is_left = False
                ok = ok and score >= CONF_THRESHOLD

            mirror = is_left and not sim_mode

            # encoding
            feats = None
            if ok and len(landmarks) == 21:
                feats = normalize_and_encode(landmarks, mirror_left=mirror)

            # save at fixed rate
            if recording and ok and feats is not None and (now - last_save) >= SAVE_INTERVAL:
                row = [f"{now:.3f}"] + feats.tolist()
                writers[current_label].writerow(row)
                save_handles[current_label].flush()
                counts[current_label] += 1
                session_saved[current_label] += 1
                last_save = now

            # overlay
            if not sim_mode:
                cv2.putText(frame, f"Label: {current_label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                state = "REC" if recording else "PAUSED"
                color = (0, 0, 255) if recording else (180, 180, 180)
                cv2.putText(frame, state, (500, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, "CAM:OK", (500, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
                cv2.putText(frame, f"F:{frame_count}", (500, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                if ok and feats is not None:
                    lm = np.array([[l.x * 640, l.y * 480] for l in landmarks], dtype=int)
                    for i, (px, py) in enumerate(lm):
                        cv2.circle(frame, (px, py), 3, (255, 255, 0), -1)
                        cv2.putText(frame, str(i), (px + 4, py - 4),
                                    cv2.FONT_HERSHEY_PLAIN, 0.6, (200, 200, 200), 1)
                # progress bar always visible
                prog = min(counts[current_label] / TARGET_PER_CLASS, 1.0)
                bar_y = 450
                cv2.rectangle(frame, (10, bar_y), (630, bar_y + 20), (60, 60, 60), -1)
                cv2.rectangle(frame, (10, bar_y), (10 + int(620 * prog), bar_y + 20),
                              (0, 200, 0), -1)
                cv2.putText(frame,
                            f"{counts[current_label]}/{TARGET_PER_CLASS}",
                            (280, bar_y + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                delta = session_saved[current_label]
                if delta > 0:
                    cv2.putText(frame, f"+{delta}", (500, bar_y + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
                if show_counts:
                    for i, l in enumerate(LABELS):
                        cv2.putText(frame, f"{l}: {counts[l]}", (10, 60 + i * 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (0, 255, 0) if l == current_label else (180, 180, 180), 1)
                cv2.imshow("Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                recording = not recording
                if recording:
                    print(f"Recording ON  (label {current_label})")
                else:
                    print(f"Recording OFF — saved {session_saved[current_label]} new samples for {current_label}")
            elif key == ord("o"):
                show_counts = not show_counts
            elif chr(key).upper() in LABELS:
                current_label = chr(key).upper()

    finally:
        for h in save_handles.values():
            h.close()
        if cap is not None:
            try: cap.release()
            except: pass
        cv2.destroyAllWindows()
        _write_splits(data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand landmark data collector")
    parser.add_argument("--target", type=int, default=TARGET_PER_CLASS)
    parser.add_argument("--rate", type=float, default=7.0, help="Samples per second")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--sim", action="store_true",
                        help="Headless simulation with synthetic landmarks")
    run(parser.parse_args())
