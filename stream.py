import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from features import normalize_and_encode

MODEL_PATH = str(Path(__file__).parent / "models" / "hand_landmarker.task")


def find_arduino_port():

    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        for p in ports:
            name = p.description.lower()
            if any(k in name for k in ["arduino", "usbserial", "ch340", "cp210", "ftdi"]):
                return p.device
        if ports:
            return ports[0].device
    except Exception:
        pass
    return None


def create_detector(conf):
    from mediapipe.tasks.python import vision, BaseOptions
    base_options = BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=conf,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect(detector, frame_bgr):
    from mediapipe import Image, ImageFormat
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None, None
    return result.hand_landmarks[0], result.handedness[0]


def parse_response(line):
    """Parse {sign,infer_us,confidence,fps,free_ram} from Arduino."""
    line = line.strip()
    if not line.startswith("{") or not line.endswith("}"):
        return None
    parts = line[1:-1].split(",")
    if len(parts) != 5:
        return None
    try:
        return {
            "sign": parts[0],
            "infer_us": int(parts[1]),
            "confidence": float(parts[2]),
            "fps": int(parts[3]),
            "free_ram": int(parts[4]),
        }
    except (ValueError, IndexError):
        return None


def main(args):
    # auto-set log file from variant if not explicitly provided
    if args.log and args.log_file is None:
        variant_map = {
            "float": "output/stream_log_float.csv",
            "int16": "output/stream_log.csv",
            "int8": "output/stream_log_int8.csv",
        }
        args.log_file = variant_map[args.variant]

    if args.echo:
        ser = None
    else:
        import serial
        port = args.port or find_arduino_port()
        if not port:
            print("No serial port found. Use --port or --echo.")
            sys.exit(1)
        ser = serial.Serial(port, args.baud, timeout=0.1)
        print(f"Connected to {port}")

    # CSV logger
    log_file = None
    log_writer = None
    if args.log and ser:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            "timestamp", "sign", "confidence", "infer_us",
            "fps", "free_ram", "latency_pc_ms"
        ])
        print(f"Logging to {log_path} (variant: {args.variant})")

    detector = create_detector(args.conf)
    # try DirectShow first, then default backend
    cap = None
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

    fps_counter = 0
    fps_time = time.time()
    fps_display = 0
    last_reopen = 0.0
    frame_count = 0
    last_stats = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                now = time.time()
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
            lm_list, handed_list = detect(detector, frame)

            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_time = time.time()

            has_hand = lm_list is not None and len(lm_list) == 21 and handed_list
            sent_this_frame = False

            if has_hand:
                hand = handed_list[0]
                # draw landmarks
                lm = np.array([[l.x * 640, l.y * 480] for l in lm_list], dtype=int)
                for i, (px, py) in enumerate(lm):
                    cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)

                if hand.score >= args.conf:
                    is_left = hand.category_name.lower().startswith("left")
                    feats = normalize_and_encode(lm_list, mirror_left=is_left)
                    cs = sum(int(v) for v in feats) % 100
                    packet = "<" + ",".join(str(int(v)) for v in feats) + "," + str(cs) + ">"
                    sent_this_frame = True
                    if ser:
                        ser.reset_input_buffer()
                        t_send = time.time()
                        ser.write((packet + "\n").encode())
                        resp_line = ser.readline().decode("utf-8", errors="ignore")
                        t_recv = time.time()
                        resp = parse_response(resp_line)
                        if resp:
                            latency_ms = (t_recv - t_send) * 1000
                            last_stats = resp
                            if log_writer:
                                log_writer.writerow([
                                    f"{t_recv:.3f}", resp["sign"],
                                    resp["confidence"], resp["infer_us"],
                                    resp["fps"], resp["free_ram"],
                                    f"{latency_ms:.1f}"
                                ])
                                log_file.flush()
                    else:
                        print(packet)

            # overlay
            h, w = frame.shape[:2]

            # draw hand skeleton if detected
            if has_hand:
                lm = np.array([[l.x * w, l.y * h] for l in lm_list], dtype=int)
                connections = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12),
                    (0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20),
                    (5,9),(9,13),(13,17),
                ]
                for i, j in connections:
                    cv2.line(frame, tuple(lm[i]), tuple(lm[j]), (0, 200, 255), 2, cv2.LINE_AA)
                for lmx, lmy in lm:
                    cv2.circle(frame, (lmx, lmy), 4, (0, 255, 255), -1, cv2.LINE_AA)

            def draw_panel(frm, x, y, pw, ph, alpha=0.6):
                ov = frm.copy()
                cv2.rectangle(ov, (x, y), (x + pw, y + ph), (30, 30, 30), -1)
                cv2.addWeighted(ov, alpha, frm, 1 - alpha, 0, frm)
                cv2.rectangle(frm, (x, y), (x + pw, y + ph), (80, 80, 80), 1, cv2.LINE_AA)

            # prediction panel (top-right)
            panel_w, panel_h = 220, 130
            ppx, ppy = w - panel_w - 10, 10
            draw_panel(frame, ppx, ppy, panel_w, panel_h)

            if last_stats and last_stats["sign"] != "?":
                letter = last_stats["sign"]
                conf = last_stats["confidence"]
                (tw, _), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)
                lx = ppx + (panel_w - tw) // 2
                cv2.putText(frame, letter, (lx, ppy + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 120), 4, cv2.LINE_AA)
                bar_x, bar_y, bar_w, bar_h = ppx + 15, ppy + 75, panel_w - 30, 16
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                fill_w = int(bar_w * min(conf, 100) / 100)
                bar_color = (0, 220, 80) if conf >= 70 else (0, 180, 255) if conf >= 40 else (0, 0, 255)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
                cv2.putText(frame, f"{conf:.0f}%", (bar_x + fill_w + 6, bar_y + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(frame, "PREDICTION", (ppx + 15, ppy + 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "WAITING", (ppx + 30, ppy + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2, cv2.LINE_AA)
                cv2.putText(frame, "Show hand", (ppx + 45, ppy + 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

            # stats panel (bottom-left)
            stats_h = 110
            draw_panel(frame, 10, h - stats_h - 10, 260, stats_h)

            rtt_ms = f"{(t_recv - t_send) * 1000:.0f} ms" if sent_this_frame else "--"
            items = [
                ("FPS", f"{fps_display}", (0, 200, 255)),
                ("Latency", f"{last_stats.get('infer_us', 0)/1000:.1f} ms", (200, 200, 200)),
                ("RAM", f"{last_stats.get('free_ram', 0)} B", (200, 200, 200)),
                ("PC RTT", rtt_ms, (200, 200, 200)),
            ]
            for i, (label, value, val_color) in enumerate(items):
                iy = h - stats_h + 8 + i * 24
                cv2.putText(frame, label, (20, iy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)
                cv2.putText(frame, value, (110, iy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, val_color, 1, cv2.LINE_AA)

            # status dot (top-left)
            dot_color = (0, 200, 0) if sent_this_frame else (0, 0, 220) if not has_hand else (0, 160, 255)
            cv2.circle(frame, (20, 20), 7, dot_color, -1, cv2.LINE_AA)
            status_text = "Connected" if sent_this_frame else "No hand" if not has_hand else "Low conf"
            cv2.putText(frame, status_text, (34, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, dot_color, 1, cv2.LINE_AA)

            cv2.imshow("Sign Language MLP", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        try: cap.release()
        except: pass
        cv2.destroyAllWindows()
        if ser:
            ser.close()
        if log_file:
            log_file.close()
            print(f"Log saved to {args.log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--conf", type=float, default=0.7)
    parser.add_argument("--echo", action="store_true")
    parser.add_argument("--log", action="store_true", help="Log Arduino responses to CSV")
    parser.add_argument("--log-file", default=None, help="Log file path (auto-set from --variant if omitted)")
    parser.add_argument("--variant", choices=["float", "int16", "int8"], default="int16",
                        help="Firmware variant for log file naming (default: int16)")
    main(parser.parse_args())
