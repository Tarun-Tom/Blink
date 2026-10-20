import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import time
import json
import argparse
import threading
from collections import deque
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eye_extractor import LandmarkEyeExtractor

# State definitions
STATE_OPEN = "OPEN"
STATE_CLOSED = "CLOSED"
STATE_UNKNOWN = "UNKNOWN"

class TemporalDrowsinessTracker:
    """
    Manages eye-state classification smoothing, sustained-closure tracking,
    and flicker-resistant alert state transitions using wall-clock elapsed time.
    """
    def __init__(
        self,
        alert_threshold: float = 1.5,
        open_grace_period: float = 0.25,
        unknown_timeout: float = 0.5,
        window_size: int = 5,
        decision_threshold: float = 0.5
    ):
        self.alert_threshold = alert_threshold
        self.open_grace_period = open_grace_period
        self.unknown_timeout = unknown_timeout
        self.window_size = window_size
        self.decision_threshold = decision_threshold

        self.p_awake_buffer = deque(maxlen=window_size)
        self.closure_duration = 0.0
        self.open_start_time = None
        self.unknown_start_time = None
        self.last_update_time = None

    def update(self, raw_p_awake: float | None, current_time: float) -> dict:
        """
        Updates the tracker with the latest frame prediction.
        
        Args:
            raw_p_awake: Model sigmoid output in [0, 1] if eye detected; None if UNKNOWN.
            current_time: Current timestamp (seconds).
        """
        if self.last_update_time is None:
            dt = 0.0
        else:
            dt = max(0.0, current_time - self.last_update_time)
            if dt > 1.0:
                dt = 0.0
        self.last_update_time = current_time

        # 1. Rolling window smoothing
        if raw_p_awake is not None:
            self.p_awake_buffer.append(raw_p_awake)
            smoothed_p_awake = float(np.mean(self.p_awake_buffer))
            frame_state = STATE_OPEN if smoothed_p_awake >= self.decision_threshold else STATE_CLOSED
        else:
            smoothed_p_awake = None
            frame_state = STATE_UNKNOWN

        # 2. Temporal State Machine & Accumulator
        if frame_state == STATE_CLOSED:
            self.open_start_time = None
            self.unknown_start_time = None
            self.closure_duration += dt
            display_state = STATE_CLOSED

        elif frame_state == STATE_OPEN:
            if self.closure_duration > 0.0:
                if self.open_start_time is None:
                    self.open_start_time = current_time

                open_elapsed = current_time - self.open_start_time
                if open_elapsed >= self.open_grace_period:
                    self.closure_duration = 0.0
                    self.open_start_time = None
            else:
                self.open_start_time = None
            display_state = STATE_OPEN

        else: # STATE_UNKNOWN
            if self.unknown_start_time is None:
                self.unknown_start_time = current_time

            unknown_elapsed = current_time - self.unknown_start_time
            if unknown_elapsed >= self.unknown_timeout:
                self.closure_duration = 0.0
                self.p_awake_buffer.clear()
            display_state = STATE_UNKNOWN

        is_alert = self.closure_duration >= self.alert_threshold

        return {
            "state": display_state,
            "raw_p_awake": raw_p_awake,
            "smoothed_p_awake": smoothed_p_awake,
            "closure_duration": self.closure_duration,
            "is_alert": is_alert
        }

    def reset(self):
        self.p_awake_buffer.clear()
        self.closure_duration = 0.0
        self.open_start_time = None
        self.unknown_start_time = None
        self.last_update_time = None

def play_audio_alert():
    """Plays non-blocking audible alert on Windows systems."""
    if HAS_WINSOUND:
        try:
            winsound.Beep(2500, 250)
        except Exception:
            pass

def draw_debug_pip(
    frame: np.ndarray,
    left_crop: np.ndarray | None,
    right_crop: np.ndarray | None,
    p_left: float | None,
    p_right: float | None,
    ear_left: float | None,
    ear_right: float | None,
    blink_left: float | None,
    blink_right: float | None
):
    """
    Renders picture-in-picture (PiP) tiles of the exact 64x64 crops fed to the CNN
    along with per-eye probabilities and landmark metrics.
    """
    h, w = frame.shape[:2]
    pip_w, pip_h = 240, 210
    pad = 12
    x1 = w - pip_w - pad
    y1 = 60
    x2 = x1 + pip_w
    y2 = y1 + pip_h

    # Card background overlay
    sub_img = frame[y1:y2, x1:x2]
    black_rect = np.full(sub_img.shape, 20, dtype=np.uint8)
    cv2.addWeighted(sub_img, 0.25, black_rect, 0.75, 0, sub_img)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 1)

    cv2.putText(frame, "CNN INPUT CROPS (64x64)", (x1 + 10, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)

    tile_size = 72
    # Right Eye (Viewer's Left)
    rx = x1 + 14
    ry = y1 + 30
    if right_crop is not None:
        r_bgr = cv2.cvtColor(right_crop, cv2.COLOR_GRAY2BGR)
        r_scaled = cv2.resize(r_bgr, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
        frame[ry:ry+tile_size, rx:rx+tile_size] = r_scaled
        r_border = (0, 255, 0) if (p_right is not None and p_right >= 0.5) else (0, 0, 255)
        cv2.rectangle(frame, (rx, ry), (rx + tile_size, ry + tile_size), r_border, 1)
        p_str = f"P:{p_right:.2f}" if p_right is not None else "P:N/A"
        cv2.putText(frame, "R Eye", (rx, ry + tile_size + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.putText(frame, p_str, (rx, ry + tile_size + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, r_border, 1)
    else:
        cv2.rectangle(frame, (rx, ry), (rx + tile_size, ry + tile_size), (80, 80, 80), 1)
        cv2.putText(frame, "No Eye", (rx + 10, ry + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    # Left Eye (Viewer's Right)
    lx = x1 + tile_size + 34
    ly = y1 + 30
    if left_crop is not None:
        l_bgr = cv2.cvtColor(left_crop, cv2.COLOR_GRAY2BGR)
        l_scaled = cv2.resize(l_bgr, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
        frame[ly:ly+tile_size, lx:lx+tile_size] = l_scaled
        l_border = (0, 255, 0) if (p_left is not None and p_left >= 0.5) else (0, 0, 255)
        cv2.rectangle(frame, (lx, ly), (lx + tile_size, ly + tile_size), l_border, 1)
        p_str = f"P:{p_left:.2f}" if p_left is not None else "P:N/A"
        cv2.putText(frame, "L Eye", (lx, ly + tile_size + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.putText(frame, p_str, (lx, ly + tile_size + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, l_border, 1)
    else:
        cv2.rectangle(frame, (lx, ly), (lx + tile_size, ly + tile_size), (80, 80, 80), 1)
        cv2.putText(frame, "No Eye", (lx + 10, ly + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    # Metric Footers
    mean_ear = np.mean([e for e in (ear_left, ear_right) if e is not None]) if any(e is not None for e in (ear_left, ear_right)) else None
    mean_blink = np.mean([b for b in (blink_left, blink_right) if b is not None]) if any(b is not None for b in (blink_left, blink_right)) else None
    ear_str = f"EAR: {mean_ear:.2f}" if mean_ear is not None else "EAR: N/A"
    blink_str = f"BlinkScore: {mean_blink:.2f}" if mean_blink is not None else "BlinkScore: N/A"
    cv2.putText(frame, f"{ear_str} | {blink_str}", (x1 + 10, y2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

def run_webcam_detector(
    model_path: str = "models/cnn_baseline_best.keras",
    landmarker_model: str = "models/face_landmarker.task",
    camera_id: int = 0,
    alert_threshold: float = 1.5,
    open_grace_period: float = 0.25,
    unknown_timeout: float = 0.5,
    window_size: int = 5,
    decision_threshold: float = 0.5,
    beep_cooldown: float = 1.0,
    max_test_frames: int = None,
    diagnostic_mode: bool = True,
    diagnostic_dir: str = "results/diagnostics/manual_samples"
):
    print("=" * 68)
    print("REAL-TIME DRIVER EYE-STATE & DROWSINESS DETECTOR (LANDMARK MESH)")
    print("=" * 68)

    # 1. Verify and Load CNN Model
    model_file = Path(model_path).resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_file}")

    print(f"Loading trained CNN checkpoint: {model_file.name}...")
    model = tf.keras.models.load_model(str(model_file))
    print(f"Loaded successfully. Model input shape: {model.input_shape}")

    # 2. Initialize Landmark Eye Extractor
    print(f"Initializing MediaPipe Face Landmarker from: {landmarker_model}...")
    extractor = LandmarkEyeExtractor(model_asset_path=landmarker_model)
    print("Face Landmarker initialized. (Landmark tracking active for open & closed eyes)")

    # 3. Initialize Temporal Tracker
    tracker = TemporalDrowsinessTracker(
        alert_threshold=alert_threshold,
        open_grace_period=open_grace_period,
        unknown_timeout=unknown_timeout,
        window_size=window_size,
        decision_threshold=decision_threshold
    )
    print(f"Temporal parameters: Alert Threshold={alert_threshold:.1f}s | Open Grace={open_grace_period*1000:.0f}ms | Unknown Timeout={unknown_timeout*1000:.0f}ms | Window={window_size}")

    # 4. Initialize Camera
    print(f"Connecting to video source (Camera ID: {camera_id})...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video device index {camera_id}.")
        print("Please verify that a webcam is plugged in and accessible, or specify a valid --camera-id.")
        return False

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("\nStarting video stream. Controls:")
    print("  [O] Key: Manually log & save current frame as OPEN ground truth")
    print("  [C] Key: Manually log & save current frame as CLOSED ground truth")
    print("  [Q] or [ESC]: Quit")
    print("-" * 68)

    last_beep_time = 0.0
    frame_count = 0
    fps_start_time = time.time()
    fps = 0.0

    manual_dir = Path(diagnostic_dir)
    manual_open_dir = manual_dir / "open"
    manual_closed_dir = manual_dir / "closed"
    manual_open_dir.mkdir(parents=True, exist_ok=True)
    manual_closed_dir.mkdir(parents=True, exist_ok=True)
    metadata_log = manual_dir / "sample_metadata.jsonl"

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Frame grab failed or video source disconnected.")
                break

            frame_count += 1
            current_time = time.time()

            # Smooth FPS calculation
            if current_time - fps_start_time >= 1.0:
                fps = frame_count / (current_time - fps_start_time)
                frame_count = 0
                fps_start_time = current_time

            h, w = frame.shape[:2]

            # Landmark-based Eye Extraction
            extraction = extractor.extract_eyes(frame)
            face_detected = extraction["face_detected"]
            right_data = extraction["right_eye"]
            left_data = extraction["left_eye"]

            p_right = None
            p_left = None
            raw_p_awake = None
            eye_preds = []

            # Right eye inference
            if right_data is not None:
                r_tensor = np.expand_dims(right_data["crop_64"], axis=(0, -1)).astype(np.float32)
                p_right = float(model(r_tensor, training=False).numpy()[0][0])
                eye_preds.append(p_right)

            # Left eye inference
            if left_data is not None:
                l_tensor = np.expand_dims(left_data["crop_64"], axis=(0, -1)).astype(np.float32)
                p_left = float(model(l_tensor, training=False).numpy()[0][0])
                eye_preds.append(p_left)

            if eye_preds:
                raw_p_awake = float(np.mean(eye_preds))

            # Update temporal tracker
            tracker_res = tracker.update(raw_p_awake, current_time)
            current_state = tracker_res["state"]
            smoothed_p_awake = tracker_res["smoothed_p_awake"]
            closure_duration = tracker_res["closure_duration"]
            is_alert = tracker_res["is_alert"]

            # Trigger audio alert with cooldown
            if is_alert:
                if (current_time - last_beep_time) >= beep_cooldown:
                    threading.Thread(target=play_audio_alert, daemon=True).start()
                    last_beep_time = current_time

            # Visual Overlays
            # 1. Landmark points & Eye boxes
            state_color = (0, 255, 0) if current_state == STATE_OPEN else ((0, 0, 255) if current_state == STATE_CLOSED else (180, 180, 180))

            if extraction["face_box"] is not None:
                fx, fy, fw, fh = extraction["face_box"]
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (80, 80, 80), 1)

            for eye_data in (right_data, left_data):
                if eye_data is not None:
                    bx, by, bw, bh = eye_data["box"]
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), state_color, 2)
                    for pt in eye_data["contour_pts"]:
                        cv2.circle(frame, pt, 1, (0, 255, 255), -1)

            # 2. Information Header Bar
            cv2.rectangle(frame, (0, 0), (w, 52), (22, 22, 22), -1)
            cv2.putText(frame, f"STATE: {current_state}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, state_color, 2)

            raw_str = f"Raw P: {raw_p_awake:.3f}" if raw_p_awake is not None else "Raw P: N/A"
            smooth_str = f"Smooth P: {smoothed_p_awake:.3f}" if smoothed_p_awake is not None else "Smooth: N/A"
            cv2.putText(frame, raw_str, (215, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            cv2.putText(frame, smooth_str, (370, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

            closed_color = (0, 0, 255) if closure_duration > 0 else (220, 220, 220)
            cv2.putText(frame, f"Closed: {closure_duration:.2f}s/{alert_threshold:.1f}s", (530, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, closed_color, 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

            # 3. Picture-in-Picture Debug Tiles
            right_crop = right_data["crop_64"] if right_data is not None else None
            left_crop = left_data["crop_64"] if left_data is not None else None
            ear_r = right_data["ear"] if right_data is not None else None
            ear_l = left_data["ear"] if left_data is not None else None
            blk_r = right_data["blink_score"] if right_data is not None else None
            blk_l = left_data["blink_score"] if left_data is not None else None

            draw_debug_pip(frame, left_crop, right_crop, p_left, p_right, ear_l, ear_r, blk_l, blk_r)

            # 4. Prominent Drowsiness Alert Banner
            if is_alert:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h // 2 - 45), (w, h // 2 + 45), (0, 0, 220), -1)
                cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

                alert_msg = f"*** DROWSINESS ALERT! EYES CLOSED > {alert_threshold:.1f}s ***"
                text_size = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(frame, alert_msg, (text_x, h // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3)

            # 5. Interactive Key Help Footer
            cv2.putText(frame, "Press [O] Log OPEN | [C] Log CLOSED | [Q] Exit", (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

            cv2.imshow("Real-Time Driver Drowsiness Detector (Landmark Mesh)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("Quit signal received ('q'/ESC). Terminating video stream.")
                break
            elif key == ord('o') or key == ord('O'):
                # Manual label OPEN
                ts = int(time.time() * 1000)
                saved_count = 0
                sample_entry = {
                    "timestamp": ts,
                    "label": "open",
                    "ground_truth": 1,
                    "raw_p_awake": raw_p_awake,
                    "smoothed_p_awake": smoothed_p_awake,
                    "p_right": p_right,
                    "p_left": p_left,
                    "ear_right": ear_r,
                    "ear_left": ear_l,
                    "blink_right": blk_r,
                    "blink_left": blk_l,
                    "predicted_state": current_state
                }

                if right_crop is not None:
                    cv2.imwrite(str(manual_open_dir / f"open_{ts}_right.png"), right_crop)
                    saved_count += 1
                if left_crop is not None:
                    cv2.imwrite(str(manual_open_dir / f"open_{ts}_left.png"), left_crop)
                    saved_count += 1
                cv2.imwrite(str(manual_open_dir / f"open_{ts}_frame.png"), frame)

                with open(metadata_log, "a") as f:
                    f.write(json.dumps(sample_entry) + "\n")

                print(f"[MANUAL LABEL: OPEN] Saved {saved_count} eye crops. Raw P: {raw_p_awake} | Smoothed P: {smoothed_p_awake} | State: {current_state}")

            elif key == ord('c') or key == ord('C'):
                # Manual label CLOSED
                ts = int(time.time() * 1000)
                saved_count = 0
                sample_entry = {
                    "timestamp": ts,
                    "label": "closed",
                    "ground_truth": 0,
                    "raw_p_awake": raw_p_awake,
                    "smoothed_p_awake": smoothed_p_awake,
                    "p_right": p_right,
                    "p_left": p_left,
                    "ear_right": ear_r,
                    "ear_left": ear_l,
                    "blink_right": blk_r,
                    "blink_left": blk_l,
                    "predicted_state": current_state
                }

                if right_crop is not None:
                    cv2.imwrite(str(manual_closed_dir / f"closed_{ts}_right.png"), right_crop)
                    saved_count += 1
                if left_crop is not None:
                    cv2.imwrite(str(manual_closed_dir / f"closed_{ts}_left.png"), left_crop)
                    saved_count += 1
                cv2.imwrite(str(manual_closed_dir / f"closed_{ts}_frame.png"), frame)

                with open(metadata_log, "a") as f:
                    f.write(json.dumps(sample_entry) + "\n")

                print(f"[MANUAL LABEL: CLOSED] Saved {saved_count} eye crops. Raw P: {raw_p_awake} | Smoothed P: {smoothed_p_awake} | State: {current_state}")
                if raw_p_awake is not None and raw_p_awake >= 0.5:
                    print(f"  >>> CNN DOMAIN MISMATCH: Closed eye scored P(awake)={raw_p_awake:.4f} >= 0.5 (False OPEN)!")

            if max_test_frames is not None and frame_count >= max_test_frames:
                print(f"Reached automated test limit ({max_test_frames} frames). Exiting test.")
                break

        return True

    except Exception as e:
        print(f"[ERROR] Exception during video processing: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera device released and OpenCV display windows closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Driver Drowsiness Detector (Landmark Mesh)")
    parser.add_argument("--model-path", type=str, default="models/cnn_baseline_best.keras", help="Path to trained CNN model")
    parser.add_argument("--landmarker-model", type=str, default="models/face_landmarker.task", help="Path to MediaPipe FaceLandmarker model")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--threshold", type=float, default=1.5, help="Sustained eye-closure threshold in seconds (default: 1.5)")
    parser.add_argument("--open-grace", type=float, default=0.25, help="OPEN reset grace period in seconds (default: 0.25)")
    parser.add_argument("--unknown-timeout", type=float, default=0.5, help="UNKNOWN timeout before resetting in seconds (default: 0.5)")
    parser.add_argument("--window-size", type=int, default=5, help="Rolling window size for smoothing (default: 5)")
    parser.add_argument("--decision-threshold", type=float, default=0.5, help="Decision threshold for P(awake) (default: 0.5)")
    parser.add_argument("--test-frames", type=int, default=None, help="Number of frames to process before auto-exit (for testing)")
    parser.add_argument("--diagnostic-dir", type=str, default="results/diagnostics/manual_samples", help="Path to save manual debug samples")
    args = parser.parse_args()

    run_webcam_detector(
        model_path=args.model_path,
        landmarker_model=args.landmarker_model,
        camera_id=args.camera_id,
        alert_threshold=args.threshold,
        open_grace_period=args.open_grace,
        unknown_timeout=args.unknown_timeout,
        window_size=args.window_size,
        decision_threshold=args.decision_threshold,
        max_test_frames=args.test_frames,
        diagnostic_dir=args.diagnostic_dir
    )
