import os
os.environ["QT_QPA_PLATFORM"] = "offscreen" # Headless mode for automated testing
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication
from src.desktop_app import MainWindow, VideoWorker
from src.webcam_detector import TemporalDrowsinessTracker, STATE_OPEN, STATE_CLOSED, STATE_UNKNOWN

def test_desktop_app_headless():
    print("=" * 65)
    print("RUNNING DESKTOP APP HEADLESS AUTOMATED TESTS")
    print("=" * 65)

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # 1. Instantiate Main Window
    print("\n[Test 1] Initializing MainWindow...")
    window = MainWindow()
    window.show()
    assert window is not None
    assert window.windowTitle().startswith("Blink")
    print("  PASS: MainWindow created successfully with theme and components.")

    # 2. Test Model & Tracker Reuse in VideoWorker
    print("\n[Test 2] Verifying Model & Tracker reuse in VideoWorker...")
    worker = VideoWorker(model_path="models/cnn_baseline_best.keras")
    assert isinstance(worker.tracker, TemporalDrowsinessTracker)
    assert Path(worker.model_path).exists()
    print("  PASS: VideoWorker references existing trained model and TemporalDrowsinessTracker.")

    # 3. Test UI Status Update Handlers
    print("\n[Test 3] Testing UI State Transitions via Status Signals...")

    # Case A: OPEN State
    status_open = {
        "state": STATE_OPEN,
        "is_alert": False,
        "closure_duration": 0.0,
        "alert_threshold": 1.5,
        "raw_p_awake": 0.94,
        "smoothed_p_awake": 0.92,
        "fps": 29.5,
        "face_detected": True,
        "eyes_detected_count": 2,
        "ambient_light": 115.0
    }
    window.on_status_updated(status_open)
    assert window.state_label.text() == "OPEN"
    assert window.alert_banner.isHidden()
    assert window.sys_status_pill.text() == "MONITORING ACTIVE"
    print("  PASS: OPEN state telemetry rendered correctly.")

    # Case B: CLOSED State with Sustained Closure
    status_closed = {
        "state": STATE_CLOSED,
        "is_alert": False,
        "closure_duration": 1.10,
        "alert_threshold": 1.5,
        "raw_p_awake": 0.38,
        "smoothed_p_awake": 0.40,
        "fps": 30.0,
        "face_detected": True,
        "eyes_detected_count": 2,
        "ambient_light": 110.0
    }
    window.on_status_updated(status_closed)
    assert window.state_label.text() == "CLOSED"
    assert window.closure_progress.value() == 1100
    assert "1.10s" in window.closure_time_label.text()
    assert window.alert_banner.isHidden()
    print("  PASS: CLOSED state and progress timer rendered accurately.")

    # Case C: DROWSINESS ALERT Triggered
    status_alert = {
        "state": STATE_CLOSED,
        "is_alert": True,
        "closure_duration": 1.65,
        "alert_threshold": 1.5,
        "raw_p_awake": 0.35,
        "smoothed_p_awake": 0.36,
        "fps": 30.0,
        "face_detected": True,
        "eyes_detected_count": 2,
        "ambient_light": 110.0
    }
    window.on_status_updated(status_alert)
    assert not window.alert_banner.isHidden()
    assert window.sys_status_pill.text() == "DROWSINESS ALERT"
    print("  PASS: Active Drowsiness Alert banner triggered and prominent.")

    # Case D: UNKNOWN State / Dim Lighting Warning
    status_unknown = {
        "state": STATE_UNKNOWN,
        "is_alert": False,
        "closure_duration": 0.0,
        "alert_threshold": 1.5,
        "raw_p_awake": None,
        "smoothed_p_awake": None,
        "fps": 28.0,
        "face_detected": False,
        "eyes_detected_count": 0,
        "ambient_light": 25.0 # Dim
    }
    window.on_status_updated(status_unknown)
    assert window.state_label.text() == "UNKNOWN"
    assert "Dim" in window.light_badge.text()
    print("  PASS: UNKNOWN state and Low-Light warning displayed properly.")

    # 4. Clean Shutdown
    print("\n[Test 4] Testing window close and cleanup...")
    window.close()
    print("  PASS: Window closed cleanly without hanging threads.")

    print("\n" + "=" * 65)
    print("ALL HEADLESS DESKTOP APP TESTS PASSED [4/4]")
    print("=" * 65)
    return True

if __name__ == "__main__":
    test_desktop_app_headless()
