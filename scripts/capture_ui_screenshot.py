import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase, QFont
from src.desktop_app import MainWindow
from src.webcam_detector import STATE_OPEN, STATE_CLOSED, STATE_UNKNOWN

def capture_ui_states():
    out_dir = Path("results/diagnostics/ui_screens")
    out_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Load Windows system fonts into Qt offscreen font database
    font_files = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/segoeuisb.ttf",
        "C:/Windows/Fonts/consolas.ttf",
        "C:/Windows/Fonts/consolab.ttf",
        "C:/Windows/Fonts/seguisym.ttf",
        "C:/Windows/Fonts/arial.ttf"
    ]
    for ff in font_files:
        if Path(ff).exists():
            QFontDatabase.addApplicationFont(ff)

    app.setFont(QFont("Segoe UI", 10))

    window = MainWindow()
    window.resize(1200, 800)
    window.show()

    # Helper to capture window
    def save_screen(name):
        app.processEvents()
        pix = window.grab()
        file_path = out_dir / f"{name}.png"
        pix.save(str(file_path))
        print(f"Saved {name} to {file_path}")
        return file_path

    # State 1: Idle
    save_screen("01_idle_standby")

    # State 2: Monitoring - OPEN
    status_open = {
        "state": STATE_OPEN,
        "is_alert": False,
        "closure_duration": 0.0,
        "alert_threshold": 1.5,
        "raw_p_awake": 0.96,
        "smoothed_p_awake": 0.94,
        "p_right": 0.95,
        "p_left": 0.97,
        "ear": 0.29,
        "blink_score": 0.02,
        "fps": 30.0,
        "face_detected": True,
        "eyes_detected_count": 2,
        "ambient_light": 115.0
    }
    window.on_status_updated(status_open)
    save_screen("02_monitoring_open")

    # State 3: Monitoring - CLOSED (Accumulating)
    status_closed = {
        "state": STATE_CLOSED,
        "is_alert": False,
        "closure_duration": 1.10,
        "alert_threshold": 1.5,
        "raw_p_awake": 0.12,
        "smoothed_p_awake": 0.18,
        "p_right": 0.10,
        "p_left": 0.14,
        "ear": 0.11,
        "blink_score": 0.88,
        "fps": 30.0,
        "face_detected": True,
        "eyes_detected_count": 2,
        "ambient_light": 115.0
    }
    window.on_status_updated(status_closed)
    save_screen("03_monitoring_closed")

    # State 4: Monitoring - ALERT (Drowsiness Alert Triggered)
    status_alert = {
        "state": STATE_CLOSED,
        "is_alert": True,
        "closure_duration": 1.65,
        "alert_threshold": 1.5,
        "raw_p_awake": 0.05,
        "smoothed_p_awake": 0.08,
        "p_right": 0.04,
        "p_left": 0.06,
        "ear": 0.08,
        "blink_score": 0.95,
        "fps": 30.0,
        "face_detected": True,
        "eyes_detected_count": 2,
        "ambient_light": 115.0
    }
    window.on_status_updated(status_alert)
    save_screen("04_drowsiness_alert")

    # State 5: Monitoring - UNKNOWN (Face not found)
    status_unknown = {
        "state": STATE_UNKNOWN,
        "is_alert": False,
        "closure_duration": 0.0,
        "alert_threshold": 1.5,
        "raw_p_awake": None,
        "smoothed_p_awake": None,
        "p_right": None,
        "p_left": None,
        "ear": None,
        "blink_score": None,
        "fps": 28.5,
        "face_detected": False,
        "eyes_detected_count": 0,
        "ambient_light": 28.0 # Dim
    }
    window.on_status_updated(status_unknown)
    save_screen("05_unknown_low_light")

    # State 6: Collapsible Panels Expanded
    window.toggle_params_panel()
    window.toggle_diag_panel()
    save_screen("06_panels_expanded")

    # State 7: Min Size Responsive (1000x680)
    window.resize(1000, 680)
    save_screen("07_min_window_1000x680")

    # State 8: Large Size Responsive (1440x900)
    window.resize(1440, 900)
    save_screen("08_large_window_1440x900")

    window.close()
    print("All UI state screenshots captured successfully with loaded fonts!")

if __name__ == "__main__":
    capture_ui_states()
