import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import time
import threading
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2

from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QColor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
    QProgressBar,
    QFrame,
    QMessageBox,
    QScrollArea
)

# Reuse existing verified tracker and audio helper
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.webcam_detector import (
    TemporalDrowsinessTracker,
    STATE_OPEN,
    STATE_CLOSED,
    STATE_UNKNOWN,
    play_audio_alert
)
from src.eye_extractor import LandmarkEyeExtractor

class VideoWorker(QThread):
    """
    Dedicated background worker thread for camera acquisition,
    MediaPipe FaceMesh ocular localization, and CNN inference.
    Runs asynchronously off the UI main thread.
    """
    frame_ready = Signal(QImage)
    status_updated = Signal(dict)
    error_occurred = Signal(str)

    def __init__(
        self,
        model_path="models/cnn_baseline_best.keras",
        camera_id=0,
        alert_threshold=1.5,
        open_grace=0.25,
        unknown_timeout=0.5,
        window_size=5,
        decision_threshold=0.5
    ):
        super().__init__()
        self.model_path = model_path
        self.camera_id = camera_id
        self.alert_threshold = alert_threshold
        self.open_grace = open_grace
        self.unknown_timeout = unknown_timeout
        self.window_size = window_size
        self.decision_threshold = decision_threshold
        
        self.is_running = False
        self.audio_enabled = True
        self.save_crop_requested = False
        self.diag_crop_dir = Path("results/diagnostics/crops")

        self.tracker = TemporalDrowsinessTracker(
            alert_threshold=self.alert_threshold,
            open_grace_period=self.open_grace,
            unknown_timeout=self.unknown_timeout,
            window_size=self.window_size,
            decision_threshold=self.decision_threshold
        )

    def update_settings(self, camera_id, threshold, grace, unknown_to, win_size):
        self.camera_id = camera_id
        self.alert_threshold = threshold
        self.open_grace = grace
        self.unknown_timeout = unknown_to
        self.window_size = win_size
        self.tracker = TemporalDrowsinessTracker(
            alert_threshold=threshold,
            open_grace_period=grace,
            unknown_timeout=unknown_to,
            window_size=win_size,
            decision_threshold=self.decision_threshold
        )

    def request_crop_save(self):
        self.save_crop_requested = True

    def stop(self):
        self.is_running = False

    def run(self):
        self.is_running = True

        # Load CNN Model
        try:
            m_path = Path(self.model_path).resolve()
            if not m_path.exists():
                self.error_occurred.emit(f"Model checkpoint not found: {m_path}")
                return
            model = tf.keras.models.load_model(str(m_path))
        except Exception as e:
            self.error_occurred.emit(f"Failed to load model: {e}")
            return

        # Initialize Landmark Eye Extractor
        try:
            extractor = LandmarkEyeExtractor(model_asset_path="models/face_landmarker.task")
        except Exception as e:
            self.error_occurred.emit(f"Face Landmarker initialization failed: {e}")
            return

        # Open Video Capture
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            self.error_occurred.emit(f"Cannot open webcam (Device Index: {self.camera_id}). Please verify camera connection.")
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        last_beep_time = 0.0
        beep_cooldown = 1.0
        frame_count = 0
        fps_start = time.time()
        fps = 0.0

        while self.is_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                self.error_occurred.emit("Video stream disconnected or frame grab failed.")
                break

            current_time = time.time()
            frame_count += 1
            if current_time - fps_start >= 1.0:
                fps = frame_count / (current_time - fps_start)
                frame_count = 0
                fps_start = current_time

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = frame.shape[:2]
            ambient_light = float(np.mean(gray))

            # Landmark-based Eye Extraction
            extraction = extractor.extract_eyes(frame)
            face_detected = extraction["face_detected"]
            right_data = extraction["right_eye"]
            left_data = extraction["left_eye"]

            raw_p_awake = None
            eye_preds = []
            detected_eyes = []
            p_right = None
            p_left = None

            if right_data is not None:
                detected_eyes.append(right_data["box"])
                r_tensor = np.expand_dims(right_data["crop_64"], axis=(0, -1)).astype(np.float32)
                p_right = float(model(r_tensor, training=False).numpy()[0][0])
                eye_preds.append(p_right)

            if left_data is not None:
                detected_eyes.append(left_data["box"])
                l_tensor = np.expand_dims(left_data["crop_64"], axis=(0, -1)).astype(np.float32)
                p_left = float(model(l_tensor, training=False).numpy()[0][0])
                eye_preds.append(p_left)

            if eye_preds:
                raw_p_awake = float(np.mean(eye_preds))

            # Optional Diagnostic Crop Saving
            if self.save_crop_requested:
                self.diag_crop_dir.mkdir(parents=True, exist_ok=True)
                if right_data is not None:
                    cv2.imwrite(str(self.diag_crop_dir / f"manual_crop_{int(current_time)}_right.png"), right_data["crop_64"])
                if left_data is not None:
                    cv2.imwrite(str(self.diag_crop_dir / f"manual_crop_{int(current_time)}_left.png"), left_data["crop_64"])
                self.save_crop_requested = False

            # Update Temporal Tracker
            tracker_out = self.tracker.update(raw_p_awake, current_time)
            state = tracker_out["state"]
            is_alert = tracker_out["is_alert"]
            closure_duration = tracker_out["closure_duration"]
            smoothed_p_awake = tracker_out["smoothed_p_awake"]

            # Audible Alert with Cooldown
            if is_alert and self.audio_enabled:
                if (current_time - last_beep_time) >= beep_cooldown:
                    threading.Thread(target=play_audio_alert, daemon=True).start()
                    last_beep_time = current_time

            # Refined Overlays on Preview Frame
            # Pitch Black Palette: Green #22C55E (OPEN), Red #EF4444 (CLOSED), Muted Gray #71717A (UNKNOWN)
            # OpenCV BGR format:
            # Green (34, 197, 94)
            # Red (68, 68, 239)
            # Gray (122, 113, 113)
            # Electric Blue contour dots: (235, 99, 37)
            eye_color = (34, 197, 94) if state == STATE_OPEN else ((68, 68, 239) if state == STATE_CLOSED else (122, 113, 113))
            if extraction["face_box"] is not None:
                fx, fy, fw, fh = extraction["face_box"]
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (38, 38, 38), 1)

            for eye_data in (right_data, left_data):
                if eye_data is not None:
                    bx, by, bw, bh = eye_data["box"]
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), eye_color, 2)
                    for pt in eye_data["contour_pts"]:
                        cv2.circle(frame, pt, 1, (235, 99, 37), -1)

            # Convert BGR to QImage for Qt UI display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bytes_per_line = 3 * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

            self.frame_ready.emit(q_img)

            # Extract EAR and Blink metrics for telemetry
            ear_val = None
            if right_data is not None and left_data is not None:
                ear_val = (right_data["ear"] + left_data["ear"]) / 2.0
            elif right_data is not None:
                ear_val = right_data["ear"]
            elif left_data is not None:
                ear_val = left_data["ear"]

            blink_val = None
            if right_data is not None and left_data is not None:
                blink_val = (right_data["blink_score"] + left_data["blink_score"]) / 2.0
            elif right_data is not None:
                blink_val = right_data["blink_score"]
            elif left_data is not None:
                blink_val = left_data["blink_score"]

            # Emit Status Dict for UI Widgets
            status_data = {
                "state": state,
                "is_alert": is_alert,
                "closure_duration": closure_duration,
                "alert_threshold": self.alert_threshold,
                "raw_p_awake": raw_p_awake,
                "smoothed_p_awake": smoothed_p_awake,
                "p_right": p_right,
                "p_left": p_left,
                "ear": ear_val,
                "blink_score": blink_val,
                "fps": fps,
                "face_detected": face_detected,
                "eyes_detected_count": len(detected_eyes),
                "ambient_light": ambient_light
            }
            self.status_updated.emit(status_data)

        cap.release()


class MainWindow(QMainWindow):
    """
    Blink | Driver Monitoring System.
    Design concept: PITCH BLACK.
    Ultra-dark, minimalist, premium desktop interface using pure black backgrounds (#000000),
    subtly differentiated dark surfaces (#050505 / #0A0A0A), thin charcoal borders (#242424),
    white typography, and restrained electric-blue accents (#2563EB / #3B82F6).
    Green for healthy attentive status, and red reserved strictly for genuine drowsiness warnings.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blink | Driver Monitoring")
        self.resize(1220, 810)
        self.setMinimumSize(1000, 680)

        self.worker = None
        self._current_progress_mode = "idle"
        self.setup_ui()
        self.apply_dark_theme()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 16, 20, 18)
        main_layout.setSpacing(14)

        # 1. INTEGRATED APP HEADER BAR
        header_bar = QWidget()
        header_bar.setObjectName("topHeader")
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(4, 2, 4, 10)
        header_layout.setSpacing(14)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        
        app_title = QLabel("BLINK")
        app_title.setObjectName("appTitle")
        
        app_badge = QLabel("CV ENGINE")
        app_badge.setObjectName("appBadge")
        
        title_row.addWidget(app_title)
        title_row.addWidget(app_badge)
        title_row.addStretch()

        app_subtitle = QLabel("Real-Time Driver Drowsiness Detection")
        app_subtitle.setObjectName("appSubtitle")
        
        title_box.addLayout(title_row)
        title_box.addWidget(app_subtitle)

        header_layout.addLayout(title_box)
        header_layout.addStretch()

        # Understated Connection & Status Badges
        self.cam_indicator = QLabel("○ Camera Standby")
        self.cam_indicator.setObjectName("camStandby")
        header_layout.addWidget(self.cam_indicator)

        self.sys_status_pill = QLabel("SYSTEM IDLE")
        self.sys_status_pill.setObjectName("statusPillIdle")
        header_layout.addWidget(self.sys_status_pill)

        main_layout.addWidget(header_bar)

        # 2. BODY WORKSPACE (Left: Camera Viewport, Right: Unified Control Deck)
        body_layout = QHBoxLayout()
        body_layout.setSpacing(16)

        # Left Column: Integrated Camera Workspace (Stretch 65%)
        self.preview_card = QFrame()
        self.preview_card.setObjectName("cardFrame")
        preview_layout = QVBoxLayout(self.preview_card)
        preview_layout.setContentsMargins(14, 14, 14, 14)
        preview_layout.setSpacing(10)

        # High-Urgency Alert Banner (Hidden by default; appears only during genuine closure alerts)
        self.alert_banner = QLabel("⚠ DROWSINESS ALERT — SUSTAINED EYE CLOSURE DETECTED")
        self.alert_banner.setObjectName("alertBanner")
        self.alert_banner.setAlignment(Qt.AlignCenter)
        self.alert_banner.setVisible(False)
        preview_layout.addWidget(self.alert_banner)

        # Viewport Frame
        self.preview_label = QLabel()
        self.preview_label.setObjectName("videoViewport")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setText("CAMERA STREAM OFFLINE\nClick 'Start Monitoring' to initiate ocular tracking")
        preview_layout.addWidget(self.preview_label, stretch=1)

        # Integrated HUD Overlay Ribbon
        hud_bar = QFrame()
        hud_bar.setObjectName("hudBar")
        hud_layout = QHBoxLayout(hud_bar)
        hud_layout.setContentsMargins(12, 6, 12, 6)
        hud_layout.setSpacing(10)

        self.fps_badge = QLabel("FPS: 0.0")
        self.fps_badge.setObjectName("hudBadge")
        self.face_badge = QLabel("Face: None")
        self.face_badge.setObjectName("hudBadge")
        self.ear_badge = QLabel("EAR: --")
        self.ear_badge.setObjectName("hudBadge")
        self.light_badge = QLabel("Lighting: Normal")
        self.light_badge.setObjectName("hudBadge")

        hud_layout.addWidget(self.fps_badge)
        hud_layout.addWidget(self.face_badge)
        hud_layout.addWidget(self.ear_badge)
        hud_layout.addWidget(self.light_badge)
        hud_layout.addStretch()
        preview_layout.addWidget(hud_bar)

        body_layout.addWidget(self.preview_card, stretch=65)

        # Right Column: Unified Control Deck (Stretch 35%)
        self.sidebar_card = QFrame()
        self.sidebar_card.setObjectName("cardFrame")
        sidebar_card_layout = QVBoxLayout(self.sidebar_card)
        sidebar_card_layout.setContentsMargins(2, 2, 2, 2)
        sidebar_card_layout.setSpacing(0)

        # Responsive Scroll Area (preserves layout proportions at smaller window sizes)
        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setObjectName("sidebarScroll")
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setFrameShape(QFrame.NoFrame)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        scroll_content = QWidget()
        scroll_content.setObjectName("scrollContent")
        sidebar_layout = QVBoxLayout(scroll_content)
        sidebar_layout.setContentsMargins(16, 16, 16, 16)
        sidebar_layout.setSpacing(14)

        # SECTION 1: OCULAR STATE MONITOR (HERO)
        state_section = QVBoxLayout()
        state_section.setSpacing(8)

        state_heading = QLabel("OCULAR STATE")
        state_heading.setObjectName("sectionHeading")
        state_section.addWidget(state_heading)

        # Prominent State Pill
        self.state_label = QLabel("UNKNOWN")
        self.state_label.setObjectName("stateUnknown")
        self.state_label.setAlignment(Qt.AlignCenter)
        self.state_label.setMinimumHeight(48)
        state_section.addWidget(self.state_label)

        # Explanatory State Subtext
        self.state_desc_label = QLabel("System on standby • Camera inactive")
        self.state_desc_label.setObjectName("stateDesc")
        self.state_desc_label.setAlignment(Qt.AlignCenter)
        state_section.addWidget(self.state_desc_label)

        # Minimal Sustained-Closure Progress & Readout
        timer_header = QHBoxLayout()
        timer_title = QLabel("Closure Duration:")
        timer_title.setObjectName("captionLabel")
        self.closure_time_label = QLabel("0.00s / 1.50s")
        self.closure_time_label.setObjectName("timerValueLabel")
        timer_header.addWidget(timer_title)
        timer_header.addStretch()
        timer_header.addWidget(self.closure_time_label)
        state_section.addLayout(timer_header)

        self.closure_progress = QProgressBar()
        self.closure_progress.setObjectName("closureProgress")
        self.closure_progress.setRange(0, 1500)
        self.closure_progress.setValue(0)
        self.closure_progress.setTextVisible(False)
        state_section.addWidget(self.closure_progress)

        grace_caption = QLabel("Grace Reset: 250ms • Unknown Timeout: 500ms")
        grace_caption.setObjectName("graceCaption")
        state_section.addWidget(grace_caption)

        sidebar_layout.addLayout(state_section)

        # Hairline Divider 1
        div1 = QFrame()
        div1.setObjectName("hairlineDivider")
        div1.setFrameShape(QFrame.HLine)
        sidebar_layout.addWidget(div1)

        # SECTION 2: COMMAND & AUDIO
        ctrl_section = QVBoxLayout()
        ctrl_section.setSpacing(10)

        ctrl_heading = QLabel("COMMAND & AUDIO")
        ctrl_heading.setObjectName("sectionHeading")
        ctrl_section.addWidget(ctrl_heading)

        self.btn_toggle_monitor = QPushButton("Start Monitoring")
        self.btn_toggle_monitor.setObjectName("btnPrimary")
        self.btn_toggle_monitor.clicked.connect(self.toggle_monitoring)
        ctrl_section.addWidget(self.btn_toggle_monitor)

        self.chk_audio = QCheckBox("Audible Buzzer Alert (2.5 kHz)")
        self.chk_audio.setChecked(True)
        self.chk_audio.toggled.connect(self.on_audio_toggled)
        ctrl_section.addWidget(self.chk_audio)

        sidebar_layout.addLayout(ctrl_section)

        # Hairline Divider 2
        div2 = QFrame()
        div2.setObjectName("hairlineDivider")
        div2.setFrameShape(QFrame.HLine)
        sidebar_layout.addWidget(div2)

        # SECTION 3: DETECTION PARAMETERS (COLLAPSIBLE)
        param_section = QVBoxLayout()
        param_section.setSpacing(6)

        self.btn_toggle_params = QPushButton("▸ Detection Parameters")
        self.btn_toggle_params.setObjectName("btnParamsToggle")
        self.btn_toggle_params.clicked.connect(self.toggle_params_panel)
        param_section.addWidget(self.btn_toggle_params)

        self.params_content_widget = QWidget()
        params_content_layout = QVBoxLayout(self.params_content_widget)
        params_content_layout.setContentsMargins(2, 6, 2, 4)
        params_content_layout.setSpacing(8)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        grid.addWidget(QLabel("Camera ID:"), 0, 0)
        self.spin_cam_id = QSpinBox()
        self.spin_cam_id.setRange(0, 10)
        self.spin_cam_id.setValue(0)
        grid.addWidget(self.spin_cam_id, 0, 1)

        grid.addWidget(QLabel("Alert Threshold (s):"), 1, 0)
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.5, 5.0)
        self.spin_threshold.setSingleStep(0.1)
        self.spin_threshold.setValue(1.5)
        grid.addWidget(self.spin_threshold, 1, 1)

        grid.addWidget(QLabel("Open Grace (s):"), 2, 0)
        self.spin_grace = QDoubleSpinBox()
        self.spin_grace.setRange(0.05, 1.0)
        self.spin_grace.setSingleStep(0.05)
        self.spin_grace.setValue(0.25)
        grid.addWidget(self.spin_grace, 2, 1)

        params_content_layout.addLayout(grid)

        self.btn_apply_settings = QPushButton("Apply Parameters")
        self.btn_apply_settings.setObjectName("btnSecondary")
        self.btn_apply_settings.clicked.connect(self.apply_settings)
        params_content_layout.addWidget(self.btn_apply_settings)

        self.params_content_widget.setVisible(False)
        param_section.addWidget(self.params_content_widget)

        sidebar_layout.addLayout(param_section)

        # Hairline Divider 3
        div3 = QFrame()
        div3.setObjectName("hairlineDivider")
        div3.setFrameShape(QFrame.HLine)
        sidebar_layout.addWidget(div3)

        # SECTION 4: ADVANCED TELEMETRY & DIAGNOSTICS (COLLAPSIBLE)
        diag_section = QVBoxLayout()
        diag_section.setSpacing(6)

        self.btn_toggle_diag = QPushButton("▸ Advanced Telemetry && Diagnostics")
        self.btn_toggle_diag.setObjectName("btnDiagToggle")
        self.btn_toggle_diag.clicked.connect(self.toggle_diag_panel)
        diag_section.addWidget(self.btn_toggle_diag)

        self.diag_content_widget = QWidget()
        diag_content_layout = QVBoxLayout(self.diag_content_widget)
        diag_content_layout.setContentsMargins(2, 6, 2, 0)
        diag_content_layout.setSpacing(6)

        self.lbl_raw_p = QLabel("Raw P(Awake): N/A")
        self.lbl_raw_p.setObjectName("diagMetric")
        self.lbl_smooth_p = QLabel("Smoothed P(Awake): N/A")
        self.lbl_smooth_p.setObjectName("diagMetric")
        self.lbl_ear_blink = QLabel("Landmarks: EAR=N/A | Blink=N/A")
        self.lbl_ear_blink.setObjectName("diagMetric")
        self.lbl_ambient = QLabel("Mean Luminance: N/A")
        self.lbl_ambient.setObjectName("diagMetric")
        self.lbl_det_status = QLabel("Detector Status: Ready")
        self.lbl_det_status.setObjectName("diagMetric")

        diag_content_layout.addWidget(self.lbl_raw_p)
        diag_content_layout.addWidget(self.lbl_smooth_p)
        diag_content_layout.addWidget(self.lbl_ear_blink)
        diag_content_layout.addWidget(self.lbl_ambient)
        diag_content_layout.addWidget(self.lbl_det_status)

        self.btn_save_crop = QPushButton("Save Diagnostic Eye Crop")
        self.btn_save_crop.setObjectName("btnSecondary")
        self.btn_save_crop.clicked.connect(self.on_save_crop_clicked)
        diag_content_layout.addWidget(self.btn_save_crop)

        self.lbl_crop_status = QLabel("")
        self.lbl_crop_status.setObjectName("cropStatus")
        self.lbl_crop_status.setAlignment(Qt.AlignCenter)
        diag_content_layout.addWidget(self.lbl_crop_status)

        self.diag_content_widget.setVisible(False)
        diag_section.addWidget(self.diag_content_widget)

        sidebar_layout.addLayout(diag_section)
        sidebar_layout.addStretch()

        self.sidebar_scroll.setWidget(scroll_content)
        sidebar_card_layout.addWidget(self.sidebar_scroll)

        body_layout.addWidget(self.sidebar_card, stretch=35)
        main_layout.addLayout(body_layout)

    def apply_dark_theme(self):
        style = """
        /* Base Pitch Black Canvas */
        QWidget {
            background-color: #000000;
            color: #EDEDED;
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Inter', Roboto, sans-serif;
            font-size: 13px;
        }

        /* Top Header Area */
        #topHeader {
            background: #000000;
            border-bottom: 1px solid #1A1A1A;
        }

        /* Subtly Differentiated Pitch Black Card Surfaces */
        #cardFrame {
            background-color: #050505;
            border: 1px solid #1A1A1A;
            border-radius: 8px;
        }

        /* Sidebar Scroll Container */
        #sidebarScroll, #scrollContent {
            background-color: transparent;
            border: none;
        }
        QScrollBar:vertical {
            background-color: #050505;
            width: 5px;
            margin: 0px;
            border-radius: 2px;
        }
        QScrollBar::handle:vertical {
            background-color: #242424;
            min-height: 24px;
            border-radius: 2px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #3B82F6;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: none;
        }

        /* Hairline Divider */
        #hairlineDivider {
            border: none;
            background-color: #1A1A1A;
            height: 1px;
            margin: 3px 0px;
        }

        /* Brand & Typography */
        #appTitle {
            font-size: 18px;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: 1.2px;
        }
        #appBadge {
            font-size: 10px;
            font-weight: 700;
            color: #3B82F6;
            background-color: #051426;
            border: 1px solid #1D4ED8;
            border-radius: 3px;
            padding: 2px 7px;
            letter-spacing: 0.6px;
        }
        #appSubtitle {
            font-size: 11px;
            color: #71717A;
            letter-spacing: 0.2px;
            background: transparent;
        }

        /* Header Status Badges */
        #camStandby {
            background-color: #0A0A0A;
            color: #71717A;
            border: 1px solid #242424;
            padding: 5px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }
        #camActive {
            background-color: #051426;
            color: #3B82F6;
            border: 1px solid #1D4ED8;
            padding: 5px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }
        #statusPillIdle {
            background-color: #0A0A0A;
            color: #71717A;
            border: 1px solid #242424;
            padding: 5px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        #statusPillActive {
            background-color: #05180C;
            color: #22C55E;
            border: 1px solid #16A34A;
            padding: 5px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        #statusPillAlert {
            background-color: #1C0709;
            color: #EF4444;
            border: 1px solid #7F1D1D;
            padding: 5px 12px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }

        /* Video Viewport */
        #videoViewport {
            background-color: #000000;
            border: 1px solid #1A1A1A;
            border-radius: 6px;
            color: #52525B;
            font-size: 12px;
            line-height: 1.6;
        }

        /* High-Urgency Alert Banner (Reserved Red) */
        #alertBanner {
            background-color: #DC2626;
            color: #FFFFFF;
            font-weight: 700;
            font-size: 12px;
            padding: 9px 14px;
            border-radius: 4px;
            letter-spacing: 0.8px;
            border: 1px solid #EF4444;
        }

        /* Viewport HUD Ribbon */
        #hudBar {
            background-color: #050505;
            border: 1px solid #1A1A1A;
            border-radius: 4px;
        }
        #hudBadge {
            background-color: #0A0A0A;
            color: #A1A1AA;
            border: 1px solid #242424;
            padding: 4px 9px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 600;
            font-family: 'Consolas', 'Courier New', monospace;
        }

        /* Section Headings */
        #sectionHeading {
            font-size: 10px;
            font-weight: 700;
            color: #71717A;
            letter-spacing: 1px;
            text-transform: uppercase;
            background: transparent;
        }
        #captionLabel {
            color: #A1A1AA;
            font-size: 12px;
            background: transparent;
        }
        #timerValueLabel {
            font-weight: 700;
            color: #FFFFFF;
            font-size: 13px;
            font-family: 'Consolas', 'Courier New', monospace;
            background: transparent;
        }
        #graceCaption {
            color: #52525B;
            font-size: 10px;
            letter-spacing: 0.2px;
            background: transparent;
        }

        /* Hero Ocular State Badges */
        #stateOpen {
            font-size: 22px;
            font-weight: 800;
            color: #22C55E;
            background-color: #05180C;
            border: 1px solid #16A34A;
            border-radius: 6px;
            padding: 12px;
            letter-spacing: 1.2px;
        }
        #stateClosed {
            font-size: 22px;
            font-weight: 800;
            color: #EF4444;
            background-color: #1C0709;
            border: 1px solid #7F1D1D;
            border-radius: 6px;
            padding: 12px;
            letter-spacing: 1.2px;
        }
        #stateUnknown {
            font-size: 22px;
            font-weight: 800;
            color: #A1A1AA;
            background-color: #0A0A0A;
            border: 1px solid #242424;
            border-radius: 6px;
            padding: 12px;
            letter-spacing: 1.2px;
        }
        #stateDesc {
            font-size: 11px;
            color: #71717A;
            padding-top: 1px;
            background: transparent;
        }

        /* Minimal Closure Progress Indicator */
        #closureProgress {
            border: 1px solid #1F1F1F;
            border-radius: 3px;
            background-color: #000000;
            height: 6px;
        }
        #closureProgress::chunk {
            background-color: #3B82F6;
            border-radius: 2px;
        }

        /* Interactive Buttons */
        #btnPrimary {
            background-color: #2563EB;
            color: #FFFFFF;
            font-weight: 700;
            font-size: 13px;
            padding: 10px 18px;
            border-radius: 5px;
            border: 1px solid #3B82F6;
        }
        #btnPrimary:hover {
            background-color: #3B82F6;
            border: 1px solid #60A5FA;
        }
        #btnPrimary:pressed {
            background-color: #1D4ED8;
            border: 1px solid #2563EB;
        }
        /* Stop Monitoring: Charcoal Neutral, NOT RED */
        #btnPrimaryActive {
            background-color: #18181B;
            color: #FFFFFF;
            font-weight: 700;
            font-size: 13px;
            padding: 10px 18px;
            border-radius: 5px;
            border: 1px solid #3F3F46;
        }
        #btnPrimaryActive:hover {
            background-color: #27272A;
            border: 1px solid #52525B;
        }
        #btnPrimaryActive:pressed {
            background-color: #09090B;
            border: 1px solid #27272A;
        }
        #btnSecondary {
            background-color: #0A0A0A;
            color: #D4D4D8;
            padding: 6px 12px;
            border-radius: 4px;
            border: 1px solid #242424;
            font-weight: 600;
            font-size: 12px;
        }
        #btnSecondary:hover {
            background-color: #141414;
            border-color: #3B82F6;
            color: #FFFFFF;
        }
        #btnSecondary:pressed {
            background-color: #050505;
            border-color: #1D4ED8;
        }
        #btnDiagToggle, #btnParamsToggle {
            background: transparent;
            color: #A1A1AA;
            text-align: left;
            padding: 4px 0;
            border: none;
            font-weight: 600;
            font-size: 12px;
        }
        #btnDiagToggle:hover, #btnParamsToggle:hover {
            color: #3B82F6;
        }
        #btnDiagToggle:pressed, #btnParamsToggle:pressed {
            color: #1D4ED8;
        }

        /* Telemetry Details */
        #diagMetric {
            color: #71717A;
            font-size: 11px;
            font-family: 'Consolas', 'Courier New', monospace;
            background: transparent;
        }
        #cropStatus {
            color: #22C55E;
            font-size: 11px;
            font-weight: 600;
            background: transparent;
        }

        /* Form Controls */
        QSpinBox, QDoubleSpinBox {
            background-color: #000000;
            border: 1px solid #242424;
            padding: 4px 8px;
            border-radius: 4px;
            color: #FFFFFF;
            font-family: 'Consolas', monospace;
            font-size: 12px;
        }
        QSpinBox:focus, QDoubleSpinBox:focus {
            border: 1px solid #3B82F6;
        }
        QCheckBox {
            color: #D4D4D8;
            font-size: 12px;
            spacing: 8px;
            background: transparent;
        }
        QCheckBox::indicator {
            width: 15px;
            height: 15px;
            border: 1px solid #2E2E2E;
            border-radius: 3px;
            background-color: #000000;
        }
        QCheckBox::indicator:hover {
            border-color: #3B82F6;
        }
        QCheckBox::indicator:checked {
            background-color: #2563EB;
            border-color: #3B82F6;
        }
        """
        self.setStyleSheet(style)

    def toggle_params_panel(self):
        visible = not self.params_content_widget.isVisible()
        self.params_content_widget.setVisible(visible)
        self.btn_toggle_params.setText("▾ Detection Parameters" if visible else "▸ Detection Parameters")

    def toggle_monitoring(self):
        if self.worker is None or not self.worker.isRunning():
            # Start Worker
            cam_id = self.spin_cam_id.value()
            threshold = self.spin_threshold.value()
            grace = self.spin_grace.value()

            self.worker = VideoWorker(
                model_path="models/cnn_baseline_best.keras",
                camera_id=cam_id,
                alert_threshold=threshold,
                open_grace=grace,
                unknown_timeout=0.5,
                window_size=5,
                decision_threshold=0.5
            )
            self.worker.audio_enabled = self.chk_audio.isChecked()
            self.worker.frame_ready.connect(self.on_frame_ready)
            self.worker.status_updated.connect(self.on_status_updated)
            self.worker.error_occurred.connect(self.on_worker_error)
            self.worker.start()

            self.btn_toggle_monitor.setText("Stop Monitoring")
            self.btn_toggle_monitor.setObjectName("btnPrimaryActive")
            self.btn_toggle_monitor.style().unpolish(self.btn_toggle_monitor)
            self.btn_toggle_monitor.style().polish(self.btn_toggle_monitor)

            self.cam_indicator.setText(f"● Cam {cam_id} Online")
            self.cam_indicator.setObjectName("camActive")
            self.cam_indicator.style().unpolish(self.cam_indicator)
            self.cam_indicator.style().polish(self.cam_indicator)

            self.sys_status_pill.setText("MONITORING ACTIVE")
            self.sys_status_pill.setObjectName("statusPillActive")
            self.sys_status_pill.style().unpolish(self.sys_status_pill)
            self.sys_status_pill.style().polish(self.sys_status_pill)

            self.state_desc_label.setText("Acquiring video stream and ocular mesh...")
        else:
            # Stop Worker
            self.stop_worker()

    def stop_worker(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(1500)
            self.worker = None

        self.btn_toggle_monitor.setText("Start Monitoring")
        self.btn_toggle_monitor.setObjectName("btnPrimary")
        self.btn_toggle_monitor.style().unpolish(self.btn_toggle_monitor)
        self.btn_toggle_monitor.style().polish(self.btn_toggle_monitor)

        self.cam_indicator.setText("○ Camera Standby")
        self.cam_indicator.setObjectName("camStandby")
        self.cam_indicator.style().unpolish(self.cam_indicator)
        self.cam_indicator.style().polish(self.cam_indicator)

        self.sys_status_pill.setText("SYSTEM IDLE")
        self.sys_status_pill.setObjectName("statusPillIdle")
        self.sys_status_pill.style().unpolish(self.sys_status_pill)
        self.sys_status_pill.style().polish(self.sys_status_pill)

        self.preview_label.setText("CAMERA STREAM OFFLINE\nClick 'Start Monitoring' to initiate ocular tracking")
        self.alert_banner.setVisible(False)

        self.state_label.setText("UNKNOWN")
        self.state_label.setObjectName("stateUnknown")
        self.state_label.style().unpolish(self.state_label)
        self.state_label.style().polish(self.state_label)

        self.state_desc_label.setText("System on standby • Camera inactive")
        self._current_progress_mode = "idle"
        self.closure_progress.setStyleSheet("")
        self.closure_progress.setValue(0)
        self.closure_time_label.setText("0.00s / 1.50s")
        self.fps_badge.setText("FPS: 0.0")
        self.face_badge.setText("Face: None")
        self.ear_badge.setText("EAR: --")

    def on_audio_toggled(self, checked):
        if self.worker is not None:
            self.worker.audio_enabled = checked

    def apply_settings(self):
        cam_id = self.spin_cam_id.value()
        threshold = self.spin_threshold.value()
        grace = self.spin_grace.value()
        self.closure_progress.setRange(0, int(threshold * 1000))
        self.closure_time_label.setText(f"0.00s / {threshold:.2f}s")
        if self.worker is not None:
            self.worker.update_settings(cam_id, threshold, grace, 0.5, 5)

    def toggle_diag_panel(self):
        visible = not self.diag_content_widget.isVisible()
        self.diag_content_widget.setVisible(visible)
        self.btn_toggle_diag.setText("▾ Advanced Telemetry && Diagnostics" if visible else "▸ Advanced Telemetry && Diagnostics")

    def on_save_crop_clicked(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_crop_save()
            self.lbl_crop_status.setText("✓ Diagnostic crop queued for next detection")
            QTimer.singleShot(4000, lambda: self.lbl_crop_status.setText(""))
        else:
            self.lbl_crop_status.setText("⚠ Start monitoring first to capture eye crop")
            QTimer.singleShot(4000, lambda: self.lbl_crop_status.setText(""))

    @Slot(QImage)
    def on_frame_ready(self, q_img):
        # Scale to fit label viewport smoothly
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled_pixmap)

    @Slot(dict)
    def on_status_updated(self, status):
        state = status["state"]
        is_alert = status["is_alert"]
        closure_sec = status["closure_duration"]
        threshold_sec = status["alert_threshold"]
        fps = status["fps"]
        face_detected = status["face_detected"]
        eyes_count = status["eyes_detected_count"]
        ambient = status["ambient_light"]

        # Update Ocular State Label and Subtext Description
        if state == STATE_OPEN:
            self.state_label.setText("OPEN")
            self.state_label.setObjectName("stateOpen")
            self.state_desc_label.setText("Driver alert & attentive • Active facial mesh tracking")
        elif state == STATE_CLOSED:
            self.state_label.setText("CLOSED")
            self.state_label.setObjectName("stateClosed")
            self.state_desc_label.setText("Ocular closure detected • Accumulating duration")
        else:
            self.state_label.setText("UNKNOWN")
            self.state_label.setObjectName("stateUnknown")
            if not face_detected:
                self.state_desc_label.setText("Face unlocalized • Driver not in camera frame")
            else:
                self.state_desc_label.setText("Landmarks unavailable • Check camera lighting")

        self.state_label.style().unpolish(self.state_label)
        self.state_label.style().polish(self.state_label)

        # Update Progress Bar & Monospace Duration
        val_ms = int(closure_sec * 1000)
        max_ms = int(threshold_sec * 1000)
        self.closure_progress.setRange(0, max_ms)
        self.closure_progress.setValue(min(val_ms, max_ms))
        self.closure_time_label.setText(f"{closure_sec:.2f}s / {threshold_sec:.2f}s")

        # Dynamic progress chunk color: Amber for accumulation, Vivid Red on alert
        target_mode = "alert" if is_alert else ("accumulating" if closure_sec > 0.0 else "idle")
        if target_mode != self._current_progress_mode:
            self._current_progress_mode = target_mode
            if target_mode == "alert":
                self.closure_progress.setStyleSheet("#closureProgress::chunk { background-color: #ef4444; border-radius: 2px; }")
            elif target_mode == "accumulating":
                self.closure_progress.setStyleSheet("#closureProgress::chunk { background-color: #f59e0b; border-radius: 2px; }")
            else:
                self.closure_progress.setStyleSheet("")

        # Alert Banner & System Status Pill
        self.alert_banner.setVisible(is_alert)
        if is_alert:
            self.sys_status_pill.setText("DROWSINESS ALERT")
            self.sys_status_pill.setObjectName("statusPillAlert")
            self.closure_time_label.setStyleSheet("color: #ef4444; font-weight: 700;")
        else:
            self.sys_status_pill.setText("MONITORING ACTIVE")
            self.sys_status_pill.setObjectName("statusPillActive")
            self.closure_time_label.setStyleSheet("")
            
        self.sys_status_pill.style().unpolish(self.sys_status_pill)
        self.sys_status_pill.style().polish(self.sys_status_pill)

        # HUD Badges
        self.fps_badge.setText(f"FPS: {fps:.1f}")
        self.face_badge.setText(f"Face: {'Mesh Locked' if face_detected else 'Searching...'}")

        ear_val = status.get("ear")
        blink_val = status.get("blink_score")
        if ear_val is not None:
            self.ear_badge.setText(f"EAR: {ear_val:.2f}")
        else:
            self.ear_badge.setText("EAR: --")

        if ambient < 40:
            self.light_badge.setText("Lighting: Dim (Low-Light Alert)")
            self.light_badge.setStyleSheet("color: #f59e0b; background-color: #1a1303; border: 1px solid #78350f;")
        else:
            self.light_badge.setText("Lighting: Adequate")
            self.light_badge.setStyleSheet("")

        # Advanced Diagnostics Panel
        raw_p = status["raw_p_awake"]
        smooth_p = status["smoothed_p_awake"]
        p_r = status.get("p_right")
        p_l = status.get("p_left")

        self.lbl_raw_p.setText(f"Raw P(Awake): {raw_p:.3f} (R:{p_r:.2f}, L:{p_l:.2f})" if (raw_p is not None and p_r is not None and p_l is not None) else (f"Raw P(Awake): {raw_p:.3f}" if raw_p is not None else "Raw P(Awake): N/A"))
        self.lbl_smooth_p.setText(f"Smoothed P(Awake): {smooth_p:.3f}" if smooth_p is not None else "Smoothed P(Awake): N/A")
        
        if ear_val is not None and blink_val is not None:
            self.lbl_ear_blink.setText(f"Landmarks: EAR={ear_val:.2f} | BlinkScore={blink_val:.2f}")
        else:
            self.lbl_ear_blink.setText("Landmarks: EAR=N/A | BlinkScore=N/A")

        self.lbl_ambient.setText(f"Mean Luminance: {ambient:.1f} / 255.0")
        self.lbl_det_status.setText(f"Detector Status: Face={'Detected' if face_detected else 'None'}, Eyes={eyes_count}")

    @Slot(str)
    def on_worker_error(self, err_msg):
        self.stop_worker()
        QMessageBox.critical(self, "Camera / Detection Error", err_msg)

    def closeEvent(self, event):
        self.stop_worker()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
