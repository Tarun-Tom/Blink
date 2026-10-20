import os
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

# Subject's Right Eye (Viewer's Left)
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
RIGHT_EYE_CORNERS = (33, 133)   # outer, inner
# Key landmarks for EAR: (p1, p4, p2, p6, p3, p5)
RIGHT_EAR_PTS = (33, 133, 160, 144, 158, 153)

# Subject's Left Eye (Viewer's Right)
LEFT_EYE_CONTOUR = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYE_CORNERS = (362, 263)   # inner, outer
LEFT_EAR_PTS = (362, 263, 385, 380, 387, 373)

class LandmarkEyeExtractor:
    """
    Extracts stable, aspect-ratio-preserved 64x64 square eye crops
    using MediaPipe FaceLandmarker. Reliable whether eyes are OPEN or CLOSED.
    Also computes landmark-based Eye Aspect Ratio (EAR) and blink blendshapes.
    """
    def __init__(self, model_asset_path: str = "models/face_landmarker.task", min_confidence: float = 0.5):
        task_path = Path(model_asset_path).resolve()
        if not task_path.exists():
            raise FileNotFoundError(f"MediaPipe face_landmarker model not found at {task_path}")

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(task_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_face_blendshapes=True
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def extract_eyes(self, bgr_frame: np.ndarray, margin_factor: float = 1.35):
        """
        Processes a BGR image frame and extracts left and right eye crops.
        
        Returns:
            dict containing:
                'face_detected': bool
                'right_eye': dict with 'crop_64', 'box', 'center', 'ear', 'blink_score' (or None)
                'left_eye': dict with 'crop_64', 'box', 'center', 'ear', 'blink_score' (or None)
                'landmarks': list of (x, y) coordinates
                'face_box': (x, y, w, h)
        """
        h, w = bgr_frame.shape[:2]
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.detector.detect(mp_image)
        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return {
                "face_detected": False,
                "right_eye": None,
                "left_eye": None,
                "landmarks": [],
                "face_box": None
            }

        face_landmarks = result.face_landmarks[0]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]

        # Extract blendshape blink scores if available
        blink_left = 0.0
        blink_right = 0.0
        if result.face_blendshapes and len(result.face_blendshapes) > 0:
            for cat in result.face_blendshapes[0]:
                if cat.category_name == "eyeBlinkLeft":
                    blink_left = float(cat.score)
                elif cat.category_name == "eyeBlinkRight":
                    blink_right = float(cat.score)

        # Compute Face Bounding Box
        all_x = [p[0] for p in pts]
        all_y = [p[1] for p in pts]
        fx1, fx2 = max(0, min(all_x)), min(w, max(all_x))
        fy1, fy2 = max(0, min(all_y)), min(h, max(all_y))
        face_box = (fx1, fy1, fx2 - fx1, fy2 - fy1)

        def _compute_ear(ear_indices):
            p1 = np.array(pts[ear_indices[0]], dtype=np.float32)
            p4 = np.array(pts[ear_indices[1]], dtype=np.float32)
            p2 = np.array(pts[ear_indices[2]], dtype=np.float32)
            p6 = np.array(pts[ear_indices[3]], dtype=np.float32)
            p3 = np.array(pts[ear_indices[4]], dtype=np.float32)
            p5 = np.array(pts[ear_indices[5]], dtype=np.float32)

            d_vert1 = np.linalg.norm(p2 - p6)
            d_vert2 = np.linalg.norm(p3 - p5)
            d_horiz = np.linalg.norm(p1 - p4)
            if d_horiz < 1e-5:
                return 0.0
            return float((d_vert1 + d_vert2) / (2.0 * d_horiz))

        def _get_eye_data(corners, contour_indices, ear_indices, blink_score):
            c1_idx, c2_idx = corners
            p1 = np.array(pts[c1_idx], dtype=np.float32)
            p2 = np.array(pts[c2_idx], dtype=np.float32)

            center = (p1 + p2) / 2.0
            eye_width = np.linalg.norm(p2 - p1)
            if eye_width < 8:
                return None

            box_size = int(eye_width * margin_factor)
            half = box_size // 2

            cx, cy = int(center[0]), int(center[1])
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w, cx + half)
            y2 = min(h, cy + half)

            crop = gray[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
                return None

            crop_64 = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
            ear = _compute_ear(ear_indices)

            return {
                "crop_64": crop_64,
                "box": (x1, y1, x2 - x1, y2 - y1),
                "center": (cx, cy),
                "ear": ear,
                "blink_score": blink_score,
                "contour_pts": [pts[i] for i in contour_indices]
            }

        right_data = _get_eye_data(RIGHT_EYE_CORNERS, RIGHT_EYE_CONTOUR, RIGHT_EAR_PTS, blink_right)
        left_data = _get_eye_data(LEFT_EYE_CORNERS, LEFT_EYE_CONTOUR, LEFT_EAR_PTS, blink_left)

        return {
            "face_detected": True,
            "right_eye": right_data,
            "left_eye": left_data,
            "landmarks": pts,
            "face_box": face_box
        }
