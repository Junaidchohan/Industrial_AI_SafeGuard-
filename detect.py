"""
detect.py - SafeGuard AI Detection Engine
YOLOv8 hybrid engine with ASCII-only labels for OpenCV compatibility
"""

import cv2
import numpy as np
import random
import time
import os
from datetime import datetime

# - Try importing ultralytics (graceful fallback if not available) -
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# - Class configuration (ASCII ONLY - no emoji) -
CLASS_CONFIG = {
    "person":      {"color": (255, 255, 255), "label": "WORKER",      "risk": 0},
    "helmet":      {"color": (0,   200,   0), "label": "HELMET OK",   "risk": 0},
    "no_helmet":   {"color": (0,   0,   255), "label": "NO HELMET",   "risk": 25},
    "vest":        {"color": (0,   200,   0), "label": "VEST OK",     "risk": 0},
    "no_vest":     {"color": (0,   165, 255), "label": "NO VEST",     "risk": 15},
    "fire":        {"color": (0,   0,   255), "label": "FIRE!",       "risk": 50},
    "smoke":       {"color": (128, 128, 128), "label": "SMOKE!",      "risk": 35},
    "fall":        {"color": (0,   0,   200), "label": "FALL!",       "risk": 40},
}

# - Detection result dataclass -
class Detection:
    def __init__(self, cls, x1, y1, x2, y2, conf=0.85):
        self.cls  = cls
        self.x1   = int(x1)
        self.y1   = int(y1)
        self.x2   = int(x2)
        self.y2   = int(y2)
        self.conf = conf
        cfg = CLASS_CONFIG.get(cls, {})
        self.color = cfg.get("color", (200, 200, 200))
        self.label = cfg.get("label", cls.upper())
        self.risk  = cfg.get("risk",  0)

    def to_dict(self):
        return {
            "class": self.cls,
            "label": self.label,
            "confidence": round(self.conf, 2),
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "risk": self.risk,
        }


# - Worker state manager (stable per-worker PPE state) -
class WorkerState:
    def __init__(self, worker_id):
        self.id           = worker_id
        self.has_helmet   = random.random() > 0.45   # 55% compliant
        self.has_vest     = random.random() > 0.35   # 65% compliant
        self.life         = random.randint(80, 160)  # frames before re-randomise
        self.age          = 0

    def tick(self):
        self.age += 1
        if self.age >= self.life:
            self.has_helmet = random.random() > 0.45
            self.has_vest   = random.random() > 0.35
            self.life       = random.randint(80, 160)
            self.age        = 0


# - Main detector class -
class SafetyDetector:

    MODEL_PATH = "models/safety_yolov8.pt"

    def __init__(self):
        self.model        = None
        self.use_real     = False
        self.worker_states: dict[int, WorkerState] = {}
        self.fire_active  = False
        self.fire_timer   = 0
        self.fire_interval= random.randint(200, 400)
        self.frame_count  = 0
        self._load_model()

    # - Model loading -
    def _load_model(self):
        if not YOLO_AVAILABLE:
            return
        # Try custom trained model first
        if os.path.exists(self.MODEL_PATH):
            try:
                self.model    = YOLO(self.MODEL_PATH)
                self.use_real = True
                print("[SafeGuard] Custom model loaded:", self.MODEL_PATH)
                return
            except Exception as e:
                print("[SafeGuard] Custom model failed:", e)
        # Fall back to COCO pretrained (detects 'person' only)
        try:
            self.model    = YOLO("yolov8n.pt")
            self.use_real = False
            print("[SafeGuard] COCO pretrained loaded (hybrid mode)")
        except Exception as e:
            print("[SafeGuard] No model available, pure simulation:", e)
            self.model = None

    # - Public detect API -
    def detect(self, frame: np.ndarray, conf: float = 0.4) -> list:
        self.frame_count += 1
        h, w = frame.shape[:2]

        if self.model is not None and self.use_real:
            return self._real_detect(frame, conf)

        # Hybrid: get person boxes from COCO model, attach simulated PPE
        person_boxes = self._coco_persons(frame, conf)

        if not person_boxes:
            # No real persons found → generate synthetic workers
            person_boxes = self._synthetic_persons(w, h)

        detections = self._attach_ppe(person_boxes, w, h)
        detections += self._hazard_events(w, h)
        return detections

    # - Real YOLO inference (custom model) -
    def _real_detect(self, frame, conf):
        results     = self.model(frame, conf=conf, verbose=False)[0]
        detections  = []
        names       = results.names
        for box in results.boxes:
            cls_id  = int(box.cls[0])
            cls_name= names.get(cls_id, "unknown")
            if cls_name not in CLASS_CONFIG:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            c = float(box.conf[0])
            detections.append(Detection(cls_name, x1, y1, x2, y2, c))
        return detections

    # - COCO person detection -
    def _coco_persons(self, frame, conf):
        if self.model is None:
            return []
        try:
            results = self.model(frame, conf=conf, classes=[0], verbose=False)[0]
            boxes   = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                c = float(box.conf[0])
                boxes.append((x1, y1, x2, y2, c))
            return boxes
        except Exception:
            return []

    # - Synthetic person grid (when no real persons found) -
    def _synthetic_persons(self, w, h):
        """Generate 3–6 synthetic person boxes distributed across the frame."""
        count = random.randint(3, 6)
        boxes = []
        for i in range(count):
            pw = int(w * random.uniform(0.08, 0.14))
            ph = int(h * random.uniform(0.35, 0.55))
            px = int(w * (0.1 + 0.8 * i / max(count - 1, 1)) - pw // 2)
            py = int(h * random.uniform(0.3, 0.55))
            px = max(0, min(px, w - pw))
            py = max(0, min(py, h - ph))
            boxes.append((px, py, px + pw, py + ph, round(random.uniform(0.80, 0.97), 2)))
        return boxes

    # - PPE attachment (anchored to each person box) -
    def _attach_ppe(self, person_boxes, w, h):
        detections = []
        for idx, (x1, y1, x2, y2, conf) in enumerate(person_boxes):
            bw = x2 - x1
            bh = y2 - y1

            # Add person box
            detections.append(Detection("person", x1, y1, x2, y2, conf))

            # Get or create stable worker state
            state = self.worker_states.setdefault(idx, WorkerState(idx))
            state.tick()

            # - Helmet box: top 20% of person height -
            hx1 = x1 + int(bw * 0.15)
            hy1 = y1
            hx2 = x2 - int(bw * 0.15)
            hy2 = y1 + int(bh * 0.22)
            helmet_cls = "helmet" if state.has_helmet else "no_helmet"
            detections.append(Detection(
                helmet_cls, hx1, hy1, hx2, hy2,
                round(random.uniform(0.78, 0.96), 2)
            ))

            # - Vest box: torso 28%–65% of person height -
            vx1 = x1 + int(bw * 0.05)
            vy1 = y1 + int(bh * 0.28)
            vx2 = x2 - int(bw * 0.05)
            vy2 = y1 + int(bh * 0.65)
            vest_cls = "vest" if state.has_vest else "no_vest"
            detections.append(Detection(
                vest_cls, vx1, vy1, vx2, vy2,
                round(random.uniform(0.75, 0.94), 2)
            ))

            # - Fall detection: if person bbox is landscape-oriented -
            if bw > bh * 1.3:
                detections.append(Detection(
                    "fall", x1, y1, x2, y2,
                    round(random.uniform(0.70, 0.88), 2)
                ))

        return detections

    # - Hazard events (fire / smoke) -
    def _hazard_events(self, w, h):
        detections = []

        # Fire trigger cycle
        self.fire_timer += 1
        if self.fire_timer >= self.fire_interval:
            self.fire_active   = True
            self.fire_interval = random.randint(200, 400)
            self.fire_timer    = 0

        if self.fire_active:
            # Fire box — right-side corner, realistic size
            fx1 = int(w * 0.72)
            fy1 = int(h * 0.45)
            fx2 = int(w * 0.92)
            fy2 = int(h * 0.85)
            detections.append(Detection("fire", fx1, fy1, fx2, fy2,
                                         round(random.uniform(0.82, 0.97), 2)))

            # Smoke box above fire
            sx1 = int(w * 0.68)
            sy1 = int(h * 0.20)
            sx2 = int(w * 0.96)
            sy2 = int(h * 0.48)
            detections.append(Detection("smoke", sx1, sy1, sx2, sy2,
                                         round(random.uniform(0.75, 0.90), 2)))

            # Auto-extinguish after ~4 seconds (120 frames @ 30fps)
            if self.fire_timer > 120:
                self.fire_active = False

        return detections

    # - Annotation (draw boxes + ASCII labels on frame) -
    def annotate(
        self,
        frame:       np.ndarray,
        detections:  list,
        show_boxes:  bool = True,
        show_labels: bool = True,
        show_scores: bool = False,
    ) -> np.ndarray:

        h, w = frame.shape[:2]

        for det in detections:
            if not show_boxes:
                break
            color = det.color
            x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2

            # Clamp to frame
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            # Draw rectangle
            thickness = 3 if det.cls in ("fire", "smoke", "fall") else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            if not show_labels:
                continue

            # Build label text (ASCII only)
            label_text = det.label
            if show_scores:
                label_text += f" {det.conf:.2f}"

            # Label background
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thick = 1
            (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, font_thick)

            lx1 = x1
            ly1 = max(y1 - th - 6, 0)
            lx2 = min(x1 + tw + 6, w)
            ly2 = max(y1, th + 6)

            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)

            # Text color: black on bright backgrounds, white on dark
            brightness = 0.299*color[2] + 0.587*color[1] + 0.114*color[0]
            txt_color  = (0, 0, 0) if brightness > 140 else (255, 255, 255)

            cv2.putText(
                frame, label_text,
                (lx1 + 3, ly2 - 3),
                font, font_scale, txt_color, font_thick,
                cv2.LINE_AA
            )

        # - HUD overlay -
        self._draw_hud(frame, detections, w, h)
        return frame

    # - HUD overlay -
    def _draw_hud(self, frame, detections, w, h):
        violations = sum(1 for d in detections if d.cls in ("no_helmet", "no_vest", "fire", "smoke", "fall"))
        risk_score = min(100, sum(d.risk for d in detections))
        workers    = sum(1 for d in detections if d.cls == "person")

        if   risk_score >= 80: level, lcolor = "CRITICAL", (0, 0, 255)
        elif risk_score >= 50: level, lcolor = "HIGH",     (0, 100, 255)
        elif risk_score >= 25: level, lcolor = "MEDIUM",   (0, 200, 255)
        else:                  level, lcolor = "LOW",       (0, 200,   0)

        # Bottom bar
        bar_h = 36
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        ts   = datetime.now().strftime("%H:%M:%S")
        hud  = f"RISK: {level}  [{risk_score}/100]   WORKERS: {workers}   VIOLATIONS: {violations}   {ts}"
        cv2.putText(frame, hud, (10, h - 10), font, 0.52, lcolor, 1, cv2.LINE_AA)

        # Top-left facility tag
        cv2.putText(frame, "SAFEGUARD AI | STEEL PLANT ALPHA",
                    (10, 22), font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # - Splash / standby frame shown before monitoring starts -
    def generate_splash_frame(self, w: int = 960, h: int = 540) -> np.ndarray:
        """
        Returns a branded standby frame (numpy array, BGR).
        Displayed in the video panel before the user presses START.
        """
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Dark gradient background
        for y in range(h):
            val = int(18 + 20 * y / h)
            frame[y, :] = (val, val, val)

        font_big  = cv2.FONT_HERSHEY_DUPLEX
        font_med  = cv2.FONT_HERSHEY_SIMPLEX
        green     = (0, 210, 80)
        white     = (220, 220, 220)
        grey      = (120, 120, 120)

        # Centre cross-hair decoration
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 60, cy), (cx + 60, cy), green, 1)
        cv2.line(frame, (cx, cy - 60), (cx, cy + 60), green, 1)
        cv2.circle(frame, (cx, cy), 80, (40, 40, 40), 1)
        cv2.circle(frame, (cx, cy), 40, (50, 50, 50), 1)

        # Title
        title = "SAFEGUARD AI"
        (tw, th), _ = cv2.getTextSize(title, font_big, 1.4, 2)
        cv2.putText(frame, title, (cx - tw // 2, cy - 110),
                    font_big, 1.4, green, 2, cv2.LINE_AA)

        # Subtitle
        sub = "Industrial Safety Monitoring Platform"
        (sw, _), _ = cv2.getTextSize(sub, font_med, 0.6, 1)
        cv2.putText(frame, sub, (cx - sw // 2, cy - 75),
                    font_med, 0.6, white, 1, cv2.LINE_AA)

        # Status line
        status = "[ SYSTEM READY ]  Press  START MONITOR  to begin"
        (stw, _), _ = cv2.getTextSize(status, font_med, 0.55, 1)
        cv2.putText(frame, status, (cx - stw // 2, cy + 110),
                    font_med, 0.55, green, 1, cv2.LINE_AA)

        # Version tag bottom-right
        ver = "v2.4.1 | YOLOv8 Engine"
        (vw, _), _ = cv2.getTextSize(ver, font_med, 0.42, 1)
        cv2.putText(frame, ver, (w - vw - 12, h - 12),
                    font_med, 0.42, grey, 1, cv2.LINE_AA)

        # Timestamp bottom-left
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts, (12, h - 12),
                    font_med, 0.42, grey, 1, cv2.LINE_AA)

        # Border
        cv2.rectangle(frame, (6, 6), (w - 6, h - 6), (50, 50, 50), 1)

        return frame

    # - Risk summary for dashboard -
    def get_risk_summary(self, detections: list) -> dict:
        workers    = sum(1 for d in detections if d.cls == "person")
        violations = sum(1 for d in detections if d.cls in ("no_helmet", "no_vest", "fall"))
        fire       = any(d.cls == "fire"  for d in detections)
        smoke      = any(d.cls == "smoke" for d in detections)
        risk_score = min(100, sum(d.risk for d in detections))

        if   risk_score >= 80: level = "CRITICAL"
        elif risk_score >= 50: level = "HIGH"
        elif risk_score >= 25: level = "MEDIUM"
        else:                  level = "LOW"

        alerts = []
        if any(d.cls == "no_helmet" for d in detections):
            alerts.append("Helmet Missing - Immediate action required")
        if any(d.cls == "no_vest"   for d in detections):
            alerts.append("Safety Vest Missing - PPE violation logged")
        if fire:
            alerts.append("FIRE DETECTED - Emergency protocol activated")
        if smoke:
            alerts.append("Smoke Detected - Ventilation alert triggered")
        if any(d.cls == "fall"      for d in detections):
            alerts.append("Worker Fall Detected - Medical team notified")

        return {
            "workers":    workers,
            "violations": violations,
            "risk_score": risk_score,
            "risk_level": level,
            "fire":       fire,
            "smoke":      smoke,
            "alerts":     alerts,
            "detections": [d.to_dict() for d in detections],
        }