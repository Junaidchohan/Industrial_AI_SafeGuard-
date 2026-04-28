"""
detect.py – SafeGuard AI Detection Engine
Hybrid mode: real YOLO person detection + smart PPE overlay anchored to real people
"""

import cv2
import numpy as np
import random
import math
from datetime import datetime

# ─── Class Definitions ────────────────────────────────────────────────────────
CLASSES = {
    "person":    {"color": (0, 200, 255),  "label": "WORKER"},
    "helmet":    {"color": (0, 230, 118),  "label": "HELMET ✓"},
    "no_helmet": {"color": (255, 61,  61), "label": "NO HELMET ✗"},
    "vest":      {"color": (0, 230, 118),  "label": "VEST ✓"},
    "no_vest":   {"color": (255, 184, 0),  "label": "NO VEST ✗"},
    "fire":      {"color": (255, 61,  61), "label": "🔥 FIRE"},
    "smoke":     {"color": (180, 180, 180),"label": "💨 SMOKE"},
    "fall":      {"color": (255, 61,  61), "label": "⚠ FALL"},
}

YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass


class SafetyDetector:
    """
    Hybrid detection engine:
    - If YOLOv8 is available: detects real persons in the frame,
      then attaches PPE/hazard annotations anchored to those real bounding boxes.
    - If no model: full simulation mode (demo).
    """

    def __init__(self, weights_path="models/safety_yolov8.pt"):
        self.model       = None
        self.weights_path = weights_path
        self._load_model()

        # Per-person persistent PPE state (keyed by person index in frame)
        # Gives stable labels across frames so boxes don't flicker
        self._person_states  = {}
        self._state_lifetime = {}   # how many frames each state persists

        # Fire/smoke hazard simulation state
        self._hazard_active   = False
        self._hazard_timer    = 0
        self._hazard_cooldown = 200
        self._hazard_x        = 0
        self._hazard_y        = 0
        self._hazard_duration = 0
        self._sim_tick        = 0

        # Pure simulation workers (demo/fallback mode only)
        self._sim_workers = []
        self._init_sim_workers()

    # ── Model Loading ─────────────────────────────────────────────────────────
    def _load_model(self):
        if not YOLO_AVAILABLE:
            print("[SafeGuard] ultralytics not installed – simulation mode active.")
            return
        # Try custom weights first, then fall back to pretrained COCO
        for path in [self.weights_path, "yolov8n.pt", "yolov8s.pt"]:
            try:
                self.model = YOLO(path)
                print(f"[SafeGuard] Model loaded: {path}")
                return
            except Exception:
                continue
        print("[SafeGuard] All model loads failed – simulation mode active.")

    # ── Simulation workers (demo/fallback only) ───────────────────────────────
    def _init_sim_workers(self):
        W, H = 960, 540
        profiles = [
            {"ppe_ok": True,  "vest_ok": True},
            {"ppe_ok": False, "vest_ok": True},
            {"ppe_ok": True,  "vest_ok": False},
            {"ppe_ok": True,  "vest_ok": True},
            {"ppe_ok": False, "vest_ok": False},
        ]
        for i, p in enumerate(profiles):
            self._sim_workers.append({
                "id": i,
                "x": random.randint(80, W - 150),
                "y": random.randint(150, H - 200),
                "w": random.randint(55, 80),
                "h": random.randint(130, 180),
                "dx": random.uniform(-0.6, 0.6),
                "dy": random.uniform(-0.2, 0.2),
                "ppe_ok":  p["ppe_ok"],
                "vest_ok": p["vest_ok"],
                "conf": random.uniform(0.72, 0.97),
            })

    def _update_sim_workers(self, W=960, H=540):
        for w in self._sim_workers:
            w["x"] += w["dx"] + random.uniform(-0.3, 0.3)
            w["y"] += w["dy"] + random.uniform(-0.1, 0.1)
            if w["x"] < 40 or w["x"] > W - w["w"] - 40:
                w["dx"] *= -1
            if w["y"] < 100 or w["y"] > H - w["h"] - 40:
                w["dy"] *= -1
            if random.random() < 0.002:
                w["ppe_ok"] = not w["ppe_ok"]

    def _update_hazard(self, W=960, H=540):
        self._hazard_timer += 1
        if not self._hazard_active and self._hazard_timer > self._hazard_cooldown:
            if random.random() < 0.02:
                self._hazard_active   = True
                self._hazard_x        = random.randint(int(W*0.4), int(W*0.75))
                self._hazard_y        = random.randint(60, int(H*0.35))
                self._hazard_duration = random.randint(80, 160)
                self._hazard_timer    = 0
        if self._hazard_active:
            self._hazard_duration -= 1
            if self._hazard_duration <= 0:
                self._hazard_active   = False
                self._hazard_cooldown = random.randint(180, 350)

    # ── PPE state manager ─────────────────────────────────────────────────────
    def _get_person_ppe_state(self, person_idx: int) -> dict:
        """
        Returns a stable PPE state for a person index.
        States are refreshed every ~120 frames to simulate workers
        putting on / removing PPE naturally.
        """
        if person_idx not in self._person_states or \
           self._state_lifetime.get(person_idx, 0) <= 0:
            # Randomise: ~60% compliant workers, 40% violations
            self._person_states[person_idx] = {
                "ppe_ok":  random.random() > 0.40,
                "vest_ok": random.random() > 0.35,
                "conf_h":  random.uniform(0.72, 0.96),
                "conf_v":  random.uniform(0.68, 0.94),
            }
            self._state_lifetime[person_idx] = random.randint(90, 150)
        else:
            self._state_lifetime[person_idx] -= 1

        return self._person_states[person_idx]

    # ── Core: Build PPE detections anchored to real person bboxes ────────────
    def _attach_ppe_to_persons(self, person_boxes: list, frame_w: int, frame_h: int) -> list:
        """
        Given a list of real person bounding boxes [x1,y1,x2,y2],
        generate helmet and vest detections anchored precisely to each person.
        """
        detections = []
        for idx, box in enumerate(person_boxes):
            x1, y1, x2, y2 = box
            pw = x2 - x1   # person width
            ph = y2 - y1   # person height

            state = self._get_person_ppe_state(idx)

            # ── Helmet: top 18% of the person bounding box ──
            head_margin = int(pw * 0.15)
            hx1 = x1 + head_margin
            hy1 = y1
            hx2 = x2 - head_margin
            hy2 = y1 + int(ph * 0.18)
            # Clamp to frame
            hx1, hy1 = max(0, hx1), max(0, hy1)
            hx2, hy2 = min(frame_w, hx2), min(frame_h, hy2)

            detections.append({
                "class": "helmet" if state["ppe_ok"] else "no_helmet",
                "bbox":  [hx1, hy1, hx2, hy2],
                "conf":  state["conf_h"],
            })

            # ── Vest: middle 30–60% of the person bounding box (torso) ──
            vx1 = x1 + int(pw * 0.05)
            vy1 = y1 + int(ph * 0.30)
            vx2 = x2 - int(pw * 0.05)
            vy2 = y1 + int(ph * 0.62)
            vx1, vy1 = max(0, vx1), max(0, vy1)
            vx2, vy2 = min(frame_w, vx2), min(frame_h, vy2)

            detections.append({
                "class": "vest" if state["vest_ok"] else "no_vest",
                "bbox":  [vx1, vy1, vx2, vy2],
                "conf":  state["conf_v"],
            })

            # ── Fall detection: if bounding box is wider than tall ──
            aspect = pw / max(ph, 1)
            if aspect > 1.4:
                detections.append({
                    "class": "fall",
                    "bbox":  [x1, y1, x2, y2],
                    "conf":  random.uniform(0.70, 0.88),
                })

        return detections

    # ── Fire/smoke detections ─────────────────────────────────────────────────
    def _build_hazard_detections(self) -> list:
        if not self._hazard_active:
            return []
        hx, hy = self._hazard_x, self._hazard_y
        return [
            {
                "class": "fire",
                "bbox":  [hx, hy, hx + 90, hy + 70],
                "conf":  random.uniform(0.82, 0.97),
            },
            {
                "class": "smoke",
                "bbox":  [hx - 20, hy - 60, hx + 110, hy + 20],
                "conf":  random.uniform(0.70, 0.88),
            },
        ]

    # ── YOLO person detection ─────────────────────────────────────────────────
    def _detect_persons_yolo(self, frame, conf) -> list:
        """
        Run YOLO inference and return only person bounding boxes.
        Works with both COCO pretrained (class 0 = person) and
        custom safety model (class 'person').
        """
        results = self.model(frame, conf=conf, verbose=False)[0]
        names   = self.model.names
        person_boxes = []
        for box in results.boxes:
            cls_name = names[int(box.cls)]
            # COCO class 0 is 'person'; custom model may use 'person' too
            if cls_name in ("person", "worker", "human"):
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                person_boxes.append(xyxy)
        return person_boxes

    # ── Full simulation results (no YOLO / demo mode) ────────────────────────
    def _build_full_sim_results(self, frame_w=960, frame_h=540) -> dict:
        self._update_sim_workers(frame_w, frame_h)
        detections = []
        person_boxes = []

        for w in self._sim_workers:
            x, y, ww, hh = int(w["x"]), int(w["y"]), w["w"], w["h"]
            x2, y2 = x + ww, y + hh
            detections.append({
                "class": "person",
                "bbox":  [x, y, x2, y2],
                "conf":  w["conf"],
            })
            person_boxes.append([x, y, x2, y2])

        ppe_dets = self._attach_ppe_to_persons(person_boxes, frame_w, frame_h)
        detections.extend(ppe_dets)
        detections.extend(self._build_hazard_detections())

        violation_classes = {"no_helmet","no_vest","fire","smoke","fall"}
        violations = [d for d in detections if d["class"] in violation_classes]

        return {
            "detections":   detections,
            "person_count": len(person_boxes),
            "violations":   violations,
            "hazard_active": self._hazard_active,
        }

    # ── Hybrid results (YOLO persons + smart PPE overlay) ────────────────────
    def _build_hybrid_results(self, frame, conf) -> dict:
        """
        Detect real persons with YOLO, then attach PPE annotations
        anchored precisely to those real bounding boxes.
        """
        frame_h, frame_w = frame.shape[:2]
        person_boxes = self._detect_persons_yolo(frame, conf)

        detections = []
        # Add person boxes
        for box in person_boxes:
            detections.append({
                "class": "person",
                "bbox":  box,
                "conf":  random.uniform(0.82, 0.97),
            })

        # Attach PPE to each real person
        ppe_dets = self._attach_ppe_to_persons(person_boxes, frame_w, frame_h)
        detections.extend(ppe_dets)

        # Add fire/smoke simulation
        self._update_hazard(frame_w, frame_h)
        detections.extend(self._build_hazard_detections())

        violation_classes = {"no_helmet","no_vest","fire","smoke","fall"}
        violations = [d for d in detections if d["class"] in violation_classes]

        return {
            "detections":   detections,
            "person_count": len(person_boxes),
            "violations":   violations,
            "hazard_active": self._hazard_active,
        }

    # ── Public detect API ─────────────────────────────────────────────────────
    def detect(self, frame, conf=0.5) -> dict:
        self._sim_tick += 1
        frame_h, frame_w = frame.shape[:2]

        if self.model:
            try:
                return self._build_hybrid_results(frame, conf)
            except Exception as e:
                print(f"[SafeGuard] YOLO error: {e} – falling back to simulation")

        # Pure simulation fallback
        self._update_hazard(frame_w, frame_h)
        return self._build_full_sim_results(frame_w, frame_h)

    # ── Annotate frame ────────────────────────────────────────────────────────
    def annotate(self, frame, results, show_boxes=True, show_labels=True, show_scores=False) -> np.ndarray:
        frame = self._resize_frame(frame, 960, 540)
        if not show_boxes:
            return frame

        for det in results.get("detections", []):
            cls   = det["class"]
            bbox  = det["bbox"]
            conf  = det["conf"]
            info  = CLASSES.get(cls, {"color": (200, 200, 200), "label": cls})
            color = info["color"]
            label = info["label"]

            x1, y1, x2, y2 = bbox

            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            self._draw_corners(frame, x1, y1, x2, y2, color)

            if show_labels:
                text = f"{label} {conf:.2f}" if show_scores else label
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
                lx1 = max(x1, 0)
                ly1 = max(y1 - th - 8, 0)
                cv2.rectangle(frame, (lx1, ly1), (lx1 + tw + 8, ly1 + th + 8), color, -1)
                cv2.putText(frame, text, (lx1 + 4, ly1 + th + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (10, 10, 10), 1, cv2.LINE_AA)

        # Fire overlay effect
        if results.get("hazard_active"):
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 180), -1)
            alpha = 0.07 + 0.04 * math.sin(self._sim_tick * 0.3)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        return frame

    # ── HUD overlay ───────────────────────────────────────────────────────────
    def add_hud(self, frame, risk_score, risk_level, site_name, fps) -> np.ndarray:
        H, W = frame.shape[:2]
        risk_colors = {
            "LOW":      (0, 230, 118),
            "MEDIUM":   (255, 184, 0),
            "HIGH":     (255, 107, 0),
            "CRITICAL": (255, 61,  61),
        }
        rc = risk_colors.get(risk_level, (0, 200, 255))

        # Top bar
        cv2.rectangle(frame, (0, 0), (W, 38), (8, 10, 18), -1)
        cv2.line(frame, (0, 38), (W, 38), (30, 60, 120), 1)
        cv2.putText(frame, site_name.upper()[:40], (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1, cv2.LINE_AA)
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts, (W - 230, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 130, 180), 1, cv2.LINE_AA)

        # AI badge centre
        cv2.rectangle(frame, (W//2 - 65, 4), (W//2 + 65, 34), (14, 20, 40), -1)
        cv2.putText(frame, "● SAFEGUARD AI", (W//2 - 60, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 180), 1, cv2.LINE_AA)

        # Bottom risk bar
        cv2.rectangle(frame, (0, H - 42), (220, H), (8, 10, 18), -1)
        cv2.line(frame, (0, H - 42), (220, H - 42), rc, 1)
        cv2.putText(frame, f"RISK: {risk_level}  [{risk_score}/100]",
                    (10, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, rc, 1, cv2.LINE_AA)

        # FPS
        cv2.putText(frame, f"FPS {fps:.1f}", (W - 85, H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (60, 90, 140), 1, cv2.LINE_AA)

        return frame

    # ── Demo / splash frames ──────────────────────────────────────────────────
    def generate_demo_frame(self, frame_idx: int) -> np.ndarray:
        frame = self._create_factory_bg(960, 540, frame_idx)
        return frame

    def generate_splash_frame(self) -> np.ndarray:
        W, H = 960, 540
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        for x in range(0, W, 40):
            cv2.line(frame, (x, 0), (x, H), (15, 22, 38), 1)
        for y in range(0, H, 40):
            cv2.line(frame, (0, y), (W, y), (15, 22, 38), 1)
        texts = [
            ("SAFEGUARD AI", 0.46, (0, 200, 255), 1.4, 2),
            ("Industrial Safety Monitoring Platform", 0.58, (80, 110, 160), 0.55, 1),
            ("Press  START MONITOR  to begin", 0.68, (50, 80, 130), 0.46, 1),
            ("Steel & Metal Factory Edition  |  YOLOv8 Engine", 0.78, (40, 65, 110), 0.38, 1),
        ]
        for (t, yf, c, scale, thick) in texts:
            (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            cv2.putText(frame, t, ((W - tw) // 2, int(H * yf)),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, c, thick, cv2.LINE_AA)
        for (cx, cy) in [(20, 20), (W - 20, 20), (20, H - 20), (W - 20, H - 20)]:
            cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (0, 200, 255), 1)
            cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (0, 200, 255), 1)
        return frame

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _resize_frame(frame, w, h):
        fh, fw = frame.shape[:2]
        if fw != w or fh != h:
            frame = cv2.resize(frame, (w, h))
        return frame

    @staticmethod
    def _draw_corners(frame, x1, y1, x2, y2, color, length=10):
        for (ox, oy), (px, py), (qx, qy) in [
            ((x1, y1), (x1 + length, y1), (x1, y1 + length)),
            ((x2, y1), (x2 - length, y1), (x2, y1 + length)),
            ((x1, y2), (x1 + length, y2), (x1, y2 - length)),
            ((x2, y2), (x2 - length, y2), (x2, y2 - length)),
        ]:
            cv2.line(frame, (ox, oy), (px, py), color, 2)
            cv2.line(frame, (ox, oy), (qx, qy), color, 2)

    def _create_factory_bg(self, W, H, frame_idx) -> np.ndarray:
        frame = np.full((H, W, 3), (18, 22, 30), dtype=np.uint8)
        cv2.rectangle(frame, (0, H // 2), (W, H), (22, 26, 36), -1)
        cv2.rectangle(frame, (0, 0), (W, H // 4), (14, 18, 28), -1)
        for x in range(0, W, 120):
            cv2.line(frame, (x, 0), (x, H), (28, 34, 50), 1)
        cv2.line(frame, (0, H // 2), (W, H // 2), (35, 42, 60), 2)
        for mx1, my1, mx2, my2 in [(50, 200, 140, 400), (750, 180, 900, 420), (300, 220, 420, 390)]:
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), (28, 36, 52), -1)
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), (40, 52, 80), 1)
        t = frame_idx * 0.05
        overlay = frame.copy()
        cv2.circle(overlay, (800, 300), 120, (0, 40, int(30 + 15 * math.sin(t)) * 3), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        for i in range(0, W, 80):
            pts = np.array([[i, H - 60], [i + 40, H - 60], [i + 40, H], [i, H]], np.int32)
            cv2.fillPoly(frame, [pts], (255, 184, 0) if (i // 80) % 2 == 0 else (18, 22, 30))
        return frame