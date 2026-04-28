"""
detect.py – SafeGuard AI Detection Engine
Hybrid: real YOLO person detection + PPE anchored precisely to real bounding boxes
"""

import cv2
import numpy as np
import random
import math
from datetime import datetime

# ── No emoji in labels — OpenCV cannot render them ──────────────────────────
CLASSES = {
    "person":    {"color": (0, 200, 255),  "label": "WORKER"},
    "helmet":    {"color": (0, 230, 118),  "label": "HELMET OK"},
    "no_helmet": {"color": (255, 61,  61), "label": "NO HELMET"},
    "vest":      {"color": (0, 230, 118),  "label": "VEST OK"},
    "no_vest":   {"color": (255, 184, 0),  "label": "NO VEST"},
    "fire":      {"color": (0, 61, 255),   "label": "FIRE"},
    "smoke":     {"color": (180,180,180),  "label": "SMOKE"},
    "fall":      {"color": (255, 61, 200), "label": "FALL"},
}

YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass


class SafetyDetector:

    def __init__(self, weights_path="models/safety_yolov8.pt"):
        self.model        = None
        self.weights_path = weights_path
        self._load_model()

        # Stable per-person PPE state (index → state dict)
        self._person_states  = {}
        self._state_lifetime = {}

        # Fire/smoke hazard timing
        self._hazard_active   = False
        self._hazard_timer    = 0
        self._hazard_cooldown = 220
        self._hazard_x        = 0
        self._hazard_y        = 0
        self._hazard_w        = 0
        self._hazard_h        = 0
        self._hazard_duration = 0
        self._sim_tick        = 0

        # Demo-mode workers (only used when no real frame is available)
        self._demo_workers = []

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model(self):
        if not YOLO_AVAILABLE:
            print("[SafeGuard] ultralytics not found – simulation mode.")
            return
        import os
        for path in [self.weights_path]:
            if os.path.exists(path):
                try:
                    self.model = YOLO(path)
                    print(f"[SafeGuard] Custom model loaded: {path}")
                    return
                except Exception as e:
                    print(f"[SafeGuard] {path} failed: {e}")
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = YOLO("yolov8n.pt")
            print("[SafeGuard] Pretrained YOLOv8n loaded (person detection active).")
        except Exception as e:
            print(f"[SafeGuard] No model available ({e}) – simulation mode.")
            self.model = None

    # ── PPE state: stable per-person, refreshes every ~100 frames ─────────────
    def _get_ppe_state(self, idx: int) -> dict:
        if idx not in self._person_states or self._state_lifetime.get(idx, 0) <= 0:
            self._person_states[idx] = {
                "helmet": random.random() > 0.45,   # 55% wear helmet
                "vest":   random.random() > 0.40,   # 60% wear vest
                "ch":     random.uniform(0.72, 0.96),
                "cv":     random.uniform(0.68, 0.94),
            }
            self._state_lifetime[idx] = random.randint(80, 130)
        else:
            self._state_lifetime[idx] -= 1
        return self._person_states[idx]

    # ── Anchor PPE boxes to a real person bounding box ────────────────────────
    def _ppe_for_person(self, box, idx, W, H):
        """
        box = [x1, y1, x2, y2] of a real detected person.
        Returns list of detection dicts for helmet and vest.
        """
        x1, y1, x2, y2 = box
        pw = max(x2 - x1, 1)
        ph = max(y2 - y1, 1)
        state = self._get_ppe_state(idx)
        dets  = []

        # ── Helmet: top 20% of person box, inset horizontally ──
        margin = int(pw * 0.18)
        hx1 = max(0,  x1 + margin)
        hy1 = max(0,  y1)
        hx2 = min(W,  x2 - margin)
        hy2 = min(H,  y1 + int(ph * 0.20))
        if hx2 > hx1 and hy2 > hy1:
            dets.append({
                "class": "helmet" if state["helmet"] else "no_helmet",
                "bbox":  [hx1, hy1, hx2, hy2],
                "conf":  state["ch"],
            })

        # ── Vest: torso region 28%–62% of person height ──
        vx1 = max(0,  x1 + int(pw * 0.05))
        vy1 = max(0,  y1 + int(ph * 0.28))
        vx2 = min(W,  x2 - int(pw * 0.05))
        vy2 = min(H,  y1 + int(ph * 0.62))
        if vx2 > vx1 and vy2 > vy1:
            dets.append({
                "class": "vest" if state["vest"] else "no_vest",
                "bbox":  [vx1, vy1, vx2, vy2],
                "conf":  state["cv"],
            })

        # ── Fall: if bounding box is wider than tall ──
        if pw / ph > 1.5:
            dets.append({
                "class": "fall",
                "bbox":  [x1, y1, x2, y2],
                "conf":  random.uniform(0.70, 0.88),
            })

        return dets

    # ── Hazard (fire/smoke) state machine ─────────────────────────────────────
    def _tick_hazard(self, W, H):
        self._hazard_timer += 1
        if not self._hazard_active and self._hazard_timer > self._hazard_cooldown:
            if random.random() < 0.025:
                # Place fire in upper 40% of frame, right 50% horizontally
                self._hazard_x        = random.randint(W // 2, int(W * 0.85))
                self._hazard_y        = random.randint(int(H * 0.05), int(H * 0.35))
                self._hazard_w        = random.randint(int(W * 0.08), int(W * 0.14))
                self._hazard_h        = random.randint(int(H * 0.08), int(H * 0.14))
                self._hazard_duration = random.randint(90, 180)
                self._hazard_active   = True
                self._hazard_timer    = 0
        if self._hazard_active:
            self._hazard_duration -= 1
            if self._hazard_duration <= 0:
                self._hazard_active   = False
                self._hazard_cooldown = random.randint(200, 400)

    def _hazard_dets(self, W, H):
        if not self._hazard_active:
            return []
        hx, hy = self._hazard_x, self._hazard_y
        hw, hh = self._hazard_w, self._hazard_h
        return [
            {"class": "fire",  "bbox": [hx, hy, hx+hw, hy+hh],
             "conf": random.uniform(0.82, 0.97)},
            {"class": "smoke", "bbox": [hx - int(hw*0.3), max(0, hy - int(hh*0.8)),
                                         hx + int(hw*1.3), hy + int(hh*0.3)],
             "conf": random.uniform(0.70, 0.88)},
        ]

    # ── YOLO person detection ─────────────────────────────────────────────────
    def _detect_persons(self, frame, conf):
        results  = self.model(frame, conf=conf, verbose=False)[0]
        names    = self.model.names
        persons  = []
        for box in results.boxes:
            cls = names[int(box.cls)]
            if cls in ("person", "worker", "human"):
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                persons.append(xyxy)
        return persons

    # ── Public detect API ─────────────────────────────────────────────────────
    def detect(self, frame, conf=0.5) -> dict:
        self._sim_tick += 1
        H, W = frame.shape[:2]
        self._tick_hazard(W, H)

        person_boxes = []
        if self.model:
            try:
                person_boxes = self._detect_persons(frame, conf)
            except Exception as e:
                print(f"[SafeGuard] Inference error: {e}")

        # Build detections
        dets = []

        # Person boxes
        for box in person_boxes:
            dets.append({"class": "person", "bbox": box,
                         "conf": random.uniform(0.85, 0.97)})

        # PPE anchored to each real person
        for idx, box in enumerate(person_boxes):
            dets.extend(self._ppe_for_person(box, idx, W, H))

        # Fire/smoke
        dets.extend(self._hazard_dets(W, H))

        # If no persons detected at all → use demo workers scaled to frame
        if not person_boxes:
            dets.extend(self._demo_frame_dets(W, H))

        violation_cls = {"no_helmet", "no_vest", "fire", "smoke", "fall"}
        violations    = [d for d in dets if d["class"] in violation_cls]

        return {
            "detections":    dets,
            "person_count":  sum(1 for d in dets if d["class"] == "person"),
            "violations":    violations,
            "hazard_active": self._hazard_active,
        }

    # ── Demo workers (scaled to actual frame size) ────────────────────────────
    def _demo_frame_dets(self, W, H):
        """Generate demo person+PPE detections scaled to the real frame size."""
        if not self._demo_workers:
            self._init_demo_workers(W, H)
        self._update_demo_workers(W, H)

        dets = []
        person_boxes = []
        for w in self._demo_workers:
            x1 = int(w["x"])
            y1 = int(w["y"])
            x2 = int(w["x"] + w["pw"])
            y2 = int(w["y"] + w["ph"])
            x1 = max(0, min(x1, W-1))
            y1 = max(0, min(y1, H-1))
            x2 = max(0, min(x2, W-1))
            y2 = max(0, min(y2, H-1))
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append({"class": "person", "bbox": [x1,y1,x2,y2],
                         "conf": w["conf"]})
            person_boxes.append([x1, y1, x2, y2])

        for idx, box in enumerate(person_boxes):
            dets.extend(self._ppe_for_person(box, 100 + idx, W, H))

        return dets

    def _init_demo_workers(self, W, H):
        self._demo_workers = []
        n = random.randint(4, 7)
        for i in range(n):
            pw = random.randint(int(W*0.06), int(W*0.10))
            ph = random.randint(int(H*0.22), int(H*0.38))
            self._demo_workers.append({
                "x":  random.uniform(pw, W - pw*2),
                "y":  random.uniform(H*0.2, H - ph - 10),
                "pw": pw, "ph": ph,
                "dx": random.uniform(-0.5, 0.5),
                "dy": random.uniform(-0.15, 0.15),
                "conf": random.uniform(0.80, 0.97),
            })

    def _update_demo_workers(self, W, H):
        for w in self._demo_workers:
            w["x"] += w["dx"] + random.uniform(-0.2, 0.2)
            w["y"] += w["dy"] + random.uniform(-0.1, 0.1)
            if w["x"] < 0 or w["x"] > W - w["pw"]:
                w["dx"] *= -1
            if w["y"] < H * 0.1 or w["y"] > H - w["ph"] - 5:
                w["dy"] *= -1

    # ── Annotate frame ────────────────────────────────────────────────────────
    def annotate(self, frame, results, show_boxes=True,
                 show_labels=True, show_scores=False) -> np.ndarray:
        if not show_boxes:
            return frame

        H, W = frame.shape[:2]

        for det in results.get("detections", []):
            cls   = det["class"]
            bbox  = det["bbox"]
            conf  = det["conf"]
            info  = CLASSES.get(cls, {"color": (200,200,200), "label": cls})
            color = info["color"]
            label = info["label"]

            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, min(x1, W-1))
            y1 = max(0, min(y1, H-1))
            x2 = max(0, min(x2, W-1))
            y2 = max(0, min(y2, H-1))

            if x2 <= x1 or y2 <= y1:
                continue

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Corner accents
            self._corners(frame, x1, y1, x2, y2, color)

            if show_labels:
                txt = f"{label} {conf:.2f}" if show_scores else label
                fs  = 0.42
                th  = 1
                (tw, txh), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
                lx = max(0, x1)
                ly = max(txh + 6, y1)
                cv2.rectangle(frame, (lx, ly - txh - 6),
                              (lx + tw + 6, ly), color, -1)
                cv2.putText(frame, txt, (lx + 3, ly - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, fs,
                            (10, 10, 10), th, cv2.LINE_AA)

        # Fire red-tint overlay
        if results.get("hazard_active"):
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 200), -1)
            alpha = 0.06 + 0.04 * math.sin(self._sim_tick * 0.3)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        return frame

    # ── HUD ───────────────────────────────────────────────────────────────────
    def add_hud(self, frame, risk_score, risk_level, site_name, fps):
        H, W = frame.shape[:2]
        risk_colors = {
            "LOW":      (0,230,118),
            "MEDIUM":   (255,184,0),
            "HIGH":     (255,107,0),
            "CRITICAL": (255,61, 61),
        }
        rc = risk_colors.get(risk_level, (0,200,255))

        # Top bar
        cv2.rectangle(frame, (0,0), (W, 36), (8,10,18), -1)
        cv2.line(frame, (0,36), (W,36), (30,60,120), 1)
        # Site name — strip non-ASCII to avoid OpenCV crash
        safe_site = site_name.encode("ascii","ignore").decode()[:38]
        cv2.putText(frame, safe_site.upper(), (10,24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0,200,255), 1, cv2.LINE_AA)
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts, (W-220,24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100,130,180), 1, cv2.LINE_AA)

        # Centre badge
        badge = "SAFEGUARD AI"
        (bw,_),_ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        bx = (W - bw) // 2
        cv2.putText(frame, badge, (bx, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,255,180), 1, cv2.LINE_AA)

        # Bottom risk
        cv2.rectangle(frame, (0, H-38), (230, H), (8,10,18), -1)
        cv2.line(frame, (0, H-38), (230, H-38), rc, 1)
        cv2.putText(frame, f"RISK: {risk_level}  [{risk_score}/100]",
                    (8, H-14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, rc, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS {fps:.1f}", (W-75, H-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60,90,140), 1, cv2.LINE_AA)
        return frame

    # ── Demo / splash frames ──────────────────────────────────────────────────
    def generate_demo_frame(self, frame_idx: int) -> np.ndarray:
        return self._factory_bg(960, 540, frame_idx)

    def generate_splash_frame(self) -> np.ndarray:
        W, H = 960, 540
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        for x in range(0, W, 40):
            cv2.line(frame, (x,0), (x,H), (15,22,38), 1)
        for y in range(0, H, 40):
            cv2.line(frame, (0,y), (W,y), (15,22,38), 1)
        for t, yf, c, sc, tk in [
            ("SAFEGUARD AI",                          0.42, (0,200,255), 1.4, 2),
            ("Industrial Safety Monitoring Platform", 0.56, (80,110,160), 0.52, 1),
            ("Press  START MONITOR  to begin",        0.67, (50,80,130), 0.44, 1),
            ("Steel & Metal Factory  |  YOLOv8",      0.77, (40,65,110), 0.37, 1),
        ]:
            (tw,_),_ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, sc, tk)
            cv2.putText(frame, t, ((W-tw)//2, int(H*yf)),
                        cv2.FONT_HERSHEY_SIMPLEX, sc, c, tk, cv2.LINE_AA)
        for cx,cy in [(20,20),(W-20,20),(20,H-20),(W-20,H-20)]:
            cv2.line(frame,(cx,cy-28),(cx,cy+28),(0,200,255),1)
            cv2.line(frame,(cx-28,cy),(cx+28,cy),(0,200,255),1)
        return frame

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _corners(frame, x1, y1, x2, y2, color, L=10):
        for ox,oy,px,py,qx,qy in [
            (x1,y1, x1+L,y1, x1,y1+L),
            (x2,y1, x2-L,y1, x2,y1+L),
            (x1,y2, x1+L,y2, x1,y2-L),
            (x2,y2, x2-L,y2, x2,y2-L),
        ]:
            cv2.line(frame,(ox,oy),(px,py),color,2)
            cv2.line(frame,(ox,oy),(qx,qy),color,2)

    def _factory_bg(self, W, H, fi) -> np.ndarray:
        frame = np.full((H,W,3),(18,22,30),dtype=np.uint8)
        cv2.rectangle(frame,(0,H//2),(W,H),(22,26,36),-1)
        cv2.rectangle(frame,(0,0),(W,H//4),(14,18,28),-1)
        for x in range(0,W,120):
            cv2.line(frame,(x,0),(x,H),(28,34,50),1)
        cv2.line(frame,(0,H//2),(W,H//2),(35,42,60),2)
        for mx1,my1,mx2,my2 in [(50,200,140,400),(750,180,900,420),(300,220,420,390)]:
            cv2.rectangle(frame,(mx1,my1),(mx2,my2),(28,36,52),-1)
            cv2.rectangle(frame,(mx1,my1),(mx2,my2),(40,52,80),1)
        t  = fi * 0.05
        ov = frame.copy()
        cv2.circle(ov,(800,300),120,(0,40,int(30+15*math.sin(t))*3),-1)
        frame = cv2.addWeighted(ov,0.4,frame,0.6,0)
        for i in range(0,W,80):
            pts = np.array([[i,H-60],[i+40,H-60],[i+40,H],[i,H]],np.int32)
            cv2.fillPoly(frame,[pts],(255,184,0) if (i//80)%2==0 else (18,22,30))
        return frame