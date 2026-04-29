"""
SafeGuard AI — Flask Backend Server
Replaces Streamlit. Serves the HTML frontend and provides API endpoints.

Run:  python server.py
Then open:  http://localhost:5000
"""

import cv2
import time
import tempfile
import threading
import numpy as np
from datetime import datetime
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

# ── Import your existing modules (unchanged) ──────────────────────────────
try:
    from detect import SafetyDetector
    from utils import AlertManager, RiskCalculator, IncidentLogger
except Exception as e:
    print(f"[WARN] Could not import detection modules: {e}")
    # Fallback stubs so the server still starts
    class SafetyDetector:
        def detect(self, frame, conf=0.5): return []
        def annotate(self, frame, results, **kw): return frame
        def generate_splash_frame(self):
            f = np.zeros((540, 960, 3), dtype=np.uint8)
            f[:] = (8, 15, 26)
            return f

    class AlertManager:
        def check_and_fire(self, results): return []

    class RiskCalculator:
        @staticmethod
        def calculate(results): return 0, "LOW", 0

    class IncidentLogger:
        def log(self, results, zone=""): return None


# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='frontend')
CORS(app)  # allow cross-origin requests from the HTML file

# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE (thread-safe via lock)
# ─────────────────────────────────────────────────────────────────────────────
state_lock = threading.Lock()
state = {
    'running':          False,
    'paused':           False,
    'source':           'demo',
    'conf':             0.5,
    'alert_thresh':     40,
    'show_boxes':       True,
    'show_labels':      True,
    'show_scores':      False,
    'zone':             'Blast Furnace — Zone B',
    'fps':              0.0,
    'workers':          0,
    'total_violations': 0,
    'risk_score':       0,
    'risk_level':       'LOW',
    'alerts':           [],
    'incidents':        [],
    'latest_frame':     None,  # JPEG bytes
}

# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
detector  = SafetyDetector()
alert_mgr = AlertManager()
logger    = IncidentLogger()

# ─────────────────────────────────────────────────────────────────────────────
# DETECTION THREAD
# ─────────────────────────────────────────────────────────────────────────────
detect_thread = None

def detection_loop():
    """Background thread: captures frames, runs YOLO, updates state."""
    cap = None
    demo_mode = state['source'] == 'demo'

    if not demo_mode:
        src = 0 if state['source'] == 'webcam' else state.get('video_path', '')
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print("[WARN] Cannot open video source, falling back to demo mode")
            demo_mode = True

    prev_time = time.time()

    while True:
        with state_lock:
            if not state['running']:
                break
            if state['paused']:
                time.sleep(0.05)
                continue
            conf        = state['conf']
            show_boxes  = state['show_boxes']
            show_labels = state['show_labels']
            show_scores = state['show_scores']
            zone        = state['zone']
            alert_thresh = state['alert_thresh']

        # ── Grab frame ────────────────────────────────────────────────────
        if demo_mode:
            frame = np.zeros((540, 960, 3), dtype=np.uint8)
            frame[:] = (8, 15, 26)
            # Draw a moving crosshair for demo visualization
            t = time.time()
            cx = int(480 + 200 * np.sin(t * 0.5))
            cy = int(270 + 100 * np.sin(t * 0.7))
            cv2.drawMarker(frame, (cx, cy), (60, 200, 60),
                           cv2.MARKER_CROSS, 30, 1, cv2.LINE_AA)
            cv2.putText(frame, "SAFEGUARD AI  —  DEMO MODE",
                        (320, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (60, 200, 60), 1, cv2.LINE_AA)
        else:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((540, 960, 3), dtype=np.uint8)

        # ── Detect ────────────────────────────────────────────────────────
        results   = detector.detect(frame, conf=conf)
        annotated = detector.annotate(
            frame.copy(), results,
            show_boxes=show_boxes,
            show_labels=show_labels,
            show_scores=show_scores,
        )

        risk_score, risk_level, violations_frame = RiskCalculator.calculate(results)
        workers = sum(1 for d in results if getattr(d, 'cls', None) == 'person')

        # ── New alerts ────────────────────────────────────────────────────
        new_alerts = alert_mgr.check_and_fire(results)
        new_incidents = []
        for a in new_alerts:
            if risk_score >= alert_thresh:
                inc = logger.log(results, zone=zone)
                if inc:
                    new_incidents.append(inc)

        # ── FPS ───────────────────────────────────────────────────────────
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 0.001)
        prev_time = curr_time

        # ── Encode frame to JPEG ──────────────────────────────────────────
        _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buf.tobytes()

        # ── Update shared state ───────────────────────────────────────────
        with state_lock:
            state['fps']            = fps
            state['workers']        = workers
            state['total_violations'] += violations_frame
            state['risk_score']     = risk_score
            state['risk_level']     = risk_level
            state['latest_frame']   = frame_bytes

            for a in new_alerts:
                state['alerts'].append(a)
            if len(state['alerts']) > 20:
                state['alerts'] = state['alerts'][-20:]

            for inc in new_incidents:
                state['incidents'].append(inc)
            if len(state['incidents']) > 50:
                state['incidents'] = state['incidents'][-50:]

        time.sleep(0.02)   # ~50 FPS cap

    if cap:
        cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the HTML frontend."""
    return send_from_directory('frontend', 'index.html')


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'time': datetime.now().isoformat()})


@app.route('/api/metrics')
def metrics():
    """Return current detection metrics as JSON."""
    with state_lock:
        return jsonify({
            'fps':              round(state['fps'], 1),
            'workers':          state['workers'],
            'total_violations': state['total_violations'],
            'risk_score':       state['risk_score'],
            'risk_level':       state['risk_level'],
            'alerts':           state['alerts'][-5:],
            'incidents':        state['incidents'][-8:],
            'running':          state['running'],
        })


@app.route('/api/control', methods=['POST'])
def control():
    """Start / stop / pause / update settings."""
    global detect_thread
    data = request.get_json(force=True)
    action = data.get('action', '')

    if action == 'start':
        with state_lock:
            state['running']          = True
            state['paused']           = False
            state['source']           = data.get('source', 'demo')
            state['conf']             = float(data.get('conf', 0.5))
            state['alert_thresh']     = int(data.get('alert_thresh', 40))
            state['zone']             = data.get('zone', 'Zone A')
            state['show_boxes']       = bool(data.get('show_boxes', True))
            state['show_labels']      = bool(data.get('show_labels', True))
            state['show_scores']      = bool(data.get('show_scores', False))
            state['total_violations'] = 0
            state['alerts']           = []
            state['incidents']        = []

        if detect_thread is None or not detect_thread.is_alive():
            detect_thread = threading.Thread(target=detection_loop, daemon=True)
            detect_thread.start()

        return jsonify({'status': 'started'})

    elif action == 'stop':
        with state_lock:
            state['running'] = False
        return jsonify({'status': 'stopped'})

    elif action == 'pause':
        with state_lock:
            state['paused'] = True
        return jsonify({'status': 'paused'})

    elif action == 'resume':
        with state_lock:
            state['paused'] = False
        return jsonify({'status': 'resumed'})

    elif action == 'settings':
        with state_lock:
            state['conf']         = float(data.get('conf', state['conf']))
            state['alert_thresh'] = int(data.get('alert_thresh', state['alert_thresh']))
            state['show_boxes']   = bool(data.get('show_boxes', state['show_boxes']))
            state['show_labels']  = bool(data.get('show_labels', state['show_labels']))
            state['show_scores']  = bool(data.get('show_scores', state['show_scores']))
        return jsonify({'status': 'settings updated'})

    return jsonify({'error': 'unknown action'}), 400


@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Accept an uploaded video file and store its path."""
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400
    f = request.files['file']
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    f.save(tmp.name)
    with state_lock:
        state['video_path'] = tmp.name
    return jsonify({'status': 'ok', 'path': tmp.name})


def gen_frames():
    """MJPEG generator: yields JPEG frames for the /video_feed endpoint."""
    splash = detector.generate_splash_frame()
    _, splash_buf = cv2.imencode('.jpg', splash)
    splash_bytes  = splash_buf.tobytes()

    while True:
        with state_lock:
            running = state['running']
            frame   = state.get('latest_frame')

        if not running or frame is None:
            # Show splash while not running
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + splash_bytes + b'\r\n')
            time.sleep(0.1)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.02)


@app.route('/video_feed')
def video_feed():
    """MJPEG stream endpoint — point <img src="/video_feed"> at this."""
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  SafeGuard AI — Flask Server")
    print("  http://localhost:5000")
    print("=" * 55)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)