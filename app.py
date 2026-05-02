"""
SafeGuard AI — Industrial Safety Monitoring Platform
Enterprise Dashboard v4.0 | Enhanced Video Upload with Icon Interface
Features: Click to Upload, Video Thumbnails, Playback Controls
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
import tempfile
from datetime import datetime
from pathlib import Path
import hashlib
from PIL import Image
import io
from ultralytics import YOLO

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="SafeGuard AI — Enterprise",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# SESSION STATE
# ============================================================================
if 'running' not in st.session_state:
    st.session_state.running = False
if 'risk_history' not in st.session_state:
    st.session_state.risk_history = []
if 'incidents' not in st.session_state:
    st.session_state.incidents = []
if 'total_violations' not in st.session_state:
    st.session_state.total_violations = 0
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if '_fps' not in st.session_state:
    st.session_state._fps = 0.0
if 'site_name' not in st.session_state:
    st.session_state.site_name = "Steel Plant Alpha — Unit 3"
if 'zone_name' not in st.session_state:
    st.session_state.zone_name = "Blast Furnace — Zone B"
if 'supervisor' not in st.session_state:
    st.session_state.supervisor = "Eng. Tariq Mahmood"
if 'session_start' not in st.session_state:
    st.session_state.session_start = datetime.now()
if 'uploaded_videos' not in st.session_state:
    st.session_state.uploaded_videos = []  # List of dicts: {name, path, thumbnail}
if 'selected_video' not in st.session_state:
    st.session_state.selected_video = None
if 'video_playing' not in st.session_state:
    st.session_state.video_playing = False
if 'current_video_path' not in st.session_state:
    st.session_state.current_video_path = None

# ============================================================================
# CSS STYLES
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0a0c10;
    --bg-secondary: #111318;
    --bg-tertiary: #181b22;
    --border-color: #2a2e3a;
    --accent-primary: #00ff88;
    --accent-secondary: #00d4ff;
    --accent-danger: #ff3366;
    --accent-warning: #ffaa00;
    --text-primary: #ffffff;
    --text-secondary: #a0a5b0;
    --text-tertiary: #5a5e6e;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

.stApp { background: var(--bg-primary); }

/* Hide Streamlit default elements */
#MainMenu, header, footer, .stDeployButton {
    display: none !important;
}

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Custom scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--accent-primary); border-radius: 3px; }

/* Top Navigation Bar */
.top-nav {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 60px;
    background: rgba(17, 19, 24, 0.95);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--border-color);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    z-index: 1000;
}

.logo-area {
    display: flex;
    align-items: center;
    gap: 12px;
}

.logo-icon {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
}

.logo-text {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 18px;
    letter-spacing: 1px;
}

.logo-text span { color: var(--accent-primary); }

.status-bar {
    display: flex;
    gap: 16px;
}

.status-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: var(--bg-tertiary);
    border-radius: 20px;
    font-family: monospace;
    font-size: 11px;
    font-weight: 500;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-primary);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.status-dot.warning { background: var(--accent-warning); }

/* Main Layout */
.main-layout {
    display: flex;
    margin-top: 60px;
    min-height: calc(100vh - 60px);
}

/* Left Sidebar */
.sidebar-left {
    width: 300px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    padding: 20px;
    overflow-y: auto;
    height: calc(100vh - 60px);
    position: sticky;
    top: 60px;
}

.sidebar-section {
    margin-bottom: 24px;
}

.sidebar-title {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--text-tertiary);
    margin-bottom: 12px;
}

/* Video Upload Area */
.upload-area {
    background: var(--bg-tertiary);
    border: 2px dashed var(--border-color);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    margin-bottom: 16px;
}

.upload-area:hover {
    border-color: var(--accent-primary);
    background: rgba(0, 255, 136, 0.05);
}

.upload-icon {
    font-size: 48px;
    margin-bottom: 12px;
}

.upload-text {
    font-size: 14px;
    color: var(--text-secondary);
    margin-bottom: 8px;
}

.upload-hint {
    font-size: 11px;
    color: var(--text-tertiary);
}

/* Video Thumbnails Grid */
.video-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-top: 12px;
    max-height: 400px;
    overflow-y: auto;
    padding: 4px;
}

.video-thumbnail-card {
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    overflow: hidden;
    cursor: pointer;
    transition: all 0.2s ease;
    position: relative;
}

.video-thumbnail-card:hover {
    border-color: var(--accent-primary);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

.video-thumbnail-card.selected {
    border: 2px solid var(--accent-primary);
    box-shadow: 0 0 0 2px rgba(0, 255, 136, 0.2);
}

.thumbnail-image {
    width: 100%;
    height: 100px;
    object-fit: cover;
    background: #000;
}

.video-info {
    padding: 8px;
}

.video-name {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 4px;
}

.video-duration {
    font-size: 10px;
    color: var(--text-tertiary);
}

.delete-video {
    position: absolute;
    top: 4px;
    right: 4px;
    background: rgba(0,0,0,0.7);
    border-radius: 50%;
    width: 22px;
    height: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.2s;
}

.delete-video:hover {
    background: var(--accent-danger);
}

/* Video Area */
.video-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
    padding: 20px;
}

.video-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-color);
}

.video-title {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: var(--text-secondary);
}

.video-stats {
    display: flex;
    gap: 20px;
}

.video-stat {
    font-family: monospace;
    font-size: 12px;
    color: var(--text-tertiary);
}

.video-stat strong { color: var(--accent-primary); }

.video-container {
    background: #000000;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border-color);
    position: relative;
    min-height: 500px;
}

.video-feed {
    width: 100%;
    height: auto;
    display: block;
}

/* Right Sidebar */
.sidebar-right {
    width: 320px;
    background: var(--bg-secondary);
    border-left: 1px solid var(--border-color);
    padding: 20px;
    overflow-y: auto;
    height: calc(100vh - 60px);
    position: sticky;
    top: 60px;
}

.kpi-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 24px;
}

.kpi-card {
    background: var(--bg-tertiary);
    border-radius: 12px;
    padding: 16px;
    transition: transform 0.2s;
}

.kpi-card:hover { transform: translateY(-2px); }

.kpi-label {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--text-tertiary);
    margin-bottom: 8px;
}

.kpi-value {
    font-family: monospace;
    font-size: 28px;
    font-weight: 700;
    color: var(--accent-primary);
}

.risk-meter {
    background: var(--bg-tertiary);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 24px;
}

.risk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}

.risk-label { font-weight: 600; }

.risk-badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-family: monospace;
    font-size: 11px;
    font-weight: 700;
}

.risk-badge.low { background: rgba(0,255,136,0.15); color: var(--accent-primary); }
.risk-badge.medium { background: rgba(255,170,0,0.15); color: var(--accent-warning); }
.risk-badge.high { background: rgba(255,51,102,0.15); color: var(--accent-danger); }

.risk-bar {
    height: 8px;
    background: var(--bg-primary);
    border-radius: 4px;
    overflow: hidden;
    margin: 12px 0;
}

.risk-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent-primary), var(--accent-warning), var(--accent-danger));
    border-radius: 4px;
    transition: width 0.3s ease;
}

.alert-item {
    background: var(--bg-tertiary);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 8px;
    border-left: 3px solid var(--accent-danger);
}

.alert-title {
    font-weight: 600;
    font-size: 13px;
    margin-bottom: 4px;
}

.alert-time {
    font-family: monospace;
    font-size: 10px;
    color: var(--text-tertiary);
}

/* Custom Button */
.custom-btn {
    width: 100%;
    padding: 10px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    margin-bottom: 8px;
}

.custom-btn:hover {
    border-color: var(--accent-primary);
    color: var(--accent-primary);
}

.custom-btn.primary {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: #000000;
    font-weight: 600;
}

/* Incident Log */
.incident-table {
    width: 100%;
    font-family: monospace;
    font-size: 12px;
}

.incident-row {
    display: grid;
    grid-template-columns: 70px 90px 1fr 80px;
    padding: 10px 0;
    border-bottom: 1px solid var(--border-color);
}

.incident-header {
    color: var(--text-tertiary);
    font-weight: 600;
    font-size: 10px;
    text-transform: uppercase;
}

.incident-critical { color: var(--accent-danger); }
.incident-high { color: var(--accent-warning); }
.incident-low { color: var(--accent-primary); }

/* Streamlit Overrides */
.stButton > button {
    width: 100%;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    border-radius: 8px;
}

.stButton > button:hover {
    border-color: var(--accent-primary);
    color: var(--accent-primary);
}

/* Hide default file uploader */
[data-testid="stFileUploader"] {
    display: none;
}
</style>

<script>
function triggerFileUpload() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'video/*';
    input.onchange = (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                const videoData = {
                    name: file.name,
                    data: event.target.result
                };
                window.videoToUpload = videoData;
            };
            reader.readAsDataURL(file);
        }
    };
    input.click();
}
</script>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_video_thumbnail(video_path, timestamp_sec=2):
    """Extract a thumbnail frame from video"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_pos = int(fps * timestamp_sec) if fps > 0 else 30
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            # Resize thumbnail
            height, width = frame.shape[:2]
            thumb_height = 100
            thumb_width = int(width * (thumb_height / height))
            thumbnail = cv2.resize(frame, (thumb_width, thumb_height))
            return cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
    except:
        pass

    # Return placeholder if thumbnail extraction fails
    placeholder = np.zeros((100, 150, 3), dtype=np.uint8)
    placeholder[:] = (30, 30, 40)
    return placeholder

def get_video_duration(video_path):
    """Get video duration in seconds"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps > 0:
            duration = frame_count / fps
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            return f"{minutes}:{seconds:02d}"
    except:
        pass
    return "0:00"

def save_uploaded_video(uploaded_file):
    """Save uploaded video to temp directory and store in session state"""
    if uploaded_file is not None:
        # Create unique filename
        file_hash = hashlib.md5(uploaded_file.name.encode()).hexdigest()[:8]
        temp_dir = tempfile.gettempdir()
        video_path = os.path.join(temp_dir, f"video_{file_hash}_{uploaded_file.name}")

        # Save file
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract thumbnail
        thumbnail = extract_video_thumbnail(video_path)
        duration = get_video_duration(video_path)

        # Add to session state
        video_info = {
            "name": uploaded_file.name,
            "path": video_path,
            "thumbnail": thumbnail,
            "duration": duration,
            "size": f"{len(uploaded_file.getbuffer()) / (1024*1024):.1f} MB"
        }

        # Check if already exists
        existing = [v for v in st.session_state.uploaded_videos if v["name"] == uploaded_file.name]
        if not existing:
            st.session_state.uploaded_videos.append(video_info)

        return video_info
    return None

def delete_video(video_name):
    """Delete video from session state and temp file"""
    for i, video in enumerate(st.session_state.uploaded_videos):
        if video["name"] == video_name:
            # Delete temp file
            if os.path.exists(video["path"]):
                try:
                    os.remove(video["path"])
                except:
                    pass
            # Remove from session
            st.session_state.uploaded_videos.pop(i)
            if st.session_state.selected_video == video_name:
                st.session_state.selected_video = None
                st.session_state.current_video_path = None
            break

def render_upload_section():
    """Render the video upload interface with clickable icon"""

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-title">📹 VIDEO LIBRARY</div>
    </div>
    """, unsafe_allow_html=True)

    # Custom HTML/CSS upload button that opens file picker
    upload_html = """
    <div id="upload-trigger" class="upload-area" onclick="triggerFileUpload()">
        <div class="upload-icon">📽️</div>
        <div class="upload-text">Click to Upload Video</div>
        <div class="upload-hint">MP4, AVI, MOV, MKV • Max 500MB</div>
    </div>
    """
    st.markdown(upload_html, unsafe_allow_html=True)

    # Hidden file uploader
    uploaded_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        label_visibility="collapsed",
        key="video_uploader"
    )

    # Process uploaded file
    if uploaded_file is not None:
        video_info = save_uploaded_video(uploaded_file)
        if video_info:
            st.toast(f"✅ {video_info['name']} uploaded successfully!", icon="🎥")
            st.rerun()

    # Display video thumbnails if any exist
    if st.session_state.uploaded_videos:
        st.markdown('<div class="sidebar-title" style="margin-top:8px;">🎬 MY VIDEOS</div>', unsafe_allow_html=True)
        st.markdown('<div class="video-grid">', unsafe_allow_html=True)

        for video in st.session_state.uploaded_videos:
            # Convert thumbnail to bytes for display
            if video["thumbnail"] is not None:
                thumb_pil = Image.fromarray(video["thumbnail"])
                buf = io.BytesIO()
                thumb_pil.save(buf, format="PNG")
                thumb_bytes = buf.getvalue()

                selected_class = "selected" if st.session_state.selected_video == video["name"] else ""

                st.markdown(f"""
                <div class="video-thumbnail-card {selected_class}" onclick="selectVideo('{video["name"]}')">
                    <img class="thumbnail-image" src="data:image/png;base64,{thumb_bytes.hex()}" />
                    <div class="video-info">
                        <div class="video-name" title="{video["name"]}">{video["name"][:25]}</div>
                        <div class="video-duration">⏱️ {video["duration"]} • {video["size"]}</div>
                    </div>
                    <div class="delete-video" onclick="event.stopPropagation(); deleteVideoConfirm('{video["name"]}')">🗑️</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # JavaScript for handling video selection and deletion
    js_code = """
    <script>
    function selectVideo(videoName) {
        const input = document.createElement('input');
        input.type = 'text';
        input.value = videoName;
        input.style.display = 'none';
        document.body.appendChild(input);

        // Create form data
        const formData = new FormData();
        formData.append('selected_video', videoName);

        // Send to Streamlit
        fetch('/_stcore/stream', {
            method: 'POST',
            body: formData
        });

        document.body.removeChild(input);
    }

    function deleteVideoConfirm(videoName) {
        if (confirm(`Delete "${videoName}"?`)) {
            const formData = new FormData();
            formData.append('delete_video', videoName);
            fetch('/_stcore/stream', { method: 'POST', body: formData });
        }
    }
    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)

# ============================================================================
# MOCK DETECTION CLASSES (for demo)
# ============================================================================

# Add at the top with other imports:
from ultralytics import YOLO

# Replace the entire SafetyDetector class with:
# Add this import at the top (around line 10):
from ultralytics import YOLO

# Then REPLACE the entire SafetyDetector class (around line 440-465) with:
class SafetyDetector:
    def __init__(self):
        # Load pre-trained COCO model (no training file needed!)
        self.model = YOLO('yolov8n.pt')  # Auto-downloads first time

    def detect(self, frame, conf=0.5):
        # Run real detection
        results = self.model(frame, conf=conf)[0]

        # Convert YOLO results to your existing detection format
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Map COCO classes to your safety categories
            if cls_id == 0:  # person in COCO
                class_name = 'person'
            elif cls_id == 67:  # cell phone in COCO
                class_name = 'mobile_phone'
            else:
                continue  # Skip other objects for now

            detections.append(type('Detection', (), {
                'bbox': [x1, y1, x2, y2],
                'cls': class_name,
                'confidence': confidence
            })())
        return detections

    def annotate(self, frame, results, show_boxes=True, show_labels=True, show_scores=False):
        # Keep your existing annotate method - it works fine!
        annotated = frame.copy()
        for det in results:
            x1, y1, x2, y2 = det.bbox
            if show_boxes:
                color = (60, 255, 184) if det.cls == 'person' else (40, 120, 240)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            if show_labels:
                label = f"{det.cls}"
                if show_scores:
                    label += f" {det.confidence:.2f}"
                cv2.putText(annotated, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return annotated

    def generate_splash_frame(self):
        # Keep your existing splash frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame, "SafeGuard AI", (400, 360),
                   cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 136), 3)
        cv2.putText(frame, "Industrial Safety Platform", (380, 420),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
        cv2.putText(frame, "Click START to begin monitoring", (440, 480),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,100), 1)
        return frame

class RiskCalculator:
    @staticmethod
    def calculate(results):
        risk_score = min(100, len(results) * 12)
        if risk_score < 30: level = "LOW"
        elif risk_score < 60: level = "MEDIUM"
        elif risk_score < 85: level = "HIGH"
        else: level = "CRITICAL"
        violations = sum(1 for d in results if d.cls in ['mobile_phone', 'smoking'])
        return risk_score, level, violations

class AlertManager:
    def check_and_fire(self, results):
        alerts = []
        for det in results:
            if det.cls in ['mobile_phone', 'smoking']:
                alerts.append({
                    'message': f"{det.cls.upper()} detected!",
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'sent_to': "Safety Control Room"
                })
        return alerts

class IncidentLogger:
    def log(self, results, zone=""):
        violations = [d for d in results if d.cls in ['mobile_phone', 'smoking']]
        if not violations:
            return None
        return {
            'id': datetime.now().strftime("%H%M%S"),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'risk_level': "HIGH" if len(violations) > 1 else "MEDIUM",
            'violations': f"{len(violations)} violation(s)",
            'action': f"Alert sent to {zone}"
        }

# Initialize detectors
detector = SafetyDetector()
alert_mgr = AlertManager()
logger = IncidentLogger()

# ============================================================================
# UI RENDER FUNCTIONS
# ============================================================================

def render_top_bar():
    is_active = st.session_state.running
    status_class = "warning" if not is_active else ""
    status_text = "ACTIVE" if is_active else "STANDBY"

    st.markdown(f"""
    <div class="top-nav">
        <div class="logo-area">
            <div class="logo-icon">🛡️</div>
            <div class="logo-text">Safe<span>Guard</span> <span style="color:var(--accent-secondary);">AI</span></div>
        </div>
        <div class="status-bar">
            <div class="status-badge"><div class="status-dot"></div>AI ENGINE</div>
            <div class="status-badge"><div class="status-dot"></div>YOLOv8</div>
            <div class="status-badge"><div class="status-dot {status_class}"></div>{status_text}</div>
            <div class="status-badge">📍 {st.session_state.site_name}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_video_area(frame, fps):
    st.markdown('<div class="video-area">', unsafe_allow_html=True)

    source_status = "LIVE" if st.session_state.running else "STANDBY"
    video_name = ""
    if st.session_state.selected_video:
        for v in st.session_state.uploaded_videos:
            if v["name"] == st.session_state.selected_video:
                video_name = v["name"][:40]
                break

    st.markdown(f"""
    <div class="video-header">
        <div class="video-title">🎥 {video_name if video_name else st.session_state.zone_name} • {source_status}</div>
        <div class="video-stats">
            <div class="video-stat">FPS: <strong>{fps:.1f}</strong></div>
            <div class="video-stat">Resolution: <strong>1280x720</strong></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    if frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, use_container_width=True, channels="RGB")
    else:
        st.markdown("""
        <div style="display:flex; align-items:center; justify-content:center; height:500px; background:#000; color:var(--text-tertiary);">
            <div style="text-align:center;">
                <div style="font-size:48px; margin-bottom:16px;">🎥</div>
                <div>Select a video from the library and click START</div>
                <div style="font-size:12px;">or use Demo Mode</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin-top:12px; text-align:right; font-family:monospace; font-size:11px; color:var(--text-tertiary);">
        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_right_sidebar(risk_score, risk_level, violations, workers):
    st.markdown('<div class="sidebar-right">', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-title">📈 KEY METRICS</div>
    <div class="kpi-grid">
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-card"><div class="kpi-label">👥 WORKERS</div><div class="kpi-value">{workers}</div></div>
    <div class="kpi-card"><div class="kpi-label">⚠️ VIOLATIONS</div><div class="kpi-value" style="color:var(--accent-danger);">{violations}</div></div>
    </div>
    """, unsafe_allow_html=True)

    session_duration = int((datetime.now() - st.session_state.session_start).total_seconds() / 60)
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-label">⏱️ SESSION</div><div class="kpi-value">{session_duration}m</div></div>
        <div class="kpi-card"><div class="kpi-label">📊 ALERTS</div><div class="kpi-value">{len(st.session_state.alerts)}</div></div>
    </div>
    """, unsafe_allow_html=True)

    risk_color = "low"
    if risk_level == "MEDIUM": risk_color = "medium"
    elif risk_level in ["HIGH", "CRITICAL"]: risk_color = "high"

    st.markdown(f"""
    <div class="risk-meter">
        <div class="risk-header"><div class="risk-label">📊 RISK INDEX</div><div class="risk-badge {risk_color}">{risk_level}</div></div>
        <div class="risk-bar"><div class="risk-fill" style="width: {risk_score}%;"></div></div>
        <div style="text-align:center; margin-top:12px;"><span style="font-size:32px; font-weight:700; color:var(--accent-primary);">{risk_score}</span><span style="color:var(--text-tertiary);"> / 100</span></div>
    </div>
    """, unsafe_allow_html=True)

    if len(st.session_state.risk_history) > 1:
        risk_df = pd.DataFrame(st.session_state.risk_history, columns=["time", "score"])
        st.markdown('<div class="sidebar-title">📉 RISK TREND</div>', unsafe_allow_html=True)
        st.line_chart(risk_df.set_index("time")["score"], height=150, color="#00ff88")

    st.markdown('<div class="sidebar-title" style="margin-top:16px;">🚨 ALERTS</div>', unsafe_allow_html=True)

    alerts = st.session_state.alerts[-5:]
    if alerts:
        for alert in alerts:
            st.markdown(f"""
            <div class="alert-item">
                <div class="alert-title">⚠️ {alert.get('message', 'Alert')}</div>
                <div class="alert-time">{alert.get('timestamp', 'Just now')} • {alert.get('sent_to', 'Control Room')}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:center; padding:32px; color:var(--text-tertiary);'>✅ All Clear</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def render_incident_log():
    st.markdown("""
    <div style="margin: 0 20px 20px 20px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
            <div style="font-weight:600;">📋 INCIDENT LOG</div>
            <div style="font-size:11px; color:var(--text-tertiary);">Last 10 incidents</div>
        </div>
    """, unsafe_allow_html=True)

    incidents = st.session_state.incidents[-10:] if st.session_state.incidents else []

    if incidents:
        st.markdown("""
        <div class="incident-table">
            <div class="incident-row incident-header"><div>ID</div><div>TIME</div><div>DESCRIPTION</div><div>SEV</div></div>
        """, unsafe_allow_html=True)

        for inc in reversed(incidents):
            severity_class = "incident-low"
            if inc.get("risk_level") == "MEDIUM": severity_class = "incident-high"
            elif inc.get("risk_level") in ["HIGH", "CRITICAL"]: severity_class = "incident-critical"

            st.markdown(f"""
            <div class="incident-row">
                <div>#{inc.get('id', '—')}</div>
                <div>{inc.get('timestamp', '')[-8:]}</div>
                <div>{inc.get('violations', '')}</div>
                <div class="{severity_class}">{inc.get('risk_level', 'LOW')[0]}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:center; padding:32px; background:var(--bg-tertiary); border-radius:12px; color:var(--text-tertiary);'>📋 No incidents logged</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# MAIN DETECTION LOOP
# ============================================================================

def run_detection_loop():
    import random

    prev_time = time.time()

    while st.session_state.running:
        loop_start = time.time()

        # Get frame based on selected video or demo
        if st.session_state.selected_video:
            # Find selected video path
            video_path = None
            for v in st.session_state.uploaded_videos:
                if v["name"] == st.session_state.selected_video:
                    video_path = v["path"]
                    break

            if video_path and os.path.exists(video_path):
                # Initialize video capture if not already
                if not hasattr(st.session_state, 'video_cap') or st.session_state.video_cap is None:
                    st.session_state.video_cap = cv2.VideoCapture(video_path)
                    st.session_state.frame_position = 0

                ret, frame = st.session_state.video_cap.read()
                if not ret:
                    # Loop video
                    st.session_state.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = st.session_state.video_cap.read()

                if ret and frame is not None:
                    # Resize for consistency
                    if frame.shape[1] > 1280:
                        scale = 1280 / frame.shape[1]
                        new_height = int(frame.shape[0] * scale)
                        frame = cv2.resize(frame, (1280, new_height))
                else:
                    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(frame, "Video Error", (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            else:
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(frame, "No video selected", (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        else:
            # Demo mode - generate frame
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            frame[:] = (20, 25, 35)
            cv2.putText(frame, "DEMO MODE", (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 136), 3)
            cv2.putText(frame, "Upload a video or use webcam", (470, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150,150,150), 1)

        if frame is not None:
            # Detection
            results = detector.detect(frame, conf=0.5)
            annotated = detector.annotate(frame, results, show_boxes=True, show_labels=True, show_scores=False)

            # Metrics
            risk_score, risk_level, violations_frame = RiskCalculator.calculate(results)
            workers = sum(1 for d in results if d.cls == 'person')

            # Update state
            st.session_state.total_violations += violations_frame
            st.session_state.risk_history.append((datetime.now().strftime("%H:%M:%S"), risk_score))
            if len(st.session_state.risk_history) > 60:
                st.session_state.risk_history.pop(0)

            # Alerts
            new_alerts = alert_mgr.check_and_fire(results)
            for alert in new_alerts:
                st.session_state.alerts.append(alert)
                if risk_score >= 40:
                    incident = logger.log(results, zone=st.session_state.zone_name)
                    if incident:
                        st.session_state.incidents.append(incident)

            # FPS
            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 0.001)
            prev_time = current_time
            st.session_state._fps = fps

            # Render
            render_video_area(annotated, fps)
            render_right_sidebar(risk_score, risk_level, st.session_state.total_violations, workers)
            render_incident_log()

        elapsed = time.time() - loop_start
        time.sleep(max(0, 0.033 - elapsed))

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    render_top_bar()

    col_left, col_center, col_right = st.columns([1, 2.2, 1.2], gap="small")

    with col_left:
        render_upload_section()

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">🎮 CONTROLS</div>', unsafe_allow_html=True)

        # Video selection dropdown (alternative to clicking thumbnails)
        if st.session_state.uploaded_videos:
            video_names = ["Demo Mode"] + [v["name"] for v in st.session_state.uploaded_videos]
            selected_idx = 0
            if st.session_state.selected_video:
                try:
                    selected_idx = video_names.index(st.session_state.selected_video)
                except:
                    selected_idx = 0

            selected = st.selectbox("Select Source", video_names, index=selected_idx, label_visibility="collapsed")

            if selected == "Demo Mode":
                st.session_state.selected_video = None
                if hasattr(st.session_state, 'video_cap'):
                    st.session_state.video_cap = None
            else:
                if st.session_state.selected_video != selected:
                    st.session_state.selected_video = selected
                    if hasattr(st.session_state, 'video_cap'):
                        st.session_state.video_cap = None

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ START", use_container_width=True):
                st.session_state.running = True
                st.session_state.risk_history = []
                st.session_state.incidents = []
                st.session_state.total_violations = 0
                st.session_state.alerts = []
                st.session_state.session_start = datetime.now()
                if hasattr(st.session_state, 'video_cap'):
                    st.session_state.video_cap = None
                st.rerun()
        with col2:
            if st.button("⏹️ STOP", use_container_width=True):
                st.session_state.running = False
                if hasattr(st.session_state, 'video_cap'):
                    st.session_state.video_cap = None
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Site info
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">🏭 SITE INFO</div>', unsafe_allow_html=True)
        st.session_state.site_name = st.text_input("Facility", st.session_state.site_name, label_visibility="collapsed")
        st.session_state.zone_name = st.text_input("Zone", st.session_state.zone_name, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_center:
        if not st.session_state.running:
            splash = detector.generate_splash_frame()
            splash_rgb = cv2.cvtColor(splash, cv2.COLOR_BGR2RGB)
            st.image(splash_rgb, use_container_width=True)
            st.info("🟢 Select a video from the library above and click **START** to begin monitoring")
            render_incident_log()
        else:
            run_detection_loop()

    with col_right:
        if not st.session_state.running:
            render_right_sidebar(0, "LOW", 0, 0)
            render_incident_log()

if __name__ == "__main__":
    main()