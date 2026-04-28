"""
AI Industrial Safety Monitoring System
Streamlit Dashboard - Production Demo
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import random
from datetime import datetime, timedelta

# ─── Safe imports for Streamlit Cloud ────────────────────────────────────────
try:
    from detect import SafetyDetector
    from utils import AlertManager, RiskCalculator, IncidentLogger
except Exception as e:
    import streamlit as st
    st.error(f"Import error: {e}")
    st.stop()

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeGuard AI – Industrial Safety Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Rajdhani:wght@400;500;600;700&family=Exo+2:wght@300;400;600;800&display=swap');

:root {
    --bg-primary:   #0a0d14;
    --bg-secondary: #0f1420;
    --bg-panel:     #131926;
    --bg-card:      #1a2235;
    --accent-blue:  #00a8ff;
    --accent-cyan:  #00ffd0;
    --accent-red:   #ff3d3d;
    --accent-amber: #ffb800;
    --accent-green: #00e676;
    --text-primary: #e8eaf0;
    --text-muted:   #6b7a99;
    --border:       #1e2d4a;
}

.stApp { background: var(--bg-primary); font-family: 'Exo 2', sans-serif; }
.stApp * { color: var(--text-primary); }
.block-container { padding: 1rem 2rem 2rem; max-width: 1600px; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.header-banner {
    background: linear-gradient(135deg, #060b18 0%, #0d1b3e 50%, #060b18 100%);
    border: 1px solid #1e3a6e;
    border-radius: 12px;
    padding: 20px 32px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-blue), transparent);
}
.header-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem; font-weight: 700;
    color: #fff; letter-spacing: 2px; margin: 0;
}
.header-title span { color: var(--accent-cyan); }
.header-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: var(--text-muted);
    letter-spacing: 3px; margin-top: 4px;
    text-transform: uppercase;
}
.header-status {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(0,230,118,0.1); border: 1px solid rgba(0,230,118,0.3);
    border-radius: 20px; padding: 4px 14px;
    font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;
    color: var(--accent-green);
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--accent-green);
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; transform: scale(1); }
    50%      { opacity:0.4; transform: scale(0.8); }
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 20px;
    position: relative; overflow: hidden;
}
.metric-card::after {
    content: ''; position: absolute;
    top: 0; left: 0; width: 3px; height: 100%;
    border-radius: 10px 0 0 10px;
}
.metric-card.blue::after  { background: var(--accent-blue); }
.metric-card.cyan::after  { background: var(--accent-cyan); }
.metric-card.amber::after { background: var(--accent-amber); }
.metric-card.red::after   { background: var(--accent-red); }
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; letter-spacing: 2px;
    color: var(--text-muted); text-transform: uppercase; margin-bottom: 6px;
}
.metric-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.4rem; font-weight: 700; line-height: 1;
}
.metric-delta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; color: var(--text-muted); margin-top: 4px;
}

.risk-badge {
    display: inline-block; padding: 6px 18px;
    border-radius: 4px; font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem; font-weight: 700; letter-spacing: 2px;
}
.risk-LOW      { background:rgba(0,230,118,0.15); border:1px solid var(--accent-green); color:var(--accent-green); }
.risk-MEDIUM   { background:rgba(255,184,0,0.15); border:1px solid var(--accent-amber); color:var(--accent-amber); }
.risk-HIGH     { background:rgba(255,107,0,0.15); border:1px solid #ff6b00; color:#ff6b00; }
.risk-CRITICAL { background:rgba(255,61,61,0.15); border:1px solid var(--accent-red); color:var(--accent-red);
                 animation: flash 0.8s infinite; }
@keyframes flash { 50% { opacity: 0.5; } }

.alert-item {
    background: var(--bg-panel); border-radius: 8px;
    padding: 10px 14px; margin-bottom: 8px;
    border-left: 3px solid var(--accent-red);
    font-size: 0.82rem; font-family: 'JetBrains Mono', monospace;
}
.alert-item.warn { border-left-color: var(--accent-amber); }
.alert-item.info { border-left-color: var(--accent-blue); }
.alert-time { color: var(--text-muted); font-size: 0.68rem; display:block; margin-top:3px; }

.log-entry {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    padding: 6px 12px; border-bottom: 1px solid var(--border);
    display: flex; gap: 12px; align-items: center;
}
.log-ts { color: var(--text-muted); min-width: 140px; }
.log-msg { color: var(--text-primary); }
.log-sev-HIGH     { color: var(--accent-red); }
.log-sev-MEDIUM   { color: var(--accent-amber); }
.log-sev-LOW      { color: var(--accent-green); }
.log-sev-CRITICAL { color: var(--accent-red); }

.risk-bar-wrap { background: #1a2235; border-radius: 4px; height: 10px; overflow: hidden; margin-top: 8px; }
.risk-bar { height: 100%; border-radius: 4px; transition: width 0.5s ease; }

.section-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
    letter-spacing: 3px; color: var(--accent-blue);
    text-transform: uppercase; margin: 16px 0 8px;
    padding-bottom: 6px; border-bottom: 1px solid var(--border);
}

[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: var(--text-muted) !important; font-size:0.75rem; }

hr { border-color: var(--border) !important; }

.stButton > button {
    background: linear-gradient(135deg, #0d2a5e, #1a4490) !important;
    border: 1px solid var(--accent-blue) !important;
    color: #fff !important; border-radius: 6px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.9rem !important; letter-spacing: 1px !important;
    font-weight: 600 !important; padding: 6px 20px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1a4490, #0d2a5e) !important;
    border-color: var(--accent-cyan) !important;
}
div[data-testid="stMetric"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ────────────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "risk_history" not in st.session_state:
    st.session_state.risk_history = []
if "incidents" not in st.session_state:
    st.session_state.incidents = []
if "total_violations" not in st.session_state:
    st.session_state.total_violations = 0
if "alerts" not in st.session_state:
    st.session_state.alerts = []

detector  = SafetyDetector()
alert_mgr = AlertManager()
logger    = IncidentLogger()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
    <div>
      <div class="header-title">SafeGuard AI</div>
      <div class="header-sub">Industrial Safety Monitoring Platform &nbsp;|&nbsp; Steel &amp; Metal Factory Edition</div>
    </div>
    <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
      <div class="header-status"><span class="status-dot"></span>SYSTEM ONLINE</div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#6b7a99;">
        v2.4.1 &nbsp;|&nbsp; YOLOv8 Engine
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">Configuration</div>', unsafe_allow_html=True)
    source_mode  = st.selectbox("Input Source", ["Demo Mode", "Upload Video", "Webcam (Live)"])
    conf_thresh  = st.slider("Detection Confidence", 0.3, 0.9, 0.5, 0.05)
    alert_thresh = st.slider("Alert Threshold (Risk Score)", 20, 80, 40, 5)

    st.markdown('<div class="section-label">Display Options</div>', unsafe_allow_html=True)
    show_boxes  = st.toggle("Bounding Boxes",     True)
    show_labels = st.toggle("Object Labels",       True)
    show_scores = st.toggle("Confidence Scores",   False)

    st.markdown('<div class="section-label">Site Info</div>', unsafe_allow_html=True)
    site_name = st.text_input("Facility Name", "Steel Plant Alpha - Unit 3")
    zone_name = st.text_input("Zone",          "Blast Furnace Area - Zone B")

    st.markdown('<div class="section-label">Supervisor Notifications</div>', unsafe_allow_html=True)
    supervisor   = st.text_input("Supervisor", "Eng. Tariq Mahmood")
    notify_email = st.toggle("Email Alerts", True)
    notify_sms   = st.toggle("SMS Alerts",   False)

    # Upload widget — only shown in Upload mode
    uploaded_file = None
    if source_mode == "Upload Video":
        st.markdown('<div class="section-label">Upload Video</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported: MP4, AVI, MOV, MKV"
        )
        if uploaded_file:
            st.success(f"Ready: {uploaded_file.name}")
        else:
            st.info("Upload a video then press START")

    st.markdown("---")
    st.markdown(
        '<div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#6b7a99;text-align:center;">'
        'SafeGuard AI &copy; 2025<br>Softnity Technologies</div>',
        unsafe_allow_html=True
    )

# ─── Control buttons ──────────────────────────────────────────────────────────
col_btn1, col_btn2, col_btn3, _ = st.columns([1, 1, 1, 5])
with col_btn1:
    if st.button("START MONITOR"):
        st.session_state.running         = True
        st.session_state.risk_history    = []
        st.session_state.incidents       = []
        st.session_state.total_violations = 0
        st.session_state.alerts          = []
with col_btn2:
    if st.button("STOP"):
        st.session_state.running = False
with col_btn3:
    if st.button("RESET"):
        st.session_state.running         = False
        st.session_state.risk_history    = []
        st.session_state.incidents       = []
        st.session_state.total_violations = 0
        st.session_state.alerts          = []

st.markdown("---")

# ─── Main Layout ──────────────────────────────────────────────────────────────
col_video, col_panel = st.columns([3, 2], gap="medium")

with col_video:
    st.markdown('<div class="section-label">Live Detection Feed</div>', unsafe_allow_html=True)
    video_placeholder = st.empty()
    fps_placeholder   = st.empty()

with col_panel:
    st.markdown('<div class="section-label">Real-Time Metrics</div>', unsafe_allow_html=True)
    metric_placeholder = st.empty()
    st.markdown('<div class="section-label">Active Alerts</div>', unsafe_allow_html=True)
    alert_placeholder  = st.empty()
    st.markdown('<div class="section-label">Risk Score Trend</div>', unsafe_allow_html=True)
    chart_placeholder  = st.empty()

st.markdown("---")
st.markdown('<div class="section-label">Incident Log</div>', unsafe_allow_html=True)
log_placeholder = st.empty()


# ─── Render helpers ───────────────────────────────────────────────────────────
def render_metrics(workers, violations, risk_score, risk_level, fps):
    color_map = {
        "LOW":      "#00e676",
        "MEDIUM":   "#ffb800",
        "HIGH":     "#ff6b00",
        "CRITICAL": "#ff3d3d",
    }
    bar_color = color_map.get(risk_level, "#00a8ff")

    metric_placeholder.markdown(f"""
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:12px;">
      <div class="metric-card blue">
        <div class="metric-label">Workers Detected</div>
        <div class="metric-value" style="color:#00a8ff;">{workers}</div>
        <div class="metric-delta">Active in frame</div>
      </div>
      <div class="metric-card {'red' if violations > 0 else 'cyan'}">
        <div class="metric-label">Total Violations</div>
        <div class="metric-value" style="color:{'#ff3d3d' if violations > 0 else '#00ffd0'};">{violations}</div>
        <div class="metric-delta">Session cumulative</div>
      </div>
      <div class="metric-card amber">
        <div class="metric-label">Risk Score</div>
        <div class="metric-value" style="color:{bar_color};">{risk_score}</div>
        <div class="risk-bar-wrap">
          <div class="risk-bar" style="width:{risk_score}%;background:{bar_color};"></div>
        </div>
      </div>
      <div class="metric-card {'red' if risk_level in ('HIGH','CRITICAL') else 'cyan'}">
        <div class="metric-label">Risk Level</div>
        <div class="metric-value" style="font-size:1.4rem;margin-top:6px;">
          <span class="risk-badge risk-{risk_level}">{risk_level}</span>
        </div>
      </div>
    </div>
    <div style="font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#6b7a99;text-align:right;">
      FPS: {fps:.1f} &nbsp;|&nbsp; {datetime.now().strftime('%H:%M:%S')} &nbsp;|&nbsp; {zone_name}
    </div>
    """, unsafe_allow_html=True)


def render_alerts(alerts):
    if not alerts:
        alert_placeholder.markdown(
            '<div class="alert-item info">All systems nominal — No active alerts</div>',
            unsafe_allow_html=True)
        return
    html = ""
    for a in alerts[-6:]:
        color = a.get("color", "red")
        css   = "warn" if color == "orange" else ("info" if color == "blue" else "")
        msg   = a.get("message", "")
        ts    = a.get("timestamp", "")
        atype = a.get("type", "")
        html += f'<div class="alert-item {css}">{atype}: {msg}<span class="alert-time">{ts}</span></div>'
    alert_placeholder.markdown(html, unsafe_allow_html=True)


def render_chart(history):
    if len(history) < 2:
        return
    df = pd.DataFrame(history, columns=["time", "score"])
    chart_placeholder.line_chart(df.set_index("time")["score"], height=100, use_container_width=True)


def render_log(incidents):
    if not incidents:
        log_placeholder.markdown(
            '<div class="log-entry">'
            '<span class="log-ts">-</span>'
            '<span class="log-msg">No incidents recorded this session.</span>'
            '</div>',
            unsafe_allow_html=True)
        return
    html = ""
    for inc in incidents[-12:][::-1]:
        if inc is None:
            continue
        level = inc.get("risk_level", "LOW")
        html += f"""<div class="log-entry">
          <span class="log-ts">{inc.get('timestamp','')}</span>
          <span class="log-sev-{level}">[{level}]</span>
          <span class="log-msg">{inc.get('id','')} | {inc.get('zone','')} | {inc.get('violations','')} | {inc.get('action','')}</span>
        </div>"""
    log_placeholder.markdown(html, unsafe_allow_html=True)


# ─── Detection loop ───────────────────────────────────────────────────────────
def run_detection_loop():
    cap       = None
    demo_mode = (source_mode == "Demo Mode")

    # ── Open video source ──
    if not demo_mode:
        if source_mode == "Upload Video" and uploaded_file is not None:
            import tempfile, os
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.flush()
            tfile.close()
            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                st.error("Could not open video file. Try MP4 format.")
                st.session_state.running = False
                return
        elif source_mode == "Webcam (Live)":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.warning("Webcam not found - switching to Demo Mode.")
                demo_mode = True
        else:
            # Upload mode but no file yet
            st.warning("Please upload a video file first.")
            st.session_state.running = False
            return

    prev_time = time.time()
    frame_idx = 0

    while st.session_state.running:

        # ── Get frame ──
        if demo_mode:
            # Generate a blank frame for demo (simulation runs on blank)
            frame = np.zeros((540, 960, 3), dtype=np.uint8)
            frame[:] = (18, 20, 25)   # dark background
        else:
            ret, frame = cap.read()
            if not ret:
                # Loop video back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((540, 960, 3), dtype=np.uint8)

        # ── Run detection ──
        results   = detector.detect(frame, conf=conf_thresh)   # returns list[Detection]
        annotated = detector.annotate(
            frame.copy(), results,
            show_boxes=show_boxes,
            show_labels=show_labels,
            show_scores=show_scores,
        )

        # ── Risk calculation ──
        # results is list[Detection] — RiskCalculator handles this correctly
        risk_score, risk_level, violations_frame = RiskCalculator.calculate(results)

        # Count workers directly from detection list
        workers = sum(1 for d in results if d.cls == "person")

        st.session_state.total_violations += violations_frame
        ts = datetime.now().strftime("%H:%M:%S")
        st.session_state.risk_history.append((ts, risk_score))
        if len(st.session_state.risk_history) > 60:
            st.session_state.risk_history.pop(0)

        # ── Alerts ──
        # alert_mgr.check_and_fire() is the correct method name
        new_alerts = alert_mgr.check_and_fire(results)
        for a in new_alerts:
            st.session_state.alerts.append(a)
            # Log incident if risk is above threshold
            if risk_score >= alert_thresh:
                incident = logger.log(results, zone=zone_name)
                if incident:
                    st.session_state.incidents.append(incident)

        # ── FPS ──
        curr_time = time.time()
        fps       = 1.0 / max(curr_time - prev_time, 0.001)
        prev_time = curr_time

        # ── Render ──
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        render_metrics(workers, st.session_state.total_violations, risk_score, risk_level, fps)
        render_alerts(st.session_state.alerts)
        render_chart(st.session_state.risk_history)
        render_log(st.session_state.incidents)

        frame_idx += 1
        time.sleep(0.03)   # ~30 fps cap

    if cap:
        cap.release()


# ─── Entry point ──────────────────────────────────────────────────────────────
if not st.session_state.running:
    # Show splash frame while idle
    splash = detector.generate_splash_frame()
    video_placeholder.image(cv2.cvtColor(splash, cv2.COLOR_BGR2RGB), use_container_width=True)
    render_metrics(0, 0, 0, "LOW", 0.0)
    render_alerts([])
    render_log([])
    fps_placeholder.markdown(
        '<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#6b7a99;text-align:center;padding:8px;">'
        'Press START MONITOR to begin</div>',
        unsafe_allow_html=True
    )
else:
    run_detection_loop()