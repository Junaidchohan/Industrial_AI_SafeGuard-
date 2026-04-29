"""
SafeGuard AI  —  Industrial Safety Monitoring Platform
Enterprise Dashboard v4.0 | Palantir/Tesla-Grade UI
Redesigned by: Senior UI/UX + Frontend Architect
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta

try:
    from detect import SafetyDetector
    from utils import AlertManager, RiskCalculator, IncidentLogger
except Exception as e:
    st.error(f"Import error: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeGuard AI — Enterprise",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

# ─────────────────────────────────────────────────────────────────────────────
# WORLD-CLASS ENTERPRISE CSS — v4.0
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800;900&family=Azeret+Mono:wght@300;400;500;600;700&family=Barlow:wght@300;400;500;600;700&family=Barlow+Condensed:wght@300;400;500;600;700;800;900&display=swap');

/* ══════════════════════════════════════════════════════════
   DESIGN TOKENS — Palantir Gotham × NVIDIA Control Suite
══════════════════════════════════════════════════════════ */
:root {
  /* Core surfaces — near-black with subtle blue undertone */
  --void:       #020408;
  --base:       #060c14;
  --s0:         #080f1a;
  --s1:         #0b1420;
  --s2:         #0f1b2a;
  --s3:         #142234;
  --s4:         #1a2a3e;
  --s5:         #203248;

  /* Edge / border system */
  --e0:         rgba(255,255,255,0.035);
  --e1:         rgba(255,255,255,0.07);
  --e2:         rgba(255,255,255,0.12);
  --e3:         rgba(255,255,255,0.2);

  /* Primary accent — acid lime (safety/operational) */
  --lime:       #b8ff3c;
  --lime-soft:  #9de832;
  --lime-dim:   rgba(184,255,60,0.08);
  --lime-mid:   rgba(184,255,60,0.15);
  --lime-glow:  rgba(184,255,60,0.3);
  --lime-pulse: rgba(184,255,60,0.5);

  /* Status palette */
  --red:        #f03d3d;
  --red-dim:    rgba(240,61,61,0.1);
  --red-glow:   rgba(240,61,61,0.4);
  --amber:      #f0a020;
  --amber-dim:  rgba(240,160,32,0.1);
  --orange:     #f07020;
  --orange-dim: rgba(240,112,32,0.1);
  --sky:        #2cb8f0;
  --sky-dim:    rgba(44,184,240,0.08);
  --violet:     #8b5cf6;
  --violet-dim: rgba(139,92,246,0.1);

  /* Typography */
  --t0: #f0f4ff;   /* primary text */
  --t1: #c8d4e8;   /* secondary */
  --t2: #7a90b0;   /* muted */
  --t3: #3d5070;   /* dim */
  --t4: #1e3050;   /* very dim */

  /* Layout */
  --radius:    8px;
  --radius-lg: 14px;
  --nav-h:     56px;
  --ease:      cubic-bezier(0.4, 0, 0.2, 1);
  --spring:    cubic-bezier(0.34, 1.56, 0.64, 1);
}

/* ── RESET & BASE ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp { background: var(--base) !important; }
.stApp {
  font-family: 'Barlow', sans-serif;
  color: var(--t0);
  -webkit-font-smoothing: antialiased;
}
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
.stDeployButton, [data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
section[data-testid="stSidebar"],
button[kind="header"], [aria-label*="sidebar"] {
  display: none !important;
  visibility: hidden !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--s0); }
::-webkit-scrollbar-thumb { background: var(--s4); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--lime); }

/* ══════════════════════════════════════════════════════════
   TOP COMMAND BAR — Mission Control aesthetic
══════════════════════════════════════════════════════════ */
.cmd-bar {
  position: sticky;
  top: 0;
  z-index: 9999;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: var(--nav-h);
  padding: 0 20px 0 16px;
  background: linear-gradient(180deg,
    rgba(6,12,20,0.98) 0%,
    rgba(6,12,20,0.92) 100%);
  border-bottom: 1px solid var(--e1);
  backdrop-filter: blur(20px) saturate(180%);
  -webkit-backdrop-filter: blur(20px) saturate(180%);
}

.cmd-bar::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg,
    transparent 0%,
    var(--lime) 30%,
    var(--lime) 70%,
    transparent 100%);
  opacity: 0.25;
}

/* Logo cluster */
.cmd-logo {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}

.cmd-logo-shield {
  position: relative;
  width: 32px; height: 32px;
  display: flex; align-items: center; justify-content: center;
}

.cmd-logo-shield svg { width: 32px; height: 32px; }

.cmd-logo-wordmark {
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.cmd-logo-name {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.05rem;
  font-weight: 800;
  letter-spacing: 2px;
  color: var(--t0);
  text-transform: uppercase;
  line-height: 1;
}

.cmd-logo-name em {
  font-style: normal;
  color: var(--lime);
}

.cmd-logo-tagline {
  font-family: 'Azeret Mono', monospace;
  font-size: 0.48rem;
  letter-spacing: 2.5px;
  color: var(--t3);
  text-transform: uppercase;
  font-weight: 500;
}

/* Divider pip */
.cmd-sep {
  width: 1px;
  height: 28px;
  background: var(--e1);
  margin: 0 16px;
  flex-shrink: 0;
}

/* System health pills */
.cmd-health {
  display: flex;
  gap: 6px;
  align-items: center;
}

.cmd-health-pill {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 4px 10px;
  border-radius: 4px;
  background: var(--s2);
  border: 1px solid var(--e1);
  font-family: 'Azeret Mono', monospace;
  font-size: 0.55rem;
  font-weight: 600;
  letter-spacing: 1px;
  color: var(--t2);
  text-transform: uppercase;
  transition: all 0.2s var(--ease);
}

.cmd-health-pill.online {
  color: var(--lime);
  border-color: rgba(184,255,60,0.25);
  background: var(--lime-dim);
}

.cmd-health-pill.warn {
  color: var(--amber);
  border-color: rgba(240,160,32,0.25);
  background: var(--amber-dim);
}

.cmd-health-pill.crit {
  color: var(--red);
  border-color: rgba(240,61,61,0.3);
  background: var(--red-dim);
}

.cmd-dot {
  width: 5px; height: 5px;
  border-radius: 50%;
  background: currentColor;
  flex-shrink: 0;
}

.cmd-dot.pulse {
  animation: dot-pulse 2s infinite;
}

@keyframes dot-pulse {
  0%, 100% { opacity: 1; box-shadow: 0 0 0 0 currentColor; }
  50% { opacity: 0.7; box-shadow: 0 0 6px 2px rgba(184,255,60,0.2); }
}

/* Right cluster */
.cmd-right {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}

.cmd-clock {
  font-family: 'Azeret Mono', monospace;
  font-size: 0.72rem;
  color: var(--t2);
  letter-spacing: 1.5px;
  font-weight: 500;
}

.cmd-clock strong {
  color: var(--t0);
  font-weight: 600;
}

.cmd-version {
  font-family: 'Azeret Mono', monospace;
  font-size: 0.5rem;
  color: var(--t4);
  letter-spacing: 2px;
  text-transform: uppercase;
  padding: 3px 8px;
  border: 1px solid var(--e0);
  border-radius: 3px;
}

/* ══════════════════════════════════════════════════════════
   LAYOUT WRAPPER
══════════════════════════════════════════════════════════ */
.sg-layout {
  display: grid;
  grid-template-columns: 220px 1fr 280px;
  gap: 0;
  min-height: calc(100vh - var(--nav-h));
  background: var(--base);
}

/* ══════════════════════════════════════════════════════════
   LEFT PANEL — Control Deck
══════════════════════════════════════════════════════════ */
.panel-left {
  background: linear-gradient(180deg, var(--s0) 0%, var(--base) 100%);
  border-right: 1px solid var(--e1);
  padding: 0;
  overflow-y: auto;
}

.panel-section {
  padding: 16px;
  border-bottom: 1px solid var(--e0);
}

.panel-label {
  font-family: 'Azeret Mono', monospace;
  font-size: 0.5rem;
  font-weight: 700;
  letter-spacing: 3px;
  color: var(--t3);
  text-transform: uppercase;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.panel-label::before {
  content: '';
  display: inline-block;
  width: 12px;
  height: 1px;
  background: var(--lime);
  opacity: 0.6;
}

/* ══════════════════════════════════════════════════════════
   VIDEO FRAME — Cinematic Command Center
══════════════════════════════════════════════════════════ */
.video-shell {
  background: var(--void);
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--e1);
}

.video-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  background: linear-gradient(180deg, var(--s1) 0%, rgba(8,15,26,0.5) 100%);
  border-bottom: 1px solid var(--e1);
  flex-shrink: 0;
}

.video-title {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 3px;
  color: var(--t1);
  text-transform: uppercase;
}

.video-meta-chips {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}

.vmeta-chip {
  font-family: 'Azeret Mono', monospace;
  font-size: 0.52rem;
  font-weight: 600;
  letter-spacing: 1.5px;
  color: var(--t3);
  padding: 3px 8px;
  background: var(--s2);
  border: 1px solid var(--e0);
  border-radius: 3px;
  text-transform: uppercase;
}

.vmeta-chip span {
  color: var(--lime);
  font-weight: 700;
}

.vmeta-chip.live {
  color: var(--lime);
  border-color: rgba(184,255,60,0.2);
  background: var(--lime-dim);
}

.vmeta-chip.live .cmd-dot { animation: dot-pulse 1.4s infinite; }

.video-viewport {
  position: relative;
  flex: 1;
  background: var(--void);
  overflow: hidden;
}

/* Corner bracket decorators on video */
.video-viewport::before,
.video-viewport::after {
  content: '';
  position: absolute;
  z-index: 10;
  pointer-events: none;
}

.video-corner {
  position: absolute;
  width: 20px;
  height: 20px;
  border-color: var(--lime);
  border-style: solid;
  opacity: 0.5;
  z-index: 10;
  pointer-events: none;
  transition: opacity 0.3s;
}
.video-corner:hover { opacity: 1; }
.vc-tl { top: 8px; left: 8px; border-width: 2px 0 0 2px; }
.vc-tr { top: 8px; right: 8px; border-width: 2px 2px 0 0; }
.vc-bl { bottom: 8px; left: 8px; border-width: 0 0 2px 2px; }
.vc-br { bottom: 8px; right: 8px; border-width: 0 2px 2px 0; }

.video-scanline {
  position: absolute;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent 0px,
    transparent 3px,
    rgba(0,0,0,0.06) 3px,
    rgba(0,0,0,0.06) 4px
  );
  pointer-events: none;
  z-index: 5;
}

.video-bottombar {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 16px;
  background: linear-gradient(0deg, var(--s1) 0%, rgba(8,15,26,0.5) 100%);
  border-top: 1px solid var(--e1);
  flex-shrink: 0;
  flex-wrap: wrap;
}

.vbottom-chip {
  font-family: 'Azeret Mono', monospace;
  font-size: 0.52rem;
  font-weight: 600;
  letter-spacing: 1px;
  color: var(--t3);
  padding: 3px 8px;
  background: var(--s2);
  border: 1px solid var(--e0);
  border-radius: 3px;
  text-transform: uppercase;
  white-space: nowrap;
}

.vbottom-chip.accent {
  color: var(--lime);
  border-color: rgba(184,255,60,0.2);
  background: var(--lime-dim);
  font-weight: 700;
}

.vbottom-ts {
  margin-left: auto;
  font-family: 'Azeret Mono', monospace;
  font-size: 0.52rem;
  color: var(--t3);
}

/* ══════════════════════════════════════════════════════════
   INCIDENT LOG TABLE — Premium data grid
══════════════════════════════════════════════════════════ */
.log-shell {
  padding: 0 2px;
}

.log-section-head {
  font-family: 'Azeret Mono', monospace;
  font-size: 0.5rem;
  font-weight: 700;
  letter-spacing: 3px;
  color: var(--t3);
  text-transform: uppercase;
  padding: 10px 4px 8px;
  border-bottom: 1px solid var(--e0);
  margin-bottom: 8px;
}

.log-table {
  width: 100%;
  overflow: hidden;
  border: 1px solid var(--e1);
  border-radius: var(--radius);
}

.log-head-row {
  display: grid;
  grid-template-columns: 58px 52px 1fr 56px;
  gap: 6px;
  padding: 7px 10px;
  background: var(--s3);
  border-bottom: 1px solid var(--e1);
  font-family: 'Azeret Mono', monospace;
  font-size: 0.45rem;
  letter-spacing: 2px;
  color: var(--t3);
  text-transform: uppercase;
  font-weight: 700;
}

.log-data-row {
  display: grid;
  grid-template-columns: 58px 52px 1fr 56px;
  gap: 6px;
  align-items: center;
  padding: 8px 10px;
  border-bottom: 1px solid var(--e0);
  font-size: 0.65rem;
  transition: background 0.15s var(--ease), padding-left 0.2s var(--ease);
  cursor: default;
  position: relative;
}

.log-data-row::before {
  content: '';
  position: absolute;
  left: 0; top: 0;
  width: 2px; height: 100%;
  background: transparent;
  transition: background 0.2s;
}

.log-data-row:hover {
  background: var(--s2);
  padding-left: 14px;
}

.log-data-row.r-critical::before { background: var(--red); }
.log-data-row.r-high::before { background: var(--orange); }
.log-data-row.r-medium::before { background: var(--amber); }
.log-data-row.r-low::before { background: var(--lime); }

.log-data-row:last-child { border-bottom: none; }

.log-id {
  font-family: 'Azeret Mono', monospace;
  font-size: 0.55rem;
  color: var(--lime);
  font-weight: 700;
  letter-spacing: 0.5px;
}

.log-ts {
  font-family: 'Azeret Mono', monospace;
  font-size: 0.52rem;
  color: var(--t3);
  letter-spacing: 0.5px;
}

.log-msg {
  color: var(--t1);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-weight: 500;
  font-size: 0.65rem;
}

.log-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-family: 'Azeret Mono', monospace;
  font-size: 0.45rem;
  font-weight: 700;
  letter-spacing: 0.5px;
  padding: 2px 6px;
  border-radius: 3px;
  text-transform: uppercase;
}

.lb-CRITICAL { background: var(--red-dim); color: var(--red); border: 1px solid rgba(240,61,61,0.3); }
.lb-HIGH { background: var(--orange-dim); color: var(--orange); border: 1px solid rgba(240,112,32,0.3); }
.lb-MEDIUM { background: var(--amber-dim); color: var(--amber); border: 1px solid rgba(240,160,32,0.3); }
.lb-LOW { background: var(--lime-dim); color: var(--lime); border: 1px solid rgba(184,255,60,0.25); }

.log-empty {
  text-align: center;
  padding: 20px 12px;
  font-family: 'Azeret Mono', monospace;
  font-size: 0.58rem;
  letter-spacing: 1.5px;
  color: var(--t3);
  border: 1px dashed var(--e0);
  border-radius: var(--radius);
  text-transform: uppercase;
}

/* ══════════════════════════════════════════════════════════
   RIGHT PANEL — Analytics Stack
══════════════════════════════════════════════════════════ */
.panel-right {
  background: linear-gradient(180deg, var(--s0) 0%, var(--base) 100%);
  overflow-y: auto;
  padding: 0;
}

.analytics-section {
  padding: 14px 14px;
  border-bottom: 1px solid var(--e0);
}

.section-eyebrow {
  font-family: 'Azeret Mono', monospace;
  font-size: 0.48rem;
  font-weight: 700;
  letter-spacing: 3px;
  color: var(--t3);
  text-transform: uppercase;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.section-eyebrow .accent-line {
  display: inline-block;
  width: 16px;
  height: 1px;
  background: var(--lime);
  opacity: 0.5;
}

/* ── KPI CARDS ── */
.kpi-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.kpi-card {
  background: var(--s2);
  border: 1px solid var(--e1);
  border-radius: var(--radius);
  padding: 12px 12px 10px;
  position: relative;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.25s var(--ease);
}

.kpi-card::after {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: transparent;
  transition: background 0.25s var(--ease);
}

.kpi-card.c-lime::after { background: var(--lime); }
.kpi-card.c-red::after { background: var(--red); }
.kpi-card.c-sky::after { background: var(--sky); }
.kpi-card.c-amber::after { background: var(--amber); }
.kpi-card.c-violet::after { background: var(--violet); }

.kpi-card:hover {
  border-color: var(--e2);
  transform: translateY(-3px);
  box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

.kpi-label {
  font-family: 'Azeret Mono', monospace;
  font-size: 0.47rem;
  font-weight: 700;
  letter-spacing: 2px;
  color: var(--t3);
  text-transform: uppercase;
  margin-bottom: 8px;
}

.kpi-value {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 2rem;
  font-weight: 800;
  line-height: 1;
  letter-spacing: -0.5px;
  margin-bottom: 4px;
}

.kpi-sub {
  font-size: 0.58rem;
  color: var(--t3);
  font-weight: 400;
  letter-spacing: 0.3px;
}

.kpi-delta {
  font-family: 'Azeret Mono', monospace;
  font-size: 0.52rem;
  font-weight: 700;
  margin-top: 6px;
}
.kpi-delta.up { color: var(--lime); }
.kpi-delta.dn { color: var(--red); }

/* ── RISK GAUGE ── */
.risk-gauge-card {
  background: var(--s2);
  border: 1px solid var(--e1);
  border-radius: var(--radius);
  padding: 16px;
  position: relative;
  overflow: hidden;
}

.risk-gauge-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
}

.risk-level-badge {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 0.7rem;
  font-weight: 800;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  padding: 3px 10px;
  border-radius: 4px;
}

.rlb-LOW    { color: var(--lime);   background: var(--lime-dim);   border: 1px solid rgba(184,255,60,0.25); }
.rlb-MEDIUM { color: var(--amber);  background: var(--amber-dim);  border: 1px solid rgba(240,160,32,0.25); }
.rlb-HIGH   { color: var(--orange); background: var(--orange-dim); border: 1px solid rgba(240,112,32,0.25); }
.rlb-CRITICAL {
  color: var(--red);
  background: var(--red-dim);
  border: 1px solid rgba(240,61,61,0.3);
  animation: crit-badge-pulse 0.9s ease-in-out infinite;
}

@keyframes crit-badge-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(240,61,61,0); }
  50% { box-shadow: 0 0 12px 2px rgba(240,61,61,0.25); }
}

.risk-score-display {
  text-align: center;
  margin-bottom: 12px;
}

.risk-score-num {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 3.8rem;
  font-weight: 900;
  line-height: 1;
  letter-spacing: -2px;
}

.risk-score-unit {
  font-family: 'Azeret Mono', monospace;
  font-size: 0.55rem;
  color: var(--t3);
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-top: 2px;
}

/* Segmented progress bar */
.risk-track {
  height: 6px;
  background: var(--s3);
  border-radius: 3px;
  overflow: visible;
  position: relative;
  margin-bottom: 8px;
  border: 1px solid var(--e0);
}

.risk-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 0.8s var(--ease), background 0.5s var(--ease);
  box-shadow: 0 0 10px currentColor;
  position: relative;
}

.risk-fill::after {
  content: '';
  position: absolute;
  right: -1px; top: -2px;
  width: 3px; height: 10px;
  background: white;
  border-radius: 2px;
  opacity: 0.9;
  box-shadow: 0 0 6px rgba(255,255,255,0.6);
}

.risk-track-labels {
  display: flex;
  justify-content: space-between;
  margin-top: 6px;
}

.risk-track-label {
  font-family: 'Azeret Mono', monospace;
  font-size: 0.44rem;
  color: var(--t4);
  font-weight: 600;
}

/* Threshold markers */
.risk-segments {
  display: flex;
  gap: 2px;
  margin-bottom: 6px;
}

.risk-seg {
  height: 3px;
  border-radius: 2px;
  flex: 1;
}

/* ── SESSION STATS ROW ── */
.stat-row {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 6px;
}

.stat-mini {
  background: var(--s2);
  border: 1px solid var(--e0);
  border-radius: var(--radius);
  padding: 10px 8px;
  text-align: center;
  transition: border-color 0.2s;
}

.stat-mini:hover { border-color: var(--e2); }

.stat-mini-val {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.4rem;
  font-weight: 800;
  color: var(--t0);
  line-height: 1;
}

.stat-mini-lbl {
  font-family: 'Azeret Mono', monospace;
  font-size: 0.44rem;
  color: var(--t3);
  letter-spacing: 1.5px;
  text-transform: uppercase;
  margin-top: 3px;
  font-weight: 600;
}

/* ── ALERTS FEED ── */
.alert-feed {
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: 280px;
  overflow-y: auto;
  padding-right: 2px;
}

.alert-item {
  background: var(--s2);
  border: 1px solid var(--e0);
  border-radius: var(--radius);
  padding: 10px 12px;
  position: relative;
  overflow: hidden;
  transition: all 0.2s var(--ease);
  cursor: pointer;
}

.alert-item::before {
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 3px;
}

.alert-item.a-critical::before { background: var(--red); }
.alert-item.a-warning::before  { background: var(--amber); }
.alert-item.a-info::before     { background: var(--sky); }
.alert-item.a-safe::before     { background: var(--lime); }

.alert-item.a-critical { border-color: rgba(240,61,61,0.2); }
.alert-item.a-warning  { border-color: rgba(240,160,32,0.15); }
.alert-item.a-info     { border-color: rgba(44,184,240,0.12); }

.alert-item:hover {
  transform: translateX(3px);
  border-color: var(--e2);
  box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}

.alert-severity {
  font-family: 'Azeret Mono', monospace;
  font-size: 0.5rem;
  font-weight: 800;
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-bottom: 4px;
}

.a-critical .alert-severity { color: var(--red); }
.a-warning  .alert-severity { color: var(--amber); }
.a-info     .alert-severity { color: var(--sky); }
.a-safe     .alert-severity { color: var(--lime); }

.alert-body {
  font-size: 0.68rem;
  color: var(--t1);
  line-height: 1.5;
  font-weight: 500;
  letter-spacing: 0.2px;
}

.alert-footer {
  font-family: 'Azeret Mono', monospace;
  font-size: 0.5rem;
  color: var(--t3);
  margin-top: 6px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.alert-empty {
  text-align: center;
  padding: 22px 12px;
  font-family: 'Azeret Mono', monospace;
  font-size: 0.55rem;
  letter-spacing: 2px;
  color: var(--t3);
  border: 1px dashed var(--e0);
  border-radius: var(--radius);
  text-transform: uppercase;
  background: var(--lime-dim);
}

.alert-clear-icon {
  font-size: 1rem;
  margin-bottom: 6px;
  display: block;
  opacity: 0.5;
}

/* ── CHART WRAPPER ── */
.chart-card {
  background: var(--s2);
  border: 1px solid var(--e1);
  border-radius: var(--radius);
  padding: 12px;
  overflow: hidden;
}

/* ══════════════════════════════════════════════════════════
   STREAMLIT WIDGET OVERRIDES
══════════════════════════════════════════════════════════ */
div[data-testid="stMetric"] { display: none !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
.element-container { margin: 0 !important; padding: 0 !important; }

/* BUTTONS */
.stButton > button {
  width: 100%;
  background: var(--s3) !important;
  border: 1px solid var(--e1) !important;
  color: var(--t2) !important;
  border-radius: var(--radius) !important;
  font-family: 'Azeret Mono', monospace !important;
  font-size: 0.58rem !important;
  letter-spacing: 2px !important;
  font-weight: 700 !important;
  padding: 9px 14px !important;
  text-transform: uppercase !important;
  transition: all 0.2s var(--ease) !important;
}

.stButton > button:hover {
  background: var(--lime-mid) !important;
  border-color: var(--lime) !important;
  color: var(--lime) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 20px rgba(184,255,60,0.15) !important;
}

.start-btn .stButton > button {
  background: linear-gradient(135deg, var(--lime) 0%, #9de832 100%) !important;
  border-color: var(--lime) !important;
  color: #020408 !important;
  font-weight: 900 !important;
  font-size: 0.6rem !important;
  letter-spacing: 2.5px !important;
  box-shadow: 0 4px 20px var(--lime-glow) !important;
}

.start-btn .stButton > button:hover {
  background: linear-gradient(135deg, #ccff55 0%, var(--lime) 100%) !important;
  color: #020408 !important;
  box-shadow: 0 8px 30px var(--lime-pulse) !important;
  transform: translateY(-3px) !important;
}

/* SELECTBOX */
[data-testid="stSelectbox"] > div > div {
  background: var(--s2) !important;
  border: 1px solid var(--e1) !important;
  color: var(--t0) !important;
  font-family: 'Barlow', sans-serif !important;
  font-size: 0.78rem !important;
  border-radius: var(--radius) !important;
  transition: border-color 0.2s !important;
}

[data-testid="stSelectbox"] > div > div:hover { border-color: var(--lime) !important; }

[data-testid="stSelectbox"] label {
  color: var(--t3) !important;
  font-family: 'Azeret Mono', monospace !important;
  font-size: 0.5rem !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  font-weight: 700 !important;
}

/* SLIDERS */
.stSlider [data-testid="stWidgetLabel"] p {
  color: var(--t3) !important;
  font-family: 'Azeret Mono', monospace !important;
  font-size: 0.5rem !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  font-weight: 700 !important;
}

[data-baseweb="slider"] div[role="slider"] {
  background: var(--lime) !important;
  border-color: var(--lime) !important;
  box-shadow: 0 0 10px var(--lime-glow) !important;
}

[data-baseweb="slider"] [data-testid="stSliderThumb"] {
  background: var(--lime) !important;
}

/* TEXT INPUTS */
.stTextInput input {
  background: var(--s2) !important;
  border: 1px solid var(--e1) !important;
  color: var(--t0) !important;
  border-radius: var(--radius) !important;
  font-family: 'Barlow', sans-serif !important;
  font-size: 0.78rem !important;
  padding: 9px 11px !important;
  transition: all 0.2s !important;
}

.stTextInput input:focus {
  border-color: var(--lime) !important;
  box-shadow: 0 0 0 3px var(--lime-dim) !important;
}

.stTextInput label p {
  color: var(--t3) !important;
  font-family: 'Azeret Mono', monospace !important;
  font-size: 0.5rem !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  font-weight: 700 !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
  background: var(--s2) !important;
  border: 1px dashed var(--e2) !important;
  border-radius: var(--radius) !important;
  padding: 12px !important;
  transition: all 0.2s !important;
}

[data-testid="stFileUploader"]:hover {
  border-color: var(--lime) !important;
  background: var(--lime-dim) !important;
}

/* TOGGLE */
[data-baseweb="checkbox"] input:checked ~ div { background: var(--lime) !important; }

/* DIVIDER */
hr { border-color: var(--e0) !important; margin: 10px 0 !important; }

/* SUCCESS/INFO messages */
[data-testid="stSuccess"] {
  background: var(--lime-dim) !important;
  border: 1px solid rgba(184,255,60,0.2) !important;
  border-radius: var(--radius) !important;
  color: var(--lime) !important;
  font-family: 'Azeret Mono', monospace !important;
  font-size: 0.6rem !important;
}

[data-testid="stInfo"] {
  background: var(--sky-dim) !important;
  border: 1px solid rgba(44,184,240,0.2) !important;
  border-radius: var(--radius) !important;
  color: var(--sky) !important;
  font-family: 'Azeret Mono', monospace !important;
  font-size: 0.6rem !important;
}

/* STREAMLIT CHART */
[data-testid="stArrowVegaLiteChart"],
.vega-embed { background: transparent !important; }

/* Streamlit line chart override */
[data-testid="stLineChartContainer"] {
  background: transparent !important;
}

/* Padding for left panel content */
.sp-inner { padding: 14px 14px 10px; }

/* Footer branding strip */
.sg-footer {
  padding: 14px 14px 18px;
  margin-top: auto;
}

.sg-footer-inner {
  padding: 10px 12px;
  border: 1px solid var(--e0);
  border-radius: var(--radius);
  background: var(--s2);
}

.sg-footer-brand {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 2px;
  color: var(--t3);
  text-transform: uppercase;
}

.sg-footer-sub {
  font-family: 'Azeret Mono', monospace;
  font-size: 0.45rem;
  color: var(--t4);
  letter-spacing: 1.5px;
  text-transform: uppercase;
  margin-top: 4px;
  line-height: 1.8;
}

.sg-footer-badge {
  display: inline-block;
  margin-top: 6px;
  padding: 2px 8px;
  background: var(--lime-dim);
  border: 1px solid rgba(184,255,60,0.2);
  border-radius: 3px;
  font-family: 'Azeret Mono', monospace;
  font-size: 0.44rem;
  font-weight: 700;
  letter-spacing: 1.5px;
  color: var(--lime);
  text-transform: uppercase;
}

/* ══════════════════════════════════════════════════════════
   BUTTON COLUMN CONTROL PANEL
══════════════════════════════════════════════════════════ */
.ctrl-section {
  padding: 10px 14px 12px;
  border-bottom: 1px solid var(--e0);
}

.ctrl-btn-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
  margin-top: 6px;
}

/* ══════════════════════════════════════════════════════════
   RESPONSIVE / UTILITY
══════════════════════════════════════════════════════════ */
.mt4  { margin-top: 4px;  }
.mt8  { margin-top: 8px;  }
.mt12 { margin-top: 12px; }
.mt16 { margin-top: 16px; }
.mb4  { margin-bottom: 4px; }
.mb8  { margin-bottom: 8px; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE  (unchanged — backend logic preserved)
# ─────────────────────────────────────────────────────────────────────────────
defaults = dict(
    running=False,
    risk_history=[],
    incidents=[],
    total_violations=0,
    alerts=[],
    _fps=0.0,
    site_name="Steel Plant Alpha — Unit 3",
    zone_name="Blast Furnace — Zone B",
    supervisor="Eng. Tariq Mahmood",
    max_risk=0,
    avg_risk=0,
    session_start=datetime.now(),
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCES  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_detector():  return SafetyDetector()
@st.cache_resource
def get_alert_mgr(): return AlertManager()
@st.cache_resource
def get_logger():    return IncidentLogger()

detector  = get_detector()
alert_mgr = get_alert_mgr()
logger    = get_logger()

# ─────────────────────────────────────────────────────────────────────────────
# TOP COMMAND BAR
# ─────────────────────────────────────────────────────────────────────────────
now_str   = datetime.now().strftime("%d %b %Y")
time_str  = datetime.now().strftime("%H:%M:%S")
is_active = st.session_state.running

engine_status = "online" if is_active else "warn"
stream_status = "online" if is_active else "warn"
status_label  = "ACTIVE" if is_active else "STANDBY"

nav_col1, nav_col2, nav_col3 = st.columns([1.0, 2.8, 1.4], gap="small")

with nav_col1:
    st.markdown(f"""
    <div class="cmd-bar" style="position:relative;z-index:9999;">
      <div class="cmd-logo">
        <div class="cmd-logo-shield">
          <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M16 2L4 7v8c0 7 5.4 13.5 12 15.5C22.6 28.5 28 22 28 15V7L16 2z"
              fill="rgba(184,255,60,0.12)" stroke="#b8ff3c" stroke-width="1.5"/>
            <path d="M12 16l3 3 5-5" stroke="#b8ff3c" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </div>
        <div class="cmd-logo-wordmark">
          <div class="cmd-logo-name">Safe<em>Guard</em> <em>AI</em></div>
          <div class="cmd-logo-tagline">Industrial Safety Platform</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with nav_col2:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;height:56px;">
      <div class="cmd-health-pill {engine_status}">
        <span class="cmd-dot {'pulse' if is_active else ''}"></span>
        AI ENGINE
      </div>
      <div class="cmd-health-pill {stream_status}">
        <span class="cmd-dot {'pulse' if is_active else ''}"></span>
        STREAM
      </div>
      <div class="cmd-health-pill online">
        <span class="cmd-dot"></span>
        YOLOv8n
      </div>
      <div class="cmd-sep" style="height:28px;width:1px;background:var(--e1);margin:0 4px;"></div>
    </div>
    """, unsafe_allow_html=True)

with nav_col3:
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:flex-end;gap:10px;height:56px;">
      <div class="cmd-version">v4.0</div>
      <div class="cmd-clock">{now_str} <strong>{time_str}</strong></div>
      <div class="cmd-health-pill {'online' if is_active else 'warn'}">
        <span class="cmd-dot {'pulse' if is_active else ''}"></span>
        {status_label}
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div style="height:1px;background:var(--e1);margin:0;"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3-COLUMN MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_center, col_right = st.columns([1.05, 3.3, 1.5], gap="small")

# ═════════════════════════════════════════════════════════════════════════════
#  LEFT PANEL — CONTROL DECK
# ═════════════════════════════════════════════════════════════════════════════
with col_left:

    # ── Transport Controls ──────────────────────────────────────────
    st.markdown("""
    <div class="ctrl-section">
      <div class="panel-label">Transport</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ctrl-section" style="border-bottom:none;padding-bottom:4px;">', unsafe_allow_html=True)
    st.markdown('<div class="start-btn">', unsafe_allow_html=True)
    if st.button("▶️  START MONITORING", key="start_btn", use_container_width=True):
        st.session_state.running = True
        st.session_state.risk_history = []
        st.session_state.incidents = []
        st.session_state.total_violations = 0
        st.session_state.alerts = []
        st.session_state.session_start = datetime.now()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        if st.button("⏸️", key="pause_btn", use_container_width=True):
            st.session_state.running = False
            st.rerun()
    with c2:
        if st.button("⏹️", key="stop_btn", use_container_width=True):
            st.session_state.running = False
            st.rerun()
    with c3:
        if st.button("↺", key="reset_btn", use_container_width=True):
            st.session_state.running = False
            st.session_state.risk_history = []
            st.session_state.incidents = []
            st.session_state.total_violations = 0
            st.session_state.alerts = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ── Input Source ─────────────────────────────────────────────────
    st.markdown('<div class="sp-inner">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Input Source</div>', unsafe_allow_html=True)
    source_mode = st.selectbox("Mode", ["Demo Mode", "Upload Video", "Webcam (Live)"], label_visibility="collapsed")
    uploaded_file = None
    if source_mode == "Upload Video":
        uploaded_file = st.file_uploader("Video file", type=["mp4","avi","mov","mkv"], label_visibility="collapsed")
        if uploaded_file:
            st.success(f"✓ {uploaded_file.name}")
    elif source_mode == "Webcam (Live)":
        st.info("📷 Webcam activates on start")
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ── Detection Settings ───────────────────────────────────────────
    st.markdown('<div class="sp-inner">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Detection Engine</div>', unsafe_allow_html=True)
    conf_thresh  = st.slider("Confidence Threshold", 0.3, 0.95, 0.5, 0.05, label_visibility="collapsed")
    alert_thresh = st.slider("Alert Threshold (%)",  20,   95,  40,    5, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ── Display Options ───────────────────────────────────────────────
    st.markdown('<div class="sp-inner">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Overlay Layers</div>', unsafe_allow_html=True)
    show_boxes  = st.toggle("Bounding Boxes",     True)
    show_labels = st.toggle("Object Labels",      True)
    show_scores = st.toggle("Confidence Scores", False)
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ── Site Config ───────────────────────────────────────────────────
    st.markdown('<div class="sp-inner">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Site Configuration</div>', unsafe_allow_html=True)
    st.session_state.site_name   = st.text_input("Facility",   st.session_state.site_name,   label_visibility="collapsed")
    st.session_state.zone_name   = st.text_input("Zone",       st.session_state.zone_name,   label_visibility="collapsed")
    st.session_state.supervisor  = st.text_input("Supervisor", st.session_state.supervisor,  label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer Branding ───────────────────────────────────────────────
    st.markdown("""
    <div class="sg-footer">
      <div class="sg-footer-inner">
        <div class="sg-footer-brand">Softnity Tech</div>
        <div class="sg-footer-sub">
          Enterprise Safety Division<br>
          SafeGuard AI Platform v4.0<br>
          © 2025 Softnity Technologies
        </div>
        <div class="sg-footer-badge">ISO 45001 Compliant</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  CENTER — VIDEO FEED (cinematic) + INCIDENT LOG
# ═════════════════════════════════════════════════════════════════════════════
with col_center:
    fps_val = st.session_state.get("_fps", 0.0)

    # ── Video Top Bar ──────────────────────────────────────────────────
    st.markdown(f"""
    <div class="video-topbar">
      <div class="video-title">Live Detection Stream</div>
      <div class="video-meta-chips">
        <div class="vmeta-chip live">
          <span class="cmd-dot {'pulse' if is_active else ''}"></span>
          {'LIVE' if is_active else 'OFFLINE'}
        </div>
        <div class="vmeta-chip">ZONE <span>{st.session_state.zone_name[:16]}</span></div>
        <div class="vmeta-chip">FPS <span>{fps_val:.1f}</span></div>
        <div class="vmeta-chip">ENGINE <span>YOLOv8n</span></div>
        <div class="vmeta-chip">CONF <span>{conf_thresh:.2f}</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Video Viewport with corner brackets ───────────────────────────
    st.markdown("""
    <div style="position:relative;">
      <div class="video-corner vc-tl"></div>
      <div class="video-corner vc-tr"></div>
      <div class="video-corner vc-bl"></div>
      <div class="video-corner vc-br"></div>
      <div class="video-scanline"></div>
    </div>
    """, unsafe_allow_html=True)

    video_placeholder = st.empty()

    # ── Video Bottom Bar ───────────────────────────────────────────────
    st.markdown(f"""
    <div class="video-bottombar">
      <div class="vbottom-chip accent">● STREAMING</div>
      <div class="vbottom-chip">{st.session_state.site_name[:24]}</div>
      <div class="vbottom-chip">1920 × 1080</div>
      <div class="vbottom-chip">H.264</div>
      <div class="vbottom-ts">{datetime.now().strftime('%H:%M:%S UTC+5')}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)

    # ── Incident Log ───────────────────────────────────────────────────
    st.markdown("""
    <div class="log-section-head">
      <span style="color:var(--lime);margin-right:6px;">▪️</span>
      Incident History Log
    </div>
    """, unsafe_allow_html=True)

    log_placeholder = st.empty()


# ═════════════════════════════════════════════════════════════════════════════
#  RIGHT PANEL — ANALYTICS STACK
# ═════════════════════════════════════════════════════════════════════════════
with col_right:

    # ── Live Metrics KPI Grid ──────────────────────────────────────────
    st.markdown("""
    <div class="analytics-section">
      <div class="section-eyebrow"><span class="accent-line"></span>Live Metrics</div>
    </div>
    """, unsafe_allow_html=True)
    metric_placeholder = st.empty()

    # ── Session Stats ─────────────────────────────────────────────────
    if st.session_state.risk_history:
        max_r = max(r[1] for r in st.session_state.risk_history)
        avg_r = sum(r[1] for r in st.session_state.risk_history) / len(st.session_state.risk_history)
        dur_m = int((datetime.now() - st.session_state.session_start).total_seconds() / 60)
        st.markdown(f"""
        <div class="analytics-section">
          <div class="section-eyebrow"><span class="accent-line"></span>Session Stats</div>
          <div class="stat-row">
            <div class="stat-mini">
              <div class="stat-mini-val" style="color:var(--red);">{max_r:.0f}</div>
              <div class="stat-mini-lbl">Peak</div>
            </div>
            <div class="stat-mini">
              <div class="stat-mini-val" style="color:var(--amber);">{avg_r:.0f}</div>
              <div class="stat-mini-lbl">Avg</div>
            </div>
            <div class="stat-mini">
              <div class="stat-mini-val" style="color:var(--sky);">{dur_m}m</div>
              <div class="stat-mini-lbl">Uptime</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Risk Score ────────────────────────────────────────────────────
    st.markdown("""
    <div class="analytics-section">
      <div class="section-eyebrow"><span class="accent-line"></span>Risk Assessment</div>
    </div>
    """, unsafe_allow_html=True)
    risk_placeholder = st.empty()

    # ── Active Alerts ─────────────────────────────────────────────────
    st.markdown("""
    <div class="analytics-section" style="padding-bottom:0;">
      <div class="section-eyebrow"><span class="accent-line"></span>Active Alerts</div>
    </div>
    """, unsafe_allow_html=True)
    alert_placeholder = st.empty()
    st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)

    # ── Risk Trend Chart ──────────────────────────────────────────────
    st.markdown("""
    <div class="analytics-section">
      <div class="section-eyebrow"><span class="accent-line"></span>Risk Trend</div>
    </div>
    """, unsafe_allow_html=True)
    chart_placeholder = st.empty()


# ─────────────────────────────────────────────────────────────────────────────
# RENDER FUNCTIONS (UI only — no backend changes)
# ─────────────────────────────────────────────────────────────────────────────

def render_metrics(workers, violations):
    viol_color = "var(--red)" if violations > 0 else "var(--lime)"
    viol_class = "c-red" if violations > 0 else "c-lime"
    delta_class = "dn" if violations > 0 else "up"
    delta_sym   = "↑" if violations > 0 else "✓"

    metric_placeholder.markdown(f"""
    <div class="analytics-section" style="padding-top:8px;">
      <div class="kpi-grid">
        <div class="kpi-card c-sky">
          <div class="kpi-label">Workers</div>
          <div class="kpi-value" style="color:var(--sky);">{workers}</div>
          <div class="kpi-sub">In frame</div>
        </div>
        <div class="kpi-card {viol_class}">
          <div class="kpi-label">Violations</div>
          <div class="kpi-value" style="color:{viol_color};">{violations}</div>
          <div class="kpi-sub">This session</div>
          <div class="kpi-delta {delta_class}">{delta_sym} {abs(violations)}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_risk(risk_score, risk_level):
    color_map = {
        "LOW":      "var(--lime)",
        "MEDIUM":   "var(--amber)",
        "HIGH":     "var(--orange)",
        "CRITICAL": "var(--red)",
    }
    fill = color_map.get(risk_level, "var(--lime)")

    # Threshold segment colors
    segs = [
        ("#1a3a1a", "#1a3a1a", "#3a2a10", "#3a1010"),  # bg all
    ]
    seg_colors = [
        f"background:{'var(--lime)' if risk_score > 0 else 'var(--s3)'}",
        f"background:{'var(--amber)' if risk_score > 35 else 'var(--s3)'}",
        f"background:{'var(--orange)' if risk_score > 65 else 'var(--s3)'}",
        f"background:{'var(--red)' if risk_score > 85 else 'var(--s3)'}",
    ]

    risk_placeholder.markdown(f"""
    <div class="analytics-section">
      <div class="risk-gauge-card">
        <div class="risk-gauge-header">
          <span style="font-family:'Azeret Mono',monospace;font-size:0.48rem;font-weight:700;
            letter-spacing:2.5px;color:var(--t3);text-transform:uppercase;">Overall Risk</span>
          <span class="risk-level-badge rlb-{risk_level}">{risk_level}</span>
        </div>
        <div class="risk-score-display">
          <div class="risk-score-num" style="color:{fill};">{risk_score}</div>
          <div class="risk-score-unit">/ 100 — RISK INDEX</div>
        </div>
        <div class="risk-segments">
          <div class="risk-seg" style="{seg_colors[0]};flex:35;"></div>
          <div class="risk-seg" style="{seg_colors[1]};flex:30;"></div>
          <div class="risk-seg" style="{seg_colors[2]};flex:20;"></div>
          <div class="risk-seg" style="{seg_colors[3]};flex:15;"></div>
        </div>
        <div class="risk-track">
          <div class="risk-fill" style="width:{risk_score}%;background:{fill};color:{fill};"></div>
        </div>
        <div class="risk-track-labels">
          <span class="risk-track-label">0 — SAFE</span>
          <span class="risk-track-label">50</span>
          <span class="risk-track-label">100 — CRITICAL</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_alerts(alerts):
    if not alerts:
        alert_placeholder.markdown("""
        <div class="analytics-section" style="padding-top:6px;">
          <div class="alert-empty">
            <span class="alert-clear-icon">✓</span>
            All Clear — No Active Alerts
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    type_map = {
        "red":    ("a-critical", "🚨  Critical"),
        "orange": ("a-critical", "🚨  Critical"),
        "amber":  ("a-warning",  "⚠️  Warning"),
        "blue":   ("a-info",     "ℹ️  Info"),
    }

    html = '<div class="analytics-section" style="padding-top:6px;padding-bottom:6px;"><div class="alert-feed">'
    for a in reversed(alerts[-5:]):
        css, label = type_map.get(a.get("color", "red"), ("a-critical", "🚨  Alert"))
        ts  = a.get("timestamp", "")
        dst = a.get("sent_to", "")
        html += f"""
        <div class="alert-item {css}">
          <div class="alert-severity">{label}</div>
          <div class="alert-body">{a.get('message', '')}</div>
          <div class="alert-footer">
            <span>{ts}</span>
            <span>{dst}</span>
          </div>
        </div>"""
    html += '</div></div>'
    alert_placeholder.markdown(html, unsafe_allow_html=True)


def render_chart(history):
    if len(history) < 2:
        chart_placeholder.markdown("""
        <div class="analytics-section">
          <div style="font-family:'Azeret Mono',monospace;font-size:0.52rem;
            color:var(--t3);padding:20px 10px;text-align:center;
            border:1px dashed var(--e0);border-radius:8px;letter-spacing:1.5px;
            text-transform:uppercase;">Awaiting data stream...</div>
        </div>
        """, unsafe_allow_html=True)
        return

    df = pd.DataFrame(history, columns=["time", "score"])
    with chart_placeholder.container():
        st.markdown('<div class="analytics-section" style="padding-bottom:8px;">', unsafe_allow_html=True)
        st.line_chart(
            df.set_index("time")["score"],
            height=130,
            use_container_width=True,
            color="#b8ff3c",
        )
        st.markdown('</div>', unsafe_allow_html=True)


def render_log(incidents):
    if not incidents:
        log_placeholder.markdown("""
        <div class="log-empty">No incidents logged this session.</div>
        """, unsafe_allow_html=True)
        return

    level_class = {
        "CRITICAL": "r-critical",
        "HIGH":     "r-high",
        "MEDIUM":   "r-medium",
        "LOW":      "r-low",
    }

    html = '<div class="log-table">'
    html += """
    <div class="log-head-row">
      <span>Incident ID</span>
      <span>Time</span>
      <span>Details</span>
      <span>Severity</span>
    </div>"""

    for inc in reversed(incidents[-8:]):
        if inc is None:
            continue
        level    = inc.get("risk_level", "LOW")
        ts_short = inc.get("timestamp", "")[-8:]
        msg      = f"{inc.get('violations', '')} · {inc.get('action', '')}"
        row_cls  = level_class.get(level, "r-low")

        html += f"""
        <div class="log-data-row {row_cls}">
          <span class="log-id">#{inc.get('id', '—')}</span>
          <span class="log-ts">{ts_short}</span>
          <span class="log-msg">{msg[:34]}</span>
          <span class="log-badge lb-{level}">{level[0]}</span>
        </div>"""

    html += '</div>'
    log_placeholder.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION LOOP  (backend unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def run_detection_loop():
    cap = None
    demo_mode = (source_mode == "Demo Mode")

    if not demo_mode:
        if source_mode == "Upload Video" and uploaded_file is not None:
            import tempfile
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.flush()
            tfile.close()
            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                st.error("Could not open video")
                st.session_state.running = False
                return
        elif source_mode == "Webcam (Live)":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.warning("Webcam not found")
                demo_mode = True

    prev_time = time.time()

    while st.session_state.running:
        if demo_mode:
            frame = np.zeros((540, 960, 3), dtype=np.uint8)
            frame[:] = (8, 15, 26)
        else:
            ret, frame = cap.read()
            if not ret:
                if cap:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((540, 960, 3), dtype=np.uint8)

        results = detector.detect(frame, conf=conf_thresh)
        annotated = detector.annotate(
            frame.copy(), results,
            show_boxes=show_boxes,
            show_labels=show_labels,
            show_scores=show_scores,
        )

        risk_score, risk_level, violations_frame = RiskCalculator.calculate(results)
        workers = sum(1 for d in results if d.cls == "person")

        st.session_state.total_violations += violations_frame
        ts = datetime.now().strftime("%H:%M:%S")
        st.session_state.risk_history.append((ts, risk_score))
        if len(st.session_state.risk_history) > 120:
            st.session_state.risk_history.pop(0)

        new_alerts = alert_mgr.check_and_fire(results)
        for a in new_alerts:
            st.session_state.alerts.append(a)
            if risk_score >= alert_thresh:
                incident = logger.log(results, zone=st.session_state.zone_name)
                if incident:
                    st.session_state.incidents.append(incident)

        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 0.001)
        prev_time = curr_time
        st.session_state["_fps"] = fps

        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        render_metrics(workers, st.session_state.total_violations)
        render_risk(risk_score, risk_level)
        render_alerts(st.session_state.alerts)
        render_chart(st.session_state.risk_history)
        render_log(st.session_state.incidents)

        time.sleep(0.02)

    if cap:
        cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.running:
    splash = detector.generate_splash_frame()
    video_placeholder.image(
        cv2.cvtColor(splash, cv2.COLOR_BGR2RGB),
        use_container_width=True,
    )
    render_metrics(0, 0)
    render_risk(0, "LOW")
    render_alerts([])
    render_chart([])
    render_log([])
else:
    run_detection_loop()