"""
utils.py – SafeGuard AI Utilities
Risk scoring, alert management, incident logging
"""

import random
from datetime import datetime


# ─── Risk Calculator ──────────────────────────────────────────────────────────
class RiskCalculator:
    """
    Converts raw detections into a business-meaningful risk score.

    Scoring model:
      no_helmet  → +25 per instance (critical PPE)
      no_vest    → +15 per instance
      fire       → +50 per instance
      smoke      → +30 per instance
      fall       → +40 per instance
    Clamped to 100, then mapped to level.
    """

    WEIGHTS = {
        "no_helmet": 25,
        "no_vest":   15,
        "fire":      50,
        "smoke":     30,
        "fall":      40,
    }

    THRESHOLDS = [
        (75, "CRITICAL"),
        (50, "HIGH"),
        (25, "MEDIUM"),
        (0,  "LOW"),
    ]

    @classmethod
    def calculate(cls, results: dict):
        score      = 0
        violations = 0

        for det in results.get("detections", []):
            w = cls.WEIGHTS.get(det["class"], 0)
            if w > 0:
                score      += w
                violations += 1

        score = min(score, 100)

        level = "LOW"
        for threshold, lbl in cls.THRESHOLDS:
            if score >= threshold:
                level = lbl
                break

        return score, level, violations


# ─── Alert Manager ────────────────────────────────────────────────────────────
class AlertManager:
    """
    Generates contextual, deduplicated alerts and notification messages.
    Simulates supervisor notification pipeline.
    """

    COOLDOWN_FRAMES = 30  # Prevent alert spam

    def __init__(self):
        self._cooldowns = {}

    def process(self, results: dict, risk_level: str, supervisor: str, site: str) -> list:
        now        = datetime.now()
        ts         = now.strftime("%H:%M:%S")
        new_alerts = []

        detected_classes = {d["class"] for d in results.get("detections", [])}

        # PPE violation
        if "no_helmet" in detected_classes and self._can_fire("no_helmet"):
            new_alerts.append({
                "icon":  "⚠️",
                "msg":   "Helmet violation detected – worker exposed",
                "level": "HIGH",
                "time":  ts,
                "log":   True,
                "notify": f"📧 Alert dispatched → {supervisor}",
                "type":  "PPE_HELMET",
            })

        if "no_vest" in detected_classes and self._can_fire("no_vest"):
            new_alerts.append({
                "icon":  "🦺",
                "msg":   "Safety vest missing – PPE non-compliance",
                "level": "MEDIUM",
                "time":  ts,
                "log":   True,
                "notify": f"📧 Alert dispatched → {supervisor}",
                "type":  "PPE_VEST",
            })

        if "fire" in detected_classes and self._can_fire("fire"):
            new_alerts.append({
                "icon":  "🔥",
                "msg":   f"FIRE DETECTED at {site} – Initiating emergency protocol",
                "level": "HIGH",
                "time":  ts,
                "log":   True,
                "notify": f"🚨 Emergency SMS → {supervisor} + Safety Team",
                "type":  "FIRE",
            })

        if "smoke" in detected_classes and self._can_fire("smoke"):
            new_alerts.append({
                "icon":  "💨",
                "msg":   "Smoke detected – possible combustion hazard",
                "level": "HIGH",
                "time":  ts,
                "log":   True,
                "notify": f"📧 Alert dispatched → {supervisor}",
                "type":  "SMOKE",
            })

        if "fall" in detected_classes and self._can_fire("fall"):
            new_alerts.append({
                "icon":  "🆘",
                "msg":   "Fall detected – worker may be injured",
                "level": "HIGH",
                "time":  ts,
                "log":   True,
                "notify": f"🚨 Emergency alert → {supervisor} + Medical",
                "type":  "FALL",
            })

        if risk_level == "CRITICAL" and self._can_fire("critical"):
            new_alerts.append({
                "icon":  "🚨",
                "msg":   "CRITICAL RISK ZONE – All non-essential personnel evacuate",
                "level": "HIGH",
                "time":  ts,
                "log":   True,
                "notify": f"🚨 CRITICAL: All channels notified",
                "type":  "CRITICAL",
            })

        # Tick cooldowns
        for key in list(self._cooldowns.keys()):
            self._cooldowns[key] -= 1
            if self._cooldowns[key] <= 0:
                del self._cooldowns[key]

        return new_alerts

    def _can_fire(self, key: str) -> bool:
        if key in self._cooldowns:
            return False
        self._cooldowns[key] = self.COOLDOWN_FRAMES
        return True


# ─── Incident Logger ──────────────────────────────────────────────────────────
class IncidentLogger:
    """
    Generates structured incident log entries.
    Simulates enterprise ITSM/HSE system logging.
    """

    INCIDENT_IDS = iter(range(10000, 99999))

    _SEVERITY_MAP = {
        "PPE_HELMET": "HIGH",
        "PPE_VEST":   "MEDIUM",
        "FIRE":       "HIGH",
        "SMOKE":      "HIGH",
        "FALL":       "HIGH",
        "CRITICAL":   "HIGH",
    }

    _SYSTEM_MSGS = {
        "PPE_HELMET": [
            "Incident logged – HSE database updated",
            "Compliance report flag raised",
            "Supervisor notification confirmed",
        ],
        "PPE_VEST": [
            "PPE violation recorded",
            "Worker ID tagged for coaching session",
            "Incident logged – HSE database updated",
        ],
        "FIRE": [
            "🔥 Fire incident logged – Emergency protocol initiated",
            "Fire suppression system notified",
            "Emergency response team alerted",
        ],
        "SMOKE": [
            "Smoke event logged – HVAC isolation triggered",
            "Ventilation system alert raised",
        ],
        "FALL": [
            "🆘 Fall incident logged – Medical response requested",
            "Worker welfare check initiated",
        ],
        "CRITICAL": [
            "🚨 Critical risk event logged",
            "Zone lockdown protocol considered",
        ],
    }

    def log(self, alert: dict, site: str, zone: str) -> dict:
        inc_id   = f"INC-{next(self.INCIDENT_IDS)}"
        ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        itype    = alert.get("type", "GENERAL")
        severity = self._SEVERITY_MAP.get(itype, "MEDIUM")

        sys_msgs = self._SYSTEM_MSGS.get(itype, ["Incident logged."])
        sys_msg  = random.choice(sys_msgs)

        return {
            "id":        inc_id,
            "timestamp": ts,
            "site":      site,
            "zone":      zone,
            "type":      itype,
            "severity":  severity,
            "message":   f"[{inc_id}] {alert['msg']} | {sys_msg}",
        }
