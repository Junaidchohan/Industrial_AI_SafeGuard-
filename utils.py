"""
utils.py - SafeGuard AI Risk Engine, Alert Manager, Incident Logger
Fixed: accepts list[Detection] OR dict from detector
Fixed: compliant workers (helmet+vest) produce ZERO risk
"""

import random
import time
from datetime import datetime
import pandas as pd


# ---------------------------------------------------------------------------
# Risk weights - ONLY violations add risk, safe classes add nothing
# ---------------------------------------------------------------------------
RISK_WEIGHTS = {
    "no_helmet": 25,
    "no_vest":   15,
    "fire":      50,
    "smoke":     35,
    "fall":      40,
    # These classes are SAFE - zero risk contribution
    "person":    0,
    "helmet":    0,
    "vest":      0,
}


class RiskCalculator:
    """
    Accepts EITHER:
      - list of Detection objects  (from detector.detect())
      - dict with 'detections' key (from detector.get_risk_summary())
    """

    @staticmethod
    def calculate(results):
        """
        Returns: (risk_score: int, risk_level: str, violation_count: int)
        """
        # --- Normalise input to a flat list of class names -----------------
        class_names = []

        if isinstance(results, list):
            # List of Detection objects
            for det in results:
                if hasattr(det, "cls"):
                    class_names.append(det.cls)
                elif isinstance(det, dict):
                    class_names.append(det.get("class", ""))

        elif isinstance(results, dict):
            # Dict format from get_risk_summary()
            for det in results.get("detections", []):
                if isinstance(det, dict):
                    class_names.append(det.get("class", ""))
                elif hasattr(det, "cls"):
                    class_names.append(det.cls)
        else:
            return 0, "LOW", 0

        # --- Count only violation classes -----------------------------------
        VIOLATION_CLASSES = {"no_helmet", "no_vest", "fire", "smoke", "fall"}
        violation_classes_found = [c for c in class_names if c in VIOLATION_CLASSES]
        violation_count = len(violation_classes_found)

        # --- Score: sum risk weights, clamp to 100 --------------------------
        raw_score = sum(RISK_WEIGHTS.get(c, 0) for c in violation_classes_found)
        risk_score = min(100, raw_score)

        # --- Level -----------------------------------------------------------
        if   risk_score >= 80: risk_level = "CRITICAL"
        elif risk_score >= 50: risk_level = "HIGH"
        elif risk_score >= 25: risk_level = "MEDIUM"
        else:                  risk_level = "LOW"

        return risk_score, risk_level, violation_count

    @staticmethod
    def get_violation_details(results):
        """Return human-readable violation list for the alert panel."""
        class_names = RiskCalculator._extract_classes(results)
        details = []
        if "no_helmet" in class_names:
            details.append(("NO HELMET",  "red",    "Helmet missing - PPE violation"))
        if "no_vest"   in class_names:
            details.append(("NO VEST",    "orange", "Safety vest missing"))
        if "fire"      in class_names:
            details.append(("FIRE",       "red",    "Fire detected - Emergency protocol"))
        if "smoke"     in class_names:
            details.append(("SMOKE",      "orange", "Smoke detected - Ventilation alert"))
        if "fall"      in class_names:
            details.append(("FALL",       "red",    "Worker fall - Medical team notified"))
        return details

    @staticmethod
    def _extract_classes(results):
        classes = []
        if isinstance(results, list):
            for det in results:
                if hasattr(det, "cls"):
                    classes.append(det.cls)
                elif isinstance(det, dict):
                    classes.append(det.get("class", ""))
        elif isinstance(results, dict):
            for det in results.get("detections", []):
                if isinstance(det, dict):
                    classes.append(det.get("class", ""))
                elif hasattr(det, "cls"):
                    classes.append(det.cls)
        return classes

    @staticmethod
    def count_workers(results):
        classes = RiskCalculator._extract_classes(results)
        return classes.count("person")


# ---------------------------------------------------------------------------
# Alert Manager
# ---------------------------------------------------------------------------
class AlertManager:
    def __init__(self, cooldown_seconds: int = 30):
        self.cooldown   = cooldown_seconds
        self.last_alert = {}   # alert_type -> last timestamp
        self.alert_log  = []   # full history

    def check_and_fire(self, results) -> list:
        """
        Returns list of new alert dicts that passed cooldown.
        """
        violation_details = RiskCalculator.get_violation_details(results)
        new_alerts = []

        for key, color, message in violation_details:
            now = time.time()
            if now - self.last_alert.get(key, 0) >= self.cooldown:
                self.last_alert[key] = now
                alert = {
                    "type":      key,
                    "message":   message,
                    "color":     color,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "sent_to":   "Supervisor (SMS + Email)",
                }
                self.alert_log.append(alert)
                new_alerts.append(alert)

        return new_alerts

    def get_recent(self, n: int = 10) -> list:
        return self.alert_log[-n:][::-1]


# ---------------------------------------------------------------------------
# Incident Logger
# ---------------------------------------------------------------------------
class IncidentLogger:
    def __init__(self):
        self.incidents = []
        self._counter  = random.randint(1000, 1999)

    def log(self, results, zone: str = "Zone A") -> dict | None:
        """Log an incident if violations exist. Returns incident dict or None."""
        score, level, count = RiskCalculator.calculate(results)
        if count == 0:
            return None

        self._counter += 1
        classes = RiskCalculator._extract_classes(results)
        viol_types = [c for c in classes if c in {"no_helmet","no_vest","fire","smoke","fall"}]

        incident = {
            "id":         f"INC-{self._counter}",
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "zone":       zone,
            "violations": ", ".join(set(viol_types)).upper(),
            "risk_score": score,
            "risk_level": level,
            "status":     "OPEN",
            "action":     self._hse_action(level),
        }
        self.incidents.append(incident)
        return incident

    def _hse_action(self, level: str) -> str:
        actions = {
            "CRITICAL": "Evacuate zone immediately - HSE notified",
            "HIGH":     "Stop work order issued - Supervisor dispatched",
            "MEDIUM":   "Warning issued - PPE compliance required",
            "LOW":      "Advisory logged - Monitor and review",
        }
        return actions.get(level, "Logged")

    def to_dataframe(self) -> pd.DataFrame:
        if not self.incidents:
            return pd.DataFrame(columns=[
                "id","timestamp","zone","violations","risk_score","risk_level","status","action"
            ])
        return pd.DataFrame(self.incidents[::-1])  # newest first

    def summary_stats(self) -> dict:
        if not self.incidents:
            return {"total": 0, "critical": 0, "high": 0, "open": 0}
        df = pd.DataFrame(self.incidents)
        return {
            "total":    len(df),
            "critical": int((df["risk_level"] == "CRITICAL").sum()),
            "high":     int((df["risk_level"] == "HIGH").sum()),
            "open":     int((df["status"] == "OPEN").sum()),
        }