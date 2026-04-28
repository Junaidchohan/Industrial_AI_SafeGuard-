# 🏭 SafeGuard AI – Industrial Safety Monitoring Platform
### Steel & Metal Factory Edition | Powered by YOLOv8

> **"Real-time AI vision that prevents accidents before they happen."**

---

## 📁 Project Structure

```
safety_monitor/
│
├── app.py              ← Streamlit dashboard (run this)
├── detect.py           ← YOLOv8 inference engine + simulation
├── train.py            ← Model training script
├── utils.py            ← Risk scoring, alerts, incident logging
│
├── data.yaml           ← Dataset configuration for training
├── requirements.txt    ← Python dependencies
│
├── models/             ← Trained weights go here
│   └── safety_yolov8.pt   (auto-generated after training)
│
├── datasets/           ← Dataset folder (download separately)
│   └── safety/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
│
└── README.md
```

---

## 🚀 Quick Start (3 Steps)

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — (Optional) Train the Model
Only needed if you have a dataset. Skip this to use Demo Mode.

```bash
# Quick training (GPU recommended)
python train.py --epochs 40 --batch 16 --device 0

# CPU-only
python train.py --epochs 40 --batch 8 --device cpu

# Or directly with ultralytics CLI
yolo detect train model=yolov8n.pt data=data.yaml epochs=40 imgsz=640
```

### Step 3 — Launch Dashboard
```bash
streamlit run app.py
```

Open in browser: **http://localhost:8501**

---

## 🎬 Running Without a Dataset (Demo Mode)

No dataset? No problem. The system includes a **high-fidelity simulation engine** that:

- Generates realistic factory floor frames
- Simulates moving workers with PPE violations
- Triggers fire/smoke events periodically
- Calculates real risk scores and alerts
- Logs incidents to the panel

Just select **🎬 Demo Mode** in the sidebar and click **▶ START MONITOR**.

---

## 📡 Detection Classes

| ID | Class      | Description              | Risk Weight |
|----|------------|--------------------------|-------------|
| 0  | person     | Worker detected          | –           |
| 1  | helmet     | PPE compliant            | –           |
| 2  | no_helmet  | PPE violation (helmet)   | +25         |
| 3  | vest       | PPE compliant            | –           |
| 4  | no_vest    | PPE violation (vest)     | +15         |
| 5  | fire       | Fire hazard              | +50         |
| 6  | smoke      | Smoke/combustion hazard  | +30         |
| 7  | fall       | Worker fall detected     | +40         |

---

## 🧮 Risk Scoring Model

```
Risk Score = Σ (violation_weight × count)  [clamped to 100]

Risk Level:
  0–24   → LOW      (green)
  25–49  → MEDIUM   (amber)
  50–74  → HIGH     (orange)
  75–100 → CRITICAL (red, flashing)
```

---

## 🗃️ Dataset Sources

### PPE Detection
| Source | Dataset | Link |
|--------|---------|------|
| Roboflow | Hard Hat Universe | https://universe.roboflow.com/roboflow-universe-projects/hard-hat-universe |
| Roboflow | Construction Safety | https://universe.roboflow.com/proyek-akhir-mikrotik/construction-safety-gsnvb |
| Roboflow | PPE Detection | https://universe.roboflow.com/new-workspace-wqunn/ppe-detection-jhbpq |

### Fire & Smoke
| Source | Dataset | Link |
|--------|---------|------|
| Kaggle | Fire Dataset | https://www.kaggle.com/datasets/phylake1337/fire-dataset |
| Roboflow | Wildfire Smoke | https://universe.roboflow.com/roboflow-100/wildfire-smoke |
| Roboflow | Fire & Smoke | https://universe.roboflow.com/test-hvhce/fire-and-smoke-detection-kxmzj |

### Quick Download via Roboflow API
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("roboflow-universe-projects").project("hard-hat-universe")
dataset  = project.version(1).download("yolov8")
```

---

## ⚙️ Training Configuration Reference

```bash
# Recommended GPU training
python train.py \
  --model yolov8n.pt \   # yolov8s.pt for better accuracy
  --data  data.yaml \
  --epochs 40 \
  --batch  16 \
  --imgsz  640 \
  --device 0

# Validate trained model
python train.py --mode val --weights models/safety_yolov8.pt

# Export to ONNX (for deployment)
python train.py --mode export --weights models/safety_yolov8.pt --format onnx
```

---

## 🎯 Client Demo Script

### Before the Demo
1. Start the dashboard: `streamlit run app.py`
2. Set facility name to client's company name (sidebar)
3. Select **Demo Mode** → Click **▶ START MONITOR**

### Talking Points by Section

**Live Feed Panel:**
> "This is the AI vision feed from your factory floor CCTV. Every worker is
> tracked in real time — no manual monitoring required."

**Violations Counter:**
> "Each PPE violation is logged automatically. No more relying on floor
> supervisors to catch every incident manually."

**Risk Score:**
> "This single number tells you exactly how dangerous the current situation is.
> Leadership can see this on any device, from anywhere."

**Alert Panel:**
> "The moment a violation occurs, an alert goes to the supervisor's phone and
> email — within seconds, not minutes."

**Incident Log:**
> "Every event is timestamped and logged. This becomes your compliance record
> for regulatory audits — zero paperwork."

### ROI Message
> "A single serious accident in a steel plant costs ₹50–200 lakh in
> compensation, production loss, and regulatory fines. SafeGuard AI prevents
> that — at a fraction of the cost."

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Webcam not found | Use Demo Mode or Upload Video |
| ultralytics not installed | `pip install ultralytics` |
| CUDA error | Use `--device cpu` in training |
| Streamlit not found | `pip install streamlit` |
| Slow performance | Reduce webcam resolution or use Demo Mode |

---

## 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| AI Model | YOLOv8 (Ultralytics) |
| Dashboard | Streamlit |
| Vision | OpenCV |
| Language | Python 3.9+ |
| Compute | CPU / NVIDIA GPU |

---

## 🏢 About

**SafeGuard AI** is developed by **Softnity Technologies**
as a turnkey AI safety solution for heavy industry.

> *"Think beyond boundaries."*

---

*Version 2.4.1 | YOLOv8 Engine | © 2025 Softnity Technologies*
