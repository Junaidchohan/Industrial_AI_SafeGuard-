"""
train.py – SafeGuard AI Training Pipeline
YOLOv8 custom training for industrial safety detection
"""

import argparse
import os
import sys
from pathlib import Path


# ─── Dependency Check ─────────────────────────────────────────────────────────
def check_deps():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)


# ─── Default Config ───────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "model":      "yolov8n.pt",   # nano – fast training; use yolov8s.pt for better accuracy
    "data":       "data.yaml",
    "epochs":     40,
    "imgsz":      640,
    "batch":      16,
    "name":       "safety_monitor_v1",
    "project":    "runs/train",
    "device":     "0",            # '0' for GPU, 'cpu' for CPU
    "patience":   10,             # Early stopping
    "workers":    4,
    "lr0":        0.01,
    "lrf":        0.001,
    "mosaic":     1.0,
    "augment":    True,
    "val":        True,
    "save":       True,
    "cache":      False,
}


def train(config: dict):
    YOLO = check_deps()

    print("\n" + "="*60)
    print("  SafeGuard AI – YOLOv8 Training Pipeline")
    print("="*60)
    print(f"  Model:   {config['model']}")
    print(f"  Data:    {config['data']}")
    print(f"  Epochs:  {config['epochs']}")
    print(f"  ImgSz:   {config['imgsz']}")
    print(f"  Batch:   {config['batch']}")
    print(f"  Device:  {config['device']}")
    print("="*60 + "\n")

    if not Path(config["data"]).exists():
        print(f"[ERROR] data.yaml not found at: {config['data']}")
        print("  Please create data.yaml with your dataset paths.")
        print("  See the sample data.yaml included in this project.\n")
        sys.exit(1)

    model = YOLO(config["model"])

    results = model.train(
        data     = config["data"],
        epochs   = config["epochs"],
        imgsz    = config["imgsz"],
        batch    = config["batch"],
        name     = config["name"],
        project  = config["project"],
        device   = config["device"],
        patience = config["patience"],
        workers  = config["workers"],
        lr0      = config["lr0"],
        lrf      = config["lrf"],
        mosaic   = config["mosaic"],
        augment  = config["augment"],
        val      = config["val"],
        save     = config["save"],
        cache    = config["cache"],
        verbose  = True,
    )

    best_weights = Path(config["project"]) / config["name"] / "weights" / "best.pt"
    dest = Path("models") / "safety_yolov8.pt"
    dest.parent.mkdir(exist_ok=True)

    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, dest)
        print(f"\n✅ Training complete. Best weights saved to: {dest}")
        print(f"   mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    else:
        print("\n⚠️  Training finished but best.pt not found. Check runs/ directory.")

    print("\n  → Run the dashboard: streamlit run app.py\n")


def validate(weights_path: str, data_path: str):
    YOLO = check_deps()
    model = YOLO(weights_path)
    metrics = model.val(data=data_path)
    print(f"\n Validation Results:")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision:{metrics.box.mp:.4f}")
    print(f"  Recall:   {metrics.box.mr:.4f}")


def export_model(weights_path: str, fmt: str = "onnx"):
    YOLO = check_deps()
    model = YOLO(weights_path)
    model.export(format=fmt)
    print(f"\n✅ Model exported to {fmt.upper()} format.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeGuard AI – Training Script")
    parser.add_argument("--mode",    default="train",  choices=["train","val","export"])
    parser.add_argument("--model",   default=DEFAULT_CONFIG["model"])
    parser.add_argument("--data",    default=DEFAULT_CONFIG["data"])
    parser.add_argument("--epochs",  type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--imgsz",   type=int, default=DEFAULT_CONFIG["imgsz"])
    parser.add_argument("--batch",   type=int, default=DEFAULT_CONFIG["batch"])
    parser.add_argument("--device",  default=DEFAULT_CONFIG["device"])
    parser.add_argument("--weights", default="models/safety_yolov8.pt", help="For val/export")
    parser.add_argument("--format",  default="onnx", help="Export format")
    args = parser.parse_args()

    if args.mode == "train":
        config = DEFAULT_CONFIG.copy()
        config.update({
            "model":   args.model,
            "data":    args.data,
            "epochs":  args.epochs,
            "imgsz":   args.imgsz,
            "batch":   args.batch,
            "device":  args.device,
        })
        train(config)

    elif args.mode == "val":
        validate(args.weights, args.data)

    elif args.mode == "export":
        export_model(args.weights, args.format)
