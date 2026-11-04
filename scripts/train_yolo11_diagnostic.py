from pathlib import Path
from ultralytics import YOLO

def main():
    dataset_yaml = Path("datasets/gtsdb/yolo/data.yaml")

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_yaml}")

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(dataset_yaml),
        epochs=1,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        project="runs/roadvision",
        name="yolo11_diagnostic",
        exist_ok=True,
        plots=True,
    )

    print("Diagnostic training completed.")
    print("Results: runs/detect/runs/roadvision/yolo11_diagnostic/")

if __name__ == "__main__":
    main()
