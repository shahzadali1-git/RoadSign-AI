from pathlib import Path
from ultralytics import YOLO

def main():
    dataset_yaml = Path("datasets/gtsdb/yolo/data.yaml")

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_yaml}")

    print("Loading pretrained YOLO11n model...", flush=True)

    model = YOLO("yolo11n.pt")

    print("Starting YOLO11n training...", flush=True)

    model.train(
        data=str(dataset_yaml),
        epochs=30,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        patience=10,
        project="runs/roadvision",
        name="yolo11_30epochs",
        exist_ok=True,
        plots=True,
        seed=42,
    )

    print("Training completed.", flush=True)

if __name__ == "__main__":
    main()
