
from pathlib import Path
from ultralytics import YOLO

def main():
    dataset_yaml = Path("datasets/gtsdb/yolo/data.yaml")

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {dataset_yaml}"
        )

    # Start with a small pretrained model for CPU training.
    model = YOLO("yolo26n.pt")

    model.train(
        data=str(dataset_yaml),
        epochs=3,           # Short test run
        imgsz=640,
        batch=4,            # Conservative for CPU and RAM
        device="cpu",
        workers=0,          # Safer on Windows
        project="runs/roadvision",
        name="first_train",
        exist_ok=True,
        plots=True,
    )

    print("Training test completed.")
    print("Check runs/roadvision/first_train/ for results.")

if __name__ == "__main__":
    main()
