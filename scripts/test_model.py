
from pathlib import Path
from ultralytics import YOLO

def main():
    model_path = Path(
        "runs/detect/runs/roadvision/first_train/weights/best.pt"
    )
    image_path = Path(
        "datasets/gtsdb/yolo/images/val/00576.jpg"
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = YOLO(str(model_path))

    results = model.predict(
        source=str(image_path),
        conf=0.10,
        save=True,
    )

    for result in results:
        print(f"Image: {image_path}")
        print(f"Detections: {len(result.boxes)}")

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            print(f"Class: sign_{class_id}, Confidence: {confidence:.2%}")

    print("Prediction test finished.")

if __name__ == "__main__":
    main()
