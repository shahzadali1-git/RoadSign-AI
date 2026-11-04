from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/roadvision/first_train/weights/best.pt"
)

image = "datasets/gtsdb/yolo/images/train/00000.jpg"

results = model.predict(
    source=image,
    imgsz=640,
    conf=0.001,
    max_det=100,
    verbose=False,
)

result = results[0]
print("Image:", image)
print("Detections at 0.1% confidence:", len(result.boxes))

if len(result.boxes):
    confidences = result.boxes.conf.tolist()
    print("Highest confidence:", f"{max(confidences):.4%}")

    for box in result.boxes[:10]:
        print(
            f"Class {int(box.cls.item())}, "
            f"confidence {float(box.conf.item()):.4%}"
        )
else:
    print("No detections even at the lower threshold.")
