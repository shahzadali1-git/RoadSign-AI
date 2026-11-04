from pathlib import Path
import cv2

root = Path("datasets/gtsdb/yolo")
image_path = root / "images" / "val" / "00281.jpg"
label_path = root / "labels" / "val" / "00281.txt"

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(image_path)

h, w = image.shape[:2]

for line in label_path.read_text(encoding="utf-8-sig").splitlines():
    cls, xc, yc, bw, bh = map(float, line.split())

    x1 = int((xc - bw / 2) * w)
    y1 = int((yc - bh / 2) * h)
    x2 = int((xc + bw / 2) * w)
    y2 = int((yc + bh / 2) * h)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"Class {int(cls)}", (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

output = Path("runs/dataset_check/class31_00281.jpg")
output.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(output), image)

print("Saved:", output)
