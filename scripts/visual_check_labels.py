from pathlib import Path
import cv2

root = Path("datasets/gtsdb/yolo")
output_dir = Path("runs/dataset_check")
output_dir.mkdir(parents=True, exist_ok=True)

samples = [
    ("train", "00000.jpg"),
    ("train", "00001.jpg"),
    ("train", "00002.jpg"),
    ("val", "00576.jpg"),
]

for split, filename in samples:
    image_path = root / "images" / split / filename
    label_path = root / "labels" / split / f"{Path(filename).stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read:", image_path)
        continue

    height, width = image.shape[:2]
    box_count = 0

    if label_path.exists():
        lines = label_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()

        for line in lines:
            values = line.split()
            if len(values) != 5:
                continue

            cls, xc, yc, bw, bh = map(float, values)

            x1 = int((xc - bw / 2) * width)
            y1 = int((yc - bh / 2) * height)
            x2 = int((xc + bw / 2) * width)
            y2 = int((yc + bh / 2) * height)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"class {int(cls)}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            box_count += 1

    output_path = output_dir / f"{split}_{filename.replace('.jpg', '')}.jpg"
    cv2.imwrite(str(output_path), image)

    print(f"{output_path} | boxes drawn: {box_count}")

print("\nOpen the images in runs/dataset_check/")
print("\nOpen the images in runs/dataset_check/")