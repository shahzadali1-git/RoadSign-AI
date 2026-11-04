from pathlib import Path
from PIL import Image

root = Path("datasets/gtsdb/yolo")

for split, filename in [
    ("train", "00000.jpg"),
    ("val", "00576.jpg"),
]:
    image_path = root / "images" / split / filename
    label_path = root / "labels" / split / f"{Path(filename).stem}.txt"

    print(f"\n--- {split.upper()} ---")
    print("Image exists:", image_path.exists())
    print("Label exists:", label_path.exists())

    if image_path.exists():
        with Image.open(image_path) as im:
            print("Image dimensions:", im.size)

    if label_path.exists():
        lines = [
            line.strip()
            for line in label_path.read_text().splitlines()
            if line.strip()
        ]
        print("Ground-truth boxes:", len(lines))

        for line in lines[:5]:
            values = line.split()
            print("Label:", line)

            if len(values) != 5:
                print("WARNING: Expected 5 values per label")
                continue

            cls, x, y, w, h = map(float, values)

            if not (
                0 <= cls < 43
                and 0 <= x <= 1
                and 0 <= y <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):
                print("WARNING: Label values outside expected ranges")
    else:
        print("No matching label file found.")
