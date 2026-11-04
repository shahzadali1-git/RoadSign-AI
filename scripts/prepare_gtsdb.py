
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

SOURCE = Path("datasets/gtsdb/TrainIJCNN2013")
OUTPUT = Path("datasets/gtsdb/yolo")
ANNOTATIONS = SOURCE / "gt.txt"

VAL_RATIO = 0.20
SEED = 42
NUM_CLASSES = 43


def main():
    if not SOURCE.exists() or not ANNOTATIONS.exists():
        raise FileNotFoundError(
            "GTSDB images or gt.txt were not found. "
            "Run this script from the RoadSign-AI project folder."
        )

    # Read the original bounding-box annotations.
    annotations = {}
    with ANNOTATIONS.open("r", newline="") as file:
        for row in csv.reader(file, delimiter=";"):
            if len(row) != 6:
                continue

            filename = row[0]
            x1, y1, x2, y2 = map(int, row[1:5])
            class_id = int(row[5])

            if not 0 <= class_id < NUM_CLASSES:
                raise ValueError(f"Unexpected class ID: {class_id}")

            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Invalid bounding box: {row}")

            annotations.setdefault(filename, []).append(
                (x1, y1, x2, y2, class_id)
            )

    # Use a reproducible, shuffled 80/20 image split.
    images = sorted(SOURCE.glob("*.ppm"))
    if not images:
        raise FileNotFoundError(f"No PPM images found in {SOURCE}")

    random.Random(SEED).shuffle(images)
    split_at = int(len(images) * (1 - VAL_RATIO))
    splits = {
        "train": images[:split_at],
        "val": images[split_at:],
    }

    class_counts = Counter()
    total_boxes = 0

    for split, split_images in splits.items():
        image_dir = OUTPUT / "images" / split
        label_dir = OUTPUT / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                width, height = image.size

                # JPEG is broadly supported by object-detection tools.
                output_image = image_dir / f"{image_path.stem}.jpg"
                image.save(output_image, "JPEG", quality=95)

            label_path = label_dir / f"{image_path.stem}.txt"
            boxes = annotations.get(image_path.name, [])

            # Create an empty label file for images with no signs.
            with label_path.open("w") as label_file:
                for x1, y1, x2, y2, class_id in boxes:
                    x_center = ((x1 + x2) / 2) / width
                    y_center = ((y1 + y2) / 2) / height
                    box_width = (x2 - x1) / width
                    box_height = (y2 - y1) / height

                    values = (
                        class_id,
                        x_center,
                        y_center,
                        box_width,
                        box_height,
                    )
                    label_file.write(
                        f"{class_id} {x_center:.6f} {y_center:.6f} "
                        f"{box_width:.6f} {box_height:.6f}\n"
                    )
                    class_counts[class_id] += 1
                    total_boxes += 1

    # Create the YOLO dataset configuration.
    dataset_path = OUTPUT.resolve().as_posix()
    names = ", ".join(
        f"{i}: sign_{i}" for i in range(NUM_CLASSES)
    )

    yaml_text = (
        f"path: '{dataset_path}'\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {NUM_CLASSES}\n"
        f"names: {{{names}}}\n"
    )
    (OUTPUT / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print("Dataset conversion completed!")
    print(f"Original images: {len(images)}")
    print(f"Training images: {len(splits['train'])}")
    print(f"Validation images: {len(splits['val'])}")
    print(f"Bounding boxes converted: {total_boxes}")
    print(f"Classes represented: {len(class_counts)} / {NUM_CLASSES}")
    print(f"YOLO configuration: {OUTPUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
