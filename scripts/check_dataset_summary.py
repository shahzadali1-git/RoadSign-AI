from pathlib import Path
from collections import Counter

root = Path("datasets/gtsdb/yolo")

for split in ("train", "val"):
    image_dir = root / "images" / split
    label_dir = root / "labels" / split

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    missing_labels = []
    empty_labels = []
    total_boxes = 0
    class_counts = Counter()

    for image in images:
        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            missing_labels.append(image.name)
            continue

        lines = [
            line.strip()
            for line in label.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]

        if not lines:
            empty_labels.append(image.name)

        total_boxes += len(lines)

        for line in lines:
            values = line.split()
            if len(values) == 5:
                class_id = int(float(values[0]))
                class_counts[class_id] += 1

    print(f"\n--- {split.upper()} DATASET ---")
    print("Images:", len(images))
    print("Missing label files:", len(missing_labels))
    print("Empty label files:", len(empty_labels))
    print("Total bounding boxes:", total_boxes)
    print("Classes represented:", len(class_counts))
    print("Images with boxes:", sum(
        1 for image in images
        if (label_dir / f"{image.stem}.txt").exists()
        and (label_dir / f"{image.stem}.txt").read_text(
            encoding="utf-8-sig"
        ).strip()
    ))
    print("Class counts:", dict(sorted(class_counts.items())))
