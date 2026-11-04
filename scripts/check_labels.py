
from pathlib import Path

ROOT = Path("datasets/gtsdb/yolo")
NUM_CLASSES = 43

errors = []
total_boxes = 0

for split in ("train", "val"):
    label_dir = ROOT / "labels" / split
    files = list(label_dir.glob("*.txt"))
    split_boxes = 0

    for file in files:
        for line_number, line in enumerate(
            file.read_text().splitlines(), start=1
        ):
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != 5:
                errors.append(
                    f"{file}:{line_number}: expected 5 values"
                )
                continue

            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                errors.append(
                    f"{file}:{line_number}: invalid number"
                )
                continue

            if not 0 <= class_id < NUM_CLASSES:
                errors.append(
                    f"{file}:{line_number}: invalid class ID"
                )

            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                errors.append(
                    f"{file}:{line_number}: coordinate out of range"
                )

            if w <= 0 or h <= 0:
                errors.append(
                    f"{file}:{line_number}: invalid box size"
                )

            split_boxes += 1

    total_boxes += split_boxes
    print(f"{split}: {len(files)} label files, {split_boxes} boxes")

print(f"\nTotal boxes: {total_boxes}")
print(f"Errors found: {len(errors)}")

for error in errors[:20]:
    print(error)

if not errors:
    print("All labels passed these basic checks.")
