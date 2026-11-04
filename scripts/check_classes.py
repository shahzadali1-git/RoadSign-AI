
from collections import Counter
from pathlib import Path

label_root = Path("datasets/gtsdb/yolo/labels")
class_counts = Counter()

for split in ("train", "val"):
    folder = label_root / split

    for file in folder.glob("*.txt"):
        for line in file.read_text().splitlines():
            if line.strip():
                class_id = int(line.split()[0])
                class_counts[class_id] += 1

print("Traffic-sign instances per class:")

for class_id in range(43):
    print(f"Class {class_id:2}: {class_counts[class_id]}")

print(f"\nClasses represented: {len(class_counts)} / 43")
print(f"Total instances: {sum(class_counts.values())}")
