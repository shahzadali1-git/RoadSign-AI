from pathlib import Path

source = Path("datasets/gtsdb/TrainIJCNN2013/gt.txt")
converted = Path("datasets/gtsdb/yolo/labels/train/00340.txt")

print("--- ORIGINAL ANNOTATIONS ---")

for line in source.read_text(encoding="utf-8-sig").splitlines():
    if line.split(";")[0].split(".")[0] == "00340":
        print(line)

print("\n--- CONVERTED YOLO LABELS ---")

if converted.exists():
    for line in converted.read_text(encoding="utf-8-sig").splitlines():
        print(line)
else:
    print("Converted label file not found.")
