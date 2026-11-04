from pathlib import Path

label_dir = Path("datasets/gtsdb/yolo/labels/val")

for file in label_dir.glob("*.txt"):
    for line in file.read_text(encoding="utf-8-sig").splitlines():
        values = line.split()

        if len(values) == 5 and int(float(values[0])) == 31:
            print("Image:", file.stem + ".jpg")
            print("Annotation:", line)
