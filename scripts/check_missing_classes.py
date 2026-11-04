from pathlib import Path

root = Path("datasets/gtsdb/yolo/labels")

def get_classes(split):
    classes = set()

    for file in (root / split).glob("*.txt"):
        for line in file.read_text(encoding="utf-8-sig").splitlines():
            values = line.split()
            if len(values) == 5:
                classes.add(int(float(values[0])))

    return classes

train_classes = get_classes("train")
val_classes = get_classes("val")

print("Classes in training:", len(train_classes))
print("Classes in validation:", len(val_classes))
print("Classes missing from training but present in validation:")
print(sorted(val_classes - train_classes))
