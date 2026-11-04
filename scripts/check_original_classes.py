import csv
from collections import defaultdict

source = "datasets/gtsdb/TrainIJCNN2013/gt.txt"

class_images = defaultdict(set)

with open(source, "r", newline="") as file:
    for row in csv.reader(file, delimiter=";"):
        if len(row) == 6:
            filename = row[0]
            class_id = int(row[5])
            class_images[class_id].add(filename)

print("\n--- ORIGINAL DATASET CLASS DISTRIBUTION ---\n")

for class_id in range(43):
    print(
        f"Class {class_id:2}: "
        f"{len(class_images[class_id]):3} images"
    )

print("\nTotal classes:", len(class_images))