import csv
from collections import Counter

source = "datasets/gtsdb/TrainIJCNN2013/gt.txt"

class_boxes = Counter()
total_boxes = 0
total_images = set()

with open(source, "r", newline="") as file:
    for row in csv.reader(file, delimiter=";"):
        if len(row) == 6:
            filename = row[0]
            class_id = int(row[5])

            total_images.add(filename)
            class_boxes[class_id] += 1
            total_boxes += 1

print("Original annotated images:", len(total_images))
print("Original total bounding boxes:", total_boxes)
print("Original classes:", len(class_boxes))

print("\nBoxes per class:")

for class_id in range(43):
    print(f"Class {class_id:2}: {class_boxes[class_id]:3} boxes")