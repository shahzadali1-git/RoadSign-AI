import csv

source = "datasets/gtsdb/TrainIJCNN2013/gt.txt"

widths = []
heights = []

with open(source, "r", newline="") as file:
    for row in csv.reader(file, delimiter=";"):
        if len(row) == 6:
            x1, y1, x2, y2 = map(int, row[1:5])

            widths.append(x2 - x1)
            heights.append(y2 - y1)

print("Total boxes:", len(widths))
print("Average width:", round(sum(widths) / len(widths), 2), "pixels")
print("Average height:", round(sum(heights) / len(heights), 2), "pixels")
print("Smallest width:", min(widths))
print("Smallest height:", min(heights))
print("Largest width:", max(widths))
print("Largest height:", max(heights))