from pathlib import Path
from collections import Counter

root = Path("datasets/gtsdb/yolo/labels")
total_duplicates = 0

for split in ("train", "val"):
    duplicate_files = []

    for label_file in (root / split).glob("*.txt"):
        lines = [
            line.strip()
            for line in label_file.read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if line.strip()
        ]

        counts = Counter(lines)
        duplicates = sum(count - 1 for count in counts.values() if count > 1)

        if duplicates:
            duplicate_files.append((label_file.name, duplicates))
            total_duplicates += duplicates

    print(f"\n--- {split.upper()} ---")
    print("Files containing duplicates:", len(duplicate_files))

    for filename, count in duplicate_files[:20]:
        print(filename, "->", count, "duplicate(s)")

print("\nTotal duplicate annotations:", total_duplicates)
