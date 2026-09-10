"""Lab 4 starter: audit dialect mix and record the implication in NOTES.md."""

from collections import Counter
import csv


DATA_PATH = "data/raw/bayan_feedback.csv"


def main():
    dialect_counts = Counter()
    total_arabic = 0

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            language = row["lang"].strip().lower()

            if language != "ar":
                continue

            dialect = row["dialect_region"].strip() or "unknown"

            dialect_counts[dialect] += 1
            total_arabic += 1

    print("Arabic dialect / region distribution:\n")

    for dialect, count in dialect_counts.most_common():
        percentage = count / total_arabic * 100

        print(f"{dialect}: {count} ({percentage:.2f}%)")

    print(f"\nTotal Arabic examples: {total_arabic}")


if __name__ == "__main__":
    main()