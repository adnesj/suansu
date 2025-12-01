import csv
import sys

def main():
    input_file = "formants_full_smoothed_2_manual.csv"

    with open(input_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        with_glide = 0
        without_glide = 0

        for row in reader:
            label = row.get("Label", "")
            word = row.get("Word", "")
            nextseg = row.get("NextSeg", "")

            if not label.endswith("_"):
                continue

            if nextseg == "j":
                without_glide += 1
                continue

            if any(g in word for g in ["j", "w", "ɰ"]):
                with_glide += 1
            else:
                without_glide += 1

    print("Rows with j/w/ɰ:", with_glide)
    print("Rows without j/w/ɰ:", without_glide)


if __name__ == "__main__":
    main()
