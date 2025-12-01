import csv
import sys
from collections import defaultdict

def main():
    input_file = "formants_full_smoothed_2_manual.csv"

    tilde_char = "̃"

    tilde_without_nextseg = 0
    total_tilde_labels = 0

    nextseg_counts = defaultdict(int)

    try:
        with open(input_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                label = row.get("Label", "")
                nextseg = row.get("NextSeg", "").strip() # have to do it now

                # Check if "Label" has the tilde character (̃)
                if tilde_char in label:
                    total_tilde_labels += 1
                    
                    if nextseg:
                        nextseg_counts[nextseg] += 1
                    else:
                        # has tilde AND nextseg is nothing
                        tilde_without_nextseg += 1

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    tilde_with_nextseg = sum(nextseg_counts.values())










    
    print(f"## 📊 Analysis of Labels Containing '{tilde_char}'")
    print("---")
    print(f"**Total rows where 'Label' contains '{tilde_char}': {total_tilde_labels}**")
    print("\n### Summary Counts")
    print(f"* Rows with '{tilde_char}' in 'Label' AND a **value** in 'NextSeg': **{tilde_with_nextseg}**")
    print(f"* Rows with '{tilde_char}' in 'Label' BUT **NO value** in 'NextSeg': **{tilde_without_nextseg}**")

    if nextseg_counts:
        print("\n### Counts for Individual NextSeg Values:")
        #sorted_nextseg = sorted(nextseg_counts)
        sorted_nextseg = sorted(nextseg_counts.items(), key=lambda item: item[1], reverse=True)
        
        for nextseg, count in sorted_nextseg:
            print(f"* NextSeg '{nextseg}': {count}")
    else:
        print("\n### Counts for Individual NextSeg Values:")
        print("No rows found where 'Label' contains the tilde and 'NextSeg' has a value.")
        
    print("---")
    print("SAMANLIKNING-------------------")
    if tilde_with_nextseg > tilde_without_nextseg:
        print(f"MORE rows with 'NextSeg' val ({tilde_with_nextseg}) than without ({tilde_without_nextseg})")
    elif tilde_without_nextseg > tilde_with_nextseg:
        print(f"MORE rows without 'NextSeg' val ({tilde_without_nextseg}) than with ({tilde_with_nextseg})")
    else:
        print(f"EQUAL, with: {tilde_with_nextseg}, without: {tilde_without_nextseg})")


if __name__ == "__main__":
    main()