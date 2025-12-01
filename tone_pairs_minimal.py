import pandas as pd
import sys
import re
from collections import defaultdict

csv_file = "formants_full_smoothed_2_manual.csv"
df = pd.read_csv(csv_file)

def normalize_word(word):
    # remove stress mark ˈ
    word_no_stress = word.replace("ˈ", "")
    # remove tone number
    word_base = re.sub(r'\d$', '', word_no_stress)
    return word_base

word_dict = defaultdict(list)
for idx, word in enumerate(df["Word"]):
    norm = normalize_word(str(word))
    word_dict[norm].append(idx)







print("Minimal tone pairs found:")
for base, indices in word_dict.items():
    tone_numbers = set(re.findall(r'(\d)$', str(df.loc[i, "Word"]))[0] for i in indices if re.search(r'\d$', str(df.loc[i, "Word"])))
    if len(indices) > 1 and len(tone_numbers) > 1:
        for i in indices:
            word = df.loc[i, "Word"]
            english = df.loc[i, "English"]
            print(f"{word} -> {english}")
        print("-" * 40)
