import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

#matplot lib settings
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16




ipa_font_prop = FontProperties(family=['Inter'], size=16)



#input_csv = "formants_full.csv"
#input_csv = "formants_full_smoothed.csv"
#input_csv = "formants_full_smoothed_2.csv"
input_csv = "formants_full_smoothed_2_manual.csv"
output_plot = "vowel_space_pure_monophthongs_custom.png"

try:
    df = pd.read_csv(input_csv)
    print(f"Loaded {input_csv}")
except FileNotFoundError:
    print(f"{input_csv} is not")
    data = 1
    df = pd.DataFrame(data)
    print("wrong file")

df_filtered = df[df["Label"].str.match(r"^_.$", na=False)].copy()
if df_filtered.empty:
    print("No rows were minor.")
    exit()

df_filtered["Vowel"] = df_filtered["Label"].str.replace("_", "", regex=False)

#COLOURS
custom_colors = {
    # Blue shades
    'a':  '#1f78b4',
    '': '#a6cee3',
    
    # Orange shades
    '': '#ff7f00',
    '': '#fdbf6f',
    
    # Purple shades
    'ə': '#6a3d9a',
    '': '#cab2d6',
    
    # Cyan / Teal shades
    '': '#008080',
    '': '#66c2a5',
    
    # Brown / Warm gray shades
    '': '#b15928',
    '': '#f4a582',
}
df_filtered["Color"] = df_filtered["Vowel"].apply(lambda v: custom_colors.get(v, "#999999"))

plt.figure(figsize=(10, 10))

for idx, row in df_filtered.iterrows():
    vowel = row["Vowel"]
    f1 = row["F1_mid"]
    f2 = row["F2_mid"]
    english = row.get("English", "")
    color = row["Color"]

    plt.scatter(f2, f1, color=color, s=60, alpha=0.8)

    label_text = f"{vowel} ({english})" if english and not pd.isna(english) else vowel
    plt.text(f2, f1, label_text, color=color, fontsize=10,
             verticalalignment='bottom', horizontalalignment='center',
             fontproperties=ipa_font_prop)

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.xlabel("F2 (Hz)")
plt.ylabel("F1 (Hz)")
#plt.title("Pure Vowel Monophthongs (Strict '_v' Filter)", fontsize=10, weight='bold')

ax = plt.gca()
ax.invert_xaxis()
ax.invert_yaxis()
ax.set_xlim(2500, 600)# F2
ax.set_ylim(850, 200) # F1
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
ax.yaxis.set_ticks_position('right')
ax.yaxis.set_label_position('right')
plt.xticks(fontproperties=ipa_font_prop)
plt.yticks(fontproperties=ipa_font_prop)

# Legende
handles = [plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=col, markersize=8, label=v)
           for v, col in custom_colors.items() if v in df_filtered["Vowel"].unique()]
plt.legend(handles=handles, title="Vowel Symbol", bbox_to_anchor=(1.02, 1), loc='upper left')

plt.grid(True, linestyle=':', alpha=0.6)
plt.xticks(fontproperties=ipa_font_prop)
plt.yticks(fontproperties=ipa_font_prop)
plt.tight_layout()

plt.savefig(output_plot, dpi=300)
plt.show()
