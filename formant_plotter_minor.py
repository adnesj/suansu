import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as mticker

"""
only minor vowels for now
"""

#CONFIG--------------------------------

#input_csv = "formants_full.csv"
#input_csv = "formants_full_smoothed.csv"
#input_csv = "formants_full_smoothed_2.csv"
input_csv = "formants_full_smoothed_2_manual.csv"
output_plot = "vowels_minor.png"
#font
ipa_font_prop = FontProperties(family=['Inter'], size=12)

df = pd.read_csv(input_csv)
print(f"Loaded data from {input_csv}")

#---------------
df_filtered = df[df["Label"].str.match(r"^_.$", na=False)].copy()
if df_filtered.empty:
    print("No rows matched the strict '_v' filter.")
    exit()

df_filtered["Vowel"] = df_filtered["Label"].str.replace("_", "", regex=False)

# COLOURS
custom_colors = {
    #"a": "#f8766d",      # H=0° (Red)
    "e": "#e38900",   # H=30° (Red-Orange)
    "i": "#c49a00",   # H=60° (Yellow)
    "o": "#99a800", # H=90° (Yellow-Green)
    "u": "#53b400",    # H=120° (Green)
    "ə": "#00bc56", # H=150° (Cyan-Green)
    "": "#01c094",     # H=180° (Cyan)
    "ɯ": "#00bfc4", # H=210° (Sky Blue)
    "blue": "#00b6eb",     # H=240° (Blue/Indigo)
    "a": "#04a4ff",   # H=270° (Violet)
    "ə": "#a58aff",  # H=300° (Magenta)
    "rose": "#df70f8",      # H=330° (Rose)
    #"a": "#fb61d7",
    "magenta": "#ff66a8",
    }

df_filtered["Color"] = df_filtered["Vowel"].apply(lambda v: custom_colors.get(v, "#999999"))

fig, ax = plt.subplots(figsize=(10, 6))  # 5:3 ratio

for idx, row in df_filtered.iterrows():
    f1 = row["F1_mid"]
    f2 = row["F2_mid"]
    color = row["Color"]
    ax.scatter(f2, f1, color=color, s=60, alpha=0.8)

#Axe settings
ax.invert_xaxis()
ax.invert_yaxis()
ax.set_xlim(2500, 600)# F2
ax.set_ylim(800, 200) # F1
ax.xaxis.set_major_locator(mticker.MultipleLocator(100))  # F2 kvar 100 hz
ax.yaxis.set_major_locator(mticker.MultipleLocator(50))   # F1 kvar 50 hz
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
ax.yaxis.set_ticks_position('right')
ax.yaxis.set_label_position('right')
ax.set_xlabel("F2 (Hz)", fontsize=12, fontproperties=ipa_font_prop)
ax.set_ylabel("F1 (Hz)", fontsize=12, fontproperties=ipa_font_prop)
#ax.set_title("Pure Vowel Monophthongs (Strict '_v' Filter)", fontsize=16, weight='bold')



#legende
unique_vowels = sorted(df_filtered["Vowel"].unique())
handles = [plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=custom_colors[v], markersize=8, label=v)
           for v in unique_vowels]
ax.legend(handles=handles, title="Vowel", bbox_to_anchor=(1, 1), loc='upper right')
ax.grid(True, linestyle=':', alpha=0.6)
plt.xticks(fontproperties=ipa_font_prop)
plt.yticks(fontproperties=ipa_font_prop)
plt.tight_layout()




# Save
plt.savefig(output_plot, dpi=300)
plt.savefig(output_plot.replace(".png", ".pdf"), format='pdf', bbox_inches='tight')

plt.show()
