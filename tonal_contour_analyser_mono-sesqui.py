import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os


INPUT_CSV_FILE = "tonal_contour_smoothed.csv"
NUM_F0_SAMPLES = 30  
plt.rcParams['font.family'] = 'Inter'
plt.rcParams['font.size'] = 12

# COLOURS
TONE_COLORS = {
    1: '#f8766d',
    2: '#01c094',
    3: '#1f77b4'
}

custom_colors = {
    "a": "#f8766d",      # H=0° (Red)
    "e": "#e38900",   # H=30° (Red-Orange)
    "i": "#c49a00",   # H=60° (Yellow)
    "o": "#99a800", # H=90° (Yellow-Green)
    "u": "#53b400",    # H=120° (Green)
    "ə": "#00bc56", # H=150° (Cyan-Green)
    "ʌ": "#01c094",     # H=180° (Cyan)
    "ɯ": "#00bfc4", # H=210° (Sky Blue)
    "blue": "#00b6eb",     # H=240° (Blue/Indigo)
    "violet": "#04a4ff",   # H=270° (Violet)
    "magenta": "#a58aff",  # H=300° (Magenta)
    "rose": "#df70f8",      # H=330° (Rose)
    "magenta": "#fb61d7",
    "magenta": "#ff66a8",
    }

# 5:3 ratio
FIGSIZE = (10, 6)







def load_and_group_data(file_path):
    if not os.path.exists(file_path):
        print(f"Feil: Finner ikke inputfilen: {file_path}")
        return None, None, None

    df = pd.read_csv(file_path)

    f0_cols = [f'F0_P{i+1}' for i in range(NUM_F0_SAMPLES)]

    def extract_tone_group(word):
        if pd.isna(word):
            return 0
        try:
            digit = int(str(word).strip()[-1])
            return digit if 1 <= digit <= 3 else 0
        except ValueError:
            return 0

    df['Tone_Group'] = df['Word'].apply(extract_tone_group)

    df['Syllabic_Type'] = df['Minor_Syllable_End_Time_s'].apply(
        lambda x: 'Monosyllabic' if pd.isna(x) or x == 0.0 else 'Sesquisyllabic'
    )

    df_filtered = df[df['Tone_Group'].isin([1, 2, 3])].copy()

    df_filtered[f0_cols] = df_filtered[f0_cols].replace(0.0, np.nan)

    # Tone 3 is actually tone 2
    df_filtered.loc[df_filtered['Tone_Group'] == 3, 'Tone_Group'] = 2

    groups = {
        1: df_filtered[df_filtered['Tone_Group'] == 1],
        2: df_filtered[df_filtered['Tone_Group'] == 2]
    }

    return df_filtered, groups, f0_cols


def compute_mean_and_sd(df, f0_cols):
    """
    Returns mean contour and standard deviation.
    """
    mean = df[f0_cols].mean(axis=0)
    sd = df[f0_cols].std(axis=0)
    return mean, sd


def plot_overall_average_contours(groups, f0_cols):
    fig = plt.figure(figsize=FIGSIZE)
    x_axis = np.linspace(0, 100, NUM_F0_SAMPLES)

    print("GNEJJOMSNITT")

    for group_num in [1, 2]:
        group_df = groups.get(group_num)

        if group_df is not None and not group_df.empty:
            mean_contour, sd = compute_mean_and_sd(group_df, f0_cols)

            plt.plot(
                x_axis,
                mean_contour,
                color=TONE_COLORS[group_num],
                linewidth=3,
                label=f'Tone {group_num} (N={len(group_df)})'
            )

            # SD shading
            plt.fill_between(
                x_axis,
                mean_contour - sd,
                mean_contour + sd,
                color=TONE_COLORS[group_num],
                alpha=0.15
            )

    plt.xlabel('Normalized Word Length (%)')
    plt.ylabel('Average Fundamental Frequency (F0) [Hz]')
    plt.legend(
        title="Tone Group",
        loc='lower left',       # anchor to bottom-left
        bbox_to_anchor=(0, 0),  # fixed position
        frameon=True,
        fontsize=10
    )
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    outname = "overall_average_contours.pdf"
    fig.savefig(outname, format="pdf")
    plt.show()

    print(f"Lagra PDF: {outname}")


def plot_average_by_structure(df_filtered, f0_cols, syllabic_type):
    df_type = df_filtered[df_filtered['Syllabic_Type'] == syllabic_type]

    if df_type.empty:
        print(f"Mangle data for {syllabic_type}, hoppe over.")
        return

    fig = plt.figure(figsize=FIGSIZE)
    x_axis = np.linspace(0, 100, NUM_F0_SAMPLES)

    for group_num in [1, 2]:
        group_df = df_type[df_type['Tone_Group'] == group_num]

        if not group_df.empty:
            mean_contour, sd = compute_mean_and_sd(group_df, f0_cols)

            plt.plot(
                x_axis,
                mean_contour,
                color=TONE_COLORS[group_num],
                linewidth=3,
                label=f'Tone {group_num} (N={len(group_df)})'
            )

            # SD shading
            plt.fill_between(
                x_axis,
                mean_contour - sd,
                mean_contour + sd,
                color=TONE_COLORS[group_num],
                alpha=0.15
            )

    plt.xlabel('Normalized Word Length (%)')
    plt.ylabel('Average Fundamental Frequency (F0) [Hz]')
    plt.legend(
        title="Tone Group",
        loc='lower left',       # where it is
        bbox_to_anchor=(0, 0),  
        frameon=True,
        fontsize=10
    )
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()


    outname = f"average_contours_{syllabic_type.lower()}.pdf"
    fig.savefig(outname, format="pdf")
    plt.show()

    print(f"Lagret PDF: {outname}")









def run_analysis():
    df_filtered, groups, f0_cols = load_and_group_data(INPUT_CSV_FILE)

    if df_filtered is None:
        return

    plot_overall_average_contours(groups, f0_cols)
    plot_average_by_structure(df_filtered, f0_cols, 'Monosyllabic')
    plot_average_by_structure(df_filtered, f0_cols, 'Sesquisyllabic')


if __name__ == '__main__':
    run_analysis()
