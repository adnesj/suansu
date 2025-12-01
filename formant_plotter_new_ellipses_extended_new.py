import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import matplotlib as mpl
from scipy.stats import chi2
import warnings
import os

from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

# no useless warnings more
warnings.simplefilter(action='ignore', category=FutureWarning)

mpl.rcParams["font.family"] = "Inter"
mpl.rcParams["font.sans-serif"] = ["Inter"]

# CONFIG################SETUP
INPUT_CSV = "formants_full_smoothed_2_manual.csv"
OUTPUT_PLOT = "vowel_space_with_ellipses_relaxed_filter.png"
EXCLUSION_FILE = "formant_plotter_new_ellipses_ignore.txt"
F1_MID_COL = 'F1_mid'
F2_MID_COL = 'F2_mid'
F3_MID_COL = 'F3_mid'
EXCLUDED_SEGMENTS = ["j", "w", "ɥ", "ɰ"]
#ellipse settings the three below
OUTLIER_Z_SCORE_THRESHOLD = 2.0
CONFIDENCE_LEVEL = 0.95
CHISQ_K = chi2.ppf(CONFIDENCE_LEVEL, 2)

def load_exclusion_list(filepath):
    """
    Loads words to be excluded, but i stopped using this
    """
    excluded_words = set()
    if not os.path.exists(filepath):
        print(f"Warning: Exclusion file '{filepath}' not found. No tokens be excluded from ellipse calculation.")
        return excluded_words
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            excluded_words = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(excluded_words)}")
        return excluded_words
    except Exception as e:
        print(f"Error: {e}")
        return excluded_words

def get_confidence_ellipse(data):
    """
    Får ellipsedata for formantverdia
    """
    if len(data) < 3:
        # minder så blitje nå særlig
        return None 

    mu = data.mean().values
    cov_matrix = np.cov(data.T)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
    
    order = eigen_values.argsort()
    eigen_values = eigen_values[order]
    eigen_vectors = eigen_vectors[:, order]
    
    major_axis_eigenvalue = eigen_values[1]
    minor_axis_eigenvalue = eigen_values[0]
    major_axis_vector = eigen_vectors[:, 1]

    width = 2 * np.sqrt(CHISQ_K * major_axis_eigenvalue)
    height = 2 * np.sqrt(CHISQ_K * minor_axis_eigenvalue)
    
    angle_rad = np.arctan2(major_axis_vector[1], major_axis_vector[0])
    angle_deg = np.degrees(angle_rad)

    return mu[0], mu[1], width, height, angle_deg


def plot_vowel_space(df_plot, excluded_words):
    if df_plot.empty:
        print("No data points left after filtering, no plot made.")
        return

    # COSTUM COLOURS
    custom_colors = {
        "a": "#b15928",
        "e": "#1f78b4",
        "i": "#6a3d9a",
        "o": "#fb9a99",
        "u": "#e31a1c",
        "ə": "#008080",
        "ʌ": "#fdbf6f",
        "ɯ": "#ff7f00"
    }

    custom_colors = {
    "a": "#f8766d",      # H=0° (Red)
    "ʌ": "#e38900",   # H=30° (Red-Orange)
    "": "#c49a00",   # H=60° (Yellow)
    "": "#bbcc00", # H=90° (Yellow-Green)
    "ə": "#53b400",    # H=120° (Green)
    "": "#00bc56", # H=150° (Cyan-Green)
    "o": "#01c094",     # H=180° (Cyan)
    "u": "#00bfc4", # H=210° (Sky Blue)
    "blue": "#00b6eb",     # H=240° (Blue/Indigo)
    "ɯ": "#04a4ff",   # H=270° (Violet)
    "i": "#a58aff",  # H=300° (Magenta)
    "rose": "#df70f8",      # H=330° (Rose)
    "e": "#fb61d7",
    "magenta": "#ff66a8",
    }


    #CUSTOM ORDER
    custom_order = ["a", "ʌ", "ə", "o", "u", "ɯ", "i", "e"]

    unique_vowels = [v for v in custom_order if v in df_plot["Vowel"].unique()]

    vowel_color_map = {v: custom_colors.get(v, "#000000") for v in unique_vowels}

    # 5:3 LOOKS NICE on thesis
    plt.figure(figsize=(15, 9))
    ax = plt.gca()

    df_for_ellipse_calc = df_plot[~df_plot['English'].isin(excluded_words)].copy()
    print(f"Tokens excluded: {len(df_plot) - len(df_for_ellipse_calc)}")

    for vowel, group_df_full in df_for_ellipse_calc.groupby("Vowel"):
        group_df = group_df_full.copy()

        if len(group_df) < 3:
            continue

        f1_z = (group_df['F1_avg'] - group_df['F1_avg'].mean()) / group_df['F1_avg'].std()
        f2_z = (group_df['F2_avg'] - group_df['F2_avg'].mean()) / group_df['F2_avg'].std()
        mask = (np.abs(f1_z) <= OUTLIER_Z_SCORE_THRESHOLD) & (np.abs(f2_z) <= OUTLIER_Z_SCORE_THRESHOLD)
        df_clean = group_df[mask].copy()

        ellipse_params = get_confidence_ellipse(df_clean[['F2_avg', 'F1_avg']])
        if ellipse_params:
            center_f2, center_f1, width, height, angle_deg = ellipse_params

            ellipse = Ellipse(
                xy=(center_f2, center_f1),
                width=width,              
                height=height,
                angle=angle_deg,
                facecolor=vowel_color_map[vowel],
                edgecolor=vowel_color_map[vowel],
                alpha=0.15,
                linewidth=2,
                zorder=1
            )
            ax.add_patch(ellipse)

    for _, row in df_plot.iterrows():
        v = row["Vowel"]
        f1 = row["F1_avg"]
        f2 = row["F2_avg"]
        lbl = row["Full_Display_Label"]

        color = vowel_color_map[v]

        ax.scatter(
            f2, f1,
            color=color,
            s=70,
            alpha=0.9,
            edgecolors='none',  #outline around points
            zorder=2
        )
        if lbl == "_#":
            lbl = ""
        lbl = lbl.lstrip("_")
        ax.text(
            f2 - 10, f1 - 0,
            lbl,
            fontsize=12,
            color="black",
            va="center",
            ha="left",
            zorder=3
        )
#AXIS SETTINGS
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("Average F2 (Hz)", fontsize=12)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_ylabel("Average F1 (Hz)", fontsize=12)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_xlim(2500, 600)
    ax.set_ylim(800, 200)
    ax.xaxis.set_major_locator(plt.MultipleLocator(100))
    ax.yaxis.set_major_locator(plt.MultipleLocator(50))
    ax.tick_params(axis='both', labelsize=12)
    ax.minorticks_on()

    #LEGEND SETTINGS
    vowel_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=vowel_color_map[v],
                   markersize=10,
                   label=v,
                   markeredgecolor='none')
        for v in unique_vowels
    ]

    ax.legend(handles=vowel_handles,
              title="Vowel Category",
              loc="lower left",
              fontsize=12,
              title_fontsize=12)

    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("vowel_space_output.pdf", dpi=300)
    plt.show()

    print("\nSaved PDF: vowel_space_output.pdf")


def main():
    # Load the exclusion list firt
    excluded_words = load_exclusion_list(EXCLUSION_FILE)
    
    try:
        # Load the data
        df = pd.read_csv(INPUT_CSV)
        print(f"Loaded {len(df)} total rows from {INPUT_CSV}.")
        
        required_cols = ['Label', 'PrevSeg', 'NextSeg', 'English', F1_MID_COL, F2_MID_COL, F3_MID_COL]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"Error: Missing required columns: {missing}. Please ensure F1_mid, F2_mid, and F3_mid are present.")
            return

    except FileNotFoundError:
        print(f"Error: The file '{INPUT_CSV}' was not found. Please check the path.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        return

    
    cond1_label_ends_underscore = df['Label'].astype(str).str.endswith('_', na=False)
    cond2_no_nasalization = ~df['Label'].astype(str).str.contains('̃', na=False)
    cond3_no_underscore_in_context = (
        ~df['PrevSeg'].astype(str).str.contains('_', na=False) &
        ~df['NextSeg'].astype(str).str.contains('_', na=False)
    )
    
    cond4_no_glides_context = (
        ~df['PrevSeg'].astype(str).isin(EXCLUDED_SEGMENTS)
    )
    

    df_filtered = df[
        cond1_label_ends_underscore & 
        cond2_no_nasalization & 
        cond3_no_underscore_in_context & 
        cond4_no_glides_context 
    ].copy()

    print(f"Filtered  to {len(df_filtered)} tokens that meet the criteria")
    
    if df_filtered.empty:
        print("Nnot workig.")
        return

    
    df_filtered['F1_avg'] = df_filtered[F1_MID_COL]
    df_filtered['F2_avg'] = df_filtered[F2_MID_COL]
    df_filtered['F3_avg'] = df_filtered[F3_MID_COL] 
    
    df_filtered['English'] = df_filtered['English'].fillna('').astype(str).str.strip()
    
    initial_rows_count = len(df_filtered)
    df_filtered.dropna(subset=['F1_avg', 'F2_avg'], inplace=True)
    rows_dropped_for_coords = initial_rows_count - len(df_filtered)
    
    print(f"Dropped {rows_dropped_for_coords} rows due to missing F1/F2 coordinates.")
    
    df_filtered['Rounding_Metric'] = df_filtered['F3_avg'] - df_filtered['F2_avg']
    
    
    df_filtered["Vowel"] = df_filtered["Label"].astype(str).str.replace("_", "", regex=False)
    
    df_filtered['PrevSeg_Display'] = df_filtered['PrevSeg'].fillna('').astype(str).replace('', '#')
    df_filtered['NextSeg_Display'] = df_filtered['NextSeg'].fillna('').astype(str).replace('', '#')
    
    df_filtered['English_Display'] = df_filtered['English'] 
    
    df_filtered['Contextual_Label'] = (
        #df_filtered['PrevSeg_Display'] + 
        '_' +
        df_filtered['NextSeg_Display']
    )
    
    def create_metric_line(row):
        metric = row['Rounding_Metric']
        if pd.notna(metric):
            return f'{metric:.0f}'
        else:
            return ''

    df_filtered['Metric_Display'] = df_filtered.apply(create_metric_line, axis=1)

    df_filtered['Full_Display_Label'] = (
        df_filtered['Contextual_Label'])

    # FINAL PLOT
    plot_vowel_space(df_filtered, excluded_words)

if __name__ == "__main__":
    main()