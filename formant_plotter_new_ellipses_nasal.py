import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import matplotlib as mpl
from scipy.stats import chi2
import warnings
import os

# make useless wasrning go away
warnings.simplefilter(action='ignore', category=FutureWarning)

mpl.rcParams["font.family"] = "Inter"
mpl.rcParams["font.sans-serif"] = ["Inter"]

# setup!!!!!
INPUT_CSV = "formants_full_smoothed_2_manual.csv"
OUTPUT_PLOT = "vowel_space_with_ellipses_nasal_glides_corrected.pdf" 
EXCLUSION_FILE = "formant_plotter_new_ellipses_nasal_ignore.txt"

F1_MID_COL = 'F1_mid'
F2_MID_COL = 'F2_mid'
F3_MID_COL = 'F3_mid'


EXCLUDED_SEGMENTS = ["w", "ɥ", "ɰ"]

# change this to adjust ellipsis shapes
OUTLIER_Z_SCORE_THRESHOLD = 2.0
CONFIDENCE_LEVEL = 0.95
CHISQ_K = chi2.ppf(CONFIDENCE_LEVEL, 2)




def reverse_tilde_for_display(ipa_symbol):
    """
    Reverse order of combining tilde and vowel so that it will
    display them correctly in legend
    """
    if '̃' in ipa_symbol:
        # Assuming the tilde is the only diacritic and is at the end
        vowel = ipa_symbol.replace('̃', '')
        return '̃' + vowel 
    return ipa_symbol

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
    """
    Making pretty plots
    """
    if df_plot.empty:
        print("No data points left after filtering, no plot made")
        return

    #! FONT ANS SIZE
    plt.rcParams['font.family'] = 'Inter'
    plt.rcParams['font.size'] = 12
    
    #! COLOURS
    custom_colors = {
        "ɛ̃": "#04a4ff", "ẽ": "#1f78b4", "ĩ": "#6a3d9a", "õ": "#fb9a99",
        "ɔ̃": "#00bc56", "ə̃": "#008080", "": "#fdbf6f", "ʌ̃": "#f8766d"
    }
    
    unique_vowels = sorted(df_plot["Vowel"].unique())
    vowel_color_map = {
        v: custom_colors.get(v, cm.get_cmap("viridis", len(unique_vowels))(i)) 
        for i, v in enumerate(unique_vowels)
    }

    # 5:3 ratio looks good in the thesis
    plt.figure(figsize=(15, 9)) 
    ax = plt.gca()

    
    df_for_ellipse_calc = df_plot[~df_plot['English'].isin(excluded_words)].copy()
    
    n_excluded_total = len(df_plot) - len(df_for_ellipse_calc)
    print(f"Tokens excluded: {n_excluded_total}")

    for vowel, group_df_full in df_for_ellipse_calc.groupby("Vowel"):
        group_df = group_df_full.copy()
        
        if len(group_df) < 3:
            print(f"[{vowel}] Not enough ellipse data after exclusion (N={len(group_df)}). Skipping ellipse.")
            continue
            
        color_for_group = vowel_color_map[vowel]
        
        f1_z = (group_df['F1_avg'] - group_df['F1_avg'].mean()) / group_df['F1_avg'].std()
        f2_z = (group_df['F2_avg'] - group_df['F2_avg'].mean()) / group_df['F2_avg'].std()
        
        outlier_mask = (np.abs(f1_z) <= OUTLIER_Z_SCORE_THRESHOLD) & \
                       (np.abs(f2_z) <= OUTLIER_Z_SCORE_THRESHOLD)
        
        df_clean = group_df[outlier_mask].copy()
        
        n_dropped = len(group_df) - len(df_clean)
        if n_dropped > 0:
            print(f"[{vowel}] Dropped {n_dropped}/{len(group_df)} remaining points (>{OUTLIER_Z_SCORE_THRESHOLD} SD) for ellipse calculation.")
        
        # CALC ELIPSE
        data_for_ellipse = df_clean[['F2_avg', 'F1_avg']] 
        ellipse_params = get_confidence_ellipse(data_for_ellipse)
        
        if ellipse_params:
            center_f2, center_f1, width, height, angle_deg = ellipse_params
            
            ellipse = Ellipse(
                xy=(center_f2, center_f1), 
                width=width, 
                height=height, 
                angle=angle_deg,
                edgecolor=color_for_group,
                facecolor=color_for_group,
                alpha=0.15, 
                linestyle='-',
                linewidth=2,
                zorder=1 
            )
            ax.add_patch(ellipse)






    for _, row in df_plot.iterrows():
        vowel = row["Vowel"]
        f1_avg = row["F1_avg"]
        f2_avg = row["F2_avg"]
        
        full_display_label = row["Full_Display_Label"]
        color_for_point = vowel_color_map[vowel]

        if row['English'] in excluded_words:
            edge_color = 'red'
            marker_style = 'D' 
            size = 90
            z_order = 3
        else:
            edge_color = 'none'
            marker_style = 'o' 
            size = 80
            z_order = 2
            
        ax.scatter(f2_avg, f1_avg, color=color_for_point, s=size, alpha=0.8, 
                   edgecolors='none', linewidths=0.8, zorder=z_order, marker=marker_style)
        
        # LABEL FOR POINT
        full_display_label = full_display_label.lstrip("_")
        ax.text(f2_avg - 10, f1_avg - 0, full_display_label, color='black', fontsize=12, 
                 verticalalignment='center', horizontalalignment='left', zorder=3)

    
    ax.invert_xaxis()
    ax.invert_yaxis()

    # axis placement and span and tick adjust
    ax.set_xlim(2500, 600) 
    ax.xaxis.set_major_locator(plt.MultipleLocator(100))
    ax.set_ylim(800, 200) 
    ax.yaxis.set_major_locator(plt.MultipleLocator(50))
    ax.set_xlabel("Average F2 (Hz)")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_ylabel("Average F1 (Hz)")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.tick_params(axis='both', labelsize=12)
    ax.minorticks_on()
    
    
    vowel_handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=vowel_color_map[v], markersize=10, 
                          label=reverse_tilde_for_display(v), 
                          markeredgecolor='none', markeredgewidth=0) # its broken
               for v in unique_vowels]
    
    all_handles = vowel_handles
    
    # PLACE LEGEDND INSIDE RAPH
    ax.legend(handles=all_handles, 
              title="Vowel Category", 
              loc='lower left', # this doesnt mean lower left of the plot area smh, how to do it
              fontsize=10,
              title_fontsize=12)

    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout() 

    # PDF SAVE
    plt.savefig(OUTPUT_PLOT, format='pdf', dpi=300)
    plt.show()
    print(f"\nSuccessfully generated plot: {OUTPUT_PLOT}")

    plt.rcParams['font.family'] = 'sans-serif' 
    plt.rcParams['font.size'] = 10


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

    # THESE are the nasl vowels
    NASAL_VOWELS = ['ʌ̃', 'ɔ̃', 'ɛ̃']

    cond_nasal_vowel = df['Label'].astype(str).str.contains('|'.join(NASAL_VOWELS), na=False)

    # ignore prev j w
    cond_prevseg_ok = ~df['PrevSeg'].astype(str).isin(['j', 'w'])

    cond_nextseg_ok = ~df['NextSeg'].astype(str).isin(EXCLUDED_SEGMENTS)

    cond_label_ends_underscore = df['Label'].astype(str).str.endswith('_', na=False)

    df_filtered = df[
        cond_nasal_vowel &
        cond_prevseg_ok &
        cond_nextseg_ok &
        cond_label_ends_underscore
    ].copy()

    print(f"Filtered to {len(df_filtered)} nasal tokens matching {NASAL_VOWELS} and valid context")
    
    if df_filtered.empty:
        print("No eligible rows remain after filtering. no analysis analysis.")
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
        df_filtered['PrevSeg_Display'] + '_' +
        df_filtered['NextSeg_Display']
    )
    
    def create_metric_line(row):
        metric = row['Rounding_Metric']
        if pd.notna(metric):
            return f'{metric:.0f}'
        else:
            return ''

    df_filtered['Metric_Display'] = df_filtered.apply(create_metric_line, axis=1)

    df_filtered['NextSeg_Display'] = df_filtered['NextSeg'].fillna('').astype(str)

    df_filtered['Full_Display_Label'] = df_filtered['NextSeg_Display'].apply(
        lambda x: f"_{x}" if x not in ['', '#'] else ''
    )


    # FINAL PLOT
    plot_vowel_space(df_filtered, excluded_words)

if __name__ == "__main__":
    main()