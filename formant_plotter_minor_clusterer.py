import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import to_rgb 
from matplotlib.colors import to_hex 
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines

# warnings gone
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
#FONT
ipa_font_prop = FontProperties(family=['Inter'], size=12)


custom_colors = {
    "a": "#f8766d",      # H=0° (Red)
    #"e": "#e38900",   # H=30° (Red-Orange)
    #"i": "#c49a00",   # H=60° (Yellow)
    #"o": "#99a800", # H=90° (Yellow-Green)
    #"u": "#53b400",    # H=120° (Green)
    "ə": "#00bc56", # H=150° (Cyan-Green)
    #"": "#01c094",     # H=180° (Cyan)
    #"ɯ": "#00bfc4", # H=210° (Sky Blue)
    #"blue": "#00b6eb",     # H=240° (Blue/Indigo)
    "e": "#04a4ff",   # H=270° (Violet)
    #"ə": "#a58aff",  # H=300° (Magenta)
    #"rose": "#df70f8",      # H=330° (Rose)
    #"a": "#fb61d7",
    #"magenta": "#ff66a8",
    }

#CSV_PATH = "formants_normalized.csv"
CSV_PATH = "formants_full_smoothed_2_manual.csv"

#FEATURE_COLUMNS = ['norm_mid_F1', 'norm_mid_F2']
FEATURE_COLUMNS = ['F1_mid', 'F2_mid']

LABEL_COLUMN = 'Label' 
K_MIN = 1
K_MAX = 6


Z_SCORE_THRESHOLD = 2.0 

CONSONANT_GROUPS = {
    # LABIAL
    'p': 'Labial', 'pʰ': 'Labial', 'b': 'Labial', 'm': 'Labial', 'f': 'Labial', 
    'v': 'Labial', 'w': 'Labial', 'mʷ': 'Labial',
    
    # CORONAL
    't': 'Coronal', 'tʰ': 'Coronal', 'd': 'Coronal', 
    'n': 'Coronal', 's': 'Coronal', 'z': 'Coronal', 
    'l': 'Coronal', 'r': 'Coronal', 'ɹ': 'Coronal',
    't͡s': 'Coronal', 't͡θ': 'Coronal', 'zʷ': 'Coronal', 
    'ð': 'Coronal', 'ɬ': 'Coronal', 'θ': 'Coronal',
    
    # VELAR
    'k': 'Velar', 'kʰ': 'Velar', 'g': 'Velar', 'ŋ': 'Velar', 'ŋʷ': 'Velar',
    
    # PALATAL (ALVEO-PALATAL)
    'j': 'Palatal', 'ʃ': 'Palatal', 'ʒ': 'Palatal', 't͡ʃ': 'Palatal', 
    't͡ʃ̺': 'Palatal', 'd͡ʒ': 'Palatal', 'h': 'Palatal',
    
    # OTHER
    'kx': 'Velar', 'q': 'Uvular', 'ʁ': 'Uvular',
    'sk': 'Coronal', 'pɹ': 'Labial', 'tɹ': 'Coronal',
    'kʰ,':'Velar',
    't͡͡ʃ̺': 'Palatal',
}


# --- Mock Data Generation ---
# IMPORTANT: This block provides data if the CSV is not found.
try:
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded data from {CSV_PATH}. Using real data.")
except FileNotFoundError:
    print(f"csv not found")
    data = 1
    df = pd.DataFrame(data)

def map_consonant_to_group(segment):
    """Mappe enkeltkonsonantsegment te gruppe."""

    if pd.isna(segment):
        return 'Unknown'
    return CONSONANT_GROUPS.get(str(segment).lower(), 'Unknown')

def find_optimal_clusters(data, k_min, k_max):
    """
    Elbow method for k clusters
    """
    inertia_scores = []
    k_range = range(k_min, k_max + 1)
    
    print(f"Elbow method test for k={k_min} to k={k_max}")
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertia_scores.append(kmeans.inertia_)
    



    plt.figure(figsize=(8, 6))
    plt.plot(k_range, inertia_scores, marker='o')
    plt.title('Elbow Method to Determine Optimal Cluster Count (k)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-Cluster Sum of Squares - WCSS)')
    plt.grid(True)
    plt.show()

def get_label_text(row, label_col):
    """Get label text"""
    
    next_seg = row['NextSeg'] if 'NextSeg' in row and pd.notna(row.get('NextSeg')) else ""
    english_gloss = row['English'] if 'English' in row and pd.notna(row.get('English')) else ""
    
    label_text = ""
    if next_seg:
        label_text = next_seg
        if english_gloss:
            label_text = f"{next_seg} ({english_gloss})"
    elif english_gloss:
        label_text = f"({english_gloss})"
    else:
        original_label = row[label_col]
        label_text = original_label.lstrip('_')
        
    return label_text
    
def plot_vowel_space(df, n_clusters, features, label_col):
    """
    Plots vowel space w clusters
    """
    print(f"\nPlotting clustered data into {n_clusters}")

    f1_col, f2_col = features
    fig, ax = plt.subplots(figsize=(10, 6))  # 5:3 ratio

    # need colour
    cluster_colors = {cluster: custom_colors.get(v, "#999999")
                      for cluster, v in enumerate(sorted(df[label_col].str.replace("_", "", regex=False).unique()))}

    # If n_clusters > number of colours
    cluster_list = sorted(df['Acoustic_Cluster'].unique())
    for i, cluster in enumerate(cluster_list):
        subset = df[df['Acoustic_Cluster'] == cluster]
        # Pick color
        color_keys = list(custom_colors.keys())
        color = custom_colors[color_keys[i % len(color_keys)]]
        ax.scatter(subset[f2_col], subset[f1_col], color=color, s=60, alpha=0.8, label=f'Cluster {cluster}')



    # Axis settings
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlim(2500, 600)
    ax.set_ylim(800, 200)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_label_position('right')
    ax.set_xlabel("F2 (Hz)", fontsize=12, fontproperties=ipa_font_prop)
    ax.set_ylabel("F1 (Hz)", fontsize=12, fontproperties=ipa_font_prop)
    ax.legend(title="Acoustic Cluster", bbox_to_anchor=(1, 1), loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(fontproperties=ipa_font_prop)
    plt.yticks(fontproperties=ipa_font_prop)
    plt.tight_layout()



    # Save
    output_plot = "vowels_acoustic_clusters.png"
    plt.savefig(output_plot, dpi=300)
    plt.savefig(output_plot.replace(".png", ".pdf"), format='pdf', bbox_inches='tight')
    plt.show()



def plot_vowel_space_by_consonant_group(df, features, label_col):
    """
    Plotte vokalan etter farge
    """
    print(f"\nPlotting data-----------")

    f1_col, f2_col = features
    fig, ax = plt.subplots(figsize=(10, 6))  # 5:3 ratio


    CUSTOM_ORDER = ['Labial', 'Coronal', 'Palatal', 'Velar', 'Uvular'] 
    
    # COLOURS
    COLOR_MAP_DEFINITIONS = {
    "Labial": "#f8766d",      # H=0° (Red)
    "e": "#e38900",   # H=30° (Red-Orange)
    "i": "#c49a00",   # H=60° (Yellow)
    "o": "#99a800", # H=90° (Yellow-Green)
    "": "#53b400",    # H=120° (Green)
    "": "#00bc56", # H=150° (Cyan-Green)
    "": "#01c094",     # H=180° (Cyan)
    "Coronal": "#00bfc4", # H=210° (Sky Blue)
    "Velar": "#00b6eb",     # H=240° (Blue/Indigo)
    "Uvular": "#04a4ff",   # H=270° (Violet)
    "Palatal": "#a58aff",  # H=300° (Magenta)
    "": "#df70f8",      # H=330° (Rose)
    "": "#fb61d7",
    "": "#ff66a8",
    }



    actual_groups = list(df['NextSeg_Group'].unique())
    group_color_map = {g: COLOR_MAP_DEFINITIONS.get(g, '#999999') for g in actual_groups}

    final_plot_order = [g for g in CUSTOM_ORDER if g in actual_groups]
    remaining_groups = sorted([g for g in actual_groups if g not in final_plot_order])
    final_plot_order.extend(remaining_groups)
    
    for group_name in final_plot_order: #fix this---------------------------------------------------------
        subset = df[df['NextSeg_Group'] == group_name]
        color = group_color_map[group_name]
        ax.scatter(subset[f2_col], subset[f1_col], color=color, s=60, alpha=0.8, label=group_name)
    
    
    # Axis settings
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlim(2500, 600)
    ax.set_ylim(800, 200)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_label_position('right')
    ax.set_xlabel("F2 (Hz)", fontsize=12, fontproperties=ipa_font_prop)
    ax.set_ylabel("F1 (Hz)", fontsize=12, fontproperties=ipa_font_prop)

    handles = [mlines.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=group_color_map[g], markersize=8, label=g)
               for g in final_plot_order] # <--- CHANGED HERE
    ax.legend(handles=handles, title="Following Consonant", bbox_to_anchor=(1, 1), loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(fontproperties=ipa_font_prop)
    plt.yticks(fontproperties=ipa_font_prop)
    plt.tight_layout()



    # Save
    output_plot = "vowels_by_consonant_group.png"
    plt.savefig(output_plot, dpi=300)
    plt.savefig(output_plot.replace(".png", ".pdf"), format='pdf', bbox_inches='tight')
    plt.show()

def main():
    global df
    
    try:
        required_cols = FEATURE_COLUMNS + [LABEL_COLUMN, 'NextSeg'] # Add nextseg
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            print(f"Error: CSV/Mock data must contain the columns {required_cols}. Missing: {missing_cols}")
            return
            
        if 'English' not in df.columns:
            print("Warning: 'English' column not found. Vowel labels will be limited to 'NextSeg'.")

        initial_count = len(df)
        
        df_filtered = df[
            df[LABEL_COLUMN].astype(str).str.match(r'^_.{1}$', na=False)
        ].copy()

        print(f"Data filtered from {initial_count} rows down to {len(df_filtered)} rows.")
        
        if df_filtered.empty:
            print(f"Error: No data rows matched the strict filter ('^_.$'). Exiting.")
            return
        df_cleaned = df_filtered.dropna(subset=FEATURE_COLUMNS)
        
        nan_dropped_count = len(df_filtered) - len(df_cleaned)
        if nan_dropped_count > 0:
            print(f"Warning: Dropped {nan_dropped_count} rows due to missing F1 or F2 values (NaN).")
        
        
        print(f"\n4. Filtering outliers (Z-score > {Z_SCORE_THRESHOLD}) for F1_mid and F2_mid...")





        z_scores = np.abs((df_cleaned[FEATURE_COLUMNS] - df_cleaned[FEATURE_COLUMNS].mean()) / df_cleaned[FEATURE_COLUMNS].std())
        
        df_outliers_removed = df_cleaned[((z_scores < Z_SCORE_THRESHOLD).all(axis=1))].copy()
        
        outliers_dropped_count = len(df_cleaned) - len(df_outliers_removed)
        
        if outliers_dropped_count > 0:
            print(f"Dropped {outliers_dropped_count} rows identified as outliers (Z-score > {Z_SCORE_THRESHOLD}).")
        
        #
        df_cleaned = df_outliers_removed # this is better

        if len(df_cleaned) < K_MIN:
            print(f"Error: Too few data points ({len(df_cleaned)}) remain after cleaning and outlier removal to run clustering.")
            return
            
        print(f"Final data size for clustering: {len(df_cleaned)} rows.")

        X = df_cleaned[FEATURE_COLUMNS].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        find_optimal_clusters(X_scaled, K_MIN, K_MAX)
        
        while True:
            try:
                n_clusters = int(input(f"\nwhere is elbow, give num between {K_MIN} and {K_MAX}): "))
                if K_MIN <= n_clusters <= K_MAX:
                    break
                else:
                    print("wtong")
            except ValueError:
                print("wrong")


        print(f"Doing final clustering with k={n_clusters}")
        kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_cleaned.loc[:, 'Acoustic_Cluster'] = kmeans_final.fit_predict(X_scaled)
        
        

        print("\n9. Mapping NextSeg consonants to broad phonetic groups...")
        df_cleaned.loc[:, 'NextSeg_Group'] = df_cleaned['NextSeg'].apply(map_consonant_to_group)

        plot_vowel_space(df_cleaned, n_clusters, FEATURE_COLUMNS, LABEL_COLUMN)
        

        plot_vowel_space_by_consonant_group(df_cleaned, FEATURE_COLUMNS, LABEL_COLUMN)
        
        # stats
        print("\nDistribution of data in new acoustic clusters:")
        print(df_cleaned['Acoustic_Cluster'].value_counts().sort_index())
        


        centroids_scaled = kmeans_final.cluster_centers_
        centroids_hz = scaler.inverse_transform(centroids_scaled)
        centroids_df = pd.DataFrame(centroids_hz, columns=FEATURE_COLUMNS)
        
        print("\nCluster centroids:")
        print(centroids_df)
        
        print("\n----ferdig analyse-------")
        print(f"brukt {len(df_cleaned)} datapunkt.")

    except FileNotFoundError:
        print(f"E: {CSV_PATH} fins itj")
    except Exception as e:
        print(f"{e}")

if __name__ == "__main__":
    main()