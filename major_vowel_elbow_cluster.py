import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import sys
#warnings away
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)





#config
#INPUT_CSV = "formants_full_smoothed_2.csv"
INPUT_CSV = "formants_full_smoothed_2_manual.csv"
OUTPUT_PLOT_CLUSTERS = "vowel_space_kmeans_clusters.png"
OUTPUT_PLOT_ELBOW = "kmeans_elbow_method.png"
F1_MID_COL = 'F1_mid'
F2_MID_COL = 'F2_mid'
F3_MID_COL = 'F3_mid' # dont need it anywas
EXCLUDED_SEGMENTS = ["j", "w", "ɥ", "ɰ"]






def plot_elbow_method(data):
    """
    Plots elbpw to graph
    """
    inertia = []
    K_range = range(1, 11)
    
    X = data[[F2_MID_COL, F1_MID_COL]].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n-------k-means eval--------")
    print("calc show for k 1-10")

    for k in K_range:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300) 
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)
            print(f"K={k}, Inertia: {kmeans.inertia_:.2f}")

    plt.figure(figsize=(8, 6))
    plt.plot(K_range, inertia, marker='o', linestyle='--', color='blue')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Within-cluster Sum of Squares)')
    plt.xticks(K_range)
    plt.grid(True, linestyle=':', alpha=0.7)
        
    plt.savefig(OUTPUT_PLOT_ELBOW, dpi=300)
    plt.show() #waits until i close
    print(f"\nElbow plot save: {OUTPUT_PLOT_ELBOW}")
    
    return X_scaled 





#K_MEANS cluster analysis and plot

def perform_kmeans_and_plot(df_plot, X_scaled, K_clusters):
    print(f"\nK-Means Clustering Results ({K_clusters})")

    kmeans = KMeans(n_clusters=K_clusters, random_state=42, n_init=10, max_iter=300)
    df_plot['Cluster'] = kmeans.fit_predict(X_scaled)
    
    cluster_centers_scaled = kmeans.cluster_centers_
    
    X_original = df_plot[[F2_MID_COL, F1_MID_COL]].values
    scaler = StandardScaler()
    scaler.fit(X_original)
    
    cluster_centers_hz = scaler.inverse_transform(cluster_centers_scaled)
    
    unique_clusters = sorted(df_plot["Cluster"].unique())
    colors = cm.get_cmap("Set1", len(unique_clusters)) 
    cluster_color_map = {c: colors(i) for i, c in enumerate(unique_clusters)}

    # SETUP
    plt.figure(figsize=(12, 12)) 
    ax = plt.gca()
    for _, row in df_plot.iterrows():
        cluster = row["Cluster"] # for colours
        f1_avg = row[F1_MID_COL]
        f2_avg = row[F2_MID_COL]
        
        full_display_label = row["Full_Display_Label"] 
        color_for_point = cluster_color_map[cluster] 

        ax.scatter(f2_avg, f1_avg, color=color_for_point, s=80, alpha=0.7, edgecolors='black', linewidths=0.5)
        
        ax.text(f2_avg + 30, f1_avg - 10, full_display_label, color='black', fontsize=8, 
                 verticalalignment='center', horizontalalignment='left')
    
    for i, (f2_center, f1_center) in enumerate(cluster_centers_hz):
        ax.scatter(f2_center, f1_center, marker='*', s=600, 
                   color='yellow', edgecolors='black', linewidths=2, 
                   zorder=10, label=f'Cluster {i} Center')
        ax.text(f2_center, f1_center + 50, f'K{i}', color='black', fontsize=18, 
                weight='bold', horizontalalignment='center', verticalalignment='top', zorder=11)


    ax.invert_xaxis()  
    ax.invert_yaxis()
    ax.set_xlabel("F2 Midpoint (Hz)", fontsize=12)
    ax.set_ylabel("F1 Midpoint (Hz)", fontsize=12)
    ax.set_title(
        f"K-Means Vowel Clustering (K={K_clusters}): Points Labeled by Context and Colored by Cluster",
        fontsize=14,
        weight='bold'
    )
    
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=cluster_color_map[c], markersize=10, 
                          label=f'Cluster {c}', markeredgecolor='black', markeredgewidth=0.5)
               for c in unique_clusters]
    
    handles.append(plt.Line2D([0], [0], marker='*', color='w',
                              markerfacecolor='yellow', markersize=15, 
                              label='Cluster Center', markeredgecolor='black', markeredgewidth=2))
               
    ax.legend(handles=handles, title="Cluster Membership", loc='upper left', bbox_to_anchor=(1.0, 1.0))

    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.88, 1]) 

    plt.savefig(OUTPUT_PLOT_CLUSTERS, dpi=300)
    plt.show()
    print(f"\ncluster gen to: {OUTPUT_PLOT_CLUSTERS}")
    
    print("\nog vowel distrib")
    
    cluster_vowel_counts = df_plot.groupby('Cluster')['Vowel'].value_counts().sort_index()
    cluster_vowel_percent = df_plot.groupby('Cluster')['Vowel'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
    
    distribution_df = pd.DataFrame({
        'Count': cluster_vowel_counts,
        'Percentage': cluster_vowel_percent
    }).reset_index()
    
    print(f"\nDistribution of og vowel in each K={K_clusters} cluster:")
    
    for cluster_id in sorted(distribution_df['Cluster'].unique()):
        cluster_data = distribution_df[distribution_df['Cluster'] == cluster_id].copy()
        
        cluster_data['Count'] = cluster_data['Count'].astype(int) 
        cluster_data = cluster_data.sort_values(by='Count', ascending=False)
        
        print(f"\nCluster {cluster_id} (Total Tokens: {df_plot[df_plot['Cluster'] == cluster_id].shape[0]}):")
        for _, row in cluster_data.iterrows():
            print(f"  - {row['Vowel']:<10}: {row['Count']:>4} tokens ({row['Percentage']:>5})")


def main():
    try:
        # Load
        df = pd.read_csv(INPUT_CSV)
        print(f"Loaded {len(df)} total rows from {INPUT_CSV}.")
        
        # Check
        required_cols = ['Label', 'PrevSeg', 'NextSeg', 'English', F1_MID_COL, F2_MID_COL, F3_MID_COL]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"Error: {missing}")
            return

    except FileNotFoundError:
        print(f"Error: no {INPUT_CSV}")
        return
    except Exception as e:
        print(f"Error: {e}")
        return

    
    cond1_label_ends_underscore = df['Label'].astype(str).str.endswith('_', na=False)
    cond2_no_nasalization = ~df['Label'].astype(str).str.contains('̃', na=False)
    cond3_no_underscore_in_context = (
        ~df['PrevSeg'].astype(str).str.contains('_', na=False) &
        ~df['NextSeg'].astype(str).str.contains('_', na=False)
    )
    cond4_no_glides_context = (
        ~df['PrevSeg'].astype(str).isin(EXCLUDED_SEGMENTS) &
        ~df['NextSeg'].astype(str).isin(EXCLUDED_SEGMENTS)
    )

    df_filtered = df[
        cond1_label_ends_underscore & 
        cond2_no_nasalization & 
        cond3_no_underscore_in_context & 
        cond4_no_glides_context
    ].copy()

    print(f"Filtered to {len(df_filtered)} tokens that meet all criteria.")
    
    if df_filtered.empty:
        print("None found")
        return

    df_filtered['F1_avg'] = df_filtered[F1_MID_COL]
    df_filtered['F2_avg'] = df_filtered[F2_MID_COL]
    df_filtered['F3_avg'] = df_filtered[F3_MID_COL] 
    
    initial_rows_count = len(df_filtered)
    df_filtered.dropna(subset=['F1_avg', 'F2_avg'], inplace=True)
    rows_dropped_for_coords = initial_rows_count - len(df_filtered)
    
    print(f"Dropped {rows_dropped_for_coords} rows due to missing F1/F2 coordinates.")
    
    
    df_filtered['Rounding_Metric'] = df_filtered['F3_avg'] - df_filtered['F2_avg']
    df_filtered["Vowel"] = df_filtered["Label"].astype(str).str.replace("_", "", regex=False)
    
    df_filtered['PrevSeg_Display'] = df_filtered['PrevSeg'].fillna('').astype(str).replace('', '#')
    df_filtered['NextSeg_Display'] = df_filtered['NextSeg'].fillna('').astype(str).replace('', '#')
    df_filtered['English_Display'] = df_filtered['English'].fillna('N/A').astype(str)
    
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

    df_filtered['Full_Display_Label'] = (
        df_filtered['Contextual_Label'] 
        .str.cat(
            '(' + df_filtered['English_Display'] + ')',
            sep='\n'
        )
        .str.cat(
            df_filtered['Metric_Display'],
            sep='\n'
        )
    )

    X_scaled = plot_elbow_method(df_filtered)
    
    
    K_CLUSTERS_TO_USE = None
    while K_CLUSTERS_TO_USE is None:
        try:
            k_input = input("\nchoose k from elbow ")
            K_CLUSTERS_TO_USE = int(k_input)
            
            if K_CLUSTERS_TO_USE < 2 or K_CLUSTERS_TO_USE > 10:
                print("maybe dont  work try 2-10")
            
        except ValueError:
            print("Invalid")
            
    print(f"\n------------------------------------------")
    print(f"{K_CLUSTERS_TO_USE} clusters to use")
    print(f"--------------------------")

    perform_kmeans_and_plot(df_filtered, X_scaled, K_CLUSTERS_TO_USE)


if __name__ == "__main__":
    main()