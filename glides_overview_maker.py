import pandas as pd
import warnings
import re
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)

INPUT_GLIDES_TXT = "glides_overview.txt"
INPUT_FORMANT_CSV = "formants_full_smoothed_2_manual.csv"
OUTPUT_CSV = "glide_predictions.csv"

def clean_word(word):
    if not isinstance(word, str):
        return ""
    
    cleaned = word.replace('ˈ', '')
    
    if cleaned.startswith('a'):
        cleaned = cleaned[1:]
    elif cleaned.startswith('ə'):
        cleaned = cleaned[1:]
        
    return cleaned

def parse_glides_txt(filepath):
    parsed_data = []
    current_phonetic_glide = None
    
    # regex for å finn dem
    line_pattern = re.compile(r"^(.*?)\s+\((.*?)\)\s*\|.*?\|\s*.*?, ?([ijuvw\?ɰ])?$")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return []
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('--- j-Onglide'):
            current_phonetic_glide = 'j'
            continue
        elif line.startswith('--- w-Onglide'):
            current_phonetic_glide = 'w' 
            continue
        elif line.startswith('--- ɰ-Onglide'):
            current_phonetic_glide = 'w' 
            continue
        elif line.startswith('---'):
            current_phonetic_glide = None
            continue

        if current_phonetic_glide:
            match = line_pattern.search(line)
            if match:
                word = match.group(1).strip()
                gloss = match.group(2).strip()
                
                annotated_glide = match.group(3).strip() if match.group(3) else '?'
                
                cleaned_word = clean_word(word)
                
                parsed_data.append({
                    'Cleaned_Word': cleaned_word,
                    'English_Gloss': gloss,
                    'Annotated_Glide': annotated_glide,
                    'Phonetic_Glide_Type': current_phonetic_glide
                })
            
    return parsed_data

def get_segment_durations_and_predict(df_formants, parsed_entries):
    """
    Matsje formant, lengde, og forutsir ka d bli, glide eller vokal
    """
    results = []

    df_formants['Word_Cleaned'] = df_formants['Word'].apply(clean_word)
    df_formants['English_Cleaned'] = df_formants['English'].str.replace('\n', ' ').str.strip()
    
    df_formants['Duration'] = pd.to_numeric(df_formants['Duration'], errors='coerce')
    
    for entry in parsed_entries:
        cleaned_word = entry['Cleaned_Word']
        gloss = entry['English_Gloss']
        annotated_glide = entry['Annotated_Glide']
        phonetic_glide_type = entry['Phonetic_Glide_Type'] # 'j' or 'w'
        
        expected_onglides = ['j'] if phonetic_glide_type == 'j' else ['w', 'ɰ']
            
        match_criteria = (
            (df_formants['Word_Cleaned'] == cleaned_word) &
            (df_formants['English_Cleaned'] == gloss) &
            (df_formants['Label'].isin(expected_onglides))
        )
        
        matching_rows = df_formants[match_criteria]
        
        if matching_rows.empty:
            results.append({
                'Cut Word': cleaned_word,
                'English Gloss': gloss,
                'Annotated Glide': annotated_glide,
                'Phonetic Glide Type': phonetic_glide_type,
                'Onglide Duration (s)': "NaN",
                'Vowel Complex Duration (V [V+j]) (s)': "NaN",
                'Ratio (V/G [V+j]/G)': "NaN",
                'Predicted Glide': 'NaN (No Match)',
                'Prediction Match (Annotated)': '?',
                'Prediction_Ratio_Numeric': np.nan # dont delete this, its needed
            })
            continue

        glide_row = matching_rows.iloc[0]
        idx = glide_row.name
            
        onglide_duration = glide_row['Duration']
        vowel_duration = np.nan
        offglide_j_duration = 0.0
            
        if idx + 1 in df_formants.index and str(df_formants.loc[idx + 1, 'Label']).endswith('_'):
            vowel_row = df_formants.loc[idx + 1]
            vowel_duration = vowel_row['Duration']
                
            if idx + 2 in df_formants.index and str(df_formants.loc[idx + 2, 'Label']) == 'j':
                offglide_row = df_formants.loc[idx + 2]
                offglide_j_duration = offglide_row['Duration'] if pd.notna(offglide_row['Duration']) else 0.0
            





        primary_ratio = np.nan
        if pd.notna(vowel_duration) and pd.notna(onglide_duration) and onglide_duration > 0:
            primary_ratio = vowel_duration / onglide_duration
            
        # Complex Ratio: (Vowel + Offglide J) / Onglide
        v_plus_j_duration = (vowel_duration + offglide_j_duration) if pd.notna(vowel_duration) else np.nan
        ratio_v_plus_j_g = (v_plus_j_duration / onglide_duration) if pd.notna(v_plus_j_duration) and onglide_duration else np.nan

        final_prediction_ratio = np.nan
        

        # is it valid?
        if offglide_j_duration > 0 and pd.notna(ratio_v_plus_j_g):
            final_prediction_ratio = ratio_v_plus_j_g
        elif pd.notna(primary_ratio):
            # Otherwise, fall back to V/G ratio
            final_prediction_ratio = primary_ratio
            
        predicted_glide = 'NaN (Ratio Missing)'
        if pd.notna(final_prediction_ratio):
            if final_prediction_ratio >= 1.0:
                # Ratio >= 1.0: Predict short approximant ('j' or 'w')
                predicted_glide = "j" if phonetic_glide_type == 'j' else "w"
            else:
                # Ratio < 1.0: Predict long vocalic element ('i' or 'u')
                predicted_glide = "i" if phonetic_glide_type == 'j' else "u"
        
        prediction_match = 'NaN (No Ratio)'
        if predicted_glide != 'NaN (Ratio Missing)':
            if annotated_glide == '?':
                prediction_match = '?'
            elif predicted_glide == annotated_glide:
                prediction_match = 'Yes'
            else:
                prediction_match = 'No'

        ratio_display = f"{primary_ratio:.2f}" if pd.notna(primary_ratio) else "NaN"
        if offglide_j_duration > 0 and pd.notna(ratio_v_plus_j_g):
            # add the (V+j)/G ratio in parentheses
            ratio_display += f" ({ratio_v_plus_j_g:.2f})"

        vowel_duration_str = f"{vowel_duration:.3f}" if pd.notna(vowel_duration) else "NaN"
        v_complex_duration_display = vowel_duration_str
        if offglide_j_duration > 0 and pd.notna(v_plus_j_duration):
            v_complex_duration_display += f" ({v_plus_j_duration:.3f})"
            




            #final result
        results.append({
            'Cut Word': cleaned_word,
            'English Gloss': gloss,
            'Annotated Glide': annotated_glide,
            'Phonetic Glide Type': phonetic_glide_type,
            'Onglide Duration (s)': f"{onglide_duration:.3f}" if pd.notna(onglide_duration) else "NaN",
            'Vowel Complex Duration (V [V+j]) (s)': v_complex_duration_display,
            'Ratio (V/G [V+j]/G)': ratio_display,
            'Predicted Glide': predicted_glide,
            'Prediction Match (Annotated)': prediction_match,
            'Updated_Annotation': predicted_glide,  # <<< NEW
            'Prediction_Ratio_Numeric': final_prediction_ratio # Internal column for stats
        })

    return pd.DataFrame(results)

def analyze_ratios_and_predictions(df_results):
    """
    Analyzes the 'Predicted Glide' column and calculates statistics for the
    i/u and j/w groups based on the ratio used for prediction (V/G or (V+j)/G).
    """
    
    df_clean = df_results.dropna(subset=['Prediction_Ratio_Numeric', 'Predicted Glide'])
    
    if df_clean.empty:
        print("\nNo valid ratio or prediction data available for statistical analysis.")
        return

    vocalic_pred_glides = ['i', 'u']
    approximant_pred_glides = ['j', 'w']
    
    vowel_subset = df_clean[df_clean['Predicted Glide'].isin(vocalic_pred_glides)]['Prediction_Ratio_Numeric']
    
    glide_subset = df_clean[df_clean['Predicted Glide'].isin(approximant_pred_glides)]['Prediction_Ratio_Numeric']
    
    print("\n--- Ratio Analysis (Vocalic Complex Duration / Onglide Duration) ---")
    print("Prediction Rule: Ratio < 1.0 -> 'i/u' (Vocalic); Ratio >= 1.0 -> 'j/w' (Approximant)")
    
    print("\nStatistics by Predicted Categories:")
    
    if not vowel_subset.empty:
        print(f"--- Predicted Vocalic 'i/u' (N={len(vowel_subset)}) ---")
        print(f"  Mean Ratio: {vowel_subset.mean():.3f}")
        print(f"  Median Ratio: {vowel_subset.median():.3f}")
    else:
        print("--- Predicted Vocalic 'i/u': No data available. ---")

    if not glide_subset.empty:
        print(f"--- Predicted Approximant 'j/w' (N={len(glide_subset)}) ---")
        print(f"  Mean Ratio: {glide_subset.mean():.3f}")
        print(f"  Median Ratio: {glide_subset.median():.3f}")
    else:
        print("--- Predicted Approximant 'j/w': No data available. ---")

def analyze_prediction_accuracy(df_results):
    """
    Calculates and prints the total number and percentage of correct predictions.
    It excludes rows where the annotation was '?' or the prediction was missing.
    """
    df_analyzed = df_results[df_results['Prediction Match (Annotated)'] != '?'].copy()
    df_analyzed = df_analyzed[df_analyzed['Predicted Glide'] != 'NaN (Ratio Missing)']
    
    total_analyzed = len(df_analyzed)
    
    print("\n--- Prediction Accuracy Analysis ---")
    
    if total_analyzed == 0:
        print("No entries with clear annotations (non-'?') and valid predictions found for accuracy analysis.")
        return

    correct_predictions = df_analyzed[df_analyzed['Prediction Match (Annotated)'] == 'Yes'].shape[0]
    
    accuracy_percentage = (correct_predictions / total_analyzed) * 100 if total_analyzed > 0 else 0.0

    print(f"Total entries with clear annotation and valid prediction: {total_analyzed}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy_percentage:.2f}%")

def list_unique_reannotated_sequences(df_results):  # HAD TO ADD THIS
    """
    Creates a list of unique segment sequences based on the updated annotation,
    allowing ja vs. ia etc. to be distinguished.
    """
    sequences = []

    for _, row in df_results.iterrows():
        updated = row['Updated_Annotation']
        word = row['Cut Word']

        if not isinstance(updated, str) or updated.startswith("NaN"):
            continue

        # detect where the glide/vowel is
        # find the first j/w/i/u occurrence in the cleaned-word
        m = re.search(r"[jiwuɰ]", word)
        if not m:
            continue

        start = m.start()

        if start + 1 < len(word):
            seq = updated + word[start+1]
        else:
            seq = updated

        sequences.append(seq)

    return sorted(set(sequences))

def main():
    print(f"Starting glide duration and prediction analysis...")
    print(f"1. Reading annotated glides from {INPUT_GLIDES_TXT}")
    
    parsed_entries = parse_glides_txt(INPUT_GLIDES_TXT)
    if not parsed_entries:
        print("No valid onglide entries found to process. Aborting.")
        return

    print(f"   -> Found {len(parsed_entries)} annotated onglide entries (j, w, and ɰ groups).")
    
    try:
        df_formants = pd.read_csv(INPUT_FORMANT_CSV)
        df_formants = df_formants.reset_index(drop=True) 
        
        required_cols = ['Word', 'English', 'Label', 'Duration']
        if not all(col in df_formants.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_formants.columns]
            print(f"Error: Missing required columns in {INPUT_FORMANT_CSV}: {missing}. Aborting.")
            return

    except FileNotFoundError:
        print(f"Error: The file '{INPUT_FORMANT_CSV}' was not found. Please check the path.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        return

    print(f"2. Matching, calculating durations (V/(G) or (V+j)/(G)), and predicting category...")
    
    df_results = get_segment_durations_and_predict(df_formants, parsed_entries)
    
    if df_results.empty:
        print("No duration data could be extracted.")
        return
        
    print(f"3. Writing results to {OUTPUT_CSV}")
    
    df_results_output = df_results.drop(columns=['Prediction_Ratio_Numeric'])
    df_results_output.to_csv(OUTPUT_CSV, index=False)
    
    analyze_ratios_and_predictions(df_results)

    analyze_prediction_accuracy(df_results)

    #NEW
    print("\n--- Unique reannotated segment sequences ---")
    uniq = list_unique_reannotated_sequences(df_results)
    for s in uniq:
        print("  ", s)
    
    print(f"\nFinal results saved to '{OUTPUT_CSV}'.")


if __name__ == "__main__":
    main()