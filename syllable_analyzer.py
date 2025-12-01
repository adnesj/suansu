import pandas as pd
import numpy as np
import collections
import warnings
import re
INPUT_CSV = "formants_full_smoothed_2_manual.csv"
OUTPUT_TXT = "syllable_analysis.txt"
GLIDES = {'j', 'w', 'ɰ'}
VOWEL_MARKER = '_'
DIGIT_PATTERN = re.compile(r'[123]')







def clean_segment_label(label):
    """
    rensk
    """
    s = str(label).strip()
    if s == '':
        return 'Ø (Null)'

    if s.startswith(VOWEL_MARKER) and s.endswith(VOWEL_MARKER) and len(s) > 2:
        base_vowel = s.strip(VOWEL_MARKER).replace("̃", "(nasal)")
        return f'Vowel /{base_vowel}/'
    
    return f'/{s}/'

def clean_vowel_label_for_word_search(label):
    """fjerne underscore"""
    s = str(label).strip()
    if s.startswith(VOWEL_MARKER) and s.endswith(VOWEL_MARKER) and len(s) > 2:
        return s.strip(VOWEL_MARKER)
    return None

def extract_word_final_env(row):
    """
    tar ut endelige
    """
    word = str(row['Word'])
    label = str(row['Label'])
    english = str(row['English'])#hvis bli feil

    target_vowel = clean_vowel_label_for_word_search(label)
    if not target_vowel:
        # This shouldn't happen
        return (None, f"ERROR: Label '{label}' not recognized as vowel.")

    start_index = word.rfind(target_vowel)

    if start_index == -1:
        failure_data = {
            'Label': label,
            'Word': word,
            'English': english
        }
        return (target_vowel, failure_data)
    
    final_environment_raw = word[start_index:]
    final_environment_cleaned = DIGIT_PATTERN.sub('', final_environment_raw)
    
    return (target_vowel, final_environment_cleaned)


def sort_counter_dict(data_dict):
    """Sortere telleordbok"""
    return sorted(data_dict.items(), key=lambda item: item[1], reverse=True)

def run_analysis():
    
    try:
        warnings.simplefilter(action='ignore', category=FutureWarning)
        df = pd.read_csv(INPUT_CSV)
        # Replace NaNs with empty strings for consistency
        df = df.replace(np.nan, '', regex=True) 
        
    except FileNotFoundError:
        print(f"{INPUT_CSV} is not")
        return
    except Exception as e:
        print(f"{e}")
        return





    glide_environments = collections.defaultdict(collections.Counter)
    df_glides = df[df['Label'].isin(GLIDES)].copy()
    
    for _, row in df_glides.iterrows():
        current_glide = str(row['Label']).strip()
        raw_prev_seg = str(row['PrevSeg']).strip()
        preceding_seg = clean_segment_label(raw_prev_seg)
        glide_environments[current_glide][preceding_seg] += 1
        
    word_final_environments = collections.defaultdict(collections.Counter)
    failed_extractions = []
    




    df_word_final_vowels = df[df['Label'].str.endswith(VOWEL_MARKER, na=False)].copy()

    for _, row in df_word_final_vowels.iterrows():
        vowel_result, env_result = extract_word_final_env(row)
        
        if isinstance(env_result, dict):
            failed_extractions.append(env_result)
        elif vowel_result:
            word_final_environments[vowel_result][env_result] += 1

        #RAPPORTSKRIVING
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(" COMPREHENSIVE PHONOLOGICAL ANALYSIS\n")
        f.write(f" (Input file: {INPUT_CSV})\n")
        f.write("=" * 80 + "\n\n")
        


        f.write("SECTION 1: DIRECT PRECEDING ENVIRONMENTS FOR GLIDES\n")
        f.write("--------------------------------------------------\n")
        f.write("Focus: What segment (PrevSeg) immediately precedes a glide (Label = G)?\n")
        f.write("Key: Vowel /.../ = Vowel Nucleus, /C/ = Consonant, Ø (Null) = Word/Syllable Initial\n\n")

        total_glide_tokens = len(df_glides)
        f.write(f"Total Glide Tokens Analyzed: {total_glide_tokens}\n")
        f.write("-" * 50 + "\n")

        if total_glide_tokens > 0:
            for glide, environments in glide_environments.items():
                f.write(f"\n--> ENVIRONMENTS PRECEDING GLIDE /{glide}/ (N={sum(environments.values())}):\n")
                
                # Sort environments by count
                for seg, count in sort_counter_dict(environments):
                    f.write(f"  {seg:<30} : {count} tokens\n")
            
            f.write("\n" + "=" * 80 + "\n\n")

        else:
            f.write("No tokens were found where the 'Label' column contained a target glide (j, w, or ɰ).\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
        
        f.write("SECTION 2: WORD FINAL ENVIRONMENT ANALYSIS\n")
        f.write("----------------------------------------\n")
        f.write("Focus: What segments follow a Vowel Nucleus at the end of a Word?\n\n")
        
        if word_final_environments:
            sorted_vowels = sorted(word_final_environments.keys())
            
            for vowel in sorted_vowels:
                environments_counter = word_final_environments[vowel]
                total_vowel_tokens = sum(environments_counter.values())
                
                f.write(f"Vowel /{vowel}/ ({total_vowel_tokens} tokens)\n")
                f.write("-" * (len(vowel) + 12) + "\n")

                unique_environments = sorted(environments_counter.keys(), key=lambda k: environments_counter[k], reverse=True)
                
                f.write("  Detailed Distribution (Final Environment: Count):\n")
                for env in unique_environments:
                    f.write(f"    '{env}' : {environments_counter[env]} tokens\n")

                final_segments = collections.Counter()
                for env, count in environments_counter.items():
                    coda = env[len(vowel):] 
                    if coda == '':
                        final_segments['Ø (Word-Final Vowel)'] += count
                    else:
                        final_segments[coda] += count
                        
                f.write("\n  Summary of Word-Final Coda/Segments Allowed:\n")
                for coda, count in sort_counter_dict(final_segments):
                    f.write(f"    {coda:<30} : {count} tokens\n")
                
                f.write("\n")

        else:
            f.write("No word-final vowels were found in the dataset to analyze final environments.\n")
            
        f.write("\n" + "=" * 80 + "\n\n")
        
        
        f.write("SECTION 3: ERROR LOG (Vowel from 'Label' NOT found in 'Word')\n")
        f.write("-------------------------------------------------------------\n")
        
        if failed_extractions:
            f.write(f"The word-final environment extraction failed for {len(failed_extractions)} tokens.\n")
            f.write("Please review these lines for segmentation/transcription errors:\n\n")
            f.write("{:<20} | {:<25} | {:<40}\n".format("LABEL (Vowel)", "WORD", "ENGLISH"))
            f.write("-" * 80 + "\n")
            
            for fail in failed_extractions:
                f.write("{:<20} | {:<25} | {:<40}\n".format(fail['Label'], fail['Word'], fail['English']))
        else:
            f.write("All Vowel Nuclei found in the data were successfully matched within their corresponding 'Word' entry. No errors to report.\n")
            
        f.write("\n" + "=" * 80 + "\n")


    print(f"\nComplete two-part analysis finished. Results written to '{OUTPUT_TXT}'.")

if __name__ == "__main__":
    run_analysis()