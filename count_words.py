import pandas as pd
import os

# --- Configuration ---
FILE_PATH = 'formants_full_smoothed_2_manual.csv'
TARGET_COLUMN = 'Label'
TARGET_SUFFIX = '_'
CONTEXT_LABELS = ['j', 'w', 'ɰ']
# ---------------------

def count_contextual_labels(file_path, target_col, target_suffix, context_labels):
    """
    Les csv-n og sjekke rada s ende på _ (så majorvokala), og telle kun ein gang for kvar avhg av samme
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure the file path is correct.")
        return

    try:
        df = pd.read_csv(file_path, low_memory=False)

        if target_col not in df.columns:
            print(f"Error: Column '{target_col}' not found in the CSV file.")
            print(f"Available columns are: {list(df.columns)}")
            return

        df[target_col] = df[target_col].astype(str)

        is_target = df[target_col].str.endswith(target_suffix)

        df['Prev_Label'] = df[target_col].shift(1)
        df['Next_Label'] = df[target_col].shift(-1)

        is_prev_context = df['Prev_Label'].isin(context_labels)
        is_next_context = df['Next_Label'].isin(context_labels)

        final_condition = is_target & (is_prev_context | is_next_context)

        final_count = final_condition.sum()

        print(f"!! FERDIG KONTEKSTANALYSE------------------")
        print(f"File analyzed: {file_path}")
        print(f"Target suffix: '{target_suffix}'")
        print(f"Required context labels: {context_labels}")
        print(f"\nTot count of target labels with required context: {final_count}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    count_contextual_labels(FILE_PATH, TARGET_COLUMN, TARGET_SUFFIX, CONTEXT_LABELS)