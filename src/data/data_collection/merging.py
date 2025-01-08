import pandas as pd

import pandas as pd

def merge_csv_files(file1: str, file2: str, output_file: str) -> pd.DataFrame:
    """
    Merges two CSV files into one, converts 'LABEL_X' sentiment values to corresponding textual labels
    (without modifying the original datasets), saves the result to a new CSV file, and returns the merged DataFrame.

    Parameters:
        file1 (str): Path to the first CSV file.
        file2 (str): Path to the second CSV file.
        output_file (str): Path to the output CSV file where the merged result will be saved.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    try:
        # Read the two CSV files
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # Define the mapping for sentiment labels
        label_mapping = {
            "LABEL_0": "negative",
            "LABEL_1": "neutral",
            "LABEL_2": "positive",
        }

        # Create temporary DataFrames to avoid modifying the originals
        temp_df1 = df1.copy()
        temp_df2 = df2.copy()

        # Standardize the 'sentiment' column for each temporary DataFrame
        if 'sentiment' in temp_df1.columns:
            temp_df1['sentiment'] = temp_df1['sentiment'].replace(label_mapping)
        if 'sentiment' in temp_df2.columns:
            temp_df2['sentiment'] = temp_df2['sentiment'].replace(label_mapping)

        # Merge the temporary DataFrames (stacking rows)
        merged_df = pd.concat([temp_df1, temp_df2], ignore_index=True)

        # Save the merged DataFrame to the output file
        merged_df.to_csv(output_file, index=False)

        print(f"Merged file saved to '{output_file}'.")
        return merged_df

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)