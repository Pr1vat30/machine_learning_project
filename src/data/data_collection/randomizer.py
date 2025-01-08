import pandas as pd


def create_sample(input_path, output_path, sample_size=100000, random_state=42):
    """
    Creates a random sample from an input dataset and saves it as a new CSV file.

    Parameters:
        - input_path (str): Path to the input CSV file.
        - output_path (str): Path to save the sampled CSV file.
        - sample_size (int): Desired sample size.
        - random_state (int): Seed for reproducibility (default: 42).

    Output:
        - Saves a CSV file with the sampled data.
    """
    try:
        # Load the original dataset
        dataset = pd.read_csv(input_path)

        # Check if the dataset has enough entries
        if len(dataset) < sample_size:
            raise ValueError("The sample size exceeds the number of entries in the original dataset.")

        # Extract a random sample
        sampled_dataset = dataset.sample(n=sample_size, random_state=random_state)

        # Save the sampled data to a new CSV file
        sampled_dataset.to_csv(output_path, index=False)

        print(f"Sample of {sample_size} entries successfully created and saved to '{output_path}'.")
    except Exception as e:
        print(f"Error: {e}")