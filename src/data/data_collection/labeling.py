import pandas as pd
from tqdm import tqdm
from transformers import pipeline


def analyze_sentiments_csv(
    input_file: str,
    output_file: str,
    text_column="text",
    sentiment_model_name="cardiffnlp/twitter-roberta-base-sentiment",
):
    """
    Analyzes the sentiment of reviews in a CSV file and saves the results to a new file.

    Args:
        input_file (str): Path to the input CSV file containing reviews.
        output_file (str): Path to the output CSV file for saving results.
        text_column (str): Name of the column containing reviews (default: "text").
        sentiment_model_name (str): Name of the sentiment analysis model to use (default: "cardiffnlp/twitter-roberta-base-sentiment").

    Returns:
        None: Saves a CSV file with analyzed sentiments.
    """
    print(f"Reading reviews from {input_file}...")
    print(f"Writing reviews to {output_file}...")
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Check if the comments column exists
    if text_column not in df.columns:
        raise ValueError(f"The CSV file must have a column named '{text_column}'")

    # Load the sentiment analysis model
    sentiment_model = pipeline("sentiment-analysis", model=sentiment_model_name)

    # Function to analyze the sentiment of a single review
    def analyze_sentiment(text):
        try:
            result = sentiment_model(text[:512])[0]  # Limit to 512 characters
            return result["label"]  # Return only the sentiment label
        except Exception as e:
            return "Error"  # Return "Error" if something goes wrong

    # Analyze sentiments for all reviews with a progress bar
    tqdm.pandas()  # Enable progress bar for DataFrame operations
    df["sentiment"] = df[text_column].progress_apply(lambda x: analyze_sentiment(x))

    # Save the results to a new CSV file
    df_output = df[[text_column, "sentiment"]]
    df_output.to_csv(output_file, index=False)

    print(f"Analysis completed. Results have been saved to '{output_file}'")