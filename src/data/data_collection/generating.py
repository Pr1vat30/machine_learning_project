import os
import json
import random
import requests
import pandas as pd
from tqdm import tqdm
import threading

class OllamaService:

    def __init__(
            self,
            input_csv,
            output_csv,
            mode="generate",  # "adjust" for corrections, "generate" for create new text
            sentiment="negative",  # "positive" - "negative" - "neutral"
            model="llama3",
            host="http://localhost:11434",
            progress_file="progress.json",
    ):
        """
        Initialize the class with the parameters needed to process the CSV file.
        """
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.mode = mode  # Mode: "adjust" or "generate"
        self.sentiment = sentiment
        self.model = model
        self.host = host
        self.progress_file = progress_file  # File to track progress

        # Load progress status if it exists
        self.progress = self.load_progress()
        self.stop_generation = False

    def adjust_text_with_ollama(self, text):
        """
        Adjust or improve a sentence using Ollama.
        """
        try:
            # Create the payload for the model
            data = {
                "model": self.model,
                "prompt": f"Adjust this sentence to improve clarity. Respond only with the new sentence : {text}",
            }

            # Make the request to the Ollama server with streaming support
            response = requests.post(
                f"{self.host}/api/generate", json=data, stream=True
            )

            # Check the response status code
            response.raise_for_status()

            # Read the response in streaming mode and construct the text
            adjusted_sentence = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            adjusted_sentence += chunk["response"]
                    except json.JSONDecodeError:
                        print(f"Error parsing the line: {line}")

            return (
                adjusted_sentence.strip()
                if adjusted_sentence
                else "No response generated."
            )

        except requests.exceptions.RequestException as e:
            return f"Error communicating with Ollama: {e}"

    def generate_text_with_ollama(self, prompt):
        """
        Generate a new sentence using Ollama.
        """
        try:
            # Create the payload for the model
            data = {
                "model": self.model,
                "prompt": prompt,
            }

            # Make the request to the Ollama server with streaming support
            response = requests.post(
                f"{self.host}/api/generate", json=data, stream=True
            )

            # Check the response status code
            response.raise_for_status()

            # Read the response in streaming mode and construct the text
            generated_sentence = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            generated_sentence += chunk["response"]
                    except json.JSONDecodeError:
                        print(f"Error parsing the line: {line}")

            return (
                generated_sentence.strip()
                if generated_sentence
                else "No response generated."
            )

        except requests.exceptions.RequestException as e:
            return f"Error communicating with Ollama: {e}"

    def load_progress(self):
        """
        Load progress status from the progress.json file, if it exists.
        """
        if os.path.exists(self.progress_file):
            with open(self.progress_file, "r") as f:
                try:
                    progress = json.load(f)
                    return progress.get("last_processed_index", 0)
                except json.JSONDecodeError:
                    print(
                        "Error loading the progress file. Starting from the beginning."
                    )
        return 0  # Start from the first item if the progress file does not exist or is corrupted.

    def save_progress(self, index):
        """
        Save progress status to the progress.json file.
        """
        with open(self.progress_file, "w") as f:
            json.dump({"last_processed_index": index}, f)

    def process_adjust(self):
        """
        Process each sentence from the input CSV file and adjust them using Ollama.
        """
        df = pd.read_csv(self.input_csv)
        df_temp = df[["text", "sentiment"]]

        if os.path.exists(self.output_csv):
            df_output = pd.read_csv(self.output_csv)
        else:
            df_output = pd.DataFrame(columns=["text", "sentiment"])

        print(f"Starting sentence processing in adjust mode...")

        # Iterate through the CSV file
        for i, sentence in tqdm(
                enumerate(df_temp["text"]),
                desc="Processing sentences",
                unit="sentence",
                initial=self.progress,
                total=len(df_temp),
        ):
            if i < self.progress:
                continue

            adjusted_sentence = self.adjust_text_with_ollama(sentence)

            # Add the processed sentence to the output DataFrame
            df_output = pd.concat(
                [df_output, pd.DataFrame({"text": [adjusted_sentence], "sentiment": [df_temp["sentiment"][i]]})],
                ignore_index=True,
            )

            # Save the output file incrementally
            df_output.to_csv(self.output_csv, index=False)

            # Save the progress status
            self.save_progress(i + 1)

        del df_temp
        print(f"Processed file saved as {self.output_csv}")

    def generate_infinite(self):
        """
        Continuously generate sentences based on a random prompt until the user presses 'q' to stop.
        """
        print(f"Starting sentence generation in generate mode...")

        # Check if the output CSV exists or create a new DataFrame
        if os.path.exists(self.output_csv):
            df_output = pd.read_csv(self.output_csv)
        else:
            df_output = pd.DataFrame(columns=["text", "sentiment"])

        def listen_for_exit():
            input_key = input("Press 'q' to stop generating sentences.")
            if input_key == 'q':
                self.stop_generation = True

        # Start a background thread to listen for 'q' press
        thread = threading.Thread(target=listen_for_exit)
        thread.daemon = True
        thread.start()

        # Infinite loop for generating sentences
        while not self.stop_generation:
            prompts = [
                f"Create a {self.sentiment} comment about a lesson. Respond ONLY with the new sentence and max {random.randint(5, 15)} words.",
                f"Write a critical review of a lesson. Keep it {self.sentiment} and within {random.randint(5, 15)} words.",
                f"Give a short {self.sentiment} comment about a lesson, focusing on what could be improved. Max {random.randint(5, 15)} words.",
                f"Write a {self.sentiment} feedback on a lesson, mentioning its weaknesses. Limit your response to {random.randint(5, 15)} words.",
                f"Provide a short and critical comment on a lesson. Respond in no more than {random.randint(5, 15)} words.",
                f"Create a brief {self.sentiment} review of a lesson, highlighting flaws. Keep it under {random.randint(5, 15)} words.",
                f"Write a sentence of {self.sentiment} feedback about a lesson. Limit your answer to a maximum of {random.randint(5, 15)} words.",
                f"Generate a short critical comment about a lesson, focusing on its {self.sentiment} aspects. Max {random.randint(5, 15)} words.",
                f"Write a short but {self.sentiment} critique of a lesson. Keep it under {random.randint(5, 15)} words.",
                f"Create a {self.sentiment} and concise feedback on a lesson. Maximum of {random.randint(5, 15)} words.",

                f"Write a {self.sentiment} comment about a teaching method. Respond ONLY with the new sentence and max {random.randint(5, 15)} words.",
                f"Provide a critical review of an educational tool. Keep it {self.sentiment} and within {random.randint(5, 15)} words.",
                f"Give a brief {self.sentiment} comment on an educational video. Limit your response to {random.randint(5, 15)} words.",
                f"Create a {self.sentiment} review about a classroom activity. Respond in no more than {random.randint(5, 15)} words.",
                f"Write a {self.sentiment} feedback on a student project. Keep your response within {random.randint(5, 15)} words.",
                f"Create a short, critical comment on a teacher’s explanation. Max {random.randint(5, 15)} words.",
                f"Write a {self.sentiment} critique of an online learning platform. Respond in no more than {random.randint(5, 15)} words.",
                f"Provide a brief {self.sentiment} review of a homework assignment. Limit your response to {random.randint(5, 15)} words.",
                f"Give a short {self.sentiment} comment on a group discussion. Max {random.randint(5, 15)} words.",
                f"Write a critical feedback about an educational experience. Keep it concise and under {random.randint(5, 15)} words.",

                f"Create a {self.sentiment} comment about an e-learning course. Respond ONLY with the new sentence and max {random.randint(5, 15)} words.",
                f"Write a critical review of a book used for learning. Limit it to {random.randint(5, 15)} words.",
                f"Provide a {self.sentiment} feedback on a tutor’s performance. Respond concisely in {random.randint(5, 15)} words or less.",
                f"Generate a {self.sentiment} comment of a lesson plan. Respond within a maximum of {random.randint(5, 15)} words.",
                f"Write a short {self.sentiment} review of an educational app. Limit your answer to {random.randint(5, 15)} words.",
                f"Give a {self.sentiment} comment on an educational game or activity. Keep it to a maximum of {random.randint(5, 15)} words.",
                f"Write a short {self.sentiment} reflection on a class discussion. Max {random.randint(5, 15)} words.",
                f"Create a {self.sentiment} comment of a training session. Respond within {random.randint(5, 15)} words.",
                f"Write a {self.sentiment} comment about a collaborative project. Keep it under {random.randint(5, 15)} words.",
                f"Give a brief {self.sentiment} feedback on an educational seminar. Limit your response to {random.randint(5, 15)} words."
            ]
            adjusted_sentence = self.generate_text_with_ollama(random.choice(prompts))

            # Save the generated sentence to the DataFrame
            df_output = pd.concat(
                [df_output, pd.DataFrame({"text": [adjusted_sentence], "sentiment": [self.sentiment]})],
                ignore_index=True,
            )

            # Save the updated DataFrame to the output CSV file
            df_output.to_csv(self.output_csv, index=False)

    def process(self):
        if self.mode == "adjust":
            self.process_adjust()
        elif self.mode == "generate":
            self.generate_infinite()
        else:
            print(f"Invalid mode: {self.mode}")
