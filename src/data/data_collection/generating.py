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
                "prompt": (
                    f"""
                    
                    Transform the given sentence into its negative version. Here are examples:
                    
                    Example 1:
                    Input: I love sunny days.
                    Output: I don’t love sunny days.
                    
                    Example 2:
                    Input: She is happy with her results.
                    Output: She is not happy with her results.
                    
                    Example 3:
                    Input: We can achieve this goal.
                    Output: We cannot achieve this goal. 
                    
                    Now, transform the following sentence into its negative form: {text}. The response must be ONLY the new sentence without any comment.
                    
                    """
                ),
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
                f"Write a {self.sentiment} comment on a lesson, focusing on key aspects to consider.",
                f"Provide a {self.sentiment} review of a lesson, offering suggestions for improvement.",
                f"Write a short {self.sentiment} reflection on a group project, highlighting strengths and weaknesses.",
                f"Generate a {self.sentiment} evaluation of a teaching method, considering its potential impact.",
                f"Offer a {self.sentiment} critique of a classroom activity, exploring possible enhancements.",
                f"Write a brief {self.sentiment} assessment of an educational tool, focusing on its advantages and limitations.",
                f"Provide a {self.sentiment} comment on how a teacher presents complex topics.",
                f"Write a {self.sentiment} review of an online learning platform, evaluating its overall effectiveness.",
                f"Give a {self.sentiment} opinion on a student’s work during a presentation.",
                f"Write {self.sentiment} feedback on an assignment, addressing areas for growth.",

                f"Create a {self.sentiment} evaluation of a student project, suggesting areas for development.",
                f"Write a {self.sentiment} critique of a lesson, examining aspects that could be improved.",
                f"Offer a {self.sentiment} review of a lesson’s content, proposing ways to increase engagement.",
                f"Write {self.sentiment} feedback on a teaching method, pointing out what works well and what doesn’t.",
                f"Provide a {self.sentiment} critique of a teaching approach, highlighting areas that could be refined.",
                f"Write a {self.sentiment} comment about a group discussion, focusing on participation and interaction.",
                f"Generate a {self.sentiment} reflection on a teacher’s classroom management techniques.",
                f"Offer a {self.sentiment} evaluation of a group project, considering both successes and challenges.",
                f"Write a {self.sentiment} review of a classroom tool, discussing how it could be optimized.",
                f"Provide a {self.sentiment} comment on a class debate, evaluating the effectiveness of the discussion.",

                f"Write a {self.sentiment} analysis of a lesson, incorporating a broad vocabulary to reflect its various aspects.",
                f"Provide a {self.sentiment} assessment of a lesson, utilizing diverse terminology to discuss its strengths and areas for improvement.",
                f"Offer a {self.sentiment} critique of a group project, using a wide range of descriptive words to highlight both successes and challenges.",
                f"Write a {self.sentiment} review of a teaching method, ensuring the use of precise and varied language to describe its impact.",
                f"Generate a {self.sentiment} evaluation of a classroom activity, employing an extensive lexicon to explore its effectiveness.",
                f"Write a {self.sentiment} reflection on an educational tool, using rich and varied language to discuss its advantages and limitations.",
                f"Provide a {self.sentiment} critique of a teacher’s explanation, incorporating diverse vocabulary to express both clarity and areas for improvement.",
                f"Write a {self.sentiment} review of an online learning platform, choosing varied and detailed language to assess its strengths.",
                f"Give a {self.sentiment} comment on a student’s presentation, utilizing a rich vocabulary to evaluate both delivery and content.",

                f"Write {self.sentiment} feedback on an assignment, demonstrating a broad vocabulary to suggest areas for improvement.",
                f"Create a {self.sentiment} evaluation of a student project, employing diverse terminology to describe both strong points and areas for growth.",
                f"Write a {self.sentiment} critique of a lesson, integrating varied and elevated language to explore how the lesson can be enhanced.",
                f"Offer a {self.sentiment} review of a lesson’s structure, utilizing sophisticated language to suggest improvements in pacing and flow.",
                f"Write {self.sentiment} feedback on a teaching technique, using varied and precise vocabulary to highlight both its strengths and weaknesses.",
                f"Provide a {self.sentiment} analysis of a teaching method, ensuring the use of a broad and advanced lexicon to describe its impact.",
                f"Write a {self.sentiment} comment on a group discussion, using an array of expressive language to analyze participation levels.",
                f"Generate a {self.sentiment} reflection on a teacher’s classroom management, employing diverse and nuanced vocabulary to evaluate effectiveness.",
                f"Offer a {self.sentiment} critique of a group project, using varied and refined language to evaluate the project’s overall success.",
                f"Write a {self.sentiment} review of a classroom tool, selecting precise and varied language to evaluate its practicality.",
                f"Provide a {self.sentiment} comment on a class debate, incorporating rich and varied vocabulary to assess the quality of the discussion."
            ]

            prompt = f"""
                    Here are some example reviews of lessons. Please note the format is a single, short sentence:

                    Example for Neutral: The lesson was okay, but I would have liked more interaction.
                    Example for Positive: The lesson was clear and concise, I understood everything well.
                    Example for Negative: I had a hard time understanding the main points of the lesson.

                    Now, please generate a review based on the following instructions:
                    
                    {random.choice(prompts)}
                    
                    Respond ONLY with the new sentence and MAX {random.randint(5, 20)} words. The output must be a single sentence and should not include any extra information like explanations.
                    """

            adjusted_sentence = self.generate_text_with_ollama(prompt)

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
