import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append("src/data/data_processing/")
from embedding import Embeddings # type: ignore

class NaiveBayesTrainer:
    """
    This class handles training a Naive Bayes classifier for sentiment analysis.
    It accepts a dataset and an embedding type for vectorizing text data (TF-IDF, Word2Vec, or BERT).
    """

    def __init__(self, dataset, test_size=0.2, embedding_type="tfidf"):
        """
        Initializes the trainer with a dataset, testing size, and embedding type.

        :param dataset: List of tuples (text, sentiment) for training.
        :param test_size: Proportion of data to be used for testing (default is 20%).
        :param embedding_type: Embedding method to use for text vectorization ('tfidf', 'word2vec', 'bert').
        """
        self.model = MultinomialNB()
        self.dataset = dataset
        self.test_size = test_size
        self.embedding_class = None
        self.embedding_type = embedding_type

    def split_data(self):
        """
        Splits the dataset into texts and sentiment labels.

        :return: Tuple of (texts, sentiments).
        """
        texts, sentiments = zip(*self.dataset)
        return texts, sentiments

    def train_model(self):
        """
        Trains the Naive Bayes model on the given dataset using the specified text embedding.
        """
        texts, sentiments = zip(*self.dataset)

        # Create the appropriate embedding class for text vectorization
        self.embedding_class = Embeddings(self.dataset)

        # Choose embedding type and transform data
        if self.embedding_type == "tfidf":
            embedding = self.embedding_class.apply_tfidf_embedding()
            X = embedding

        elif self.embedding_type == "word2vec":
            scaler = MinMaxScaler()
            embedding = self.embedding_class.apply_word2vec_embedding()
            X = scaler.fit_transform(embedding)

        elif self.embedding_type == "bert":
            scaler = MinMaxScaler()
            embedding = self.embedding_class.apply_bert_embedding()
            X = scaler.fit_transform(embedding)

        else:
            raise ValueError(f"Embedding type '{self.embedding_type}' is not supported.")

        # Split the data into training and testing sets
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, sentiments,
                test_size=self.test_size,
                random_state=42
            )
        except ValueError as e:
            print(f"Error during data split: {e}")
            return

        # Train the model
        try:
            self.model.fit(X_train, y_train)

            self.X_test, self.y_test = X_test, y_test
            self.X_train, self.y_train = X_train, y_train

            print("Model trained successfully!")
        except Exception as e:
            print(f"Error during model training: {e}")

    def save_model(self, filepath):
        """
        Saves the trained Naive Bayes model to a file.

        :param filepath: Path to save the model.
        """
        try:
            with open(filepath, "wb") as f:
                pickle.dump(
                    {
                        "model": self.model,
                        "embedding_class": self.embedding_class,
                    },
                    f
                )
                print(f"Model saved successfully to {filepath}!")

        except Exception as e:
            print(f"Error during model saving: {e}")
