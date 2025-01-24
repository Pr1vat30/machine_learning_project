import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append("src/data/data_processing/")
from embedding import Embeddings # type: ignore

class SVMTrainer:
    """
    Handles the training of an SVM classifier for sentiment analysis.
    It accepts a dataset and an embedding type for vectorizing text data (TF-IDF, Word2Vec, or BERT).
    """

    def __init__(self, dataset, test_size=0.2, embedding_type="tfidf"):
        """
        Initializes the trainer with a dataset, testing size, and embedding type.

        :param dataset: List of tuples (text, sentiment) for training.
        :param test_size: Proportion of data to be used for testing (default is 20%).
        :param embedding_type: Embedding method to use for text vectorization ('tfidf', 'word2vec', 'bert').
        """
        self.model = None
        self.dataset = dataset
        self.test_size = test_size
        self.embedding_class = None
        self.embedding_type = embedding_type

    def train_model(self):
        """
        Trains an SVM model on the given dataset using the specified embedding method.
        """
        try:
            # Extract texts and sentiments from the dataset
            texts, sentiments = zip(*self.dataset)

            # Initialize the embedding class and create embeddings
            self.embedding_class = Embeddings(self.dataset)

            if self.embedding_type == "tfidf":
                self.embedding_class.load_embedding("./tfidf_embedding.pkl")
                embedding = self.embedding_class.tfidf_embeddings
                X = embedding.toarray()  # Convert sparse matrix to dense

            elif self.embedding_type == "word2vec":
                scaler = MinMaxScaler()
                embedding = self.embedding_class.apply_word2vec_embedding()
                X = scaler.fit_transform(embedding)
                # X = embedding

            elif self.embedding_type == "bert":
                scaler = MinMaxScaler()
                embedding = self.embedding_class.apply_bert_embedding()
                X = scaler.fit_transform(embedding)
                # X = embedding

            else:
                raise ValueError(
                    f"Embedding type '{self.embedding_type}' is not supported. Use 'tfidf', 'word2vec', or 'bert'."
                )

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, sentiments, test_size=self.test_size, random_state=42
            )

            # Initialize and train the SVM model
            self.model = SVC(
                C=1.0,  # Regularization parameter
                random_state=42  # For reproducibility
            )
            self.model.fit(X_train, y_train)
            self.X_test = X_test
            self.y_test = y_test
            print("SVM model trained successfully!")

        except ValueError as ve:
            print(f"Data or configuration error: {ve}")

        except Exception as e:
            print(f"Error during model training: {e}")

    def save_model(self, filepath):
        """
        Saves the trained SVM model and embedding class to disk.

        :param filepath: Path to save the model.
        """
        try:
            with open(filepath, "wb") as f:
                pickle.dump(
                    {
                        "model": self.model,
                        "embedding_class": self.embedding_class,
                    },
                    f,
                )
            print(f"Model and embeddings saved successfully to {filepath}!")

        except Exception as e:
            print(f"Error saving the model: {e}")