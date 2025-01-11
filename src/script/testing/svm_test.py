import pickle
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class SVMPredictor:
    """
    Handles prediction and evaluation of an SVM model for sentiment analysis.
    It uses the trained model to classify new text or evaluate model performance.
    """

    def __init__(self, trained_model=None, embedding_class=None, embedding_type="tfidf"):
        """
        Initializes the predictor with a trained SVM model, embedding class, and embedding type.

        :param trained_model: The trained SVM model.
        :param embedding_class: The class used for text vectorization.
        :param embedding_type: The type of embedding used ('tfidf', 'word2vec', 'bert').
        """
        self.model = trained_model
        self.embedding_class = embedding_class
        self.embedding_type = embedding_type

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the model on testing data and returns evaluation metrics.

        :param X_test: Test feature matrix.
        :param y_test: True sentiment labels for the testing data.
        :return: A dictionary with evaluation metrics (precision, recall, F1 score, accuracy).
        """
        y_pred = self.model.predict(X_test)

        evaluation_metrics = {
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
            "accuracy": accuracy_score(y_test, y_pred),
        }

        return evaluation_metrics

    def use_model(self, text):
        """
        Classifies a new text using the trained SVM model.

        :param text: The text to classify.
        :return: The predicted sentiment class.
        """
        # Transform the new text into the appropriate embedding
        if self.embedding_type == "tfidf":
            embeddings = self.embedding_class.tfidf_model.transform([text])
            X = embeddings.toarray()

        elif self.embedding_type == "word2vec":
            embeddings = [
                self.embedding_class.w2v_model.wv[word]
                for word in text.split() if word in self.embedding_class.w2v_model.wv
            ]
            X = np.mean(embeddings, axis=0).reshape(1, -1)

        elif self.embedding_type == "bert":
            embeddings = self.embedding_class.bert_model.encode([text])
            X = np.mean(embeddings, axis=0).reshape(1, -1)

        else:
            raise ValueError(f"Embedding type '{self.embedding_type}' is not supported.")

        # Predict the sentiment class
        return self.model.predict(X)[0]


    def load_model(self, filepath):
        """
        Carica un modello, il vettorizzatore e il trasformatore TF-IDF da disco.
        :param filepath: Percorso del file da cui caricare il modello.
        :return: Il modello caricato e la classe di embedding.
        """
        try:
            # Carica il modello e la classe di embedding dal file
            with open(filepath, "rb") as f:
                saved_objects = pickle.load(f)
                self.model = saved_objects["model"]
                self.embedding_class = saved_objects["embedding_class"]

            print(f"Modello e embedding caricati correttamente da {filepath}!")

        except Exception as e:
            print(f"Errore durante il caricamento del modello: {e}")