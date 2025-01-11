import pickle
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class LogisticRegressionPredictor:
    """
    Gestisce l'utilizzo e la valutazione di un modello di Regressione Logistica per l'analisi del sentiment.
    """

    def __init__(self, trained_model=None, embedding_class=None, embedding_type="tfidf"):
        """
        Inizializza la classe con un modello addestrato e il metodo di embedding utilizzato.

        :param trained_model: Il modello di regressione logistica addestrato.
        :param embedding_class: La classe usata per la vettorizzazione del testo.
        :param embedding_type: Il tipo di embedding utilizzato ('tfidf', 'word2vec', 'bert').
        """
        self.model = trained_model
        self.embedding_class = embedding_class
        self.embedding_type = embedding_type

    def evaluate_model(self, X_test, y_test):
        """
        Valuta il modello utilizzando il set di testing.

        :param X_test: Feature matrix del testing set.
        :param y_test: Etichette reali del testing set.
        :return: Un dizionario contenente precision, recall, f1-score e accuracy.
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
        Predice la classe di un nuovo testo utilizzando il modello allenato.

        :param text: Il testo da classificare.
        :return: La classe predetta.
        """
        # Creazione dell'embedding per il nuovo testo
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
            raise ValueError(f"Tipo di embedding '{self.embedding_type}' non supportato.")

        # Predizione
        return self.model.predict(X)[0]

    def load_model(self, filepath):
        """
        Carica un modello e la classe di embedding da un file salvato.

        :param filepath: Percorso del file salvato.
        """
        try:
            with open(filepath, "rb") as f:
                saved_objects = pickle.load(f)
                self.model = saved_objects["model"]
                self.embedding_class = saved_objects["embedding_class"]

            print(f"Modello e embedding caricati correttamente da {filepath}!")
        except Exception as e:
            print(f"Errore durante il caricamento del modello: {e}")