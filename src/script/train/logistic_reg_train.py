import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append("src/data/data_processing/")
from embedding import Embeddings # type: ignore

class LogisticRegressionTrainer:
    """
    Gestisce l'addestramento di un modello di Regressione Logistica per l'analisi del sentiment.
    Accetta un dataset e un tipo di embedding per la vettorizzazione del testo.
    """

    def __init__(self, dataset, test_size=0.2, embedding_type="tfidf"):
        """
        Inizializza la classe con un dataset, la dimensione del testing set e il tipo di embedding.

        :param dataset: Lista di tuple (text, sentiment) per il training.
        :param test_size: Proporzione dei dati da utilizzare per il testing (default 20%).
        :param embedding_type: Metodo di embedding da utilizzare ('tfidf', 'word2vec', 'bert').
        """
        self.model = LogisticRegression(
            solver="lbfgs", max_iter=1000, random_state=42
        )
        self.dataset = dataset
        self.test_size = test_size
        self.embedding_class = None
        self.embedding_type = embedding_type

    def train_model(self):
        """
        Addestra il modello di regressione logistica sul dataset fornito.
        """
        try:
            # Estrazione dei testi e delle etichette dal dataset
            texts, sentiments = zip(*self.dataset)

            # Creazione degli embedding
            self.embedding_class = Embeddings(self.dataset)

            if self.embedding_type == "tfidf":
                embedding = self.embedding_class.apply_tfidf_embedding()
                X = embedding.toarray()

            elif self.embedding_type == "word2vec":
                scaler = MinMaxScaler()
                embedding = self.embedding_class.apply_word2vec_embedding()
                X = scaler.fit_transform(embedding)

            elif self.embedding_type == "bert":
                scaler = MinMaxScaler()
                embedding = self.embedding_class.apply_bert_embedding()
                X = scaler.fit_transform(embedding)

            else:
                raise ValueError(
                    f"Tipo di embedding '{self.embedding_type}' non supportato. Usa 'tfidf', 'word2vec', o 'bert'."
                )

            # Divisione dei dati in training e testing
            X_train, X_test, y_train, y_test = train_test_split(
                X, sentiments, test_size=self.test_size, random_state=42
            )

            # Addestramento del modello
            self.model.fit(X_train, y_train)
            self.X_test = X_test
            self.y_test = y_test

            print("Modello di regressione logistica allenato correttamente!")

        except Exception as e:
            print(f"Errore durante l'addestramento: {e}")

    def save_model(self, filepath):
        """
        Salva il modello allenato e la classe di embedding su disco.

        :param filepath: Percorso del file in cui salvare il modello.
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
            print(f"Modello e embedding salvati correttamente in {filepath}!")
        except Exception as e:
            print(f"Errore durante il salvataggio del modello: {e}")