import pickle
import numpy as np
from keras.src.saving import load_model
from tf_keras.src.utils import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class LSTMPredictor:
    def __init__(self, model_path):
        """
        Inizializza la classe per il testing del modello.
        :param model_path: Percorso del modello salvato
        :param max_len: Lunghezza massima delle sequenze di input
        """

        try:
            self.model = load_model(model_path)
            with open(f"{model_path}_tokenizer.pkl", "rb") as f:
                self.tokenizer = pickle.load(f)
            print(f"Modello e tokenizer caricati correttamente da {model_path}!")
        except Exception as e:
            print(f"Errore durante il caricamento del modello: {e}")

    def evaluate_model(self, X_test, y_test):
        y_pred_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)

        evaluation_metrics = {
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1_score": f1_score(y_true, y_pred, average="weighted"),
            "accuracy": accuracy_score(y_true, y_pred),
        }

        return evaluation_metrics

    def use_model(self, text):
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, padding="post")
        prediction = self.model.predict(padded_sequence)
        return np.argmax(prediction)