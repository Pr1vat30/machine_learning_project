import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import LabelBinarizer

class NeuralNetworkPredictor:
    """
    Questa classe gestisce la valutazione e l'utilizzo dei modelli addestrati.
    """

    def __init__(self, trained_model=None, embedding_class=None, embedding_type="tfidf"):
        """
        Inizializza il predictor con il modello addestrato, la classe di embedding e il tipo di embedding.

        :param trained_model: Il modello addestrato (feedforward).
        :param embedding_class: La classe utilizzata per la vettorizzazione del testo.
        :param embedding_type: Tipo di embedding utilizzato ('tfidf', 'word2vec', 'bert').
        """
        self.model = trained_model
        self.embedding_class = embedding_class
        self.embedding_type = embedding_type

    def evaluate_model(self, X_test, y_test):
        """
        Valuta il modello addestrato sui dati di test e restituisce le metriche di valutazione.

        :param X_test: Dati di test (embedding del testo).
        :param y_test: Etichette di test.
        :return: Un dizionario contenente le metriche di valutazione (precision, recall, F1 score, accuracy, ROC AUC).
        """
        y_pred = self.model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)

        evaluation_metrics = {
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
            "accuracy": accuracy_score(y_test, y_pred),
        }

        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)
        y_prob = self.model.predict(X_test)
        num_classes = y_prob.shape[1]
        roc_auc_scores = {}

        # Calcola ROC AUC per ogni classe
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc_scores[f"class_{i}"] = auc(fpr, tpr)

        evaluation_metrics["roc_auc"] = roc_auc_scores

        return evaluation_metrics

    def use_model(self, text):
        """
        Classifica un nuovo campione di testo utilizzando il modello addestrato.

        :param text: Il testo da classificare.
        :return: La classe di sentiment predetta.
        """
        if not hasattr(self, "model"):
            raise ValueError("Il modello non è stato addestrato o caricato.")

        # Trasforma il nuovo testo nel formato di embedding appropriato
        if self.embedding_type == "tfidf":
            embeddings = self.embedding_class.model.transform([text])
            X = embeddings.toarray()

        elif self.embedding_type == "word2vec":
            embeddings = [
                self.embedding_class.model.wv[word]
                for word in text.split() if word in self.embedding_class.model.wv
            ]
            X = np.mean(embeddings, axis=0).reshape(1, -1)

        elif self.embedding_type == "bert":
            embeddings = self.embedding_class.model.encode([text])
            X = np.mean(embeddings, axis=0).reshape(1, -1)

        else:
            raise ValueError(f"Tipo di embedding '{self.embedding_type}' non supportato.")

        prediction = self.model.predict(X)

        predicted_class = np.argmax(prediction, axis=1)

        mapping = {0: "negative", 1: "neutral", 2: "positive"}

        return mapping[predicted_class[0]] if len(predicted_class) == 1 else [mapping[c] for c in predicted_class]

    def load_model(self, filepath):
        """
        Carica un modello, la classe di embedding e lo scaler da disco.

        :param filepath: Percorso del file da cui caricare il modello.
        """
        try:
            with open(filepath, "rb") as f:
                saved_objects = pickle.load(f)
                self.model = saved_objects["model"]
                self.embedding_class = saved_objects["embedding_class"]
                if "scaler" in saved_objects:
                    self.scaler = saved_objects["scaler"]

            print(f"Modello e embedding caricati correttamente da {filepath}!")

        except Exception as e:
            print(f"Errore durante il caricamento del modello: {e}")

    """
    Funzioni di visualizzazione per il modello addestrato
    """

    def plot_roc_curve(self, X_test, y_test):
        """
        Traccia la curva ROC per il modello sui dati di test.

        :param X_test: Dati di test (embedding del testo).
        :param y_test: Etichette di test.
        """
        # Ottieni le previsioni di probabilità
        y_prob = self.model.predict(X_test)  # y_prob dovrebbe essere una matrice (n_samples, n_classes)

        # Converti y_test in formato binario
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)  # y_test_bin sarà una matrice (n_samples, n_classes)

        # Calcola la curva ROC e l'AUC per ogni classe
        num_classes = y_prob.shape[1]  # Numero di classi
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"Classe {i} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--", label="Guess casuale")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Curva ROC")
        plt.legend(loc="lower right")
        plt.show()

    def plot_confusion_matrix(self, X_test, y_test):
        """
        Traccia la matrice di confusione per il modello sui dati di test.

        :param X_test: Dati di test (embedding del testo).
        :param y_test: Etichette di test.
        """
        y_pred = self.model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))

        disp.plot(cmap="Blues", values_format="d")
        plt.title("Matrice di Confusione")
        plt.show()

    def plot_epoch_convergence(self, X_train, y_train, X_test, y_test, epochs=5, batch_size=32):
        """
        Traccia il grafico di convergenza durante le epoche per il modello.

        :param X_train: Dati di training (embedding del testo).
        :param y_train: Etichette di training.
        :param X_test: Dati di test (embedding del testo).
        :param y_test: Etichette di test.
        :param epochs: Numero di epoche per il training.
        :param batch_size: Dimensione del batch per il training.
        """
        # Addestra il modello e registra la cronologia
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        # Ottieni le metriche dal training e dal validation set
        train_accuracy = history.history["accuracy"]
        val_accuracy = history.history["val_accuracy"]
        train_loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        # Traccia i grafici della convergenza
        plt.figure(figsize=(14, 6))

        # Grafico dell'accuratezza
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), train_accuracy, "o-", label="Training Accuracy")
        plt.plot(range(1, epochs + 1), val_accuracy, "o-", label="Validation Accuracy")
        plt.xlabel("Epoche")
        plt.ylabel("Accuratezza")
        plt.title("Convergenza dell'accuratezza")
        plt.legend(loc="best")
        plt.grid()

        # Grafico della perdita
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), train_loss, "o-", label="Training Loss")
        plt.plot(range(1, epochs + 1), val_loss, "o-", label="Validation Loss")
        plt.xlabel("Epoche")
        plt.ylabel("Perdita")
        plt.title("Convergenza della perdita")
        plt.legend(loc="best")
        plt.grid()

        plt.tight_layout()
        plt.show()