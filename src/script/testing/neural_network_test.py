import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import LabelBinarizer

class NeuralNetworkPredictor:
    """
    Questa classe gestisce la valutazione e l'utilizzo dei modelli addestrati dalla classe SentimentAnalysisModel.
    Supporta modelli feedforward, LSTM e BERT.
    """

    def __init__(self, trained_model=None, embedding_class=None, embedding_type="tfidf"):
        """
        Inizializza il predictor con il modello addestrato, la classe di embedding e il tipo di embedding.

        :param trained_model: Il modello addestrato (feedforward, LSTM o BERT).
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
            embeddings = self.embedding_class.apply_tfidf_embedding(texts=[text])
            X = embeddings.toarray()

        elif self.embedding_type == "word2vec":
            embeddings = self.embedding_class.apply_word2vec_embedding(texts=[text])
            X = np.mean(embeddings, axis=0).reshape(1, -1)

        elif self.embedding_type == "bert":
            embeddings = self.embedding_class.apply_bert_embedding(texts=[text])
            X = np.mean(embeddings, axis=0).reshape(1, -1)

        else:
            raise ValueError(f"Tipo di embedding '{self.embedding_type}' non supportato.")

        # Predice la classe di sentiment
        return self.model.predict(X)[0]

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

    def plot_learning_curve(self, X_train, y_train, X_test, y_test):
        """
        Traccia la curva di apprendimento per il modello sui dati di training.

        :param X_train: Dati di training (embedding del testo).
        :param y_train: Etichette di training.
        :param X_test: Dati di test (embedding del testo).
        :param y_test: Etichette di test.
        :param cv: Numero di fold per la cross-validation (non utilizzato in questa implementazione).
        """
        # Definisci le porzioni del dataset di training da utilizzare
        train_sizes = np.linspace(0.1, 1.0, 10)  # Usa 10 punti tra il 10% e il 100% del dataset
        train_scores = []  # Accuratezza sul training set
        test_scores = []   # Accuratezza sul validation set

        for size in train_sizes:
            # Seleziona una porzione del dataset di training
            n_samples = int(size * X_train.shape[0])
            X_subset, _, y_subset, _ = train_test_split(X_train, y_train, train_size=n_samples, random_state=42)

            # Addestra il modello sulla porzione selezionata
            self.model.fit(X_subset, y_subset, epochs=5, batch_size=32, verbose=1)

            # Valuta il modello sul training set e sul validation set
            train_score = self.model.evaluate(X_subset, y_subset, verbose=0)[1]  # Accuratezza
            test_score = self.model.evaluate(X_test, y_test, verbose=0)[1]       # Accuratezza

            train_scores.append(train_score)
            test_scores.append(test_score)

        # Traccia la curva di apprendimento
        plt.plot(train_sizes, train_scores, "o-", color="r", label="Training score")
        plt.plot(train_sizes, test_scores, "o-", color="g", label="Validation score")
        plt.xlabel("Porzione del Training Set")
        plt.ylabel("Accuratezza")
        plt.title("Curva di Apprendimento")
        plt.legend(loc="best")
        plt.grid()
        plt.show()