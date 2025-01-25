import pickle, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelBinarizer

class NaiveBayesPredictor:
    """
    This class handles predictions and model evaluation after the Naive Bayes classifier is trained.
    It uses the saved model for text classification and evaluates the model performance.
    """

    def __init__(self, trained_model=None, embedding_class=None, embedding_type="tfidf"):
        """
        Initializes the predictor with the trained model, embedding class, and embedding type.

        :param trained_model: The trained Naive Bayes model.
        :param embedding_class: The class used for text vectorization.
        :param embedding_type: Embedding type used ('tfidf', 'word2vec', 'bert').
        """
        self.model = trained_model
        self.embedding_class = embedding_class
        self.embedding_type = embedding_type

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the trained Naive Bayes model on the testing data and returns evaluation metrics.

        :return: A dictionary containing evaluation metrics (precision, recall, F1 score, accuracy, ROC AUC).
        """

        # Proceed with calculating the evaluation metrics
        y_pred = self.model.predict(X_test)

        evaluation_metrics = {
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
            "accuracy": accuracy_score(y_test, y_pred),
        }

        # Convert y_test to a binary format for multi-class ROC AUC calculation
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)

        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(X_test)
            num_classes = y_prob.shape[1]
            roc_auc_scores = {}

            # Compute ROC AUC for each class
            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                roc_auc_scores[f"class_{i}"] = auc(fpr, tpr)

            evaluation_metrics["roc_auc"] = roc_auc_scores

        return evaluation_metrics

    def use_model(self, text):
        """
        Classifies a new text sample using the trained Naive Bayes model.

        :param text: The text to classify.
        :return: The predicted sentiment class.
        """
        if not hasattr(self, "model"):
            raise ValueError("The model has not been trained or loaded.")

        # Transform the new text into the appropriate embedding format
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

    """
    Visualization functions for the trained model
    """
    def plot_roc_curve(self, X_test, y_test):
        """
        Plots the ROC curve for the model on the test data.

        :param X_test: Test features.
        :param y_test: True labels for the test set.
        """
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("The model does not support probability predictions required for ROC.")

        # Get probability predictions
        y_prob = self.model.predict_proba(X_test)

        # Convert y_test to a binary format
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)

        # Compute ROC curve and AUC for each class
        num_classes = y_prob.shape[1]
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--", label="Random guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.show()

    def plot_confusion_matrix(self, X_test, y_test):
        """
        Plots the confusion matrix for the model on the test data.

        :param X_test: Test features.
        :param y_test: True labels for the test set.
        """
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))

        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix")
        plt.show()

    def plot_learning_curve(self, X_train, y_train, cv=5):
        """
        Plots the learning curve for the model on the training data.

        :param X_train: Training features.
        :param y_train: True labels for the training set.
        :param cv: Number of cross-validation folds.
        """
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")

        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.title("Learning Curve")
        plt.legend(loc="best")
        plt.grid()
        plt.show()