import os
from src.data.data_processing.processing import Preprocessor
from src.script.testing.logistic_reg_test import LogisticRegressionPredictor
from src.script.testing.naive_bayes_test import NaiveBayesPredictor
from src.script.train.logistic_reg_train import LogisticRegressionTrainer

class UseScript:

    def __init__(self, path):
        self.predictor = None
        self.model_path = path
        self.check_and_load_model()

    def check_and_load_model(self):
        """
        Controlla se il file del modello esiste. Se sì, lo carica.
        Altrimenti, costruisce e salva il modello, quindi lo carica.
        """
        if os.path.exists(self.model_path):
            print(f"Il modello esiste: {self.model_path}. Caricamento in corso...")
            self.load_model(self.model_path)
        else:
            print(f"Il modello non esiste: {self.model_path}. Addestramento in corso...")
            self.build_model()
            self.load_model(self.model_path)

    def load_model(self, path):
        """
        Carica il modello dal percorso specificato.
        """
        self.predictor = NaiveBayesPredictor(embedding_type="tfidf")
        self.predictor.load_model(path)

    def use_model(self, text):
        """
        Utilizza il modello per effettuare una previsione.
        """
        if not self.predictor:
            raise ValueError("Il modello non è stato caricato.")
        prediction = self.predictor.use_model(text)
        return prediction

    def build_model(self):
        """
        Addestra e salva il modello.
        """
        print("Addestramento del modello...")
        d_processing = Preprocessor()
        d_processing.load_dataset("./src/dataset/gold/merged_undersample_processed.csv")

        trainer = LogisticRegressionTrainer(d_processing.data_list, embedding_type="tfidf")
        trainer.train_model()
        trainer.save_model(self.model_path)
        print(f"Modello salvato in: {self.model_path}")
