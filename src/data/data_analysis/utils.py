import random
import pandas as pd
from collections import Counter

from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer


class Utils:

    @staticmethod
    def save_results_to_csv(results, csv_file):
        """
        Salva una lista di risultati in formato CSV.

        Args:
            results (list): Lista di tuple del tipo [(text, sentiment)].
            csv_file (str): Nome del file CSV di output.
        """
        try:
            df = pd.DataFrame(results, columns=["text", "sentiment"])
            df.to_csv(csv_file, index=False)
            print(f"Risultati salvati in {csv_file}.")
            return df
        except Exception as e:
            print(f"Errore nel salvataggio del file CSV {csv_file}: {e}")

    @staticmethod
    def split_data(data, validation_split=0.2):
        """
        Divide i dati in set di training e validation.

        Args:
            data (list): Lista di tuple (es. [(text, sentiment), ...]).
            validation_split (float): Percentuale di dati per il validation set.

        Returns:
            tuple: training_set, validation_set.
        """
        try:
            index = int((1 - validation_split) * len(data))
            random.shuffle(data)
            return data[:index], data[index:]
        except Exception as e:
            print(f"Errore nella divisione dei dati: {e}")
            return [], []

    @staticmethod
    def undersample_class(data, class_to_undersample, target_column, max_entries=None):
        """
        Esegue undersampling sulla classe specificata per bilanciare il dataset.

        Args:
            data (pd.DataFrame): Il dataset originale.
            class_to_undersample (str): La classe da ridurre (es. 'positive').
            target_column (str): La colonna che contiene le etichette di classe.
            max_entries (int, optional): Numero massimo di entry per la classe sottocampionata.

        Returns:
            pd.DataFrame: Il dataset bilanciato con undersampling.
        """
        try:
            # Conta le occorrenze di ciascuna classe
            class_counts = Counter(data[target_column])
            print(f"Distribuzione originale delle classi: {class_counts}")

            # Determina la dimensione massima delle altre classi
            max_class_size = max(
                count
                for cls, count in class_counts.items()
                if cls != class_to_undersample
            )

            # Se max_entries è specificato, usa max_class_size
            if max_entries is not None:
                max_class_size = max_entries

            # Suddividi il dataset per classe
            class_groups = data.groupby(target_column)

            # Mantieni tutte le istanze delle altre classi e sottocampiona la classe specificata
            balanced_data = pd.concat(
                [
                    (
                        group
                        if name != class_to_undersample
                        else group.sample(max_class_size, random_state=42)
                    )
                    for name, group in class_groups
                ]
            )
            balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(
                drop=True
            )

            # Conta le occorrenze dopo l'undersampling
            new_class_counts = Counter(balanced_data[target_column])
            print(f"Distribuzione dopo undersampling: {new_class_counts}")

            return balanced_data

        except Exception as e:
            print(f"Errore durante l'undersampling: {e}")
            return data

    @staticmethod
    def oversample_with_smote(df, sentiment, text_col='text', sentiment_col='sentiment'):
        """
        Applica SMOTE per fare oversampling delle classi 'negative' e 'neutral' in un dataframe di sentiment analysis.

        :param df: pandas DataFrame contenente i dati
        :param text_col: Nome della colonna contenente il testo
        :param sentiment_col: Nome della colonna contenente il sentiment (target)
        :param random_state: Random state per la riproducibilità
        :return: DataFrame bilanciato
        """
        # Sostituisci valori mancanti nella colonna testo con una stringa vuota
        df[text_col] = df[text_col].fillna("")

        # Estrai testo e sentiment
        X = df[text_col]
        y = df[sentiment_col]

        # Converti il testo in rappresentazioni numeriche usando TF-IDF
        tfidf = TfidfVectorizer()
        X_tfidf = tfidf.fit_transform(X)

        # Determina il numero massimo di campioni (classe di maggioranza)
        max_class_count = y.value_counts().max()

        # Configura la strategia per portare ogni classe al massimo livello
        sampling_strategy = {f"{sentiment}": max_class_count}

        # Applica SMOTE
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

        # Converti i dati oversamplati in un nuovo dataframe
        resampled_text = tfidf.inverse_transform(X_resampled)
        resampled_df = pd.DataFrame({
            text_col: [' '.join(tokens) for tokens in resampled_text],
            sentiment_col: y_resampled
        })

        return resampled_df

    @staticmethod
    def map_sentiment_labels(dataset, column_name):
        """
        Sostituisce i valori di una colonna specifica in un dataset con una mappatura predefinita.

        Parameters:
            dataset (pd.DataFrame): Il dataset da processare.
            column_name (str): Il nome della colonna contenente le etichette da mappare.

        Returns:
            pd.DataFrame: Il dataset con la colonna aggiornata.
        """
        # Definisci la mappatura
        label_mapping = {
            "LABEL_0": "negative",  # LABEL_0 -> negative
            "LABEL_1": "neutral",  # LABEL_1 -> neutral
            "LABEL_2": "positive"  # LABEL_2 -> positive
        }

        # Applica la mappatura solo se la colonna esiste
        if column_name in dataset.columns:
            dataset[column_name] = dataset[column_name].map(label_mapping).fillna(dataset[column_name])
        else:
            raise KeyError(f"La colonna '{column_name}' non è presente nel dataset.")

        return dataset

    @staticmethod
    def remove_void_or_null(dataset_path):
        """
        Rimuove tutte le righe dal file CSV in cui uno dei valori è vuoto o uguale a null
        e salva il file modificato in loco.

        Args:
            dataset_path (str): Il percorso al file CSV contenente il dataset (text, sentiment).
        """
        # Carica il dataset con pandas
        df = pd.read_csv(dataset_path, header=None, names=["text", "sentiment"])

        # Filtra le righe in cui 'text' o 'sentiment' sono vuoti o uguali a 'null' (case insensitive)
        filtered_df = df[
            df["text"].notnull() &
            df["sentiment"].notnull() &
            (df["text"].str.strip().str.lower() != "null") &
            (df["sentiment"].str.strip().str.lower() != "null") &
            (df["text"].str.strip().str.lower() != "") &
            (df["sentiment"].str.strip().str.lower() != "")
            ]

        # Sovrascrive il file originale con i dati filtrati
        filtered_df.to_csv(dataset_path, index=False, header=False)


# ut = Utils()
# dt = pd.read_csv("../../dataset/gold/es2/defmerge_undersample.csv")
# newdt = ut.map_sentiment_labels(dt, "sentiment")
# newdt.to_csv("../../dataset/silver/rtt.csv", index=False)

# dt = pd.read_csv("../../dataset/gold/es2/defmerge_undersample.csv")
# dt2 = ut.undersample_class(dt, "positive", "sentiment", 35000)
# dt2.to_csv("../../dataset/gold/defmerge_undersample2.csv", index=False)

# ut.remove_void_or_null("../../dataset/silver/tmp3.csv")

# tt1 = ut.oversample_with_smote(dt, "negative")
# tt1.to_csv("../../dataset/gold/def_merged_test.csv", index=False)

# tt1 = ut.oversample_with_smote(tt1, "neutral")
# tt1.to_csv("../../dataset/gold/def_merged_test.csv", index=False)