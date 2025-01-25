import random
import pandas as pd
from tqdm import tqdm
from langdetect import detect
from collections import Counter
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer


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
    def oversample_with_smote(df, sentiment, text_col='text', sentiment_col='sentiment', model_name="bert-base-uncased"):
        """
        Applica SMOTE per fare oversampling delle classi in un dataframe di sentiment analysis,
        utilizzando BERT uncased (con SentenceTransformer) per rappresentare il testo.

        :param df: pandas DataFrame contenente i dati
        :param sentiment: classe di sentiment da bilanciare
        :param text_col: Nome della colonna contenente il testo
        :param sentiment_col: Nome della colonna contenente il sentiment (target)
        :param model_name: Nome del modello pre-addestrato BERT
        :return: DataFrame bilanciato
        """
        # Sostituisci valori mancanti nella colonna testo con una stringa vuota
        df[text_col] = df[text_col].fillna("")

        # Estrai testo e sentiment
        X = df[text_col]
        y = df[sentiment_col]

        # Carica il modello SentenceTransformer
        model = SentenceTransformer(model_name)

        # Aggiungi la barra di progresso con tqdm
        tqdm.pandas(desc="Generazione embeddings BERT")

        # Calcola gli embeddings BERT per tutto il dataset, con barra di progresso
        X_bert = X.progress_apply(lambda x: model.encode(x))

        # Determina il numero massimo di campioni (classe di maggioranza)
        max_class_count = y.value_counts().max()

        # Configura la strategia per portare ogni classe al massimo livello
        sampling_strategy = {f"{sentiment}": max_class_count}

        # Applica SMOTE sugli embeddings
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_bert.tolist(), y)

        # Converti i dati oversamplati in un nuovo dataframe
        resampled_df = pd.DataFrame({
            text_col: X_resampled,  # Gli embeddings come lista di vettori
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

    @staticmethod
    def rating_to_sentiment(input_file, output_file):
        # Carica il file CSV
        df = pd.read_csv(input_file)

        # Elimina le righe con valori nulli
        df.dropna(subset=['reviews', 'rating'], inplace=True)

        # Crea la colonna 'text' che contiene il testo della recensione
        df['text'] = df['reviews']

        # Crea la colonna 'sentiment' mappando il rating
        def map_sentiment(rating):
            if 1 <= rating < 2.5:
                return 'negative'
            elif 2.5 <= rating < 3.5:
                return 'neutral'
            elif 3.5 <= rating <= 5:
                return 'positive'
            return None  # In caso di rating fuori da 1-5 (se ci sono valori strani)

        df['sentiment'] = df['rating'].apply(map_sentiment)

        # Se non è presente la colonna 'sentiment', eliminare la riga
        df = df[df['sentiment'].notnull()]

        # Seleziona solo le colonne 'text' e 'sentiment' da esportare
        df_final = df[['text', 'sentiment']]

        # Salva il nuovo dataset in un file CSV
        df_final.to_csv(output_file, index=False)

    @staticmethod
    def remove_other_language(df, text_column):
        """
        Mantiene solo le righe di un DataFrame in cui il testo nella colonna specificata è in inglese.

        Args:
            df (pd.DataFrame): Il DataFrame da processare.
            colonna_testo (str): Il nome della colonna contenente il testo.

        Returns:
            pd.DataFrame: Il DataFrame contenente solo righe con testo in inglese.
        """

        def is_english(text):
            try:
                return detect(text) == 'en'
            except:
                return False  # Esclude righe problematiche

        # Filtra il DataFrame
        tqdm.pandas(desc="Filtrando righe non in inglese")
        df_filtrato = df[df[text_column].progress_apply(is_english)]
        return df_filtrato

    @staticmethod
    def process_short_comments(dataset):
        """
        Elimina il 60% dei commenti con meno di 3 parole per ogni classe di sentiment.

        Args:
            dataset (pd.DataFrame): Il dataset deve avere colonne 'text' e 'sentiment'.

        Returns:
            pd.DataFrame: Il dataset processato.
        """

        def count_words(text):
            """Conta il numero di parole in un testo."""
            return len(text.split())

        # Aggiungi una colonna per contare le parole in ogni commento
        dataset['word_count'] = dataset['text'].apply(count_words)

        # Filtra i commenti con meno di 3 parole
        short_comments = dataset[dataset['word_count'] <= 4]

        # Per ogni classe, rimuovi il 60% dei commenti sotto la soglia
        indices_to_remove = []
        for sentiment_class in short_comments['sentiment'].unique():
            class_subset = short_comments[short_comments['sentiment'] == sentiment_class]
            num_to_remove = int(len(class_subset) * 1)
            if num_to_remove > 0:
                indices_to_remove.extend(class_subset.sample(n=num_to_remove, random_state=42).index)

        # Rimuovi i commenti selezionati dal dataset
        dataset = dataset.drop(index=indices_to_remove)

        # Rimuovi la colonna temporanea 'word_count'
        dataset = dataset.drop(columns=['word_count'])

        return dataset