import re as regex
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("stopwords")

class Preprocessor:

    def __init__(self):
        """
        data_list è una lista di tuple del tipo (text, sentiment)
        """
        self.dataset = None
        self.data_list = None
        
    def load_dataset(self, file_path: str):
        """
        Carica un dataset in formato csv usando pandas.
        """
        try:
            self.dataset = pd.read_csv(file_path)
            
            self.data_list = list(
                zip(
                    self.dataset["text"].tolist(),
                    self.dataset["sentiment"].tolist(),
                )
            )
                    
        except Exception as e:
            print(f"Errore durante il caricamento del dataset: {e}")

    def lowercase(self):
        """
        Trasforma il testo in minuscolo.
        """
        self.data_list = [
            (text.lower(), sentiment) for text, sentiment in self.data_list
        ]
        
    def remove_digits(self):
        """
        Elimina digit dal testo.
        """
        self.data_list = [
            (regex.sub(r'\d+', '', text), sentiment) for text, sentiment in self.data_list
        ]
     
    def remove_punctuation(self):
        """
        Rimuove la punteggiatura dal testo.
        """
        self.data_list = [
            (regex.sub(r"[^\w\s]", "", text), sentiment)
            for text, sentiment in self.data_list
        ]

    def remove_non_english_chars(self):
        """
        Rimuove tutti i caratteri che non appartengono all'alfabeto inglese o ai simboli di punteggiatura.
        """
        self.data_list = [
            (regex.sub(r'[^a-zA-Z0-9\s.,!?\'"]+', '', text), sentiment)
            for text, sentiment in self.data_list
        ]

    def remove_emojis_and_symbols(self):
        """
        Rimuove tutte le emoji e i simboli particolari dai testi.
        """
        emoji_pattern = regex.compile(
            "["
            "\U0001F600-\U0001F64F"  # Emoticon
            "\U0001F300-\U0001F5FF"  # Simboli e pittogrammi
            "\U0001F680-\U0001F6FF"  # Trasporto e simboli
            "\U0001F1E0-\U0001F1FF"  # Bandiere
            "\U00002702-\U000027B0"  # Simboli aggiuntivi
            "\U000024C2-\U0001F251"  # Altri simboli
            "]+",
            flags=regex.UNICODE
        )

        self.data_list = [
            (emoji_pattern.sub('', text), sentiment)
            for text, sentiment in self.data_list
        ]

    def remove_stopwords(self):
        """
        Rimuove le stopword.
        """
        stop_words = set(stopwords.words("english"))
        self.data_list = [
            (
                " ".join(
                    [word for word in word_tokenize(text) if word not in stop_words]
                ),
                sentiment,
            )
            for text, sentiment in self.data_list
        ]

    def lemmatize(self):
        """
        Applica la lemmatizzazione al testo.
        """
        lemmatizer = WordNetLemmatizer()
        self.data_list = [
            (
                " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(text)]),
                sentiment,
            )
            for text, sentiment in self.data_list
        ]

    def remove_short_lines(self, min_words):
        """
        Rimuove le righe che hanno meno di `min_words` parole.
        """
        self.data_list = [
            (text, sentiment)
            for text, sentiment in self.data_list
            if len(word_tokenize(text)) >= min_words
        ]
    
    def remove_long_lines(self, max_words=50):
        """
        Rimuove le righe che hanno piu di `max_words` parole.
        """
        self.data_list = [
            (text, sentiment)
            for text, sentiment in self.data_list
            if len(word_tokenize(text)) <= max_words
        ]     
    
    def clean_text_spaces(self):
        """
        Rimuove spazi in eccesso dai testi in una lista di tuple (testo, sentiment).
        """
        cleaned_list = []

        for text, sentiment in self.data_list:
            # Rimuove spazi all'inizio e alla fine, e normalizza gli spazi tra le parole
            cleaned_text = " ".join(text.split())
            cleaned_list.append((cleaned_text, sentiment))
        
        self.data_list = cleaned_list    
    
    def preprocess_text(self, file_path: str):
        """
        Esegue tutti le operazioni di preprocessing in sequenza.
        """
        try:
            self.lowercase()
            self.remove_digits()
            self.remove_non_english_chars()
            self.remove_emojis_and_symbols()
            self.clean_text_spaces()
            self.remove_punctuation()
            self.remove_stopwords()
            self.lemmatize()
            self.remove_long_lines(50)
            self.remove_short_lines(1)

            if file_path:
                df = pd.DataFrame(self.data_list, columns=["text", "sentiment"])
                df.to_csv(file_path, index=False)
                print(f"Preprocessing completato, dataset salvato in {file_path}")
            else:
                print("Preprocessing completato, dataset non salvato")

            return self.data_list

        except Exception as e:
            print(f"Errore durante il preprocessing: {e}")