import os, pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class Embeddings:
    """
    Classe per la generazione di embedding dai testi utilizzando TF-IDF, Word2Vec o BERT.
    """

    def __init__(self, data_list):
        """
        Inizializza la classe con una lista di tuple (text, sentiment).
        """
        self.data = data_list
        self.model = None
        self.embeddings = None

    def tokenize_texts(self):
        """
        Tokenizza i testi presenti nella lista di tuple (text, sentiment).
        """
        return [text.split() for text, _ in self.data]

    def save_embedding(self, filename, model, embeddings):
        """
        Salva i dati della classe, inclusi gli embedding, in un file pickle.
        """
        try:
            data_to_save = {
                "model": model,
                "embeddings": embeddings,
            }
            with open(filename, 'wb') as f:
                pickle.dump(data_to_save, f)
            print(f"Dati relativi alla classe ed embedding salvati in {filename}")
        except Exception as e:
            print(f"Errore durante il salvataggio: {e}")

    def load_embedding(self, filename):
        """
        Carica i dati essenziali della classe, inclusi gli embedding, da un file pickle.
        """
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            print(f"Dati relativi alla classe ed embedding caricati con successo da {filename}")
            return data.get("model"), data.get("embeddings")
        except Exception as e:
            print(f"Errore durante il caricamento: {e}")
            return None, None

    def apply_tfidf_embedding(self, filename="tfidf_embedding.pkl"):
        """
        Genera embeddings TF-IDF dai testi, controlla se esistono e li salva se necessario.
        """
        if os.path.exists(filename):
            print(f"TF-IDF embedding trovato in {filename}, caricamento in corso...")
            self.model, self.embeddings = self.load_embedding(filename)
        else:
            try:
                texts = [text for text, _ in self.data]
                vectorizer = TfidfVectorizer()
                self.embeddings = vectorizer.fit_transform(texts)
                self.model = vectorizer
                self.save_embedding(filename, self.model, self.embeddings)
                print(f"TF-IDF embedding completato. Shape: {self.embeddings.shape}")
            except Exception as e:
                print(f"Errore durante il calcolo del TF-IDF: {e}")
        return self.embeddings

    def apply_word2vec_embedding(self, filename="word2vec_embedding.pkl", vector_size=300, window=25, min_count=1):
        """
        Genera embeddings Word2Vec dai testi tokenizzati, controlla se esistono e li salva se necessario.
        """
        if os.path.exists(filename):
            print(f"Word2Vec embedding trovato in {filename}, caricamento in corso...")
            self.model, self.embeddings = self.load_embedding(filename)
        else:
            try:
                tokenized_texts = self.tokenize_texts()
                model = Word2Vec(
                    sentences=tokenized_texts,
                    vector_size=vector_size,
                    window=window,
                    min_count=min_count,
                    workers=4,
                )
                self.embeddings = np.array(
                    [
                        np.mean(
                            [model.wv[word] for word in words if word in model.wv]
                            or [np.zeros(vector_size)],
                            axis=0,
                        )
                        for words in tokenized_texts
                    ]
                )
                self.model = model
                self.save_embedding(filename, self.model, self.embeddings)
                print(f"Word2Vec embedding completato. Shape: {self.embeddings.shape}")
            except Exception as e:
                print(f"Errore durante il calcolo del Word2Vec: {e}")
        return self.embeddings

    def apply_bert_embedding(self, filename="bert_embedding.pkl", model_name="bert-base-uncased"):
        """
        Genera embeddings BERT, controlla se esistono e li salva se necessario, con barra di progresso.
        """
        if os.path.exists(filename):
            print(f"BERT embedding trovato in {filename}, caricamento in corso...")
            self.model, self.embeddings = self.load_embedding(filename)
        else:
            try:
                texts = [text for text, _ in self.data]
                model = SentenceTransformer(model_name)

                # Inizializza la barra di progresso
                print("Calcolo embeddings BERT...")
                self.embeddings = []
                for text in tqdm(texts, desc="Generazione embeddings", unit="text"):
                    embedding = model.encode(text, show_progress_bar=False)
                    self.embeddings.append(embedding)

                self.embeddings = np.array(self.embeddings)  # Converte in array numpy
                self.model = model

                # Salva gli embeddings
                self.save_embedding(filename, self.model, self.embeddings)
                print(f"BERT embedding completato. Shape: {self.embeddings.shape}")
            except Exception as e:
                print(f"Errore durante il calcolo del BERT embedding: {e}")

        return self.embeddings
