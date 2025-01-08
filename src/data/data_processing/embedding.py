import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer


class Embeddings:

    def __init__(self, data_list):
        """
        Inizializza la classe con una lista di tuple (text, sentiment).
        """
        self.data = data_list
        self.tfidf_model = None
        self.w2v_model = None
        self.bert_model = None

    def tokenize_texts(self):
        """
        Tokenizza i testi presenti nella lista di tuple (text, sentiment).
        """
        return [text.split() for text, _ in self.data]

    def apply_tfidf_embedding(self):
        """
        Genera embeddings TF-IDF dai testi.
        """
        try:
            texts = [text for text, _ in self.data]
            vectorizer = TfidfVectorizer()
            tfidf_embeddings = vectorizer.fit_transform(texts)
            print(f"TF-IDF embedding completato. Shape: {tfidf_embeddings.shape}")
            self.tfidf_model = vectorizer
            return tfidf_embeddings
        except Exception as e:
            print(f"Errore durante il calcolo del TF-IDF: {e}")

    def apply_word2vec_embedding(self, vector_size=100, window=5, min_count=1):
        """
        Genera embeddings Word2Vec dai testi tokenizzati.
        """
        try:
            tokenized_texts = self.tokenize_texts()
            model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=4,
            )
            word2vec_embeddings = np.array(
                [
                    np.mean(
                        [model.wv[word] for word in words if word in model.wv]
                        or [np.zeros(vector_size)],
                        axis=0,
                    )
                    for words in tokenized_texts
                ]
            )
            print(f"Word2Vec embedding completato. Shape: {word2vec_embeddings.shape}")
            self.w2v_model = model
            return word2vec_embeddings
        except Exception as e:
            print(f"Errore durante il calcolo del Word2Vec: {e}")

    def apply_bert_embedding(self, model_name="all-MiniLM-L6-v2"):
        """
        Genera embeddings usando un modello pre-addestrato Sentence Transformer (BERT).
        """
        try:
            texts = [text for text, _ in self.data]
            model = SentenceTransformer(model_name)
            bert_embeddings = model.encode(texts)
            print(f"BERT embedding completato. Shape: {bert_embeddings.shape}")
            self.bert_model = model
            return bert_embeddings
        except Exception as e:
            print(f"Errore durante il calcolo del BERT embedding: {e}")
