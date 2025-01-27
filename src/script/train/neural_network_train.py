import pickle, tensorflow as tf

import numpy as np
import tf_keras.optimizers.legacy
import torch
from keras.src.callbacks import EarlyStopping

from tf_keras.optimizers.legacy import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dropout, Dense, Input
from keras.src.optimizers import Adam
from transformers import TFBertForSequenceClassification, BertTokenizer, TFDistilBertForSequenceClassification

import sys
sys.path.append("src/data/data_processing/")
from embedding import Embeddings  # type: ignore

class NeuralNetworkTrainer:
    """
    Gestisce l'addestramento di una serie di modelli basati su reti neurali.
    Supporta feedforward, LSTM e BERT in base al tipo di embedding scelto.
    """

    def __init__(self, dataset, test_size=0.2, embedding_type="tfidf"):
        """
        Inizializza la classe con un dataset, la dimensione del testing set e il tipo di embedding.

        :param dataset: Lista di tuple (text, sentiment) per il training.
        :param test_size: Proporzione dei dati da utilizzare per il testing (default 20%).
        :param embedding_type: Metodo di embedding da utilizzare ('tfidf', 'word2vec', 'bert').
        """
        self.dataset = dataset
        self.test_size = test_size
        self.embedding_type = embedding_type
        self.model = None
        self.embedding_class = None
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()

    def _create_feedforward(self, input_shape):
        """Crea una rete neurale fully connected (feedforward)."""
        self.model = Sequential()
        self.model.add(Input(shape=(input_shape,)))  # Aggiungi un layer Input
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3, activation='softmax'))  # 3 classi in output
        self.model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def _create_lstm(self, input_shape):
        """Crea un modello LSTM per il testo."""
        self.model = Sequential()
        self.model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=input_shape))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3, activation='softmax'))  # 3 classi in output
        self.model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def _create_bert(self):
        # Carica il modello BERT pre-addestrato
        self.model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3, attn_implementation="sdpa")

        # Congela tutti i layer tranne gli ultimi 4
        for layer in self.model.distilbert.transformer.layer[:-2]:
            layer.trainable = False

        # Configura l'ottimizzatore
        optimizer = tf_keras.optimizers.legacy.Adam(learning_rate=2e-5)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self):
        """
        Addestra il modello in base al tipo di embedding scelto.
        """
        try:
            # Estrazione dei testi e delle etichette dal dataset
            texts, sentiments = zip(*self.dataset)

            # Codifica le etichette in numeri (0, 1, 2)
            y = self.label_encoder.fit_transform(sentiments)

            # Creazione degli embedding
            self.embedding_class = Embeddings(self.dataset)


            if self.embedding_type == "tfidf":
                embedding = self.embedding_class.apply_tfidf_embedding()
                X = embedding

                # Divisione dei dati in training e testing
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_size, random_state=42
                )

                # Crea e allena la rete feedforward
                self._create_feedforward(input_shape=X_train.shape[1])
                self.model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)


            elif self.embedding_type == "word2vec":
                embedding = self.embedding_class.apply_word2vec_embedding()
                X = self.scaler.fit_transform(embedding)

                # Divisione dei dati in training e testing
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_size, random_state=42
                )

                # Adatta la forma dell'input per la LSTM
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

                # Crea e allena la LSTM
                self._create_lstm(input_shape=(X_train.shape[1], 1))
                self.model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)


            elif self.embedding_type == "bert":

                # Tokenizzazione dei testi

                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

                tokenized_inputs = tokenizer(
                    texts, padding=True, truncation=True, return_tensors="tf"
                )

                input_ids = tokenized_inputs["input_ids"]
                attention_mask = tokenized_inputs["attention_mask"]

                # Divisione dei dati in training e testing

                indices = np.arange(len(y))

                train_indices, test_indices, y_train, y_test = train_test_split(
                    indices, y, test_size=self.test_size, random_state=42
                )

                # Applica la divisione ai tensori tokenizzati

                X_train = {
                    "input_ids": tf.gather(input_ids, train_indices),
                    "attention_mask": tf.gather(attention_mask, train_indices),
                }

                X_test = {
                    "input_ids": tf.gather(input_ids, test_indices),
                    "attention_mask": tf.gather(attention_mask, test_indices),
                }

                # Crea e allena il modello BERT

                self._create_bert()

                # Addestra il modello
                history = self.model.fit(
                    X_train, y_train,
                    epochs=3,
                    batch_size=32,
                    validation_split=0.2,
                )


            else:
                raise ValueError(
                    f"Tipo di embedding '{self.embedding_type}' non supportato. Usa 'tfidf', 'word2vec', o 'bert'."
                )

            self.X_test, self.y_test = X_test, y_test
            self.X_train, self.y_train = X_train, y_train

            print(f"Modello allenato correttamente con embedding {self.embedding_type}!")

        except Exception as e:
            print(f"Errore durante l'addestramento: {e}")

    def save_model(self, filepath):
        """
        Salva il modello allenato e la classe di embedding su disco.

        :param filepath: Percorso del file in cui salvare il modello.
        """
        try:
            with open(filepath, "wb") as f:
                pickle.dump(
                    {
                        "model": self.model,
                        "embedding_class": self.embedding_class,
                        "scaler": self.scaler if self.embedding_type != "bert" else None,
                        "label_encoder": self.label_encoder,  # Salva anche il label encoder
                    },
                    f,
                )
            print(f"Modello e embedding salvati correttamente in {filepath}!")
        except Exception as e:
            print(f"Errore durante il salvataggio del modello: {e}")