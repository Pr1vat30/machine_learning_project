import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.src.models import Sequential
from keras.src.layers import Dropout, Dense, Input, BatchNormalization, LeakyReLU, ReLU

import sys
sys.path.append("src/data/data_processing/")
from embedding import Embeddings  # type: ignore

class NeuralNetworkTrainer:
    """
    Gestisce l'addestramento di una serie di modelli basati su reti neurali.
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

    def _create_feedforward_tfidf(self, input_shape):
        """ Crea una rete neurale fully connected (feedforward) """
        self.model = Sequential()
        self.model.add(Input(shape=(input_shape,)))  # Aggiungi un layer Input
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3, activation='softmax'))  # 3 classi in output
        self.model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def _create_feedforward_word2vec(self, input_shape):
        """ Crea una rete neurale fully connected (feedforward) """
        self.model = Sequential([
            Input(shape=(input_shape,)),

            # Primo blocco
            Dense(256, activation=None),
            BatchNormalization(),
            ReLU(),  # Attivazione
            Dropout(0.3),

            # Secondo blocco
            Dense(128, activation=None),
            BatchNormalization(),
            ReLU(),
            Dropout(0.3),

            # Terzo blocco
            Dense(64, activation=None),
            BatchNormalization(),
            ReLU(),
            Dropout(0.4),

            # Output
            Dense(3, activation="softmax")  # 3 classi
        ])
        self.model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def _create_feedforward_bert(self, input_shape):
        """ Crea una rete neurale fully connected (feedforward) """
        self.model = Sequential([
            Input(shape=(input_shape,)),

            # Primo blocco
            Dense(512),
            LeakyReLU(negative_slope=0.1),  # Attivazione avanzata
            BatchNormalization(),
            Dropout(0.3),

            # Secondo blocco
            Dense(256),
            LeakyReLU(negative_slope=0.1),
            BatchNormalization(),
            Dropout(0.3),

            # Terzo blocco
            Dense(128),
            LeakyReLU(negative_slope=0.1),
            BatchNormalization(),
            Dropout(0.4),

            # Quarto blocco
            Dense(64),
            LeakyReLU(negative_slope=0.1),
            Dropout(0.4),

            # Output
            Dense(3, activation='softmax')  # 3 classi
        ])
        self.model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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
                self._create_feedforward_tfidf(input_shape=X_train.shape[1])
                self.model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)


            elif self.embedding_type == "word2vec":
                embedding = self.embedding_class.apply_word2vec_embedding()
                X = self.scaler.fit_transform(embedding)

                # Divisione dei dati in training e testing
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_size, random_state=42
                )

                # Crea e allena la rete neurale
                self._create_feedforward_word2vec(input_shape=(X_train.shape[1]))
                self.model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


            elif self.embedding_type == "bert":
                embedding = self.embedding_class.apply_bert_embedding()
                X = self.scaler.fit_transform(embedding)

                # Divisione dei dati in training e testing
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_size, random_state=42
                )

                # Crea e allena la rete neurale
                self._create_feedforward_bert(input_shape=(X_train.shape[1]))
                self.model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

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