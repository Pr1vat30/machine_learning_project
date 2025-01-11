import pickle
from keras.src.models import Sequential
from keras.src.layers import Bidirectional, Embedding, LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split
from tf_keras.src.preprocessing.text import Tokenizer
from tf_keras.src.utils import pad_sequences, to_categorical


class LSTMTrainer:
    def __init__(self, dataset, test_size=0.2, max_words=15000):
        """
        Inizializza la classe per l'addestramento del modello.
        :param dataset: Lista di tuple (text, sentiment)
        :param test_size: Proporzione dei dati da utilizzare per il testing
        :param max_words: Numero massimo di parole nel vocabolario del tokenizer
        """
        self.dataset = dataset
        self.test_size = test_size
        self.max_words = max_words
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
        self.model = None

    def split_data(self):
        texts, sentiments = zip(*self.dataset)
        return texts, sentiments

    def preprocess_data(self):
        texts, sentiments = self.split_data()
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, padding="post")

        label_map = {label: idx for idx, label in enumerate(set(sentiments))}
        sentiments_numeric = [label_map[sentiment] for sentiment in sentiments]

        print(label_map)

        num_classes = len(label_map)
        sentiments_encoded = to_categorical(sentiments_numeric, num_classes=num_classes)
        return padded_sequences, sentiments_encoded, num_classes

    def build_model(self, num_classes):
        self.model = Sequential([
            Embedding(input_dim=self.max_words, output_dim=128),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            LSTM(32),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(num_classes, activation="softmax")
        ])
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        print("Modello LSTM costruito correttamente!")

    def train_model(self, epochs=10, batch_size=32):
        X, y, num_classes = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)

        if self.model is None:
            self.build_model(num_classes)

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        self.X_test, self.y_test = X_test, y_test
        print("Modello LSTM allenato correttamente!")

    def save_model(self, filepath):
        try:
            self.model.save(filepath)
            with open(f"{filepath}_tokenizer.pkl", "wb") as f:
                pickle.dump(self.tokenizer, f)
            print(f"Modello e tokenizer salvati correttamente in {filepath}!")
        except Exception as e:
            print(f"Errore durante il salvataggio del modello: {e}")