import sys, argparse

sys.path.append("src/data/data_processing/")
from processing import Preprocessor # type: ignore

from train.lstm_train import LSTMTrainer
from testing.lstm_test import LSTMPredictor

from train.naive_bayes_train import NaiveBayesTrainer
from testing.naive_bayes_test import NaiveBayesPredictor

from train.svm_train import SVMTrainer
from testing.svm_test import SVMPredictor

from train.logistic_reg_train import LogisticRegressionTrainer
from testing.logistic_reg_test import LogisticRegressionPredictor


def naive_bayes_predict_1(d_processing):
    # Train the model
    trainer = NaiveBayesTrainer(d_processing.data_list, embedding_type="tfidf")
    trainer.train_model()
    trainer.save_model("./src/model/naive_bayes/sentiment_model_tfidf.pkl")

    # Evaluate the model
    predictor = NaiveBayesPredictor(
        trainer.model,
        trainer.embedding_class,
        embedding_type="tfidf",
    )
    metrics = predictor.evaluate_model(trainer.X_test, trainer.y_test)
    print("Metrics (TF-IDF):", metrics)

    new_text = "didn't sleep well class like make feel stupid"
    prediction = predictor.use_model(new_text)
    print(f"Predicted sentiment (TF-IDF): {prediction}")

def naive_bayes_predict_2(d_processing):
    # Repeat for Word2Vec and BERT
    trainer = NaiveBayesTrainer(d_processing.data_list, embedding_type="word2vec")
    trainer.train_model()
    trainer.save_model("./src/model/naive_bayes/sentiment_model_word2vec.pkl")

    predictor = NaiveBayesPredictor(
        trainer.model,
        trainer.embedding_class,
        embedding_type="word2vec"
    )
    metrics = predictor.evaluate_model(trainer.X_test, trainer.y_test)
    print("Metrics (Word2Vec):", metrics)

    new_text = "didn't sleep well class like make feel stupid"
    prediction = predictor.use_model(new_text)
    print(f"Predicted sentiment (Word2Vec): {prediction}")

def naive_bayes_predict_3(d_processing):
    # Repeat for BERT
    trainer = NaiveBayesTrainer(d_processing.data_list, embedding_type="bert")
    trainer.train_model()
    trainer.save_model("./src/model/naive_bayes/sentiment_model_bert.pkl")

    predictor = NaiveBayesPredictor(
        trainer.model,
        trainer.embedding_class,
        embedding_type="bert"
    )
    metrics = predictor.evaluate_model(trainer.X_test, trainer.y_test)
    print("Metrics (BERT):", metrics)

    new_text = "didn't sleep well class like make feel stupid"
    prediction = predictor.use_model(new_text)
    print(f"Predicted sentiment (BERT): {prediction}")

def naive_bayes_predict_loop(embedding_type):

    predictor = NaiveBayesPredictor(embedding_type=f"{embedding_type}")
    predictor.load_model(f"./src/model/naive_bayes/sentiment_model_{embedding_type}.pkl")

    while True:
        new_text = input("Inserisci una frase (digita 'exit' per terminare): ")

        if new_text.lower() == "exit":
            print("Programma terminato.")
            break

        prediction = predictor.use_model(new_text)
        print(f"Il sentiment predetto è: {prediction}")

def svm_predict_1(d_processing):
    # Train the SVM model
    trainer = SVMTrainer(d_processing.data_list, embedding_type="tfidf")
    trainer.train_model()
    trainer.save_model("./src/model/svm/svm_model_tfidf.pkl")

    # Evaluate the model
    predictor = SVMPredictor(trainer.model, trainer.embedding_class, embedding_type="tfidf")
    metrics = predictor.evaluate_model(trainer.X_test, trainer.y_test)
    print("Metrics (TF-IDF):", metrics)

    # Predict new text
    new_text = "didn't sleep well class like make feel stupid"
    prediction = predictor.use_model(new_text)
    print(f"Predicted sentiment: {prediction}")

def svm_predict_2(d_processing):
    # Train the SVM model
    trainer = SVMTrainer(d_processing.data_list, embedding_type="word2vec")
    trainer.train_model()
    trainer.save_model("./src/model/svm/svm_model_word2vec.pkl")

    # Evaluate the model
    predictor = SVMPredictor(trainer.model, trainer.embedding_class, embedding_type="word2vec")
    metrics = predictor.evaluate_model(trainer.X_test, trainer.y_test)
    print("Metrics (word2vec):", metrics)

    # Predict new text
    new_text = "didn't sleep well class like make feel stupid"
    prediction = predictor.use_model(new_text)
    print(f"Predicted sentiment: {prediction}")

def svm_predict_3(d_processing):
    # Train the SVM model
    trainer = SVMTrainer(d_processing.data_list, embedding_type="bert")
    trainer.train_model()
    trainer.save_model("./src/model/svm/svm_model_bert.pkl")

    # Evaluate the model
    predictor = SVMPredictor(trainer.model, trainer.embedding_class, embedding_type="bert")
    metrics = predictor.evaluate_model(trainer.X_test, trainer.y_test)
    print("Metrics (bert):", metrics)

    # Predict new text
    new_text = "didn't sleep well class like make feel stupid"
    prediction = predictor.use_model(new_text)
    print(f"Predicted sentiment: {prediction}")

def svm_predict_loop(embedding_type):

    predictor = SVMPredictor(embedding_type=f"{embedding_type}")
    predictor.load_model(f"./src/model/svm/svm_model_{embedding_type}.pkl")

    while True:
        new_text = input("Inserisci una frase (digita 'exit' per terminare): ")

        if new_text.lower() == "exit":
            print("Programma terminato.")
            break

        prediction = predictor.use_model(new_text)
        print(f"Il sentiment predetto è: {prediction}")


def log_reg_predict_1(d_processing):
    # Addestramento del modello
    trainer = LogisticRegressionTrainer(d_processing.data_list, embedding_type="tfidf")
    trainer.train_model()
    trainer.save_model("./src/model/logistic_regression/logistic_model_tfidf.pkl")

    # Valutazione del modello
    predictor = LogisticRegressionPredictor(trainer.model, trainer.embedding_class, embedding_type="tfidf")
    metrics = predictor.evaluate_model(trainer.X_test, trainer.y_test)
    print("Metriche (TF-IDF):", metrics)

    # Uso del modello per fare predizioni
    new_text = "didn't sleep well class like make feel stupid"
    prediction = predictor.use_model(new_text)
    print(f"Sentiment predetto: {prediction}")

def log_reg_predict_2(d_processing):
    # Addestramento del modello
    trainer = LogisticRegressionTrainer(d_processing.data_list, embedding_type="word2vec")
    trainer.train_model()
    trainer.save_model("./src/model/logistic_regression/logistic_model_word2vec.pkl")

    # Valutazione del modello
    predictor = LogisticRegressionPredictor(trainer.model, trainer.embedding_class, embedding_type="word2vec")
    metrics = predictor.evaluate_model(trainer.X_test, trainer.y_test)
    print("Metriche (word2vec):", metrics)

    # Uso del modello per fare predizioni
    new_text = "didn't sleep well class like make feel stupid"
    prediction = predictor.use_model(new_text)
    print(f"Sentiment predetto: {prediction}")

def log_reg_predict_3(d_processing):
    # Addestramento del modello
    trainer = LogisticRegressionTrainer(d_processing.data_list, embedding_type="bert")
    trainer.train_model()
    trainer.save_model("./src/model/logistic_regression/logistic_model_bert.pkl")

    # Valutazione del modello
    predictor = LogisticRegressionPredictor(trainer.model, trainer.embedding_class, embedding_type="bert")
    metrics = predictor.evaluate_model(trainer.X_test, trainer.y_test)
    print("Metriche (bert):", metrics)

    # Uso del modello per fare predizioni
    new_text = "didn't sleep well class like make feel stupid"
    prediction = predictor.use_model(new_text)
    print(f"Sentiment predetto: {prediction}")

def log_reg_predict_loop(embedding_type):

    predictor = LogisticRegressionPredictor(embedding_type=f"{embedding_type}")
    predictor.load_model(f"./src/model/logistic_regression/logistic_model_{embedding_type}.pkl")

    while True:
        new_text = input("Inserisci una frase (digita 'exit' per terminare): ")

        if new_text.lower() == "exit":
            print("Programma terminato.")
            break

        prediction = predictor.use_model(new_text)
        print(f"Il sentiment predetto è: {prediction}")


def lstm_predict_1(d_processing):
    # Addestramento
    trainer = LSTMTrainer(d_processing.data_list, max_words=30000)
    trainer.train_model(epochs=10)
    trainer.save_model("./src/model/lstm/lstm_model.keras")

    # Testing
    tester = LSTMPredictor("./src/model/lstm/lstm_model.keras")
    metrics = tester.evaluate_model(trainer.X_test, trainer.y_test)
    print("Metriche di valutazione:", metrics)

    # Utilizzo
    test_text = "Adoro questo prodotto"
    prediction = tester.use_model(test_text)
    print(f"Predizione per '{test_text}': Classe {prediction}")

def lstm_predict_loop():

    predictor = LSTMPredictor("./src/model/lstm/lstm_model.keras")

    while True:
        new_text = input("Inserisci una frase (digita 'exit' per terminare): ")

        if new_text.lower() == "exit":
            print("Programma terminato.")
            break

        prediction = predictor.use_model(new_text)
        print(f"Il sentiment predetto è: {prediction}")


def evaluation_workflow(model, embedding):
    print("Sorry, not implemented.")


def main():
    # Main parser
    parser = argparse.ArgumentParser(description="A tool to manage machine learning workflows.")

    # Adding arguments
    parser.add_argument(
        "--mode",
        type=str,
        choices=["training", "use", "evaluation"],
        required=True,
        help="Specify the mode of operation: training, use, or evaluation."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["svm", "logistic-regression", "naive-bayes", "lstm"],
        required=True,
        help="Choose the model to use: SVM, Logistic Regression, Naive Bayes, or LSTM."
    )
    parser.add_argument(
        "--embedding",
        type=str,
        choices=["word2vec", "tfidf", "bert"],
        required=True,
        help="Select the embedding method: Word2Vec (word2vec), TF-IDF (tfidf), or BERT (bert)."
    )

    args = parser.parse_args()

    d_processing = Preprocessor()
    # d_processing.load_dataset("./src/dataset/bronze/data_augmented.csv")
    # d_processing.load_dataset("./src/dataset/gold/merged_processed_test.csv")
    d_processing.load_dataset("./src/dataset/gold/merged_undersample_processed.csv")

    if args.mode == "training":

        if args.model == "naive-bayes" and args.embedding == "tfidf":
            naive_bayes_predict_1(d_processing)
        elif args.model == "naive-bayes" and args.embedding == "word2vec":
            naive_bayes_predict_2(d_processing)
        elif args.model == "naive-bayes" and args.embedding == "bert":
            naive_bayes_predict_3(d_processing)
        elif args.model == "svm" and args.embedding == "tfidf":
            svm_predict_1(d_processing)
        elif args.model == "svm" and args.embedding == "word2vec":
            svm_predict_2(d_processing)
        elif args.model == "svm" and args.embedding == "bert":
            svm_predict_3(d_processing)
        elif args.model == "logistic-regression" and args.embedding == "tfidf":
            log_reg_predict_1(d_processing)
        elif args.model == "logistic-regression" and args.embedding == "word2vec":
            log_reg_predict_2(d_processing)
        elif args.model == "logistic-regression" and args.embedding == "bert":
            log_reg_predict_3(d_processing)
        elif args.model == "lstm":
            lstm_predict_1(d_processing)
        else: print("Programa terminato con errore.")

    elif args.mode == "use":

        if args.model == "naive-bayes" and args.embedding:
            naive_bayes_predict_loop(args.embedding)
        elif args.model == "svm" and args.embedding:
            svm_predict_loop(args.embedding)
        elif args.model == "logistic-regression" and args.embedding:
            log_reg_predict_loop(args.embedding)
        elif args.model == "lstm":
            lstm_predict_loop()
        else: print("Programa terminato con errore.")

    elif args.mode == "evaluation":
        evaluation_workflow(args.model, args.embedding)

if __name__ == "__main__":
    main()