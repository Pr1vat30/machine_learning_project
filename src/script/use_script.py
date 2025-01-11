from src.script.testing.naive_bayes_test import NaiveBayesPredictor

class UseScript:

    def __init__(self, path):
        self.predictor = None
        self.load_model(path)

    def load_model(self, path):
        self.predictor = NaiveBayesPredictor(embedding_type="tfidf")
        self.predictor.load_model(path)


    def use_model(self, text):
        prediction = self.predictor.use_model(text)
        return prediction