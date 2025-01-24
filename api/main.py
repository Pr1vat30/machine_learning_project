from src.script.use_script import UseScript

class Predict:

    def __init__(self, path):
        self.use = UseScript(path)

    def predict(self, text):
        self.use.use_model(text)



