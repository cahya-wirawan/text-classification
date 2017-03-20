import logging
from textcnn import TextCNNEvaluator
from classifier import Classifier


class ClassifierCnn(Classifier):

    def __init__(self, cfg=None, categories=None, current_category=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.categories = categories
        self.current_category = current_category
        self.evaluator = TextCNNEvaluator(cfg, self.current_category)

    def fit(self, dataset, filename):
        self.logger.info("train")

    def reload(self, filename):
        self.logger.info("reload")

    def predict(self, data):
        self.logger.info("predict")
        predicted = self.evaluator.predict(data)
        predicted = [self.categories[i] for i in predicted]
        return predicted
