import logging
from textcnn import TextCNNTraining, TextCNNEvaluator
from classifier import Classifier


class ClassifierCnn(Classifier):

    def __init__(self, cfg=None, categories=None, current_category=None, load=True):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg
        self.categories = categories
        self.current_category = current_category
        if load:
            self.evaluator = TextCNNEvaluator(self.cfg, self.current_category)
        self.clf = None

    def fit(self, dataset, filename):
        self.logger.info("train")
        self.clf = TextCNNTraining(self.cfg)
        self.clf.fit(dataset, filename)

    def reload(self, filename):
        self.logger.info("reload")

    def predict(self, data):
        self.logger.info("predict")
        predicted = self.evaluator.predict(data)
        predicted = [self.categories[i] for i in predicted]
        return predicted
