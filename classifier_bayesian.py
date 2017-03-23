from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.externals import joblib
from classifier import Classifier


class ClassifierBayesian(Classifier):

    def __init__(self, cfg=None, categories=None, current_category=None, load=True):
        super().__init__()
        self.categories = categories
        self.current_category = current_category
        if load:
            self.clf = joblib.load(cfg['training_file'][self.current_category])

    def fit(self, dataset, filename):
        self.logger.debug("fit")
        self.clf = Pipeline([('vect', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf', MultinomialNB())
                             ])
        self.clf.fit(dataset.get_dataset()['data'], dataset.get_dataset()['target'])
        joblib.dump(self.clf, filename, compress=9)

    def reload(self, filename):
        self.logger.info("reload")
        self.clf = joblib.load(filename)

    def predict(self, data):
        self.logger.debug("predict")
        predicted = self.clf.predict(data)
        predicted = [self.categories[i] for i in predicted]
        return predicted
        # self.logger.debug('Naive Bayes correct prediction: {:4.2f}'.format(np.mean(predicted == twenty_test.target)))
        # self.logger.debug(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
