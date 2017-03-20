from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.externals import joblib
from classifier import Classifier


class ClassifierBayesian(Classifier):

    def __init__(self, cfg=None, categories=None):
            super().__init__()
            self.clf = Pipeline([('vect', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf', MultinomialNB())
                                  ])

    def fit(self, dataset, filename):
        self.logger.debug("fit")
        self.clf.fit(dataset.get_dataset()['data'], dataset.get_dataset()['target'])
        joblib.dump(self.clf, filename, compress=9)

    def reload(self, filename):
        self.logger.info("reload")
        self.clf = joblib.load(filename)

    def predict(self, data):
        self.logger.debug("predict")
        predicted = self.clf.predict(data)
        return predicted
        # self.logger.debug('Naive Bayes correct prediction: {:4.2f}'.format(np.mean(predicted == twenty_test.target)))
        # self.logger.debug(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
