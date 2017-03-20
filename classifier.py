import os
import logging
from setup_logging import setup_logging
from abc import ABC, abstractmethod


class Classifier(ABC):
    __instances__ = dict()
    __training_dir__ = "run/{}".format(__name__)

    def __init__(self):
        setup_logging()
        self.logger = logging.getLogger(__name__)
        Classifier.__instances__[self.__class__.__name__] = self
        try:
            os.makedirs(self.__training_dir__, exist_ok=True)
        except OSError as err:
            self.logger.error("OS error: {0}".format(err))

    @abstractmethod
    def fit(self, dataset, filename):
        pass

    @abstractmethod
    def reload(self, filename):
        pass

    @abstractmethod
    def predict(self, data):
        pass