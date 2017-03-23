#! /usr/bin/env python

import logging
import yaml
from datetime import datetime
from setup_logging import setup_logging
from dataset import Dataset

class TextClassificationTraining(object):
    """
    Class for using TextClassificationServer with a network socket
    """
    classifiers = dict()

    def __init__(self):
        """
        class initialisation
        """
        self.logger = logging.getLogger(__name__)

        with open("textclassification.yml", 'r') as ymlfile:
            self.cfg = yaml.load(ymlfile)

        for classifier_name in self.cfg['classifier']:
            module_name = "classifier_" + classifier_name
            module = __import__(module_name)
            class_ = getattr(module, ''.join(module_name.title().split('_')))
            if class_:
                classifier = dict()
                classifier['enabled'] = self.cfg['classifier'][classifier_name]['enabled']
                default_dataset = self.cfg['datasets']['default']
                classifier['class'] = class_(self.cfg['classifier'][classifier_name],
                                             self.cfg['datasets'][default_dataset]['categories'],
                                             default_dataset, False)
                TextClassificationTraining.classifiers[classifier_name] = classifier

    def start(self):
        self.logger.info("Training starts")
        dataset_name = self.cfg['training']['dataset']['name']
        dataset = Dataset.create_dataset(self.cfg['training']['dataset'])
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        TCT = TextClassificationTraining
        for classifier in TCT.classifiers:
            TCT.classifiers[classifier]['class'].fit(dataset, "{}/{}_{}_{}.pkl".format(self.cfg['training']['data_dir'],
                                                                                       classifier, dataset_name, now))
        self.logger.info("Training end")

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)

    train = TextClassificationTraining()
    train.start()


