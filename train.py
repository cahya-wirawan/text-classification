#! /usr/bin/env python

import logging
import yaml
import argparse
from datetime import datetime
from setup_logging import setup_logging
from dataset import Dataset

class TextClassificationTraining(object):
    """
    Class for using TextClassificationServer with a network socket
    """
    classifiers = dict()

    def __init__(self, cfg):
        """
        class initialisation
        """
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg

        for classifier_name in self.cfg["classifier"]:
            if classifier_name == "default":
                continue
            module_name = "classifier_" + classifier_name
            module = __import__(module_name)
            class_ = getattr(module, ''.join(module_name.title().split('_')))
            if class_:
                classifier = dict()
                classifier['enabled'] = self.cfg["classifier"][classifier_name]['enabled']
                default_dataset = self.cfg["datasets"]["default"]
                classifier['class'] = class_(self.cfg["classifier"][classifier_name],
                                             self.cfg["datasets"][default_dataset]["categories"],
                                             default_dataset, False)
                TextClassificationTraining.classifiers[classifier_name] = classifier

    def start(self, cn=None, dn=None):
        self.logger.info("Training starts")
        if cn:
            classifier_name = cn
        else:
            classifier_name = self.cfg["classifier"]["default"]
        if classifier_name not in TextClassificationTraining.classifiers.keys() and classifier_name != "all":
            print("The classifier {} doesn't exist".format(classifier_name))
            return 1
        if dn:
            dataset_name = dn
        else:
            dataset_name = self.cfg["datasets"]["default"]
        if dataset_name not in self.cfg["datasets"]:
            print("The dataset {} doesn't exist".format(dataset_name))
            return 1
        dataset = Dataset.create_dataset(self.cfg["datasets"][dataset_name])
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        TCT = TextClassificationTraining
        if classifier_name == "all":
            for classifier_name in TCT.classifiers:
                result_name = "{}/{}_{}_{}".format(self.cfg['data_dir'], classifier_name, dataset_name, now)
                TCT.classifiers[classifier_name]['class'].fit(dataset, result_name)
                print("The training of {} classifier for the dataset {} is done.".format(classifier_name, dataset_name))
                print("The result is saved in: {}(.pkl)".format(result_name))
        else:
            result_name = "{}/{}_{}_{}".format(self.cfg['data_dir'], classifier_name, dataset_name, now)
            TCT.classifiers[classifier_name]['class'].fit(dataset, result_name)
            print("The training of {} classifier for the dataset {} is done.".format(classifier_name, dataset_name))
            print("The result is saved in: {}(.pkl)".format(result_name))

        self.logger.info("Training end")

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--classifier",
                        help="set classifier to use for the training (support currently bayesian, svm or cnn)")
    parser.add_argument("-C", "--configuration_file", default="./textclassification.yml",
                        help="set the configuration file")
    parser.add_argument("-d", "--dataset", help="set dataset to use for the training")

    args = parser.parse_args()
    with open(args.configuration_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    training = TextClassificationTraining(cfg['training'])
    training.start(cn=args.classifier, dn=args.dataset)


