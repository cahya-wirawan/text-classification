#! /usr/bin/env python

import logging
import yaml
import argparse
from textclassification import TextClassificationTraining
from setup_logging import setup_logging


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

    training = TextClassificationTraining(cfg["training"])
    training.start(cn=args.classifier, dn=args.dataset)


