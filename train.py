#! /usr/bin/env python
import os.path
import logging
import yaml
import argparse
from textclassification import TextClassificationTraining
from setup_logging import setup_logging


if __name__ == "__main__":
    search_directories = [".", "/etc/textclassification"]
    setup_logging()
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--classifier",
                        help="set classifier to use for the training (support currently bayesian, svm or cnn)")
    parser.add_argument("-C", "--configuration_file", help="set the configuration file")
    parser.add_argument("-d", "--dataset", help="set dataset to use for the training")
    args = parser.parse_args()

    config_found = False
    cfg = None
    config_file_path = None
    if args.configuration_file:
        config_file_path = args.configuration_file
        if os.path.isfile(config_file_path) and os.access(config_file_path, os.R_OK):
            config_found = True
    else:
        for directory in search_directories:
            config_file_path = os.path.join(directory, "textclassification.yml")
            if os.path.isfile(config_file_path) and os.access(config_file_path, os.R_OK):
                config_found = True
                break
    if config_found:
        with open(config_file_path, 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
    else:
        print("The configuration file is not found.")
        exit(1)

    training = TextClassificationTraining(cfg["training"])
    training.start(cn=args.classifier, dn=args.dataset)


