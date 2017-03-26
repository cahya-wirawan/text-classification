#!/usr/bin/env python
import os.path
import logging
import yaml
import argparse
from setup_logging import setup_logging
from textclassification import TextClassificationServer


if __name__ == "__main__":
    search_directories = [".", "/etc/textclassification"]
    setup_logging()
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--address", help="define the address for the server")
    parser.add_argument("-C", "--configuration_file", default="./textclassification.yml",
                        help="set the configuration file")
    parser.add_argument("-p", "--port", help="define the port number which the server uses to listen")
    parser.add_argument("-t", "--timeout", type=float, help="define the port number which the server uses to listen")
    args = parser.parse_args()

    config_found = False
    cfg = None
    for directory in search_directories:
        path = os.path.join(directory, "textclassification.yml")
        if os.path.isfile(path) and os.access(path, os.R_OK):
            config_found = True
            break
    if config_found:
        with open(args.configuration_file, 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
    else:
        print("The configuration file is not found.")
        exit(1)

    tcs = TextClassificationServer(cfg=cfg["server"])
    try:
        logger.info("Server start")
        tcs.start(address=args.address, port=args.port, timeout=args.timeout)
    except KeyboardInterrupt as err:
        logger.info("Server quit")
