#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import yaml
import argparse
from setup_logging import setup_logging
from textclassification import TextClassificationServer


if __name__ == "__main__":
    # Port 0 means to select an arbitrary unused port
    setup_logging()
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--address", help="define the address for the server")
    parser.add_argument("-C", "--configuration_file", default="./textclassification.yml",
                        help="set the configuration file")
    parser.add_argument("-p", "--port", help="define the port number which the server uses to listen")
    parser.add_argument("-t", "--timeout", type=float, help="define the port number which the server uses to listen")
    args = parser.parse_args()
    with open(args.configuration_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    tcs = TextClassificationServer(cfg=cfg["server"])
    try:
        logger.info("Server start")
        tcs.start(address=args.address, port=args.port, timeout=args.timeout)
    except KeyboardInterrupt as err:
        logger.info("Server quit")
