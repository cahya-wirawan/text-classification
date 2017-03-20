#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from setup_logging import setup_logging
from textclassification import TextClassificationServer


if __name__ == "__main__":
    # Port 0 means to select an arbitrary unused port
    host, port = "localhost", 3333

    setup_logging()
    logger = logging.getLogger(__name__)
    tcs = TextClassificationServer(host=host, port=port)
    try:
        logger.info("Server start")
        tcs.start()
    except KeyboardInterrupt as err:
        logger.info("Server quit")