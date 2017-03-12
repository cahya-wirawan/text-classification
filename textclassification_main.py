#!/usr/bin/env python
# -*- coding: utf-8 -*-

from textclassification import TextClassificationServer


if __name__ == "__main__":
    # Port 0 means to select an arbitrary unused port
    host, port = "localhost", 3333

    tcs = TextClassificationServer(host=host, port=port)
    tcs.start()
