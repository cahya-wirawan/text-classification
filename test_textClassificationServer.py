import unittest
from textclassification import TextClassificationServer, TextClassificationClient
import json
import tempfile
import os
import time
import logging
from setup_logging import setup_logging
from _version import __version__


class TestTextClassificationServer(unittest.TestCase):
    setup_logging()
    logger = logging.getLogger(__name__)
    run_server = False
    host, port = "localhost", 3333
    server = None
    tcc = None
    data = b"Text Classification"*1000
    # md5sum = md5(data).hexdigest()
    md5sum = "4b1c78bb298ef3d3d3ee9a244cb5e0c6"
    x_raw = ["an idealistic love story that brings out the latent 15 year old romantic in everyone",
             "there are enough moments of heartbreaking honesty to keep one glued to the screen",
             "the kind of nervous film that will either give you a mild headache or exhilarate you"]

    @classmethod
    def setUpClass(cls):
        TestTextClassificationServer.logger.debug("setUpClass")
        cls.tcc = TextClassificationClient(address=cls.host, port=cls.port)
        if cls.run_server:
            cls.server = TextClassificationServer(host=cls.host, port=cls.port)
            cls.server.start(run_forever=False)

    @classmethod
    def tearDownClass(cls):
        TestTextClassificationServer.logger.debug("tearDownClass")
        if cls.run_server:
            cls.server.shutdown()

    def setUp(self):
        self.logger.debug("setUp")

    def tearDown(self):
        self.logger.debug("tearDown")

    def test_ping(self):
        response = self.tcc.command("PING\n")
        response = json.loads(response.decode('utf-8'))
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual('PONG', response['result'])

    def test_version(self):
        response = self.tcc.command("VERSION\n")
        response = json.loads(response.decode('utf-8'))
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual(__version__, response['result'])

    def test_md5_file(self):
        fd, temp_path = tempfile.mkstemp()
        file = os.fdopen(fd, "wb")
        file.write(self.data)
        file.close()
        response = self.tcc.md5_file(temp_path)
        response = json.loads(response.decode('utf-8'))
        os.remove(temp_path)
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual(self.md5sum, response['result'])

    def test_md5_stream(self):
        response = self.tcc.md5_stream(self.data)
        response = json.loads(response.decode('utf-8'))
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual(self.md5sum, response['result'])

    def test_predict_stream_0(self):
        start = time.time()
        response = self.tcc.predict_stream(self.x_raw[0].encode('utf-8'))
        response = json.loads(response.decode('utf-8'))
        end = time.time()
        self.logger.debug("Time elapsed: {}".format(end - start))
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual({'bayesian': None, 'svm': None, 'cnn': ['positive_data']}, response['result'])

    def test_predict_stream_1(self):
        start = time.time()
        response = self.tcc.predict_stream(self.x_raw[1].encode('utf-8'))
        response = json.loads(response.decode('utf-8'))
        end = time.time()
        self.logger.debug("Time elapsed: {}".format(end - start))
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual({'bayesian': None, 'svm': None, 'cnn': ['positive_data']}, response['result'])

    def test_predict_stream_2(self):
        start = time.time()
        response = self.tcc.predict_stream('\n'.join(self.x_raw).encode('utf-8'))
        response = json.loads(response.decode('utf-8'))
        end = time.time()
        self.logger.debug("Time elapsed: {}".format(end - start))
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual({'bayesian': None, 'svm': None, 'cnn': ['positive_data', 'positive_data', 'negative_data']},
                         response['result'])

    def test_predict_file(self):
        fd, temp_path = tempfile.mkstemp()
        file = os.fdopen(fd, "wb")
        file.write(self.x_raw[1].encode('utf-8'))
        file.close()
        response = self.tcc.predict_file(temp_path)
        response = json.loads(response.decode('utf-8'))
        os.remove(temp_path)
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual({'bayesian': None, 'svm': None, 'cnn': ['positive_data']}, response['result'])

    def test_predict_file_multilines(self):
        fd, temp_path = tempfile.mkstemp()
        file = os.fdopen(fd, "wb")
        file.write('\n'.join(self.x_raw).encode('utf-8'))
        file.close()
        response = self.tcc.predict_file(temp_path)
        response = json.loads(response.decode('utf-8'))
        os.remove(temp_path)
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual({'bayesian': None, 'svm': None, 'cnn': ['positive_data', 'positive_data', 'negative_data']},
                         response['result'])

    def test_unknown_command(self):
        response = self.tcc.command("Unknown command\n")
        response = json.loads(response.decode('utf-8'))
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual('Unknown Command', response['result'])

    def test_close(self):
        response = self.tcc.command("CLOSE\n")
        response = json.loads(response.decode('utf-8'))
        self.logger.debug("{}: {}".format(self._testMethodName, response))
        self.assertEqual('Bye', response['result'])
        TestTextClassificationServer.tcc = TextClassificationClient(address=self.host, port=self.port)

if __name__ == '__main__':
    unittest.main(verbosity=2)
