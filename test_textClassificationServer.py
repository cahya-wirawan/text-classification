import unittest
from textclassification import TextClassificationServer
from textclassification_client import client, scan, instream
import json
import tempfile
import os


class TestTextClassificationServer(unittest.TestCase):
    host, port = "localhost", 3333
    server = None
    data = b"Text Classification"

    @classmethod
    def setUpClass(self):
        print("setUpClass")
        self.server = TextClassificationServer(host=self.host, port=self.port)
        self.server.start(run_forever=False)

    @classmethod
    def tearDownClass(self):
        print("tearDownClass")
        self.server.shutdown()

    def setUp(self):
        print("setUp")

    def tearDown(self):
        print("tearDown")

    def test_ping(self):
        response = client(self.host, self.port, "PING\n")
        response = json.loads(response.decode('utf-8'))
        self.assertEqual('PONG', response['result'])

    def test_version(self):
        response = client(self.host, self.port, "VERSION\n")
        response = json.loads(response.decode('utf-8'))
        self.assertEqual('version', response['result'])

    def test_scan(self):
        fd, temp_path = tempfile.mkstemp()
        file = os.fdopen(fd, "wb")
        file.write(self.data)
        file.close()
        response = scan(self.host, self.port, temp_path)
        response = json.loads(response.decode('utf-8'))
        os.remove(temp_path)
        self.assertEqual('87d9718b72893660ff7556dc83cb9bb6', response['result'])

    def test_instream(self):
        response = instream(self.host, self.port, self.data)
        response = json.loads(response.decode('utf-8'))
        self.assertEqual('87d9718b72893660ff7556dc83cb9bb6', response['result'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
