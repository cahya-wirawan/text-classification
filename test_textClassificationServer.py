import unittest
from textclassification import TextClassificationServer
from textclassification_client import client, scan, instream
import json

class TestTextClassificationServer(unittest.TestCase):
    host, port = "localhost", 3333

    def setUp(self):
        print("setUp")

    def tearDown(self):
        print("shutDown")

    def test_ping(self):
        response = client(self.host, self.port, "PING\n")
        response = json.loads(response.decode('utf-8'))
        self.assertEqual('PONG', response['result'])

    def test_version(self):
        response = client(self.host, self.port, "VERSION\n")
        response = json.loads(response.decode('utf-8'))
        self.assertEqual('version', response['result'])

    def test_scan(self):
        response = scan(self.host, self.port, "/bin/sh")
        response = json.loads(response.decode('utf-8'))
        self.assertEqual('7ecb352cf8cb6a957edf9ff1253ccc68', response['result'])

    def test_instream(self):
        response = instream(self.host, self.port, "/bin/sh")
        response = json.loads(response.decode('utf-8'))
        self.assertEqual('7ecb352cf8cb6a957edf9ff1253ccc68', response['result'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
