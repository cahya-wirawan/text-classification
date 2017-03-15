import unittest
from textclassification import TextClassificationServer
from textclassification_client import client, scan, instream, predict_stream
import json
import tempfile
import os


class TestTextClassificationServer(unittest.TestCase):
    host, port = "localhost", 3333
    server = None
    data = b"Text Classification"*1000
    # md5sum = md5(data).hexdigest()
    md5sum = "4b1c78bb298ef3d3d3ee9a244cb5e0c6"
    x_raw = [b"a masterpiece four years in the making", b"everything is off."]

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
        self.assertEqual(self.md5sum, response['result'])

    def test_instream(self):
        response = instream(self.host, self.port, self.data)
        response = json.loads(response.decode('utf-8'))
        self.assertEqual(self.md5sum, response['result'])

    def test_predict_stream(self):
        response = predict_stream(self.host, self.port, self.x_raw[0])
        response = json.loads(response.decode('utf-8'))
        self.assertEqual("positive_data", response['result'][0])

if __name__ == '__main__':
    unittest.main(verbosity=2)
