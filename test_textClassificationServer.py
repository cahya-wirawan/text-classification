import unittest
from textclassification import TextClassificationServer


class TestTextClassificationServer(unittest.TestCase):
    host, port = "localhost", 3333

    def setUp(self):
        self.tcs = TextClassificationServer(host=self.host, port=self.port)

    def test_host(self):
        self.assertEqual(self.tcs.host, self.host)

    def test_port(self):
        self.assertEqual(self.tcs.port, self.port)

    def test_start(self):
        self.fail()


if __name__ == '__main__':
    unittest.main(verbosity=2)
