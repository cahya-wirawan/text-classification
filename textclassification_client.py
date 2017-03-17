import socket
import os
import struct
import logging
from setup_logging import setup_logging

MAX_BUFFER_SIZE = 1024


class SimpleSocket(object):
    max_data_size = 4096

    def __init__(self, address=None, port=None):
        self.logger = logging.getLogger(__name__)
        self._timeout = None
        self._address = address
        self._port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self._address, self._port))

    def close(self):
        self.logger.debug("Closing main socket")
        self._close_connection()

    def _close_connection(self):
        self.socket.close()

    def send(self, data):
        size = len(data)
        packed_header = struct.pack('=I', size)
        self.socket.sendall(packed_header + data)

    def receive(self):
        packed_header = self.socket.recv(4)
        (size, ) = struct.unpack('=I', packed_header)
        if size == 0 or size > self.max_data_size:
            return None
        data = self.socket.recv(size)
        return data


class TextClassificationClient(object):

    def __init__(self, address='localhost', port=3333):
        setup_logging()
        self.address = address
        self.port = port
        self.simple_socket = SimpleSocket(address=self.address, port=self.port)

    def client(self, message=None):
        logger = logging.getLogger(__name__)
        try:
            self.simple_socket.send(message.encode('utf-8'))
            response = self.simple_socket.receive()
            logger.debug("Client Received: {}".format(response))
            return response
        except ConnectionError as err:
            logger.error("OS error: {0}".format(err))
        # finally:
        #     if self.sso:
        #        self.sso.close()

    def scan(self, file_name=None):
        logger = logging.getLogger(__name__)
        try:
            statinfo = os.stat(file_name)
            if statinfo is not None:
                command = "SCAN:{}\n".format(file_name)
                self.simple_socket.send(command.encode('utf-8'))
                response = self.simple_socket.receive()
                logger.debug("Client Received: {}".format(response))
                return response
            else:
                return None
        except OSError as err:
            logger.error("OS error: {0}".format(err))

    def instream(self, data=None):
        logger = logging.getLogger(__name__)
        command = "INSTREAM\n"
        self.simple_socket.send(command.encode('utf-8'))
        try:
            data_len = len(data)
            start_pos = 0
            end_pos = MAX_BUFFER_SIZE
            while start_pos < data_len:
                end_pos = min(end_pos, data_len)
                self.simple_socket.send(data[start_pos:end_pos])
                start_pos += MAX_BUFFER_SIZE
                end_pos += MAX_BUFFER_SIZE
            self.simple_socket.send(b'')
            response = self.simple_socket.receive()
            logger.debug("Client Received: {}".format(response))
            return response
        except OSError as err:
            logger.error("OS error: {0}".format(err))

    def predict_stream(self, data=None):
        logger = logging.getLogger(__name__)
        command = "PREDICT_STREAM\n"
        self.simple_socket.send(command.encode('utf-8'))
        try:
            data_len = len(data)
            start_pos = 0
            end_pos = MAX_BUFFER_SIZE
            while start_pos < data_len:
                end_pos = min(end_pos, data_len)
                self.simple_socket.send(data[start_pos:end_pos])
                start_pos += MAX_BUFFER_SIZE
                end_pos += MAX_BUFFER_SIZE
            self.simple_socket.send(b'')
            response = self.simple_socket.receive()
            logger.debug("Client Received: {}".format(response))
            return response
        except OSError as err:
            logger.error("OS error: {0}".format(err))

    def predict_file(self, file_name=None):
        logger = logging.getLogger(__name__)
        try:
            statinfo = os.stat(file_name)
            if statinfo is not None:
                command = "PREDICT_FILE:{}\n".format(file_name)
                self.simple_socket.send(command.encode('utf-8'))
                response = self.simple_socket.receive()
                logger.debug("Client Received: {}".format(response))
                return response
            else:
                return None
        except OSError as err:
            logger.error("OS error: {0}".format(err))
