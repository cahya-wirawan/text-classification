import socket
import os
import struct
import logging
from setup_logging import setup_logging

MAX_BUFFER_SIZE = 1024


class SimpleSocket(object):
    max_data_size = 4096

    def __init__(self, address='127.0.0.1', port=3333):
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
    address = 'localhost'
    port = 3333

    def __init__(self):
        setup_logging()

    def client(self, address=address, port=port, message=None):
        logger = logging.getLogger(__name__)
        ss = None
        try:
            ss = SimpleSocket(address=address, port=port)
            ss.send(message.encode('utf-8'))
            response = ss.receive()
            logger.debug("Client Received: {}".format(response))
            return response
        except ConnectionError as err:
            logger.error("OS error: {0}".format(err))
        finally:
            if ss:
                ss.close()

    def scan(self, address=address, port=port, file_name=None):
        logger = logging.getLogger(__name__)
        try:
            statinfo = os.stat(file_name)
            if statinfo is not None:
                ss = SimpleSocket(address=address, port=port)
                command = "SCAN:{}\n".format(file_name)
                ss.send(command.encode('utf-8'))
                response = ss.receive()
                logger.debug("Client Received: {}".format(response))
                return response
            else:
                return None
        except OSError as err:
            logger.error("OS error: {0}".format(err))

    def instream(self, address=address, port=port, data=None):
        logger = logging.getLogger(__name__)
        ss = SimpleSocket(address=address, port=port)
        command = "INSTREAM\n"
        ss.send(command.encode('utf-8'))
        try:
            data_len = len(data)
            start_pos = 0
            end_pos = MAX_BUFFER_SIZE
            while start_pos < data_len:
                end_pos = min(end_pos, data_len)
                ss.send(data[start_pos:end_pos])
                start_pos += MAX_BUFFER_SIZE
                end_pos += MAX_BUFFER_SIZE
            ss.send(b'')
            response = ss.receive()
            logger.debug("Client Received: {}".format(response))
            return response
        finally:
            ss.close()

    def predict_stream(self, address=address, port=port, data=None):
        logger = logging.getLogger(__name__)
        ss = SimpleSocket(address=address, port=port)
        command = "PREDICT_STREAM\n"
        ss.send(command.encode('utf-8'))
        try:
            data_len = len(data)
            start_pos = 0
            end_pos = MAX_BUFFER_SIZE
            while start_pos < data_len:
                end_pos = min(end_pos, data_len)
                ss.send(data[start_pos:end_pos])
                start_pos += MAX_BUFFER_SIZE
                end_pos += MAX_BUFFER_SIZE
            ss.send(b'')
            response = ss.receive()
            logger.debug("Client Received: {}".format(response))
            return response
        finally:
            ss.close()

    def predict_file(self, address=address, port=port, file_name=None):
        logger = logging.getLogger(__name__)
        try:
            statinfo = os.stat(file_name)
            if statinfo is not None:
                ss = SimpleSocket(address=address, port=port)
                command = "PREDICT_FILE:{}\n".format(file_name)
                ss.send(command.encode('utf-8'))
                response = ss.receive()
                logger.debug("Client Received: {}".format(response))
                return response
            else:
                return None
        except OSError as err:
            logger.error("OS error: {0}".format(err))
