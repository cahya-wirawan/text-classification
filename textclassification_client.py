import socket
import os
import struct
import logging

logger = logging.getLogger("jsonSocket")
logger.setLevel(logging.DEBUG)
FORMAT = '[%(asctime)-15s][%(levelname)s][%(funcName)s] %(message)s'
logging.basicConfig(format=FORMAT)

MAX_BUFFER_SIZE = 1024

class simple_socket(object):
    max_data_size = 4096

    def __init__(self, address='127.0.0.1', port=3333):
        self._timeout = None
        self._address = address
        self._port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self._address, self._port))

    def close(self):
        logger.debug("closing main socket")
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


def client(address='localhost', port=3333, message=None):
    ss = None
    try:
        ss = simple_socket(address=address, port=port)
        ss.send(message.encode('utf-8'))
        response = ss.receive()
        print("Client Received: {}".format(response))
        return response
    except ConnectionError as err:
        print("OS error: {0}".format(err))
    finally:
        if ss:
            ss.close()


def scan(address='localhost', port=3333, file_name=None):
    try:
        statinfo = os.stat(file_name)
        if statinfo is not None:
            ss = simple_socket(address=address, port=port)
            file_size = statinfo.st_size
            command = "SCAN:{}\n".format(file_name)
            ss.send(command.encode('utf-8'))
            response = ss.receive()
            print("Client Received: {}".format(response))
            return response
        else:
            return None
    except OSError as err:
        print("OS error: {0}".format(err))


def instream(address='localhost', port=3333, data=None):
    ss = simple_socket(address=address, port=port)
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
        print("Client Received: {}".format(response))
        return response
    finally:
        ss.close()

# address, port = "localhost", 3333
# client(address, port, "PING\n")
# client(address, port, "VERSION\n")
# client(address, port, "RELOAD\n")
# scan(address, port, "/bin/sh")
# instream(address, port, b"Text Classification")
