import socket
import sys
import os
import struct
import logging

logger = logging.getLogger("jsonSocket")
logger.setLevel(logging.DEBUG)
FORMAT = '[%(asctime)-15s][%(levelname)s][%(funcName)s] %(message)s'
logging.basicConfig(format=FORMAT)

MAX_BUFFER_SIZE = 1024
data = " ".join(sys.argv[1:])

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
    try:
        ss = simple_socket(address=address, port=port)
        ss.send(message.encode('utf-8'))
        response = ss.receive()
        print("Client Received: {}".format(response))
    except ConnectionError as err:
        print("OS error: {0}".format(err))
    finally:
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
    except OSError as err:
        print("OS error: {0}".format(err))

def instream(ip, port, file_name=None):
    statinfo = os.stat(file_name)
    if statinfo is not None:
        ss = simple_socket(address=address, port=port)
        command = "INSTREAM:{}\n".format(file_name)
        ss.send(command.encode('utf-8'))
        try:
            with open(file_name, 'rb') as f:
                while True:
                    data = f.read(MAX_BUFFER_SIZE)
                    if not data:
                        break
                    ss.send(data)
            ss.send(b'')
            response = ss.receive()
            print("Client Received: {}".format(response))
        finally:
            ss.close()

# client(ip, port, "Hello World 1:ABC:12\n")
# client(ip, port, "Hello World 2:DEF:11\n")

address, port = "localhost", 3333
client(address, port, "PING\n")
client(address, port, "VERSION\n")
client(address, port, "RELOAD\n")
scan(address, port, "/bin/sh")
instream(address, port, "/bin/sh")
