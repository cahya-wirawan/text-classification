#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
import socketserver
import sys
import socket
import hashlib
import struct


class TextClassificationServer(object):
    """
    Class for using TextClassificationServer with a network socket
    """

    def __init__(self, host='127.0.0.1', port=3333, timeout=None):
        """
        class initialisation
        host (string) : hostname or ip address
        port (int) : TCP port
        timeout (float or None) : socket timeout
        """
        self.__host = host
        self.__port = port
        self.__timeout = timeout
        self.server = None

    @property
    def host(self):
        return self.__host

    @property
    def port(self):
        return self.__port

    def start(self):
        try:
            self.server = self.ThreadedTCPServer((self.host, self.port),
                                                 self.ThreadedTCPRequestHandler)
            self.server.socket.settimeout(self.__timeout)

            # Start a thread with the server -- that thread will then start one
            # more thread for each request
            server_thread = threading.Thread(target=self.server.serve_forever)
            # Exit the server thread when the main thread terminates
            server_thread.daemon = True
            server_thread.start()
            print("Server loop running in thread:", server_thread.name)
            self.server.serve_forever()
            print("Will never go here")
        except socket.error:
            e = sys.exc_info()[1]
            raise ConnectionError(e)

    class ThreadedTCPRequestHandler(socketserver.StreamRequestHandler):
        max_buffer_size = 4096

        def handle(self):
            data = self.receive().rstrip()
            cur_thread = threading.current_thread()
            print("Server {} received: {}".format(cur_thread.name, data))
            header = data.split(b':')
            if header[0] == b'PING':
                self.ping()
            elif header[0] == b'VERSION':
                self.version()
            elif header[0] == b'RELOAD':
                self.reload()
            elif header[0] == b'LIST_ALGORITHM':
                self.list_algoritm()
            elif header[0] == b'SET_ALGORITHM':
                self.set_algoritm()
            elif header[0] == b'SCAN':
                file_name = header[1]
                self.scan(file_name=file_name)
            elif header[0] == b'INSTREAM':
                self.instream(file_name=cur_thread.name)

        def send(self, data):
            size = len(data)
            packed_header = struct.pack('=I', size)
            self.request.sendall(packed_header + data)

        def receive(self):
            packed_header = self.rfile.read(4)
            (size, ) = struct.unpack('=I', packed_header)
            if size == 0 or size > self.max_buffer_size:
                return None
            data = self.rfile.read(size)
            return data

        def ping(self):
            response = b'PONG'
            self.send(response)

        def version(self):
            response = b'version'
            self.send(response)

        def reload(self):
            response = b'reload'
            self.send(response)

        def list_algoritm(self):
            response = b'list_algoritm'
            self.send(response)

        def set_algoritm(self):
            response = b'set_algoritm'
            self.send(response)

        def scan(self, file_name=None):
            hash_md5 = hashlib.md5(open(file_name, 'rb').read())
            if hash_md5:
                response = bytes("OK: {}: {}".format(file_name, hash_md5.hexdigest()), 'utf-8')
            else:
                response = bytes("Not OK: {}: {}".format(file_name, ""), 'utf-8')
            self.send(response)

        def instream(self, file_name=None):
            hash_md5 = hashlib.md5()
            while True:
                data = self.receive()
                if data is None:
                    break
                hash_md5.update(data)
            if hash_md5:
                response = bytes("OK: {}: {}".format(file_name, hash_md5.hexdigest()), 'utf-8')
            else:
                response = bytes("Not OK: {}: {}".format(file_name, ""), 'utf-8')
            self.send(response)

    class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        pass
