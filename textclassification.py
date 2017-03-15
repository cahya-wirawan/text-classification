#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
import socketserver
import sys
import socket
import hashlib
import struct
import json
from textcnn import TextCNNEvaluator, TextCNN

_eval = None

class TextClassificationServer(object):
    """
    Class for using TextClassificationServer with a network socket
    """
    evaluator = None

    def __init__(self, host='127.0.0.1', port=3333, timeout=None):
        """
        class initialisation
        host (string) : hostname or ip address
        port (int) : TCP port
        timeout (float or None) : socket timeout
        """
        global _eval
        self.__host = host
        self.__port = port
        self.__timeout = timeout
        self.server = None
        self.evaluator = TextCNNEvaluator()
        _eval = self.evaluator
        print("evaluator: {}".format(self.evaluator))

    @property
    def host(self):
        return self.__host

    @property
    def port(self):
        return self.__port

    def start(self, run_forever=True):
        try:
            self.server = self.ThreadedTCPServer((self.__host, self.__port),
                                                 self.ThreadedTCPRequestHandler)
            self.server.socket.settimeout(self.__timeout)
            self.server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Start a thread with the server -- that thread will then start one
            # more thread for each request
            server_thread = threading.Thread(target=self.server.serve_forever)
            # Exit the server thread when the main thread terminates
            server_thread.daemon = True
            server_thread.start()
            print("Server loop running in thread:", server_thread.name)
            if run_forever:
                self.server.serve_forever()
        except socket.error:
            e = sys.exc_info()[1]
            raise ConnectionError(e)

    def shutdown(self):
        self.server.shutdown()

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
                self.instream()
            elif header[0] == b'PREDICT_STREAM':
                self.predict_stream()

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
            response = dict()
            response["status"] = "OK"
            response["result"] = "PONG"
            response = json.dumps(response).encode('utf-8')
            self.send(response)

        def version(self):
            response = dict()
            response["status"] = "OK"
            response["result"] = "version"
            response = json.dumps(response).encode('utf-8')
            self.send(response)

        def reload(self):
            response = b'reload'
            self.send(response)

        def list_algoritm(self):
            response = dict()
            response["status"] = "OK"
            response["result"] = "list_algoritm"
            response = json.dumps(response).encode('utf-8')
            self.send(response)

        def set_algoritm(self):
            response = dict()
            response["status"] = "OK"
            response["result"] = "set_algoritm"
            response = json.dumps(response).encode('utf-8')
            self.send(response)

        def scan(self, file_name=None):
            hash_md5 = hashlib.md5(open(file_name, 'rb').read())
            response = dict()
            if hash_md5:
                response["status"] = "OK"
                response["result"] = hash_md5.hexdigest()
                response = json.dumps(response).encode('utf-8')
            else:
                response["status"] = "Error"
                response["result"] = ""
                response = json.dumps(response).encode('utf-8')
            self.send(response)

        def instream(self):
            hash_md5 = hashlib.md5()
            while True:
                data = self.receive()
                if data is None:
                    break
                hash_md5.update(data)
            response = dict()
            if hash_md5:
                response["status"] = "OK"
                response["result"] = hash_md5.hexdigest()
                response = json.dumps(response).encode('utf-8')
            else:
                response["status"] = "Error"
                response["result"] = ""
                response = json.dumps(response).encode('utf-8')
            self.send(response)

        def predict_stream(self):
            server = TextClassificationServer
            print("server: {}".format(server))
            evaluator = TextClassificationServer.evaluator
            evaluator = _eval
            stream = b''
            while True:
                data = self.receive()
                print("data:{}".format(data))
                if data is None:
                    break
                stream += data
            stream = stream.decode('utf-8')
            response = dict()
            response["status"] = "OK"
            response["result"] = evaluator.predict([stream])
            response = json.dumps(response).encode('utf-8')
            self.send(response)

    class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True
        pass
