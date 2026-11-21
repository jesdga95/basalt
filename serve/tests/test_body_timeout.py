"""A request body that stops arriving is a 408 and gives the handler thread back; a client that keeps sending,
however slowly, is never cut off.  handler.HTTP_BODY_S is a per-read deadline, not a total one."""
from __future__ import annotations

import json
import socket
import time
import unittest
from pathlib import Path
from unittest import mock

from serve import handler
from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate
from serve.handler import serve
from serve.service import Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
CRLF = b"\r\n"
REQ = json.dumps({"model": "basalt", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 20}).encode()


class BodyTimeout(unittest.TestCase):
    def start(self, body_s: float):
        """A server whose connection deadline is `body_s` seconds (the handler reads the constant when it is built)."""
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "hello there", max_context=4096), tok,
                      ChatTemplate.from_file(ROOT / "serve/chat_template.jinja"))
        with mock.patch.object(handler, "HTTP_BODY_S", body_s):
            return serve(svc, port=0)

    def stop(self, httpd):
        httpd.shutdown()
        httpd.server_close()

    @staticmethod
    def head(extra: list[bytes]) -> bytes:
        return (b"POST /v1/chat/completions HTTP/1.1" + CRLF + b"Host: 127.0.0.1" + CRLF
                + b"Content-Type: application/json" + CRLF + CRLF.join(extra) + CRLF + CRLF)

    @staticmethod
    def status(s: socket.socket) -> int:
        """The status of the one answer, or -1 when the server closed the connection without one."""
        data = b""
        while CRLF + CRLF not in data:
            piece = s.recv(65536)
            if not piece:
                return -1
            data += piece
        return int(data.split(b" ", 2)[1])

    def test_a_body_that_stops_arriving_is_a_408(self):
        """The client this is for: a connection that dies without a close (a laptop sleeping, a link dropping
        mid-body).  Without the deadline the thread sits in rfile.read() until the server ends."""
        httpd = self.start(0.5)
        try:
            with socket.create_connection(("127.0.0.1", httpd.server_address[1]), timeout=20) as s:
                s.sendall(self.head([b"Content-Length: 400"]) + REQ[:10])       # 390 bytes never come
                t0 = time.monotonic()
                self.assertEqual(self.status(s), 408)
                self.assertLess(time.monotonic() - t0, 10)                      # the deadline, not a hang
        finally:
            self.stop(httpd)

    def test_a_chunked_body_that_stops_is_a_408(self):
        httpd = self.start(0.5)
        try:
            with socket.create_connection(("127.0.0.1", httpd.server_address[1]), timeout=20) as s:
                s.sendall(self.head([b"Transfer-Encoding: chunked"]) + b"190" + CRLF + REQ[:8])
                self.assertEqual(self.status(s), 408)
        finally:
            self.stop(httpd)

    def test_a_slow_body_that_keeps_arriving_is_never_cut_off(self):
        """Six times slower than the deadline in total, and served: the deadline is one read, not the request."""
        httpd = self.start(1.0)
        try:
            with socket.create_connection(("127.0.0.1", httpd.server_address[1]), timeout=30) as s:
                s.sendall(self.head([f"Content-Length: {len(REQ)}".encode()]))
                for i in range(0, len(REQ), 8):
                    s.sendall(REQ[i:i + 8])
                    time.sleep(0.2)                                             # every read stays under 1 s
                self.assertEqual(self.status(s), 200)
        finally:
            self.stop(httpd)

    def test_a_connection_that_sends_nothing_is_released(self):
        """The header side of the same deadline: handle_one_request ends the request, the thread goes back."""
        httpd = self.start(0.5)
        try:
            with socket.create_connection(("127.0.0.1", httpd.server_address[1]), timeout=20) as s:
                self.assertEqual(self.status(s), -1)                            # closed, not held
        finally:
            self.stop(httpd)


if __name__ == "__main__":
    unittest.main()
