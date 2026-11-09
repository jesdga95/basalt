"""A request body sent as Transfer-Encoding: chunked (a relay or proxy) is decoded on every route that reads a body;
a malformed or oversized body (chunked or by Content-Length) is a 400 / 413, not an empty body."""
from __future__ import annotations

import json
import socket
import unittest
from pathlib import Path

from serve import handler
from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate
from serve.handler import serve
from serve.service import Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
CRLF = b"\r\n"


def chunked(data: bytes, size: int = 7, trailers: bytes = b"") -> bytes:
    out = b""
    for i in range(0, len(data), size):
        piece = data[i:i + size]
        out += b"%x;ext=1" % len(piece) + CRLF + piece + CRLF
    return out + b"0" + CRLF + trailers + CRLF


class ChunkedBody(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = MockEngine(tok, "hello there", max_context=4096)
        cls.svc = Service(cls.engine, tok, ChatTemplate.from_file(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def send(self, head: bytes, body: bytes):
        """-> (status, JSON body) of one raw request."""
        with socket.create_connection(("127.0.0.1", self.port), timeout=20) as s:
            s.sendall(b"POST " + head + b"Host: 127.0.0.1" + CRLF + b"Connection: close" + CRLF + CRLF + body)
            s.shutdown(socket.SHUT_WR)
            data = b""
            while True:
                piece = s.recv(65536)
                if not piece:
                    break
                data += piece
        status = int(data.split(b" ", 2)[1])
        payload = data.split(CRLF + CRLF, 1)[1]
        return status, (json.loads(payload) if payload.strip().startswith(b"{") else payload)

    def chat(self, extra: bytes, body: bytes, path: bytes = b"/v1/chat/completions"):
        return self.send(path + b" HTTP/1.1" + CRLF + b"Content-Type: application/json" + CRLF + extra, body)

    REQ = json.dumps({"model": "basalt", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 20}).encode()

    def test_a_chunked_chat_request_is_read(self):
        status, body = self.chat(b"Transfer-Encoding: chunked" + CRLF, chunked(self.REQ))
        self.assertEqual(status, 200, body)
        self.assertIn("choices", body)

    def test_chunked_wins_over_content_length(self):
        status, body = self.chat(b"Transfer-Encoding: chunked" + CRLF + b"Content-Length: 3" + CRLF,
                                 chunked(self.REQ, trailers=b"X-Trailer: 1" + CRLF))
        self.assertEqual(status, 200, body)

    def test_a_content_length_request_is_unchanged(self):
        status, body = self.chat(b"Content-Length: %d" % len(self.REQ) + CRLF, self.REQ)
        self.assertEqual(status, 200, body)

    def test_every_body_route_reads_chunked(self):
        for path in (b"/v1/messages", b"/v1/messages/count_tokens", b"/v1/responses"):
            body = json.dumps({"model": "basalt", "max_tokens": 8, "input": "hi",
                               "messages": [{"role": "user", "content": "hi"}]}).encode()
            with self.subTest(path=path):
                status, got = self.chat(b"Transfer-Encoding: chunked" + CRLF, chunked(body), path)
                self.assertEqual(status, 200, got)

    def test_a_bad_chunk_size_is_a_400(self):
        status, body = self.chat(b"Transfer-Encoding: chunked" + CRLF, b"zz" + CRLF + b"{}" + CRLF + b"0" + CRLF + CRLF)
        self.assertEqual(status, 400, body)
        self.assertIn("chunk", body["error"]["message"])

    def test_a_truncated_chunked_body_is_a_400(self):
        status, body = self.chat(b"Transfer-Encoding: chunked" + CRLF, b"10" + CRLF + b"{}")
        self.assertEqual(status, 400, body)

    def test_an_oversized_chunked_body_is_a_413_and_is_never_allocated(self):
        old = self.svc.max_payload
        self.svc.max_payload = 1000
        try:
            status, body = self.chat(b"Transfer-Encoding: chunked" + CRLF, b"FFFFFFFF" + CRLF)   # announces 4 GiB
            self.assertEqual(status, 413, body)
            status, body = self.chat(b"Transfer-Encoding: chunked" + CRLF, chunked(b"x" * 3000, size=500))
            self.assertEqual(status, 413, body)
        finally:
            self.svc.max_payload = old

    def test_an_oversized_content_length_body_is_a_413(self):
        old = self.svc.max_payload
        self.svc.max_payload = 1000
        try:
            status, body = self.chat(b"Content-Length: 5000" + CRLF, b"{}")
            self.assertEqual(status, 413, body)
        finally:
            self.svc.max_payload = old

    def test_the_body_cap_defaults_to_256_mib_and_the_413_names_a_custom_one(self):
        self.assertEqual(self.svc.max_payload, 256 << 20)
        old = self.svc.max_payload
        self.svc.max_payload = 3 << 20
        try:
            status, body = self.chat(b"Content-Length: " + str(4 << 20).encode() + CRLF, b"{}")
            self.assertEqual(status, 413, body)
            self.assertIn("larger than 3 MiB", body["error"]["message"])
        finally:
            self.svc.max_payload = old

    def test_a_bad_content_length_is_a_400(self):
        for value in (b"abc", b"-5"):
            with self.subTest(value=value):
                status, body = self.chat(b"Content-Length: " + value + CRLF, b"{}")
                self.assertEqual(status, 400, body)

    def test_a_chunked_control_body_is_consumed(self):
        head = b"/unload HTTP/1.1" + CRLF + b"Content-Type: application/json" + CRLF + b"Transfer-Encoding: chunked" + CRLF
        status, got = self.send(head, chunked(b"{}"))
        self.assertIn(status, (200, 409), got)

    def test_an_oversized_chunked_control_body_is_a_413(self):
        head = b"/load HTTP/1.1" + CRLF + b"Content-Type: application/json" + CRLF + b"Transfer-Encoding: chunked" + CRLF
        status, got = self.send(head, b"FFFFFF" + CRLF)
        self.assertEqual(status, 413, got)


if __name__ == "__main__":
    unittest.main()
