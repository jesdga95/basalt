"""POST /v1/engine/threads/{attach,fork} through the real HTTP handler with an engine that speaks THREAD_*."""
import http.client
import json
import unittest
from pathlib import Path

from serve.engine.mock import MockEngine
from serve.engine.process import ProcessEngine
from serve.frontend import ChatTemplate
from serve.handler import serve
from serve.service import Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]


class TapeEngine(MockEngine):
    _THREAD_NAME = ProcessEngine._THREAD_NAME

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.tapes = {}

    def thread_command(self, command, timeout=None):
        w = command.split()
        if w[0] == "THREAD_ATTACH":
            self.tapes[w[1]] = (w[2], [int(x) for x in w[3].split(",")])
            return [w[0], w[1]]
        raise ValueError("unsupported")

    def thread_fork(self, parent, parent_head, child, child_head, keep=0):
        if parent not in self.tapes:
            raise ValueError("unknown thread tape")
        head, ids = self.tapes[parent]
        if head != parent_head:
            raise ValueError("stale thread tape head")
        self.tapes[child] = (child_head, ids[:keep or len(ids)])
        return len(self.tapes[child][1])


class Routes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = TapeEngine(tok, "ok", max_context=4096)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, json.loads(r.read())

    def test_attach_then_fork(self):
        self.assertEqual(self.post("/v1/engine/threads/attach", {"thread": "p", "head": "v1", "ids": [1, 2, 3, 4]})[0], 200)
        s, b = self.post("/v1/engine/threads/fork", {"parent": "p", "parent_head": "v1", "child": "c", "child_head": "v0", "keep_tokens": 2})
        self.assertEqual((s, b), (200, {"shared_prefix_tokens": 2}))
        self.assertEqual(self.engine.tapes["c"], ("v0", [1, 2]))

    def test_bad_names_are_refused_without_taking_the_engine(self):
        self.svc.fifo.acquire()                      # a busy engine: a valid request would answer 503, a bad one 409
        try:
            s, _ = self.post("/v1/engine/threads/fork", {"parent": "p", "parent_head": "v1", "child": "c c", "child_head": "v0"})
        finally:
            self.svc.fifo.release()
        self.assertEqual(s, 409)

    def test_errors(self):
        self.assertEqual(self.post("/v1/engine/threads/fork", {"parent": "x"})[0], 422)
        self.assertEqual(self.post("/v1/engine/threads/fork", {"parent": "nope", "parent_head": "v1", "child": "c2", "child_head": "v0"})[0], 409)
        self.assertEqual(self.post("/v1/engine/threads/attach", {"thread": "p", "head": "v1", "ids": []})[0], 409)
        self.assertEqual(self.post("/v1/engine/threads/fork", {"parent": "p", "parent_head": "v1", "child": "c3", "child_head": "v0", "keep_tokens": True})[0], 422)


if __name__ == "__main__":
    unittest.main()
