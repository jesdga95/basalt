"""llama.cpp's /tokenize and /detokenize, over HTTP against the mock engine. No engine, no pack."""
from __future__ import annotations

import http.client
import json
import unittest

from serve.handler import serve
from serve.tests.test_reasoning_effort import qwen_service


class Tokenize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = qwen_service()
        cls.httpd = serve(cls.svc, port=0)

    @classmethod
    def tearDownClass(cls):
        cls.svc.stop()
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        c = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=30)
        try:
            c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            return r.status, json.loads(r.read())
        finally:
            c.close()

    def test_round_trip(self):
        text = "héllo <|im_end|> wörld"
        status, out = self.post("/tokenize", {"content": text})
        self.assertEqual(status, 200)
        self.assertEqual(out["tokens"], self.svc.tok.encode(text, parse_special=True))
        self.assertEqual(self.post("/detokenize", {"tokens": out["tokens"]}), (200, {"content": text}))

    def test_parse_special(self):
        on = self.post("/tokenize", {"content": "<|im_end|>"})[1]["tokens"]
        off = self.post("/tokenize", {"content": "<|im_end|>", "parse_special": False})[1]["tokens"]
        self.assertEqual(len(on), 1)
        self.assertEqual(off, list(b"<|im_end|>"))

    def test_bad_bodies_are_400(self):
        for path, body in (("/tokenize", {}), ("/tokenize", {"content": 5}), ("/tokenize", {"content": "a", "parse_special": 1}),
                           ("/detokenize", {}), ("/detokenize", {"tokens": [1, "2"]}), ("/detokenize", {"tokens": [-1]}),
                           ("/detokenize", {"tokens": [True]}), ("/detokenize", {"tokens": [10 ** 9]}),
                           ("/apply-template", {}), ("/apply-template", {"messages": "hi"})):
            with self.subTest(path=path, body=body):
                self.assertEqual(self.post(path, body)[0], 400)

    def test_apply_template_renders_without_generating(self):
        msgs = [{"role": "user", "content": "hi"}]
        status, out = self.post("/apply-template", {"messages": msgs})
        self.assertEqual(status, 200)
        want, _ = self.svc.render_text(msgs, None, self.svc.thinking_kwargs({}))
        self.assertEqual(out, {"prompt": want})

    def test_apply_template_reasoning_effort(self):
        msgs = [{"role": "user", "content": "hi"}]
        renders = {}
        for effort in ("none", "low", "medium", "high"):
            status, out = self.post("/apply-template", {"messages": msgs, "reasoning_effort": effort})
            self.assertEqual(status, 200)
            renders[effort] = out["prompt"]
        self.assertNotEqual(renders["none"], renders["low"])
        self.assertNotEqual(renders["low"], renders["high"])


if __name__ == "__main__":
    unittest.main()
