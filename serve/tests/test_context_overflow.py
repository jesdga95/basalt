"""A request over the context window is a 400 before any streaming, in the wording agent harnesses recognise."""
import http.client
import json
import unittest
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.engine.mock import MockEngine
from serve.handler import serve
from serve.service import CTX_SLACK, Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
CTX = 131072
APIS = (("/v1/chat/completions", "max_tokens"), ("/v1/messages", "max_tokens"), ("/v1/responses", "max_output_tokens"))


class ContextOverflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = MockEngine(tok, "ok", max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.svc.fit_max_tokens = False

    def post(self, path, key, prompt_tokens, max_tokens=None, stream=False):
        body = {"model": "basalt", "input": "hi"} if path == "/v1/responses" else {"model": "basalt", "messages": [{"role": "user", "content": "hi"}]}
        body["stream"] = stream
        if max_tokens is not None:
            body[key] = max_tokens
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            with mock.patch.object(self.svc, "encode_prompt", return_value=[ord("x")] * prompt_tokens):
                c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
                r = c.getresponse()
                return r.status, r.getheader("Content-Type"), json.loads(r.read())
        finally:
            c.close()

    def test_overflow_is_a_recognisable_400_before_streaming(self):
        cases = ((150722, None, False, "no room to answer"), (150722, 16, True, "no room to answer"),
                 (150722, 16, False, "+ max tokens (16)"), (1024, CTX, False, f"+ max tokens ({CTX})"))
        for path, key in APIS:
            for stream in (False, True):
                for prompt, budget, fit, detail in cases:
                    with self.subTest(path=path, stream=stream, prompt=prompt, budget=budget, fit=fit):
                        self.svc.fit_max_tokens = fit
                        with mock.patch.object(self.engine, "generate") as generate:
                            status, content_type, body = self.post(path, key, prompt, budget, stream)
                        generate.assert_not_called()
                        self.assertEqual(status, 400, body)
                        self.assertIn("application/json", content_type)
                        message = body["error"]["message"]
                        self.assertIn("exceeds the context window", message)
                        self.assertIn(f"prompt ({prompt} tokens)", message)
                        self.assertIn(detail, message)

    def test_one_token_of_room_answers_and_none_is_rejected(self):
        for path, key in APIS:
            for fit in (False, True):
                with self.subTest(path=path, fit=fit):
                    self.svc.fit_max_tokens = fit
                    status, _, body = self.post(path, key, CTX - CTX_SLACK - 1, 1)
                    self.assertEqual(status, 200, body)
                    status, _, body = self.post(path, key, CTX - CTX_SLACK, 1)
                    self.assertEqual(status, 400, body)
                    self.assertIn("exceeds the context window", body["error"]["message"])


if __name__ == "__main__":
    unittest.main()
