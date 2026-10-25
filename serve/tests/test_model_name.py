"""A request must name the served model; unknown or missing is a 404 in the API's own shape (mock engine).
the any-model-name option ignores the field, whatever it holds."""
from __future__ import annotations

import http.client
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.engine.mock import MockEngine  # noqa: E402
from serve.handler import serve  # noqa: E402
from serve.service import Service  # noqa: E402
from serve.util.text import ByteTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ALIAS, FILE = "Qwen 3.8 FN", "Qwen3.8-FN-NVFP4.basalt"
MSGS = [{"role": "user", "content": "hi"}]
PATHS = {"/v1/chat/completions": {"messages": MSGS, "max_tokens": 8},
         "/v1/messages": {"messages": MSGS, "max_tokens": 8},
         "/v1/messages/count_tokens": {"messages": MSGS},
         "/v1/responses": {"input": "hi", "max_output_tokens": 8}}


class ModelName(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, ["Hello."], max_context=4096)
        engine.model_path = "/packs/" + FILE
        self.httpd = serve(Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), model_name=ALIAS),
                           port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            c.request("POST", path, body=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
            r = c.getresponse()
            return r.status, r.read().decode()
        finally:
            c.close()

    def test_the_listed_name_and_the_file_name_are_served(self):
        for path, body in PATHS.items():
            for name in (ALIAS, FILE):
                with self.subTest(path=path, name=name):
                    status, _ = self.post(path, {"model": name, **body})
                    self.assertEqual(status, 200)

    def test_unknown_is_404_in_the_api_shape(self):
        for path, body in PATHS.items():
            with self.subTest(path=path):
                status, raw = self.post(path, {"model": "gpt-x", **body})
                self.assertEqual(status, 404)
                if path.startswith("/v1/messages"):
                    self.assertEqual(json.loads(raw), {"type": "error", "error": {
                        "type": "not_found_error", "message": "model: gpt-x"}})
                else:
                    self.assertEqual(json.loads(raw), {"error": {
                        "message": "The model 'gpt-x' does not exist", "type": "invalid_request_error",
                        "param": "model", "code": "model_not_found"}})

    def test_missing_or_null_is_404(self):
        for path, body in PATHS.items():
            for extra in ({}, {"model": None}):
                with self.subTest(path=path, extra=extra):
                    status, raw = self.post(path, {**extra, **body})
                    self.assertEqual(status, 404)
                    err = json.loads(raw)
                    if path.startswith("/v1/messages"):
                        self.assertEqual(err["error"], {"type": "not_found_error", "message": "model: required"})
                    else:
                        self.assertEqual(err["error"]["message"], "model is required")
                        self.assertEqual(err["error"]["param"], "model")

    def test_the_response_names_the_served_model(self):
        for stream in (False, True):
            _, raw = self.post("/v1/chat/completions", {"model": FILE, "stream": stream, **PATHS["/v1/chat/completions"]})
            self.assertIn(f'"model": "{ALIAS}"', raw)
            self.assertNotIn(FILE, raw)
            _, raw = self.post("/v1/messages", {"model": FILE, "stream": stream, **PATHS["/v1/messages"]})
            self.assertIn(f'"model": "{ALIAS}"', raw)
            self.assertNotIn(FILE, raw)
            _, raw = self.post("/v1/responses", {"model": FILE, "stream": stream, **PATHS["/v1/responses"]})
            self.assertIn(f'"model": "{ALIAS}"', raw)
            self.assertNotIn(FILE, raw)


class AnyModelName(unittest.TestCase):
    """any-model-name: a generation request is answered whatever model it names, or none (mock engine)."""

    def setUp(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, ["Hello."], max_context=4096)
        engine.model_path = "/packs/" + FILE
        self.httpd = serve(Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), model_name=ALIAS,
                                   any_model_name=True), port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            c.request("POST", path, body=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
            r = c.getresponse()
            return r.status, r.read().decode()
        finally:
            c.close()

    def test_any_name_or_none_is_served(self):
        for path, body in PATHS.items():
            for name in ("gpt-x", "", FILE.upper(), ALIAS + "-other", 7, None):
                with self.subTest(path=path, name=name):
                    self.assertEqual(self.post(path, {"model": name, **body})[0], 200)
            with self.subTest(path=path, name="the field absent"):
                self.assertEqual(self.post(path, body)[0], 200)

    def test_the_response_still_names_the_served_model(self):
        for stream in (False, True):
            _, raw = self.post("/v1/chat/completions",
                               {"model": "gpt-x", "stream": stream, **PATHS["/v1/chat/completions"]})
            self.assertIn(f'"model": "{ALIAS}"', raw)
            self.assertNotIn("gpt-x", raw)

    def test_the_control_routes_still_check_the_name(self):
        self.assertEqual(self.post("/v1/load", {"model": "gpt-x"})[0], 404)


if __name__ == "__main__":
    unittest.main()
