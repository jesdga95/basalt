"""logit_bias on Chat Completions: parsing, validation before streaming, and what reaches the engine."""
from __future__ import annotations

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.api.logit_bias import REQUEST_KEY, checked, engine_key, normalize  # noqa: E402
from serve.engine.mock import MockEngine  # noqa: E402
from serve.engine.process import ProcessEngine  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.handler import serve  # noqa: E402
from serve.service import Service  # noqa: E402
from serve.util.text import ByteTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


class Parsing(unittest.TestCase):
    def test_a_map_of_ids_to_biases(self):
        self.assertEqual(normalize({"7": -100, "9": 2.5, 11: 1}, 20), {7: -100.0, 9: 2.5, 11: 1.0})
        for empty in (None, {}):
            self.assertEqual(normalize(empty, 20), {})

    def test_bad_values_are_named(self):
        bad = [True, 1, "x", [[1, 2]], {"-1": 0}, {"1.5": 0}, {"a": 0}, {"20": 0}, {"1": 101}, {"1": -101},
               {"1": True}, {"1": "5"}, {"1": None}, {"1": float("nan")}, {"1": float("inf")}, {"1": 10 ** 400},
               {1: 0, "1": 3}, {"99999999999": 0}]
        for value in bad:
            with self.subTest(value=str(value)[:40]), self.assertRaisesRegex(ValueError, "logit_bias"):
                normalize(value, 20)

    def test_banning_the_whole_vocabulary_is_refused(self):
        with self.assertRaisesRegex(ValueError, "every token"):
            normalize({str(i): -100 for i in range(10)}, 10)
        self.assertEqual(len(normalize({str(i): -100 for i in range(9)}, 10)), 9)

    def test_engine_key(self):
        self.assertEqual(engine_key(normalize({"9": 2.5, "7": -100}, 20)), " logit_bias=7:-100,9:2.5")
        self.assertEqual(engine_key({}), "")
        self.assertEqual(engine_key(None), "")
        self.assertEqual(engine_key({"7": -100.0}), "")           # a body's own string keys are never forwarded
        self.assertEqual(ProcessEngine.sampling_keys({REQUEST_KEY: {7: -100.0, 3: 100.0}, "temperature": 0.5}),
                         " temperature=0.5 logit_bias=3:100,7:-100")
        self.assertEqual(ProcessEngine.sampling_keys({"logit_bias": {"7": -100}}), "")

    def test_engine_support(self):
        req = {"logit_bias": {"5": 1}}
        started = SimpleNamespace(info={"context": 4096, "logit_bias": 10})
        self.assertEqual(checked(req, started), {5: 1.0})
        with self.assertRaisesRegex(ValueError, "outside the model's vocabulary"):
            checked({"logit_bias": {"10": 1}}, started)
        with self.assertRaisesRegex(ValueError, "cannot apply"):
            checked(req, SimpleNamespace(info={"context": 4096}))
        with self.assertRaisesRegex(ValueError, "parallel"):
            checked(req, SimpleNamespace(info={"context": 4096, "logit_bias": 10}, batch=2))
        self.assertEqual(checked(req, SimpleNamespace(info={"version": None})), {5: 1.0})   # not started: not judged
        self.assertEqual(checked(req, SimpleNamespace()), {5: 1.0})
        self.assertEqual(checked({}, SimpleNamespace(info={"context": 1}, batch=2)), {})


class Http(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = MockEngine(tok, "ok", max_context=4096)
        self.httpd = serve(Service(self.engine, tok, ChatTemplate.from_file(ROOT / "serve/chat_template.jinja")), port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, **extra):
        body = {"model": "basalt", "max_tokens": 4, "messages": [{"role": "user", "content": "hi"}], **extra}
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read().decode()

    def test_a_valid_map_reaches_the_engine(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.engine.requests.clear()
                code, _ = self.post("/v1/chat/completions", logit_bias={"7": -100, "9": 2.5}, stream=stream)
                self.assertEqual(code, 200)
                self.assertEqual(self.engine.requests[0][1][REQUEST_KEY], {7: -100.0, 9: 2.5})

    def test_no_bias_sends_none(self):
        for extra in ({}, {"logit_bias": {}}, {"logit_bias": None}):
            with self.subTest(extra=extra):
                self.engine.requests.clear()
                self.assertEqual(self.post("/v1/chat/completions", **extra)[0], 200)
                self.assertFalse(self.engine.requests[0][1][REQUEST_KEY])

    def test_a_bad_map_is_a_400_before_the_stream(self):
        for bad in (True, {"x": 1}, {"1": 500}, [[1, 2]]):
            for stream in (False, True):
                with self.subTest(bad=bad, stream=stream):
                    code, body = self.post("/v1/chat/completions", logit_bias=bad, stream=stream)
                    self.assertEqual(code, 400, body)
                    self.assertIn("logit_bias", json.loads(body)["error"]["message"])
        self.assertEqual(self.engine.requests, [])

    def test_other_apis_do_not_forward_it(self):
        self.engine.requests.clear()
        body = {"logit_bias": {"7": -100}, "_logit_bias": {7: -100.0}}
        self.assertEqual(self.post("/v1/messages", **body)[0], 200)
        self.assertEqual(self.post("/v1/responses", input="hi", **body)[0], 200)
        self.assertTrue(self.engine.requests)
        for _, sampling in self.engine.requests:
            self.assertNotIn("logit_bias", ProcessEngine.sampling_keys(sampling))


if __name__ == "__main__":
    unittest.main()
