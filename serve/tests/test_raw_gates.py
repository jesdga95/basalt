"""--more-samplers and --gbnf-grammar are off unless the server is started with them on: a raw request that sets tfs,
typical, top_a, xtc, dynatemp or smoothing, or that carries a grammar, gets a 400 naming the flag; the same request
is served when the flag is on. The values that switch nothing (tfs 1, top_a 0...) pass either way."""
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.config import build_parser, more_samplers_on
from serve.frontend import ChatTemplate
from serve.handler import serve
from serve.service import Service
from serve.tests.test_lifecycle import ResidentEngine
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]


class Gates(unittest.TestCase):
    def start(self, **options):
        tok = ByteTokenizer()
        self.engine = ResidentEngine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate.from_file(ROOT / "serve/chat_template.jinja"), **options)
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.engine.close()

    def post(self, path, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status, r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8")

    def completions(self, **body):
        return self.post("/v1/completions", {"prompt": "Once upon a", "max_tokens": 8, **body})

    def kobold(self, **body):
        return self.post("/api/v1/generate", {"prompt": "Once upon a", "max_length": 8, **body})

    def chat(self, **body):
        return self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8, **body})


class MoreSamplersOff(Gates):
    def setUp(self):
        self.start()                                   # the defaults: both flags off

    def test_a_sampler_any_raw_api_sets_is_a_400_naming_the_flag(self):
        for send in (self.completions, self.kobold):
            for body in ({"tfs": 0.9}, {"typical": 0.8}, {"top_a": 0.2}, {"xtc_probability": 0.5},
                         {"dynatemp_min": 0.5, "dynatemp_max": 1.5}, {"smoothing_factor": 0.3}):
                with self.subTest(api=send.__name__, body=body):
                    status, text = send(**body)
                    self.assertEqual(status, 400, text)
                    self.assertIn("--more-samplers on", text)

    def test_a_sampler_the_chat_api_sets_is_a_400_too(self):
        status, text = self.chat(tfs=0.9)
        self.assertEqual(status, 400, text)
        self.assertIn("--more-samplers on", text)

    def test_values_that_switch_nothing_pass(self):
        for body in ({"tfs": 1.0, "typical": 1.0, "top_a": 0.0, "xtc_probability": 0.0, "smoothing_factor": 0.0},
                     {"dynatemp_min": 1.0, "dynatemp_max": 1.0}):
            with self.subTest(body=body):
                self.assertEqual(self.completions(**body)[0], 200)
                self.assertEqual(self.kobold(**body)[0], 200)
                self.assertEqual(self.chat(**body)[0], 200)

    def test_a_plain_request_is_served(self):
        self.assertEqual(self.completions(temperature=0.7, top_k=20)[0], 200)
        self.assertEqual(self.kobold(rep_pen=1.1, temperature=0.7)[0], 200)

    def test_a_grammar_is_a_400_naming_the_flag(self):
        for send in (self.completions, self.kobold):
            for body in ({"grammar": 'root ::= "yes"'}, {"grammar_schema": {"type": "object"}}):
                with self.subTest(api=send.__name__, body=body):
                    status, text = send(**body)
                    self.assertEqual(status, 400, text)
                    self.assertIn("--gbnf-grammar on", text)


class MoreSamplersOn(Gates):
    def setUp(self):
        self.start(more_samplers=True, gbnf_grammar=True)

    def test_the_samplers_are_served(self):
        for send in (self.completions, self.kobold):
            status, text = send(tfs=0.9, typical=0.8, top_a=0.2, top_k=20, temperature=0.7)
            self.assertEqual(status, 200, text)
        self.assertEqual(self.chat(tfs=0.9)[0], 200)

    def test_a_grammar_is_taken(self):
        status, text = self.completions(grammar='root ::= "yes"')
        self.assertEqual(status, 200, text)
        status, text = self.kobold(grammar='root ::= "yes"')
        self.assertEqual(status, 200, text)
        status, text = self.completions(grammar_schema='{"type": "object"}')
        self.assertEqual(status, 200, text)

    def test_a_bad_grammar_field_is_a_400(self):
        for body in ({"grammar": "  "}, {"grammar": 5}, {"grammar_schema": "{"}, {"grammar_schema": [1]}):
            with self.subTest(body=body):
                status, text = self.completions(**body)
                self.assertEqual(status, 400, text)


class Options(unittest.TestCase):
    def test_both_are_off_unless_given(self):
        a = build_parser().parse_args([])
        self.assertEqual((a.more_samplers, a.gbnf_grammar), (None, None))
        a = build_parser().parse_args(["--more-samplers", "on", "--gbnf-grammar", "on"])
        self.assertEqual((a.more_samplers, a.gbnf_grammar), ("on", "on"))

    def test_which_samplers_a_request_switches_on(self):
        self.assertEqual(more_samplers_on({}), [])
        self.assertEqual(more_samplers_on({"tfs": 1.0, "top_a": 0.0, "dynatemp_min": 1.0, "dynatemp_max": 1.0}), [])
        self.assertEqual(more_samplers_on({"tfs": 0.9, "top_a": 0.0, "smoothing_factor": 0.2}), ["tfs", "smoothing_factor"])
        self.assertEqual(more_samplers_on({"dynatemp_min": 0.5, "dynatemp_max": 1.5}), ["dynatemp_min"])


if __name__ == "__main__":
    unittest.main()
