"""JSON response formats through HTTP: the request is checked, the engine gets the grammar, the text passes as is."""
import json
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.engine.mock import MockEngine
from serve.handler import serve
from serve.service import Service
from serve.util.text import ByteTokenizer


STORY = {"title": "Photobooth", "sheet": "A single continuous shot.", "cast": [],
         "boxes": [{"box": 1, "duration_ms": 15000, "new_shot": True, "beat": "Hailee changes poses.",
                    "on_screen": ["Hailee"], "references": [], "ends": "She smiles."}], "suggestions": []}


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


TEXT = {"type": "string"}
STORY_SCHEMA = object_schema({
    "title": TEXT, "sheet": TEXT, "cast": {"type": "array", "items": object_schema({"name": TEXT,
        "references": {"type": "array", "items": TEXT, "maxItems": 0}})},
    "boxes": {"type": "array", "minItems": 1, "maxItems": 1, "items": object_schema({
        "box": {"type": "integer", "minimum": 1, "maximum": 1},
        "duration_ms": {"type": "integer", "minimum": 1, "maximum": 15000}, "new_shot": {"type": "boolean"},
        "beat": TEXT, "on_screen": {"type": "array", "items": TEXT},
        "references": {"type": "array", "items": TEXT, "maxItems": 0}, "ends": TEXT})},
    "suggestions": {"type": "array", "items": object_schema({"what": TEXT, "why": TEXT,
        "boxes": {"type": "array", "items": {"type": "integer"}}, "description": TEXT})}})
FORMAT = {"type": "json_schema", "json_schema": {"name": "h3_story", "strict": True, "schema": STORY_SCHEMA}}


class Structured(unittest.TestCase):
    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read().decode()
            return response.status, data if "event-stream" in response.headers.get("Content-Type", "") else json.loads(data)

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, json.dumps(STORY), max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).resolve().parents[1] / "chat_template.jinja"),
                           grammar=True)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def chat(self, script=None, **kwargs):
        if script is not None:
            self.engine = self.svc.engine = MockEngine(self.tok, script, max_context=16384)
        req = {"model": "qwen3.8-flash-next", "messages": [{"role": "system", "content": "Write a film plan."},
                            {"role": "user", "content": "Photobooth. 15 seconds."}],
               "response_format": FORMAT, "reasoning_effort": "none", **kwargs}
        return self.request("/v1/chat/completions", req)

    def test_schema_request_carries_the_grammar_and_no_prompt_directive(self):
        code, reply = self.chat()
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(reply["choices"][0]["message"]["content"]), STORY)
        self.assertEqual(self.engine.requests[-1][1]["_grammar"], {"kind": "json_schema", "schema": STORY_SCHEMA})
        self.assertNotIn("OUTPUT FORMAT REQUIREMENT", self.tok.decode(self.engine.last_prompt))

    def test_stream_with_reasoning(self):
        code, raw = self.chat("Consider the shot.</think>\n" + json.dumps(STORY), stream=True, reasoning_effort="low")
        self.assertEqual(code, 200)
        chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        self.assertEqual(json.loads(content), STORY)
        self.assertIn("reasoning_content", raw)
        self.assertIn("data: [DONE]", raw)

    def test_the_answer_is_not_checked_after_the_fact(self):
        for script in ("## Film Plan - Photobooth", '{"title":"Only a title"}'):
            code, reply = self.chat(script)
            self.assertEqual(code, 200)
            self.assertEqual(reply["choices"][0]["message"]["content"], script)

    def test_object_mode(self):
        code, reply = self.chat('{"answer":4}', response_format={"type": "json_object"})
        self.assertEqual(code, 200)
        self.assertEqual(self.engine.requests[-1][1]["_grammar"], {"kind": "json_object"})
        # Plain text callers get no grammar.
        self.assertEqual(self.chat("Prose.", response_format={"type": "text"})[0], 200)
        self.assertIsNone(self.engine.requests[-1][1]["_grammar"])

    def test_rejected_schema_does_not_load_the_engine(self):
        with mock.patch.object(self.svc, "load") as load:
            code, _ = self.chat(response_format={"type": "json_schema"})
            self.assertEqual(code, 400)
            load.assert_not_called()

    def test_remote_refs_are_400_and_local_refs_work(self):
        remote = {"type": "object", "$ref": "https://other.example/schema"}
        fmt = {"type": "json_schema", "json_schema": {"name": "test", "strict": True, "schema": remote}}
        self.assertEqual(self.chat(response_format=fmt)[0], 400)
        schema = object_schema({"answer": {"$ref": "#/$defs/n"}})
        schema["$defs"] = {"n": {"type": "integer"}}
        fmt = {"type": "json_schema", "json_schema": {"name": "local", "schema": schema}}
        self.assertEqual(self.chat('{"answer":4}', response_format=fmt)[0], 200)

    def test_malformed_formats_are_rejected_before_loading(self):
        formats = ([], {"type": "unsupported"},
                   {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
                   {"type": "json_schema", "json_schema": {"name": "x", "strict": "true", "schema": {"type": "object"}}})
        with mock.patch.object(self.svc, "load") as load:
            for fmt in formats:
                with self.subTest(fmt=fmt):
                    self.assertEqual(self.chat(response_format=fmt)[0], 400)
            self.assertEqual(self.request("/v1/chat/completions", [])[0], 400)
            load.assert_not_called()

    def test_no_grammar_option_leaves_the_key_out(self):
        self.svc.grammar = False
        self.chat()
        self.assertEqual(self.engine.requests[-1][1]["_grammar"], {"kind": "json_schema", "schema": STORY_SCHEMA})  # strict: true
        n = len(self.engine.requests)
        status, body = self.chat(response_format={"type": "json_object"})
        self.assertEqual(status, 400)
        self.assertIn("need grammar enabled", str(body))
        self.assertEqual(len(self.engine.requests), n)    # refused before the engine


if __name__ == "__main__":
    unittest.main()
