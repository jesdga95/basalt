"""The `grammar` key of the engine request and the `grammar` server option, against the mock engine."""
from __future__ import annotations

import json
import socket
import sys
import tempfile
import types
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.api.grammar import engine_key  # noqa: E402
from serve.config import GRAMMAR_DEFAULT, build_parser, grammar_words  # noqa: E402
from serve.engine.mock import MockEngine  # noqa: E402
from serve.engine.process import ProcessEngine  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.handler import serve  # noqa: E402
from serve.service import Service  # noqa: E402
from serve.util.text import ByteTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FN = {"name": "get_weather", "description": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}
TOOL = {"type": "function", "function": FN}
STRICT_TOOL = {"type": "function", "function": {**FN, "strict": True}}
SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}}


class Fixture:
    grammar = False

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = MockEngine(tok, "</think>\n\n{\"a\": 1}", max_context=4096)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), grammar=cls.grammar)
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def sent(self, path, body):
        """The grammar value the engine received for one request (None: no key)."""
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
        return self.engine.requests[-1][1].get("_grammar")

    def refused(self, path, body):
        """(status, error body) of a request the server answers with an HTTP error."""
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=30)
        return cm.exception.code, json.loads(cm.exception.read())["error"]

    def chat(self, **extra):
        return self.sent("/v1/chat/completions", {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": "hi"}], **extra})

    def json_schema(self, **spec):
        return {"type": "json_schema", "json_schema": {"name": "s", "schema": SCHEMA, **spec}}


class GrammarOff(Fixture, unittest.TestCase):
    def test_default_is_the_named_constant(self):
        self.assertEqual(build_parser().get_default("grammar"), GRAMMAR_DEFAULT)
        self.assertTrue(GRAMMAR_DEFAULT)

    def test_no_key_without_strict(self):
        self.assertIsNone(self.chat(tools=[TOOL]))
        self.assertIsNone(self.chat(response_format={"type": "text"}))

    def test_chat_json_format_is_refused(self):
        msgs = [{"role": "user", "content": "hi"}]
        for fmt in (self.json_schema(), {"type": "json_object"}):
            code, err = self.refused("/v1/chat/completions", {"model": "qwen3.8-flash-next", "messages": msgs, "response_format": fmt})
            self.assertEqual(code, 400)
            self.assertEqual(err["type"], "invalid_request_error")
            self.assertIn("need grammar enabled", err["message"])

    def test_responses_json_format_is_refused(self):
        for fmt in ({"type": "json_schema", "name": "s", "schema": SCHEMA}, {"type": "json_object"}):
            code, err = self.refused("/v1/responses", {"model": "qwen3.8-flash-next", "input": "hi", "text": {"format": fmt}})
            self.assertEqual(code, 400)
            self.assertEqual(err["param"], "text.format")
            self.assertIn("text.format json_schema and json_object (structured output) need grammar enabled", err["message"])

    def test_anthropic_json_format_is_refused(self):
        code, err = self.refused("/v1/messages", {"model": "qwen3.8-flash-next", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}],
                                                  "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}}})
        self.assertEqual(code, 400)
        self.assertEqual(err["type"], "invalid_request_error")
        self.assertIn("output_config.format json_schema", err["message"])
        self.assertIn("need grammar enabled", err["message"])

    def test_anthropic_output_format_is_refused(self):
        code, _ = self.refused("/v1/messages", {"model": "qwen3.8-flash-next", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}],
                                                "output_format": {"type": "json_schema", "schema": SCHEMA}})
        self.assertEqual(code, 400)

    def test_responses_text_format_is_allowed(self):
        self.assertIsNone(self.sent("/v1/responses", {"model": "qwen3.8-flash-next", "input": "hi", "text": {"format": {"type": "text"}}}))

    def test_strict_json_schema(self):
        self.assertEqual(self.chat(response_format=self.json_schema(strict=True)),
                         {"kind": "json_schema", "schema": SCHEMA})

    def test_strict_tool(self):
        g = self.chat(tools=[STRICT_TOOL], tool_choice="required")
        self.assertEqual(g, {"kind": "tools", "tools": [STRICT_TOOL], "tool_choice": "required"})

    def test_anthropic_strict_tool(self):
        g = self.sent("/v1/messages", {"model": "qwen3.8-flash-next", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}],
                                       "tools": [{"name": "get_weather", "strict": True,
                                                  "input_schema": FN["parameters"]}]})
        self.assertEqual(g["kind"], "tools")
        self.assertEqual(g["tools"][0]["function"]["name"], "get_weather")

    def test_anthropic_tools_without_strict(self):
        self.assertIsNone(self.sent("/v1/messages", {"model": "qwen3.8-flash-next", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}],
                                                     "tools": [{"name": "t", "input_schema": {"type": "object"}}]}))

    def test_responses_strict_tool(self):
        g = self.sent("/v1/responses", {"model": "qwen3.8-flash-next", "input": "hi", "tools": [{"type": "function", "name": "get_weather", "strict": True,
                                                                   "parameters": FN["parameters"]}]})
        self.assertEqual(g["kind"], "tools")
        self.assertEqual(g["tools"][0]["function"]["name"], "get_weather")

    def test_responses_strict_schema(self):
        g = self.sent("/v1/responses", {"model": "qwen3.8-flash-next", "input": "hi", "text": {"format": {"type": "json_schema", "name": "s",
                                                                            "schema": SCHEMA, "strict": True}}})
        self.assertEqual(g, {"kind": "json_schema", "schema": SCHEMA})


class GrammarOn(Fixture, unittest.TestCase):
    grammar = True

    def test_tools(self):
        g = self.chat(tools=[TOOL])
        self.assertEqual(g, {"kind": "tools", "tools": [TOOL], "tool_choice": "auto"})

    def test_tool_choice_none_offers_no_tools(self):
        self.assertIsNone(self.chat(tools=[TOOL], tool_choice="none"))

    def test_json_object(self):
        self.assertEqual(self.chat(response_format={"type": "json_object"}), {"kind": "json_object"})

    def test_json_schema(self):
        self.assertEqual(self.chat(response_format=self.json_schema()), {"kind": "json_schema", "schema": SCHEMA})

    def test_plain_chat_has_no_key(self):
        self.assertIsNone(self.chat())
        self.assertIsNone(self.chat(response_format={"type": "text"}))

    def test_anthropic_tools(self):
        g = self.sent("/v1/messages", {"model": "qwen3.8-flash-next", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}],
                                       "tools": [{"name": "get_weather", "description": "weather",
                                                  "input_schema": FN["parameters"]}], "tool_choice": {"type": "any"}})
        self.assertEqual(g, {"kind": "tools", "tools": [TOOL], "tool_choice": {"type": "any"}})

    def test_responses_tools(self):
        g = self.sent("/v1/responses", {"model": "qwen3.8-flash-next", "input": "hi", "tools": [{"type": "function", "name": "get_weather",
                                                                   "description": "weather",
                                                                   "parameters": FN["parameters"]}]})
        self.assertEqual(g["kind"], "tools")
        self.assertEqual(g["tools"], [TOOL])

    def test_responses_json_object(self):
        self.assertEqual(self.sent("/v1/responses", {"model": "qwen3.8-flash-next", "input": "hi", "text": {"format": {"type": "json_object"}}}),
                         {"kind": "json_object"})

    def anthropic(self, **extra):
        return {"model": "qwen3.8-flash-next", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}], **extra}

    def test_anthropic_json_schema(self):
        want = {"kind": "json_schema", "schema": SCHEMA}
        fmt = {"type": "json_schema", "schema": SCHEMA}
        for stream in (False, True):
            self.assertEqual(self.sent("/v1/messages", self.anthropic(stream=stream, output_config={"format": fmt})), want)
        self.assertEqual(self.sent("/v1/messages", self.anthropic(output_config={"effort": "low", "format": fmt})), want)

    def test_anthropic_output_format_alias(self):
        fmt = {"type": "json_schema", "schema": SCHEMA}
        self.assertEqual(self.sent("/v1/messages", self.anthropic(output_format=fmt)), {"kind": "json_schema", "schema": SCHEMA})

    def test_anthropic_plain_has_no_key(self):
        self.assertIsNone(self.sent("/v1/messages", self.anthropic()))
        self.assertIsNone(self.sent("/v1/messages", self.anthropic(stream=True, output_config={"effort": "low"})))

    def test_anthropic_bad_format(self):
        for body in ({"output_config": {"format": {"type": "json_schema"}}},
                     {"output_config": {"format": {"type": "text"}}},
                     {"output_format": {"type": "json_schema", "schema": {"type": "string"}}},
                     {"output_config": {"format": {"type": "json_schema", "schema": SCHEMA}}, "tools": [
                         {"name": "t", "input_schema": {"type": "object"}}]}):
            code, err = self.refused("/v1/messages", self.anthropic(**body))
            self.assertEqual((code, err["type"]), (400, "invalid_request_error"))


class EngineWithoutTheFlag(unittest.TestCase):
    def test_no_key_sent(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), grammar=True,
                      grammar_engine=False)
        self.assertIsNone(svc.grammar_for([TOOL], None, [STRICT_TOOL], None))


class Wire(unittest.TestCase):
    def test_key_is_one_word(self):
        keys = ProcessEngine.sampling_keys({"_grammar": {"kind": "json_schema", "schema": {"description": "a b\n"}}})
        self.assertEqual(len(keys.split()), 1)
        self.assertEqual(json.loads(keys.split("grammar=")[1]),
                         {"kind": "json_schema", "schema": {"description": "a b\n"}})

    def test_no_spec_no_key(self):
        self.assertEqual(engine_key(None), "")
        self.assertNotIn("grammar", ProcessEngine.sampling_keys({"_grammar": None}))

    def test_engine_words(self):
        self.assertEqual(grammar_words({"--kv"}), [])           # an engine without the flag
        self.assertEqual(grammar_words(None), [])
        self.assertEqual(grammar_words({"--grammar"}), ["--grammar", "on"])

    def test_off_forms(self):
        from serve.config import load_config_file, parse_settings
        ap = build_parser()
        self.assertTrue(parse_settings(ap, ["--grammar"])[0].grammar)
        self.assertFalse(parse_settings(ap, ["--no-grammar"])[0].grammar)
        with tempfile.TemporaryDirectory() as d:
            for value, cli, want in ((True, [], True), (False, [], False), (True, ["--no-grammar"], False),
                                     (False, ["--grammar"], True)):
                p = Path(d) / "c.json"
                p.write_text(json.dumps({"server": {"grammar": value}}))
                self.assertEqual(parse_settings(build_parser(), ["--config", str(p)] + cli)[0].grammar, want)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


ONE_CARD = {"devices": "0", "cards": [{"smi": 0}], "notes": [], "warning": "", "error": ""}


class EngineStart(unittest.TestCase):
    """What main() hands the engine when `grammar` comes from the config file or the command line."""

    def engine_args(self, server_block, cli=(), flags=b"\0--grammar\0--kv\0"):
        from unittest import mock
        from serve import server
        from serve.engine.errors import EngineStartError
        seen = {}

        def fake(binary, args, **kw):
            seen["args"] = list(args)
            raise EngineStartError("stopped for the test", None)
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d) / "eng"
            eng.write_bytes(flags)
            conf = Path(d) / "c.json"
            conf.write_text(json.dumps({"server": {"model": "m.basalt", **server_block}}))
            argv = ["x", "--config", str(conf), "--engine-bin", str(eng), "--port", str(free_port()), *cli]
            tokdir = Path(d) / "tok"
            tokdir.mkdir()
            (tokdir / "vocab.json").write_text("{}")
            (tokdir / "merges.txt").write_text("")
            (tokdir / "token_type.json").write_text("[]")
            cfg = {"engine_bin": str(eng), "args": ["--serve"], "log": None, "tokenizer": str(tokdir)}
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch("serve.engine.modelfile.model_config", return_value=cfg), \
                    mock.patch.dict(sys.modules, {"tokenizer": types.SimpleNamespace(Tokenizer=lambda *a: ByteTokenizer())}), \
                    mock.patch.object(server, "ProcessEngine", fake), \
                    mock.patch.object(server, "gpu_plan", return_value=ONE_CARD):
                server.main()
        return seen["args"]

    def test_config_file(self):
        for block in ({"grammar": True}, {"grammar": False}, {}):
            self.assertEqual(self.engine_args(block), ["--serve", "--gpu", "0", "--grammar", "on"], block)

    def line(self, block, cli=()):
        import io
        from contextlib import redirect_stdout
        out = io.StringIO()
        with redirect_stdout(out):
            self.engine_args(block, cli)
        return next(x for x in out.getvalue().splitlines() if x.startswith("[server] engine "))

    def test_startup_line(self):
        self.assertEqual(self.line({}), "[server] engine   defaults")
        self.assertEqual(self.line({"grammar": True}), "[server] engine   grammar=on (from config)")
        self.assertEqual(self.line({"grammar": True}, ["--no-grammar"]), "[server] engine   grammar=off (from cli)")
        self.assertEqual(self.line({}, ["--grammar"]), "[server] engine   grammar=on (from cli)")

    def test_command_line(self):
        self.assertEqual(self.engine_args({}, ["--grammar"]), ["--serve", "--gpu", "0", "--grammar", "on"])

    def test_engine_without_the_flag(self):
        self.assertEqual(self.engine_args({"grammar": True}, flags=b"\0--kv\0"), ["--serve", "--gpu", "0"])


if __name__ == "__main__":
    unittest.main()
