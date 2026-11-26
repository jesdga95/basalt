"""Frontend request validation, Anthropic tool-result ordering, and enum schema decoding parity."""
from __future__ import annotations

import http.client
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from serve.engine.mock import MockEngine  # noqa: E402
from serve.frontend import (ChatTemplate, OutputParser, _decode_param, _text_kinds,  # noqa: E402
                            anthropic_to_messages, openai_to_messages, parse_tool_call)
from serve.handler import serve  # noqa: E402
from serve.service import Service  # noqa: E402
from serve.util.text import ByteTokenizer  # noqa: E402

TEMPLATE = ChatTemplate.from_file(ROOT / "serve/chat_template.jinja")


class EnumDecodingParity(unittest.TestCase):
    """_text_kinds and _decode_param match C++ schema.cpp for string and mixed/nullable enums."""

    def test_all_string_enum_without_type_stays_string(self):
        schema = {"enum": ["true", "false", "123", "null"]}
        self.assertEqual(_text_kinds(schema), (True, False))
        for raw in ("true", "false", "123", "null"):
            self.assertEqual(_decode_param(raw, schema), raw)

    def test_nullable_or_mixed_enum_is_json_decoded(self):
        nullable_enum = {"type": ["string", "null"], "enum": ["fast", "slow", None]}
        self.assertEqual(_text_kinds(nullable_enum), (False, False))
        self.assertEqual(_decode_param('"fast"', nullable_enum), "fast")
        self.assertIsNone(_decode_param("null", nullable_enum))

        mixed_enum = {"enum": ["fast", 1, True]}
        self.assertEqual(_text_kinds(mixed_enum), (False, False))
        self.assertEqual(_decode_param('"fast"', mixed_enum), "fast")
        self.assertEqual(_decode_param("1", mixed_enum), 1)
        self.assertIs(_decode_param("true", mixed_enum), True)

    def test_empty_enum_falls_through_to_type(self):
        self.assertEqual(_text_kinds({"type": "string", "enum": []}), (True, False))
        self.assertEqual(_text_kinds({"enum": []}), (False, False))

    def test_parse_and_stream_tool_calls_with_enums(self):
        tool = {"name": "cfg", "parameters": {"type": "object", "properties": {
            "flag": {"enum": ["true", "false"]},
            "mode": {"type": ["string", "null"], "enum": ["fast", "slow", None]},
        }}}
        body = ('<function=cfg>\n<parameter=flag>\ntrue\n</parameter>\n'
                '<parameter=mode>\n"fast"\n</parameter>\n</function>')
        self.assertEqual(parse_tool_call(body, tool).arguments, {"flag": "true", "mode": "fast"})
        full = f"<tool_call>\n{body}\n</tool_call>"
        for step in (1, 7, len(full)):
            p = OutputParser(thinking=False, tools=[tool], stream_tools=True)
            evts = []
            for i in range(0, len(full), step):
                evts.extend(p.feed(full[i:i + step]))
            evts.extend(p.finish())
            streamed = "".join(e.text for e in evts if e.kind == "tool_args")
            self.assertEqual(json.loads(streamed), {"flag": "true", "mode": "fast"})


class AnthropicToolResultOrdering(unittest.TestCase):
    """anthropic_to_messages preserves call IDs and reorders parallel tool_result blocks."""

    def test_reversed_tool_results_are_reordered_to_match_tool_use_ids(self):
        req = {"messages": [
            {"role": "user", "content": "Read both files."},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Reading."},
                {"type": "tool_use", "id": "tu_a", "name": "read", "input": {"path": "a.txt"}},
                {"type": "tool_use", "id": "tu_b", "name": "read", "input": {"path": "b.txt"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_b", "content": "CONTENT_B"},
                {"type": "tool_result", "tool_use_id": "tu_a", "content": "CONTENT_A"},
                {"type": "text", "text": "Now summarize."},
            ]},
        ]}
        msgs, _, _ = anthropic_to_messages(req)
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "tool", "tool", "user"])
        self.assertEqual([msgs[2]["content"], msgs[3]["content"]], ["CONTENT_A", "CONTENT_B"])
        self.assertEqual(msgs[4]["content"], "Now summarize.")
        # Internal _call_id markers must be stripped
        for m in msgs:
            self.assertNotIn("_call_id", m)
            for c in m.get("tool_calls") or []:
                self.assertNotIn("_call_id", c)

    def test_missing_or_ambiguous_ids_keep_received_order(self):
        req = {"messages": [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_a", "name": "read", "input": {}},
                {"type": "tool_use", "name": "read", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_b", "content": "FIRST"},
                {"type": "tool_result", "tool_use_id": "tu_a", "content": "SECOND"},
            ]},
        ]}
        msgs, _, _ = anthropic_to_messages(req)
        self.assertEqual([m["content"] for m in msgs if m["role"] == "tool"], ["FIRST", "SECOND"])


class MalformedInputValidation(unittest.TestCase):
    """Malformed message/tool shapes raise ValueError (HTTP 400) or normalize cleanly for Jinja."""

    def test_openai_malformed_inputs_raise_value_error(self):
        bad_reqs = [
            {"messages": [{"content": "missing role"}]},
            {"messages": [{"role": None, "content": "null role"}]},
            {"messages": [{"role": 123, "content": "int role"}]},
            {"messages": [{"role": "user", "content": 12345}]},
            {"messages": [{"role": "user", "content": [{"type": "text", "text": 42}]}]},
            {"messages": [{"role": "assistant", "content": "hi", "reasoning_content": 99}]},
            {"messages": [{"role": "user", "content": "hi"}], "chat_template_kwargs": "not-a-dict"},
            {"messages": [{"role": "user", "content": "hi"}], "chat_template_kwargs": [1, 2]},
            {"messages": [{"role": "user", "content": "hi"}], "chat_template_kwargs": 42},
            {"messages": [{"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": 123, "arguments": "{}"}}]}]},
        ]
        for req in bad_reqs:
            with self.subTest(req=req), self.assertRaises(ValueError):
                openai_to_messages(req)

    def test_anthropic_malformed_inputs_raise_value_error(self):
        bad_reqs = [
            {"messages": [{"content": "missing role"}]},
            {"messages": [{"role": "", "content": "empty role"}]},
            {"messages": [{"role": "user", "content": 123}]},
            {"messages": [{"role": "user", "content": ["not-a-dict"]}]},
            {"messages": [{"role": "user", "content": [123]}]},
            {"messages": [{"role": "assistant", "content": [{"type": "thinking", "thinking": 123}]}]},
            {"messages": [{"role": "assistant", "content": [{"type": "tool_use", "name": None, "input": {}}]}]},
            {"system": 123, "messages": [{"role": "user", "content": "hi"}]},
        ]
        for req in bad_reqs:
            with self.subTest(req=req), self.assertRaises(ValueError):
                anthropic_to_messages(req)

    def test_anthropic_null_thinking_and_non_dict_tool_args_normalize_and_render(self):
        # thinking: null must not crash with TypeError in "".join(reasoning)
        msgs, _, _ = anthropic_to_messages({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": None},
                {"type": "tool_use", "id": "t1", "name": "f", "input": [1, 2, 3]},
            ]},
        ]})
        self.assertEqual(msgs[1]["tool_calls"][0]["function"]["arguments"], {"arguments": [1, 2, 3]})
        rendered = TEMPLATE.render(msgs, enable_thinking=False)
        self.assertIn("<parameter=arguments>", rendered)

        # OpenAI non-dict tool_calls arguments (JSON array/scalar or raw list/int) normalize to {"arguments": value}
        for raw_args, expected in (("[1, 2]", {"arguments": [1, 2]}), ("123", {"arguments": 123}),
                                   ([1, 2], {"arguments": [1, 2]}), (False, {"arguments": False}),
                                   (None, {}), ("", {})):
            with self.subTest(raw_args=raw_args):
                omsgs, _, _ = openai_to_messages({"messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": raw_args}}]},
                ]})
                self.assertEqual(omsgs[1]["tool_calls"][0]["function"]["arguments"], expected)
                TEMPLATE.render(omsgs, enable_thinking=False)

    def test_http_routes_return_400_for_malformed_requests(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=4096), tok, TEMPLATE)
        httpd = serve(svc, port=0)
        port = httpd.server_address[1]
        try:
            cases = [
                ("/v1/chat/completions", {"model": "basalt", "messages": [{"content": "no role"}]}),
                ("/v1/chat/completions", {"model": "basalt", "messages": [{"role": "user", "content": 7}]}),
                ("/v1/chat/completions", {"model": "basalt", "messages": [{"role": "user", "content": "hi"}],
                                          "chat_template_kwargs": "bad"}),
                ("/v1/messages", {"model": "basalt", "messages": [{"content": "no role"}]}),
                ("/v1/messages", {"model": "basalt", "messages": [{"role": "user", "content": ["bad"]}]}),
            ]
            for path, body in cases:
                with self.subTest(path=path, body=body):
                    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                    try:
                        c.request("POST", path, body=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"})
                        r = c.getresponse()
                        raw = r.read().decode()
                        self.assertEqual(r.status, 400, raw)
                    finally:
                        c.close()
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
