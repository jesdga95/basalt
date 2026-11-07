"""A `<tool_call>` inside a code fence (``` or ~~~) or inline code in the visible answer is text, not a call."""
from __future__ import annotations

import json
import random
import unittest
import urllib.request
from pathlib import Path

from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate, OutputParser
from serve.handler import serve
from serve.service import Service
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
CTX = 4096
CALL = "<tool_call>\n<function=write>\n<parameter=path>\nfile.txt\n</parameter>\n</function>\n</tool_call>"
RM = CALL.replace("file.txt", "rm -rf build")
PROPS = {"path": {"type": "string"}}
SCHEMA = [{"name": "write", "parameters": {"properties": PROPS}}]
WIDTHS = (1, 2, 3, 7, 0, 100_000)


def parse(text, width, tools=SCHEMA, stream_tools=True):
    """Events for `text` fed `width` characters at a time (0 = random 1-9, seeded)."""
    p = OutputParser(thinking=True, tools=tools, stream_tools=stream_tools)
    evs, rnd, i = [], random.Random(7), 0
    while i < len(text):
        k = width or rnd.randint(1, 9)
        evs += p.feed(text[i:i + k])
        i += k
    return evs + p.finish()


def calls(evs):
    return [e.call.name for e in evs if e.kind == "tool_call"]


class VisibleAnswer(unittest.TestCase):
    HEAD = "ok</think>\n\n"

    def run_all(self, text):
        res = set()
        for width in WIDTHS:
            for st in (True, False):
                evs = parse(text, width, SCHEMA, st)
                res.add((tuple(calls(evs)), "".join(e.text for e in evs if e.kind == "content")))
        self.assertEqual(len(res), 1, res)               # every chunking agrees
        return res.pop()

    def test_quoted_calls_stay_text(self):
        for body in ("Format:\n```xml\n" + RM + "\n```\nDone.", "Format:\n~~~\n" + RM + "\n~~~\nDone.",
                     "Use `" + RM + "` like this.", "Use `x`, then `\n" + RM + "`", "```\n" + RM,
                     "Text\n\n```python\nx = 1\n```\n```\n" + CALL):
            with self.subTest(body=body[:30]):
                got, text = self.run_all(self.HEAD + body)
                self.assertEqual(got, ())
                self.assertEqual(text, body)             # nothing lost

    def test_real_calls_still_fire(self):
        for body in (CALL, "Let me do it.\n\n" + CALL, "Let me do it. " + CALL, "```\ncode\n```\n" + CALL,
                     "`a` and `b` " + CALL, "~~~\nx\n~~~\n\n" + CALL):
            with self.subTest(body=body[:30]):
                self.assertEqual(self.run_all(self.HEAD + body)[0], ("write",))
        self.assertEqual(self.run_all(self.HEAD + CALL + "\n" + CALL)[0], ("write", "write"))

    def test_without_thinking(self):
        p = OutputParser(thinking=False, tools=SCHEMA)
        evs = p.feed("```\n" + RM + "\n```") + p.finish()
        self.assertEqual(calls(evs), [])


class OverHttp(unittest.TestCase):
    """The same rules through the real server: OpenAI chat (whole and streamed), Anthropic and Responses."""

    def ask(self, script, max_tokens=800):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, script, max_context=CTX), tok, ChatTemplate.from_file(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        msgs = [{"role": "user", "content": "save my notes"}]
        reqs = {
            "openai": ("/v1/chat/completions", {"tools": [{"type": "function", "function": {
                "name": "write", "parameters": {"type": "object", "properties": PROPS}}}], "messages": msgs,
                "max_tokens": max_tokens}),
            "anthropic": ("/v1/messages", {"tools": [{"name": "write", "input_schema": {
                "type": "object", "properties": PROPS}}], "messages": msgs, "max_tokens": max_tokens}),
            "responses": ("/v1/responses", {"tools": [{"type": "function", "name": "write", "parameters": {
                "type": "object", "properties": PROPS}}], "input": "save my notes", "max_output_tokens": max_tokens}),
        }
        out = {}
        try:
            for api, (path, body) in reqs.items():
                for stream in (False, True):
                    req = urllib.request.Request(base + path, data=json.dumps(dict(body, model="basalt", stream=stream)).encode(),
                                                 headers={"Content-Type": "application/json",
                                                          "anthropic-version": "2023-06-01"})
                    with urllib.request.urlopen(req, timeout=30) as r:
                        raw = r.read().decode()
                    out[api, stream] = self.tool_uses(api, stream, raw)
        finally:
            httpd.shutdown()
            httpd.server_close()
        return out

    @staticmethod
    def tool_uses(api, stream, raw):
        """How many tool calls the answer carries."""
        if not stream:
            m = json.loads(raw)
            if api == "openai":
                return len(m["choices"][0]["message"].get("tool_calls") or [])
            if api == "anthropic":
                return sum(b["type"] == "tool_use" for b in m["content"])
            return sum(o["type"] == "function_call" for o in m["output"])
        evs = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
        if api == "openai":
            return len({tc["index"] for e in evs for c in e["choices"] for tc in c["delta"].get("tool_calls") or []})
        if api == "anthropic":
            return sum(e["type"] == "content_block_start" and e["content_block"]["type"] == "tool_use" for e in evs)
        return sum(e["type"] == "response.output_item.added" and e["item"]["type"] == "function_call" for e in evs)

    def test_visible_answer_fences_on_all_three_apis(self):
        for name, body, want in (("fence", "Format:\n```xml\n" + RM + "\n```\nDone.", 0),
                                 ("inline", "Use `" + RM + "` here.", 0),
                                 ("real", "Doing it.\n\n" + CALL, 1)):
            with self.subTest(name):
                a = self.ask("ok</think>\n\n" + body)
                self.assertEqual(set(a.values()), {want}, a)


if __name__ == "__main__":
    unittest.main()
