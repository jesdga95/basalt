"""tool_choice on Chat Completions, Messages and Responses against a scripted engine (no GPU, no pack): the server
writes the opening of the call, after the thinking or at the end of the prompt, so the model can only go on with one."""
from __future__ import annotations

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.engine.mock import MockEngine  # noqa: E402
from serve.frontend import ChatTemplate, forced_call, tool_choice_of  # noqa: E402
from serve.handler import serve  # noqa: E402
from serve.service import REASONING_CLOSE, Service  # noqa: E402
from serve.util.text import ByteTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ChatTemplate(ROOT / "serve/chat_template.jinja")
ARGS = "<parameter=q>\n2+2\n</parameter>\n</function>\n</tool_call>"


class CallingEngine(MockEngine):
    """Thinks, then answers in text, unless its prompt already opens a call: then it finishes that call."""
    THOUGHT = "I could look this up. " * 8
    ANSWER = "The answer is 4."

    def __init__(self, tok):
        super().__init__(tok, "x", max_context=32768)
        self.prompts = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        prompt = self.tok.decode(ids)
        if prompt.endswith("<function="):
            text = "search>\n" + ARGS
        elif prompt.endswith("<function=search>\n"):
            text = ARGS
        elif prompt.endswith("</think>\n\n"):
            text = self.ANSWER
        else:
            text = self.THOUGHT + "</think>\n\n" + self.ANSWER
        for t in (self.tok.encode(text) + self.tok.encode("<|im_end|>", parse_special=True))[:max_new]:
            if cancel.is_set():
                return
            yield t


FN = [{"type": "function", "function": {"name": "search", "description": "search the web",
                                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}},
      {"type": "function", "function": {"name": "done", "parameters": {"type": "object", "properties": {}}}}]
AN = [{"name": "search", "description": "search the web",
       "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}},
      {"name": "done", "input_schema": {"type": "object", "properties": {}}}]
RS = [{"type": "function", "name": "search", "description": "search the web",
       "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}},
      {"type": "function", "name": "done", "parameters": {"type": "object", "properties": {}}}]


class Parsing(unittest.TestCase):
    def test_shapes_of_every_api(self):
        cases = [(None, ("auto", None)), ("auto", ("auto", None)), ("none", ("none", None)),
                 ("required", ("required", None)), ({"type": "any"}, ("required", None)),
                 ({"type": "auto"}, ("auto", None)), ({"type": "none"}, ("none", None)),
                 ({"type": "function", "function": {"name": "a"}}, ("named", "a")),
                 ({"type": "function", "name": "a"}, ("named", "a")),
                 ({"type": "function", "namespace": "ns", "name": "a"}, ("named", "ns.a")),
                 ({"type": "tool", "name": "a"}, ("named", "a"))]
        for value, want in cases:
            with self.subTest(value=value):
                self.assertEqual(tool_choice_of(value), want)

    def test_bad_shapes_are_errors(self):
        for value in ("sometimes", 3, [], {"type": "banana"}, {"type": "function"}, {"type": "tool", "name": ""}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "tool_choice"):
                tool_choice_of(value)

    def test_forcing(self):
        tools = [{"name": "a"}, {"name": "b"}]
        self.assertEqual(forced_call("auto", tools), (tools, None))
        self.assertEqual(forced_call("none", tools), (None, None))
        self.assertEqual(forced_call("required", tools), (tools, "<tool_call>\n<function="))
        self.assertEqual(forced_call("required", tools[:1]), (tools[:1], "<tool_call>\n<function=a>\n"))
        self.assertEqual(forced_call({"type": "tool", "name": "b"}, tools), (tools, "<tool_call>\n<function=b>\n"))
        for value, given in (("required", None), ({"type": "tool", "name": "c"}, tools)):
            with self.assertRaises(ValueError):
                forced_call(value, given)
        self.assertEqual(forced_call("none", None), (None, None))


class Base(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = CallingEngine(self.tok)
        self.svc = Service(self.engine, self.tok, TEMPLATE)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode()
                return r.status, (raw if body.get("stream") else json.loads(raw))
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def offered(self):
        return "search the web" in self.tok.decode(self.engine.prompts[0])

    def opening(self, name):
        return "<tool_call>\n<function=" + (name + ">\n" if name else "")


class OpenAI(Base):
    NAMED = {"type": "function", "function": {"name": "search"}}
    NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}

    def chat(self, **extra):
        return self.post("/v1/chat/completions", {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": "2+2?"}],
                                                  "max_tokens": 400, "tools": FN, **extra})

    def call_of(self, code, b, stream):
        """-> (finish_reason, [(name, arguments)], reasoning)"""
        self.assertEqual(code, 200, b)
        if not stream:
            choice = b["choices"][0]
            calls = [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                     for c in choice["message"].get("tool_calls") or []]
            return choice["finish_reason"], calls, choice["message"].get("reasoning_content") or ""
        names, args, reasoning, finish = {}, {}, "", None
        for line in b.splitlines():
            if not line.startswith("data: {"):
                continue
            c = json.loads(line[6:])
            d = c["choices"][0]["delta"]
            reasoning += d.get("reasoning_content") or ""
            for tc in d.get("tool_calls") or []:
                names.setdefault(tc["index"], tc["function"].get("name"))
                args[tc["index"]] = args.get(tc["index"], "") + tc["function"].get("arguments", "")
            finish = c["choices"][0]["finish_reason"] or finish
        return finish, [(names[i], json.loads(args[i])) for i in sorted(names)], reasoning

    def test_forced_after_the_thinking(self):
        for choice, name in (("required", None), (self.NAMED, "search")):
            for stream in (False, True):
                with self.subTest(choice=choice, stream=stream):
                    self.engine.prompts = []
                    finish, calls, reasoning = self.call_of(*self.chat(tool_choice=choice, stream=stream), stream)
                    self.assertEqual((finish, calls), ("tool_calls", [("search", {"q": "2+2"})]))
                    self.assertEqual(reasoning, CallingEngine.THOUGHT)
                    first, second = self.engine.prompts
                    # written where the thinking ended, after the blank line the template puts before a call
                    self.assertEqual(second, first + self.tok.encode(CallingEngine.THOUGHT + "</think>\n\n"
                                                                     + self.opening(name)))

    def test_forced_without_thinking_ends_the_prompt(self):
        for choice, name in (("required", None), (self.NAMED, "search")):
            for stream in (False, True):
                with self.subTest(choice=choice, stream=stream):
                    self.engine.prompts = []
                    out = self.chat(tool_choice=choice, stream=stream, **self.NO_THINKING)
                    finish, calls, reasoning = self.call_of(*out, stream)
                    self.assertEqual((finish, calls, reasoning), ("tool_calls", [("search", {"q": "2+2"})], ""))
                    self.assertEqual(len(self.engine.prompts), 1)
                    self.assertTrue(self.tok.decode(self.engine.prompts[0]).endswith("</think>\n\n"
                                                                                     + self.opening(name)))

    def test_budget_wrap_up_opens_the_call(self):
        finish, calls, reasoning = self.call_of(*self.chat(tool_choice="required", reasoning_budget_tokens=20), False)
        self.assertEqual((finish, calls), ("tool_calls", [("search", {"q": "2+2"})]))
        first, second = self.engine.prompts
        extra = self.tok.encode(REASONING_CLOSE + "<tool_call>\n<function=", parse_special=True)
        self.assertEqual(second, first + self.tok.encode(CallingEngine.THOUGHT[:20]) + extra)

    def test_auto_absent_and_none_leave_it_to_the_model(self):
        for extra in ({}, {"tool_choice": "auto"}, {"tool_choice": "none"}):
            with self.subTest(extra=extra):
                self.engine.prompts = []
                code, b = self.chat(**extra)
                self.assertEqual(code, 200, b)
                self.assertEqual(b["choices"][0]["message"]["content"], CallingEngine.ANSWER)
                self.assertEqual(len(self.engine.prompts), 1)
                self.assertEqual(self.offered(), extra.get("tool_choice") != "none")

    def test_flat_shape_names_a_function(self):
        finish, calls, _ = self.call_of(*self.chat(tool_choice={"type": "function", "name": "search"}), False)
        self.assertEqual((finish, calls), ("tool_calls", [("search", {"q": "2+2"})]))

    def test_cut_by_max_tokens_is_not_a_tool_call(self):
        finish, calls, _ = self.call_of(*self.chat(tool_choice="required", max_tokens=5, **self.NO_THINKING), False)
        self.assertEqual((finish, calls), ("length", []))

    def test_invalid_choices_are_400s_naming_tool_choice(self):
        missing = {"type": "function", "function": {"name": "nope"}}
        for extra in ({"tool_choice": missing}, {"tool_choice": "required", "tools": None},
                      {"tool_choice": "sometimes"}, {"tool_choice": {"type": "banana"}}, {"tool_choice": 7}):
            for stream in (False, True):
                with self.subTest(extra=extra, stream=stream):
                    code, b = self.chat(stream=stream, **extra)
                    self.assertEqual(code, 400, b)
                    self.assertIn("tool_choice", b["error"]["message"])
        self.assertEqual(self.engine.prompts, [])


class Anthropic(Base):
    NO_THINKING = {"thinking": {"type": "disabled"}}
    THINKING = {"thinking": {"type": "enabled", "budget_tokens": 1024}}

    def msg(self, **extra):
        return self.post("/v1/messages", {"model": "qwen3.8-flash-next", "max_tokens": 400, "tools": AN,
                                          "messages": [{"role": "user", "content": "2+2?"}], **extra})

    def blocks(self, code, b, stream):
        self.assertEqual(code, 200, b)
        if not stream:
            return b["stop_reason"], b["content"]
        stop, out = None, {}
        for line in b.splitlines():
            if not line.startswith("data: {"):
                continue
            e = json.loads(line[6:])
            if e["type"] == "content_block_start":
                out[e["index"]] = dict(e["content_block"], _json="")
            elif e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta":
                out[e["index"]]["_json"] += e["delta"]["partial_json"]
            elif e["type"] == "message_delta":
                stop = e["delta"]["stop_reason"]
        return stop, [dict(v, input=json.loads(v["_json"])) if v["type"] == "tool_use" else v
                      for _, v in sorted(out.items())]

    def test_any_and_tool_with_and_without_thinking(self):
        for choice, name in (({"type": "any"}, None), ({"type": "tool", "name": "search"}, "search")):
            for mode in (self.THINKING, self.NO_THINKING):
                for stream in (False, True):
                    with self.subTest(choice=choice, thinking=bool(mode["thinking"]["type"] == "enabled"),
                                      stream=stream):
                        self.engine.prompts = []
                        stop, content = self.blocks(*self.msg(tool_choice=choice, stream=stream, **mode), stream)
                        calls = [c for c in content if c["type"] == "tool_use"]
                        self.assertEqual(stop, "tool_use")
                        self.assertEqual([(c["name"], c["input"]) for c in calls], [("search", {"q": "2+2"})])
                        if mode is self.THINKING:
                            first, second = self.engine.prompts
                            self.assertEqual(second, first + self.tok.encode(
                                CallingEngine.THOUGHT + "</think>\n\n" + self.opening(name)))
                        else:
                            self.assertTrue(self.tok.decode(self.engine.prompts[0]).endswith(self.opening(name)))

    def test_auto_and_none(self):
        for choice in (None, {"type": "auto"}, {"type": "none"}):
            with self.subTest(choice=choice):
                self.engine.prompts = []
                extra = {} if choice is None else {"tool_choice": choice}
                code, b = self.msg(**extra)
                self.assertEqual(code, 200, b)
                self.assertEqual(b["stop_reason"], "end_turn")
                self.assertEqual(self.offered(), (choice or {}).get("type") != "none")

    def test_invalid_choices_are_400s(self):
        for extra in ({"tool_choice": {"type": "tool", "name": "nope"}}, {"tool_choice": {"type": "any"}, "tools": []},
                      {"tool_choice": {"type": "banana"}}, {"tool_choice": "any"}):
            with self.subTest(extra=extra):
                code, b = self.msg(**extra)
                self.assertEqual(code, 400, b)
                self.assertIn("tool_choice", json.dumps(b))
        self.assertEqual(self.engine.prompts, [])


class Responses(Base):
    def resp(self, **extra):
        return self.post("/v1/responses", {"model": "qwen3.8-flash-next", "input": "2+2?", "max_output_tokens": 400, "tools": RS,
                                           **extra})

    def calls(self, code, b):
        self.assertEqual(code, 200, b)
        return [(i["name"], json.loads(i["arguments"])) for i in b["output"] if i["type"] == "function_call"]

    def test_required_and_named(self):
        for choice, name in (("required", None), ({"type": "function", "name": "search"}, "search")):
            with self.subTest(choice=choice):
                self.engine.prompts = []
                self.assertEqual(self.calls(*self.resp(tool_choice=choice)), [("search", {"q": "2+2"})])
                first, second = self.engine.prompts
                self.assertEqual(second, first + self.tok.encode(CallingEngine.THOUGHT + "</think>\n\n"
                                                                 + self.opening(name)))

    def test_without_thinking_and_streamed(self):
        code, b = self.resp(tool_choice="required", reasoning={"effort": "none"}, stream=True)
        self.assertEqual(code, 200)
        self.assertIn('"type":"function_call"', b.replace(" ", ""))
        self.assertTrue(self.tok.decode(self.engine.prompts[0]).endswith(self.opening(None)))

    def test_auto_and_none(self):
        for choice in ("auto", "none"):
            with self.subTest(choice=choice):
                self.engine.prompts = []
                code, b = self.resp(tool_choice=choice)
                self.assertEqual(self.calls(code, b), [])
                self.assertEqual(self.offered(), choice == "auto")

    def test_invalid_choices_are_400s(self):
        for choice in ({"type": "function", "name": "nope"}, "sometimes", {"type": "web_search"}):
            with self.subTest(choice=choice):
                code, b = self.resp(tool_choice=choice)
                self.assertEqual(code, 400, b)
                self.assertEqual(b["error"]["param"], "tool_choice")
        code, b = self.resp(tool_choice="required", tools=[])
        self.assertEqual((code, b["error"]["param"]), (400, "tool_choice"))
        self.assertEqual(self.engine.prompts, [])


if __name__ == "__main__":
    unittest.main()
