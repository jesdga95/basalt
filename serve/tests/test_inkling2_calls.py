"""Inkling 2 tool calls that reach the reply in a form other than the one in the template (#540, #553): the native
call with no name in its header, with no header, or in a block left open, and the call written as XML
(`<invoke name=...><parameter name=...>`, bare or wrapped in `<tool_call><function_calls>`). Each is a call whole and
streamed, in every chunking, through the OpenAI and Anthropic routes; text that only talks about the tags stays text."""
import json
import unittest
import urllib.request

from serve.handler import serve
from serve.models.inkling2 import END, HEADER, TEXT, THINK, TOOL_JSON, Inkling2Parser
from serve.tests.test_inkling2_dialect import merged, service

ARGS = {"path": "src/app.py", "old_string": 'x = "a"\n    y = 1', "new_string": "x = 'b'\n    y = 2 < 3"}
TOOLS = [{"name": "edit_file", "description": "Edit a file", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"},
                                     "new_string": {"type": "string"}, "count": {"type": "integer"}}}}]
WIDTHS = (None, 1, 2, 3, 5, 11, 64)
CALL = ("call", "edit_file", ARGS)
ARGS_TEXT = json.dumps(ARGS, separators=(",", ":"))
ENVELOPE = TOOL_JSON + '{"name":"edit_file","args":' + json.dumps(ARGS, separators=(",", ":")) + "}"


def invoke(args=ARGS, string=False):
    attr = ' string="true"' if string else ""
    return '<invoke name="edit_file">' + "".join(f'<parameter name="{k}"{attr}>{v}</parameter>' for k, v in args.items()) + "</invoke>"


def wrapped(inner):
    return "<tool_call><function_calls>" + inner + "</function_calls></tool_call>"


def run(text, width=None, tools=TOOLS, stream_tools=True):
    p = Inkling2Parser(tools=tools, stream_tools=stream_tools)
    events = []
    if width is None:
        events += p.feed(text)
    else:
        for i in range(0, len(text), width):
            events += p.feed(text[i:i + width])
    return events + p.finish()


class Native(unittest.TestCase):
    def agree(self, reply, want, tools=TOOLS):
        for width in WIDTHS:
            for stream in (True, False):
                got = merged(run(reply, width, tools, stream))
                if not stream:
                    got = [g for g in got if g[0] != "args"]
                self.assertEqual(got, want if stream else [w for w in want if w[0] != "args"], (reply, width, stream))

    def test_a_header_with_no_name(self):
        self.agree(HEADER + ENVELOPE + END, [("args", ARGS_TEXT), CALL])

    def test_a_header_with_no_name_and_no_end(self):
        self.agree(HEADER + ENVELOPE, [("args", ARGS_TEXT), CALL])

    def test_a_bare_marker(self):
        self.agree(ENVELOPE + END, [("args", ARGS_TEXT), CALL])
        self.agree(ENVELOPE, [("args", ARGS_TEXT), CALL])

    def test_after_text_and_a_stray_end(self):
        self.agree(HEADER + TEXT + "On it." + END + HEADER + ENVELOPE + END, [("content", "On it."), ("args", ARGS_TEXT), CALL])
        self.agree(END + HEADER + ENVELOPE + END, [("args", ARGS_TEXT), CALL])
        self.agree("Sure." + ENVELOPE + END, [("content", "Sure."), ("args", ARGS_TEXT), CALL])

    def test_a_block_left_open(self):
        named = HEADER + "edit_file" + TOOL_JSON + '{"name":"edit_file","args":{"path":"a"}}' + END
        for opened, kind in ((TEXT + "On it.", "content"), (THINK + "Edit it.", "reasoning")):
            self.agree(HEADER + opened + named, [(kind, "On it." if kind == "content" else "Edit it."),
                                                 ("args", '{"path":"a"}'), ("call", "edit_file", {"path": "a"})])
            self.agree(HEADER + opened + HEADER + ENVELOPE + END, [(kind, "On it." if kind == "content" else "Edit it."),
                                                                  ("args", ARGS_TEXT), CALL])

    def test_the_closing_brace_and_the_whitespace_after_it_stay_out_of_the_arguments(self):
        for end in ("", END):
            p = Inkling2Parser(tools=TOOLS)
            events = []
            for piece in (HEADER + "edit_file" + TOOL_JSON + '{"name":"edit_file","args":{"path":"a"}}', "\n ", end):
                events += p.feed(piece)
            events += p.finish()
            self.assertEqual(merged(events), [("args", '{"path":"a"}'), ("call", "edit_file", {"path": "a"})], end)

    def test_a_marker_that_names_nothing_is_text(self):
        for body in ('{"args":{"q":"one"}}', '{"nam', "not json"):
            for end in (END, ""):
                events = run(TOOL_JSON + body + end)
                self.assertFalse([e for e in events if e.kind == "tool_call"], body)
                self.assertEqual("".join(e.text for e in events if e.kind == "content"), TOOL_JSON + body, body)

    def test_a_call_cut_off_inside_its_arguments_is_not_a_call(self):
        for reply in (HEADER + "edit_file" + TOOL_JSON + '{"name":"edit_file","args":{"path":"a', ENVELOPE[:-9]):
            self.assertFalse([e for e in run(reply) if e.kind == "tool_call"], reply)


class Xml(unittest.TestCase):
    FORMS = {"bare": invoke(), "bare with string": invoke(string=True), "wrapped": wrapped(invoke()),
             "wrapped with string": wrapped(invoke(string=True)),
             "wrapped with spaces": "<tool_call>\n<function_calls>\n" + invoke() + "\n</function_calls>\n</tool_call>",
             "function_calls only": "<function_calls>" + invoke() + "</function_calls>"}

    def blocks(self, body):
        return HEADER + TEXT + body + END

    def test_every_form_is_a_call_whole_and_streamed(self):
        for name, body in self.FORMS.items():
            for reply in (self.blocks(body), HEADER + body + END, body, self.blocks(body).replace(END, "")):
                results = set()
                for width in WIDTHS:
                    for stream in (True, False):
                        got = merged(run(reply, width, TOOLS, stream))
                        results.add(json.dumps([g for g in got if stream or g[0] != "args"]) if stream else None)
                        self.assertEqual([g for g in got if g[0] != "args"], [CALL], (name, width, stream, got))
                self.assertEqual(results - {None}, {json.dumps([["args", ARGS_TEXT], list(CALL)])}, name)

    def test_text_around_the_call_stays(self):
        body = wrapped(invoke())
        for reply, want in (("I'll edit it.\n\n" + body, [("content", "I'll edit it."), CALL]),
                            ("I'll edit it. " + body + "\n\nDone.", [("content", "I'll edit it."), CALL, ("content", "Done.")]),
                            (body + "\nDone.", [CALL, ("content", "Done.")])):
            for width in WIDTHS:
                self.assertEqual([g for g in merged(run(self.blocks(reply), width)) if g[0] != "args"], want, (reply, width))

    def test_parallel_calls(self):
        second = invoke({"path": "b.py", "old_string": "1", "new_string": "2"})
        for body in (wrapped(invoke() + second), invoke() + second, wrapped(invoke()) + wrapped(second)):
            for width in WIDTHS:
                got = [g for g in merged(run(self.blocks(body), width)) if g[0] != "args"]
                self.assertEqual(got, [CALL, ("call", "edit_file", {"path": "b.py", "old_string": "1", "new_string": "2"})],
                                 (body, width))

    def test_typed_parameters(self):
        body = '<invoke name="edit_file"><parameter name="path">a.py</parameter><parameter name="count">3</parameter>' \
               '<parameter name="note" string="true">7</parameter><parameter name="flag" string="false">true</parameter></invoke>'
        want = ("call", "edit_file", {"path": "a.py", "count": 3, "note": "7", "flag": True})
        for width in WIDTHS:
            self.assertEqual([g for g in merged(run(body, width)) if g[0] != "args"], [want], width)

    def test_a_close_inside_a_value_is_the_value(self):
        args = {"path": "a.xml", "old_string": "<a>1</parameter>\nmore", "new_string": "x</invoke> y"}
        for width in WIDTHS:
            got = [g for g in merged(run(wrapped(invoke(args)), width)) if g[0] != "args"]
            self.assertEqual(got, [("call", "edit_file", args)], width)

    def test_talk_about_the_tags_is_text(self):
        for text in ("Use <invoke name=\"edit_file\"> to call it.",
                     "Wrap calls in <tool_call> and <function_calls> tags.",
                     "```xml\n" + wrapped(invoke()) + "\n```\nlike that.",
                     "Write `" + invoke() + "` for a call.",
                     "~~~\n" + invoke() + "\n~~~",
                     invoke().replace("edit_file", "rm_rf"),                    # a tool the request does not have
                     invoke()[:-len("</invoke>")],                             # never closed
                     wrapped(invoke())[:30],                                    # cut off
                     "<invoke name=\"edit_file\">no parameters here, just prose",
                     "<tool_call>not a call</tool_call>"):
            for width in WIDTHS:
                for stream in (True, False):
                    events = run(self.blocks(text), width, TOOLS, stream)
                    self.assertFalse([e for e in events if e.kind in ("tool_call", "tool_start")], (text, width))
                    self.assertEqual("".join(e.text for e in events if e.kind == "content"), text, (text, width))

    def test_without_tools_it_is_text(self):
        for tools in (None, []):
            events = run(self.blocks(wrapped(invoke())), 7, tools)
            self.assertFalse([e for e in events if e.kind == "tool_call"])
            self.assertEqual("".join(e.text for e in events if e.kind == "content"), wrapped(invoke()))

    def test_reasoning_is_never_a_call(self):
        events = run(HEADER + THINK + wrapped(invoke()) + END, 5)
        self.assertFalse([e for e in events if e.kind == "tool_call"])
        self.assertEqual("".join(e.text for e in events if e.kind == "reasoning"), wrapped(invoke()))

    def test_a_call_then_a_native_call_keep_their_order(self):
        reply = self.blocks(wrapped(invoke())) + HEADER + "edit_file" + TOOL_JSON + '{"name":"edit_file","args":{"path":"z"}}' + END
        for width in WIDTHS:
            self.assertEqual([g for g in merged(run(reply, width)) if g[0] != "args"],
                             [CALL, ("call", "edit_file", {"path": "z"})], width)


class Routes(unittest.TestCase):
    """The same replies through the server: OpenAI and Anthropic, whole and streamed."""

    OPENAI = ("/v1/chat/completions", {"tools": [{"type": "function", "function": {
        "name": "edit_file", "parameters": TOOLS[0]["parameters"]}}]})
    ANTHROPIC = ("/v1/messages", {"tools": [{"name": "edit_file", "input_schema": TOOLS[0]["parameters"]}]})

    def ask(self, script, route, stream):
        httpd = serve(service(script), port=0)
        path, body = route
        body = dict(body, model="ink", stream=stream, max_tokens=900, messages=[{"role": "user", "content": "edit it"}])
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{path}", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode()
        finally:
            httpd.shutdown()
            httpd.server_close()

    @staticmethod
    def openai(raw, stream):
        """(content, [(name, arguments)])."""
        if not stream:
            m = json.loads(raw)["choices"][0]["message"]
            return m.get("content") or "", [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in m.get("tool_calls") or []]
        names, args, text = {}, {}, ""
        for line in raw.splitlines():
            if line.startswith("data: {"):
                for c in json.loads(line[6:])["choices"]:
                    text += c["delta"].get("content") or ""
                    for tc in c["delta"].get("tool_calls") or []:
                        names.setdefault(tc["index"], tc.get("function", {}).get("name"))
                        args[tc["index"]] = args.get(tc["index"], "") + (tc.get("function", {}).get("arguments") or "")
        return text, [(names[i], json.loads(args[i])) for i in sorted(names)]

    @staticmethod
    def anthropic(raw, stream):
        if not stream:
            blocks = json.loads(raw)["content"]
            return ("".join(b["text"] for b in blocks if b["type"] == "text"),
                    [(b["name"], b["input"]) for b in blocks if b["type"] == "tool_use"])
        text, calls = "", []
        for line in raw.splitlines():
            if line.startswith("data: {"):
                e = json.loads(line[6:])
                if e["type"] == "content_block_start" and e["content_block"]["type"] == "tool_use":
                    calls.append([e["content_block"]["name"], ""])
                elif e["type"] == "content_block_delta":
                    if e["delta"]["type"] == "text_delta":
                        text += e["delta"]["text"]
                    elif e["delta"]["type"] == "input_json_delta":
                        calls[-1][1] += e["delta"]["partial_json"]
        return text, [(n, json.loads(a or "{}")) for n, a in calls]

    def test_every_form_is_a_call_on_both_routes(self):
        replies = {"headerless native": HEADER + ENVELOPE + END, "bare marker": ENVELOPE + END,
                   "bare xml": HEADER + TEXT + invoke() + END,
                   "wrapped xml": HEADER + TEXT + wrapped(invoke()) + END,
                   "wrapped xml with string": HEADER + TEXT + wrapped(invoke(string=True)) + END}
        for name, reply in replies.items():
            for route, read in ((self.OPENAI, self.openai), (self.ANTHROPIC, self.anthropic)):
                for stream in (False, True):
                    with self.subTest(form=name, route=route[0], stream=stream):
                        text, calls = read(self.ask(reply, route, stream), stream)
                        self.assertEqual(calls, [("edit_file", ARGS)])
                        self.assertEqual(text.strip(), "")

    def test_a_mention_is_content_on_both_routes(self):
        talk = "Use <invoke name=\"edit_file\"> for that, inside <tool_call> tags."
        for route, read in ((self.OPENAI, self.openai), (self.ANTHROPIC, self.anthropic)):
            for stream in (False, True):
                with self.subTest(route=route[0], stream=stream):
                    text, calls = read(self.ask(HEADER + TEXT + talk + END, route, stream), stream)
                    self.assertEqual((text, calls), (talk, []))


if __name__ == "__main__":
    unittest.main()
