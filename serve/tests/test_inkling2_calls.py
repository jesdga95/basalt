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
            for stream in (True, False):
                evs = run(reply, stream_tools=stream)
                self.assertFalse([e for e in evs if e.kind == "tool_call"], (reply, stream))
                if not stream:
                    self.assertFalse([e for e in evs if e.kind in ("tool_start", "tool_args")], reply)
                    self.assertFalse([e for e in evs if e.kind == "content" and TOOL_JSON in e.text], reply)

    def test_name_state_handles_end_and_header_before_later_tool_json(self):
        call_block = HEADER + "edit_file" + ENVELOPE + END
        for reply in (HEADER + "Plain text." + END + call_block, HEADER + "Plain text." + call_block):
            self.agree(reply, [("content", "Plain text."), ("args", ARGS_TEXT), CALL])


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

    def test_prose_xml_openers_do_not_stall_streaming_without_gt(self):
        # prose containing <tool_call>, <function_calls> or <invoke name=...> must flush during feed(), not wait for '>'
        for opener, follow in (("<tool_call>", " tags in prose without angle brackets"),
                               ("<function_calls>", " in prose without angle brackets"),
                               ('<invoke name="edit_file">', " is how you invoke it")):
            p = Inkling2Parser(tools=TOOLS)
            got = ""
            got += "".join(e.text for e in p.feed(HEADER + TEXT + "Use " + opener) if e.kind == "content")
            for ch in follow:
                got += "".join(e.text for e in p.feed(ch) if e.kind == "content")
            self.assertEqual(got, "Use " + opener + follow, opener)

    def test_zero_arg_xml_call_and_trailing_prose_without_gt(self):
        body = '<invoke name="edit_file"></invoke>'
        for width in WIDTHS:
            got = [g for g in merged(run(self.blocks(body), width)) if g[0] != "args"]
            self.assertEqual(got, [("call", "edit_file", {})], width)
        # trailing prose with no '>' after </invoke> flushes the call and text during feed()
        p = Inkling2Parser(tools=TOOLS)
        evs = p.feed(HEADER + TEXT + body)
        evs += p.feed(" Done now.")
        self.assertEqual([(e.kind, e.call.name if e.call else e.text) for e in evs if e.kind in ("tool_call", "content")],
                         [("tool_call", "edit_file"), ("content", "Done now.")])


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

    def test_non_streaming_truncated_native_call_does_not_leak(self):
        for script in (HEADER + "edit_file" + ENVELOPE + END, HEADER + ENVELOPE + END):
            tok = service().tok
            cut = len(tok.encode(script[:script.index("old_string")], parse_special=True))
            for path, extra in (self.OPENAI, self.ANTHROPIC):
                svc = service(script)
                svc.answer_room = 0     # no thinking budget: MockEngine replays from token 0 on a second pass
                httpd = serve(svc, port=0)
                body = dict(extra, model="ink", stream=False, max_tokens=cut,
                            messages=[{"role": "user", "content": "edit it"}])
                try:
                    req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{path}",
                                                 data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
                    with urllib.request.urlopen(req, timeout=30) as r:
                        res = json.loads(r.read().decode())
                finally:
                    httpd.shutdown()
                    httpd.server_close()
                if path == "/v1/chat/completions":
                    self.assertEqual(res["choices"][0]["finish_reason"], "length", script)
                    self.assertNotIn("tool_calls", res["choices"][0]["message"], script)
                else:
                    self.assertEqual(res["stop_reason"], "max_tokens", script)
                    self.assertEqual([b for b in res["content"] if b["type"] == "tool_use"], [], script)



class NoneChoice(unittest.TestCase):
    """tool_choice "none" offers no tools, so the reply has no calls: a native call the model writes anyway is text,
    shown whole and in the order written, with no call event of any kind, whole and streamed. The thinking stays
    reasoning, and a reply that is cut off or empty shows as such. Under "auto" the same reply is the call."""

    TOOL = {"type": "function", "function": {"name": "read_file", "description": "Read a file from the workspace",
                                             "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}
    THOUGHT = "I should look at the file first."
    CALL = HEADER + "read_file" + TOOL_JSON + '{"name":"read_file","args":{"path":"src/app.py"}}' + END
    BARE = HEADER + TOOL_JSON + '{"name":"read_file","args":{"path":"src/app.py"}}' + END
    TEXT_OF = "read_file" + TOOL_JSON + '{"name":"read_file","args":{"path":"src/app.py"}}'

    def ask(self, script, stream, max_tokens=900, **extra):
        """(reply: content, reasoning, calls, finish and every tool_calls delta, the prompt the engine read)."""
        svc = service(script)
        svc.answer_room = 0     # no thinking budget: a thinking request's first pass would stop at it (MockEngine replays)
        httpd = serve(svc, port=0)
        body = dict(model="ink", stream=stream, max_tokens=max_tokens, tools=[self.TOOL],
                    messages=[{"role": "user", "content": "what is in src/app.py?"}], **extra)
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode()
        finally:
            httpd.shutdown()
            httpd.server_close()
        prompt = svc.tok.decode(svc.engine.requests[0][0])
        if not stream:
            choice = json.loads(raw)["choices"][0]
            m = choice["message"]
            calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in m.get("tool_calls") or []]
            return dict(content=m.get("content"), reasoning=m.get("reasoning_content") or "", calls=calls,
                        finish=choice["finish_reason"], deltas=len(m.get("tool_calls") or [])), prompt
        content, reasoning, names, args, deltas, finish = "", "", {}, {}, 0, None
        for line in raw.splitlines():
            if not line.startswith("data: {"):
                continue
            c = json.loads(line[6:])["choices"][0]
            content += c["delta"].get("content") or ""
            reasoning += c["delta"].get("reasoning_content") or ""
            for tc in c["delta"].get("tool_calls") or []:
                deltas += 1
                names.setdefault(tc["index"], tc.get("function", {}).get("name"))
                args[tc["index"]] = args.get(tc["index"], "") + (tc.get("function", {}).get("arguments") or "")
            finish = c["finish_reason"] or finish
        calls = [(names[i], json.loads(args[i])) for i in sorted(names)]
        return dict(content=content or None, reasoning=reasoning, calls=calls, finish=finish, deltas=deltas), prompt

    def test_a_native_call_under_none_is_text(self):
        for name, call in (("named", self.CALL), ("headerless", self.BARE)):
            script = HEADER + THINK + self.THOUGHT + END + call
            text = self.TEXT_OF if name == "named" else self.TEXT_OF[len("read_file"):]
            for choice in ("none", {"type": "none"}):
                for stream in (False, True):
                    with self.subTest(form=name, choice=choice, stream=stream):
                        got, prompt = self.ask(script, stream, tool_choice=choice)
                        self.assertNotIn("Read a file from the workspace", prompt)     # no tools offered
                        self.assertEqual(got, dict(content=text, reasoning=self.THOUGHT, calls=[], finish="stop",
                                                   deltas=0))

    def test_text_and_the_call_under_none_keep_their_order(self):
        script = HEADER + THINK + self.THOUGHT + END + HEADER + TEXT + "Let me read it." + END + self.CALL
        for stream in (False, True):
            with self.subTest(stream=stream):
                got, _ = self.ask(script, stream, tool_choice="none")
                self.assertEqual(got, dict(content="Let me read it." + self.TEXT_OF, reasoning=self.THOUGHT, calls=[],
                                           finish="stop", deltas=0))

    def test_the_same_reply_under_auto_is_the_call(self):
        script = HEADER + THINK + self.THOUGHT + END + self.CALL
        for choice in ("auto", None):
            for stream in (False, True):
                with self.subTest(choice=choice, stream=stream):
                    extra = {} if choice is None else {"tool_choice": choice}
                    got, prompt = self.ask(script, stream, **extra)
                    self.assertIn("Read a file from the workspace", prompt)
                    self.assertEqual({k: got[k] for k in ("content", "reasoning", "calls", "finish")},
                                     dict(content=None, reasoning=self.THOUGHT, calls=[("read_file", {"path": "src/app.py"})],
                                          finish="tool_calls"))

    def test_a_call_cut_off_under_none_shows_as_cut_off_text(self):
        tok = service().tok
        script = HEADER + THINK + self.THOUGHT + END + self.CALL
        cut = script.index('"src/app')                  # the output ends inside the arguments
        n = len(tok.encode(script[:cut], parse_special=True))
        for stream in (False, True):
            with self.subTest(stream=stream):
                got, _ = self.ask(script, stream, max_tokens=n, tool_choice="none")
                self.assertEqual(got, dict(content=self.TEXT_OF[:self.TEXT_OF.index('"src/app')], reasoning=self.THOUGHT,
                                           calls=[], finish="length", deltas=0))

    def test_an_empty_answer_under_none_is_empty(self):
        script = HEADER + THINK + self.THOUGHT + END
        for stream in (False, True):
            with self.subTest(stream=stream):
                got, _ = self.ask(script, stream, tool_choice="none")
                self.assertEqual(got, dict(content=None, reasoning=self.THOUGHT, calls=[], finish="stop", deltas=0))


if __name__ == "__main__":
    unittest.main()
