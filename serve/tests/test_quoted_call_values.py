"""String arguments preserve quoted calls, heredocs and repeated parameter markers."""
import json
import unittest

from serve.frontend import (CALL_END, CALL_START, FUNC_END, FUNC_START, PARAM_END, PARAM_START, OutputParser,
                            hold_back, parse_tool_call)

SCHEMA = {"name": "write", "parameters": {"properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}
QUOTED = (f"{CALL_START}\n{FUNC_START}write>\n{PARAM_START}path>\nx.txt\n{PARAM_END}\n{FUNC_END}\n{CALL_END}")
VALUES = {
    "quoted call": f"example transcript:\n{QUOTED}\nafter block",
    "quoted call at the end": f"example transcript:\n{QUOTED}",
    "heredoc": f"python3 - <<'PY'\nprint('''{QUOTED}''')\nPY",
    "repeated parameter": f"head part\n{PARAM_END}\n{PARAM_START}content>\ntail part",
    "unclosed opener in prose": f"A call starts with `{FUNC_START}NAME>` and its parameters follow.",
    "unclosed opener in code": f'opener = "{FUNC_START}"\n',
    "unclosed function with parameter": f"example:\n{FUNC_START}foo>\n{PARAM_START}x>\nunfinished",
    # the three closing tags on their own lines inside a string, with no call opened before them
    "whole close inside": (f"line 1\n{PARAM_END}\n{FUNC_END}\n{CALL_END}\n"
                           "line 5 with \"quotes\" and <tags>\nlast line END"),
    "whole close twice": f"a\n{PARAM_END}\n{FUNC_END}\n{CALL_END}\nb\n{PARAM_END}\n{FUNC_END}\n{CALL_END}\nc",
    "whole close at the end": f"the parser stops at:\n{PARAM_END}\n{FUNC_END}\n{CALL_END}",
}


def call_text(value):
    return (f"{CALL_START}\n{FUNC_START}write>\n{PARAM_START}path>\nprobe.txt\n{PARAM_END}\n"
            f"{PARAM_START}content>\n{value}\n{PARAM_END}\n{FUNC_END}\n{CALL_END}")


class QuotedCallValues(unittest.TestCase):
    def test_values_are_exact_whole_streamed_and_chunked(self):
        for label, value in VALUES.items():
            text = call_text(value)
            expected = {"path": "probe.txt", "content": value}
            self.assertEqual(parse_tool_call(text[len(CALL_START):-len(CALL_END)], SCHEMA).arguments, expected)
            for stream in (False, True):
                for size in (1, 7, len(text)):
                    with self.subTest(label=label, stream=stream, size=size):
                        parser = OutputParser(thinking=False, tools=[SCHEMA], stream_tools=stream)
                        events = []
                        for i in range(0, len(text), size):
                            events.extend(parser.feed(text[i:i + size]))
                        events.extend(parser.finish())
                        calls = [e.call for e in events if e.kind == "tool_call"]
                        self.assertEqual([c.arguments for c in calls], [expected])
                        self.assertEqual("".join(e.text for e in events if e.kind == "content").strip(), "")
                        if stream:
                            self.assertEqual(json.loads("".join(e.text for e in events if e.kind == "tool_args")),
                                             expected)

    def test_prose_between_two_calls_still_ends_the_first(self):
        first, second = call_text("one"), call_text("two")
        parser = OutputParser(thinking=False, tools=[SCHEMA], stream_tools=True)
        events = parser.feed(first + "\nnow the second:\n" + second) + parser.finish()
        self.assertEqual([e.call.arguments["content"] for e in events if e.kind == "tool_call"], ["one", "two"])

    def test_whole_close_inside_a_value_then_a_second_call(self):
        value = VALUES["whole close inside"]
        text = call_text(value) + "\n" + call_text("two")
        for stream in (False, True):
            for size in (1, 7, len(text)):
                with self.subTest(stream=stream, size=size):
                    parser = OutputParser(thinking=False, tools=[SCHEMA], stream_tools=stream)
                    events = []
                    for i in range(0, len(text), size):
                        events.extend(parser.feed(text[i:i + size]))
                    events.extend(parser.finish())
                    calls = [e.call for e in events if e.kind == "tool_call"]
                    self.assertEqual([c.arguments["content"] for c in calls], [value, "two"])
                    self.assertEqual("".join(e.text for e in events if e.kind == "content").strip(), "")
                    if stream:
                        for call in calls:
                            pieces = "".join(e.text for e in events if e.kind == "tool_args" and e.call.id == call.id)
                            self.assertEqual(json.loads(pieces), call.arguments)

    def test_prose_after_a_call_that_never_closes_again_is_content(self):
        text = call_text("one") + "\ndone, the file is written."
        for stream in (False, True):
            for size in (1, 7, len(text)):
                with self.subTest(stream=stream, size=size):
                    parser = OutputParser(thinking=False, tools=[SCHEMA], stream_tools=stream)
                    events = []
                    for i in range(0, len(text), size):
                        events.extend(parser.feed(text[i:i + size]))
                    events.extend(parser.finish())
                    self.assertEqual([e.call.arguments["content"] for e in events if e.kind == "tool_call"], ["one"])
                    self.assertEqual("".join(e.text for e in events if e.kind == "content"),
                                     "done, the file is written.")

    def test_unclosed_quoted_function_preserves_following_text_and_calls(self):
        value = VALUES["unclosed function with parameter"]
        first, second = call_text(value), call_text("two")
        for tail in ("\nafter the call", "\nnow the second:\n" + second):
            text = first + tail
            for stream, tools in ((False, [SCHEMA]), (True, [SCHEMA]), (True, None)):
                for size in (1, 7, len(text)):
                    with self.subTest(tail=tail, stream=stream, tools=tools, size=size):
                        parser = OutputParser(thinking=False, tools=tools, stream_tools=stream)
                        events = []
                        for i in range(0, len(text), size):
                            events.extend(parser.feed(text[i:i + size]))
                        self.assertFalse(any(e.kind == "tool_call" for e in events))
                        events.extend(parser.finish())
                        calls = [e.call for e in events if e.kind == "tool_call"]
                        expected = [value, "two"] if tail.endswith(second) else [value]
                        self.assertEqual([c.arguments["content"] for c in calls], expected)
                        self.assertEqual("".join(e.text for e in events if e.kind == "content"),
                                         "now the second:" if len(expected) == 2 else "after the call")
                        if stream:
                            for call in calls:
                                pieces = "".join(e.text for e in events
                                                 if e.kind == "tool_args" and e.call.id == call.id)
                                self.assertEqual(json.loads(pieces), call.arguments)

    def test_malformed_tool_call_falls_back_to_content(self):
        for stream in (False, True):
            parser = OutputParser(thinking=False, tools=[SCHEMA], stream_tools=stream)
            events = parser.feed(f"{CALL_START}\nnot a function call\n{CALL_END} after") + parser.finish()
            self.assertFalse(any(e.kind == "tool_call" for e in events))
            content = "".join(e.text for e in events if e.kind == "content")
            self.assertIn("not a function call", content)
            self.assertIn("after", content)

    def test_missing_tool_call_closer_preserves_parsed_arguments(self):
        text = f"{CALL_START}\n{FUNC_START}write_file>\n{PARAM_START}content>\nhello\n{PARAM_END}\n{FUNC_END}"
        parser = OutputParser(thinking=False, tools=[SCHEMA], stream_tools=True)
        events = parser.feed(text) + parser.finish()
        calls = [e.call for e in events if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "write_file")
        self.assertEqual(calls[0].arguments, {"content": "hello"})

    def test_hold_back_matches_reference_across_edge_cases(self):
        def ref_hold_back(text, tags):
            best = 0
            for tag in tags:
                for n in range(1, len(tag)):
                    if text.endswith(tag[:n]):
                        best = max(best, n)
            return best

        tag_sets = [
            (),
            ("", "x"),
            ("</think>",),
            ("<tool_call>",),
            ("</think>", "\n\n<tool_call>", "<tool_call>"),
            ("ababa", "baba", "abc"),
            ["</parameter>", "</function>"],
        ]
        texts = [
            "", "hello world", "<", "</", "</t", "</think", "</think>",
            "abc\n", "abc\n\n", "abc\n\n<tool_cal", "abc\n\n<tool_call>",
            "zzabab", "zzaba", "zzb", "zz</param", "zz</func",
        ]
        for tags in tag_sets:
            for text in texts:
                with self.subTest(tags=tags, text=text):
                    self.assertEqual(hold_back(text, tags), ref_hold_back(text, tags))

    def test_large_multichunk_tool_call_and_external_buf_access(self):
        schema = {
            "name": "write",
            "parameters": {
                "properties": {
                    "content": {"type": "string"},
                    "meta": {"type": "object"},
                }
            },
        }
        big = ("line with some code and text 0123456789\n" * 1000)
        meta = {"items": list(range(200)), "ok": True}
        text = (
            f"before\n{CALL_START}\n{FUNC_START}write>\n"
            f"{PARAM_START}content>\n{big}\n{PARAM_END}\n"
            f"{PARAM_START}meta>\n{json.dumps(meta)}\n{PARAM_END}\n"
            f"{FUNC_END}\n{CALL_END}\nafter"
        )
        for stream in (False, True):
            with self.subTest(stream=stream):
                parser = OutputParser(thinking=False, tools=[schema], stream_tools=stream)
                events = []
                checked_buf = False
                for i in range(0, len(text), 7):
                    events.extend(parser.feed(text[i:i + 7]))
                    if not checked_buf and parser.state == "call" and i > 500:
                        # Reading parser.buf mid-call must materialize buffered chunks accurately.
                        self.assertIn("line with some code", parser.buf)
                        checked_buf = True
                events.extend(parser.finish())
                self.assertTrue(checked_buf)
                calls = [e.call for e in events if e.kind == "tool_call"]
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0].arguments, {"content": big, "meta": meta})
                self.assertEqual("".join(e.text for e in events if e.kind == "content"), "beforeafter")
                if stream:
                    streamed = "".join(e.text for e in events if e.kind == "tool_args")
                    self.assertEqual(json.loads(streamed), {"content": big, "meta": meta})

        # External buf mutation while in call state keeps CALL_END tracking consistent.
        p = OutputParser(thinking=False, tools=[SCHEMA], stream_tools=False)
        p.feed(f"{CALL_START}\n{FUNC_START}write>\n{PARAM_START}path>\na.txt\n{PARAM_END}\n")
        self.assertEqual(p.state, "call")
        p.buf = p.buf + f"{FUNC_END}\n{CALL_END}"
        events = p.feed("") + p.finish()
        calls = [e.call for e in events if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].arguments, {"path": "a.txt"})

        # Unterminated call flushes buffered parts as content on finish().
        p2 = OutputParser(thinking=False, tools=[SCHEMA], stream_tools=False)
        p2.feed(f"{CALL_START}\n{FUNC_START}write>\n")
        for _ in range(5):
            self.assertEqual(p2.feed("chunk "), [])
        ev = p2.finish()
        self.assertEqual([(e.kind, e.text) for e in ev],
                         [("content", f"{CALL_START}\n{FUNC_START}write>\n" + "chunk " * 5)])

    def test_stream_tools_event_by_event_parity_and_linear_scaling(self):
        import time
        from serve.frontend import Event, ToolCall, _text_kinds, _tool_properties, call_end, param_end

        class ReferenceOutputParser(OutputParser):
            """Pre-optimization reference using flat _buf and direct call_end/param_end on every feed()."""

            def _scan(self, final: bool = False):
                out = []

                def args(s):
                    if s:
                        out.append(Event("tool_args", s, call=self.scall))

                while True:
                    rest = self._buf[self.sp:]
                    if self.ss == "name":
                        a = rest.find("<function=")
                        b = rest.find(">", a + 10) if a >= 0 else -1
                        if b < 0:
                            return out
                        name = rest[a + 10:b]
                        self.scall = ToolCall(name=name, arguments={})
                        self.sdeclared = _tool_properties(self.schemas.get(name))
                        out.append(Event("tool_start", call=self.scall))
                        args("{")
                        self.sp += b + 1
                        self.ss = "between"
                    elif self.ss == "between":
                        stripped = rest.lstrip()
                        self.sp += len(rest) - len(stripped)
                        if stripped.startswith("<parameter="):
                            b = stripped.find(">")
                            if b < 0:
                                return out
                            pname = stripped[11:b]
                            args(("" if self.sfirst else ",") + json.dumps(pname) + ":")
                            self.sfirst = False
                            self.sp += b + 1
                            self.spname = pname
                            text, null = _text_kinds(self.sdeclared.get(pname))
                            if text and not null:
                                args('"')
                                self.ss, self.sval_started = "str", False
                            else:
                                self.ss = "raw"
                        elif stripped.startswith("</function>"):
                            args("}")
                            self.sp += len("</function>")
                            self.ss = "done"
                        else:
                            return out
                    elif self.ss == "str":
                        if not self.sval_started:
                            if not rest:
                                return out
                            if rest[0] == "\n":
                                self.sp += 1
                                rest = rest[1:]
                            self.sval_started = True
                            self.svstart = self.sp
                        end = param_end(self._buf[self.svstart:], final=final, name=self.spname) if PARAM_END in rest else -1
                        end = end - (self.sp - self.svstart) if end >= 0 else end
                        if end >= 0:
                            value = rest[:end]
                            if value.endswith("\n"):
                                value = value[:-1]
                            args(json.dumps(value)[1:-1] + '"')
                            self.sp += end + len(PARAM_END)
                            self.ss = "between"
                            continue
                        safe = rest.find(PARAM_END) if end == -2 else len(rest) - self._hold(rest, (PARAM_END,))
                        if safe > 0 and rest[safe - 1] == "\n":
                            safe -= 1
                        if safe > 0:
                            args(json.dumps(rest[:safe])[1:-1])
                            self.sp += safe
                        return out
                    elif self.ss == "raw":
                        end = param_end(rest, final=final, name=self.spname) if PARAM_END in rest else -1
                        if end < 0:
                            return out
                        value = rest[:end]
                        if value.startswith("\n"):
                            value = value[1:]
                        if value.endswith("\n"):
                            value = value[:-1]
                        from serve.frontend import _decode_param
                        args(json.dumps(_decode_param(value, self.sdeclared.get(self.spname)), ensure_ascii=False))
                        self.sp += end + len(PARAM_END)
                        self.ss = "between"
                    else:
                        return out

            def feed(self, delta: str, *, final: bool = False):
                self._buf += delta
                out = []
                while True:
                    if self.state != "call":
                        self._call_parts.clear()
                        return out + super().feed("", final=final)
                    i = call_end(self._buf, final=final) if CALL_END in self._buf else -1
                    if self.stream_tools:
                        if i >= 0:
                            whole, self._buf = self._buf, self._buf[:i + len(CALL_END)]
                            out += self._scan(final=True)
                            out += self._close_scan()
                            self._buf = whole
                        else:
                            out += self._scan(final=final)
                    if i < 0:
                        return out
                    body = self._buf[:i]
                    self._buf = self._buf[i + len(CALL_END):]
                    name = body.strip()[len("<function="):].split(">", 1)[0]
                    try:
                        call = parse_tool_call(body, self.schemas.get(name))
                    except ValueError:
                        out.append(Event("content", CALL_START + body + CALL_END))
                        self._reset_scan()
                        self.state, self.lead = "content", True
                        continue
                    if self.scall is not None:
                        call.id = self.scall.id
                    out.append(Event("tool_call", call=call))
                    self._reset_scan()
                    self.state, self.lead = "content", True

        def sig(evts):
            return [(e.kind, e.text, e.call.name if e.call else None, e.call.arguments if e.call else None)
                    for e in evts]

        raw_schema = {"name": "write", "parameters": {"properties": {
            "path": {"type": "string"}, "content": {"type": "string"}, "meta": {"type": "object"}}}}
        samples = [
            *[call_text(v) for v in VALUES.values()],
            call_text(VALUES["whole close inside"]) + "\n" + call_text("two"),
            call_text(VALUES["unclosed function with parameter"]) + "\nnow the second:\n" + call_text("two"),
            call_text("one") + "\nprose between\n" + call_text("two") + "\ndone.",
            f"plan\n</think>\n\n{CALL_START}\n{FUNC_START}write>\n{PARAM_START}meta>\n"
            f'{{"note": "a {PARAM_END} b {CALL_END} c", "nums": [1, 2, 3]}}\n{PARAM_END}\n'
            f"{PARAM_START}content>\nhello\n{PARAM_END}\n{FUNC_END}\n{CALL_END}\ntail",
            f"{CALL_START}\nnot a call\n{CALL_END}\nafter",
            f"{CALL_START}\n{FUNC_START}write>\n{PARAM_START}path>\na.txt\n{PARAM_END}\n{FUNC_END}",
        ]
        for idx, text in enumerate(samples):
            thinking = "</think>" in text
            for tools in ([raw_schema], None):
                for stream in (False, True):
                    for step in (1, 3, 7, 19, len(text)):
                        with self.subTest(idx=idx, tools=bool(tools), stream=stream, step=step):
                            fast = OutputParser(thinking=thinking, tools=tools, stream_tools=stream)
                            ref = ReferenceOutputParser(thinking=thinking, tools=tools, stream_tools=stream)
                            for pos in range(0, len(text), step):
                                chunk = text[pos:pos + step]
                                self.assertEqual(sig(fast.feed(chunk)), sig(ref.feed(chunk)))
                            self.assertEqual(sig(fast.finish()), sig(ref.finish()))

        # O(N) performance verification under stream_tools=True for 64 KiB raw JSON, quoted </tool_call>, and quoted whole-close.
        big_json = json.dumps({"items": ["x" * 16 for _ in range(3000)]})
        big_str = "abcd" * 16000
        cases = [
            (f"{CALL_START}\n{FUNC_START}write>\n{PARAM_START}meta>\n{big_json}\n{PARAM_END}\n{FUNC_END}\n{CALL_END}",
             [raw_schema], "raw_json"),
            (f"{CALL_START}\n{FUNC_START}write>\n{PARAM_START}content>\nquoted {CALL_END}\n{big_str}\n{PARAM_END}\n{FUNC_END}\n{CALL_END}",
             [raw_schema], "quoted_call_end"),
            (f"{CALL_START}\n{FUNC_START}write>\n{PARAM_START}content>\n{PARAM_END}\n{FUNC_END}\n{CALL_END}\n{big_str}\n{PARAM_END}\n{FUNC_END}\n{CALL_END}",
             [raw_schema], "quoted_whole_close"),
        ]
        for payload, tools, label in cases:
            with self.subTest(perf=label):
                p = OutputParser(thinking=False, tools=tools, stream_tools=True)
                t0 = time.monotonic()
                evts = []
                for pos in range(0, len(payload), 4):
                    evts.extend(p.feed(payload[pos:pos + 4]))
                evts.extend(p.finish())
                dt = time.monotonic() - t0
                calls = [e.call for e in evts if e.kind == "tool_call"]
                self.assertEqual(len(calls), 1)
                streamed = "".join(e.text for e in evts if e.kind == "tool_args")
                self.assertEqual(json.loads(streamed), calls[0].arguments)
                self.assertLess(dt, 2.0, f"{label} took {dt:.2f}s (expected O(N) < 2s)")


if __name__ == "__main__":
    unittest.main()
