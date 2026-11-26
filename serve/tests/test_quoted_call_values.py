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


if __name__ == "__main__":
    unittest.main()
