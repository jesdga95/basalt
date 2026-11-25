"""A stray backtick in the visible answer must not turn the rest of its paragraph into inline code and hide the
`<tool_call>` written after it. A span that genuinely stays open - a wrapped `example - still quotes what follows."""
from __future__ import annotations

import random
import unittest

from serve.frontend import OutputParser

CALL = "<tool_call>\n<function=write>\n<parameter=path>\nfile.txt\n</parameter>\n</function>\n</tool_call>"
SCHEMA = [{"name": "write", "parameters": {"properties": {"path": {"type": "string"}}}}]
HEAD = "ok</think>\n\n"                       # the thinking ends; what follows is the visible answer
WIDTHS = (1, 2, 3, 7, 0, 100_000)


def parse(text, width, stream_tools=True):
    """Events for `text` fed `width` characters at a time (0 = random 1-9, seeded)."""
    p = OutputParser(thinking=True, tools=SCHEMA, stream_tools=stream_tools)
    evs, rnd, i = [], random.Random(7), 0
    while i < len(text):
        k = width or rnd.randint(1, 9)
        evs += p.feed(text[i:i + k])
        i += k
    return evs + p.finish()


def calls(evs):
    return [e.call.name for e in evs if e.kind == "tool_call"]


def text_of(evs):
    return "".join(e.text or "" for e in evs if e.kind == "content")


class StrayBacktick(unittest.TestCase):
    def run_all(self, body, want):
        """The same answer under every chunk width: `want` calls, and a quoted call lost from nothing."""
        seen = set()
        for width in WIDTHS:
            for st in (True, False):
                evs = parse(HEAD + body, width, st)
                seen.add((tuple(calls(evs)), text_of(evs)))
        self.assertEqual(len(seen), 1, seen)               # every chunking agrees
        got, text = seen.pop()
        self.assertEqual(got, want)
        if not want:
            self.assertEqual(text, body)                   # quoted: the answer keeps the call as text

    def test_a_stray_backtick_before_a_call_keeps_the_call(self):
        """Same-paragraph stray backtick (single newline, no blank line): must not hide the call."""
        self.run_all("Let me use `write to save it.\n" + CALL, ("write",))

    def test_a_stray_on_an_earlier_line_of_the_paragraph_keeps_the_call(self):
        """Stray backtick on an earlier line of the same paragraph (no blank line before CALL)."""
        self.run_all("I have a `typo here and keep talking,\nthen I do it.\n" + CALL, ("write",))

    def test_two_strays_do_not_hide_a_call(self):
        """Two strays leave the count even, which is why the old count happened to work here - it must keep working."""
        self.run_all("The `first and the `second, then:\n" + CALL, ("write",))

    def test_a_wrapped_span_closed_on_a_later_line_keeps_the_call(self):
        """A run that ENDS its line opens a span; its closing run ends a line too, and the paragraph is paired again."""
        self.run_all("a `span that stays\nopen until here`\n" + CALL, ("write",))

    def test_a_span_still_open_quotes_the_call(self):
        """The other half of the rule: the last run ended its line and nothing closed it, so the call is an example."""
        self.run_all("look at this `\n" + CALL + "\n", ())

    def test_a_call_in_an_indented_block_stays_text(self):
        """Four spaces after a blank line is a code block; a call written in one is an example."""
        self.run_all("The format is:\n\n    " + CALL.replace("\n", "\n    ") + "\n\nDone.", ())

    def test_a_longer_fence_holds_a_shorter_one(self):
        """A 4-backtick fence containing an inner ``` line must not close early and fire the quoted call."""
        self.run_all("````\n```\n" + CALL + "\n````\n\nDone.", ())

    def test_a_six_backtick_fence_opens_and_quotes_the_call(self):
        """Fences of 6+ backticks (or tildes) must open properly even though s[3:] starts with backticks."""
        self.run_all("``````\n```\n" + CALL + "\n``````\n\nDone.", ())
        self.run_all("~~~~~~\n~~~\n" + CALL + "\n~~~~~~\n\nDone.", ())

    def test_a_fence_line_with_info_string_does_not_close_an_open_fence(self):
        """Per CommonMark, a closing fence line cannot have trailing non-whitespace text (info string)."""
        self.run_all("```\n```python\n" + CALL + "\n```\n\nDone.", ())

    def test_a_call_after_a_closed_fence_still_fires(self):
        self.run_all("```\ncode\n```\n\n" + CALL, ("write",))

    def test_a_normal_inline_span_before_a_call_is_unchanged(self):
        """The reading a model is written for: `a` and `b` pair up, the call after them is a call."""
        self.run_all("`a` and `b` " + CALL, ("write",))


if __name__ == "__main__":
    unittest.main()
