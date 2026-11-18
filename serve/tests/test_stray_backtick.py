"""A stray backtick in the visible answer must not turn the rest of its paragraph into inline code and hide the
`<tool_call>` written after it.  A span that genuinely stays open - a wrapped `example - still quotes what follows."""
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
        """The shape that lost the call: "let me use `write to save" leaves one backtick behind, and counting the
        paragraph's backticks read every later line as inline code."""
        self.run_all("Let me use `write to save it.\n\n" + CALL, ("write",))

    def test_a_stray_on_an_earlier_line_of_the_paragraph_keeps_the_call(self):
        self.run_all("I have a `typo here and keep talking,\nthen I do it.\n\n" + CALL, ("write",))

    def test_two_strays_do_not_hide_a_call(self):
        """Two strays leave the count even, which is why the old count happened to work here - it must keep working."""
        self.run_all("The `first and the `second, then:\n\n" + CALL, ("write",))

    def test_a_wrapped_span_closed_on_a_later_line_keeps_the_call(self):
        """A run that ENDS its line opens a span; its closing run ends a line too, and the paragraph is paired again."""
        self.run_all("a `span that stays\nopen until here`\n\n" + CALL, ("write",))

    def test_a_span_still_open_quotes_the_call(self):
        """The other half of the rule: the last run ended its line and nothing closed it, so the call is an example."""
        self.run_all("look at this `\n" + CALL + "\n", ())

    def test_a_call_in_an_indented_block_stays_text(self):
        """Four spaces after a blank line is a code block; a call written in one is an example."""
        self.run_all("The format is:\n\n    " + CALL.replace("\n", "\n    ") + "\n\nDone.", ())

    def test_a_longer_fence_holds_a_shorter_one(self):
        """A fence closes only against a line of at least as many of the same character (CommonMark), so a call in a
        ```` example that shows ``` fences stays text until the long fence closes."""
        self.run_all("````\n" + CALL + "\n````\n\nDone.", ())

    def test_a_call_after_a_closed_fence_still_fires(self):
        self.run_all("```\ncode\n```\n\n" + CALL, ("write",))

    def test_a_normal_inline_span_before_a_call_is_unchanged(self):
        """The reading a model is written for: `a` and `b` pair up, the call after them is a call."""
        self.run_all("`a` and `b` " + CALL, ("write",))


if __name__ == "__main__":
    unittest.main()
