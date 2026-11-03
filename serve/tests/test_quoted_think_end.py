"""A `</think>` the reasoning quotes stays reasoning; the real end is followed by a line break."""
import unittest

from serve.frontend import OutputParser

WIDTHS = (1, 2, 3, 7, 100_000)
THINK_END = "</think>"


def run(text, width):
    p = OutputParser(thinking=True)
    out = []
    for i in range(0, len(text), width):
        out += p.feed(text[i:i + width])
    return out + p.finish()


def joined(evs, kind):
    return "".join(e.text for e in evs if e.kind == kind)


class QuotedThinkEnd(unittest.TestCase):
    def test_a_quoted_tag_stays_reasoning(self):
        for quote in ("a `</think>` could arrive here",
                      'the engine returns "</think>\\n\\n2", no call',
                      "split it on '</think>' first"):
            text = "plan\n" + quote + "\ndone\n</think>\n\nanswer"
            for width in WIDTHS:
                with self.subTest(quote=quote[:20], width=width):
                    evs = run(text, width)
                    self.assertEqual(joined(evs, "reasoning"), "plan\n" + quote + "\ndone\n")
                    self.assertEqual(joined(evs, "content"), "answer")

    def test_the_end_after_a_quote_still_ends_the_thinking(self):
        for thought in ('she said "yes"', "it is `x`", "it is 'x'"):
            for width in WIDTHS:
                with self.subTest(thought=thought, width=width):
                    evs = run(thought + "</think>\n\nanswer", width)
                    self.assertEqual(joined(evs, "reasoning"), thought)
                    self.assertEqual(joined(evs, "content"), "answer")

    def test_an_output_that_ends_on_the_tag(self):
        for width in WIDTHS:
            evs = run('say "</think>', width)
            self.assertEqual(joined(evs, "reasoning"), 'say "')
            self.assertEqual(joined(evs, "content"), "")

    def test_an_unquoted_tag_ends_the_thinking(self):
        for width in WIDTHS:
            evs = run("plan</think>answer", width)
            self.assertEqual(joined(evs, "reasoning"), "plan")
            self.assertEqual(joined(evs, "content"), "answer")


def split(text, width):
    """(reasoning, content) of a run at this width."""
    evs = run(text, width)
    return joined(evs, "reasoning"), joined(evs, "content")


class MentionCase(unittest.TestCase):
    def check(self, text, reasoning, content):
        for width in WIDTHS:
            with self.subTest(width=width):
                r, c = split(text, width)
                self.assertEqual(r, reasoning)
                self.assertEqual(c, content)


class EscapedMention(MentionCase):
    """A tag the model writes about: its quote sits a character or two back, not right before the tag."""

    def test_a_typed_escape_before_the_tag_is_a_mention(self):
        # the live turn: the model writes its own closing tag, escaped, inside a backtick span
        mention = "the marker `\\n" + THINK_END + "\\n\\n` appears once"
        self.check("plan\n" + mention + "\ndone\n" + THINK_END + "\n\nanswer",
                   "plan\n" + mention + "\ndone\n", "answer")

    def test_the_same_shape_in_double_quotes(self):
        mention = 'the marker "\\n' + THINK_END + '\\n\\n" appears once'
        self.check("plan\n" + mention + "\ndone\n" + THINK_END + "\n\nanswer",
                   "plan\n" + mention + "\ndone\n", "answer")

    def test_a_real_end_after_a_typed_escape_and_a_line_break(self):
        # the escaped shape is held for the character after the tag; a line break is the real end
        self.check("the flag `\\n" + THINK_END + "\n\nanswer", "the flag `\\n", "answer")

    def test_a_real_end_in_the_escaped_shape_without_a_line_break(self):
        # known limit: an answer that starts immediately reads as part of the mention, so the whole
        # turn stays reasoning. Pinned so a change to it is deliberate.
        text = "the flag `\\n" + THINK_END + "answer"
        self.check(text, text, "")

    def test_an_unquoted_mention_still_ends_the_thinking(self):
        # known limit: with no quote in the window this is the shape of a real end. Pinned as it is.
        self.check("the marker \\n" + THINK_END + "\\n appears\n" + THINK_END + "\n\nanswer",
                   "the marker \\n", "\\n appears\n" + THINK_END + "\n\nanswer")


class QuoteBeforeTheTag(MentionCase):
    """Which side of the tag a quote is on decides whether it quotes the tag or closes a sentence."""

    def test_a_closing_quote_before_the_tag_is_a_real_end(self):
        self.check('The user said "only."' + THINK_END + "2", 'The user said "only."', "2")

    def test_an_opening_quote_before_the_tag_quotes_it(self):
        mention = 'and he wrote "' + THINK_END + '" in the log'
        self.check("plan\n" + mention + "\ndone\n" + THINK_END + "\n\nanswer",
                   "plan\n" + mention + "\ndone\n", "answer")

    def test_a_tag_after_an_opening_quote_with_prose_behind_it(self):
        mention = 'he said "' + THINK_END + ' again" and more'
        self.check("plan\n" + mention + "\ndone\n" + THINK_END + "\n\nanswer",
                   "plan\n" + mention + "\ndone\n", "answer")


if __name__ == "__main__":
    unittest.main()
