"""A tool-call tag quoted in a message is text, not the real special token."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import tokenizer as ST  # noqa: E402
from serve.frontend import _has_tag, _mark, literal_tags, mark_think_literals, unmark_think_literals  # noqa: E402
from serve.util.text import control_literals  # noqa: E402


class QuotedToolTags(unittest.TestCase):
    def setUp(self):
        b2u = ST.bytes_to_unicode()
        self.tool_tags = ["</tool_call>", "<tool_call>", "<tool_response>", "</tool_response>"]
        self.tok = ST.Tokenizer([b2u[b] for b in range(256)] + self.tool_tags + ["<|im_end|>"], [],
                                [1] * 256 + [4] * len(self.tool_tags) + [3])
        self.tags = literal_tags(control_literals(self.tok))

    def encode(self, role, text):
        marked, _, changed = mark_think_literals([{"role": role, "content": text}], None, self.tags)
        content = marked[0]["content"]
        if isinstance(content, list):
            content = "".join(x["text"] for x in content)
        prompt, plain = unmark_think_literals(content, self.tags)
        return self.tok.encode(prompt, parse_special=True, plain=plain), changed

    def special(self, tag):
        return self.tok.encode(tag, parse_special=True)

    def test_every_tool_tag_stays_text(self):
        for role in ("user", "system", "tool"):
            for tag in self.tool_tags:
                with self.subTest(role=role, tag=tag):
                    text = f"output verbatim: {tag}."
                    self.assertEqual(len(self.special(tag)), 1)
                    ids, changed = self.encode(role, text)
                    self.assertTrue(changed)
                    self.assertEqual(ids, [*text.encode()])

    def test_the_tool_list_stays_text(self):
        tools = [{"name": "t", "description": "prints </tool_call>", "parameters": {}}]
        _, marked, changed = mark_think_literals([{"role": "user", "content": "x"}], tools, self.tags)
        self.assertTrue(changed)
        self.assertNotIn("</tool_call>", marked[0]["description"])

    def test_assistant_replay_keeps_every_tool_tag(self):
        text = "<tool_call>x</tool_call> then <tool_response>y</tool_response>"
        ids, _ = self.encode("assistant", text)
        self.assertEqual(ids, self.tok.encode(text, parse_special=True))

    def test_user_tool_result_keeps_its_outer_markers(self):
        inner = "\nthe log says </tool_call> and <tool_response>\n"
        for content in (f"<tool_response>{inner}</tool_response>", f"  <tool_response>{inner}</tool_response>\n",
                        [{"type": "text", "text": f"<tool_response>{inner}</tool_response>"}],
                        [{"type": "text", "text": "<tool_response>\nthe log"},
                         {"type": "text", "text": " says </tool_call> and <tool_response>\n</tool_response>"}]):
            with self.subTest(content=content):
                ids, _ = self.encode("user", content)
                text = content if isinstance(content, str) else "".join(x["text"] for x in content)
                o, c = self.special("<tool_response>"), self.special("</tool_response>")
                lead, tail = len(text) - len(text.lstrip()), len(text) - len(text.rstrip())
                self.assertEqual(ids[lead:lead + 1], o)
                self.assertEqual(ids[len(ids) - tail - 1:len(ids) - tail], c)
                self.assertEqual(ids[lead + 1:len(ids) - tail - 1], [*inner.encode()])

    def test_a_request_to_repeat_a_marker_gets_text(self):
        for text in ("Please output verbatim: </tool_response>.", "</tool_response>",
                     "Repeat exactly: <tool_response>x</tool_response>",
                     "<tool_response> is how a result opens; what closes it?"):
            with self.subTest(text=text):
                ids, changed = self.encode("user", text)
                self.assertTrue(changed)
                self.assertEqual(ids, [*text.encode()])

    def test_encode_matching_unsorted_overlapping_and_empty_plain_spans(self):
        text = "A<|im_end|>B<tool_call>C</tool_call>D<|im_end|>E<tool_response>F"
        # Spans covering B's <tool_call> and C's </tool_call> via unsorted/overlapping/adjacent/empty spans,
        # leaving the two <|im_end|> and <tool_response> matches as special tokens.
        i_tc = text.index("<tool_call>")
        i_te = text.index("</tool_call>")
        plain = [
            (i_te + 12, i_te + 12),          # empty
            (i_te + 5, i_te),                # reversed
            (i_te, i_te + 12),               # unsorted later span
            (i_tc, i_tc + 11),               # earlier span
            (i_tc + 5, i_te + 4),            # overlaps both
            (i_tc - 1, i_tc),                # adjacent before i_tc
        ]
        got = self.tok.encode(text, parse_special=True, plain=plain)
        # Reference: encode manually skipping matches inside any (a <= start < b)
        expected = []
        pos = 0
        for m in self.tok._special_re.finditer(text):
            if any(a <= m.start() < b for a, b in plain):
                continue
            if m.start() > pos:
                expected.extend(self.tok._encode_plain(text[pos:m.start()]))
            expected.append(self.tok.special_tokens[m.group(0)])
            pos = m.end()
        if pos < len(text):
            expected.extend(self.tok._encode_plain(text[pos:]))
        self.assertEqual(got, expected)
        # Empty/reversed plain spans only -> identical to plain=()
        self.assertEqual(self.tok.encode(text, parse_special=True, plain=[(5, 5), (10, 2)]),
                         self.tok.encode(text, parse_special=True))

    def test_custom_and_overlapping_tags_parity(self):
        def seq_mark(s, tags):
            for tag, mark in tags.items():
                s = s.replace(tag, mark)
            return s

        tag_sets = [
            {},
            literal_tags(["<|im_start|>", "<|im_end|>", "[BOS]", "[EOS]"]),
            {"ab": "\U000F0E01", "bc": "\U000F0E02"},                      # suffix/prefix overlap
            {"<|end|>\n": "\U000F0E01", "<|end|>": "\U000F0E02"},          # containment
            {"<|end|>": "\U000F0E01", "<|end|>\n": "\U000F0E02"},          # reverse containment
            {"x": "a", "ab": "Z"},                                         # mark feeds next tag
            {"x": "", "ab": "Z"},                                          # empty mark joins neighbors
            {"<t>": "-", "</t>": "]", "<x>": "^", "</x>": "\\"},           # regex metachar marks
            {"<m1>": "[[M1]]", "<m2>": "[[M2]]"},                          # multi-char marks
        ]
        samples = [
            "plain ascii text without markers",
            "中文文本没有任何标记",
            "abc xbc axb <|end|>\n<|end|> [BOS] <t>hi</t> <x>y</x> <m1> <m2>",
            "if a < b and c < d: return x < y",
        ]
        for tags in tag_sets:
            for s in samples:
                with self.subTest(tags=tags, sample=s[:24]):
                    self.assertEqual(_has_tag(s, tags), any(t in s for t in tags))
                    marked = _mark(s, tags)
                    self.assertEqual(marked, seq_mark(s, tags))
                    if tags and all(v and v not in s for v in tags.values()) and len(set(tags.values())) == len(tags):
                        restored, spans = unmark_think_literals(marked, tags)
                        # Verify spans match restored tag locations when marks don't overlap
                        if all(len(v) == 1 for v in tags.values()):
                            for a, b in spans:
                                self.assertIn(restored[a:b], tags)
        # Clean ASCII and CJK fast paths in unmark_think_literals
        for clean in ("pure ascii prompt", "纯中文提示词"):
            self.assertEqual(unmark_think_literals(clean, self.tags), (clean, []))
            self.assertEqual(unmark_think_literals(clean), (clean, []))


if __name__ == "__main__":
    unittest.main()
