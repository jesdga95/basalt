"""The tokenizer's piece memo changes speed, never ids; the files it extracts are the same bytes on every OS.

    python3 -m unittest tools.test_tokenizer
    BASALT_TOKENIZER_DIR=<pack>/tokenizer python3 -m unittest tools.test_tokenizer   # also the real vocabulary
"""
import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tokenizer as T  # noqa: E402

CORPUS = [
    "", "Hello, world!", "  leading and trailing  ", "a\n\n\nb", "def f(x):\n\treturn x  # comment\n",
    "你好，世界", "مرحبا", "\U0001f600\U0001f680\U0001f1fa\U0001f1f8",
    "é́ combining", "\x00\x01\x7f control", "1234567890", "MixedCASE_and-dashes", "\r\n\r\n",
    "it's we're I'll they'd", "x" * 500,
    "<|im_start|>system\nYou are an agent.<|im_end|>\n<|im_start|>user\nread a.py<|im_end|>\n",
    "<|im_start|>assistant\n<think>\nI should read the file.\n</think>\n<tool_call>\n<function=read>\n"
    "<parameter=path>a.py</parameter>\n</function>\n</tool_call><|im_end|>\n",
    "<tool_response>\n{\"ok\": true, \"lines\": [1, 2, 3]}\n</tool_response>",
    "int main(int argc, char** argv) {\n    for (int i = 0; i < argc; ++i) printf(\"%s\\n\", argv[i]);\n}\n",
]


def synthetic(texts, n_merges=300, extra=("<think>", "</think>", "<tool_call>", "</tool_call>")):
    """A small byte-level BPE trained on `texts`, with CONTROL and USER_DEFINED literals."""
    import regex
    pre = regex.compile(T.QWEN35_PATTERN)
    words = Counter()
    for t in texts:
        for p in pre.findall(t):
            words[tuple("".join(T.BYTE_TO_UNICODE[b] for b in p.encode()))] += 1
    tokens = [T.BYTE_TO_UNICODE[b] for b in range(256)]
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for w, c in words.items():
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += c
        if not pairs:
            break
        (a, b), _c = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        merges.append("%s %s" % (a, b))
        tokens.append(a + b)
        new = {}
        for w, c in words.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            new[tuple(out)] = new.get(tuple(out), 0) + c
        words = new
    types = [1] * len(tokens)
    for lit, ty in (("<|im_start|>", 3), ("<|im_end|>", 3)) + tuple((e, 4) for e in extra):
        tokens.append(lit)
        types.append(ty)
    return tokens, merges, types


def make(tokens, merges, types, cap=None):
    tk = T.Tokenizer(tokens, merges, types)
    if cap is not None:
        tk.PIECE_CACHE_MAX = cap
    return tk


class PieceMemo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = synthetic(CORPUS * 2)
        real = os.environ.get("BASALT_TOKENIZER_DIR")
        cls.real = None
        if real:
            d = Path(real)
            vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
            toks = [None] * len(vocab)
            for t, i in vocab.items():
                toks[i] = t
            cls.real = (toks, (d / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((d / "token_type.json").read_text(encoding="utf-8")))

    def check_identical(self, data):
        plain = make(*data, cap=0)          # a full memo: nothing is remembered
        memo = make(*data)
        for rounds in range(2):             # cold, then warm
            for s in CORPUS + ["".join(CORPUS)]:
                for special in (False, True):
                    self.assertEqual(memo.encode(s, parse_special=special), plain.encode(s, parse_special=special), s[:40])
        self.assertEqual(plain._piece_ids, {})
        self.assertGreater(len(memo._piece_ids), 0)
        for s in CORPUS:
            self.assertEqual(memo.decode(memo.encode(s)), s)

    def test_identical_synthetic(self):
        self.check_identical(self.data)

    def test_identical_real_vocabulary(self):
        if self.real is None:
            self.skipTest("set BASALT_TOKENIZER_DIR to also check the real vocabulary")
        self.check_identical(self.real)

    def test_plain_span_and_resend(self):
        memo, plain = make(*self.data), make(*self.data, cap=0)
        s = "<think>x</think> <tool_call>"
        span = [(0, 7)]
        self.assertEqual(memo.encode(s, plain=span), plain.encode(s, plain=span))
        conv = "".join(CORPUS)
        first = memo.encode(conv, parse_special=True)
        self.assertEqual(memo.encode(conv + "more", parse_special=True)[:len(first) - 1], first[:-1])

    def test_bounded(self):
        memo = make(*self.data, cap=5)
        plain = make(*self.data, cap=0)
        for s in CORPUS:
            self.assertEqual(memo.encode(s), plain.encode(s))
        self.assertLessEqual(len(memo._piece_ids), 5)

    def test_cached_lists_not_aliased(self):
        memo = make(*self.data)
        a = memo.encode("hello world")
        a.append(-1)
        self.assertEqual(memo.encode("hello world")[-1] == -1, False)


class Extract(unittest.TestCase):
    def test_line_endings(self):
        # the engine splits merges.txt on LF: a CR left at a line's end would name a token outside the vocabulary
        tokens, merges, types = synthetic(CORPUS)
        md = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.tokens": tokens, "tokenizer.ggml.merges": merges,
              "tokenizer.ggml.token_type": types, "tokenizer.chat_template": "{{ a }}\n{{ b }}\n"}
        with tempfile.TemporaryDirectory() as d:
            T.extract_metadata(md, d)
            for name in ("merges.txt", "tokenizer.json", "chat_template.jinja"):
                data = (Path(d) / "tokenizer" / name).read_bytes()
                self.assertTrue(b"\n" in data and b"\r" not in data, "%s: lines do not end in LF alone" % name)


class PreTokenizer(unittest.TestCase):
    """The pre-tokenizer a pack names is the one it gets: the wrong regex splits a prompt differently from the
    reference engine, and a round trip of the pieces back to text cannot show that."""

    def make(self, pre):
        tokens, merges, types = synthetic(CORPUS)
        return T.Tokenizer(tokens, merges, types, pre=pre)

    def test_glm4_split(self):
        tk = self.make("glm4")
        self.assertTrue(tk.ignore_merges)                                     # llama.cpp gives CHATGLM4 ignore_merges
        self.assertEqual(tk._split("1234567890"), ["123", "456", "789", "0"])   # digit runs of at most three
        self.assertEqual(tk._split("it's here"), ["it", "'s", " here"])         # the contraction stays its own piece
        self.assertEqual(tk._split("  blank  "), [" ", " blank", "  "])         # a space run splits at the word

    def test_glm5_is_the_same_pre_tokenizer(self):
        glm4, glm5 = self.make("glm4"), self.make("glm5")
        self.assertTrue(glm5.ignore_merges)
        for text in CORPUS:
            self.assertEqual(glm4._split(text), glm5._split(text), text[:20])

    def test_a_pack_names_the_pattern_it_split_with(self):
        tokens, merges, types = synthetic(CORPUS)
        md = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.tokens": tokens, "tokenizer.ggml.merges": merges,
              "tokenizer.ggml.token_type": types, "tokenizer.ggml.pre": "glm4"}
        with tempfile.TemporaryDirectory() as d:
            cfg = T.extract_metadata(md, d)
        self.assertEqual(cfg["pre"], "glm4")
        self.assertEqual(cfg["pre_patterns"], T.GLM4_PATTERNS)
        self.assertIn("CHATGLM4", cfg["pre_pattern_source"])


if __name__ == "__main__":
    unittest.main()
