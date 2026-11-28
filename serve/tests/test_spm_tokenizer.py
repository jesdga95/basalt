"""The Python tokenizer's SentencePiece mode (tools/tokenizer.py, a model whose pre-tokenizer is in SENTENCEPIECE) on a
small vocabulary: pieces start from characters or their bytes' fallback tokens, U+2581 reads as a space, merges by rank,
and fallback tokens decode to their bytes. No pack."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from tokenizer import BYTE_TO_UNICODE, Tokenizer  # noqa: E402


def spell(text: str) -> str:
    return "".join(BYTE_TO_UNICODE[b] for b in text.encode("utf-8"))


def vocab():
    tokens = ["<pad>", "<|turn>"] + [f"<0x{b:02X}>" for b in range(256)]
    types = [3, 3] + [6] * 256
    for t in (" ", "a", "b", "é", "ab", " a", " ab"):
        tokens.append(spell(t))
        types.append(1)
    merges = [f"{spell(' ')} {spell('a')}", f"{spell(' a')} {spell('b')}", f"{spell('a')} {spell('b')}"]
    return Tokenizer(tokens, merges, types, "gemma4")


class SentencePiece(unittest.TestCase):
    def setUp(self):
        self.tk = vocab()

    def ids(self, *texts):
        return [self.tk.ids[spell(t)] for t in texts]

    def test_merges_from_characters(self):
        self.assertEqual(self.tk.encode(" ab"), self.ids(" ab"))
        self.assertEqual(self.tk.encode("aé ab"), self.ids("a", "é", " ab"))

    def test_u2581_is_a_space(self):
        self.assertEqual(self.tk.encode("▁ab"), self.tk.encode(" ab"))

    def test_byte_fallback(self):
        ids = self.tk.encode("ü")
        self.assertEqual([self.tk.tokens[i] for i in ids], ["<0xC3>", "<0xBC>"])
        self.assertEqual(self.tk.decode(ids), "ü")
        self.assertEqual(self.tk.decode(self.tk.encode("a ü ab")), "a ü ab")

    def test_specials_split_pieces(self):
        ids = self.tk.encode("ab<|turn>ab", parse_special=True)
        self.assertEqual(ids, self.ids("ab") + [1] + self.ids("ab"))


if __name__ == "__main__":
    unittest.main()
