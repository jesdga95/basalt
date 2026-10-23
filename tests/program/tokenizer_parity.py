#!/usr/bin/env python3
"""tokenizer_parity: the C++ tokenizer must give tools/tokenizer.py's ids, exactly, over a corpus.

usage: tokenizer_parity.py <tokenizer_test binary> <tokenizer dir of a pack> [--big FILE ...]

The golden ids are made here, with the reference tokenizer, from the pack's vocab.json, merges.txt and token_type.json.
Needs the `regex` package; without it the test is skipped (exit 77).
"""
import json
import os
import pathlib
import random
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))


def load_reference(tdir: pathlib.Path):
    import tokenizer as ST
    vocab = json.loads((tdir / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for k, i in vocab.items():
        tokens[i] = k
    merges = (tdir / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tdir / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)


def corpus(tk, big):
    cases = []

    def add(name, text, special=False, plain=()):
        cases.append((name, text, special, list(plain)))

    base = ["", "Hello, world!", "  leading and trailing  ", "a\n\n\nb", "def f(x):\n\treturn x  # comment\n",
            "你好，世界", "مرحبا", "\U0001f600\U0001f680\U0001f1fa\U0001f1f8",
            "é́ combining", "\x00\x01\x7f control", " non-breaking space", "1234567890",
            "MixedCASE_and-dashes", "\r\n\r\n", "  ", "x" * 5000, "it's I'M YOU'RE we'VE they'LL he'd 'S 'x",
            "กินข้าว", "हिन्दी कार", "á!́b",
            "!́?", "\n \n  \n   x", "x   \n", "x   ", "   ", "\t\t\t", "\r\r\n\n\r", " 　 x", "\u001c\u001d\u001e\u001f x",
            "​‌‍⁠ zero width", "﻿bom", "퟿￿".replace("￿", "�"),
            "\U0010ffff\U0001fffe", "1²Ⅰ٣ numbers", "a" * 63 + "b", "ab" * 40, "ab" * 100, "abc " * 300,
            "word" * 1500, "你" * 400, "The quick brown fox " * 200]
    for i, s in enumerate(base):
        add("base%d" % i, s)
        add("base%d_sp" % i, s, True)
    specials = sorted(tk.special_tokens)
    for lit in specials:
        add("lit_%s" % lit, "x" + lit + "y")
        add("lit_%s_sp" % lit, "x" + lit + "y", True)
        add("lit_%s_plain" % lit, "x" + lit + "y", True, [(1, 1 + len(lit))])
        add("lit_%s_dbl" % lit, lit + lit, True)
        add("lit_%s_edge" % lit, lit[:-1] + lit, True)
    chat = "<|im_start|>user\nHi <think>x</think> <tool_call>{}</tool_call><|im_end|>\n<|im_start|>assistant\n"
    add("chat", chat)
    add("chat_sp", chat, True)
    add("chat_plain1", chat, True, [(0, 14)])
    add("chat_plain2", chat, True, [(0, 14), (24, 31), (40, 60)])
    add("chat_plain_all", chat, True, [(0, len(chat))])
    rng = random.Random(1)
    pools = [
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789٣²Ⅰ", " \t\n\r  　\u0085\u001c",
        "'.,;:!?-_/\\()[]{}<>|@#$%^&*+=~`\"", "éñüßőαβджאام",
        "̀́̂ःिัิ‍‌", "你好世界あア가", "\U0001f600\U0001f680❤️✅",
        "  ​﻿­", "힣\U000e0001\U0001d7ce", "<|im_start|>", "<think>", "</think>", "'s", "'re"]
    for n in range(400):
        k = rng.choice([3, 8, 20, 60, 200])
        parts = [rng.choice(rng.choice(pools)) if rng.random() < 0.8 else rng.choice(pools[-4:]) for _ in range(k)]
        s = "".join(parts)
        add("fuzz%d" % n, s)
        add("fuzz%d_sp" % n, s, True)
    for p in sorted((ROOT / "src").rglob("*.cpp"))[:40]:
        add("src_" + p.name, p.read_text(encoding="utf-8", errors="replace"), True)
    for p in sorted((ROOT / "serve").rglob("*.py"))[:20]:
        add("py_" + p.name, p.read_text(encoding="utf-8", errors="replace"))
    for f in big:
        add("big_" + pathlib.Path(f).name, pathlib.Path(f).read_text(encoding="utf-8"), True)
    return cases


def main():
    exe, tdir = sys.argv[1], pathlib.Path(sys.argv[2])
    big = sys.argv[4:] if len(sys.argv) > 3 and sys.argv[3] == "--big" else []
    try:
        import regex  # noqa: F401
    except ImportError:
        print("the regex package is missing")
        return 77
    tk = load_reference(tdir)
    cases = corpus(tk, big)
    with tempfile.TemporaryDirectory() as d:
        g = pathlib.Path(d) / "golden.bin"
        with open(g, "wb") as f:
            for name, text, special, plain in cases:
                ids = tk.encode(text, parse_special=special, plain=plain)
                raw = text.encode("utf-8")
                spans = ",".join("%d:%d" % (len(text[:a].encode()), len(text[:b].encode())) for a, b in plain)
                f.write(("CASE %s %d %d %d %d\n" % (name.replace(" ", "_"), int(special), len(raw), len(ids), len(plain))).encode())
                f.write(raw + b"\n")
                f.write((spans + "\n").encode())
                f.write((",".join(map(str, ids)) + "\n").encode())
        r = subprocess.run([exe, str(tdir), str(g)])
    print("%d cases" % len(cases))
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
