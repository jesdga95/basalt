"""The reference byte-level BPE tokenizer, read back from the model's `tokenizer/` directory.

The GGUF metadata carries the whole tokenizer (gpt2 model, tokens, merges, token types, `qwen35` pre-tokenizer,
chat template), so no weights are needed. Any byte maps to a printable character, so there is no UNK token, and
`decode(encode(s)) == s` is the check that catches byte-offset, merge-order and pre-tokenizer mistakes.
"""
from __future__ import annotations

import json
import pathlib
import sys

import regex

# ------------------------------------------------------------------ byte <-> unicode (GPT-2 byte-level BPE)
def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte encoder: bytes 33..126, 161..172, 174..255 map to themselves, the other 68 are shifted to
    256+n in order; a wrong order is invisible in the ids."""
    bs = (list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_TO_UNICODE = bytes_to_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}

# The `qwen35` pre-tokenizer, transcribed from llama.cpp's LLAMA_VOCAB_PRE_TYPE_QWEN35. Unlike the QWEN3 pattern it
# keeps `\p{M}` in the letter class and out of the punctuation class; without that a combining mark is swallowed into
# the next punctuation run. Round trips cannot see the difference; only an oracle comparison can.
QWEN35_PATTERN = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


class Tokenizer:
    def __init__(self, tokens: list[str], merges: list[str], token_types: list[int] | None = None,
                 pre: str = "qwen35", special_ids: dict[str, int] | None = None):
        self.tokens = tokens
        self.pre = pre
        self.token_types = token_types
        self.special_ids = special_ids or {}
        self.ids = {t: i for i, t in enumerate(tokens)}
        if len(self.ids) != len(tokens):
            raise ValueError("vocabulary has duplicate tokens: %d entries, %d unique"
                             % (len(tokens), len(self.ids)))
        # merges as (left, right) -> rank; a non-total order or an unknown token would silently change ids, so check
        self.ranks: dict[tuple[str, str], int] = {}
        for i, m in enumerate(merges):
            parts = m.split(" ")
            if len(parts) != 2:
                raise ValueError("merge %d is not a pair: %r" % (i, m))
            if parts[0] not in self.ids or parts[1] not in self.ids:
                raise ValueError("merge %d names a token outside the vocabulary: %r" % (i, m))
            self.ranks[(parts[0], parts[1])] = i
        self._re = regex.compile(QWEN35_PATTERN)

        # Literals matched directly instead of through BPE (GGUF token types):
        #   * type 4 (USER_DEFINED: `<think>`, `<tool_call>`, ...) always matches, with or without parse_special;
        #   * type 3 (CONTROL: `<|im_start|>`, `<|im_end|>`, ...) matches only when parse_special is set.
        # This mirrors llama.cpp; treating them alike changes chat prompts.
        self.special_tokens: dict[str, int] = {}
        if token_types:
            for i, ty in enumerate(token_types):
                if ty in (3, 4):
                    self.special_tokens[tokens[i]] = i
        always = [t for t, i in self.special_tokens.items() if token_types and token_types[i] == 4]
        # longest literal first so `<|im_end|>` cannot match a shorter prefix; regex.escape for metacharacters
        self._always_re = self._alt(always)
        self._special_re = self._alt(list(self.special_tokens))

    @staticmethod
    def _alt(literals: list[str]):
        if not literals:
            return None
        return regex.compile("|".join(regex.escape(s) for s in sorted(literals, key=len, reverse=True)))

    # -------------------------------------------------------------- constructors
    @classmethod
    def from_gguf(cls, path) -> "Tokenizer":
        return cls.from_metadata(_gguf_metadata(path))

    @classmethod
    def from_metadata(cls, md: dict) -> "Tokenizer":
        """From GGUF tokenizer metadata (`tokenizer.ggml.*`), as a GGUF file or a converter carries it."""
        need = ["tokenizer.ggml.tokens", "tokenizer.ggml.merges"]
        missing = [k for k in need if k not in md]
        if missing:
            raise ValueError("GGUF is missing %s; it does not carry a tokenizer" % ", ".join(missing))
        model = md.get("tokenizer.ggml.model")
        if model != "gpt2":
            raise ValueError("expected a byte-level BPE tokenizer (model 'gpt2'), got %r" % model)
        special = {k: int(md[k]) for k in md
                   if k.startswith("tokenizer.ggml.") and k.endswith("_token_id")}
        return cls(list(md["tokenizer.ggml.tokens"]), list(md["tokenizer.ggml.merges"]),
                   list(md.get("tokenizer.ggml.token_type") or []) or None,
                   md.get("tokenizer.ggml.pre", "qwen35"), special)

    # -------------------------------------------------------------- the algorithm
    def _bpe(self, word: str) -> list[str]:
        """Merge `word` (already byte-mapped) by lowest rank first, not left to right; long words go to _bpe_heap."""
        if len(word) > self.HEAP_MIN:
            return self._bpe_heap(word)
        ranks_get = self.ranks.get      # read once: the scan below works on a local, not on `self` per symbol pair
        parts = list(word)
        while len(parts) > 1:
            best, best_rank = None, None
            for i in range(len(parts) - 1):
                r = ranks_get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            parts[best:best + 2] = [parts[best] + parts[best + 1]]
        return parts

    HEAP_MIN = 64       # longer words use the heap

    def _bpe_heap(self, word: str) -> list[str]:
        """_bpe's result in O(n log n): a linked list of symbols and a heap of ranked pairs keyed (rank, position);
        stale pairs are skipped."""
        import heapq
        parts = list(word)
        n = len(parts)
        nxt = list(range(1, n)) + [-1]
        prv = list(range(-1, n - 1))
        ranks = self.ranks
        heap = []
        for i in range(n - 1):
            r = ranks.get((parts[i], parts[i + 1]))
            if r is not None:
                heap.append((r, i, parts[i], parts[i + 1]))
        heapq.heapify(heap)
        while heap:
            r, i, left, right = heapq.heappop(heap)
            j = nxt[i]
            if parts[i] != left or j < 0 or parts[j] != right:
                continue                                # stale: a symbol has merged since
            parts[i] = left + right
            parts[j] = None                             # j merged into i
            k = nxt[j]
            nxt[i] = k
            if k >= 0:
                prv[k] = i
                r2 = ranks.get((parts[i], parts[k]))
                if r2 is not None:
                    heapq.heappush(heap, (r2, i, parts[i], parts[k]))
            p = prv[i]
            if p >= 0:
                r2 = ranks.get((parts[p], parts[i]))
                if r2 is not None:
                    heapq.heappush(heap, (r2, p, parts[p], parts[i]))
        return [s for s in parts if s is not None]

    PIECE_CACHE_MAX = 200_000    # pre-tokenizer pieces remembered; bounds the memo's memory

    def _encode_plain(self, text: str) -> list[int]:
        out: list[int] = []
        # A piece's ids depend on the piece alone, so repeated pieces (most of a resent conversation) are looked up
        # instead of merged again. Same ids as running _bpe each time. Idea from Strata f346dc6 (MIT).
        cache = self.__dict__.setdefault("_piece_ids", {})
        for piece in self._re.findall(text):
            got = cache.get(piece)
            if got is None:
                mapped = "".join(BYTE_TO_UNICODE[b] for b in piece.encode("utf-8"))
                got = []
                for tok in self._bpe(mapped):
                    i = self.ids.get(tok)
                    if i is None:
                        raise KeyError("BPE produced a token outside the vocabulary: %r" % tok)
                    got.append(i)
                if len(cache) < self.PIECE_CACHE_MAX:
                    cache[piece] = got
            out.extend(got)
        return out

    def _encode_matching(self, text: str, pat, plain=()) -> list[int]:
        """Encode `text`, emitting literal `pat` matches as single tokens and BPE-ing the rest. The split is on raw
        text; matches inside `plain` spans are left to the text around them."""
        if pat is None:
            return self._encode_plain(text)
        out: list[int] = []
        pos = 0
        for m in pat.finditer(text):
            if plain and any(a <= m.start() < b for a, b in plain):
                continue
            if m.start() > pos:
                out.extend(self._encode_plain(text[pos:m.start()]))
            out.append(self.special_tokens[m.group(0)])
            pos = m.end()
        if pos < len(text):
            out.extend(self._encode_plain(text[pos:]))
        return out

    def encode(self, text: str, parse_special: bool = False, plain=()) -> list[int]:
        """Tokenize `text`. `parse_special` governs only type-3 CONTROL literals; type-4 USER_DEFINED ones always match.
        `plain`: (start, end) spans that stay ordinary text even where they spell a literal."""
        return self._encode_matching(text, self._special_re if parse_special else self._always_re, plain)

    def token_bytes(self, i: int) -> bytes:
        """The raw bytes of one token (a multi-byte character can be split across tokens)."""
        cache = self.__dict__.setdefault("_bytes_cache", {})
        b = cache.get(i)
        if b is None:
            if i < 0 or i >= len(self.tokens):
                raise IndexError("token id %d is outside the vocabulary (%d)" % (i, len(self.tokens)))
            raw = bytearray()
            for ch in self.tokens[i]:
                v = UNICODE_TO_BYTE.get(ch)
                if v is None:
                    raise KeyError("token %d contains a character outside the byte alphabet: %r" % (i, ch))
                raw.append(v)
            b = cache[i] = bytes(raw)
        return b

    def decode(self, ids: list[int], errors: str = "replace") -> str:
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors=errors)


# ------------------------------------------------------------------ the pack's tokenizer/ directory
def _gguf_metadata(path) -> dict:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from gguf_reader import GGUFFile
    return GGUFFile(pathlib.Path(path)).metadata


def extract(gguf_path, out_dir) -> dict:
    """Write the tokenizer into `<out_dir>/tokenizer/` so the engine never opens the weight shards for it."""
    return extract_metadata(_gguf_metadata(gguf_path), out_dir)


def extract_metadata(md: dict, out_dir) -> dict:
    """`extract` from GGUF tokenizer metadata held in a dict."""
    out = pathlib.Path(out_dir) / "tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    tk = Tokenizer.from_metadata(md)
    cfg = {
        "model": "gpt2",
        "pre": tk.pre,
        "vocab_size": len(tk.tokens),
        "n_merges": len(tk.ranks),
        "special_ids": tk.special_ids,
        "add_bos_token": False,
        # shipped rather than re-derived by readers, so a port cannot lose the `\p{M}` classes
        "pre_pattern": QWEN35_PATTERN,
        "pre_pattern_source": "llama.cpp src/llama-vocab.cpp (LLAMA_VOCAB_PRE_TYPE_QWEN35)",
    }
    (out / "vocab.json").write_text(json.dumps(tk.ids, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("\n".join("%s %s" % k for k, _ in
                                              sorted(tk.ranks.items(), key=lambda kv: kv[1])), encoding="utf-8",
                                    newline="\n")
    (out / "token_type.json").write_text(json.dumps(tk.token_types), encoding="utf-8")
    (out / "tokenizer.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8",
                                        newline="\n")
    # fine-tunes change the chat template, so the model's own is shipped
    tpl = md.get("tokenizer.chat_template")
    if tpl:
        (out / "chat_template.jinja").write_text(tpl, encoding="utf-8", newline="\n")
    return cfg


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="round-trip a corpus and report")
    args = ap.parse_args()
    cfg = extract(args.gguf, args.out)
    print("tokenizer/: vocab %d, merges %d, pre %s, specials %s"
          % (cfg["vocab_size"], cfg["n_merges"], cfg["pre"], cfg["special_ids"]))
    if args.check:
        tk = Tokenizer.from_gguf(args.gguf)
        corpus = ["", "Hello, world!", "  leading and trailing  ", "a\n\n\nb",
                  "def f(x):\n\treturn x  # comment\n", "\u4f60\u597d\uff0c\u4e16\u754c", "\u0645\u0631\u062d\u0628\u0627",
                  "\U0001f600\U0001f680\U0001f1fa\U0001f1f8", "e\u0301\u0301 combining", "\x00\x01\x7f control",
                  "\u00a0non-breaking\u00a0space", "1234567890", "MixedCASE_and-dashes", "\r\n\r\n", "\u2028\u2029",
                  "x" * 5000]
        bad = 0
        for s in corpus:
            ids = tk.encode(s)
            back = tk.decode(ids)
            if back != s:
                print("  *** ROUND TRIP FAILED *** %r -> %d ids -> %r" % (s[:40], len(ids), back[:40]))
                bad += 1
        print("round trip: %d strings, %d failed" % (len(corpus), bad))
        return 0 if bad == 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
