"""Token log probabilities: OpenAI `logprobs` / `top_logprobs`, computed by the engine and only attached here.

The engine sends `P <id> <lp> [<id>:<lp> ...]` ahead of a pick's `T` line (`BP <slot> ...` ahead of a slot's `BT`
line). The process layer hands the pick on as a `Token` carrying the parsed line; the service turns it into an entry
in OpenAI's shape, with the token's text and bytes from the tokenizer."""
from __future__ import annotations

# the engine-side key of a request: the handler stores the checked count of alternatives there
REQUEST_KEY = "_logprobs"
MAX_TOP = 20                                         # OpenAI's limit on top_logprobs, and the engine's


class Token(int):
    """A pick's id, with the engine's P line for it (`lp`: (logprob, [(id, logprob), ...]))."""
    lp: tuple[float, list[tuple[int, float]]] | None = None


def checked(req: dict) -> int | None:
    """The alternatives a Chat Completions request asks for (0..20), or None for no logprobs; ValueError if bad.
    `top_logprobs` needs `logprobs: true`, as OpenAI has it."""
    on, top = req.get("logprobs"), req.get("top_logprobs")
    if on is not None and not isinstance(on, bool):
        raise ValueError("logprobs: expected true or false")
    if top is not None and (isinstance(top, bool) or not isinstance(top, int) or not 0 <= top <= MAX_TOP):
        raise ValueError(f"top_logprobs: an integer from 0 to {MAX_TOP}")
    if top is not None and not on:
        raise ValueError("top_logprobs: requires logprobs to be true")
    return (top or 0) if on else None


def engine_key(top: int | None) -> str:
    """The engine protocol's ` logprobs=K`, or nothing."""
    return f" logprobs={int(top)}" if isinstance(top, int) and not isinstance(top, bool) and top >= 0 else ""


def parse_line(fields: list[str]) -> tuple[int, float, list[tuple[int, float]]] | None:
    """`<id> <lp> [<id>:<lp> ...]` (a P line after its tag, a BP line after its slot); None if malformed."""
    try:
        alts = []
        for f in fields[2:]:
            i, _, v = f.partition(":")
            alts.append((int(i), float(v)))
        return int(fields[0]), float(fields[1]), alts
    except (IndexError, ValueError):
        return None


def token(tid: int, parsed) -> int:
    """`tid` as a Token carrying its P line when the line is for this id, else the plain id."""
    if parsed is None or parsed[0] != tid:
        return tid
    t = Token(tid)
    t.lp = (parsed[1], parsed[2])
    return t


def entry(tok, t: Token) -> dict:
    """OpenAI's logprobs entry for pick `t`: token text, logprob, UTF-8 bytes and the likeliest alternatives."""
    def one(tid: int, lp: float) -> dict:
        raw = tok.token_bytes(tid) if hasattr(tok, "token_bytes") else tok.decode([tid]).encode("utf-8")
        return {"token": raw.decode("utf-8", errors="replace"), "logprob": lp, "bytes": list(raw)}
    lp, alts = t.lp
    return {**one(int(t), lp), "top_logprobs": [one(i, v) for i, v in alts]}
