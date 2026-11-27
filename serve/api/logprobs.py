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


def logprobs_part(content: list, reasoning: list, tools: dict) -> dict:
    """The `logprobs` object: the answer's tokens (OpenAI's `content`), then the thinking's and each tool call's tokens when there are any."""
    part = {"content": content[:]}
    if reasoning:
        part["reasoning_content"] = reasoning[:]
    if tools:
        part["tool_calls"] = [{"index": i, "content": tools[i][:]} for i in sorted(tools)]
    return part


_BETWEEN_TOKENS = frozenset((
    "<|message_model|>", "<|content_thinking|>", "<|content_text|>",
    "<|content_tool_error|>", "<|content_invoke_tool_text|>",
    "<|end_message|>", "<|content_model_end_sampling|>",
))


class Channels:
    """Which text each generated token's entry belongs to: the thinking ("reasoning"), the answer ("content") or tool
    call i ("tool", i). A token the parser holds (part of a tag, a blank) waits for the next event that shows text; the
    thinking's close, block framing tokens and the blank after thinking are in neither text and are left out, so each
    channel's entries join to its text (a call's entries are its tokens as the model wrote them, markup included)."""

    def __init__(self, thinking: bool):
        self.thinking, self.pending, self.frame, self.calls, self.answered, self.last = thinking, [], [], 0, False, None

    def _drop_tag(self, entry: dict | None) -> None:
        """Drop held pieces of a split closing tag from pending; keep any earlier real text (e.g. whitespace)."""
        if entry is not None and entry["token"].startswith("<"):
            return
        for i in range(len(self.pending) - 1, -1, -1):
            if "<" in self.pending[i]["token"]:
                del self.pending[i:]
                return
        self.pending.clear()

    def token(self, entry: dict | None, state0: str, state1: str, evs) -> list[tuple]:
        """A token fed to the parser (`entry` None: one the server wrote itself), the parser's state before and after
        it and its events: the entries whose channel is now known, as (channel, entry, call index or None)."""
        shows = [ev for ev in evs if ev.text or ev.kind in ("tool_start", "tool_call")]
        if state0 == "reasoning" and state1 != "reasoning":
            entry = None                                 # the thinking's close: in neither text
            if not any(ev.kind == "reasoning" and ev.text for ev in evs):
                self.pending.clear()                     # (held pieces of the tag)
        elif state0 != "between" and state1 == "between":
            if state0 in ("content", "literal_call") and not any(ev.kind == "content" and ev.text for ev in evs):
                self._drop_tag(entry)
            entry = None                                 # a block's close (or an unclosed block's next header)
        elif state0 == "between":
            if state1 != "between":
                self.frame.clear()                       # header/opener tokens before the block's body
                if not shows and state1 in ("reasoning", "content"):
                    entry = None                         # block opener tag (<|content_thinking|>, <|content_text|>, ...)
            elif not shows:
                if entry is not None and entry["token"] not in _BETWEEN_TOKENS:
                    self.frame.append(entry)
                return []
            elif self.frame:
                self.pending.extend(self.frame)
                self.frame.clear()
        if entry is not None and (state0 == "between" or state1 == "between") and entry["token"] in _BETWEEN_TOKENS:
            entry = None
        if entry is not None:
            self.pending.append(entry)
        if not shows:
            return []
        out = self._flush(shows[0].kind)
        self.calls += sum(ev.kind == "tool_call" for ev in evs)
        return out

    def finish(self, evs) -> list[tuple]:
        """The parser's last events: the entries still waiting go with them, else with the last call or the answer."""
        shows = [ev for ev in evs if ev.text or ev.kind in ("tool_start", "tool_call")]
        if shows:
            if self.frame:
                self.pending.extend(self.frame)
                self.frame.clear()
            return self._flush(shows[0].kind)
        self.frame.clear()
        if self.pending and self.last is not None and self.last[0] == "tool":
            return self._flush("tool_call", self.last[1])  # a call's closing markup after its event
        return self._flush("content") if self.pending else []

    def _flush(self, kind: str, index: int | None = None) -> list[tuple]:
        channel = "reasoning" if kind == "reasoning" else "content" if kind == "content" else "tool"
        if channel == "tool" and index is None:
            index = self.calls
        out, self.pending = self.pending, []
        if channel == "content" and self.thinking and not self.answered:
            while out and not out[0]["token"].strip():   # the blank between the thinking and the answer
                out.pop(0)
        self.answered = self.answered or channel == "content"
        self.last = (channel, index if channel == "tool" else None)
        return [(channel, e, self.last[1]) for e in out]
