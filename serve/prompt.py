"""A prompt sent to the engine as text: the engine tokenizes it (GENT) instead of the server.

`PromptText` stands where the id list used to: `len()` is the token count once the engine has said `PT` (before that
an estimate, good for a status line only), `ids()` tokenizes in Python for the paths that need the ids (a second
reply segment, engines that take ids), and `+` gives the id list those paths extend.
"""
from __future__ import annotations

from serve.engine.errors import split_err


CTX_SLACK = 8               # the engine rejects prompt + max_new + 8 > context


def room_error(n: int, ctx: int) -> ValueError:
    return ValueError("request exceeds the context window: "
                      f"prompt ({n} tokens) leaves no room to answer in the context "
                      f"({ctx}); requests are never truncated")


def overflow_error(n: int, max_new: int, ctx: int) -> ValueError:
    room = ctx - CTX_SLACK - n
    return ValueError("request exceeds the context window: "
                      f"prompt ({n} tokens) + max tokens ({max_new}) exceeds the context "
                      f"({ctx}); requests are never truncated. Send a smaller "
                      f"max_tokens (at most {max(0, room)} here), or start the server with "
                      "--fit-max-tokens (or set \"fit-max-tokens\": true in the \"server\" section of "
                      "the --config file) to shorten it to the room left")


class PromptText:
    def __init__(self, tokenizer, text: str, plain=(), tail: str = "", clip: bool = False):
        self.tok = tokenizer
        self.text = text            # what the engine tokenizes: the rendered prompt, then `tail`
        self.body = text[:len(text) - len(tail)] if tail else text
        self.tail = tail            # the forced opening of a call, encoded on its own like the engine does
        self.plain = list(plain)    # character spans of `body` where special-token text stays text
        self.clip = clip            # max_new is a cap: cut to the room the context leaves
        self.n: int | None = None   # prompt tokens, from the engine's PT line
        self.max_new: int | None = None   # the answer cap the engine settled on
        self._ids: list[int] | None = None

    def ids(self) -> list[int]:
        if self._ids is None:
            ids = self.tok.encode(self.body, parse_special=True, plain=self.plain) if self.plain \
                else self.tok.encode(self.body, parse_special=True)
            if self.tail:
                ids = ids + self.tok.encode(self.tail, parse_special=True)
            self._ids = ids
            self.n = len(ids)
        return self._ids

    def __len__(self) -> int:
        if self.n is not None:
            return self.n
        return max(1, len(self.text) // 4)     # not yet counted: a display estimate

    def __add__(self, other):
        return self.ids() + list(other)

    def byte_spans(self) -> list[tuple[int, int]]:
        """The plain spans as byte offsets into the UTF-8 text."""
        if not self.plain:
            return []
        body = self.body
        n = len(body)
        norm = [(a if 0 <= a <= n else (n if a > n else max(n + a, 0)),
                 b if 0 <= b <= n else (n if b > n else max(n + b, 0))) for a, b in self.plain]
        if body.isascii():
            return norm
        pts = sorted({i for pair in norm for i in pair})
        offsets: dict[int, int] = {}
        pos = byte_off = 0
        for p in pts:
            if p > pos:
                byte_off += len(body[pos:p].encode("utf-8"))
                pos = p
            offsets[p] = byte_off
        return [(offsets[a], offsets[b]) for a, b in norm]

    def set_counted(self, n: int, max_new: int) -> None:
        self.n, self.max_new = n, max_new

    def settle(self, n: int, max_new: int, ctx: int) -> int:
        """The answer cap for a prompt of `n` tokens counted by the engine, as the engine itself settles a text prompt:
        cut to the room the context leaves when clipping, else a ValueError when it does not fit."""
        room = ctx - CTX_SLACK - n
        if self.clip:
            if room < 1:
                raise room_error(n, ctx)
            max_new = min(max_new, room)
        elif max_new > room:
            raise overflow_error(n, max_new, ctx)
        self.set_counted(n, max_new)
        return max_new

    def resolve(self, max_new: int, ctx: int) -> tuple[list[int], int]:
        """(ids, max_new) for an engine that takes ids: the answer cap the engine would have settled on."""
        ids = self.ids()
        return ids, self.settle(len(ids), max_new, ctx)

    def explain(self, err: str) -> str:
        """The engine's ERR text in the server's words, as the id path would have raised it: a coded context error
        (the engine's own fields) becomes the API's message, any other error is shown as the engine worded it."""
        code, f, message = split_err(err)
        if code == "context_overflow" and "slot" not in f and {"prompt", "max_new", "context"} <= f.keys():
            self.n = f["prompt"]
            return str(overflow_error(f["prompt"], f["max_new"], f["context"]))
        if code == "context_no_room" and {"prompt", "context"} <= f.keys():
            self.n = f["prompt"]
            return str(room_error(f["prompt"], f["context"]))
        return message


def tokens_known(prompt) -> bool:
    """False for a text prompt the engine has not counted yet."""
    return not isinstance(prompt, PromptText) or prompt.n is not None


def primed(events, prompt):
    """`events` (a generator) advanced until the engine has counted a text prompt, so an error the count finds (a prompt
    over the context) is raised before a stream's headers go out, as it was when the server counted the tokens. The
    items read on the way are replayed first. A prompt that is not text needs no wait."""
    if not isinstance(prompt, PromptText) or tokens_known(prompt):
        return events
    held, later = [], None
    try:
        while not tokens_known(prompt):
            held.append(next(events))
    except StopIteration:
        pass
    except ValueError:
        events.close()
        raise
    except BaseException as e:      # a dead engine and the like reach the stream as they always did
        later = e

    def replay():
        try:
            yield from held
            if later is not None:
                raise later
            yield from events
        finally:
            events.close()
    return replay()
