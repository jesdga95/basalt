"""Responses event streaming."""
from __future__ import annotations

from serve.engine.errors import ClientGone, EngineError


def generate(svc, ids, thinking, tools, max_new, req, cancel, force, asm, *, session=None, live=None):
    """Yield assembled Responses events, closing inference before completion and cancelling it on interruption."""
    iterator = None
    try:
        if cancel.is_set():
            raise ClientGone
        yield from asm.start()
        if cancel.is_set():
            return
        done = None
        iterator = svc.run(ids, thinking, tools, max_new, req, cancel, force=force, session=session, live=live)
        for kind, value in iterator:
            if kind == "ping":
                yield None
            elif kind == "event":
                yield from asm.feed(value)
            elif kind == "logprobs":
                asm.logprob(*value)
            elif kind == "done":
                done = value
        primary, iterator = iterator, None
        primary.close()
        if done is None:
            if cancel.is_set():
                return
            raise EngineError("generation ended without an outcome")
        if done["finish"] != "cancel" and not cancel.is_set():
            yield from asm.finish(done)
    except BaseException:
        cancel.set()
        raise
    finally:
        if iterator is not None:
            iterator.close()
