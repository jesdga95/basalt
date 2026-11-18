"""Responses event streaming."""

from contextlib import closing

from serve.engine.errors import ClientGone, EngineError


def generate(svc, ids, thinking, tools, max_new, req, cancel, force, asm, *, session=None, live=None):
    """Assemble response events, releasing inference before reporting an outcome."""
    try:
        if cancel.is_set():
            raise ClientGone()

        yield from asm.start()
        if cancel.is_set():
            raise ClientGone()

        outcome = None
        with closing(svc.run(ids, thinking, tools, max_new, req, cancel, force=force,
                             session=session, live=live)) as inference:
            try:
                for kind, value in inference:
                    if kind == "ping":
                        yield None
                    elif kind == "event":
                        yield from asm.feed(value)
                    elif kind == "logprobs":
                        asm.logprob(*value)
                    elif kind == "done":
                        outcome = value
            except BaseException:
                # Mark an interrupted request before Service.run() receives close().
                cancel.set()
                raise

        if cancel.is_set():
            raise ClientGone()
        if outcome is None:
            raise EngineError("generation ended without an outcome")
        if outcome["finish"] == "cancel":
            raise EngineError("generation was cancelled without a client disconnect")

        yield from asm.finish(outcome)
    except BaseException:
        cancel.set()
        raise
