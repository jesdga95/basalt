"""The requests being answered, by id and by session, so a client can steer one while it runs (answer now)."""
from __future__ import annotations

import collections
import threading

FINISHED_KEPT = 1024                # finished ids remembered, so a late call hears "finished" rather than "unknown"

# What a request is doing.  "queued": waiting for its turn; "prompt": the engine reads the prompt; "thinking" and
# "answering": writing its reply, in or after the thinking; "finished": ended.
STATES = ("queued", "prompt", "thinking", "answering", "finished")


class LiveRequest:
    """One request's id, session and state, and its answer-now signal, which Service.run reads in its token loop."""

    def __init__(self, rid: str, session: str | None, client_id: str | None):
        self.id = rid
        self.session = session
        self.client_id = client_id
        self.state = "queued"
        self.thinking = None                     # whether the reply thinks; None until the prompt is rendered
        self.answer_now = threading.Event()


class AnswerNowRefused(Exception):
    """An answer-now call that cannot apply: the HTTP status, a stable code, the sentence and the request's state."""

    def __init__(self, status: int, code: str, message: str, state: str | None = None):
        super().__init__(message)
        self.status, self.code, self.state = status, code, state

    def body(self) -> dict:
        err = {"type": "not_found_error" if self.status == 404 else "invalid_request_error", "code": self.code,
               "message": str(self)}
        if self.state is not None:
            err["state"] = self.state
        return {"error": err}


class LiveRequests:
    """Every running request across the served models, and the ids of the last ones that finished."""

    def __init__(self):
        self.lock = threading.Lock()
        self.live: dict[str, LiveRequest] = {}
        self.finished: collections.OrderedDict[str, str] = collections.OrderedDict()   # id -> its request id

    def open(self, rid: str, session: str | None = None, client_id: str | None = None) -> LiveRequest:
        req = LiveRequest(rid, session, client_id)
        with self.lock:
            self.live[rid] = req
        return req

    def close(self, req: LiveRequest) -> None:
        with self.lock:
            req.state = "finished"
            if self.live.get(req.id) is req:
                del self.live[req.id]
            for key in (req.id, req.client_id):
                if key:
                    self.finished[key] = req.id
                    self.finished.move_to_end(key)
            while len(self.finished) > FINISHED_KEPT:
                self.finished.popitem(last=False)

    def _find(self, request_id: str | None, session: str | None) -> LiveRequest:
        """The live request named by its id (Basalt's or the client's x-client-request-id), else the session's newest."""
        if request_id:
            req = self.live.get(request_id) or next((r for r in reversed(self.live.values())
                                                     if r.client_id == request_id), None)
            if req is not None:
                return req
            if request_id in self.finished:
                raise AnswerNowRefused(409, "not_thinking", f"request {self.finished[request_id]} has finished",
                                       "finished")
            raise AnswerNowRefused(404, "unknown_request", f"no running or recent request {request_id}")
        req = next((r for r in reversed(self.live.values()) if r.session == session), None)
        if req is None:
            raise AnswerNowRefused(404, "unknown_session", f"no running request in session {session}")
        return req

    def answer_now(self, request_id: str | None = None, session: str | None = None) -> dict:
        """Ask a request to end its thinking and answer; the reply names its state.  A request still queued or reading
        its prompt takes it too: its thinking closes at its first token.  AnswerNowRefused when it cannot apply."""
        if not request_id and not session:
            raise AnswerNowRefused(400, "no_target", "name the request: request_id, or session_id (or a session header)")
        with self.lock:
            req = self._find(request_id, session)
            state = req.state
            if req.thinking is False:
                raise AnswerNowRefused(409, "not_thinking", f"request {req.id} runs with thinking off", state)
            if state not in ("queued", "prompt", "thinking"):
                raise AnswerNowRefused(409, "not_thinking", f"request {req.id} is already {state}", state)
            req.answer_now.set()
        return {"object": "answer_now", "request_id": req.id, "state": state}
