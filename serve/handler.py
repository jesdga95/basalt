"""The HTTP server and its request handler: routes, authentication, CORS and streaming."""
from __future__ import annotations

import json
import math
import os
import re
import select
import socket
import sys
import threading
import time
import uuid
from contextlib import closing
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from serve.api import responses as responses_api
from serve.api.logs import stream_logs
from serve.api.responses import error_body as responses_error_body
from serve import ROOT
from serve.api.anthropic import anthropic_collect, anthropic_events
from serve.api.documents import sources as document_sources
from serve.engine.errors import ClientGone, EngineDied, EngineError, EngineStarting, EngineStuck, GpuBusy, ModelBusy
from serve.api.grammar import REQUEST_KEY as GRAMMAR_KEY
from serve.api.logit_bias import REQUEST_KEY as BIAS_KEY, checked as checked_bias
from serve.api.logprobs import REQUEST_KEY as LOGPROBS_KEY, checked as checked_logprobs
from serve.config import checked_sampling
from serve.frontend import anthropic_to_messages, forced_call, openai_to_messages
from serve.api.openai import checked_return_progress, openai_chunks, openai_collect
from serve.api.responses import ResponsesError
from serve.prompt import primed
from serve.security import host_allowed, host_names_for, key_matches, origin_allowed, peer_of, remote_reason
from serve.live import AnswerNowRefused
from serve.monitor import clear as clear_monitor, gather
from serve.router import Router
from serve.service import LOG_FIRST_LINES, LOG_MAX_LINES, QUEUE_BEAT_S, Service, _debug_req
from serve.api.structured import anthropic_format, check_format
from serve.util.text import stop_strings
from serve.util.ui_state import UIState, stream_state


CONTROL_BODY_MAX = 64 << 10                 # the body limit for control routes (/load, /unload, /api/pricing, ...)
HOSTED_TOOLS_NOTED_MAX = 64                 # cap tracked unknown hosted-tool names per service
# Routes that mutate server/model state: require JSON + same-origin (or trusted origin) before reading any body
OWN_PAGE_ROUTES = {
    "/api/monitor/clear": "the request monitor can be cleared",
    "/unload": "the model can be loaded or unloaded",
    "/load": "the model can be loaded or unloaded",
    "/v1/load": "the model can be loaded or unloaded",
    "/v1/unload": "the model can be loaded or unloaded",
    "/models/load": "the model can be loaded or unloaded",
    "/models/unload": "the model can be loaded or unloaded",
    "/api/pricing": "pricing can be changed",
    "/api/totals-reset": "totals can be reset",
    "/v1/vram": "the VRAM reserve can be changed",
}
_CORS_HEADER_RE = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z, ]+")
# What this server offers beyond the OpenAI, Anthropic and llama.cpp APIs, listed under /props "basalt"
CAPABILITIES = ["answer_now", "logprobs", "documents"]
TURN_POLL_S = 0.5                           # how soon a request waiting for its turn notices its client left
# The longest a connection may stay silent before its handler thread gives it up. Without one, a client whose
# connection died without a close (a laptop sleeping, a link dropping mid-body) left the thread blocked in
# rfile.read() for the life of the server: the request never reached the engine and the thread was lost. It is a
# per-read deadline, not a total one - a slow client that keeps sending bytes is never cut off. A header read that
# expires is ended by handle_one_request itself; a body that stops arriving is the 408 in _body.
HTTP_BODY_S = 60.0
# Anthropic clients add /v1/messages to their base URL themselves, so a base URL that already ends in /v1 makes them
# post /v1/v1/messages. These two paths mean the same as the real ones; nothing else is aliased.
V1_V1_ALIASES = {"/v1/v1/messages": "/v1/messages", "/v1/v1/messages/count_tokens": "/v1/messages/count_tokens"}


class BadBody(Exception):
    """A request body that cannot be read (malformed or oversized): the status and the sentence."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def request_path(target: str) -> str:
    """The route of a request target: no query, no trailing slash, and a doubled /v1 of the two Anthropic paths undone."""
    path = target.split("?")[0].rstrip("/")
    return V1_V1_ALIASES.get(path, path)


def _size_text(n: int) -> str:
    return f"{n >> 20} MiB" if n >= 1 << 20 else f"{n >> 10} KiB"


def _checked_max_tokens(req: dict, *keys: str) -> int:
    """Validate max_completion_tokens / max_tokens (int, not bool; None/0/-1 = rest of context)."""
    picked = 0
    for key in keys:
        v = req.get(key)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{key}: expected an integer, not {v!r}")
        if v < -1:
            raise ValueError(f"{key}: expected an integer >= -1, not {v!r}")
        if not picked and v:
            picked = v
    return picked


def _clean_log_token(text: str, limit: int = 64) -> str:
    """Strip ASCII control characters and bound length before printing an untrusted identifier."""
    clean = re.sub(r"[\x00-\x1f\x7f]+", " ", text).strip()
    return clean[:limit] + ("..." if len(clean) > limit else "")


def make_handler(target: Service | Router):
    router = target if isinstance(target, Router) else Router([target])
    svc = router.front                                      # the settings every model shares

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"                       # SSE ends by closing the connection
        timeout = HTTP_BODY_S                               # StreamRequestHandler.setup puts it on the connection:
        # every header, body read and write carries the deadline, so no half-open client holds a thread forever
        disable_nagle_algorithm = True                      # streamed events go at once: no waiting for a coalesced segment

        watch_done = None                                   # stops this request's disconnect watcher
        cancel = None                                       # set when the client left
        turn = None                                         # this request's router turn
        sse_open = False                                    # the stream's headers went out
        live = None                                         # this request's entry in router.live (answer now)

        # a request's session identity, the headers a load balancer names its flows by (tinyllb's order), then the
        # request body's own field; the slot the engine runs a conversation in follows it.  The grok build CLI names
        # its conversations with x-grok-conv-id / x-grok-session-id on every request, to any base URL.  Deliberately
        # not here: x-client-request-id, which identifies one request and would give every turn a session of its own.
        SESSION_HEADERS = ("x-llm-flow-id", "x-claude-code-session-id", "x-session-id", "x-session-affinity",
                           "x-grok-conv-id", "x-grok-session-id")

        def _session_id(self, req) -> str | None:
            """The conversation this request belongs to, or None when it names no session (admission is prompt-only)."""
            for h in self.SESSION_HEADERS:
                v = (self.headers.get(h) or "").strip()
                if v:
                    return v
            v = req.get("session_id") if isinstance(req, dict) else None
            return v if isinstance(v, str) and v else None

        def log_message(self, fmt, *args):
            pass

        def end_headers(self):
            """A chat response names its request id."""
            if self.live is not None:
                self.send_header("X-Request-Id", self.live.id)
            super().end_headers()

        def _open_live(self, rid: str, req: dict) -> None:
            """Register this request under its response id (and the client's x-client-request-id) for answer now."""
            client = (self.headers.get("x-client-request-id") or "").strip()[:256] or None
            self.live = router.live.open(rid, self._session_id(req), client)

        def parse_request(self):
            """Without an API key, check the Host header (DNS-rebinding protection); a key makes the check moot."""
            if not super().parse_request():
                return False
            host = self.headers.get("Host")
            if svc.api_key or host_allowed(host, svc.host_names, "*" in svc.allowed_hosts):
                return True
            print(f"[server] error: refused Host {host!r} from {self.client_address[0]} "
                  f"(--api-key KEY, or --allowed-host)",
                  flush=True)
            self._json(403, {"error": {"type": "forbidden", "message":
                             f"Host {host!r} is not allowed (DNS rebinding protection). Reaching the server under this "
                             f"name on purpose? Start the server with --allowed-host NAME, "
                             f"or with --api-key KEY, which turns this check off"}})
            return False

        def _chunked(self) -> bool:
            """The body is sent as Transfer-Encoding: chunked (it wins over a Content-Length, RFC 9112)."""
            te = self.headers.get("Transfer-Encoding", "") or ""
            return te.split(",")[-1].strip().lower() == "chunked"

        def _read(self, n: int) -> bytes:
            """rfile.read with the stalled client named: the handler's timeout (HTTP_BODY_S) turns a connection that
            stopped arriving into a 408 and gives the thread back, instead of holding it for the life of the server."""
            try:
                return self.rfile.read(n)
            except (TimeoutError, OSError) as e:
                self.close_connection = True
                raise BadBody(408, "the request body stopped arriving") from e

        def _readline(self, n: int) -> bytes:
            """rfile.readline under the same deadline."""
            try:
                return self.rfile.readline(n)
            except (TimeoutError, OSError) as e:
                self.close_connection = True
                raise BadBody(408, "the request body stopped arriving") from e

        def _read_chunked(self, limit: int, keep: bool = True) -> bytes:
            """Decode a chunked body: at most `limit` bytes (413 beyond it; the size is checked before a chunk is
            read, so a huge announced chunk is never allocated), 400 for a malformed one. keep=False drops it."""
            parts, total = [], 0
            while True:
                line = self._readline(1025)
                if not line.endswith(b"\n"):
                    raise BadBody(400, "malformed chunked request body")
                try:
                    size = int(line.split(b";", 1)[0].strip(), 16)
                except ValueError:
                    raise BadBody(400, "malformed chunk size in the request body") from None
                if size < 0:
                    raise BadBody(400, "malformed chunk size in the request body")
                if size == 0:
                    for _ in range(64):                       # the trailers, up to the blank line
                        if self._readline(8193).strip() == b"":
                            break
                    return b"".join(parts)
                total += size
                if total > limit:
                    self.close_connection = True
                    raise BadBody(413, f"the request body is larger than {_size_text(limit)}")
                left = size
                while left > 0:
                    piece = self._read(min(left, 1 << 20))
                    if not piece:
                        raise BadBody(400, "the chunked request body ended early")
                    left -= len(piece)
                    if keep:
                        parts.append(piece)
                if self._read(2) != b"\r\n":
                    raise BadBody(400, "malformed chunked request body")

        def _body(self, limit: int | None = None) -> bytes:
            """The request body, chunked or by Content-Length, at most `limit` (default svc.max_payload) bytes."""
            limit = svc.max_payload if limit is None else min(limit, svc.max_payload)
            if self._chunked():
                return self._read_chunked(limit)
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                raise BadBody(400, "invalid Content-Length") from None
            if length < 0:
                raise BadBody(400, "invalid Content-Length")
            if length > limit:
                self.close_connection = True
                raise BadBody(413, f"the request body is larger than {_size_text(limit)}")
            return self._read(length)

        def _is_json_content_type(self) -> bool:
            return (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower() == "application/json"

        def _foreign_page(self) -> bool:
            """Without an API key, refuse a /v1 or tokenizer POST from another site's page: its Origin must be
            allowed (no Origin is fine) and, with one, the body must be JSON."""
            origin = self.headers.get("Origin")
            if svc.api_key or not origin:
                return False
            if not origin_allowed(origin, self.headers.get("Host"), svc.host_names,
                                  [*svc.trusted_origins, *svc.cors_origins]):
                print(f"[server] error: refused a request from web page {origin!r} "
                      f"(--api-key KEY, --allowed-host, --cors-origin)", flush=True)
                self._json(403, {"error": {"type": "forbidden", "message":
                                 f"web pages of {origin} may not use this server without an API key; start the server with --api-key KEY, "
                                 f"or add the page's host with --allowed-host (or its origin with --cors-origin)"}})
                return True
            if not self._is_json_content_type():
                self._json(415, {"error": {"message": "send application/json"}})
                return True
            return False

        def _image_access(self):
            """What this request may name as an image, from its own connection."""
            why = "remote-only" if svc.image_sources.remote_only else remote_reason(self.client_address, self.headers)
            return replace(svc.image_sources, local=not why, why=why, peer=peer_of(self.client_address))

        def _client_cancel(self) -> threading.Event:
            """The event set when this request's client hangs up, watched from the first call."""
            if self.cancel is None:
                self.cancel = threading.Event()
                self._watch_client(self.cancel)
            return self.cancel

        def _take_turn(self, stream: bool = False) -> None:
            """Wait for this request's model to be loaded and its turn to come, which may take a swap. A stream gets
            its headers and keep-alives while it waits; a client that leaves gives its place up (ClientGone)."""
            self.turn = router.join(self.svc)
            if router.wait(self.turn, 0):
                return
            cancel, beat = self._client_cancel(), time.monotonic()
            while not router.wait(self.turn, TURN_POLL_S):
                if cancel.is_set():
                    raise ClientGone
                if stream and time.monotonic() - beat >= QUEUE_BEAT_S:
                    beat = time.monotonic()
                    try:
                        self._sse()
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                    except OSError:
                        raise ClientGone from None

        def _watch_client(self, cancel: threading.Event) -> None:
            """Cancel the request when its client hangs up, polling every 0.5 s: a readable socket with nothing to
            read is EOF (the request is fully read, so data is not a hang-up)."""
            done = self.watch_done = threading.Event()
            sock = self.connection

            def watch():
                while not done.wait(0.5) and not cancel.is_set():
                    try:
                        readable, _, _ = select.select([sock], [], [], 0)
                        gone = bool(readable) and sock.recv(1, socket.MSG_PEEK) == b""
                    except (ConnectionError, TimeoutError):
                        gone = True
                    except (OSError, ValueError):            # socket closed: the request has ended
                        return
                    if gone:
                        cancel.set()
                        return

            threading.Thread(target=watch, daemon=True, name="client-watch").start()

        def _cors(self):
            """CORS headers, only for /v1/* and a listed origin."""
            if not svc.cors_origins or not self.path.split("?")[0].startswith("/v1/"):
                return
            origin = (self.headers.get("Origin") or "").rstrip("/")
            if "*" in svc.cors_origins:
                allow = "*"
            elif origin and origin in svc.cors_origins:
                allow = origin
                self.send_header("Vary", "Origin")
            else:
                return
            self.send_header("Access-Control-Allow-Origin", allow)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            # echo the requested headers: "*" does not cover Authorization
            asked = (self.headers.get("Access-Control-Request-Headers") or "").strip()
            if not asked or len(asked) > 4096 or not _CORS_HEADER_RE.fullmatch(asked):
                asked = "Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta"
            self.send_header("Access-Control-Allow-Headers", asked)
            self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Access-Control-Expose-Headers", "X-Request-Id")    # readable by a page's script

        def do_OPTIONS(self):
            # preflights carry no credentials, so no API key check
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _stream_error(self, obj) -> None:
            """An error after the stream's headers went out (a request that waited for a swap), in the API's own form."""
            path = request_path(self.path)
            err = obj.get("error") if isinstance(obj.get("error"), dict) else {"message": str(obj)}
            if path == "/v1/messages":
                data = b"event: error\ndata: " + json.dumps({"type": "error", "error": err}, ensure_ascii=False).encode() + b"\n\n"
            elif path == "/v1/responses":
                data = b"event: error\ndata: " + json.dumps({"type": "error", "code": err.get("code"), "message": err.get("message")},
                                                            ensure_ascii=False).encode() + b"\n\n"
            else:
                data = b"data: " + json.dumps({"error": err}, ensure_ascii=False).encode() + b"\n\ndata: [DONE]\n\n"
            self.wfile.write(data)

        def _json(self, code, obj):
            if getattr(self, "svc", None) is not None:
                self.svc.monitor.json(code, obj)
            if self.sse_open:
                return self._stream_error(obj)
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _died(self, e) -> str:
            """An engine error's message; the restart hint applies while the server runs."""
            return str(e) if router.stopping else f"{e}; the next request restarts it"

        def _authorized(self) -> bool:
            if not svc.api_key:
                return True
            auth = self.headers.get("Authorization", "")
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            if key_matches(given, svc.api_key):
                return True
            self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong API key"}})
            return False

        def do_GET(self):
            path = request_path(self.path)
            if path.startswith("/fonts/"):
                name = path[len("/fonts/"):]
                f = ROOT / "serve" / "web" / "fonts" / name
                if "/" in name or "\\" in name or not name.endswith(".woff2") or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "font/woff2")
                self.send_header("Cache-Control", "max-age=86400")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path.startswith("/web/"):
                name = path[len("/web/"):]
                types = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                         ".svg": "image/svg+xml", ".png": "image/png"}
                f = ROOT / "serve" / "web" / name
                ext = os.path.splitext(name)[1]
                if "/" in name or "\\" in name or ext not in types or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", types[ext])
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api-monitor":                      # the page is inert without the data, so it needs no key
                body = (ROOT / "serve" / "web" / "monitor.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/monitor":
                if not self._authorized():
                    return self._json(401, {"error": {"message": "unauthorized"}})
                q = parse_qs(urlsplit(self.path).query)
                rid = (q.get("id") or [None])[0]
                data = gather(router.services, rid)
                if rid is not None and data is None:
                    return self._json(404, {"error": {"message": "request not found"}})
                return self._json(200, data)
            if path == "/metrics":
                if self._authorized():
                    qs = parse_qs(urlsplit(self.path).query)
                    name = qs.get("model", [None])[0]
                    fmt = qs.get("format", [None])[0]
                    from serve.prometheus import CONTENT_TYPE as PROM_CT, render_prometheus, wants_prometheus
                    if wants_prometheus(self.headers.get("Accept"), fmt):
                        if name is not None:
                            svcs = [router.service_for(name)]
                            if svcs[0] is None:
                                self._json(404, {"error": {"message": "model not found"}})
                                return
                        else:
                            svcs = list(router.services) if hasattr(router, "services") else [router.shown()]
                        entries = [(router.metrics(s), s.prom_snapshot()) for s in svcs]
                        body = render_prometheus(entries).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", PROM_CT)
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    shown = router.shown() if name is None else router.service_for(name)
                    if shown is None:
                        self._json(404, {"error": {"message": "model not found"}})
                    else:
                        self._json(200, router.metrics(shown))
            elif path == "/api/logs":
                if self._authorized():
                    self._logs()
            elif path == "/api/logs/stream":
                if self._authorized():
                    stream_logs(self, svc.logs)
            elif path == "/api/state/stream":
                if self._authorized():
                    stream_state(self, self.server.ui_state)
            elif path == "":
                body = (ROOT / "serve" / "web" / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path in ("/health", "/api/health"):
                shown = router.shown()
                load = shown.load_progress()
                self._json(200, {"status": "ok", "max_context": shown.engine.max_context, "model": shown.model,
                                 "images": shown.vision is not None, "api_key": bool(svc.api_key),
                                 "loaded": shown.loaded(), "loading": shown.loading, "service": "basalt",
                                 "percent": load.get("percent") if load else None, "swap": router.swap_status()})
            elif path == "/status":
                if not self._authorized():                  # it shows the last answer's tail
                    return
                shown = router.shown()
                with shown.status_lock:
                    s = dict(shown.status)
                now = time.time()
                if s.get("busy"):
                    s["elapsed_s"] = round(now - s["started"], 1)
                    if s.get("first_token"):
                        s["tokens_per_s"] = round(shown._tok_s(), 1)
                        s["tokens_per_s_mean"] = round(shown._tok_s_mean(), 1)
                for k in ("started", "first_token"):
                    s.pop(k, None)
                self._json(200, s)
            elif path in ("/v1/models", "/models"):
                if self._authorized():
                    models = router.models()
                    # "data" is the OpenAI list shape; "models" is what newer Codex builds decode
                    self._json(200, {"object": "list", "data": models, "models": models})
            elif path == "/props":
                if self._authorized():
                    self._props()
            elif path == "/slots":
                if self._authorized():
                    shown = router.shown()
                    loaded = not hasattr(shown.engine, "alive") or shown.engine.alive()
                    self._json(200, shown.slot_table()[0] if loaded else [])
            elif path == "/v1/status":
                if self._authorized():
                    self._json(200, {**router.shown().v1_status(), "models": router.models(), "swap": router.swap_status()})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if not self._authorized():
                return
            path = request_path(self.path)   # clients post /v1/messages?beta=true
            if (path.startswith("/v1/") or path in ("/tokenize", "/detokenize")) and self._foreign_page():
                return
            # own page and JSON only before reading any body: a form POST from any site must not control the server
            own_what = OWN_PAGE_ROUTES.get(path)
            if own_what is not None and not self._own_page(own_what):
                return
            if path in ("/unload", "/load", "/api/monitor/clear") and not self._control_body():
                return
            if path == "/api/monitor/clear":
                return self._json(200, clear_monitor(router.services))
            if path == "/unload":                            # unload between requests
                try:
                    r = router.unload(router.shown())
                except EngineStuck as e:
                    self._json(503, {"error": {"type": "server_error", "message": str(e)}})
                    return
                self._json(409 if r == "busy" else 200, {"status": r})
                return
            if path == "/load":                              # load ahead of a request
                self.svc = router.shown()
                try:
                    self._take_turn()
                    self.svc.load()
                    self._json(200, {"status": "loaded"})
                except (GpuBusy, EngineDied) as e:
                    self._json(503, {"error": {"type": "server_error", "message": str(e)}})
                except ClientGone:
                    pass
                finally:
                    self._end_request()
                return
            try:
                body_limit = CONTROL_BODY_MAX if own_what is not None else None
                req = json.loads(self._body(body_limit) or b"{}")
                if not isinstance(req, dict):
                    raise ValueError("send a JSON object")
                if path.startswith("/v1/responses/"):        # nothing is stored
                    self._json(404, responses_error_body("this server keeps no responses (stateless); send the "
                                                         "whole conversation to POST /v1/responses", code="not_found"))
                    return
                if path in ("/v1/load", "/v1/unload", "/models/load", "/models/unload"):
                    target = router.shown() if req.get("model") is None else router.service_for(req["model"])
                    if target is None:
                        self._json(404, {"error": {"message": "model not found"}})
                        return
                    self.svc = target
                    if path in ("/v1/unload", "/models/unload"):
                        result = router.unload(target)
                        if result == "busy":
                            raise ModelBusy("a request is running or queued")
                    elif target.loading:                 # a load in progress: wait for it like any request
                        self._take_turn()
                        target.load()
                        result = "loaded"
                    else:
                        self._take_turn()                # another model's turn: this load waits for it
                        if not target.fifo.acquire(blocking=False):
                            raise ModelBusy("a request is running or queued")
                        try:
                            with target.status_lock:
                                if target.status.get("busy") or target.status.get("queued"):
                                    raise ModelBusy("a request is running or queued")
                            target.ensure_loaded()
                            target.last_request_at = time.time()
                        finally:
                            target.fifo.release()
                        result = "loaded"
                    self._json(200, {"status": result, **target.v1_status()})
                    return
                if path == "/api/pricing":
                    shown = router.shown()
                    updates: dict[str, float | None] = {}
                    for field in ("cost_input_per_million", "cost_output_per_million",
                                  "cost_cache_write_per_million", "cost_cache_hit_per_million"):
                        if field in req:
                            v = req[field]
                            if v is not None:
                                if isinstance(v, bool) or not isinstance(v, (int, float)):
                                    raise ValueError(f"{field}: a number or null")
                                if not math.isfinite(v):
                                    raise ValueError(f"{field}: a finite number or null")
                                if v < 0:
                                    raise ValueError(f"{field}: a non-negative number or null")
                            updates[field] = float(v) if v is not None else None
                    from serve.totals import totals_with_cost
                    with shown.status_lock:
                        for attr, val in updates.items():
                            setattr(shown, attr, val)
                        if shown.totals_path:
                            shown._save_totals()
                        totals = {"since": int(shown.started_at), "requests": shown.requests_done, **shown.totals}
                        totals_with_cost(totals, shown.cost_input_per_million, shown.cost_output_per_million,
                                         shown.cost_cache_write_per_million, shown.cost_cache_hit_per_million)
                    self._json(200, {"status": "ok", "changed": list(updates), "totals": totals})
                    return
                if path == "/api/totals-reset":
                    shown = router.shown()
                    if not shown.totals_path:
                        self._json(400, {"error": {"message": "no totals file configured"}})
                        return
                    from serve.totals import totals_reset_file
                    with shown.status_lock:
                        prev = totals_reset_file(shown.totals_path)
                        shown.totals = {"prompt": 0, "reused": 0, "written": 0, "prompt_ms": 0.0, "decode_ms": 0.0}
                        shown.requests_done = 0
                        shown.started_at = time.time()
                        shown._save_totals()
                    self._json(200, {"status": "reset", "previous": str(prev) if prev else None})
                    return
                if path == "/v1/answer_now":
                    rid = req.get("request_id")
                    if rid is not None and not isinstance(rid, str):
                        raise ValueError("request_id: a string")
                    try:
                        self._json(200, router.live.answer_now(rid, None if rid else self._session_id(req)))
                    except AnswerNowRefused as e:
                        self._json(e.status, e.body())
                    return
                if path == "/v1/vram":                       # resize the VRAM reserve
                    r = req.get("reserve_mib")
                    if r is not None and (isinstance(r, bool) or not isinstance(r, int) or r < 0):
                        raise ValueError("reserve_mib: a whole number of MiB (0 or more), or null for the start's")
                    try:
                        out = router.vram(r)
                        self._json(202 if out.get("status") == "pending" else 200, out)
                    except EngineDied as e:
                        self._json(503, {"error": {"type": "server_error", "message": str(e)}})
                    return
                requested = req.get("model")
                # llama.cpp's tokenizer routes name no model: the shown one's tokenizer, as /props describes it
                self.svc = router.shown() if requested is None and path in ("/tokenize", "/detokenize") else \
                    router.service_for(requested)
                if self.svc is None:
                    self._unknown_model(path, requested)
                    return
                if path in ("/v1/chat/completions", "/v1/messages", "/v1/responses"):
                    self.svc.monitor.begin(path, req)
                self.svc.request_ctx.path = path
                self.svc.request_ctx.model = self.svc.model  # always the served name, never the requested one
                self.svc.request_ctx.arrived = time.time()   # arrival timestamp for end-to-end / TTFT histograms
                with document_sources(self._image_access()):   # a document by URL or path follows the image rules
                    if path == "/v1/responses":
                        self._responses(req)
                    elif path == "/v1/chat/completions":
                        self._openai(req)
                    elif path == "/v1/messages":
                        self._anthropic(req)
                    elif path == "/v1/messages/count_tokens":
                        self._count_tokens(req)
                    elif path == "/tokenize":
                        self._tokenize(req)
                    elif path == "/detokenize":
                        self._detokenize(req)
                    else:
                        self._json(404, {"error": {"message": "not found"}})
            except EngineError as e:                         # the engine failed: not the request's fault
                if path == "/v1/responses":
                    self._json(500, responses_error_body(str(e), "server_error", code="server_error"))
                else:
                    self._json(500, {"error": {"type": "server_error", "message": str(e)}})
            except BadBody as e:
                if path == "/v1/responses":
                    self._json(e.status, responses_error_body(str(e)))
                else:
                    self._json(e.status, {"error": {"type": "invalid_request_error", "message": str(e)}})
            except ValueError as e:
                if path == "/v1/responses":
                    self._json(400, responses_error_body(str(e)))
                else:
                    self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
            except ModelBusy as e:
                self._json(409, {"error": {"type": "model_busy", "message": str(e)}})
            except (GpuBusy, EngineStarting) as e:
                self._json(503, {"error": {"type": "server_error", "message": str(e)}})
            except EngineDied as e:                          # before the answer started
                self._json(503, {"error": {"type": "server_error", "message": self._died(e)}})
            except EngineStuck as e:                         # the engine could not be ended
                self._json(503, {"error": {"type": "server_error", "message": str(e)}})
            except ClientGone:                               # it left while queued
                pass
            finally:
                self._end_request()

        def _end_request(self) -> None:
            if getattr(self, "svc", None) is not None:
                self.svc.monitor.finish()
            if self.turn is not None:
                router.leave(self.turn)
                self.turn = None
            if self.watch_done is not None:
                self.watch_done.set()
            if self.live is not None:
                router.live.close(self.live)

        def _props(self):
            name = parse_qs(urlsplit(self.path).query).get("model", [None])[0]
            svc = router.shown() if name is None else router.service_for(name)
            if svc is None:
                self._json(404, {"error": {"message": "model not found"}})
                return
            try:
                if not router.multi:
                    svc.wait_load()                          # a load in progress is waited for, not refused
            except (GpuBusy, EngineDied) as e:
                self._json(503, {"error": {"type": "server_error", "message": str(e)}})
                return
            if not svc.loaded() and not getattr(svc.engine, "unloaded", False):
                self._json(503, {"error": {"message": "the engine is not running"}})
                return
            names = {"repetition_penalty": "repeat_penalty", "penalty_last_n": "repeat_last_n"}
            params = {names.get(k, k): v for k, v in svc.sampling_defaults.items()
                      if k in ("temperature", "top_p", "top_k", "min_p", "seed", "repetition_penalty",
                               "presence_penalty", "frequency_penalty", "penalty_last_n")}
            params["n_predict"] = -1
            props = {"default_generation_settings": {"n_ctx": svc.engine.max_context, "params": params},
                     "total_slots": svc.slot_stats()["total"], "model_alias": svc.model, "chat_template": svc.template.source,
                     "modalities": {"vision": svc.vision is not None, "video": svc.video_on()},
                     "models_autoload": hasattr(svc.engine, "restart"),
                     "is_sleeping": not svc.loaded(), "chat_template_caps": dict(svc.dialect.caps)}
            if getattr(svc.engine, "model_path", None):
                props["model_path"] = svc.engine.model_path
            version = getattr(svc.engine, "info", {}).get("version")
            if version:
                props["build_info"] = "basalt " + str(version)
            # "video" only where a video can be read: image input on, a model that reads video, FFmpeg found
            props["basalt"] = {"version": str(version) if version else None,
                               "capabilities": list(CAPABILITIES) + (["video"] if svc.video_on() else [])}
            self._json(200, props)

        def _control_body(self) -> bool:
            """Read the unused body before replying so the client sees no TCP reset."""
            chunked = self._chunked()
            length = 0
            if not chunked:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._json(400, {"error": {"message": "invalid Content-Length"}})
                    return False
                if not 0 <= length <= CONTROL_BODY_MAX:
                    self._json(413, {"error": {"message": "control request body is limited to 64 KiB"}})
                    return False
            timeout = self.connection.gettimeout()
            try:
                self.connection.settimeout(2.0)
                if chunked:
                    self._read_chunked(CONTROL_BODY_MAX, keep=False)
                    complete = True
                else:
                    complete = len(self.rfile.read(length)) == length
            except BadBody as e:
                self._json(e.status, {"error": {"message": f"control request: {e}"}})
                return False
            except OSError:
                complete = False
            finally:
                self.connection.settimeout(timeout)
            if not complete:
                self._json(400, {"error": {"message": "incomplete control request body"}})
            return complete

        def _own_page(self, what) -> bool:
            """Require a JSON body and no foreign Origin, so another site's page cannot control the model."""
            if not self._is_json_content_type():
                self._json(415, {"error": {"message": "send application/json"}})
                return False
            # Origin must be this server's address (http/https) or a trusted origin; proxy-added headers prove nothing
            origin = (self.headers.get("Origin") or "").strip().rstrip("/")
            if origin:
                host = (self.headers.get("Host") or "").strip().lower()
                scheme, sep, rest = origin.lower().partition("://")
                same_host = bool(sep and host and scheme in ("http", "https") and rest == host)
                trusted = any(origin.lower() == t.strip().rstrip("/").lower() for t in svc.trusted_origins)
                if not same_host and not trusted:
                    self._json(403, {"error": {"message": f"{what} only from the server's own page"}})
                    return False
            return True

        def _logs(self):
            """GET /api/logs?after=<id>&limit=<n>: the newest lines after that id, and the newest id."""
            query = parse_qs(urlsplit(self.path).query)

            def number(name, default):
                try:
                    return max(0, int(query.get(name, [default])[0]))
                except ValueError:
                    return default
            after = number("after", None) if "after" in query else None
            limit = min(number("limit", LOG_FIRST_LINES), LOG_MAX_LINES) if after is None else \
                min(number("limit", LOG_MAX_LINES), LOG_MAX_LINES)
            lines, last = svc.logs.since(after or 0, limit)
            self._json(200, {"lines": lines, "last": last})

        def _sse(self):
            if self.sse_open:
                return
            self.sse_open = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")   # proxies pass each event at once
            self._cors()
            self.end_headers()

        def _unknown_model(self, path, name):
            """404 for a name that serves nothing, or no name where several models are configured, in the shape of the
            API called. A request naming no model reaches here only through `Router.service_for`."""
            anthropic = path.startswith("/v1/messages")
            if name is None:
                text = "model: required" if anthropic else "model is required"
            else:
                text = f"model: {name}" if anthropic else f"The model '{name}' does not exist"
            if anthropic:
                self._json(404, {"type": "error", "error": {"type": "not_found_error", "message": text}})
            else:
                self._json(404, responses_error_body(text, "invalid_request_error", "model", "model_not_found"))

        def _openai(self, req):
            messages, tools, kw = openai_to_messages(req)
            tools, force = forced_call(req.get("tool_choice"), tools, self.svc.dialect.forced_opening)   # a bad choice is a 400 before streaming
            check_format(req.get("response_format"))
            self.svc.require_grammar(req.get("response_format"))
            checked_sampling(req)                             # a sampling value out of range is a 400 likewise
            checked_return_progress(req)                      # and a return_progress that is not a boolean
            max_new = _checked_max_tokens(req, "max_completion_tokens", "max_tokens")   # 0/-1: the rest of the context
            self._open_live("chatcmpl-" + uuid.uuid4().hex[:24], req)
            req[LOGPROBS_KEY] = checked_logprobs(req)         # and a bad logprobs / top_logprobs
            self._take_turn(bool(req.get("stream")))
            self.svc.load()
            req[BIAS_KEY] = checked_bias(req, self.svc.engine)     # a bad logit_bias is a 400 before streaming
            req[GRAMMAR_KEY] = self.svc.grammar_for(tools, req.get("tool_choice"), req.get("tools"),
                                               req.get("response_format"))
            self.svc.reasoning_budget(req)                         # a bad value is a 400 before streaming
            stop_strings(req)                                 # likewise a bad stop
            ids, thinking, max_new = self.svc.prepare(messages, tools, kw, max_new, force=force,
                                                 image_access=self._image_access())
            _debug_req("openai", req, messages, tools, max_new, thinking, len(ids))
            cancel = self._client_cancel()
            chunks = primed(self.svc.monitor.capture(
                openai_chunks(self.svc, req, ids, thinking, tools, max_new, cancel, force=force,
                              session=self._session_id(req), live=self.live), "openai"), ids)
            if not req.get("stream"):
                return self._json(200, openai_collect(chunks))
            self._sse()
            try:
                for c in chunks:
                    if c is None:
                        self.wfile.write(b": keep-alive\n\n")      # SSE comment
                    else:
                        self.wfile.write(b"data: " + json.dumps(c, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
            except OSError:
                cancel.set()                                 # client left: stop the engine
                chunks.close()
            except EngineDied as e:                          # mid-stream
                err = {"error": {"type": "server_error", "message": self._died(e)}}
                self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\ndata: [DONE]\n\n")
            except ValueError as e:                          # engine ERR after the stream started: no 400 now
                err = {"error": {"type": "server_error", "message": str(e)}}
                self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\ndata: [DONE]\n\n")

        def _responses(self, req):
            """POST /v1/responses (serve/api/responses.py); after the stream starts, errors are response.failed."""
            try:
                ids, thinking, tools, max_new, asm, req, force = self._responses_prepare(req)
            except ResponsesError as e:
                return self._json(e.status, e.body())
            except ModelBusy as e:
                return self._json(409, responses_error_body(str(e), "server_error", code="model_busy"))
            except (GpuBusy, EngineStarting, EngineStuck, EngineDied) as e:
                return self._json(503, responses_error_body(str(e), "server_error", code="server_error"))
            cancel = self._client_cancel()

            def events():
                yield from asm.start()
                done = None
                with closing(self.svc.run(ids, thinking, tools, max_new, req, cancel, force=force,
                                          session=self._session_id(req), live=self.live)) as inference:
                    for kind, x in inference:
                        if kind == "ping":
                            yield None
                        elif kind == "event":
                            yield from asm.feed(x)
                        elif kind == "logprobs":
                            asm.logprob(*x)
                        elif kind == "done":
                            done = x
                if done is not None and done["finish"] != "cancel" and not cancel.is_set():
                    yield from asm.finish(done)

            generation = events()
            try:
                items = primed(self.svc.monitor.capture(generation, "responses"), ids)
            except EngineError:
                raise
            except ValueError as e:                          # too long for the context: found when the engine counted
                err = ResponsesError(str(e), "input", "context_length_exceeded" if "context" in str(e) else None)
                return self._json(err.status, err.body())
            if not req.get("stream"):
                try:
                    result = responses_api.collect(items)
                except EngineDied as e:
                    return self._json(503, responses_error_body(self._died(e), "server_error",
                                                                code="server_error"))
                except EngineError as e:                     # the engine failed
                    return self._json(500, responses_error_body(str(e), "server_error", code="server_error"))
                except ValueError as e:                      # the engine refused the request
                    err = ResponsesError(str(e), "input", "context_length_exceeded" if "context" in str(e) else None)
                    return self._json(err.status, err.body())
                return self._json(200, result)
            last = time.monotonic()

            def send(e):
                self.wfile.write(f"event: {e['type']}\n".encode() + b"data: " +
                                 json.dumps(e, ensure_ascii=False).encode() + b"\n\n")

            try:
                self._sse()
                for e in items:
                    if e is None:
                        # clients' idle timers count events, not SSE comments: send in_progress every 15 s
                        if time.monotonic() - last >= 15:
                            send(asm.in_progress())
                            last = time.monotonic()
                        else:
                            self.wfile.write(b": keep-alive\n\n")
                    else:
                        send(e)
                        last = time.monotonic()
                    self.wfile.flush()
            except OSError:
                cancel.set()                                 # client left: stop the engine
            except EngineDied as e:                          # mid-stream: response.failed
                self._responses_failed(asm, self._died(e), "server_error", send)
            except ValueError as e:                          # engine ERR after the stream started
                self._responses_failed(asm, str(e), "server_error", send)
            finally:
                try:
                    items.close()
                finally:
                    # A primed replay may never have started if writing the headers failed.
                    generation.close()

        def _responses_failed(self, asm, message, code, send):
            e = asm.failed(message, code)
            try:
                send(e)
                self.wfile.flush()
            except OSError:
                pass

        def _responses_prepare(self, req):
            responses_api.check_request(req)
            messages = responses_api.input_messages(req)
            tools, names, skipped = responses_api.request_tools(req)
            try:
                tools, force = forced_call(req.get("tool_choice"), tools, self.svc.dialect.forced_opening)
            except ValueError as e:
                raise ResponsesError(str(e), "tool_choice") from None
            kw = responses_api.template_kwargs(req)
            try:
                check_format(responses_api.text_format(req))
            except ValueError as e:
                raise ResponsesError(str(e).replace("response_format", "text.format"), "text.format") from None
            try:
                self.svc.require_grammar(responses_api.text_format(req))
            except ValueError as e:
                raise ResponsesError(str(e).replace("response_format", "text.format"), "text.format") from None
            req[GRAMMAR_KEY] = self.svc.grammar_for(tools, req.get("tool_choice"), req.get("tools"),
                                               responses_api.text_format(req))
            noted = self.svc.__dict__.setdefault("hosted_tools_noted", set())   # once per tool (bounded)
            fresh = sorted(set(skipped) - noted)
            if fresh:
                to_log = []
                for name in fresh:
                    if name in responses_api.HOSTED_TOOLS:
                        noted.add(name)
                        to_log.append(name)
                    elif len(noted) < HOSTED_TOOLS_NOTED_MAX:
                        noted.add(name)
                        to_log.append(_clean_log_token(name))
                if to_log:
                    print(f"[server] warning: /v1/responses ignores hosted tools {', '.join(to_log)}", flush=True)
            try:
                self.svc.reasoning_budget(req)                    # a bad value is a 400 before streaming
            except ValueError as e:
                raise ResponsesError(str(e), "reasoning_budget_tokens") from None
            try:
                checked_sampling(req)
            except ValueError as e:
                raise ResponsesError(str(e), str(e).split(":")[0]) from None
            req[LOGPROBS_KEY] = responses_api.requested_logprobs(req)
            try:
                stop_strings(req)
            except ValueError as e:
                raise ResponsesError(str(e), "stop") from None
            self._open_live(responses_api.new_id("resp"), req)
            self._take_turn(bool(req.get("stream")))
            self.svc.load()
            try:
                ids, thinking, max_new = self.svc.prepare(messages, tools, kw, req.get("max_output_tokens") or 0, force=force,
                                                     image_access=self._image_access())
            except ResponsesError:
                raise
            except ValueError as e:                          # too long for the context, or an image without vision
                raise ResponsesError(str(e), "input", "context_length_exceeded" if "context" in str(e) else None) \
                    from None
            _debug_req("responses", req, messages, tools, max_new, thinking, len(ids))
            include = req.get("include") if isinstance(req.get("include"), list) else []
            asm = responses_api.Assembler(req, self.svc.model, len(ids), names,
                                          "reasoning.encrypted_content" in include, rid=self.live.id)
            return ids, thinking, tools, max_new, asm, req, force

        def _tokenize(self, req):
            """llama.cpp's /tokenize: the model's token ids for `content`, special-token text read as its token unless
            `parse_special` is false."""
            content, parse = req.get("content"), req.get("parse_special", True)
            if not isinstance(content, str):
                raise ValueError("content: a string")
            if not isinstance(parse, bool):
                raise ValueError("parse_special: true or false")
            self._json(200, {"tokens": self.svc.tok.encode(content, parse_special=parse)})

        def _detokenize(self, req):
            """llama.cpp's /detokenize: the text of `tokens`, a list of the model's token ids."""
            ids = req.get("tokens")
            if not isinstance(ids, list) or not all(isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in ids):
                raise ValueError("tokens: a list of token ids")
            try:
                text = self.svc.tok.decode(ids)
            except (IndexError, KeyError):
                raise ValueError("tokens: an id outside the model's vocabulary") from None
            self._json(200, {"content": text})

        def _count_tokens(self, req):
            """Anthropic's token count: the rendered, tokenized prompt, without running the model."""
            messages, tools, kw = anthropic_to_messages(req, self.svc.anthropic_think_unasked)
            tools, _ = forced_call(req.get("tool_choice"), tools, self.svc.dialect.forced_opening)   # same 400 as the request itself ("none": no tools)
            stop_strings(req)                                 # same 400 as the request itself
            self._json(200, {"input_tokens": len(self.svc.encode_prompt(messages, tools, kw))})

        def _anthropic(self, req):
            messages, tools, kw = anthropic_to_messages(req, self.svc.anthropic_think_unasked)
            tools, force = forced_call(req.get("tool_choice"), tools, self.svc.dialect.forced_opening)   # "any" / {"type": "tool", "name": N}
            max_new = _checked_max_tokens(req, "max_tokens")           # 0/-1: the rest of the context
            response_format = anthropic_format(req)            # structured outputs
            try:
                self.svc.require_grammar(response_format)
            except ValueError as e:
                raise ValueError(str(e).replace("response_format json_schema and json_object", "output_config.format json_schema")) from None
            req[GRAMMAR_KEY] = self.svc.grammar_for(tools, req.get("tool_choice"), req.get("tools"), response_format)
            self.svc.reasoning_budget(req)                         # a bad value is a 400 before streaming
            checked_sampling(req)                             # likewise a sampling value out of range
            req[LOGPROBS_KEY] = checked_logprobs(req)         # Basalt's logprobs, as Chat Completions takes them
            stop_strings(req)                                 # likewise a bad stop
            self._open_live("msg_" + uuid.uuid4().hex[:24], req)
            self._take_turn(bool(req.get("stream")))
            self.svc.load()
            ids, thinking, max_new = self.svc.prepare(messages, tools, kw, max_new, force=force,
                                                 image_access=self._image_access())
            _debug_req("anthropic", req, messages, tools, max_new, thinking, len(ids))
            cancel = self._client_cancel()
            events = primed(self.svc.monitor.capture(
                anthropic_events(self.svc, req, ids, thinking, tools, max_new, cancel, force=force,
                                  session=self._session_id(req), live=self.live), "anthropic"), ids)
            if not req.get("stream"):
                return self._json(200, anthropic_collect(events))
            self._sse()
            try:
                for item in events:
                    if item is None:
                        self.wfile.write(b": keep-alive\n\n")
                    else:
                        name, e = item
                        self.wfile.write(f"event: {name}\n".encode() + b"data: " +
                                         json.dumps(e, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
            except OSError:
                cancel.set()
                events.close()
            except EngineDied as e:                          # mid-stream: an Anthropic error event
                err = {"type": "error", "error": {"type": "api_error", "message": self._died(e)}}
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")
            except ValueError as e:                          # engine ERR after the stream started
                err = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")

    return Handler


class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second process hijack an already-bound port; use SO_EXCLUSIVEADDRUSE instead.
    allow_reuse_address = os.name != "nt"
    request_queue_size = 256      # socketserver's backlog of 5 resets some clients of a burst of simultaneous connections

    router: Router | None = None

    def __init__(self, *args, **kwargs):
        self.log_stop = threading.Event()
        super().__init__(*args, **kwargs)

    def server_bind(self):
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def shutdown(self):
        self.log_stop.set()
        if getattr(self, "ui_state", None) is not None:
            self.ui_state.close()
        super().shutdown()

    def server_close(self):
        self.log_stop.set()
        if getattr(self, "ui_state", None) is not None:
            self.ui_state.close()
        super().server_close()
        if self.router is not None:
            self.router.stop_telemetry()           # serve() started the sampler, so closing the server ends it

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], ConnectionError):   # a hang-up needs no stack trace
            super().handle_error(request, client_address)


def serve(target: Service | Router, host="127.0.0.1", port=8080) -> ThreadingHTTPServer:
    router = target if isinstance(target, Router) else Router([target])
    svc = router.front
    svc.host_names = host_names_for(host, svc.allowed_hosts, svc.trusted_origins)
    router.start_telemetry()
    httpd = Server((host, port), make_handler(router))
    httpd.router = router
    httpd.ui_state = UIState(router)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd
