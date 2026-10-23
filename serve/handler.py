"""The HTTP server and its request handler: routes, authentication, CORS and streaming."""
from __future__ import annotations

import json
import os
import select
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from serve.api import responses as responses_api
from serve.api.responses import error_body as responses_error_body
from serve import ROOT
from serve.api.anthropic import anthropic_collect, anthropic_events
from serve.engine.errors import EngineDied, EngineStarting, EngineStuck, GpuBusy, ModelBusy
from serve.api.grammar import REQUEST_KEY as GRAMMAR_KEY
from serve.api.logit_bias import REQUEST_KEY as BIAS_KEY, checked as checked_bias
from serve.frontend import anthropic_to_messages, forced_call, openai_to_messages
from serve.api.openai import openai_chunks, openai_collect
from serve.api.responses import ResponsesError
from serve.prompt import primed
from serve.security import host_allowed, host_names_for, key_matches, origin_allowed
from serve.service import LOG_FIRST_LINES, LOG_MAX_LINES, Service, _debug_req
from serve.api.structured import anthropic_format, check_format
from serve.util.text import stop_strings


BODY_MAX = 256 << 20                        # the most a JSON request body may hold (read into memory)
CONTROL_BODY_MAX = 64 << 10                 # the unused body of /load and /unload


class BadBody(Exception):
    """A request body that cannot be read (malformed or oversized): the status and the sentence."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _size_text(n: int) -> str:
    return f"{n >> 20} MiB" if n >= 1 << 20 else f"{n >> 10} KiB"


def make_handler(svc: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"                       # SSE ends by closing the connection

        watch_done = None                                   # stops this request's disconnect watcher

        def log_message(self, fmt, *args):
            pass

        def parse_request(self):
            """Without an API key, check the Host header (DNS-rebinding protection); a key makes the check moot."""
            if not super().parse_request():
                return False
            host = self.headers.get("Host")
            if svc.api_key or host_allowed(host, svc.host_names, "*" in svc.allowed_hosts):
                return True
            print(f"[server] refused Host {host!r} from {self.client_address[0]} "
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

        def _read_chunked(self, limit: int, keep: bool = True) -> bytes:
            """Decode a chunked body: at most `limit` bytes (413 beyond it; the size is checked before a chunk is
            read, so a huge announced chunk is never allocated), 400 for a malformed one. keep=False drops it."""
            parts, total = [], 0
            while True:
                line = self.rfile.readline(1025)
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
                        if self.rfile.readline(8193).strip() == b"":
                            break
                    return b"".join(parts)
                total += size
                if total > limit:
                    self.close_connection = True
                    raise BadBody(413, f"the request body is larger than {_size_text(limit)}")
                left = size
                while left > 0:
                    piece = self.rfile.read(min(left, 1 << 20))
                    if not piece:
                        raise BadBody(400, "the chunked request body ended early")
                    left -= len(piece)
                    if keep:
                        parts.append(piece)
                if self.rfile.read(2) != b"\r\n":
                    raise BadBody(400, "malformed chunked request body")

        def _body(self) -> bytes:
            """The request body, chunked or by Content-Length, at most BODY_MAX bytes."""
            limit = BODY_MAX
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
            return self.rfile.read(length)

        def _foreign_page(self) -> bool:
            """Without an API key, refuse a /v1 POST from another site's page: its Origin must be allowed (no Origin
            is fine) and, with one, the body must be JSON."""
            origin = self.headers.get("Origin")
            if svc.api_key or not origin:
                return False
            if not origin_allowed(origin, self.headers.get("Host"), svc.host_names,
                                  [*svc.trusted_origins, *svc.cors_origins]):
                print(f"[server] refused a request from web page {origin!r} "
                      f"(--api-key KEY, --allowed-host, --cors-origin)", flush=True)
                self._json(403, {"error": {"type": "forbidden", "message":
                                 f"web pages of {origin} may not use this server without an API key; start the server with --api-key KEY, "
                                 f"or add the page's host with --allowed-host (or its origin with --cors-origin)"}})
                return True
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(415, {"error": {"message": "send application/json"}})
                return True
            return False

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
            asked = self.headers.get("Access-Control-Request-Headers")
            self.send_header("Access-Control-Allow-Headers",
                             asked or "Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta")
            self.send_header("Access-Control-Max-Age", "600")

        def do_OPTIONS(self):
            # preflights carry no credentials, so no API key check
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

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
            path = self.path.split("?")[0].rstrip("/")
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
                         ".svg": "image/svg+xml"}
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
            if path == "/metrics":
                if self._authorized():
                    self._json(200, svc.metrics())
            elif path == "/api/logs":
                if self._authorized():
                    self._logs()
            elif path == "":
                body = (ROOT / "serve" / "web" / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path in ("/health", "/api/health"):
                self._json(200, {"status": "ok", "max_context": svc.engine.max_context, "model": svc.model,
                                 "images": svc.vision is not None, "api_key": bool(svc.api_key),
                                 "loaded": svc.loaded(), "service": "basalt"})
            elif path == "/status":
                if not self._authorized():                  # it shows the last answer's tail
                    return
                with svc.status_lock:
                    s = dict(svc.status)
                now = time.time()
                if s.get("busy"):
                    s["elapsed_s"] = round(now - s["started"], 1)
                    if s.get("first_token"):
                        s["tokens_per_s"] = round(svc._tok_s(), 1)
                        s["tokens_per_s_mean"] = round(svc._tok_s_mean(), 1)
                for k in ("started", "first_token"):
                    s.pop(k, None)
                self._json(200, s)
            elif path in ("/v1/models", "/models"):
                if self._authorized():
                    loaded = svc.loaded()
                    model = {"id": svc.model, "object": "model", "status": {"value": "loaded"},
                             "meta": {"n_ctx": svc.engine.max_context},
                             "architecture": {"input_modalities": ["text", "image"] if svc.vision is not None else ["text"],
                                              "output_modalities": ["text"]}}
                    if not loaded and (svc.idle_unload_s or getattr(svc.engine, "unloaded", False)):
                        model["status"] = {"value": "unloaded"}   # listed, loads on use
                        loaded = True
                    self._json(200, {"object": "list", "data": [model] if loaded else []})
            elif path == "/props":
                if self._authorized():
                    self._props()
            elif path == "/slots":
                if self._authorized():
                    loaded = not hasattr(svc.engine, "alive") or svc.engine.alive()
                    self._json(200, svc.slot_table()[0] if loaded else [])
            elif path == "/v1/status":
                if self._authorized():
                    self._json(200, svc.v1_status())
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if not self._authorized():
                return
            path = self.path.split("?")[0].rstrip("/")   # clients post /v1/messages?beta=true
            if path.startswith("/v1/") and self._foreign_page():
                return
            if path in ("/unload", "/load") and not self._control_body():
                return
            # own page and JSON only: a form POST from any site must not control the model
            if path in ("/unload", "/load") and not self._own_page("the model can be loaded or unloaded"):
                return
            if path == "/unload":                            # unload between requests
                try:
                    r = svc.unload()
                except EngineStuck as e:
                    self._json(503, {"error": {"type": "server_error", "message": str(e)}})
                    return
                self._json(409 if r == "busy" else 200, {"status": r})
                return
            if path == "/load":                              # load ahead of a request
                try:
                    svc.load()
                    self._json(200, {"status": "loaded"})
                except (GpuBusy, EngineDied) as e:
                    self._json(503, {"error": {"type": "server_error", "message": str(e)}})
                return
            try:
                req = json.loads(self._body() or b"{}")
                if not isinstance(req, dict):
                    raise ValueError("send a JSON object")
                if path.startswith("/v1/responses/"):        # nothing is stored
                    self._json(404, responses_error_body("this server keeps no responses (stateless); send the "
                                                         "whole conversation to POST /v1/responses", code="not_found"))
                    return
                if path in ("/v1/load", "/v1/unload"):
                    if not self._own_page("the model can be loaded or unloaded"):
                        return
                    if req.get("model") not in (None, *svc.served_names()):
                        self._json(404, {"error": {"message": "model not found"}})
                        return
                    if path == "/v1/unload":
                        result = svc.unload()
                        if result == "busy":
                            raise ModelBusy("a request is running or queued")
                    else:
                        if not svc.fifo.acquire(blocking=False):
                            raise ModelBusy("a request is running or queued")
                        try:
                            with svc.status_lock:
                                if svc.status.get("busy") or svc.status.get("queued"):
                                    raise ModelBusy("a request is running or queued")
                            svc.ensure_loaded()
                            svc.last_request_at = time.time()
                        finally:
                            svc.fifo.release()
                        result = "loaded"
                    self._json(200, {"status": result, **svc.v1_status()})
                    return
                if path == "/v1/vram":                       # resize the VRAM reserve
                    if not self._own_page("the VRAM reserve can be changed"):
                        return
                    r = req.get("reserve_mib")
                    if r is not None and (isinstance(r, bool) or not isinstance(r, int) or r < 0):
                        raise ValueError("reserve_mib: a whole number of MiB (0 or more), or null for the start's")
                    try:
                        self._json(200, svc.vram(r))
                    except EngineDied as e:
                        self._json(503, {"error": {"type": "server_error", "message": str(e)}})
                    return
                requested = req.get("model")
                if not (isinstance(requested, str) and requested in svc.served_names()):
                    self._unknown_model(path, requested)
                    return
                svc.request_ctx.path = path
                svc.request_ctx.model = svc.model            # always the served name, never the requested one
                if path == "/v1/responses":
                    self._responses(req)
                elif path == "/v1/chat/completions":
                    self._openai(req)
                elif path == "/v1/messages":
                    self._anthropic(req)
                elif path == "/v1/messages/count_tokens":
                    self._count_tokens(req)
                else:
                    self._json(404, {"error": {"message": "not found"}})
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
                self._json(503, {"error": {"type": "server_error", "message": f"{e}; the next request restarts it"}})
            except EngineStuck as e:                         # the engine could not be ended
                self._json(503, {"error": {"type": "server_error", "message": str(e)}})

            finally:
                if self.watch_done is not None:
                    self.watch_done.set()

        def _props(self):
            model = parse_qs(urlsplit(self.path).query).get("model", [svc.model])[0]
            if model != svc.model:
                self._json(404, {"error": {"message": "model not found"}})
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
                     "modalities": {"vision": svc.vision is not None}, "models_autoload": hasattr(svc.engine, "restart"),
                     "is_sleeping": not svc.loaded()}
            if getattr(svc.engine, "model_path", None):
                props["model_path"] = svc.engine.model_path
            version = getattr(svc.engine, "info", {}).get("version")
            if version:
                props["build_info"] = "basalt " + str(version)
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
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(415, {"error": {"message": "send application/json"}})
                return False
            # Origin must be this server's address or a trusted origin; proxy-added headers prove nothing
            origin = (self.headers.get("Origin") or "").rstrip("/")
            if origin and origin.split("://", 1)[-1] != self.headers.get("Host", "") and \
                    origin not in svc.trusted_origins:
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
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")   # proxies pass each event at once
            self._cors()
            self.end_headers()

        def _unknown_model(self, path, name):
            """404 for a missing or unserved model, in the shape of the API called."""
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
            tools, force = forced_call(req.get("tool_choice"), tools)   # a bad choice is a 400 before streaming
            if check_format(req.get("response_format")) and tools:
                raise ValueError("structured response_format with tools is not supported")
            svc.require_grammar(req.get("response_format"))
            svc.load()
            req[BIAS_KEY] = checked_bias(req, svc.engine)     # a bad logit_bias is a 400 before streaming
            req[GRAMMAR_KEY] = svc.grammar_for(tools, req.get("tool_choice"), req.get("tools"),
                                               req.get("response_format"))
            max_new = int(req.get("max_completion_tokens") or req.get("max_tokens") or 0)   # 0/-1: the rest of the context
            svc.reasoning_budget(req)                         # a bad value is a 400 before streaming
            stop_strings(req)                                 # likewise a bad stop
            ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new, force=force)
            _debug_req("openai", req, messages, tools, max_new, thinking, len(ids))
            cancel = threading.Event()
            self._watch_client(cancel)
            chunks = primed(openai_chunks(svc, req, ids, thinking, tools, max_new, cancel, force=force), ids)
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
                err = {"error": {"type": "server_error", "message": f"{e}; the next request restarts it"}}
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
            cancel = threading.Event()
            self._watch_client(cancel)

            def events():
                yield from asm.start()
                done = None
                for kind, x in svc.run(ids, thinking, tools, max_new, req, cancel, force=force):
                    if kind == "ping":
                        yield None
                    elif kind == "event":
                        yield from asm.feed(x)
                    elif kind == "done":
                        done = x
                if done is not None and done["finish"] != "cancel" and not cancel.is_set():
                    yield from asm.finish(done)
            try:
                items = primed(events(), ids)
            except ValueError as e:                          # too long for the context: found when the engine counted
                err = ResponsesError(str(e), "input", "context_length_exceeded" if "context" in str(e) else None)
                return self._json(err.status, err.body())
            if not req.get("stream"):
                try:
                    result = responses_api.collect(items)
                except EngineDied as e:
                    return self._json(503, responses_error_body(f"{e}; the next request restarts it", "server_error",
                                                                code="server_error"))
                except ValueError as e:                      # the engine's ERR line
                    return self._json(500, responses_error_body(str(e), "server_error", code="server_error"))
                return self._json(200, result)
            self._sse()
            last = time.monotonic()

            def send(e):
                self.wfile.write(f"event: {e['type']}\n".encode() + b"data: " +
                                 json.dumps(e, ensure_ascii=False).encode() + b"\n\n")

            try:
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
                items.close()
            except EngineDied as e:                          # mid-stream: response.failed
                self._responses_failed(asm, f"{e}; the next request restarts it", "server_error", send)
            except ValueError as e:                          # engine ERR after the stream started
                self._responses_failed(asm, str(e), "server_error", send)

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
                tools, force = forced_call(req.get("tool_choice"), tools)
            except ValueError as e:
                raise ResponsesError(str(e), "tool_choice") from None
            kw = responses_api.template_kwargs(req)
            try:
                structured = check_format(responses_api.text_format(req))
            except ValueError as e:
                raise ResponsesError(str(e).replace("response_format", "text.format"), "text.format") from None
            if structured and tools:
                raise ResponsesError("a JSON text.format with tools is not supported", "text.format",
                                     "unsupported_parameter")
            try:
                svc.require_grammar(responses_api.text_format(req))
            except ValueError as e:
                raise ResponsesError(str(e).replace("response_format", "text.format"), "text.format") from None
            req[GRAMMAR_KEY] = svc.grammar_for(tools, req.get("tool_choice"), req.get("tools"),
                                               responses_api.text_format(req))
            noted = svc.__dict__.setdefault("hosted_tools_noted", set())   # once per tool
            if set(skipped) - noted:
                print(f"[server] warning: /v1/responses ignores hosted tools {', '.join(sorted(set(skipped) - noted))}", flush=True)
                noted.update(skipped)
            try:
                svc.reasoning_budget(req)                    # a bad value is a 400 before streaming
            except ValueError as e:
                raise ResponsesError(str(e), "reasoning_budget_tokens") from None
            try:
                stop_strings(req)
            except ValueError as e:
                raise ResponsesError(str(e), "stop") from None
            svc.load()
            try:
                ids, thinking, max_new = svc.prepare(messages, tools, kw, req.get("max_output_tokens") or 0, force=force)
            except ResponsesError:
                raise
            except ValueError as e:                          # too long for the context, or an image without vision
                raise ResponsesError(str(e), "input", "context_length_exceeded" if "context" in str(e) else None) \
                    from None
            _debug_req("responses", req, messages, tools, max_new, thinking, len(ids))
            include = req.get("include") if isinstance(req.get("include"), list) else []
            asm = responses_api.Assembler(req, svc.model, len(ids), names,
                                          "reasoning.encrypted_content" in include)
            return ids, thinking, tools, max_new, asm, req, force

        def _count_tokens(self, req):
            """Anthropic's token count: the rendered, tokenized prompt, without running the model."""
            messages, tools, kw = anthropic_to_messages(req, svc.anthropic_think_unasked)
            tools, _ = forced_call(req.get("tool_choice"), tools)   # same 400 as the request itself ("none": no tools)
            stop_strings(req)                                 # same 400 as the request itself
            self._json(200, {"input_tokens": len(svc.encode_prompt(messages, tools, kw))})

        def _anthropic(self, req):
            svc.load()
            messages, tools, kw = anthropic_to_messages(req, svc.anthropic_think_unasked)
            tools, force = forced_call(req.get("tool_choice"), tools)   # "any" / {"type": "tool", "name": N}
            max_new = int(req.get("max_tokens") or 0)                  # 0/-1: the rest of the context
            response_format = anthropic_format(req)            # structured outputs
            if response_format and tools:
                raise ValueError("a JSON output format with tools is not supported")
            try:
                svc.require_grammar(response_format)
            except ValueError as e:
                raise ValueError(str(e).replace("response_format json_schema and json_object", "output_config.format json_schema")) from None
            req[GRAMMAR_KEY] = svc.grammar_for(tools, req.get("tool_choice"), req.get("tools"), response_format)
            svc.reasoning_budget(req)                         # a bad value is a 400 before streaming
            stop_strings(req)                                 # likewise a bad stop
            ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new, force=force)
            _debug_req("anthropic", req, messages, tools, max_new, thinking, len(ids))
            cancel = threading.Event()
            self._watch_client(cancel)
            events = primed(anthropic_events(svc, req, ids, thinking, tools, max_new, cancel, force=force), ids)
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
                err = {"type": "error", "error": {"type": "api_error", "message": f"{e}; the next request restarts it"}}
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")
            except ValueError as e:                          # engine ERR after the stream started
                err = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")

    return Handler


class Server(ThreadingHTTPServer):
    allow_reuse_address = True
    request_queue_size = 256      # socketserver's backlog of 5 resets some clients of a burst of simultaneous connections

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], ConnectionError):   # a hang-up needs no stack trace
            super().handle_error(request, client_address)


def serve(svc: Service, host="127.0.0.1", port=8080) -> ThreadingHTTPServer:
    svc.host_names = host_names_for(host, svc.allowed_hosts, svc.trusted_origins)
    svc.start_telemetry()
    httpd = Server((host, port), make_handler(svc))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd
