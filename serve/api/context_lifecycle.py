"""The agent-context lifecycle of the solo serve session.

Three parties conduct one context: the registry (basalt-thread-api) owns the FSM and leases, the engine owns
the bytes (`CONTEXT_*` commands, see src/program/serve/context_cmds.cpp), and this module runs one request's
attach -> restore -> publish dance between them, plus the idle reaper that cools HOT contexts to disk and the
tombstone drain that frees a deleted context's payload. The registry locator carries the engine's snapshot
meta verbatim: `g<generation>,<meta>`.

No context path may fail a request: every registry or engine refusal degrades to the transcript replay the
caller was already prepared to send. All of it is off unless the registry adapter is configured.
"""
from __future__ import annotations

import hashlib
import shutil
import os
import tempfile
import threading
import time

from serve.api.context_adapter import ContextAdapter, RegistryError
from serve.engine.errors import EngineDied
from serve.prompt import PromptText

_SNAP_WORDS = 8       # CONTEXT_OK SNAP <id> <gen> <bytes> <tokens> <prefix> <meta>


class _Handle:
    """One request's lease on one context: mode is hot | restoring | replay for settle."""
    __slots__ = ("cid", "lease", "version", "mode", "gen")

    def __init__(self, cid: str, lease: int | None, version: int, mode: str, gen: int):
        self.cid, self.lease, self.version, self.mode, self.gen = cid, lease, version, mode, gen


class ContextLifecycle:
    def __init__(self, service, adapter: ContextAdapter):
        self.svc = service
        self.ad = adapter
        self.cool_idle_s = float(os.environ.get("BASALT_CONTEXT_COOL_IDLE_S", "600") or 0)
        # publishing a snapshot onto a nearly full disk is how a host dies: refuse it fail-closed
        self.min_free_disk_bytes = int(float(os.environ.get("BASALT_CONTEXT_MIN_FREE_DISK_GIB", "16") or 0)) * (1 << 30)
        self.store_dir = os.environ.get("BASALT_CONTEXT_STORE_DIR") or ""
        self.lock = threading.Lock()
        self.fp: str | None = None
        self.epoch: str | None = None
        self.proc_key = object()          # the engine process the fingerprint belongs to
        self.live_id: str | None = None   # the context the engine session's conversation currently is
        self.hot: dict[str, dict] = {}    # HOT contexts: {"used": monotonic, "gen": int}
        self.drops: set[str] = set()      # tombstoned ids whose engine payload is not yet freed
        self._stop = threading.Event()
        if self.cool_idle_s > 0:
            threading.Thread(target=self._reap_loop, daemon=True, name="basalt-ctx-reaper").start()

    # -- engine identity -------------------------------------------------------
    def _identity(self) -> "tuple[str, str]":
        """(model_fingerprint, engine_epoch) of the running engine, fetched once per process."""
        proc = getattr(self.svc.engine, "proc", None)
        if self.fp is None or proc is not self.proc_key:
            digest = hashlib.sha256(self.svc.template.source.encode("utf-8")).hexdigest()
            words = self.svc.engine.context_command("CONTEXT_FINGERPRINT " + digest).split()
            self.fp, self.epoch, self.proc_key = words[2], words[3], proc
            with self.lock:                 # a new engine process holds no conversation
                self.live_id = None
        return self.fp, self.epoch

    def _disk_ok(self) -> bool:
        if not (self.store_dir and self.min_free_disk_bytes):
            return True
        try:
            return shutil.disk_usage(self.store_dir).free >= self.min_free_disk_bytes
        except OSError:
            return False

    # -- one request -----------------------------------------------------------
    def attach(self, req, ids) -> "_Handle | None":
        """Lease the request's bound context and, when the registry points at a snapshot, load it into the
        engine. Returns the settle handle (possibly mode "replay") or None. Never raises."""
        if not isinstance(req, dict):
            return None
        try:
            return self._attach(req, ids)
        except (RegistryError, ValueError, EngineDied, OSError) as e:
            print(f"[server] context: attach fell back to replay ({e})", flush=True)
            return None

    def _attach(self, req, ids):
        res = self.ad.attach(req)
        if res.mode != "context":           # no binding, or the registry said replay outright
            return None
        cid = res.context_id
        detail = self.ad._call("GET", f"/v1/contexts/{cid}", None, None, (200,))
        state = detail.get("state")
        locator = detail.get("locator")
        gen, meta = self._locator(locator)
        position = int((detail.get("position") or {}).get("prefix_tokens") or 0)
        h = _Handle(cid, res.lease_token, int(detail.get("version") or 0), "replay", gen)
        with self.lock:
            live = self.live_id == cid
        if state == "HOT" and live:         # the engine session still holds this conversation: continue it
            h.mode = "hot"
            return h
        if state == "HOT" and not locator:  # lost with nothing on disk: materialize is illegal from HOT
            self._release(h)
            return None
        if not meta or position <= 0:
            return h                        # nothing published yet (or nothing to restore): replay, publish later
        try:
            _fp, epoch = self._identity()
            if state == "HOT":              # a dead engine's HOT: demote it to COLD under our lease, then restore
                self._release(h)
                self._transition(h, "cool", lease=False)
                self._transition(h, "chill", lease=False)
                h.lease = self._resume(cid)
            elif state == "COOLING":
                self._transition(h, "chill", lease=False)
            if state in ("HOT", "COOLING", "WARM", "COLD"):
                self._transition(h, "restore", patch={"engine_epoch": epoch})
                if self._engine_restore(h, meta, ids, position):
                    h.mode = "restoring"
                    return h
                self._transition(h, "fail")  # the snapshot no longer fits: replay, and republish at settle
        except RegistryError as e:
            print(f"[server] context: restore path fell back to replay ({e.reason})", flush=True)
            self._fail_quietly(h)
        return h

    def settle(self, h: "_Handle | None", ok: bool) -> None:
        """Close one request: publish what the engine now holds, mark the FSM, release the lease. Never raises."""
        try:
            self._settle(h, ok)
        except (RegistryError, ValueError, EngineDied, OSError) as e:
            print(f"[server] context: settle gave up ({e})", flush=True)
        finally:
            if h is not None and h.lease is not None:
                try:
                    self.ad.release(h.cid, h.lease)
                except (RegistryError, OSError):
                    pass                    # the lease expires on its own
                h.lease = None

    def _settle(self, h, ok):
        if h is None:
            with self.lock:                 # an unbound request moved the session's conversation on
                self.live_id = None
            return
        if not ok:
            if h.mode == "restoring":       # the restored conversation never completed: rebuild from transcript
                try:
                    self._transition(h, "fail")
                except RegistryError as e:
                    print(f"[server] context: fail after a broken request did not land ({e.reason})", flush=True)
            with self.lock:
                if self.live_id == h.cid:
                    self.live_id = None
            return
        if h.mode == "restoring":
            _fp, epoch = self._identity()
            self._transition(h, "ready", patch={"engine_epoch": epoch})   # identity: the stored one must hold
            self._mark_hot(h.cid, h.gen)
        elif h.mode == "replay":
            fp, epoch = self._identity()
            gen = h.gen + 1
            if not self._disk_ok():
                print("[server] context: disk budget full, publishing skipped (replay stays correct)", flush=True)
                return
            words = self.svc.engine.context_command(f"CONTEXT_SNAP {h.cid} {gen}").split()
            if len(words) != _SNAP_WORDS:
                raise ValueError("the engine's SNAP answer is malformed")
            tokens, prefix, meta = int(words[5]), words[6], words[7]
            self._transition(h, "materialize")
            self._transition(h, "ready", patch={
                "model_fingerprint": fp, "prefix_hash": prefix, "prefix_tokens": tokens,
                "locator": f"g{gen},{meta}", "validated": True, "engine_epoch": epoch})
            self._mark_hot(h.cid, gen)
        elif h.mode == "hot":
            self._mark_hot(h.cid, h.gen)    # the freshest bytes stay in RAM until the reaper cools them

    # -- tombstones --------------------------------------------------------------
    def note_delete(self, cid: str) -> None:
        """A DELETE landed: the row is tombstoned, the engine's payload has to go with it."""
        with self.lock:
            self.drops.add(cid)
            self.hot.pop(cid, None)
            if self.live_id == cid:
                self.live_id = None
        self.drain_drops()

    REAPER_FIFO_WAIT_S = 25.0               # a reaper round's patience for the solo turn

    def drain_drops(self, fifo_wait_s: float = 0.0) -> None:
        """Free tombstoned payloads while the engine sits idle, then confirm to the registry so GC can purge.

        The reaper passes a bounded wait: under sustained agent load the turn lock passes straight from one
        request to the next and a non-blocking probe loses every race, so tombstones would never drain.
        """
        with self.lock:
            pending = list(self.drops)
        if not pending or not self.svc.loaded():
            return
        if fifo_wait_s > 0:
            got = self.svc.fifo.acquire(timeout=fifo_wait_s)
        else:
            got = self.svc.fifo.acquire(blocking=False)
        if not got:
            return                          # a request holds the turn: the caller retries
        try:
            for cid in pending:
                try:
                    words = self.svc.engine.context_command(f"CONTEXT_DROP {cid}", timeout=120.0).split()
                    print(f"[server] context: {cid[:8]} engine freed {words[3] if len(words) > 3 else '?'} files", flush=True)
                    if self.store_dir:
                        try:
                            os.rmdir(os.path.join(self.store_dir, cid))   # the engine freed the files; the empty folder goes with them
                        except OSError:
                            pass
                except (ValueError, EngineDied) as e:
                    print(f"[server] context: engine drop of {cid[:8]} refused ({e})", flush=True)
                try:
                    self.ad._call("POST", f"/v1/contexts/{cid}/payload/released", None, None, (200,))
                except RegistryError as e:
                    if e.status == 404:
                        with self.lock:
                            self.drops.discard(cid)   # the row is already purged: nothing left to confirm
                        continue
                    print(f"[server] context: release of {cid[:8]} did not land ({e.reason}); retrying", flush=True)
                    continue                # registry unreachable: retry on the next round
                except OSError as e:
                    print(f"[server] context: release of {cid[:8]} did not land ({e}); retrying", flush=True)
                    continue
                with self.lock:
                    self.drops.discard(cid)
        finally:
            self.svc.fifo.release()

    # -- the idle reaper -----------------------------------------------------------
    def _reap_loop(self) -> None:
        while not self._stop.wait(30.0):
            try:
                self.drain_drops(self.REAPER_FIFO_WAIT_S)
                self._cool_idle()
            except Exception as e:          # a daemon thread must outlive any one round
                print(f"[server] context: reaper round failed ({type(e).__name__}: {e})", flush=True)

    def _cool_idle(self) -> None:
        now = time.monotonic()
        with self.lock:
            stale = [cid for cid, i in self.hot.items() if now - i["used"] > self.cool_idle_s]
        if not stale or not self.svc.loaded():
            return
        if not self.svc.fifo.acquire(timeout=self.REAPER_FIFO_WAIT_S):
            return
        try:
            for cid in stale:
                try:
                    self._cool_one(cid)
                except (RegistryError, ValueError, EngineDied, OSError) as e:
                    print(f"[server] context: cooling {cid[:8]} gave up ({e})", flush=True)
        finally:
            self.svc.fifo.release()

    def _cool_one(self, cid: str) -> None:
        detail = self.ad._call("GET", f"/v1/contexts/{cid}", None, None, (200,))
        state = detail.get("state")
        if state not in ("HOT", "COOLING"):
            with self.lock:
                self.hot.pop(cid, None)
            return
        if int((detail.get("refs") or {}).get("leases") or 0) > 0:
            return                          # a request just leased it: try again next round
        h = _Handle(cid, None, int(detail.get("version") or 0), "cool", 0)
        with self.lock:
            live = self.live_id == cid
        if state == "HOT" and live:
            try:                            # the session still holds it: publish the freshest generation first
                if self._disk_ok():
                    gen = h.gen + 1
                    words = self.svc.engine.context_command(f"CONTEXT_SNAP {cid} {gen}").split()
                    if len(words) != _SNAP_WORDS:
                        raise ValueError("the engine's SNAP answer is malformed")
                    self._transition(h, "cool", lease=False)
                    self._transition(h, "chill", patch={"locator": f"g{gen},{words[7]}"}, lease=False)
                    with self.lock:
                        self.hot.pop(cid, None)
                        if self.live_id == cid:
                            self.live_id = None
                    return
            except (ValueError, EngineDied) as e:
                print(f"[server] context: refresh before cooling {cid[:8]} failed ({e})", flush=True)
                                        # fall through: cool at the generation the registry already has
        if state == "HOT":
            self._transition(h, "cool", lease=False)
        self._transition(h, "chill", lease=False)   # COOLING -> COLD keeps the published locator
        with self.lock:
            self.hot.pop(cid, None)
            if self.live_id == cid:
                self.live_id = None

    # -- internals -----------------------------------------------------------------
    @staticmethod
    def _locator(locator) -> "tuple[int, str]":
        """`g<generation>,<meta>` back into its parts; anything else is (0, "")."""
        if isinstance(locator, str) and locator.startswith("g") and "," in locator:
            gen_text, meta = locator[1:].split(",", 1)
            if gen_text.isdigit() and meta:
                return int(gen_text), meta
        return 0, ""

    def _mark_hot(self, cid: str, gen: int) -> None:
        with self.lock:
            self.live_id = cid
            self.hot[cid] = {"used": time.monotonic(), "gen": gen}

    def _release(self, h: _Handle) -> None:
        if h.lease is not None:
            try:
                self.ad.release(h.cid, h.lease)
            except (RegistryError, OSError):
                pass
            h.lease = None

    def _fail_quietly(self, h: _Handle) -> None:
        if h.mode == "restoring":
            try:
                self._transition(h, "fail")
            except RegistryError:
                pass
            h.mode = "replay"

    def _resume(self, cid: str) -> int:
        res = self.ad._call("POST", f"/v1/contexts/{cid}/resume", {}, None, (200,))
        token = (res.get("lease") or {}).get("token")
        if isinstance(token, bool) or not isinstance(token, int):
            raise RegistryError("invalid_response")
        return token

    def _transition(self, h: _Handle, event: str, patch: dict | None = None, lease: bool = True) -> dict:
        body = {"event": event, "expected_version": h.version, "lease_token": h.lease if lease else None}
        if patch:
            body.update(patch)
        out = self.ad._call("POST", f"/v1/contexts/{h.cid}/transition", body, None, (200,))
        h.version = int(out.get("version") or h.version + 1)
        return out

    def _engine_restore(self, h: _Handle, meta: str, ids, position: int) -> bool:
        """Load the published prefix into the engine; False means replay (a mismatch, or no engine)."""
        try:
            token_ids = ids.ids() if isinstance(ids, PromptText) else list(ids)
        except Exception:
            return False
        if position > len(token_ids):
            print(f"[server] context: {h.cid[:8]} the transcript holds {len(token_ids)} tokens, fewer than the published {position}", flush=True)
            return False                    # the client's prompt is not an extension of the published prefix
        fd, path = tempfile.mkstemp(prefix="basalt-ctx-", suffix=".ids")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(",".join(str(t) for t in token_ids[:position]) + "\n")
            self.svc.engine.context_command(f"CONTEXT_RESTORE {h.cid} {h.gen} {path} {meta}")
            return True
        except (ValueError, EngineDied, OSError) as e:
            print(f"[server] context: {h.cid[:8]} the engine refused the restore ({e})", flush=True)
            return False
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
