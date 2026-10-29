"""OpenAI, Anthropic and Responses endpoints over an engine that maps token ids to tokens.

    basalt serve --model FILE.basalt [--alias NAME] [--host H] [--port P] [--gpu 0,1] [--api-key KEY] ...
    python -m serve.server --model FILE.basalt --engine-bin bin/basalt ...
    python -m serve.server --config config.json
    python -m serve.server --mock-engine --port 8080

The engine boundary is `Engine.generate(prompt_ids, max_new, sampling, cancel)` -> token ids. ProcessEngine keeps
one `basalt --serve` process resident; MockEngine is a scripted stand-in for tests. Prompts that do not fit the
context are rejected with a 400, never truncated.
"""
from __future__ import annotations

import json
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # also runnable as a script
from serve import ROOT, service  # noqa: E402
from serve.config import (build_parser, check_engine_flags,  # noqa: E402
                          engine_flags_of, engine_line, grammar_words, parse_settings, sampling_defaults_from_config,
                          sampling_line)
from serve.engine import process as engine_process  # noqa: E402
from serve.engine.errors import EngineStartError  # noqa: E402
from serve.engine.flags import flag_index, flag_value  # noqa: E402
from serve.engine.mock import MockEngine  # noqa: E402
from serve.engine.process import ProcessEngine  # noqa: E402
from serve.engine.vision import (EngineVision, Vision, check_gpu, child_env, engine_args, gpu_plan,  # noqa: E402
                                 gpu_text, gpus_in_play, vision_card, vision_env, vision_line)
from serve.frontend import ChatTemplate  # noqa: E402
from serve.handler import Server, serve  # noqa: E402
from serve.security import (LOOPBACK_NAMES, allowed_hosts_of, api_key_of, exposed_without_key,  # noqa: E402
                            lan_addresses, origins_of)
from serve.service import Service  # noqa: E402
from serve.util.logbuf import ColorStream, LogBuffer, TeeStream, tail_file  # noqa: E402
from serve.util.console import LEGEND, tagged  # noqa: E402
from serve.util.text import ByteTokenizer  # noqa: E402


def fail(msg: str) -> None:
    """Print a startup error and exit with status 2."""
    print(f"[server] error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def warn_tight_ram(arena_mib) -> None:
    """Warn at start when less than ~6 GB of RAM is left beside the expert arena (INFO arena_mib)."""
    if not isinstance(arena_mib, int) or arena_mib <= 0:
        return
    try:
        import psutil
        total = psutil.virtual_memory().total
    except Exception:  # noqa: BLE001 - psutil is optional here
        return
    left = total / 2**30 - arena_mib / 1024
    if left < 6:
        print(tagged("warning", f"low RAM: {left:.1f} of {total / 2**30:.0f} GB left after experts"), flush=True)


def grammar_entry(on: bool, cli: list[str], used: list[str]) -> tuple[bool, str] | None:
    """(on, origin) for the startup line when the command line or the config file set grammar; None for the default."""
    names = ("--grammar", "--no-grammar")
    origin = "cli" if any(w in names for w in cli) else "config" if any(w in names for w in used) else ""
    return (on, origin) if origin else None


def main() -> int:
    ap = build_parser()
    try:
        a, used_argv, file_engine, cli_engine = parse_settings(ap, sys.argv[1:])
    except ValueError as e:
        fail(f"--config: {e}")
    service.DEBUG, engine_process.REQUEST_LINES = a.debug, a.request_lines
    logs = LogBuffer()                                  # this process's and the engine's output
    sys.stdout, sys.stderr = (TeeStream(ColorStream.wrap(sys.stdout), logs, "srv"),
                              TeeStream(ColorStream.wrap(sys.stderr), logs, "srv"))
    print(tagged("legend", LEGEND), flush=True)          # first line, so the key precedes every value
    cfg = {}                                            # what the model file implies
    try:
        check_gpu(a.gpu)
    except ValueError:
        fail(f"--gpu {gpu_text(a.gpu)}: expected numbers like 0 or 0,1, or auto")
    if not a.mock_engine and not (a.model and a.engine_bin):
        fail("the server needs --model and the engine binary: run it as basalt serve --config FILE (or --mock-engine)")
    if (file_engine or cli_engine) and not a.model:
        fail("engine flags need --model")
    if a.model:
        if not a.engine_bin:
            fail("--model needs --engine-bin BINARY")
        from serve.engine import modelfile
        try:
            known = engine_flags_of(a.engine_bin)
            check_engine_flags(file_engine, known, f"--config {a.config}")
            check_engine_flags(cli_engine, known, "command line")
        except ValueError as e:
            fail(str(e))
        mode = a.vision
        vision_arg = False if a.mock_engine or mode == "off" else mode
        try:
            cfg.update({k: v for k, v in modelfile.model_config(
                a.model, a.engine_bin, vision=vision_arg,
                engine_words=file_engine + cli_engine).items()
                if k not in cfg})
        except (OSError, ValueError, KeyError) as e:
            fail(f"{a.model}: {e}")
        if not a.mock_engine:
            # the engine's own choice, printed once here; the engine then sees only these cards, main first
            try:
                plan = gpu_plan(a.engine_bin, a.gpu)
            except ValueError as e:
                fail(f"gpu: {e}")
            card = vision_card(mode)
            if card is not None:                       # a numbered sidecar card, checked like gpu before it starts
                try:
                    gpu_plan(a.engine_bin, str(card))
                except ValueError as e:
                    fail(f"vision: {e}")
            for note in plan.get("notes") or []:
                print(f"[server] {note}", flush=True)
            if plan.get("warning"):
                print(f"[server] warning: {plan['warning']}", flush=True)
            cfg["gpu"] = plan["devices"]
            cfg["gpu_smi"] = [c.get("smi", -1) for c in plan.get("cards") or []]
            cfg["args"] += ["--gpu", ",".join(str(i) for i in range(len(plan.get("cards") or [])))]
        if not a.mock_engine:
            print(engine_line(file_engine, cli_engine,
                                    (a.shared_context, a.parallel) if getattr(a, "shared_context", None) else None,
                                    grammar_entry(a.grammar, sys.argv[1:], used_argv)), flush=True)
    if a.alias:
        cfg["model_name"] = a.alias
    try:                                                # fail fast if the port is taken
        Server((a.host, a.port), BaseHTTPRequestHandler).server_close()
    except OSError:
        fail(f"port {a.port} is in use (--port N)")
    if cfg.get("tokenizer"):
        a.tokenizer = cfg["tokenizer"]
    tok = ByteTokenizer()
    tpath = Path(a.tokenizer)
    if not a.mock_engine and not (tpath / "vocab.json").exists():
        fail(f"tokenizer missing: {tpath / 'vocab.json'}")
    if (tpath / "vocab.json").exists():
        import tokenizer as ST
        vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
        types = json.loads((tpath / "token_type.json").read_text())
        tok = ST.Tokenizer(tokens, merges, types)
    if not a.mock_engine:
        vision = None
        env = child_env(cfg)
        sampling_defaults = {**sampling_defaults_from_config(cfg), **a.sampling_launch}
        if sampling_defaults:
            print(sampling_line(sampling_defaults, a.sampling_cli, set(a.sampling_launch) - a.sampling_cli), flush=True)
        lazy = a.lazy
        if lazy and cfg.get("vision"):
            print(tagged("warning", "lazy loading is text-only, vision off"), flush=True)
            while (at := flag_index(cfg["args"], "--vision-mmproj")) is not None:
                del cfg["args"][at:at + 2]
            del cfg["vision"]
        if cfg.get("vision"):
            mode = cfg["vision"].get("mode")
            if mode == "dynamic":
                vision = EngineVision(cfg["vision"])
            else:
                vision = Vision(cfg["vision"], log=open(cfg["log"], "a", encoding="utf-8") if cfg.get("log") else None,
                                env=vision_env(env, mode))
        print(vision_line(cfg.get("vision"), cfg, vision.ready_s if vision else 0.0), flush=True)
        if lazy:
            print(tagged("model", "unloaded, the first request loads it"), flush=True)
        if cfg.get("log"):                              # follow the engine's log file
            threading.Thread(target=tail_file, args=(logs, cfg["log"], "eng"), daemon=True).start()
        else:
            ProcessEngine.stderr_sink = lambda line: logs.add("eng", line)
        known_flags = engine_flags_of(cfg["engine_bin"])
        has_grammar = known_flags is not None and "--grammar" in known_flags
        if a.grammar and not has_grammar:
            print(tagged("warning", "grammar is on but this engine has no --grammar; requests go without one"), flush=True)
        gwords = grammar_words(known_flags)
        cdir = flag_value(engine_args(cfg), "--conversation-cache-dir")
        cmib = flag_value(engine_args(cfg), "--conversation-cache-disk-mib") or "0"
        if cdir and cmib != "0":
            print(tagged("disk", f"conversations={cdir} disk-mib={cmib}"), flush=True)
        # always built unloaded: the load itself (below) is the lazy path's, started here instead of by a request
        engine = ProcessEngine(cfg["engine_bin"], engine_args(cfg) + gwords, log=cfg.get("log"), env=env, lazy=True)
        if isinstance(vision, EngineVision):
            vision.attach(engine)
    else:
        engine, vision, sampling_defaults = MockEngine(tok, a.script or [
            "Thinking about it.</think>\n\nHello from the mock engine."], batch=a.parallel or 0), None, dict(a.sampling_launch)
    # the model's exported template, else the bundled one
    tpl = tpath / "chat_template.jinja"
    svc = Service(engine, tok, ChatTemplate(tpl if tpl.exists() else ROOT / "serve/chat_template.jinja"),
                  model_name=cfg.get("model_name", "qwen3.8-flash-next"), vision=vision,
                  sampling_defaults=sampling_defaults, fit_max_tokens=a.fit_max_tokens, grammar=a.grammar,
                  grammar_engine=bool(a.mock_engine or has_grammar))
    svc.logs = logs
    if any(x == "--api-key" and i + 1 < len(used_argv) and not used_argv[i + 1].strip() or x.strip() == "--api-key="
           for i, x in enumerate(used_argv)):
        # an empty key would silently disable authentication
        print("[server] error: --api-key is empty (give a key, or leave it out)", file=sys.stderr)
        return 2
    try:
        svc.api_key = api_key_of(a.api_key)
    except ValueError as e:
        print(f"[server] error: {e}: give a key, or leave --api-key out", file=sys.stderr)
        return 2
    try:
        svc.cors_origins = origins_of(a.cors_origin, "--cors-origin", wildcard=True)
        svc.allowed_hosts = allowed_hosts_of(a.allowed_host)
    except (ValueError, SystemExit) as e:
        fail(str(e).removeprefix("[server] error: "))
    if svc.allowed_hosts:
        print(tagged("warning", "Host check off, any name is accepted (--allowed-host *)") if "*" in svc.allowed_hosts
              else tagged("hosts", ",".join(svc.allowed_hosts)), flush=True)
    if svc.cors_origins:
        print(tagged("cors", ",".join(svc.cors_origins)), flush=True)
        if "*" in svc.cors_origins and not svc.api_key:
            print(tagged("warning", "any web page can use the model (--api-key KEY)"), flush=True)
    svc.idle_unload_s = a.idle_unload or 0
    svc.idle_sleep_s = a.idle_sleep or 0
    svc.min_free_vram_mib = a.min_free_vram_mib or 0
    svc.reasoning_budget_tokens = a.reasoning_budget_tokens or 0
    svc.answer_room = a.answer_room
    svc.gpu_indices = gpus_in_play(cfg, engine_args(cfg) if not a.mock_engine else [])   # the cards /metrics reads
    svc.gpu_index = svc.gpu_indices[0]
    httpd = serve(svc, host=a.host, port=a.port)
    svc.start_idle_unload()
    svc.start_idle_sleep()
    here = "127.0.0.1" if a.host in ("0.0.0.0", "", "::") else a.host

    def announce():
        if not a.mock_engine and not lazy:
            warn_tight_ram(engine.info.get("arena_mib"))
        if getattr(engine, "batch", 0):
            print(tagged("slots", f"par={engine.batch} ctx/slot={engine.max_context}"
                         + (" (slot-full-context)" if a.slot_full_context else "") + ", others queue"), flush=True)
        tail = f"ctx={engine.max_context} images={'on' if vision else 'off'}" + (" api-key=on" if svc.api_key else "")
        print(tagged("ready", f"http://{here}:{a.port}/v1  openai+anthropic  {tail}"), flush=True)

    print(tagged("stats", f"http://{here}:{a.port}/"), flush=True)
    if a.mock_engine or lazy:
        announce()
    if a.host not in LOOPBACK_NAMES:
        # reachable from other devices: print the addresses
        ips = lan_addresses()
        for ip in ips:
            print(tagged("network", f"http://{ip}:{a.port}/v1"), flush=True)
        if not ips:
            print(tagged("network", f"port={a.port}"), flush=True)
        if exposed_without_key(a.host, svc.api_key):
            print(tagged("warning", "no api key, open to the network (--api-key KEY)"), flush=True)
    if not a.mock_engine and not lazy:
        svc.start_load(announce)                        # the page and the API are up; requests wait for the load
    if a.open:
        import webbrowser
        webbrowser.open(f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else a.host}:{a.port}/")
    # SIGTERM takes Ctrl+C's path (QUIT to the engine) so a container stop does not orphan it
    def on_sigterm(signum, frame):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, on_sigterm)
    except (ValueError, OSError, AttributeError):         # not the main thread
        pass
    rc = 0
    try:
        while True:
            time.sleep(1)                               # a timed wait keeps Ctrl+C deliverable
            if svc.boot_error is not None:              # the start-up load failed: requests waiting on it get the error
                time.sleep(2)
                rc = 1
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        if not rc:
            print("[server] stopping (Ctrl+C again to force)", flush=True)
        closers = [svc.stop, httpd.shutdown, getattr(engine, "close", None), vision.close if vision else None]
        for close in filter(None, closers):
            try:
                close()
            except KeyboardInterrupt:                   # second Ctrl+C: do not wait for the engine
                if getattr(engine, "proc", None):
                    engine.proc.kill()
        print("[server] stopped", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
