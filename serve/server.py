"""OpenAI, Anthropic and Responses endpoints over engines that map token ids to tokens, one model loaded at a time.

    basalt serve --model FILE.basalt [--alias NAME] [--host H] [--port P] [--gpu 0,1] [--api-key KEY] ...
    python -m serve.server --model FILE.basalt --engine-bin bin/basalt ...
    python -m serve.server --config config.json          (each entry of its "models" is one model)
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
                          engine_flags_of, grammar_words, model_line, model_settings, sampling_defaults_from_config,
                          sampling_items, shared_line, slot_share, start_items, with_spec)
from serve.engine import process as engine_process  # noqa: E402
from serve.engine.errors import EngineStartError  # noqa: E402
from serve.engine.image_source import image_sources_of  # noqa: E402
from serve.engine.flags import flag_index, flag_value  # noqa: E402
from serve.engine.mock import MockEngine  # noqa: E402
from serve.engine.process import ProcessEngine  # noqa: E402
from serve.engine.vision import (EngineVision, Vision, check_gpu, child_env, engine_args, flag_refused, gpu_card_name,  # noqa: E402
                                 gpu_text, gpus_in_play, model_plan, vision_card, vision_env, vision_item)
from serve.models import dialect_for  # noqa: E402
from serve.handler import Server, serve  # noqa: E402
from serve.security import (LOOPBACK_NAMES, allowed_hosts_of, api_key_of, exposed_without_key,  # noqa: E402
                            lan_addresses, origins_of)
from serve.router import Router  # noqa: E402
from serve.service import Service  # noqa: E402
from serve.util.logbuf import ColorStream, LogBuffer, TeeStream, tail_file  # noqa: E402
from serve.util.console import LEGEND, tagged  # noqa: E402
from serve.util.text import ByteTokenizer  # noqa: E402


SWAP_START_TRIES = 2      # a swap's load: a second try for VRAM still being released, not a third for clients to wait on


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


def open_when_ready(svc, url, now=False, opener=None):
    """Open the page once, after the first weights load; at once when no load is under way (mock, lazy)."""
    if opener is None:
        import webbrowser
        opener = webbrowser.open
    if now:
        opener(url)
        return
    done, previous = [], svc.on_loaded

    def loaded():
        if previous is not None:
            previous()
        if not done:
            done.append(1)
            opener(url)
    svc.on_loaded = loaded


def check_model(a, file_engine: list[str], cli_engine: list[str], used_argv: list[str], known) -> dict:
    """What one configured model's file and flags imply, by the engine's own rules and without loading anything; fail()
    for what stops the start."""
    cfg = {}                                            # what the model file implies
    lines = cfg["start_lines"] = []                     # printed with the model's other lines once all are checked
    try:
        check_gpu(a.gpu)
    except ValueError:
        fail(f"--gpu {gpu_text(a.gpu)}: expected numbers like 0 or 0,1, or auto")
    if not a.mock_engine and not (a.model and a.engine_bin):
        fail("the server needs --model and the engine binary: run it as basalt serve --config FILE (or --mock-engine)")
    if (file_engine or cli_engine) and not a.model:
        fail("engine flags need --model")
    pack = bool(a.model) and not (a.mock_engine and a.config)   # the scripted engine of a config file reads no model files
    if pack:
        if not a.engine_bin:
            fail("--model needs --engine-bin BINARY")
        from serve.engine import modelfile
        try:
            check_engine_flags(file_engine, known, f"--config {a.config}")
            check_engine_flags(cli_engine, known, "command line")
        except ValueError as e:
            fail(str(e))
        mode = a.vision
        spec = modelfile.ModelSpec(a.model, alias=a.alias, gpu=a.gpu,
                                   vision=False if a.mock_engine or mode == "off" else mode,
                                   engine_words=tuple(file_engine + cli_engine))
        try:
            cfg.update(modelfile.model_config(spec, a.engine_bin))
        except (OSError, ValueError, KeyError) as e:
            fail(f"{a.model}: {e}")
        if not a.mock_engine:
            # the engine's own choice and its own flag rules, printed once here; the engine then sees only these cards,
            # main first
            try:
                plan = model_plan(a.engine_bin, a.model, spec.gpu, list(spec.engine_words))
            except ValueError as e:
                fail(f"{a.model}: {e}")
            card = vision_card(mode)
            if card is not None:                       # a numbered sidecar card: the engine's sm_120 rule is the
                if gpu_card_name(card) is None:        # engine's; the sidecar is its own build and says whether
                    fail(f"vision: GPU {card} is not found by nvidia-smi")   # the sidecar checks the card itself
            for note in plan.get("notes") or []:
                lines.append(f"[server] {note}")
            if plan.get("warning"):
                lines.append(f"[server] warning: {plan['warning']}")
            cfg["gpu"] = plan["devices"]
            cfg["spec"] = plan.get("spec")
            if getattr(a, "split_default", False) and plan.get("max_context"):
                # the engine's default context, shared by the slots
                a.shared_context = int(plan["max_context"])
                part = ["--max-context", str(slot_share(a.shared_context, a.parallel))]
                cfg["args"] += part
                cli_engine += part
            cfg["gpu_smi"] = [c.get("smi", -1) for c in plan.get("cards") or []]
            cfg["args"] += ["--gpu", ",".join(str(i) for i in range(len(plan.get("cards") or [])))]
    for flag in a.skipped:
        lines.append(tagged("engine", f"{flag[2:]} from the top level: this model has no such flag, skipped"))
    cfg["items"] = start_items(file_engine, cli_engine,
                               (a.shared_context, a.parallel) if getattr(a, "shared_context", None) else None,
                               grammar_entry(a.grammar, sys.argv[1:], used_argv), a.debug) if pack and not a.mock_engine else []
    cfg["items"] = with_spec(cfg["items"], cfg.get("spec"))
    if a.alias and not pack:
        cfg["model_name"] = a.alias
    if not cfg.get("model_name") and a.model:
        cfg["model_name"] = Path(a.model).name
    return cfg


def drop_sidecar_vision(cfg: dict) -> None:
    """Only a sidecar encoder holds VRAM before the first request; `dynamic` encodes inside the engine, which the
    request loads first anyway."""
    if not cfg.get("vision") or cfg["vision"].get("mode") == "dynamic":
        return
    print(tagged("warning", "lazy loading drops a sidecar image encoder, vision off; --vision dynamic stays on"), flush=True)
    while (at := flag_index(cfg["args"], "--vision-mmproj")) is not None:
        del cfg["args"][at:at + 2]
    del cfg["vision"]


def build_model(a, cfg: dict, logs, multi: bool, known) -> Service:
    """The Service of one checked model, unloaded. With several models a numbered vision sidecar waits for its model's
    turn too."""
    tokenizer = cfg.get("tokenizer") or a.tokenizer
    tok = ByteTokenizer()
    tpath = Path(tokenizer)
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
        pre = json.loads((tpath / "tokenizer.json").read_text(encoding="utf-8")).get("pre", "qwen35") \
            if (tpath / "tokenizer.json").exists() else "qwen35"
        tok = ST.Tokenizer(tokens, merges, types, pre=pre)
    has_grammar = False
    if not a.mock_engine:
        vision = None
        env = child_env(cfg)
        sampling_defaults = {**sampling_defaults_from_config(cfg), **a.sampling_launch}
        if a.lazy and not multi:
            drop_sidecar_vision(cfg)
        shown = [x for x in cfg["items"] if x not in cfg.get("shared_items", ())]
        shown += sampling_items(sampling_defaults) + [x for x in [vision_item(cfg.get("vision"), cfg)] if x]
        print(model_line(cfg["model_name"], cfg.get("name_width", 0), shown), flush=True)
        for line in cfg["start_lines"]:
            print(line, flush=True)
        if cfg.get("vision"):
            mode = cfg["vision"].get("mode")
            if mode == "dynamic":
                vision = EngineVision(cfg["vision"])
            else:
                vision = Vision(cfg["vision"], log=open(cfg["log"], "a", encoding="utf-8") if cfg.get("log") else None,
                                env=vision_env(env, mode), start=not multi)
        if a.lazy and not multi:
            print(tagged("model", "unloaded, the first request loads it"), flush=True)
        if cfg.get("log"):                              # follow the engine's log file
            threading.Thread(target=tail_file, args=(logs, cfg["log"], "eng"), daemon=True).start()
        else:
            ProcessEngine.stderr_sink = lambda line: logs.add("eng", line)
        has_grammar = known is not None and "--grammar" in known
        if a.grammar and not has_grammar:
            print(tagged("warning", "grammar is on but this engine has no --grammar; requests go without one"), flush=True)
        gwords = grammar_words(known)
        cdir = flag_value(engine_args(cfg), "--conversation-cache-dir")
        cmib = flag_value(engine_args(cfg), "--conversation-cache-disk-mib") or "0"
        if cdir and cmib != "0" and a.debug:
            print(tagged("disk", f"conversations={cdir} disk-mib={cmib}"), flush=True)
        # always built unloaded: the load itself is the lazy path's, started by the caller instead of by a request
        engine = ProcessEngine(cfg["engine_bin"], engine_args(cfg) + gwords, log=cfg.get("log"), env=env, lazy=True)
        if isinstance(vision, EngineVision):
            vision.attach(engine)
    else:
        engine, vision, sampling_defaults = MockEngine(tok, a.script or [
            "Thinking about it.</think>\n\nHello from the mock engine."], batch=a.parallel or 0), None, dict(a.sampling_launch)
    # the model's exported template, else the bundled one
    dialect = dialect_for(cfg.get("arch"))
    tpl = tpath / "chat_template.jinja"
    svc = Service(engine, tok, dialect.template(tpl if tpl.exists() else ROOT / "serve/chat_template.jinja"),
                  model_name=cfg.get("model_name", "basalt"), vision=vision,
                  sampling_defaults=sampling_defaults, fit_max_tokens=a.fit_max_tokens, grammar=a.grammar,
                  grammar_engine=bool(a.mock_engine or has_grammar), dialect=dialect)
    svc.logs = logs
    svc.idle_unload_s = a.idle_unload or 0
    svc.idle_sleep_s = a.idle_sleep or 0
    svc.min_free_vram_mib = a.min_free_vram_mib or 0
    svc.reasoning_budget_tokens = a.reasoning_budget_tokens or 0
    svc.answer_room = a.answer_room
    svc.thinking_loop_tokens = a.thinking_loop_tokens
    svc.gpu_indices = gpus_in_play(cfg, engine_args(cfg) if not a.mock_engine else [])   # the cards /metrics reads
    svc.gpu_index = svc.gpu_indices[0]
    if multi:
        svc.start_tries = SWAP_START_TRIES
    return svc


def check_models(services: list[Service]) -> None:
    """Refuse model names that answer for two entries, and one conversation folder shared by two: each model's chats keep
    their own budget."""
    seen: dict[str, Service] = {}
    folders: dict[str, str] = {}
    for svc in services:
        for name in svc.served_names():
            if seen.setdefault(name, svc) is not svc:
                fail(f"--config: {name!r} names two models: give each its own alias")
        args = getattr(svc.engine, "spawn", (None, []))[1]
        where = flag_value(args, "--conversation-cache-dir")
        if where and (flag_value(args, "--conversation-cache-disk-mib") or "0") != "0" and \
                folders.setdefault(where, svc.model) != svc.model:
            fail(f"--config: {svc.model} and {folders[where]} share the conversation folder {where}: give each its own")


def main() -> int:
    ap = build_parser()
    engine_bin = ap.parse_known_args(sys.argv[1:])[0].engine_bin
    try:
        settings = model_settings(ap, sys.argv[1:], (lambda path, group: flag_refused(engine_bin, path, group))
                                  if engine_bin else None)
    except ValueError as e:
        fail(f"--config: {e}")
    a = settings[0][0]                                  # the options every model shares read the same in each
    multi = len(settings) > 1
    service.DEBUG, engine_process.REQUEST_LINES = a.debug, a.request_lines
    logs = LogBuffer()                                  # this process's and the engine's output
    sys.stdout, sys.stderr = (TeeStream(ColorStream.wrap(sys.stdout), logs, "srv"),
                              TeeStream(ColorStream.wrap(sys.stderr), logs, "srv"))
    print(tagged("legend", LEGEND), flush=True)          # first line, so the key precedes every value
    known = engine_flags_of(a.engine_bin) if a.engine_bin else None
    cfgs = [check_model(m, file_engine, cli_engine, used_argv, known)
            for m, used_argv, file_engine, cli_engine in settings]
    if (a.image_url_host or a.image_dir) and not a.mock_engine and all(m[0].vision == "off" for m in settings):
        fail("--image-url-host and --image-dir need --vision (image input is off)")
    try:
        image_sources = image_sources_of(a.image_url_host, a.image_dir, a.image_remote_only)
    except ValueError as e:
        fail(str(e))
    try:                                                # fail fast if the port is taken
        Server((a.host, a.port), BaseHTTPRequestHandler).server_close()
    except OSError:
        fail(f"port {a.port} is in use (--port N)")
    width = max(len(c.get("model_name") or "") for c in cfgs) + 1
    shared = [x for x in cfgs[0].get("items", []) if all(x in c.get("items", []) for c in cfgs[1:])] if multi else []
    line, shared_sleep = shared_line(shared, [m[0].idle_sleep for m in settings])
    if line:
        print(tagged("engine", line), flush=True)
    for c in cfgs:
        c["shared_items"], c["name_width"] = shared, width
    services = [build_model(m[0], cfg, logs, multi, known) for m, cfg in zip(settings, cfgs)]
    check_models(services)
    router = Router(services)
    svc = router.front
    svc.image_sources = image_sources
    if image_sources.url_hosts or image_sources.dirs or image_sources.remote_only:
        print(tagged("images", f"from other machines{' and this one' if image_sources.remote_only else ''}: "
                               f"url-hosts={','.join(a.image_url_host) or '-'} "
                               f"dirs={','.join(image_sources.dirs) or '-'}"), flush=True)
    used_argv = settings[0][1]
    if any(x == "--api-key" and i + 1 < len(used_argv) and not used_argv[i + 1].strip() or x.strip() == "--api-key="
           for i, x in enumerate(used_argv)):
        # an empty key would silently disable authentication
        print("[server] error: --api-key is empty (give a key, or leave it out)", file=sys.stderr)
        return 2
    if a.max_payload_mib < 1:
        fail(f"--max-payload-mib {a.max_payload_mib} must be at least 1")
    svc.max_payload = a.max_payload_mib << 20
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
    httpd = serve(router, host=a.host, port=a.port)
    here = "127.0.0.1" if a.host in ("0.0.0.0", "", "::") else a.host

    def announcer(s: Service, m):
        first_load = [True]

        def announce():
            """After every weights load: the slot table once, then the weights line (no URL)."""
            engine = s.engine
            if first_load[0]:
                first_load[0] = False
                if not a.mock_engine:
                    warn_tight_ram(engine.info.get("arena_mib"))
                if getattr(engine, "batch", 0):
                    print(tagged("slots", f"par={engine.batch} ctx/slot={engine.max_context}"
                                 + (" (slot-full-context)" if m.slot_full_context else "") + ", others queue"), flush=True)
            print(tagged("weights", f"loaded{f' {s.model}' if multi else ''}, ctx={engine.max_context} "
                                    f"images={'on' if s.vision else 'off'}"), flush=True)
        return announce

    for s, m in zip(services, (x[0] for x in settings)):
        s.start_idle_unload()
        s.start_idle_sleep(announce=not shared_sleep, name=s.model if multi else None)
        s.on_loaded = announcer(s, m)

    # the one place the addresses and the APIs are printed: the server listens before the model loads
    print(tagged("ready", f"http://{here}:{a.port}/v1  openai+anthropic  stats http://{here}:{a.port}/"
                 + (" api-key=on" if svc.api_key else "")), flush=True)
    if a.mock_engine:
        for s in services:
            s.on_loaded()
    if a.host not in LOOPBACK_NAMES:
        # reachable from other devices: print the addresses
        ips = lan_addresses()
        for ip in ips:
            print(tagged("network", f"http://{ip}:{a.port}/v1"), flush=True)
        if not ips:
            print(tagged("network", f"port={a.port}"), flush=True)
        if exposed_without_key(a.host, svc.api_key):
            print(tagged("warning", "no api key, open to the network (--api-key KEY)"), flush=True)
    if not a.mock_engine and not a.lazy:
        if multi:
            router.preload(svc)                         # the first entry; a failed load leaves the others servable
        else:
            svc.start_load()                            # the page and the API are up; requests wait for the load
    if a.open:
        open_when_ready(svc, f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else a.host}:{a.port}/",
                        now=bool(a.mock_engine or a.lazy))
    # SIGTERM takes Ctrl+C's path (QUIT to the engine) so a container stop does not orphan it
    def on_sigterm(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)   # one stop: a later TERM must not cut the engine's QUIT short
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
        closers = [router.stop, httpd.shutdown]
        for s in services:
            closers += [getattr(s.engine, "close", None), s.vision.close if s.vision else None]
        for close in filter(None, closers):
            try:
                close()
            except KeyboardInterrupt:                   # second Ctrl+C: do not wait for the engine
                for s in services:
                    if getattr(s.engine, "proc", None):
                        s.engine.proc.kill()
        print("[server] stopped", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
