"""Server settings: the command-line parser, the --config file and engine flag checks."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from serve import ROOT
from serve.engine.vision import VISION_MODES, VISION_USAGE, vision_mode


MIN_SLOT_CONTEXT = 256          # the least context a slot is given when --max-context is shared
ENGINE_DEFAULT_CONTEXT = 4096     # the engine's context without --max-context
# server options that are engine flags (the "engine" section of the config file)
ENGINE_PASSTHROUGH = ("spec", "spec-min-p", "mtp-max-t", "peer-exclusive", "max-context")
PARALLEL_MAX = 8                            # the engine's batch slots
CONFIG_PATHS = ("model", "tokenizer")       # relative to the config file
ENGINE_WORDS = {"--control-vector-layer-range": 2}   # engine flags that take several values: --flag A B


def engine_flags_of(engine_bin: str) -> set[str] | None:
    """The long flags the engine binary accepts; None when it cannot be read."""
    try:
        with open(engine_bin, "rb") as f:
            data = f.read()
    except OSError:
        return None
    return {m.decode() for m in re.findall(rb"(?<![\x21-\x7e])--[a-z0-9][a-z0-9-]*(?![\x21-\x7e])", data)}


def server_options(ap: argparse.ArgumentParser) -> dict:
    """{name: action} of the server's own options (names without the dashes), apart from --config and --help."""
    return {o[2:]: act for act in ap._actions for o in act.option_strings
            if o.startswith("--") and o[2:] not in ("help", "config")}


def split_engine_extras(extras: list[str]) -> dict:
    """Engine flags among the unknown command-line words: `--flag VALUE`, `--flag=VALUE`, `--flag A B` for the
    flags in ENGINE_WORDS (a list), or a bare switch; ValueError for a stray word."""
    out, i = {}, 0
    while i < len(extras):
        word = extras[i]
        if not word.startswith("--") or len(word) < 3:
            raise ValueError(f"unexpected {word!r}")
        flag, eq, value = word.partition("=")
        if flag in ENGINE_WORDS:
            n = ENGINE_WORDS[flag]
            words = extras[i + 1:i + 1 + n]
            if eq or len(words) < n or any(w.startswith("--") for w in words):
                raise ValueError(f"{flag} takes {n} values: {flag} " + " ".join(f"V{k + 1}" for k in range(n)))
            out[flag] = words
            i += n
        elif eq:
            out[flag] = value
        elif i + 1 < len(extras) and not extras[i + 1].startswith("--"):
            out[flag] = extras[i + 1]
            i += 1
        else:
            out[flag] = True
        i += 1
    return out


def load_config_file(path: str, ap: argparse.ArgumentParser) -> tuple[list[str], dict]:
    """(argv, engine) from a --config JSON file {"server": {...}, "engine": {...}}. `server` options become
    command-line words so every command-line check applies; `engine` maps flags to values (true: switch, false:
    off). Paths are relative to the file; ValueError for anything else."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except OSError as e:
        raise ValueError(f"cannot read: {e.strerror or e}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON: {e}") from None
    if not isinstance(raw, dict) or set(raw) - {"server", "engine"}:
        raise ValueError('expected {"server": {...}, "engine": {...}}'
                         + (f", not the key {sorted(set(raw) - {'server', 'engine'})[0]!r}" if isinstance(raw, dict) else ""))
    sections = {k: raw.get(k, {}) for k in ("server", "engine")}
    for k, v in sections.items():
        if not isinstance(v, dict):
            raise ValueError(f'"{k}" must be an object')
    here = Path(path).resolve().parent
    known = server_options(ap)
    argv: list[str] = []
    for key in (*sections["server"], *sections["engine"]):
        if re.sub(r"[^a-z]", "", str(key).lower()).startswith("apikey"):
            raise ValueError("api-key can't be set in a config file; pass --api-key on the command line")
    for key, value in sections["server"].items():
        name = str(key).strip().lower().replace("_", "-").removeprefix("--")
        act = known.get(name)
        if name == "engine-bin":
            raise ValueError("server.engine-bin: engine-bin is set by basalt serve")
        if name in ENGINE_PASSTHROUGH:
            raise ValueError(f'server.{key}: an engine flag, set it under "engine"')
        if act is None:
            raise ValueError(f"server.{key}: not a server option")
        flag = "--" + name
        if name == "vision":                               # a mode; booleans are refused
            try:
                mode = vision_mode(value)
            except ValueError as e:
                raise ValueError(f"server.{key}: {e}") from None
            argv += ["--vision", mode]
            continue
        if act.nargs == 0:                                 # a switch
            if not isinstance(value, bool):
                raise ValueError(f"server.{key}: true or false")
            argv += [flag] if value else []
            continue
        items = value if isinstance(value, list) else [value]
        if isinstance(value, list) and not isinstance(act, argparse._AppendAction):
            raise ValueError(f"server.{key}: one value, not a list")
        for item in items:
            if isinstance(item, bool) or not isinstance(item, (str, int, float)):
                raise ValueError(f"server.{key}: expected a string or number, not {item!r}")
            item = str(item)
            if name in CONFIG_PATHS and item:
                item = str((here / os.path.expanduser(item)))
            argv += [flag, item]
    engine = {}
    for key, value in sections["engine"].items():
        flag = "--" + str(key).strip().lower().replace("_", "-").removeprefix("--")
        if flag[2:] in known and flag[2:] not in ENGINE_PASSTHROUGH:
            raise ValueError(f'engine.{key}: a server option, set it under "server"')
        if flag in ENGINE_WORDS:
            n = ENGINE_WORDS[flag]
            if not (isinstance(value, list) and len(value) == n
                    and all(isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in value)):
                raise ValueError(f"engine.{key}: a list of {n} values, not {value!r}")
            engine[flag] = [str(v) for v in value]
        elif isinstance(value, bool) or (isinstance(value, (str, int, float)) and str(value).strip() != ""):
            engine[flag] = value
        else:
            raise ValueError(f"engine.{key}: expected a string or number, or true/false for a switch, not {value!r}")
    return argv, engine


def parse_settings(ap: argparse.ArgumentParser, cli: list[str]):
    """(args, argv, file_engine, cli_engine): parsed options, their source words and the engine flags from each.
    Command line > config file > defaults; SystemExit(2) for a bad value, ValueError for a bad config or stray word."""
    a_cli, extras = ap.parse_known_args(cli)
    cli_engine = split_engine_extras(extras)
    file_argv, file_engine = [], {}
    if a_cli.config:
        file_argv, file_engine = load_config_file(a_cli.config, ap)
    a = ap.parse_known_args(file_argv + cli)[0]
    if a.vision == "":                                     # a bare --vision
        ap.error(VISION_USAGE)
    for act in ap._actions:                                # a repeatable command-line option replaces the file's
        if isinstance(act, argparse._AppendAction) and getattr(a_cli, act.dest) != act.default:
            setattr(a, act.dest, getattr(a_cli, act.dest))
    for flag, value in (("--spec", a.spec), ("--spec-min-p", a.spec_min_p), ("--mtp-max-t", a.mtp_max_t),
                        ("--max-context", a.max_context), ("--peer-exclusive", a.peer_exclusive or None)):
        if value is not None:
            cli_engine[flag] = value
    if a.parallel and a.parallel > 1:
        for flag in ("--batch", "--slots"):
            if flag in cli_engine or flag in file_engine:
                raise ValueError(f"parallel and engine {flag} both set: use parallel")
        (cli_engine if a_cli.parallel else file_engine)["--batch"] = a.parallel
        if not a.slot_full_context:
            # --max-context is shared by the slots, as in llama.cpp: each holds an equal part (the solo path, used while
            # a request is alone, has the same length, so a request never outgrows its slot)
            src = cli_engine if "--max-context" in cli_engine or "--max-context" not in file_engine else file_engine
            ctx = int(src.get("--max-context", ENGINE_DEFAULT_CONTEXT))
            src["--max-context"] = max(MIN_SLOT_CONTEXT, ctx // a.parallel)
            a.shared_context = ctx                         # the total, for the startup line
    return a, file_argv + cli, file_engine, cli_engine


def check_engine_flags(flags: dict, known: set[str] | None, where: str) -> None:
    """ValueError naming the first flag the engine rejects (unchecked when the binary is unreadable)."""
    for flag in flags:
        if known is not None and flag not in known:
            raise ValueError(f"{where}: {flag} is not an engine flag")


def engine_settings(file_flags: dict, cli_flags: dict) -> tuple[dict, dict]:
    """(flags, sources): the config file's flags overridden by the command line's, with each one's origin."""
    flags, sources = {}, {}
    for src, items in (("config", file_flags), ("cli", cli_flags)):
        for flag, value in items.items():
            flags[flag], sources[flag] = value, src
    return flags, sources


def engine_line(flags: dict, sources: dict, config: bool = True, shared: tuple[int, int] | None = None) -> str:
    """The startup line listing engine settings from --config or the command line; without a config file it says
    the built-in defaults apply. `shared` is (total, slots) when --max-context was divided among the slots."""
    def one(f, v):
        if v is False:
            return f"no {f} ({sources[f]})"
        note = f"{sources[f]}: {shared[0]} / {shared[1]} slots" if shared and f == "--max-context" else sources[f]
        shown_v = " ".join(v) if isinstance(v, list) else v
        return f"{f}{'' if v is True else f' {shown_v}'} ({note})"
    shown = ", ".join(one(f, v) for f, v in flags.items())
    if config:
        return f"[server] engine: {shown}" if shown else "[server] engine: config sets nothing, built-in defaults"
    spec = flags.get("--spec", 0)
    state = f"spec {'off' if not int(spec) else spec}, {flags.get('--max-context', ENGINE_DEFAULT_CONTEXT)} context"
    return f"[server] engine: built-in defaults (no --config): {state}" + (f"; {shown}" if shown else "")


def sampling_defaults_from_config(cfg: dict) -> dict:
    """The model file's `sampling` block: defaults for fields a request omits (temperature, top_p, top_k, min_p,
    presence/repetition/frequency penalties, penalty_last_n, seed). Request fields win; a bad value refuses to
    start; unknown keys are named and ignored."""
    out = {}
    for key, value in (cfg.get("sampling") or {}).items():
        if value is None:
            continue
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if key == "temperature":
            if not number or value < 0:
                raise SystemExit(f"[server] error: sampling.temperature={value!r}: expected a number >= 0 (0 = greedy)")
            out[key] = float(value)
        elif key == "top_p":
            if not number or not 0 < value <= 1:
                raise SystemExit(f"[server] error: sampling.top_p={value!r}: expected 0 < top_p <= 1")
            out[key] = float(value)
        elif key == "min_p":
            if not number or not 0 <= value <= 1:
                raise SystemExit(f"[server] error: sampling.min_p={value!r}: expected 0 <= min_p <= 1")
            out[key] = float(value)
        elif key == "top_k":
            if not number or value != int(value) or not 1 <= value <= 64:
                raise SystemExit(f"[server] error: sampling.top_k={value!r}: the sampled path takes an integer 1..64")
            out[key] = int(value)
        elif key == "presence_penalty":
            if not number or value < 0:
                raise SystemExit(f"[server] error: sampling.presence_penalty={value!r}: expected a number >= 0")
            out[key] = float(value)
        elif key == "frequency_penalty":
            if not number or value < 0:
                raise SystemExit(f"[server] error: sampling.frequency_penalty={value!r}: expected a number >= 0")
            out[key] = float(value)
        elif key == "repetition_penalty":
            if not number or value <= 0:
                raise SystemExit(f"[server] error: sampling.repetition_penalty={value!r}: expected a number > 0 (1 = off)")
            out[key] = float(value)
        elif key == "penalty_last_n":
            if not number or value != int(value) or value < 0:
                raise SystemExit(f"[server] error: sampling.penalty_last_n={value!r}: expected a non-negative integer")
            out[key] = int(value)
        elif key == "seed":
            if not number or value != int(value) or value <= 0:
                raise SystemExit(f"[server] error: sampling.seed={value!r}: expected a positive integer")
            out[key] = int(value)
        elif key == "experimental_speed_projection":
            if not isinstance(value, bool):
                raise SystemExit(f"[server] error: sampling.experimental_speed_projection={value!r}: expected true or "
                                 "false (the default for requests that leave it out, when the engine has the vector)")
            out[key] = value
        else:
            print(f"[server] warning: sampling.{key} is not a known key, ignored", flush=True)
    return out


def vision_setting(text: str) -> str:
    """argparse type: one of the --vision modes."""
    try:
        return vision_mode(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def parallel_count(text: str) -> int:
    """argparse type: a whole number of request slots, 1..PARALLEL_MAX."""
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number 1..{PARALLEL_MAX}, not {text!r}") from None
    if not 1 <= n <= PARALLEL_MAX:
        raise argparse.ArgumentTypeError(f"expected 1..{PARALLEL_MAX}, not {n}")
    return n


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--mock-engine", action="store_true",
                    help="run a scripted engine instead of the model, for clients and tests (no --model needed)")
    ap.add_argument("--model", help="the .basalt model file: the engine, tokenizer, sampling and vision come from it")
    ap.add_argument("--engine-bin", help=argparse.SUPPRESS)      # set by `basalt serve`
    ap.add_argument("--alias", help="the model name the API reports (default: the name stored in the model file)")
    ap.add_argument("--spec", type=int, help="tokens per verify window, the drafter's cap (default: the model file's; "
                                             "0 = no speculation, the drafter is not loaded)")
    ap.add_argument("--spec-min-p", type=float, help="stop drafting below this draft probability (default: the "
                                                     "model file's; 0 = always draft to the cap)")
    ap.add_argument("--mtp-max-t", type=int, help="the drafter's own window cap (default 0 = --spec)")
    ap.add_argument("--max-context", type=int, help="the longest context the engine allows, in tokens (default: the "
                                                    "engine's)")
    ap.add_argument("--parallel", type=parallel_count, metavar="N",
                    help=f"requests served at once, 1..{PARALLEL_MAX} (default 1: one at a time); more requests wait "
                         "their turn. --max-context is shared: each slot holds that divided by N (so 4 slots of 262144 "
                         "need --max-context 1048576, or --slot-full-context). The slots' KV cache comes out of the "
                         "expert cache")
    ap.add_argument("--slot-full-context", action="store_true",
                    help="with --parallel, give every slot the whole --max-context instead of an equal part: N slots "
                         "cost N times the KV cache")
    ap.add_argument("--config", metavar="FILE",
                    help="a JSON object of engine flags, e.g. {\"max-context\": 32768, \"spec\": 7, "
                         "\"resident-experts\": true}; the server's own flags win over it")
    ap.add_argument("--vision", nargs="?", const="", default="off", type=vision_setting, choices=VISION_MODES, metavar="MODE",
                    help="image input: off (default), on (the encoder always on the main GPU, ~2.2 GiB of VRAM), "
                         "peer (always on the second GPU), dynamic (inside the engine: it borrows expert-cache slots "
                         "on the main GPU while it reads an image, no standing VRAM) or cpu (no VRAM, slow); a value is "
                         "required")
    ap.add_argument("--peer-exclusive", action="store_true",
                    help="the peer card's experts live only there, not also in RAM: less RAM, slower prefill")
    ap.add_argument("--host", default="127.0.0.1",
                    help="the address to listen on: 127.0.0.1 = this PC only (the default), 0.0.0.0 = also other "
                         "devices on your network (give --api-key)")
    ap.add_argument("--script", action="append",
                    help="the mock engine's answer (default: a short greeting); given more than once, requests get "
                         "them in turn and the last one repeats")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--gpu", help="the GPUs to use, as nvidia-smi numbers them: the first runs the model, a second is "
                                  "the peer expert tier (\"0,1\"; default: every visible GPU)")
    ap.add_argument("--tokenizer", default=str(ROOT / "pack/full/tokenizer"),
                    help="tokenizer directory for the mock engine (a byte tokenizer if absent)")
    ap.add_argument("--open", action="store_true", help="open the local page in the browser once the model is ready")
    ap.add_argument("--fit-max-tokens", action="store_true",
                    help="clamp max_tokens to the remaining context instead of rejecting the request with 400")
    ap.add_argument("--api-key", default="",
                    help="require this key on /v1/* (Authorization: Bearer ... or x-api-key); command line only, "
                         "not allowed in a --config file")
    ap.add_argument("--allowed-host", action="append", default=[], metavar="NAME",
                    help="also answer to this Host name (repeatable; \"*\" turns the check off)")
    ap.add_argument("--cors-origin", action="append", default=[], metavar="ORIGIN",
                    help="let web pages from this origin use /v1/* (repeatable)")
    ap.add_argument("--debug", action="store_true",
                    help="log one line per request and the raw model text after each reply")
    ap.add_argument("--request-lines", action="store_true",
                    help="echo the engine's per-request summary on stdout, for a supervisor that only sees this output")
    ap.add_argument("--lazy", action="store_true", help="start the text-only API unloaded; load on first request")
    ap.add_argument("--idle-unload", type=float, default=None, metavar="SECONDS",
                    help="unload the model after this many seconds without requests; the next request loads it again")
    ap.add_argument("--min-free-vram-mib", type=int, default=None,
                    help="load an unloaded model only when this much VRAM is free, else answer 503")
    return ap
