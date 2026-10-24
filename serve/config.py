"""Server settings: the command-line parser, the --config file and engine flag checks."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

from serve import ROOT
from serve.engine.flags import flag_index
from serve.engine.vision import VISION_MODES, VISION_USAGE, vision_mode


MIN_SLOT_CONTEXT = 256          # the least context a slot is given when --max-context is shared
ENGINE_DEFAULT_CONTEXT = 4096     # the engine's context without --max-context
PARALLEL_MAX = 8                            # the engine's batch slots
CONFIG_PATHS = ("model", "tokenizer")       # relative to the config file
GRAMMAR_DEFAULT = True                      # the `grammar` option when neither the file nor the command line sets it


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


def grammar_words(known: set[str] | None) -> list[str]:
    """The engine words that make grammar enforcement available: `--grammar on` whenever the engine binary has the
    flag (a request's grammar key enforces; the server option only decides which requests carry one), so an engine
    without it still starts."""
    return ["--grammar", "on"] if known is not None and "--grammar" in known else []


def split_engine_extras(extras: list[str]) -> list[str]:
    """The unknown command-line words, which go to the engine as typed: it parses its own flags and their values.
    A `--flag=VALUE` word is split in two, as the engine has no `=` form. ValueError for a leading word that is not
    a flag."""
    if extras and not (extras[0].startswith("--") and len(extras[0]) > 2):
        raise ValueError(f"unexpected {extras[0]!r}")
    out: list[str] = []
    for word in extras:
        flag, eq, value = word.partition("=")
        if eq and word.startswith("--") and len(flag) > 2:
            out += [flag, value]
        else:
            out.append(word)
    return out


def engine_words(key: str, value) -> list[str]:
    """The engine words for one config entry: a switch is true (the flag) or false (nothing), a string or number is
    one value, and a list gives one word per item (a flag that takes several values). ValueError otherwise."""
    flag = "--" + str(key).strip().lower().replace("_", "-").removeprefix("--")
    if isinstance(value, bool):
        return [flag] if value else []
    items = value if isinstance(value, list) else [value]
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (str, int, float)) or str(item).strip() == "":
            raise ValueError(f"engine.{key}: expected a string or number, a list of them for several values, "
                             f"or true/false for a switch, not {value!r}")
    if not items:
        raise ValueError(f"engine.{key}: an empty list gives the flag no value")
    return [flag, *(str(item) for item in items)]


def load_config_file(path: str, ap: argparse.ArgumentParser) -> tuple[list[str], list[str]]:
    """(argv, engine) from a --config JSON file {"server": {...}, "engine": {...}}. `server` options become
    command-line words so every command-line check applies; `engine` maps flags to values and becomes the words
    for the engine (true: switch, false: off, a list: several values). Paths are relative to the file; ValueError
    for anything else."""
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
        if act is None:
            raise ValueError(f'server.{key}: not a server option (engine flags go under "engine")')
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
            argv += [flag] if value else (["--no-grammar"] if name == "grammar" else [])
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
    engine: list[str] = []
    for key, value in sections["engine"].items():
        flag = "--" + str(key).strip().lower().replace("_", "-").removeprefix("--")
        if flag[2:] in known:
            raise ValueError(f'engine.{key}: a server option, set it under "server"')
        engine += engine_words(key, value)
    return argv, engine


def parse_settings(ap: argparse.ArgumentParser, cli: list[str]):
    """(args, argv, file_words, cli_words): parsed options, their source words and the engine words from each.
    The engine reads its flags in order and a later one wins, so file words then command-line words keep
    command line > config file > defaults; SystemExit(2) for a bad value, ValueError for a bad config or stray word."""
    a_cli, extras = ap.parse_known_args(cli)
    cli_words = split_engine_extras(extras)
    file_argv, file_words = [], []
    if a_cli.config:
        file_argv, file_words = load_config_file(a_cli.config, ap)
    a = ap.parse_known_args(file_argv + cli)[0]
    a.sampling_launch = {k: getattr(a, k) for k in SAMPLING_KEYS if getattr(a, k) is not None}
    a.sampling_cli = {k for k in a.sampling_launch if getattr(a_cli, k) is not None}
    if a.vision == "":                                     # a bare --vision
        ap.error(VISION_USAGE)
    for act in ap._actions:                                # a repeatable command-line option replaces the file's
        if isinstance(act, argparse._AppendAction) and getattr(a_cli, act.dest) != act.default:
            setattr(a, act.dest, getattr(a_cli, act.dest))
    if a.parallel and a.parallel > 1:
        for flag in ("--batch", "--slots"):
            if flag in file_words or flag in cli_words:
                raise ValueError(f"parallel and engine {flag} both set: use parallel")
        (cli_words if a_cli.parallel else file_words).extend(["--batch", str(a.parallel)])
        if not a.slot_full_context:
            # --max-context is shared by the slots, as in llama.cpp: each holds an equal part (the solo path, used while
            # a request is alone, has the same length, so a request never outgrows its slot)
            src = cli_words if "--max-context" in cli_words or "--max-context" not in file_words else file_words
            at = flag_index(src, "--max-context")
            if at is not None and at + 1 >= len(src):
                raise ValueError("--max-context needs a value")
            ctx = int(src[at + 1]) if at is not None else ENGINE_DEFAULT_CONTEXT
            part = str(max(MIN_SLOT_CONTEXT, ctx // a.parallel))
            if at is None:
                src += ["--max-context", part]
            else:
                src[at + 1] = part
            a.shared_context = ctx                         # the total, for the startup line
    return a, file_argv + cli, file_words, cli_words


def check_engine_flags(words: list[str], known: set[str] | None, where: str) -> None:
    """ValueError naming the first flag (a word starting with `--`) the engine rejects; value words are not names
    (unchecked when the binary is unreadable)."""
    for word in words:
        if word == "--gpu":
            raise ValueError(f"{where}: gpu is a server setting (--gpu, or \"gpu\" in the server section)")
        if known is not None and word.startswith("--") and word not in known:
            raise ValueError(f"{where}: {word} is not an engine flag")


def engine_groups(words: list[str]) -> list[list[str]]:
    """`words` as [flag, value, ...] groups, each starting at a word that begins with `--`."""
    groups: list[list[str]] = []
    for word in words:
        if word.startswith("--") or not groups:
            groups.append([word])
        else:
            groups[-1].append(word)
    return groups


def engine_line(file_words: list[str], cli_words: list[str],
                shared: tuple[int, int] | None = None, grammar: tuple[bool, str] | None = None) -> str:
    """The startup line listing the engine words from --config and the command line, each flag with its values and
    origin; with none it says the default flags are used. `shared` is (total, slots) when --max-context
    was divided among the slots. `grammar` is (on, origin) with origin "config" or "cli", None when left at its default (no entry)."""
    groups = [(g, "config") for g in engine_groups(file_words)] + [(g, "cli") for g in engine_groups(cli_words)]
    last_ctx = max((i for i, (g, _) in enumerate(groups) if g[0] == "--max-context"), default=None)
    def one(i, g, src):
        note = f"{src}: {shared[0]} / {shared[1]} slots" if shared and i == last_ctx else src
        return f"{' '.join(g)} ({note})"
    shown = ", ".join(one(i, g, src) for i, (g, src) in enumerate(groups))
    if grammar is not None:
        on, origin = grammar
        entry = f"grammar {'on' if on else 'off'} ({origin})"
        shown = f"{shown}, {entry}" if shown else entry
        base = "using default flags, " + entry if not groups else shown
    else:
        base = shown or "using default flags"
    return f"[server] engine: {base}"


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _whole(v):
    return _num(v) and v == int(v)


# sampling key -> (converter, test, what is expected); the model file and the launch options share it
SAMPLING_KEYS = {
    "temperature": (float, lambda v: _num(v) and v >= 0, "a number >= 0 (0 = greedy)"),
    "top_p": (float, lambda v: _num(v) and 0 < v <= 1, "0 < top_p <= 1"),
    "min_p": (float, lambda v: _num(v) and 0 <= v <= 1, "0 <= min_p <= 1"),
    "top_k": (int, lambda v: _whole(v) and 1 <= v <= 64, "an integer 1..64"),
    "presence_penalty": (float, lambda v: _num(v) and v >= 0, "a number >= 0"),
    "frequency_penalty": (float, lambda v: _num(v) and v >= 0, "a number >= 0"),
    "repetition_penalty": (float, lambda v: _num(v) and v > 0, "a number > 0 (1 = off)"),
    "penalty_last_n": (int, lambda v: _whole(v) and v >= 0, "a non-negative integer"),
    "seed": (int, lambda v: _whole(v) and v > 0, "a positive integer"),
}


def sampling_defaults_from_config(cfg: dict) -> dict:
    """The model file's `sampling` block: defaults for fields a request omits (temperature, top_p, top_k, min_p,
    presence/repetition/frequency penalties, penalty_last_n, seed). Request fields win; a bad value refuses to
    start; unknown keys are named and ignored."""
    out = {}
    for key, value in (cfg.get("sampling") or {}).items():
        if value is None:
            continue
        if key in SAMPLING_KEYS:
            convert, ok, expected = SAMPLING_KEYS[key]
            if not ok(value):
                raise SystemExit(f"[server] error: sampling.{key}={value!r}: expected {expected}")
            out[key] = convert(value)
        elif key == "experimental_speed_projection":
            if not isinstance(value, bool):
                raise SystemExit(f"[server] error: sampling.{key}={value!r}: expected true or "
                                 "false (the default for requests that leave it out, when the engine has the vector)")
            out[key] = value
        else:
            print(f"[server] warning: sampling.{key} is not a known key, ignored", flush=True)
    return out


def sampling_option(key: str):
    """argparse type for the launch option of one sampling key: the model file's check on the typed value."""
    convert, ok, expected = SAMPLING_KEYS[key]

    def parse(text: str):
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected {expected}, not {text!r}") from None
        if not ok(value):
            raise argparse.ArgumentTypeError(f"expected {expected}, not {text!r}")
        return convert(value)
    return parse


def sampling_line(effective: dict, cli: set[str], config: set[str]) -> str:
    """The startup line for the effective sampling defaults, each marked with where it came from: the keys in
    `cli` were typed on the command line, those in `config` come from the config file; the rest are the model
    file's."""
    mark = lambda k: " (command line)" if k in cli else " (config)" if k in config else ""
    shown = ", ".join(f"{k}={v}{mark(k)}" for k, v in effective.items())
    return f"[server] sampling defaults: {shown}"


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


def budget_count(text: str) -> int:
    """argparse type: a thinking budget in tokens, a whole number >= 0 (0: none)."""
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number >= 0, not {text!r}") from None
    if n < 0:
        raise argparse.ArgumentTypeError(f"expected a whole number >= 0, not {n}")
    return n


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--mock-engine", action="store_true",
                    help="run a scripted engine instead of the model, for clients and tests (no --model needed)")
    ap.add_argument("--model", help="the .basalt model file: the engine, tokenizer, sampling and vision come from it")
    ap.add_argument("--engine-bin", help=argparse.SUPPRESS)      # set by `basalt serve`
    ap.add_argument("--alias", help="the model name the API reports (default: the name stored in the model file)")
    ap.add_argument("--parallel", type=parallel_count, metavar="N",
                    help=f"requests served at once, 1..{PARALLEL_MAX} (default 1: one at a time); more requests wait "
                         "their turn. --max-context is shared: each slot holds that divided by N (so 4 slots of 262144 "
                         "need --max-context 1048576, or --slot-full-context). The slots' KV cache comes out of the "
                         "expert cache")
    ap.add_argument("--slot-full-context", action="store_true",
                    help="with --parallel, give every slot the whole --max-context instead of an equal part: N slots "
                         "cost N times the KV cache")
    ap.add_argument("--config", metavar="FILE",
                    help="a JSON file {\"server\": {...}, \"engine\": {...}}: server options and engine flags named "
                         "without the dashes, e.g. \"engine\": {\"max-context\": 32768, \"spec\": 7, "
                         "\"resident-experts\": true}; a list gives a flag several values; the command line "
                         "wins over it")
    ap.add_argument("--vision", nargs="?", const="", default="off", type=vision_setting, metavar="MODE|GPU",
                    help="image input: off (default), dynamic (inside the engine: it borrows expert-cache slots "
                         "on the main GPU while it reads an image, no standing VRAM), cpu (no VRAM, slow) or a "
                         "GPU number, as nvidia-smi numbers them: the encoder sidecar, always on that card")
    ap.add_argument("--host", default="127.0.0.1",
                    help="the address to listen on: 127.0.0.1 = this PC only (the default), 0.0.0.0 = also other "
                         "devices on your network (give --api-key)")
    ap.add_argument("--script", action="append",
                    help="the mock engine's answer (default: a short greeting); given more than once, requests get "
                         "them in turn and the last one repeats")
    ap.add_argument("--port", type=int, default=8080, help="the TCP port to listen on (default 8080)")
    ap.add_argument("--gpu", help="the GPUs to use, as nvidia-smi numbers them: exactly these, the first is the main "
                                  "card and the engine gives the rest their roles (\"0,1\"); auto = every Blackwell "
                                  "card, the fastest as main (default: GPU 0 alone)")
    ap.add_argument("--tokenizer", default=str(ROOT / "pack/full/tokenizer"),
                    help="tokenizer directory for the mock engine (a byte tokenizer if absent)")
    ap.add_argument("--open", action="store_true", help="open the local page in the browser once the model is ready")
    ap.add_argument("--fit-max-tokens", action="store_true",
                    help="clamp max_tokens to the remaining context instead of rejecting the request with 400")
    ap.add_argument("--grammar", action=argparse.BooleanOptionalAction, default=GRAMMAR_DEFAULT,
                    help="hold every tool call and JSON response format to a grammar in the engine (on by default; a request "
                         "with strict: true gets it either way); --no-grammar turns it off")
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
    for key, text in (
            ("temperature", "0 = greedy; higher is more random"),
            ("top-p", "nucleus cutoff, 0 < value <= 1 (1 = off)"),
            ("top-k", "keep the K likeliest tokens, 1..64"),
            ("min-p", "drop tokens less likely than this times the likeliest, 0..1 (0 = off)"),
            ("presence-penalty", "flat penalty on a token seen in the window, >= 0"),
            ("frequency-penalty", "penalty growing with how often a token was seen, >= 0"),
            ("repetition-penalty", "multiplicative penalty on seen tokens, > 0 (1 = off)"),
            ("penalty-last-n", "the window the penalties look back over, in tokens"),
            ("seed", "fix the random draw, a positive integer")):
        ap.add_argument(f"--{key}", type=sampling_option(key.replace("-", "_")), metavar="N", default=None,
                        help=f"default {key.replace('-', '_')} for requests that leave it out, over the model file's: {text}")
    ap.add_argument("--min-free-vram-mib", type=int, default=None,
                    help="load an unloaded model only when this much VRAM is free, else answer 503")
    ap.add_argument("--reasoning-budget-tokens", type=budget_count, default=None, metavar="N",
                    help="default thinking budget for requests that leave reasoning_budget_tokens out (0 = none): at "
                         "the budget the thinking is closed and the answer starts, so set it below the request's "
                         "max_tokens")
    return ap
