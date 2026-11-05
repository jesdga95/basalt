"""Server settings: the command-line parser, the --config file and engine flag checks."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from serve import ROOT
from serve.engine.flags import flag_index
from serve.engine.image_source import parse_url_host
from serve.engine.vision import VISION_MODES, VISION_USAGE, vision_mode
from serve.util.console import tagged


MIN_SLOT_CONTEXT = 256          # the least context a slot is given when --max-context is shared
CONFIG_VERSION = 2                          # the config file's "version"; a file without one is this format's first
PARALLEL_MAX = 8                            # the engine's batch slots
CONFIG_PATHS = ("model", "tokenizer", "image-dir")       # relative to the config file
# a model entry's own, and what it may also inherit from the top-level "engine"; other server options are the server's
ENTRY_OPTIONS = frozenset({"model", "alias"})
MODEL_OPTIONS = frozenset({
    "gpu", "vision", "parallel", "slot-full-context", "idle-sleep", "idle-unload", "min-free-vram-mib", "fit-max-tokens",
    "grammar", "reasoning-budget-tokens", "thinking-loop-tokens", "answer-room", "temperature", "top-p", "top-k", "min-p",
    "presence-penalty", "frequency-penalty", "repetition-penalty", "penalty-last-n", "seed"})
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


@dataclass
class ModelEntry:
    """One entry of "models": its server options as command-line words, the engine words of the top-level "engine" (taken
    where the model has the flag) and its own (a flag it lacks stops the start)."""
    path: str
    argv: list[str] = field(default_factory=list)
    shared_words: list[str] = field(default_factory=list)
    own_words: list[str] = field(default_factory=list)


@dataclass
class ConfigFile:
    server_argv: list[str]
    models: list[ModelEntry]


def option_words(where: str, key, value, act, here: Path) -> list[str]:
    """The command-line words of one server option from a config file, so every command-line check applies."""
    name = config_name(key)
    flag = "--" + name
    if name == "vision":                               # a mode; booleans are refused
        try:
            return ["--vision", vision_mode(value)]
        except ValueError as e:
            raise ValueError(f"{where}.{key}: {e}") from None
    if act.nargs == 0:                                 # a switch
        if not isinstance(value, bool):
            raise ValueError(f"{where}.{key}: true or false")
        return [flag] if value else ([f"--no-{name}"] if isinstance(act, argparse.BooleanOptionalAction) else [])
    items = value if isinstance(value, list) else [value]
    if isinstance(value, list) and not isinstance(act, argparse._AppendAction):
        raise ValueError(f"{where}.{key}: one value, not a list")
    words: list[str] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (str, int, float)):
            raise ValueError(f"{where}.{key}: expected a string or number, not {item!r}")
        item = str(item)
        if name in CONFIG_PATHS and item:
            item = str((here / os.path.expanduser(item)))
        words += [flag, item]
    return words


def config_name(key) -> str:
    return str(key).strip().lower().replace("_", "-").removeprefix("--")


def read_config(path: str) -> dict:
    """The file's object, of this format and version; ValueError for anything else."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except OSError as e:
        raise ValueError(f"cannot read: {e.strerror or e}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON: {e}") from None
    if not isinstance(raw, dict):
        raise ValueError('expected {"version": ' + str(CONFIG_VERSION) + ', "server": {...}, "engine": {...}, "models": [...]}')
    version = raw.get("version", CONFIG_VERSION)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError('"version" must be a whole number')
    if version > CONFIG_VERSION:
        raise ValueError(f"this is a v{version} config; this Basalt reads v{CONFIG_VERSION}")
    if version < CONFIG_VERSION or "models" not in raw or "v" in raw:
        old = Path(path).resolve()
        new = old.with_name(f"{old.stem}.v{CONFIG_VERSION}{old.suffix}")
        raise ValueError(f"Basalt requires v{CONFIG_VERSION} config, please upgrade: "
                         f"python3 {ROOT / 'tools/migrations/config_v1_to_v2.py'} {old} -o {new}")
    extra = set(raw) - {"version", "server", "engine", "models"}
    if extra:
        raise ValueError(f"unexpected key {sorted(extra)[0]!r}")
    return raw


def load_config_file(path: str, ap: argparse.ArgumentParser) -> ConfigFile:
    """A --config JSON file {"version": 2, "server": {...}, "engine": {...}, "models": [{...}]}. `server` holds what the
    whole server shares; `engine` what applies to every model that takes it; each entry of `models` names one model, and
    its settings win over `engine`'s. Engine flags map to values (true: switch, false: off, a list: several values). Paths
    are relative to the file; ValueError for anything else."""
    raw = read_config(path)
    server, shared, entries = raw.get("server", {}), raw.get("engine", {}), raw["models"]
    for k, v in (("server", server), ("engine", shared)):
        if not isinstance(v, dict):
            raise ValueError(f'"{k}" must be an object')
    if not isinstance(entries, list) or not entries or not all(isinstance(e, dict) for e in entries):
        raise ValueError('"models" must be a list of at least one {"model": FILE, ...}')
    for section in (server, shared, *entries):
        for key in section:
            if re.sub(r"[^a-z]", "", str(key).lower()).startswith("apikey"):
                raise ValueError("api-key can't be set in a config file; pass --api-key on the command line")
    here = Path(path).resolve().parent
    known = server_options(ap)
    server_argv: list[str] = []
    for key, value in server.items():
        name = config_name(key)
        if name == "engine-bin":
            raise ValueError("server.engine-bin: engine-bin is set by basalt serve")
        if name in ENTRY_OPTIONS or name in MODEL_OPTIONS:
            raise ValueError(f'server.{key}: a model setting, set it under "engine" or in a model entry')
        if name not in known:
            raise ValueError(f'server.{key}: not a server option (engine flags go under "engine")')
        server_argv += option_words("server", key, value, known[name], here)
    shared_argv: list[str] = []
    shared_words: list[str] = []
    for key, value in shared.items():
        name = config_name(key)
        if name in ENTRY_OPTIONS:
            raise ValueError(f"engine.{key}: set it in a model entry")
        if name in MODEL_OPTIONS:
            shared_argv += option_words("engine", key, value, known[name], here)
        elif name in known:
            raise ValueError(f'engine.{key}: a server option, set it under "server"')
        else:
            shared_words += engine_words(key, value)
    models: list[ModelEntry] = []
    for i, entry in enumerate(entries):
        where = f"models[{i}]"
        if not isinstance(entry.get("model"), str) or not entry["model"].strip():
            raise ValueError(f'{where}: "model" names the model file')
        out = ModelEntry(str(here / os.path.expanduser(entry["model"])), list(shared_argv), list(shared_words))
        for key, value in entry.items():
            name = config_name(key)
            if name in ENTRY_OPTIONS or name in MODEL_OPTIONS:
                out.argv += option_words(where, key, value, known[name], here)
            elif name in known:
                raise ValueError(f'{where}.{key}: a server option, set it under "server"')
            else:
                out.own_words += engine_words(key, value)
        models.append(out)
    return ConfigFile(server_argv, models)


def model_settings(ap: argparse.ArgumentParser, cli: list[str], refused=None) -> list[tuple]:
    """parse_settings() for each model the server runs: the --config file's entries, else the command line's one.
    `refused(path, group)` says a model has no such engine flag; a top-level setting it refuses is left out of that model's
    words and named in `a.skipped`. With --config the command line's settings apply to every entry."""
    a_cli = ap.parse_known_args(cli)[0]
    if not a_cli.config:
        return [parse_settings(ap, cli)]
    if a_cli.model or a_cli.alias:
        raise ValueError("--model and --alias are set in a model entry")
    cfg = load_config_file(a_cli.config, ap)
    out = []
    for entry in cfg.models:
        kept, skipped = [], []
        for group in engine_groups(entry.shared_words):
            if refused is not None and refused(entry.path, group):
                skipped.append(group[0])
            else:
                kept += group
        settings = parse_settings(ap, cli, cfg.server_argv, entry.argv, kept + entry.own_words)
        settings[0].skipped = skipped
        out.append(settings)
    return out


def parse_settings(ap: argparse.ArgumentParser, cli: list[str], server_argv=(), model_argv=(), words=()):
    """(args, argv, file_words, cli_words): parsed options, their source words and the engine words from each.
    The engine reads its flags in order and a later one wins, so file words then command-line words keep
    command line > config file > defaults; SystemExit(2) for a bad value, ValueError for a stray word or a clash."""
    a_cli, extras = ap.parse_known_args(cli)
    cli_words = split_engine_extras(extras)
    file_argv, file_words = [*server_argv, *model_argv], list(words)
    a = ap.parse_known_args(file_argv + cli)[0]
    a.skipped = []
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
            # a request is alone, has the same length, so a request never outgrows its slot). Without the flag the
            # engine's default is shared, once the server has asked the engine for it (split_default).
            src = cli_words if "--max-context" in cli_words or "--max-context" not in file_words else file_words
            at = flag_index(src, "--max-context")
            if at is not None and at + 1 >= len(src):
                raise ValueError("--max-context needs a value")
            if at is None:
                a.split_default = True
            else:
                ctx = int(src[at + 1])
                src[at + 1] = str(slot_share(ctx, a.parallel))
                a.shared_context = ctx                     # the total, for the startup line
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


# flag -> the console's name; a -mib or -gib flag's unit follows its value
ENGINE_NAMES = {"max-context": "ctx", "pool-workers": "pool", "conversation-cache-disk-mib": "disk-cache",
                "conversation-cache-mib": "ram-cache", "vram-reserve-mib": "reserve"}
UNIT_SUFFIXES = {"-mib": "MiB", "-gib": "GiB"}


def engine_item(group: list[str]) -> str:
    """One engine flag with its values as `name=value`; a switch is its bare name."""
    flag, values = group[0][2:], group[1:]
    unit = next((u for suffix, u in UNIT_SUFFIXES.items() if flag.endswith(suffix) and len(values) == 1), "")
    name = ENGINE_NAMES.get(flag, flag[:-4] if unit else flag)
    return f"{name}={','.join(values)}{unit}" if values else name


def start_items(file_words: list[str], cli_words: list[str], shared: tuple[int, int] | None = None,
                grammar: tuple[bool, str] | None = None, debug: bool = False) -> list[str]:
    """The engine words from --config and the command line as key=value items for a model's startup line, a flag given
    twice shown once with the value that wins. `shared` is (total, slots) when --max-context was divided among the slots;
    `grammar` is (on, origin), None when left at its default. The conversation folder shows only with --debug."""
    groups = engine_groups(file_words) + engine_groups(cli_words)
    last = {g[0]: i for i, g in enumerate(groups)}
    items = []
    for i, g in enumerate(groups):
        if last[g[0]] != i or (g[0] == "--conversation-cache-dir" and not debug):
            continue
        item = engine_item(g)
        if shared and g[0] == "--max-context":
            item += f" ({shared[0]}/{shared[1]} slots)"
        items.append(item)
    if grammar is not None:
        items.append(f"grammar={'on' if grammar[0] else 'off'}")
    return items


def slot_share(total: int, parallel: int) -> int:
    """Each of `parallel` slots' part of a shared --max-context."""
    return max(MIN_SLOT_CONTEXT, total // parallel)


def with_spec(items: list[str], spec) -> list[str]:
    """A model's items with `spec=N` the window the engine will run (its --gpu-plan answer), not the flag as given."""
    return items if not isinstance(spec, int) else [x for x in items if not x.startswith("spec=")] + [f"spec={spec}"]


def shared_line(shared: list[str], sleeps: list) -> tuple[str, bool]:
    """The items every model shares, with the idle sleep when all of several models sleep after the same time; and
    whether it took the sleep (the models then do not say it each)."""
    s = {x or 0 for x in sleeps}
    sleep = len(sleeps) > 1 and len(s) == 1 and next(iter(s)) > 0
    return " ".join(shared + ([f"sleep={next(iter(s)):g}s"] if sleep else [])), sleep


def model_line(name: str, width: int, items: list[str]) -> str:
    """A model's startup line: its name padded to the column of the longest name, then its items."""
    return f"[server] {name:<{width}} {' '.join(items)}".rstrip()


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _whole(v):
    return _num(v) and v == int(v)


# the engine's penalty window holds at most this many tokens (src/program/serve/state.hpp, kPenaltyWindowCap)
PENALTY_WINDOW_MAX = 4096
# sampling key -> (converter, test, what is expected); the model file, the launch options and every request share it
SAMPLING_KEYS = {
    "temperature": (float, lambda v: _num(v) and v >= 0, "a number >= 0 (0 = greedy)"),
    "top_p": (float, lambda v: _num(v) and 0 < v <= 1, "0 < top_p <= 1"),
    "min_p": (float, lambda v: _num(v) and 0 <= v <= 1, "0 <= min_p <= 1"),
    "top_k": (int, lambda v: _whole(v) and v >= -1, "an integer >= 0 (0 or -1 = off: every token)"),
    "presence_penalty": (float, _num, "a number"),
    "frequency_penalty": (float, _num, "a number"),
    "repetition_penalty": (float, lambda v: _num(v) and v > 0, "a number > 0 (1 = off)"),
    "penalty_last_n": (int, lambda v: _whole(v) and 0 <= v <= PENALTY_WINDOW_MAX,
                       f"an integer 0..{PENALTY_WINDOW_MAX} (the engine's penalty window)"),
    "seed": (int, lambda v: _whole(v) and -2 ** 63 <= v < 2 ** 64, "an integer"),
}


def checked_sampling(req: dict) -> None:
    """A request's sampling fields against SAMPLING_KEYS (the model file's rules): ValueError naming the first one out
    of range, so the request gets a 400 before streaming; a field left out or null takes the defaults.  A value is
    never changed: what passes reaches the engine as sent."""
    for key, (_, ok, expected) in SAMPLING_KEYS.items():
        value = req.get(key)
        if value is not None and not ok(value):
            raise ValueError(f"{key}: expected {expected}, not {value!r}")
    rd = req.get("relaxed_drafting")
    if rd is not None and rd is not False and not (_num(rd) and 0 <= rd <= 1):
        raise ValueError(f"relaxed_drafting: expected false or a number 0..1, not {rd!r}")


def sampling_defaults_from_config(cfg: dict) -> dict:
    """The model file's sampling defaults (its GGUF metadata's general.sampling.* keys, serve/engine/modelfile.py):
    defaults for fields a request omits (temperature, top_p, top_k, min_p,
    presence/repetition/frequency penalties, penalty_last_n, seed). Request fields win; a bad value refuses to
    start; unknown keys are named and ignored."""
    out = {}
    for key, value in (cfg.get("sampling") or {}).items():
        if value is None or (key == "top_k" and value == 0):   # 0: no top-k cut
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


SAMPLING_NAMES = {"temperature": "temp", "presence_penalty": "pres", "frequency_penalty": "freq",
                  "repetition_penalty": "rep"}      # the console's short names
SAMPLING_OFF = {"min_p": 0, "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty": 1, "top_k": -1}


def sampling_items(effective: dict) -> list[str]:
    """The effective sampling defaults as items, leaving out those that change nothing."""
    return [f"{SAMPLING_NAMES.get(k, k)}={v}" for k, v in effective.items()
            if not (k in SAMPLING_OFF and v == SAMPLING_OFF[k]) and not (k == "top_k" and v == 0)]


def vision_setting(text: str) -> str:
    """argparse type: one of the --vision modes."""
    try:
        return vision_mode(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def image_url_host(text: str) -> str:
    """argparse type: HOST or HOST:PORT an image URL from another machine may name."""
    try:
        parse_url_host(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    return text.strip()


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
    """argparse type: a token count, a whole number >= 0 (0: none)."""
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
                    help="a JSON file {\"server\": {...}, \"engine\": {...}, \"models\": [{\"model\": FILE, ...}]}: settings "
                         "named without the dashes. \"server\" holds what the server shares, \"engine\" what applies to "
                         "every model that takes it, a model entry that model's own; a list gives a flag several "
                         "values; the command line wins over it, for every model")
    ap.add_argument("--vision", nargs="?", const="", default="off", type=vision_setting, metavar="MODE|GPU",
                    help="image input: off (default), dynamic (inside the engine: it borrows expert-cache slots "
                         "on the main GPU while it reads an image, no standing VRAM), cpu (no VRAM, slow) or a "
                         "GPU number, as nvidia-smi numbers them: the encoder sidecar, always on that card")
    ap.add_argument("--image-url-host", action="append", default=[], metavar="HOST[:PORT]", type=image_url_host,
                    help="let requests from other machines name http(s) image URLs on exactly this host (repeatable; "
                         "without a port, 80 and 443). A name must resolve to public addresses; an IP address is "
                         "taken as written. Requests from this machine need no flag")
    ap.add_argument("--image-dir", action="append", default=[], metavar="DIR",
                    help="let requests from other machines name image files under this directory (repeatable). "
                         "Requests from this machine need no flag")
    ap.add_argument("--image-remote-only", action="store_true",
                    help="treat requests from this machine as remote too, so they get only data: URLs and what "
                         "--image-url-host and --image-dir allow. For a server behind anything on this machine that "
                         "forwards other people's requests")
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
    ap.add_argument("--lazy", action="store_true", help="start the API unloaded; load on first request; a sidecar image encoder is dropped, --vision dynamic stays")
    ap.add_argument("--idle-unload", type=float, default=None, metavar="SECONDS",
                    help="unload the model after this many seconds without requests; the next request loads it again")
    ap.add_argument("--idle-sleep", type=float, default=None, metavar="SECONDS",
                    help="put the model to sleep after this many seconds without requests: its VRAM is freed, its RAM "
                         "copy, conversations and learned expert placement stay, and the next request wakes it in a few "
                         "seconds (Linux; the driver suspends the engine's GPU state)")
    for key, text in (
            ("temperature", "0 = greedy; higher is more random"),
            ("top-p", "nucleus cutoff, 0 < value <= 1 (1 = off)"),
            ("top-k", "keep the K likeliest tokens (0 = off: every token)"),
            ("min-p", "drop tokens less likely than this times the likeliest, 0..1 (0 = off)"),
            ("presence-penalty", "flat penalty on a token seen in the window"),
            ("frequency-penalty", "penalty growing with how often a token was seen"),
            ("repetition-penalty", "multiplicative penalty on seen tokens, > 0 (1 = off)"),
            ("penalty-last-n", "the window the penalties look back over, in tokens (at most 4096)"),
            ("seed", "fix the random draw, an integer")):
        ap.add_argument(f"--{key}", type=sampling_option(key.replace("-", "_")), metavar="N", default=None,
                        help=f"default {key.replace('-', '_')} for requests that leave it out, over the model file's: {text}")
    ap.add_argument("--min-free-vram-mib", type=int, default=None,
                    help="load an unloaded model only when this much VRAM is free, else answer 503")
    ap.add_argument("--reasoning-budget-tokens", type=budget_count, default=None, metavar="N",
                    help="default thinking budget for requests that leave reasoning_budget_tokens out (0 = none): at "
                         "the budget the thinking is closed and the answer starts, so set it below the request's "
                         "max_tokens")
    ap.add_argument("--thinking-loop-tokens", type=budget_count, default=256, metavar="N",
                    help="end the thinking and let the model answer when it repeats one token this many times in a "
                         "row inside the thinking (0 = never)")
    ap.add_argument("--answer-room", action=argparse.BooleanOptionalAction, default=True,
                    help="reserve space for closing the thinking and the answer when the output cap is at least 1024 tokens; "
                         "an explicit request budget of 0 opts out")
    return ap
