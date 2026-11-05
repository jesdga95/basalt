#!/usr/bin/env python3
"""Turn a single-model server config into the v2 shape with "models": prints it, or writes it to a new file.

    python3 tools/migrations/config_v1_to_v2.py basalt_config.json [-o NEW.json]

The input is only read. What says which model it is goes in the model entry; the rest goes in the top-level "engine", where
it applies to every model added later; what the whole server shares stays in "server".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VERSION = 2
# stays with the model: its identity, what it costs and how it answers
ENTRY_KEYS = {"model", "alias", "gpu", "vision", "parallel", "slot-full-context", "max-context", "kv",
              "temperature", "top-p", "top-k", "min-p", "presence-penalty", "frequency-penalty", "repetition-penalty",
              "penalty-last-n", "seed", "reasoning-budget-tokens", "thinking-loop-tokens", "answer-room", "grammar",
              "fit-max-tokens"}
# server options that apply per model and leave "server"
MODEL_OPTIONS = ENTRY_KEYS | {"idle-sleep", "idle-unload", "min-free-vram-mib"}


def name_of(key) -> str:
    return str(key).strip().lower().replace("_", "-").removeprefix("--")


def migrated(old) -> dict:
    """The v2 object for a v1 one; ValueError when `old` is not a v1 config that names its model."""
    if not isinstance(old, dict) or "models" in old:
        raise ValueError("not a v1 config: it already has \"models\"" if isinstance(old, dict) else "expected a JSON object")
    extra = set(old) - {"v", "server", "engine"}
    if extra:
        raise ValueError(f"not a v1 config: unexpected key {sorted(extra)[0]!r}")
    server, engine, entry, shared = {}, {}, {}, {}
    for key, value in old.get("server", {}).items():
        name = name_of(key)
        if name in ENTRY_KEYS:
            entry[name] = value
        elif name in MODEL_OPTIONS:
            shared[name] = value
        else:
            server[name] = value
    for key, value in old.get("engine", {}).items():
        name = name_of(key)
        (entry if name in ENTRY_KEYS else shared)[name] = value
    if "model" not in entry:
        raise ValueError("this config names no model (\"model\" in \"server\"): add it first")
    engine = shared
    out = {"version": VERSION, "models": [entry]}
    if engine:
        out["engine"] = engine
    if server:
        out["server"] = server
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="the old config file; only read")
    ap.add_argument("-o", "--output", help="write here instead of printing; the file must not exist")
    a = ap.parse_args(argv)
    try:
        with open(a.config, encoding="utf-8") as f:
            new = migrated(json.load(f))
    except (OSError, ValueError) as e:
        print(f"{a.config}: {e}", file=sys.stderr)
        return 2
    text = json.dumps(new, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if not a.output:
        sys.stdout.write(text)
        return 0
    try:
        with Path(a.output).open("x", encoding="utf-8") as f:
            f.write(text)
    except OSError as e:
        print(f"{a.output}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
