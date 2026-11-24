"""Persistent token counts and pricing for the basalt server.

The counts the /metrics endpoint returns (requests, prompt tokens, output tokens, etc.) live only in
the process, so a restart loses them. This module keeps them in a small JSON file next to the config,
read when the server starts and written when a request ends. Pricing turns those counts into a dollar
figure: what an API would have charged for the same work.

    python -m unittest serve.totals -v
"""
from __future__ import annotations
import json
import os
import tempfile
import time
from pathlib import Path

TOTALS_KEYS = ("requests", "prompt", "reused", "written", "prompt_ms", "decode_ms")


def totals_path_of(given: str, config: str | None) -> str | None:
    """Where the lifetime counters live: --totals-file, else '<config>.totals.json' next to the config.
    'none' or 'off' disables persistence. No config and no --totals-file: nowhere to keep them."""
    if given and given.lower() in ("none", "off"):
        return None
    if given:
        return given
    if not config:
        return None
    return str(Path(config).with_suffix("")) + ".totals.json"


def totals_load_file(path: str | Path) -> dict:
    """Read a saved totals file. A missing or broken file is a first start: returns {}.
    A file whose top-level value is not a dict (a list, string, number, etc.) is treated the same way."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def totals_add_saved(totals: dict, saved: dict) -> dict:
    """Add a saved file's counts into the live totals dict. Only numbers are added; a file another
    version wrote or a hand-edit cannot spoil a start. The 'since' timestamp moves back to the
    saved one so the printed window covers every start the file remembers."""
    for key in TOTALS_KEYS:
        v = saved.get(key)
        if isinstance(v, (int, float)):
            totals[key] = totals.get(key, 0) + v
    if isinstance(saved.get("since"), (int, float)) and saved["since"] < totals.get("since", float("inf")):
        totals["since"] = saved["since"]
    return totals


def totals_write_file(path: str | Path, totals: dict) -> None:
    """Write the totals to a file atomically (via a temp file in the same directory)."""
    data = {"since": totals.get("since", time.time())}
    for key in TOTALS_KEYS:
        data[key] = totals.get(key, 0)
    # pricing fields (informational, not reloaded as config)
    for key in ("cost_input_per_million", "cost_output_per_million",
                "cost_cache_write_per_million", "cost_cache_hit_per_million", "cost_usd"):
        if key in totals:
            data[key] = totals[key]
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def totals_reset_file(path: str | Path) -> Path | None:
    """Close a costing window: move the current file to .prev.json and return its path, or None
    if there is nothing to reset."""
    path = Path(path)
    if not path.is_file():
        return None
    prev = path.with_suffix(".prev.json")
    path.replace(prev)
    return prev


def totals_cost(totals: dict, in_per_million=0, out_per_million=0,
                cache_write_per_million=None, cache_hit_per_million=None) -> float:
    """The prices applied to the counts, in US dollars. A prompt splits into cache-write (new tokens)
    and cache-hit (reused tokens), each at its own rate. A cache rate left as None takes the plain
    input price — an API priced without prompt caching."""
    given = totals.get("prompt", 0) or 0
    held = min(totals.get("reused", 0) or 0, given)
    plain = float(in_per_million or 0)
    write = plain if cache_write_per_million is None else float(cache_write_per_million or 0)
    read = plain if cache_hit_per_million is None else float(cache_hit_per_million or 0)
    return ((given - held) * write + held * read
            + (totals.get("written", 0) or 0) * float(out_per_million or 0)) / 1e6


def totals_with_cost(totals: dict, in_per_million=0, out_per_million=0,
                     cache_write_per_million=None, cache_hit_per_million=None) -> dict:
    """Add the prices and computed cost to a copy of totals for the /metrics response."""
    plain = in_per_million or 0
    totals["cost_input_per_million"] = plain
    totals["cost_output_per_million"] = out_per_million or 0
    totals["cost_cache_write_per_million"] = plain if cache_write_per_million is None else cache_write_per_million
    totals["cost_cache_hit_per_million"] = plain if cache_hit_per_million is None else cache_hit_per_million
    totals["cost_usd"] = round(totals_cost(totals, in_per_million, out_per_million,
                                           cache_write_per_million, cache_hit_per_million), 4)
    return totals
