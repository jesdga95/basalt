#!/usr/bin/env python3
"""Validate docs/community/benchmarks.json (stdlib only). Exits 1 with per-row messages."""
import json
import re
import sys

REQUIRED = {"submitter", "date", "commit", "preset", "gpus", "ram_gib", "os",
            "quant", "prefill_16k_tps", "prefill_64k_tps", "decode_tps"}
OPTIONAL = {"notes"}
GPU_KEYS = {"name", "vram_gib", "power_w"}
PLACEHOLDER = "YOUR_NAME"
CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")
SYSTEMS = ("Linux", "Windows", "WSL2")


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def is_pos(v):
    return is_num(v) and v > 0


def is_text(v):
    return isinstance(v, str) and v.strip() != ""


def check_gpu(g, k, err):
    if not isinstance(g, dict):
        err(f"gpus[{k}] must be an object")
        return
    for key in sorted(set(g) - GPU_KEYS):
        err(f"gpus[{k}]: unknown key '{key}'")
    if not is_text(g.get("name")):
        err(f"gpus[{k}].name must be a non-empty string")
    for key in ("vram_gib", "power_w"):
        if not is_pos(g.get(key)):
            err(f"gpus[{k}].{key} must be a number > 0")


def check_row(r, err):
    if not isinstance(r, dict):
        err("row must be an object")
        return
    for key in sorted(REQUIRED - set(r)):
        err(f"missing required key '{key}'")
    if "link" in r:
        err("'link' is no longer a field; remove it")
    for key in sorted(set(r) - REQUIRED - OPTIONAL - {"link"}):
        err(f"unknown key '{key}'")
    if "submitter" in r:
        if not is_text(r["submitter"]):
            err("submitter must be a non-empty string")
        elif r["submitter"].strip().upper() == PLACEHOLDER:
            err(f"submitter is still the placeholder {PLACEHOLDER}; use the name you want shown")
        elif len(r["submitter"]) > 40:
            err("submitter must be at most 40 characters")
        elif CONTROL.search(r["submitter"]):
            err("submitter must not contain control characters")
    if "date" in r:
        d = r["date"]
        ok = isinstance(d, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d)
        if ok:
            try:
                import datetime
                datetime.date.fromisoformat(d)
            except ValueError:
                ok = False
        if not ok:
            err("date must be a real date as YYYY-MM-DD")
    if "commit" in r and not (isinstance(r["commit"], str) and re.fullmatch(r"[0-9a-fA-F]{7,40}", r["commit"])):
        err("commit must be 7-40 hex characters")
    if "preset" in r and not is_text(r["preset"]):
        err("preset must be a non-empty string")
    if "gpus" in r:
        g = r["gpus"]
        if not isinstance(g, list) or not g:
            err("gpus must be a non-empty list")
        else:
            for k, x in enumerate(g):
                check_gpu(x, k, err)
    for key in ("ram_gib", "prefill_16k_tps", "prefill_64k_tps", "decode_tps"):
        if key in r and not is_pos(r[key]):
            err(f"{key} must be a number > 0")
    if "os" in r and r["os"] not in SYSTEMS:
        err("os must be one of " + ", ".join(SYSTEMS))
    if "quant" in r and not is_text(r["quant"]):
        err("quant must be a non-empty string")
    if "notes" in r and not (isinstance(r["notes"], str) and len(r["notes"]) <= 200):
        err("notes must be a string of at most 200 characters")


def main(argv):
    path = argv[1] if len(argv) > 1 else "docs/community/benchmarks.json"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        print(f"{path}: {e}")
        return 1
    problems = []
    if not isinstance(data, dict):
        problems.append("top level must be an object")
    else:
        if not is_text(data.get("preset")):
            problems.append("top-level 'preset' must be a non-empty string")
        if not isinstance(data.get("runs"), list):
            problems.append("top-level 'runs' must be a list")
        for key in sorted(set(data) - {"preset", "runs"}):
            problems.append(f"unknown top-level key '{key}'")
        for i, r in enumerate(data["runs"] if isinstance(data.get("runs"), list) else []):
            who = r.get("submitter") if isinstance(r, dict) else None
            label = f"runs[{i}]" + (f" ({who})" if isinstance(who, str) and who else "")
            check_row(r, lambda m, label=label: problems.append(f"{label}: {m}"))
    for p in problems:
        print(f"{path}: {p}")
    if problems:
        return 1
    print(f"{path}: ok, {len(data['runs'])} runs")
    return 0


sys.exit(main(sys.argv))
