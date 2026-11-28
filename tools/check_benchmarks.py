#!/usr/bin/env python3
"""Validate the community run files (stdlib only). Exits 1 with per-file messages.

    check_benchmarks.py [--fix] [DIR]       DIR is docs/community by default

DIR holds preset.json and runs/<model slug>/<id>.json, one row per file. Every file must be in the house format, named
by its content (the id of row_id) and in the folder of its model's slug, and no two files may hold the same setup.
--fix rewrites the files that are valid JSON in the house format, then validates.
"""
import hashlib
import json
import math
import os
import re
import sys

REQUIRED = {"submitter", "date", "commit", "preset", "model", "gpus", "ram_gib", "os",
            "quant", "kv", "prefill_16k_tps", "prefill_64k_tps", "decode_tps"}
TESTS = ("conv", "struct", "prose")   # the decode fixtures of the preset
TEST_TPS = tuple(f"decode_{t}_tps" for t in TESTS)
TEST_ACCEPT = tuple(f"accept_{t}_pct" for t in TESTS)
TEST_DRAFT = tuple(f"draft_len_{t}" for t in TESTS)
OPTIONAL = {"notes", "cpu", "ram_gbps", "engine_args", *TEST_TPS, *TEST_ACCEPT, *TEST_DRAFT}
GPU_KEYS = {"name", "vram_gib", "power_w", "pcie", "h2d_gbps"}
PCIE = re.compile(r"[1-9](\.0)? x(1|2|4|8|16|32)")
PLACEHOLDER = "YOUR_NAME"
CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")
SYSTEMS = ("Linux", "Windows", "WSL2")
MODELS = ("Qwen3.8-Flash-Next", "Qwen3.8-27B", "DeepSeek-V4.1-Flash", "Hy4-flash", "Inkling-2-Mini", "ERNIE-5.5-Flash",
          "Gemma-4-26B-A4B", "Gemma-4-31B", "Gemma-4-12B")   # the engine's model names (model_name of each adapter)
KV_FORMATS = ("fp16", "int8", "q4_0", "k8v4", "fp8+fp4", "fp8")  # the cache formats the engine reports (DeepSeek --kv fp8 is fp8+fp4)


def _reject_constant(token):
    raise ValueError(f"non-finite JSON constant {token!r}")


def loads_strict(text):
    return json.loads(text, parse_constant=_reject_constant)


def is_num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        return math.isfinite(v)
    except OverflowError:
        return False


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
    if "pcie" in g and not (isinstance(g["pcie"], str) and PCIE.fullmatch(g["pcie"])):
        err(f"gpus[{k}].pcie must look like '5.0 x16'")
    if "h2d_gbps" in g and not is_pos(g["h2d_gbps"]):
        err(f"gpus[{k}].h2d_gbps must be a number > 0")


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
        elif re.search(r"\s", r["submitter"]):
            err("submitter must be one name with no spaces")
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
    for key in ("ram_gib", "prefill_16k_tps", "prefill_64k_tps", "decode_tps", *TEST_TPS, *TEST_DRAFT):
        if key in r and not is_pos(r[key]):
            err(f"{key} must be a number > 0")
    for key in TEST_ACCEPT:
        if key in r and not (is_num(r[key]) and 0 <= r[key] <= 100):
            err(f"{key} must be a number from 0 to 100")
    if "engine_args" in r and not (is_text(r["engine_args"]) and len(r["engine_args"]) <= 300 and not CONTROL.search(r["engine_args"])):
        err("engine_args must be a non-empty string of at most 300 characters, without control characters")
    if "ram_gbps" in r and not is_pos(r["ram_gbps"]):
        err("ram_gbps must be a number > 0")
    if "cpu" in r and not (is_text(r["cpu"]) and len(r["cpu"]) <= 80 and not CONTROL.search(r["cpu"])):
        err("cpu must be a non-empty string of at most 80 characters, without control characters")
    if "os" in r and r["os"] not in SYSTEMS:
        err("os must be one of " + ", ".join(SYSTEMS))
    if "quant" in r and not is_text(r["quant"]):
        err("quant must be a non-empty string")
    if "model" in r and r["model"] not in MODELS:
        err("model must be one of " + ", ".join(MODELS))
    if "kv" in r and r["kv"] not in KV_FORMATS:
        err("kv must be one of " + ", ".join(KV_FORMATS))
    if "notes" in r and not (isinstance(r["notes"], str) and len(r["notes"]) <= 200):
        err("notes must be a string of at most 200 characters")


ROW_ORDER = ("submitter", "date", "commit", "preset", "model", "gpus", "ram_gib", "cpu", "ram_gbps", "os",
             "quant", "kv", "prefill_16k_tps", "prefill_64k_tps", "decode_tps", *TEST_TPS, *TEST_ACCEPT, *TEST_DRAFT,
             "engine_args", "notes")
GPU_ORDER = ("name", "vram_gib", "power_w", "pcie", "h2d_gbps")
RUNS_DIR = "runs"
PRESET_FILE = "preset.json"
DEFAULT_DIR = "docs/community"
ID_LEN = 12
FIX_HINT = "python3 tools/check_benchmarks.py --fix"


def ordered(d, order):
    """Known keys in the fixed order, unknown keys after them in file order."""
    if not isinstance(d, dict):
        return d
    return {**{k: d[k] for k in order if k in d}, **{k: v for k, v in d.items() if k not in order}}


def canonical_row(row):
    """The one accepted form of a run file: fixed key order, 2-space indent, one trailing newline."""
    row = ordered(row, ROW_ORDER)
    if isinstance(row, dict) and isinstance(row.get("gpus"), list):
        row["gpus"] = [ordered(g, GPU_ORDER) for g in row["gpus"]]
    return json.dumps(row, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def canonical_preset(preset):
    return json.dumps({"preset": preset}, indent=2, ensure_ascii=False) + "\n"


def lower_ascii(s):
    """Case folding that C++ (community_row.cpp) reproduces exactly: ASCII letters only."""
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in s)


def model_slug(model):
    """The folder of a model's runs: its name in ASCII lower case."""
    return lower_ascii(model)


def row_id(row):
    """The file name (without .json) of a run: the first ID_LEN hex digits of the sha256 of the setup the "same setup"
    rule compares, taken as exact values: the submitter in ASCII lower case, model, quant, cpu, ram_gib, os and every
    GPU's name, vram_gib and power_w. A field the row lacks is empty. The text hashed is a compact JSON array, numbers
    as "%.0f" and strings quoted as json_quote in community_row.cpp does, so the bench computes the same id."""
    def q(v):
        return json.dumps(v if isinstance(v, str) else "", ensure_ascii=False)

    def n(v):
        return "%.0f" % v if is_num(v) else ""
    gpus = row.get("gpus")
    gpus = [g for g in gpus if isinstance(g, dict)] if isinstance(gpus, list) else []
    sub = row.get("submitter")
    parts = [q(lower_ascii(sub) if isinstance(sub, str) else ""), q(row.get("model")), q(row.get("quant")), q(row.get("cpu")),
             n(row.get("ram_gib")), q(row.get("os")),
             "[" + ",".join(f"[{q(g.get('name'))},{n(g.get('vram_gib'))},{n(g.get('power_w'))}]" for g in gpus) + "]"]
    text = "[" + ",".join(parts) + "]"
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:ID_LEN]


def run_path(row):
    """<model slug>/<id>.json, relative to the runs folder."""
    return f"{model_slug(row['model'])}/{row_id(row)}.json"


def same_setup(a, b):
    """True when two rows are one setup: submitter (ASCII case-insensitive), model, quant, every GPU (name, vram_gib,
    power_w, in order), cpu, ram_gib and os all match. A field missing from either row matches; kv is not compared (the
    preset fixes it and only the engine version changes it)."""
    def eq(x, y):
        if x is None or y is None:
            return True
        if isinstance(x, str) and isinstance(y, str):
            return x == y
        return is_num(x) and is_num(y) and x == y
    sa, sb = a.get("submitter"), b.get("submitter")
    if isinstance(sa, str) and isinstance(sb, str):
        sa, sb = lower_ascii(sa), lower_ascii(sb)
    if not all(eq(x, y) for x, y in ((sa, sb), *((a.get(k), b.get(k)) for k in ("model", "quant", "cpu", "ram_gib", "os")))):
        return False
    ga, gb = a.get("gpus"), b.get("gpus")
    if isinstance(ga, list) and isinstance(gb, list):
        if len(ga) != len(gb):
            return False
        for x, y in zip(ga, gb):
            if not (isinstance(x, dict) and isinstance(y, dict)) or not all(eq(x.get(k), y.get(k)) for k in ("name", "vram_gib", "power_w")):
                return False
    return True


def read_text(path):
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def list_runs(root):
    """(relative path, absolute path) of every entry under root/runs, sorted; problems for what is not a run file."""
    found, problems = [], []
    base = os.path.join(root, RUNS_DIR)
    if not os.path.isdir(base):
        return found, [f"{RUNS_DIR}: the runs folder is missing"]
    for slug in sorted(os.listdir(base)):
        d = os.path.join(base, slug)
        rel = f"{RUNS_DIR}/{slug}"
        if not os.path.isdir(d):
            problems.append(f"{rel}: only model folders belong directly in {RUNS_DIR}/")
            continue
        for name in sorted(os.listdir(d)):
            full = os.path.join(d, name)
            if os.path.isfile(full) and re.fullmatch(r"[0-9a-f]{%d}\.json" % ID_LEN, name):
                found.append((f"{rel}/{name}", full))
            else:
                problems.append(f"{rel}/{name}: not a run file (<{ID_LEN} hex digits>.json)")
    return found, problems


def check_file(rel, text, err):
    """Validate one run file's text; returns the row when it parsed to an object."""
    try:
        row = loads_strict(text)
    except ValueError as e:
        err(f"not valid JSON: {e}")
        return None
    if not isinstance(row, dict):
        err("the file must hold one row object")
        return None
    problems = []
    check_row(row, problems.append)
    for p in problems:
        err(p)
    if not problems:
        want = run_path(row)
        if rel.split("/", 1)[1] != want:
            err(f"the file belongs at {RUNS_DIR}/{want} (the model's folder and the id of its setup)")
    if text != canonical_row(row):
        err(f"not in the house format; run: {FIX_HINT}")
    return row


def check_dir(root, fix=False):
    """Validate root. Returns (problems, rows) where rows is [(relative path, row)] of the files that parsed."""
    problems, rows = [], []
    found, stray = list_runs(root)
    problems += stray
    rewritten = 0
    for rel, full in found:
        text = read_text(full)
        if fix:
            try:
                parsed = loads_strict(text)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict) and text != canonical_row(parsed):
                text = canonical_row(parsed)
                write_text(full, text)
                rewritten += 1
        row = check_file(rel, text, lambda m, rel=rel: problems.append(f"{rel}: {m}"))
        if row is not None:
            rows.append((rel, row))
    pp = os.path.join(root, PRESET_FILE)
    try:
        text = read_text(pp)
        data = loads_strict(text)
        if not isinstance(data, dict) or set(data) != {"preset"} or not is_text(data["preset"]):
            problems.append(f"{PRESET_FILE}: must be an object with one key, 'preset', a non-empty string")
        elif text != canonical_preset(data["preset"]):
            if fix:
                write_text(pp, canonical_preset(data["preset"]))
                rewritten += 1
            else:
                problems.append(f"{PRESET_FILE}: not in the house format; run: {FIX_HINT}")
    except (OSError, ValueError) as e:
        problems.append(f"{PRESET_FILE}: {e}")
    # one run per setup: the same submitter, model, quant and hardware (a missing field matches, so a row from before
    # a field existed still collides with the same setup that has it). Only rows of one submitter and model can match.
    buckets = {}
    for rel, row in rows:
        if isinstance(row.get("submitter"), str):
            buckets.setdefault((row.get("model"), lower_ascii(row["submitter"])), []).append((rel, row))
    for group in buckets.values():
        for i, (rel, row) in enumerate(group):
            first = next((o for o, orow in group[:i] if same_setup(orow, row)), None)
            if first is not None:
                problems.append(f"{rel}: same submitter, model, quant and hardware as {first}: update that run (the same file) instead")
    if fix and rewritten:
        print(f"{rewritten} file(s) rewritten in the house format")
    return problems, rows


def main(argv):
    args = argv[1:]
    fix = "--fix" in args
    args = [a for a in args if a != "--fix"]
    root = args[0] if args else DEFAULT_DIR
    problems, rows = check_dir(root, fix)
    for p in problems:
        print(f"{root}/{p}")
    if problems:
        return 1
    print(f"{root}: ok, {len(rows)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
