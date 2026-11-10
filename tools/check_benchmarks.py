#!/usr/bin/env python3
"""Validate docs/community/benchmarks.json (stdlib only). Exits 1 with per-row messages.

--fix rewrites a valid-JSON file in the canonical format, then validates it.
"""
import json
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
MODELS = ("Qwen3.8-Flash-Next", "DeepSeek-V4.1-Flash")   # the engine's model names (model_name of each adapter)
KV_FORMATS = ("fp16", "int8", "q4_0", "k8v4", "fp8+fp4")  # the cache formats the engine reports (DeepSeek --kv fp8 is fp8+fp4)


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
TOP_ORDER = ("preset", "runs")
FIX_HINT = "python3 tools/check_benchmarks.py --fix docs/community/benchmarks.json"


def ordered(d, order):
    """Known keys in the fixed order, unknown keys after them in file order."""
    if not isinstance(d, dict):
        return d
    return {**{k: d[k] for k in order if k in d}, **{k: v for k, v in d.items() if k not in order}}


def canonical(data):
    """The one accepted file form: fixed key order, 2-space indent, no trailing newline."""
    data = ordered(data, TOP_ORDER)
    if isinstance(data.get("runs"), list) if isinstance(data, dict) else False:
        rows = []
        for r in data["runs"]:
            r = ordered(r, ROW_ORDER)
            if isinstance(r, dict) and isinstance(r.get("gpus"), list):
                r["gpus"] = [ordered(g, GPU_ORDER) for g in r["gpus"]]
            rows.append(r)
        data["runs"] = rows
    return json.dumps(data, indent=2, ensure_ascii=False)


def lower_ascii(s):
    """Case folding that C++ (community_row.cpp) reproduces exactly: ASCII letters only."""
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in s)


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


def validate(data):
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
        rows = data["runs"] if isinstance(data.get("runs"), list) else []
        for i, r in enumerate(rows):
            who = r.get("submitter") if isinstance(r, dict) else None
            label = f"runs[{i}]" + (f" ({who})" if isinstance(who, str) and who else "")
            check_row(r, lambda m, label=label: problems.append(f"{label}: {m}"))
            first = next((j for j in range(i) if isinstance(r, dict) and isinstance(rows[j], dict) and same_setup(rows[j], r)), None)
            if first is not None:
                problems.append(f"{label}: same submitter, model, quant and hardware as runs[{first}]: update your run instead")
    return problems


def main(argv):
    args = argv[1:]
    fix = "--fix" in args
    args = [a for a in args if a != "--fix"]
    path = args[0] if args else "docs/community/benchmarks.json"
    try:
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
        data = json.loads(text)
    except (OSError, ValueError) as e:
        print(f"{path}: {e}")
        return 1
    problems = validate(data)
    want = canonical(data)
    if fix and text != want:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(want)
        print(f"{path}: rewritten in the canonical format")
        text = want
    elif text != want:
        problems.append(f"not in the canonical format; run: {FIX_HINT}")
    for p in problems:
        print(f"{path}: {p}")
    if problems:
        return 1
    print(f"{path}: ok, {len(data['runs'])} runs")
    return 0


sys.exit(main(sys.argv))
