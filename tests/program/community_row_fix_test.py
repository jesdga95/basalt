#!/usr/bin/env python3
"""community_row_test against tools/check_benchmarks.py: the file the bench writes is the house format byte for byte, at the path
and with the id the checker computes, and the checker's rules hold.

    community_row_fix_test.py COMMUNITY_ROW_TEST_EXE REPO_ROOT

The rows built here must match plain_row()/odd_row() in community_row_test.cpp.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile

exe, root = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(root, "tools"))
import check_benchmarks as cb  # noqa: E402

checker = os.path.join(root, "tools", "check_benchmarks.py")
failed = 0


def report(name, ok, *details):
    global failed
    print(f"community_row_fix {name}: " + ("ok" if ok else "FAILED"))
    if not ok:
        failed = 1
        for d in details:
            print(d)


def row(odd=False):
    r = {"submitter": "tester", "date": "2026-10-24", "commit": "abc1234", "preset": "community-1", "model": "Qwen3.8-Flash-Next",
         "gpus": [{"name": "RTX 5090", "vram_gib": 32, "power_w": 400, "pcie": "5.0 x16", "h2d_gbps": 55.3},
                  {"name": "RTX 5060 Ti", "vram_gib": 16, "power_w": 180, "pcie": "5.0 x8", "h2d_gbps": 27.9}],
         "ram_gib": 64, "cpu": "Test CPU 9, 16 cores", "ram_gbps": 78.2, "os": "Linux", "quant": "NVFP4", "kv": "int8",
         "prefill_16k_tps": 1234.5, "prefill_64k_tps": 2345.6, "decode_tps": 345.7,
         "decode_conv_tps": 400.1, "decode_struct_tps": 500.3, "decode_prose_tps": 300.2,
         "accept_conv_pct": 62.5, "accept_struct_pct": 80.0, "draft_len_conv": 3.2, "draft_len_struct": 4.1,
         "engine_args": "--gpu 0,1 --max-context 8192 --kv int8"}
    if odd:
        r["submitter"] = "Ünï\"q\"\\☃日本"
        r["gpus"] = [{"name": "Card \"X\"\tTi", "vram_gib": 24, "power_w": 300}]
        for key in [k for k in r if k.startswith(("decode_", "accept_", "draft_len_")) and k != "decode_tps"] + ["cpu", "ram_gbps", "engine_args"]:
            del r[key]
        r["notes"] = "line\nbreak € \x01\x7f"
    return r


def read(p):
    with open(p, "rb") as f:
        return f.read()


def put(community, r, name=None):
    """Writes the row as the run file `name` (default: where it belongs) of the community folder; returns the path."""
    path = os.path.join(community, cb.RUNS_DIR, name or cb.run_path(r))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cb.write_text(path, cb.canonical_row(r))
    return path


def fresh(d, *rows):
    community = tempfile.mkdtemp(dir=d)
    cb.write_text(os.path.join(community, cb.PRESET_FILE), cb.canonical_preset("basalt-bench --community"))
    for r in rows:
        put(community, r)
    return community


def check(community, *flags):
    r = subprocess.run([sys.executable, checker, *flags, community], capture_output=True, text=True)
    return r.returncode, r.stdout


with tempfile.TemporaryDirectory() as d:
    # the bench's file is the checker's house format, at the checker's path
    for case in ("plain", "odd"):
        r = row(case == "odd")
        community = fresh(d)
        ran = subprocess.run([exe, "--apply", community, case], capture_output=True, text=True)
        want = os.path.join(community, cb.RUNS_DIR, cb.run_path(r))
        ok = ran.returncode == 0 and os.path.normpath(ran.stdout.strip()) == os.path.normpath(want) and os.path.isfile(want) and read(want) == cb.canonical_row(r).encode("utf-8")
        report(f"{case} file is the house format at the checker's path", ok, ran.stdout, ran.stderr)
        report(f"{case} file passes the checker", check(community)[0] == 0, check(community)[1])
        again = subprocess.run([exe, "--apply", community, case], capture_output=True, text=True)
        n = sum(len(fs) for _, _, fs in os.walk(os.path.join(community, cb.RUNS_DIR)))
        report(f"{case} run again overwrites its file", again.returncode == 0 and n == 1 and check(community)[0] == 0)

    # the id agrees with the checker's on every real run file and on the odd rows; the setup rule agrees too
    real_files = [full for _, full in cb.list_runs(os.path.join(root, "docs", "community"))[0]]
    odd = os.path.join(d, "odd.json")
    cb.write_text(odd, cb.canonical_row(row(True)))
    files = real_files + [odd]
    ran = subprocess.run([exe, "--ids", *files], capture_output=True, text=True)
    want = [cb.row_id(json.loads(cb.read_text(f))) for f in files]
    report(f"ids of {len(files)} files match row_id", ran.returncode == 0 and ran.stdout.split() == want, ran.stdout[:200], ran.stderr)
    ran = subprocess.run([exe, "--dup", *real_files], capture_output=True, text=True)
    report("the real runs hold no duplicate setup", ran.returncode == 0 and ran.stdout.strip() == "" and check(os.path.join(root, "docs", "community"))[0] == 0, ran.stdout)

    # the setup rule: (name, change, same setup?); C++ and Python agree on each, and the checker refuses the pairs that
    # have two ids (a row that lacks a field matches the row that has it)
    def edit(**kw):
        return lambda r: r.update(kw)

    def gpu(k, **kw):
        return lambda r: r["gpus"][k].update(kw)

    def drop(*keys):
        return lambda r: [r.pop(key) for key in keys]
    variants = [
        ("identical", edit(), True), ("submitter case", edit(submitter="TESTer"), True), ("kv", edit(kv="fp16"), True),
        ("numbers and notes", edit(decode_tps=1.5, prefill_16k_tps=2.5, date="2026-11-01", commit="def5678", notes="n"), True),
        ("link figures", gpu(0, pcie="4.0 x8", h2d_gbps=1.5), True), ("ram_gbps", edit(ram_gbps=1.5), True),
        ("cpu missing", drop("cpu", "ram_gbps"), True),
        ("submitter", edit(submitter="other"), False), ("model", edit(model="DeepSeek-V4.1-Flash"), False), ("quant", edit(quant="IQ4_XS"), False),
        ("gpu power", gpu(0, power_w=401), False), ("gpu vram", gpu(1, vram_gib=17), False), ("gpu name", gpu(1, name="RTX 5070"), False),
        ("gpu order", lambda r: r["gpus"].reverse(), False), ("one gpu fewer", lambda r: r["gpus"].pop(), False),
        ("cpu", edit(cpu="Other CPU"), False), ("ram_gib", edit(ram_gib=65), False), ("os", edit(os="WSL2"), False),
    ]
    for name, change, same in variants:
        other = row()
        change(other)
        a, b = os.path.join(d, "a.json"), os.path.join(d, "b.json")
        cb.write_text(a, cb.canonical_row(row()))
        cb.write_text(b, cb.canonical_row(other))
        ran = subprocess.run([exe, "--dup", a, b], capture_output=True, text=True)
        cpp_same = ran.stdout.split() == ["1", "0"]
        ids = subprocess.run([exe, "--ids", a, b], capture_output=True, text=True).stdout.split()
        ok = cpp_same == same == cb.same_setup(row(), other) and ids == [cb.row_id(row()), cb.row_id(other)]
        if ok and same and ids[0] != ids[1]:   # one setup under two ids: the checker names the clash
            code, out = check(fresh(d, row(), other))
            ok = code == 1 and "same submitter, model, quant and hardware as" in out
        elif ok and not same:
            ok = check(fresh(d, row(), other))[0] == 0
        report(f"setup rule: {name}", ok, ran.stdout, ids)

    # a file the checker refuses
    good = row()
    c = fresh(d, good)
    cb.write_text(os.path.join(c, cb.RUNS_DIR, "qwen3.8-flash-next", "0123456789ab.json"), cb.canonical_row(good))   # another id, same content
    code, out = check(c)
    report("a name that is not the content's id", code == 1 and "belongs at runs/qwen3.8-flash-next/" + cb.row_id(good) in out, out)
    c = fresh(d)
    put(c, good, "wrong-folder/" + cb.row_id(good) + ".json")
    code, out = check(c)
    report("a folder that is not the model's slug", code == 1 and "belongs at" in out, out)
    c = fresh(d, good)
    cb.write_text(os.path.join(c, cb.RUNS_DIR, "notes.txt"), "x")
    os.makedirs(os.path.join(c, cb.RUNS_DIR, "qwen3.8-flash-next", "sub"))
    code, out = check(c)
    report("a stray file or folder", code == 1 and "notes.txt" in out and "sub" in out, out)
    c = fresh(d, good)
    p = os.path.join(c, cb.RUNS_DIR, cb.run_path(good))
    cb.write_text(p, json.dumps(good))   # compact JSON: valid but not the house format
    code, out = check(c)
    code_fix, _ = check(c, "--fix")
    report("a file not in the house format is refused and --fix rewrites it", code == 1 and "house format" in out and code_fix == 0 and
           read(p) == cb.canonical_row(good).encode("utf-8") and check(c)[0] == 0, out)
    c = fresh(d, good)
    p = os.path.join(c, cb.RUNS_DIR, cb.run_path(good))
    cb.write_text(p, cb.canonical_row(good).replace("\n", "\r\n"))
    code, out = check(c)
    code_fix, _ = check(c, "--fix")
    report("a CRLF file is refused and --fix turns it into LF", code == 1 and code_fix == 0 and b"\r" not in read(p))
    c = fresh(d, good)
    os.remove(os.path.join(c, cb.PRESET_FILE))
    report("a missing preset.json", check(c)[0] == 1)
    cb.write_text(os.path.join(c, cb.PRESET_FILE), '{"preset": ""}\n')
    report("an empty preset", check(c)[0] == 1)
    c = fresh(d, good)
    cb.write_text(os.path.join(c, cb.RUNS_DIR, "qwen3.8-flash-next", "x.json"), "{")
    cb.write_text(os.path.join(c, cb.RUNS_DIR, "qwen3.8-flash-next", "00000000000a.json"), "{")
    report("a file that is not JSON", "not valid JSON" in check(c)[1])
    c = fresh(d, good)
    p = os.path.join(c, cb.RUNS_DIR, cb.run_path(good))
    cb.write_text(p, "[1]\n")
    report("a file that is not a row", "one row object" in check(c)[1])

    # the model and the KV format are required and checked against the names the engine uses; every model the engine names is accepted
    for key, bad in (("model", "Qwen"), ("kv", "int4"), ("model", None), ("kv", None)):
        r = row()
        if bad is None:
            del r[key]
        else:
            r[key] = bad
        c = fresh(d)
        put(c, r, "qwen3.8-flash-next/" + cb.row_id(r) + ".json")
        code, out = check(c)
        report(f"bad {key}={bad}", code == 1 and key in out, out)
    for model, kv in (("Qwen3.8-Flash-Next", "int8"), ("Qwen3.8-27B", "int8"), ("DeepSeek-V4.1-Flash", "fp8+fp4"),
                      ("Hy4-flash", "fp16"), ("Hy4-flash", "fp8"), ("Inkling-2-Mini", "fp16"), ("Inkling-2-Mini", "fp8")):
        r = row()
        r["model"], r["kv"] = model, kv
        report(f"good {model}/{kv}", check(fresh(d, r))[0] == 0)

    # the per-test fields are optional and type-checked
    for key, value, valid in (("decode_conv_tps", 400.1, True), ("decode_prose_tps", 0, False), ("decode_struct_tps", "400", False),
                              ("accept_conv_pct", 0, True), ("accept_prose_pct", 100.0, True), ("accept_struct_pct", 100.5, False),
                              ("accept_conv_pct", -1, False), ("draft_len_conv", 3.2, True), ("draft_len_prose", 0, False),
                              ("draft_len_struct", True, False), ("engine_args", "--kv int8", True), ("engine_args", "", False),
                              ("engine_args", 5, False), ("engine_args", "x" * 301, False), ("engine_args", "a\nb", False),
                              ("decode_other_tps", 1.5, False)):
        r = row()
        r[key] = value
        code, out = check(fresh(d, r))
        report(f"field {key}={value!r:.20}", (code == 0) if valid else (code == 1 and key in out), out)
sys.exit(failed)
