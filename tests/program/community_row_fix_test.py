#!/usr/bin/env python3
"""community_row_test --apply against tools/check_benchmarks.py --fix, byte for byte.

For each input (the real docs/community/benchmarks.json, an empty runs list) and each row (plain, odd strings) the
expected file is the input with the row appended in plain json and then rewritten by check_benchmarks.py --fix. The row
values here must match plain_row()/odd_row() in community_row_test.cpp. The written file is also checked as it is.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

exe, root = sys.argv[1], sys.argv[2]
checker = os.path.join(root, "tools", "check_benchmarks.py")
real = os.path.join(root, "docs", "community", "benchmarks.json")


def row(odd):
    r = {"submitter": "tester", "date": "2026-10-24", "commit": "abc1234", "preset": "community-1", "model": "Qwen3.8-Flash-Next",
         "gpus": [{"name": "RTX 5090", "vram_gib": 32, "power_w": 400, "pcie": "5.0 x16", "h2d_gbps": 55.3},
                  {"name": "RTX 5060 Ti", "vram_gib": 16, "power_w": 180, "pcie": "5.0 x8", "h2d_gbps": 27.9}],
         "ram_gib": 64, "cpu": "Test CPU 9, 16 cores", "ram_gbps": 78.2, "os": "Linux", "quant": "NVFP4", "kv": "int8", "prefill_16k_tps": 1234.5, "prefill_64k_tps": 2345.6, "decode_tps": 345.7,
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


def write(p, text):
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def load(p):
    return json.loads(read(p).decode("utf-8"))


failed = 0
with tempfile.TemporaryDirectory() as d:
    empty = os.path.join(d, "empty.json")
    write(empty, json.dumps({"preset": "basalt-bench --community", "runs": []}, indent=2))
    # the real file with CRLF line endings, as a checkout that translates them gives
    crlf = os.path.join(d, "real_crlf.json")
    with open(crlf, "wb") as f:
        f.write(read(real).replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    for name, src, eol in (("real", real, b"\n"), ("empty", empty, b"\n"), ("real-crlf", crlf, b"\r\n")):
        for case in ("plain", "odd"):
            data = load(src)
            data["runs"].append(row(case == "odd"))
            want = os.path.join(d, f"want_{name}_{case}.json")
            write(want, json.dumps(data, ensure_ascii=False))
            fix = subprocess.run([sys.executable, checker, "--fix", want], capture_output=True, text=True)
            got = os.path.join(d, f"got_{name}_{case}.json")
            ran = subprocess.run([exe, "--apply", src, got, case], capture_output=True, text=True)
            want_bytes = read(want).replace(b"\n", eol)   # --fix writes LF; a CRLF input stays CRLF
            same = ran.returncode == 0 and fix.returncode == 0 and read(got) == want_bytes
            # the checker takes the LF result as it is, the CRLF one once converted back
            lf = os.path.join(d, f"lf_{name}_{case}.json")
            with open(lf, "wb") as f:
                f.write(read(got).replace(b"\r\n", b"\n"))
            check = subprocess.run([sys.executable, checker, lf], capture_output=True, text=True)
            ok = same and check.returncode == 0
            print(f"community_row_fix {name}/{case}: " + ("ok" if ok else "FAILED"))
            if not ok:
                failed = 1
                print(ran.stdout, ran.stderr, fix.stdout, check.stdout)
    # a CRLF file is not canonical for the checker, and --fix turns it into LF without failing
    bad = subprocess.run([sys.executable, checker, crlf], capture_output=True, text=True)
    fixed = subprocess.run([sys.executable, checker, "--fix", crlf], capture_output=True, text=True)
    ok = bad.returncode == 1 and fixed.returncode == 0 and b"\r" not in read(crlf)
    print("community_row_fix crlf checker: " + ("ok" if ok else "FAILED"))
    failed |= not ok
# the model and the KV format are required and checked against the names the engine uses
for key, bad in (("model", "Qwen"), ("kv", "int4"), ("model", None), ("kv", None)):
    data = load(real)
    if bad is None:
        del data["runs"][0][key]
    else:
        data["runs"][0][key] = bad
    p = os.path.join(tempfile.gettempdir(), "cb_bad_row.json")
    write(p, json.dumps(data, indent=2, ensure_ascii=False))
    r = subprocess.run([sys.executable, checker, p], capture_output=True, text=True)
    os.remove(p)
    ok = r.returncode == 1 and key in r.stdout
    print(f"community_row_fix bad {key}={bad}: " + ("ok" if ok else "FAILED"))
    failed |= not ok


# the per-test fields are optional and type-checked; rows without them (every old row) are valid
def check_with(key, value):
    data = load(real)
    if value is None:
        data["runs"][0].pop(key, None)
    else:
        data["runs"][0][key] = value
    p = os.path.join(tempfile.gettempdir(), "cb_field_row.json")
    write(p, json.dumps(data, indent=2, ensure_ascii=False))
    r = subprocess.run([sys.executable, checker, "--fix", p], capture_output=True, text=True)   # --fix puts the key in order first
    os.remove(p)
    return r


for key, value, valid in (("decode_conv_tps", 400.1, True), ("decode_prose_tps", 0, False), ("decode_struct_tps", "400", False),
                          ("accept_conv_pct", 0, True), ("accept_prose_pct", 100.0, True), ("accept_struct_pct", 100.5, False),
                          ("accept_conv_pct", -1, False), ("draft_len_conv", 3.2, True), ("draft_len_prose", 0, False),
                          ("draft_len_struct", True, False), ("engine_args", "--kv int8", True), ("engine_args", "", False),
                          ("engine_args", 5, False), ("engine_args", "x" * 301, False), ("engine_args", "a\nb", False),
                          ("decode_other_tps", 1.5, False)):
    r = check_with(key, value)
    ok = (r.returncode == 0) if valid else (r.returncode == 1 and key in r.stdout)
    print(f"community_row_fix field {key}={value!r:.20}: " + ("ok" if ok else "FAILED"))
    failed |= not ok

# Two rows of one setup are refused. Each variant is checked against a base row alone: (name, change, same setup?)
def variants():
    def edit(**kw):
        def f(r):
            r.update(kw)
        return f

    def gpu(k, **kw):
        def f(r):
            r["gpus"][k].update(kw)
        return f

    def drop(*keys):
        def f(r):
            for key in keys:
                del r[key]
        return f

    def swap(r):
        r["gpus"].reverse()

    def one(r):
        del r["gpus"][1]
    return [
        ("identical", edit(), True), ("submitter case", edit(submitter="TESTer"), True), ("kv", edit(kv="fp16"), True),
        ("numbers and notes", edit(decode_tps=1.5, prefill_16k_tps=2.5, date="2026-11-01", commit="def5678", notes="n"), True),
        ("link figures", gpu(0, pcie="4.0 x8", h2d_gbps=1.5), True), ("ram_gbps", edit(ram_gbps=1.5), True),
        ("cpu missing", drop("cpu", "ram_gbps"), True),
        ("submitter", edit(submitter="other"), False), ("model", edit(model="DeepSeek-V4.1-Flash"), False), ("quant", edit(quant="IQ4_XS"), False),
        ("gpu power", gpu(0, power_w=401), False), ("gpu vram", gpu(1, vram_gib=17), False), ("gpu name", gpu(1, name="RTX 5070"), False),
        ("gpu order", swap, False), ("one gpu fewer", one, False), ("cpu", edit(cpu="Other CPU"), False), ("ram_gib", edit(ram_gib=65), False),
        ("os", edit(os="WSL2"), False),
    ]


def dup_pairs(path):
    out = subprocess.run([sys.executable, checker, path], capture_output=True, text=True).stdout
    pat = re.compile(r"runs\[(\d+)\] \([^)]*\): same submitter, model, quant and hardware as runs\[(\d+)\]: update your run instead")
    return [(int(a), int(b)) for a, b in pat.findall(out)], out


def cpp_pairs(path):
    """The same pairs from community_row_test --dup, on the file in the canonical form."""
    with tempfile.TemporaryDirectory() as t:
        copy = os.path.join(t, "canonical.json")
        write(copy, read(path).decode("utf-8"))
        subprocess.run([sys.executable, checker, "--fix", copy], capture_output=True, text=True)
        ran = subprocess.run([exe, "--dup", copy], capture_output=True, text=True)
    return [tuple(map(int, line.split())) for line in ran.stdout.split("\n") if line.strip()] if ran.returncode == 0 else None


def dup_fixture(change):
    r = row(False)
    change(r)
    return {"preset": "basalt-bench --community", "runs": [row(False), r]}


with tempfile.TemporaryDirectory() as d:
    for name, change, same in variants():
        p = os.path.join(d, "dup.json")
        write(p, json.dumps(dup_fixture(change), indent=2, ensure_ascii=False))
        pairs, out = dup_pairs(p)
        ok = pairs == ([(1, 0)] if same else []) and (same or "ok, 2 runs" in out) and cpp_pairs(p) == pairs
        print(f"community_row_fix dup {name}: " + ("ok" if ok else "FAILED"))
        failed |= not ok
    # the message names the earlier run and says what to do
    p = os.path.join(d, "dup.json")
    write(p, json.dumps(dup_fixture(lambda r: None), indent=2, ensure_ascii=False))
    ok = "runs[1] (tester): same submitter, model, quant and hardware as runs[0]: update your run instead" in dup_pairs(p)[1]
    print("community_row_fix dup message: " + ("ok" if ok else "FAILED"))
    failed |= not ok
    # the real file has no duplicate; adding the same setup again to it is one
    data = load(real)
    ok = dup_pairs(real)[0] == [] and cpp_pairs(real) == []
    data["runs"].append(json.loads(json.dumps(data["runs"][4])))
    write(p, json.dumps(data, indent=2, ensure_ascii=False))
    ok = ok and dup_pairs(p)[0] == [(len(data["runs"]) - 1, 4)] and cpp_pairs(p) == [(len(data["runs"]) - 1, 4)]
    print("community_row_fix dup real file: " + ("ok" if ok else "FAILED"))
    failed |= not ok
sys.exit(failed)
