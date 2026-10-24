#!/usr/bin/env python3
"""community_row_test --apply against tools/check_benchmarks.py --fix, byte for byte.

For each input (the real docs/community/benchmarks.json, an empty runs list) and each row (plain, odd strings) the
expected file is the input with the row appended in plain json and then rewritten by check_benchmarks.py --fix. The row
values here must match plain_row()/odd_row() in community_row_test.cpp. The written file is also checked as it is.
"""
import json
import os
import subprocess
import sys
import tempfile

exe, root = sys.argv[1], sys.argv[2]
checker = os.path.join(root, "tools", "check_benchmarks.py")
real = os.path.join(root, "docs", "community", "benchmarks.json")


def row(odd):
    r = {"submitter": "tester", "date": "2026-10-24", "commit": "abc1234", "preset": "community-1",
         "gpus": [{"name": "RTX 5090", "vram_gib": 32, "power_w": 400, "pcie": "5.0 x16", "h2d_gbps": 55.3},
                  {"name": "RTX 5060 Ti", "vram_gib": 16, "power_w": 180, "pcie": "5.0 x8", "h2d_gbps": 27.9}],
         "ram_gib": 64, "cpu": "Test CPU 9, 16 cores", "ram_gbps": 78.2, "os": "Linux", "quant": "NVFP4", "prefill_16k_tps": 1234.5, "prefill_64k_tps": 2345.6, "decode_tps": 345.7}
    if odd:
        r["submitter"] = "Ünï \"q\" \\ ☃ 日本"
        r["gpus"] = [{"name": "Card \"X\"\tTi", "vram_gib": 24, "power_w": 300}]
        del r["cpu"], r["ram_gbps"]
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
sys.exit(failed)
