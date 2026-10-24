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
         "gpus": [{"name": "RTX 5090", "vram_gib": 32, "power_w": 400}, {"name": "RTX 5060 Ti", "vram_gib": 16, "power_w": 180}],
         "ram_gib": 64, "os": "Linux", "quant": "NVFP4", "prefill_16k_tps": 1234.5, "prefill_64k_tps": 2345.6, "decode_tps": 345.7}
    if odd:
        r["submitter"] = "Ünï \"q\" \\ ☃ 日本"
        r["gpus"] = [{"name": "Card \"X\"\tTi", "vram_gib": 24, "power_w": 300}]
        r["notes"] = "line\nbreak € \x01\x7f"
    return r


def read(p):
    with open(p, "rb") as f:
        return f.read()


failed = 0
with tempfile.TemporaryDirectory() as d:
    empty = os.path.join(d, "empty.json")
    with open(empty, "w", encoding="utf-8") as f:
        json.dump({"preset": "basalt-bench --community", "runs": []}, f, indent=2)
    for name, src in (("real", real), ("empty", empty)):
        for case in ("plain", "odd"):
            with open(src, encoding="utf-8") as f:
                data = json.load(f)
            data["runs"].append(row(case == "odd"))
            want = os.path.join(d, f"want_{name}_{case}.json")
            with open(want, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            fix = subprocess.run([sys.executable, checker, "--fix", want], capture_output=True, text=True)
            got = os.path.join(d, f"got_{name}_{case}.json")
            ran = subprocess.run([exe, "--apply", src, got, case], capture_output=True, text=True)
            same = ran.returncode == 0 and fix.returncode == 0 and read(got) == read(want)
            check = subprocess.run([sys.executable, checker, got], capture_output=True, text=True)
            ok = same and check.returncode == 0
            print(f"community_row_fix {name}/{case}: " + ("ok" if ok else "FAILED"))
            if not ok:
                failed = 1
                print(ran.stdout, ran.stderr, fix.stdout, check.stdout)
sys.exit(failed)
