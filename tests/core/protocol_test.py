"""Runs protocol_test and checks every stdout line is a whole BT or PL line (CPU only)."""
import re
import subprocess
import sys

out = subprocess.run([sys.argv[1]], capture_output=True, text=True, check=True).stdout.split("\n")
assert out.pop() == "", "output does not end with a newline"
bt = re.compile(r"^BT [02]( \d+){8}$")
pl = re.compile(r"^PL \d+ 2000 12 3\.5$")
bad = [l for l in out if not (bt.match(l) or pl.match(l))]
assert len(out) == 4 * 2000, f"{len(out)} lines"
assert not bad, f"{len(bad)} broken lines, first: {bad[0]!r}"
print(f"protocol {len(out)} lines whole: ok")
