#!/usr/bin/env python3
"""A reply read again keeps the bits its decode wrote (`--serve --prompt-cache 0`).

    reply_read_test.py ENGINE MODEL.basalt [ENGINE ARGS...]

The server moves a request between engine runs (solo to a batch slot when another request arrives, a thinking budget's
wrap-up) and sends the next run the prompt, the reply so far and `reply=N`. With no prompt cache that run reads all of it
again. A model whose prompt path and decode path differ in the last bits must read the last prompt token and the reply
through decode windows, or the continuation diverges from the run that was never moved: the greedy reply is then not the
one the request gives alone.

For several split points the test generates a greedy reply, then sends the prompt, the first J tokens of that reply and
`reply=N`, and checks that the continuation is the rest of the reply.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS = (1, 9, 40, 120)
TOKENS = 200


def default_fixtures(model: str) -> Path:
    base = ROOT / "data" / "bench"
    low = Path(model).name.lower()
    if "inkling" in low and (base / "inkling2" / "conv.ids").is_file():
        return base / "inkling2"
    return base


class Engine:
    def __init__(self, cmd: list[str], log_path: Path):
        self.log = open(log_path, "w")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"engine exited before READY (see {log_path})")
            if line.startswith("READY"):
                break

    def run(self, keys: str, ids: list[int], max_new: int) -> list[int]:
        """One greedy GEN to its DONE: the tokens it sent."""
        self.p.stdin.write(f"GEN {max_new} temperature=0{keys} {','.join(map(str, ids))}\n")
        self.p.stdin.flush()
        toks: list[int] = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine exited unexpectedly")
            f = line.split()
            if f and f[0] == "ERR":
                raise RuntimeError(line)
            if f and f[0] == "T":
                toks.append(int(f[1]))
            if f and f[0] == "DONE":
                return toks

    def stop(self) -> None:
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=60)
        except Exception:
            self.p.kill()
        finally:
            self.log.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine")
    ap.add_argument("model")
    ap.add_argument("--fixtures", default="")
    args, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    fixtures = Path(args.fixtures) if args.fixtures else default_fixtures(args.model)
    prompt = [int(t) for t in (fixtures / "conv.ids").read_text().splitlines()[0].split(",") if t.strip()]
    n = len(prompt)

    failures = 0
    with tempfile.TemporaryDirectory(prefix="basalt_reply_read_") as tmp:
        eng = Engine([args.engine, "--model", args.model, "--serve", "--prompt-cache", "0", "--max-context", "4096", *extra],
                     Path(tmp) / "engine.log")
        try:
            reply = eng.run("", prompt, TOKENS)
            for j in SPLITS:
                rest = eng.run(f" reply={n}", prompt + reply[:j], TOKENS - j)
                same = rest == reply[j:j + len(rest)] and len(rest) == len(reply) - j
                print(f"split at {j}: {'ok' if same else 'FAIL'}")
                failures += not same
        finally:
            eng.stop()
    if failures:
        return 1
    print("reply_read_test: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
