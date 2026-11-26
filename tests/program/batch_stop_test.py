#!/usr/bin/env python3
"""A batch slot stopped by its client holds what the client saw (`--serve --batch 2 --spec 1 --prompt-cache 1`).

    batch_stop_test.py ENGINE MODEL.basalt [ENGINE ARGS...]

A client that stops a reply at a token (a thinking budget) continues the conversation with the tokens it kept and a
close of its own. The engine answers a BSTOP before the next window feeds a token, like the solo loop does, so the slot
holds at most the one window that was already running and the continuation reads only its tail. A slot that ran one more
window held a token the client never saw: the continuation did not match and read the whole reply again through the
prompt path, whose cells differ from the decode windows' in the last bits.

For several stop points the test reads J tokens of a reply in slot 0, sends BSTOP, then sends the conversation as the
client keeps it (the prompt, the J tokens, a token of its own) and checks that all but the tail is reused.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STOPS = (6, 17, 40)


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

    def send(self, line: str) -> None:
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def read(self) -> str:
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("engine exited unexpectedly")
        line = line.rstrip("\n")
        if line.startswith("ERR"):
            raise RuntimeError(line)
        return line

    def stop(self) -> None:
        try:
            self.send("QUIT")
            self.p.wait(timeout=60)
        except Exception:
            self.p.kill()
        finally:
            self.log.close()


def admit(eng: Engine, ids: list[int], max_new: int) -> tuple[list[int], int]:
    """BGEN into slot 0: the tokens the admission wrote and the tokens it reused; the slot decodes on."""
    eng.send(f"BGEN 0 {max_new} {','.join(map(str, ids))}")
    toks: list[int] = []
    reused = -1
    while True:
        f = eng.read().split()
        if f[0] == "T":
            toks.append(int(f[1]))
        elif f[0] == "REUSED":
            reused = int(f[1])
        elif f[0] == "BADM":
            if f[2] == "0":
                raise RuntimeError("the slot did not go on decoding")
            return toks, reused


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine")
    ap.add_argument("model")
    ap.add_argument("--fixtures", default="")
    args, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    fixtures = Path(args.fixtures) if args.fixtures else default_fixtures(args.model)
    prompt = [int(t) for t in (fixtures / "conv.ids").read_text().splitlines()[0].split(",") if t.strip()][:200]

    failures = 0
    with tempfile.TemporaryDirectory(prefix="basalt_batch_stop_") as tmp:
        eng = Engine([args.engine, "--model", args.model, "--serve", "--batch", "2", "--spec", "1", "--prompt-cache", "1",
                      "--max-context", "4096", *extra], Path(tmp) / "engine.log")
        try:
            for j in STOPS:
                toks, _ = admit(eng, prompt, 512)
                while len(toks) < j:
                    f = eng.read().split()
                    if f[0] == "BT":
                        toks.extend(int(x) for x in f[2:])
                eng.send("BSTOP 0")
                kept = toks[:j]
                while eng.read().split()[0] != "BDONE":
                    pass
                conversation = prompt + kept + [prompt[-1]]
                _, reused = admit(eng, conversation, 2)
                eng.send("BSTOP 0")
                while eng.read().split()[0] != "BDONE":
                    pass
                want = len(prompt) + j - 1
                ok = reused >= want
                print(f"stop after {j} tokens: continuation reused {reused} of {len(conversation) - 1} (at least {want}): "
                      f"{'ok' if ok else 'FAIL'}")
                failures += not ok
        finally:
            eng.stop()
    if failures:
        print(f"batch_stop_test: {failures} stop points re-read the reply")
        return 1
    print("batch_stop_test: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
