#!/usr/bin/env python3
"""A pass that continues a reply takes no turn checkpoint inside it (`--serve --prompt-cache 1`).

    reply_checkpoint_test.py ENGINE MODEL.basalt [ENGINE ARGS...]

With a chain of one checkpoint, the turn boundary of a prompt is what the chat's next turn mounts through (the reply
it wrote, thinking included, is not sent back). The pass that follows a thinking budget extends the prompt by the
reply so far: a checkpoint at its last token would replace the turn boundary and the next turn would read everything
again. `reply=N` names where the reply begins. The test runs a prompt, a continuation of it with and without the key,
then a next turn that diverges inside the reply, and checks what each leaves to reuse.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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

    def run(self, keys: str, ids: list[int], max_new: int = 4) -> int:
        """One GEN to its DONE: the tokens it reused."""
        self.p.stdin.write(f"GEN {max_new}{keys} {','.join(map(str, ids))}\n")
        self.p.stdin.flush()
        reused = -1
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine exited unexpectedly")
            f = line.split()
            if f and f[0] == "ERR":
                raise RuntimeError(line)
            if f and f[0] == "REUSED":
                reused = int(f[1])
            if f and f[0] == "DONE":
                return reused

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
    ids = [int(t) for t in (fixtures / "conv.ids").read_text().splitlines()[0].split(",") if t.strip()]
    third = len(ids) // 3
    prompt, reply, other = ids[:third], ids[third:2 * third], ids[2 * third:3 * third]
    n = len(prompt)

    results = {}
    with tempfile.TemporaryDirectory(prefix="basalt_reply_ckpt_") as tmp:
        for name, keys in (("with reply=N", f" reply={n}"), ("without", "")):
            eng = Engine([args.engine, "--model", args.model, "--serve", "--prompt-cache", "1", "--max-context", "4096", *extra],
                         Path(tmp) / "engine.log")
            try:
                eng.run("", prompt)                               # the turn boundary becomes the chain's checkpoint
                eng.run(keys, prompt + reply)                     # the pass that continues the reply
                results[name] = eng.run("", prompt + other)       # the next turn: the reply is not sent back
            finally:
                eng.stop()
    ok = results["with reply=N"] >= n - 1 and results["without"] < n - 1
    print(f"next turn reused {results['with reply=N']} tokens with reply=N and {results['without']} without "
          f"(the prompt is {n}): {'ok' if ok else 'FAIL'}")
    if not ok:
        return 1
    print("reply_checkpoint_test: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
