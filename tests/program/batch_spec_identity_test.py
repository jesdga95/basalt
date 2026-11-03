#!/usr/bin/env python3
"""Speculative multi-slot serving (`--serve --batch 2/4/8 --spec 8`) matches `--batch 1` token for token.

    batch_spec_identity_test.py ENGINE MODEL.basalt [ENGINE ARGS...]

Runs the model's benchmark token fixtures (`conv`, `struct`, `prose`) under greedy decoding (`--spec 8 --prompt-cache 0`)
first in a single-slot engine (`--serve --batch 1`), then in `--serve --batch B` for B in (2, 4, 8):
  1. Each slot's prompt run alone in slot 0 must match the `--batch 1` reference.
  2. All B slots run concurrently (cold / graph-capture pass) must match the `--batch 1` reference.
  3. All B slots run concurrently a second time in the same engine (warm / graph-replay pass) must match the reference.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ("conv", "struct", "prose")


def default_fixtures(model: str) -> Path:
    base = ROOT / "data" / "bench"
    low = Path(model).name.lower()
    if "inkling" in low and (base / "inkling2" / "conv.ids").is_file():
        return base / "inkling2"
    return base


def fixture_ids(fixtures: Path, name: str) -> str:
    line = (fixtures / f"{name}.ids").read_text().splitlines()[0].strip()
    return ",".join(t.strip() for t in line.split(",") if t.strip())


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
        assert self.p.stdin is not None
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def read(self) -> str:
        assert self.p.stdout is not None
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("engine exited unexpectedly")
        return line.rstrip("\n")

    def stop(self) -> None:
        try:
            self.send("QUIT")
            self.p.wait(timeout=60)
        except Exception:
            self.p.kill()
        finally:
            self.log.close()


def run_slots(eng: Engine, jobs: list[tuple[int, str]], max_new: int) -> dict[int, list[int]]:
    toks: dict[int, list[int]] = {b: [] for b, _ in jobs}
    live: set[int] = set()

    def handle(line: str, admitting: int) -> int | None:
        f = line.split()
        if not f:
            return None
        if f[0] == "T" and admitting >= 0:
            toks[admitting].append(int(f[1]))
        elif f[0] == "BT":
            toks[int(f[1])].extend(int(x) for x in f[2:])
        elif f[0] == "BDONE":
            live.discard(int(f[1]))
        elif f[0] == "BADM":
            b = int(f[1])
            if f[2] != "0":
                live.add(b)
            return b
        elif f[0] == "ERR":
            raise RuntimeError(line)
        return None

    for b, ids in jobs:
        eng.send(f"BGEN {b} {max_new} {ids}")
        while handle(eng.read(), b) != b:
            pass
    while live:
        handle(eng.read(), -1)
    return toks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine")
    ap.add_argument("model")
    ap.add_argument("--slots", default="2,4,8")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--fixtures", default="")
    args, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    if "--spec" not in extra:
        extra = ["--spec", "8", *extra]

    fixtures = Path(args.fixtures) if args.fixtures else default_fixtures(args.model)
    slot_counts = [int(x) for x in args.slots.split(",") if x.strip()]

    with tempfile.TemporaryDirectory(prefix="basalt_batch_spec_") as tmp:
        tmp_dir = Path(tmp)
        ref_eng = Engine(
            [args.engine, "--model", args.model, "--serve", "--batch", "1", "--prompt-cache", "0", *extra],
            tmp_dir / "ref.log",
        )
        try:
            ref = {name: run_slots(ref_eng, [(0, fixture_ids(fixtures, name))], args.tokens)[0] for name in PROMPTS}
        finally:
            ref_eng.stop()

        failures = 0
        for n in slot_counts:
            ids = [fixture_ids(fixtures, PROMPTS[j % len(PROMPTS)]) for j in range(n)]
            eng = Engine(
                [args.engine, "--model", args.model, "--serve", "--batch", str(n), "--prompt-cache", "0", *extra],
                tmp_dir / f"batch_{n}.log",
            )
            try:
                alone = [run_slots(eng, [(0, ids[j])], args.tokens)[0] for j in range(n)]
                cold = run_slots(eng, list(enumerate(ids)), args.tokens)
                warm = run_slots(eng, list(enumerate(ids)), args.tokens)
            finally:
                eng.stop()

            for b in range(n):
                want = ref[PROMPTS[b % len(PROMPTS)]]
                ok_alone = alone[b] == want
                ok_cold = cold[b] == want
                ok_warm = warm[b] == want
                if not (ok_alone and ok_cold and ok_warm):
                    failures += 1
                    print(
                        f"FAIL --batch {n} slot {b} ({PROMPTS[b % len(PROMPTS)]}): "
                        f"alone={'ok' if ok_alone else 'DIFF'} "
                        f"cold={'ok' if ok_cold else 'DIFF'} "
                        f"warm={'ok' if ok_warm else 'DIFF'}"
                    )
            if failures == 0:
                print(f"--batch {n} ({args.tokens} tok/slot, alone + cold + warm): IDENTICAL")

    if failures:
        print(f"batch_spec_identity_test: {failures} slot failures")
        return 1
    print("batch_spec_identity_test: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
