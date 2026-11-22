"""Hy4-flash: a prompt's last-row logits do not depend on how the prompt is cut. One chunk of 4,500 rows (three attention
steps, the sparse layers selecting past 2,048 positions) against chunks of 1,024 rows (one step each); the two may round
differently (KL well under 0.01), a row that reads another step's spans or cells does not (KL about 0.5)."""
import argparse
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROWS = 4500
KL_MAX = 0.01


def tokens(n: int) -> list[int]:
    """Reproducible ids in the common part of the vocabulary (a linear congruential walk)."""
    x, out = 12345, []
    for _ in range(n):
        x = (1103515245 * x + 12345) % (1 << 31)
        out.append(1000 + x % 60000)
    return out


def last_row(path: Path, vocab: int) -> list[float]:
    data = path.read_bytes()
    if len(data) < vocab * 4:
        raise AssertionError(f"{path}: {len(data)} bytes, not a logits row")
    return list(struct.unpack(f"<{vocab}f", data[-vocab * 4:]))


def log_softmax(x: list[float]) -> list[float]:
    m = max(x)
    s = math.log(sum(math.exp(v - m) for v in x))
    return [v - m - s for v in x]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vocab", type=int, default=120832)
    args = parser.parse_args()
    if not Path(args.model).is_file():
        print("SKIP: model pack unavailable")
        return 77
    with tempfile.TemporaryDirectory(prefix="basalt-hy4-steps-") as temp:
        ids = Path(temp) / "prompt.ids"
        ids.write_text(",".join(map(str, tokens(ROWS))) + "\n")
        rows = {}
        for name, extra in (("one chunk", []), ("1,024-row chunks", ["--diag-rows", "1024"])):
            out = Path(temp) / (name.split()[0] + ".f32")
            cmd = [str(Path(args.engine).resolve()), "--model", str(Path(args.model).resolve()), "--tokens-file", str(ids),
                   "--max-new", "1", "--temperature", "0", "--max-context", "5120", "--spec", "0",
                   "--diag-logits", str(out), *extra]
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if run.returncode != 0:
                print(run.stderr[-3000:], file=sys.stderr)
                raise AssertionError(f"{name}: the engine exited with {run.returncode}")
            rows[name] = last_row(out, args.vocab)
    a, b = rows["1,024-row chunks"], rows["one chunk"]
    la, lb = log_softmax(a), log_softmax(b)
    kl = sum(math.exp(p) * (p - q) for p, q in zip(la, lb))
    top_a, top_b = a.index(max(a)), b.index(max(b))
    print(f"KL(1,024-row chunks || one chunk) {kl:.6f}, top-1 {top_a} / {top_b}")
    if kl > KL_MAX or top_a != top_b:
        print(f"FAIL: the prompt's cut changes its logits (KL limit {KL_MAX})")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
