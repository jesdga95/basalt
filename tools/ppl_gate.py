#!/usr/bin/env python3
"""Perplexity and KL divergence through the decode path, against a llama.cpp base.

The base is `llama-perplexity --kl-divergence-base`'s file (wikitext-2, 512-token chunks, second half scored). The
engine runs as `--serve` with the model plus --ppl-base and --ppl-out; per chunk it reads the first half as a prompt
and teacher-forces the rest through verify windows of T tokens (see the engine's ppl_gate.hpp). It sets --spec 5 and,
when there are two visible GPUs, --gpu 0,1 --gpu-mode peer (--gpu-mode goes only with two or more cards, so a --gpu
that names one card drops it). `--architecture qwen4exp` also disables adaptive swaps so placement stays
fixed. Flags passed after the tool's options replace these defaults.

  python3 tools/ppl_gate.py --model FILE.basalt --architecture qwen4exp|inkling2|other \
                             --base base.kld --out DIR [--engine-bin BIN] [--chunks 20] [--t 3]
                             [--gpu 0 ...]
  python3 tools/ppl_gate.py --report DIR/rows.tsv          # summary of a finished run
  python3 tools/ppl_gate.py --base-stats base.kld          # what the base says alone: llama.cpp's perplexity
  python3 tools/ppl_gate.py --diff A/rows.tsv B/rows.tsv   # row by row: identical rows, largest NLL/KLD change
  python3 tools/ppl_gate.py --model F --base B --out D --architecture other --chunks 1 --repeat 2
      # one chunk scored twice in the same process: the spread between the passes answers in a minute whether a
      #   reused session restarts its recurrent state for every request (a nat says it does not)
"""
from __future__ import annotations

import argparse
import math
import struct
import subprocess
import sys
import time
from pathlib import Path

ENGINE_BIN = Path(__file__).resolve().parents[1] / "bin" / "basalt"
SHUTDOWN_TIMEOUT_S = 60


def _max_or_nan(vals) -> float:
    best = -math.inf
    for v in vals:
        if math.isnan(v):
            return math.nan
        if v > best:
            best = v
    return best


def _safe_exp(v: float) -> float:
    try:
        return math.exp(v)
    except OverflowError:
        return math.inf


def read_base(path: str):
    with open(path, "rb") as f:
        if f.read(8) != b"_logits_":
            sys.exit(f"{path}: not a llama.cpp KL-divergence base")
        n_ctx, n_vocab, n_chunk = struct.unpack("<Iii", f.read(12))
        toks = struct.unpack(f"<{n_ctx * n_chunk}i", f.read(4 * n_ctx * n_chunk))
    return n_ctx, n_vocab, n_chunk, [toks[k * n_ctx:(k + 1) * n_ctx] for k in range(n_chunk)]


def summary(rows_path: str, extra: dict | None = None) -> str:
    nll, nllb, kld, same, dlp = [], [], [], 0, []
    nonfinite = 0
    by: dict[int, list] = {}
    with open(rows_path) as fh:
        for line in fh:
            f = line.split("\t")
            if len(f) < 10:
                continue
            v_nll, v_nllb, v_kld, v_dlp = float(f[3]), float(f[4]), float(f[5]), float(f[9])
            if not (math.isfinite(v_nll) and math.isfinite(v_nllb) and math.isfinite(v_kld) and math.isfinite(v_dlp)):
                nonfinite += 1
            by.setdefault(int(f[0]), []).append((v_nll, v_nllb, v_kld, int(f[6]), int(f[1])))
            nll.append(v_nll); nllb.append(v_nllb); kld.append(v_kld)
            same += int(f[6]); dlp.append(v_dlp)
    n = len(nll)
    if n == 0:
        return "no rows"
    has_kld_nan = any(math.isnan(v) for v in kld)
    if has_kld_nan:
        kld_med = kld_p99 = kld_max = math.nan
    else:
        ks = sorted(kld)
        pct = lambda q: ks[min(n - 1, int(round(q * (n - 1))))]
        kld_med, kld_p99, kld_max = pct(0.5), pct(0.99), ks[-1]
    mn = sum(nll) / n
    var = sum(v * v for v in nll) / n - mn * mn
    se = math.nan if math.isnan(var) else math.sqrt(max(0.0, var) / max(1, n - 1))
    ppl = _safe_exp(mn)
    base_mean = sum(nllb) / n
    out = [f"chunks {len(by)}  rows {n}",
           f"PPL {ppl:.4f} +/- {ppl * se:.4f}   base PPL {_safe_exp(base_mean):.4f}   "
           f"ln(PPL/base) {mn - base_mean:.5f}",
           f"KLD mean {sum(kld) / n:.5f}  median {kld_med:.5f}  p99 {kld_p99:.4f}  max {kld_max:.4f}",
           f"same top-1 {100.0 * same / n:.2f}%   mean |dlogp| of the base top-1 {sum(dlp) / n:.5f}"]
    if nonfinite:
        out.append(f"WARNING: {nonfinite} row(s) contain non-finite values (NaN/Inf)")
    if len(by) > 1:
        # The shape of the gap across chunks is the diagnosis: a ramp over the chunks says state carried from one
        # request into the next, a flat row of ratios says the decode path tracks the base.
        out.append("per chunk (nll, base nll, PPL ratio, KL, same top-1):")
        for c in sorted(by):
            v, m = by[c], len(by[c])
            a = sum(x[0] for x in v) / m
            b = sum(x[1] for x in v) / m
            out.append(f"  {c:3d}  {a:6.3f}  {b:6.3f}  x{_safe_exp(a - b):5.2f}   "
                       f"{sum(x[2] for x in v) / m:6.3f}   {100.0 * sum(x[3] for x in v) / m:5.1f}%")
    prof: dict[int, list[float]] = {}
    for v in by.values():
        for x in v:
            prof.setdefault(x[4] // 32, []).append(x[0] - x[1])
    if len(prof) > 1:
        # The excess over the base by position inside the window: carried state is strongest at the first scored
        # row and decays as the window's own tokens overwrite it, while a wrong scale or quantisation is position
        # independent. A flat column near zero is what fidelity looks like.
        out.append("mean excess nll over the base, by position in the window:")
        for k in sorted(prof):
            v = prof[k]
            out.append(f"  {k * 32:4d}-{k * 32 + 31:4d}  {sum(v) / len(v):+.3f}  ({len(v)} rows)")
    if extra:
        out.append("  ".join(f"{k} {v}" for k, v in extra.items()))
    return "\n".join(out)


def base_stats(path: str) -> str:
    """What the base file says on its own: llama.cpp's perplexity over the scored rows, without the engine.

    The rows are the base file's own (positions n_ctx/2 .. n_ctx-2, log-probs as uint16 over [min, max]), and
    `nll_base` is the same arithmetic the engine applies to them (`row()` in basalt/program/ppl_gate.hpp), so
    this is the target the engine's `nll` has to come near - and if the engine's own `nll_base` column
    disagrees, one of the two reads the file differently. Stdlib only: it seeks to the head and the target
    code instead of touching all 154880 columns of every row.
    """
    n_ctx, n_vocab, n_chunk, toks = read_base(path)
    first, rows = n_ctx // 2, n_ctx - 1 - n_ctx // 2
    row_bytes = (4 + n_vocab + n_vocab % 2) * 2
    body = 20 + 4 * n_ctx * n_chunk
    nll, top_lp = [], []
    with open(path, "rb") as f:
        for c in range(n_chunk):
            for r in range(rows):
                f.seek(body + (c * rows + r) * row_bytes)
                scale, min_lp = struct.unpack("<ff", f.read(8))
                tgt = toks[c][first + r + 1]
                f.seek(2 * tgt, 1)   # the head is 8 bytes, so code[tgt] starts 2*tgt past it
                code = struct.unpack("<H", f.read(2))[0]
                nll.append(-(scale * code + min_lp))
                top_lp.append(min_lp + 65535 * scale)
    if not nll:
        return "no rows"
    n = len(nll)
    mean = sum(nll) / n
    se = math.sqrt(max(0.0, sum(v * v for v in nll) / n - mean * mean) / max(1, n - 1))
    return (f"base: {n_chunk} chunks, {rows} scored rows each, {n} rows read; vocab {n_vocab}\n"
            f"base PPL {math.exp(mean):.4f} +/- {math.exp(mean) * se:.4f} (mean nll {mean:.5f})   "
            f"the row top log-prob the format claims: {min(top_lp):.4f} .. {max(top_lp):.4f} (should sit at 0)")


def carryover(rows_path: str, repeat: int) -> str:
    """The same chunk scored `repeat` times inside one engine process. The rows file is append-written, so a
    (chunk, pos) that came back `repeat` times can be compared pass against pass, which asks of a real pack in a
    minute whether a request starts where the previous one stopped: a session that carries its recurrent state into
    the next request shows about a nat of difference, one that restarts cleanly shows only the expert tier's two
    quantizers, a few thousandths. Inside one process the tier's placement also moves between the passes, so a
    `--diag-tier-frozen` run next to this one says which of the two the gap was."""
    seen: dict[tuple[int, int], list[float]] = {}
    for line in open(rows_path):
        f = line.split("\t")
        if len(f) < 10:
            continue
        seen.setdefault((int(f[0]), int(f[1])), []).append(float(f[3]))
    worst, pair, n = 0.0, None, 0
    for key, v in seen.items():
        if len(v) < repeat:
            continue
        n += 1
        if max(v) - min(v) > worst:
            worst, pair = max(v) - min(v), key
    if n == 0:
        return f"carryover: nothing repeated {repeat} times - the engine wrote fewer rows than the passes asked for"
    where = f" (chunk {pair[0]} pos {pair[1]})" if pair else ""
    return (f"state carryover across {repeat} passes: max |d nll| {worst:.5f} nats over {n} positions{where} - "
            f"a few thousandths is the tier's card/CPU rounding, about a nat is a request inheriting the "
            f"previous one's caches")


def diff(pa: str, pb: str) -> str:
    def load(p):
        with open(p) as fh:
            return {(f[0], f[1]): f for f in (l.rstrip("\n").split("\t") for l in fh) if len(f) >= 10}
    a, b = load(pa), load(pb)
    keys = sorted(set(a) & set(b), key=lambda k: (int(k[0]), int(k[1])))
    if not keys:
        return "no common rows"
    def row_same(k):
        return a[k][3:] == b[k][3:] and all(math.isfinite(float(a[k][c])) for c in (3, 4, 5, 9))
    same = sum(row_same(k) for k in keys)
    top = sum(a[k][7] == b[k][7] for k in keys)
    dn = _max_or_nan(abs(float(a[k][3]) - float(b[k][3])) for k in keys)
    dk = _max_or_nan(abs(float(a[k][5]) - float(b[k][5])) for k in keys)
    first = next((k for k in keys if not row_same(k)), None)
    return (f"rows {len(keys)}: identical {same} ({100.0 * same / len(keys):.2f}%), same top-1 {top}, "
            f"max |dNLL| {dn:.6f}, max |dKLD| {dk:.6f}, first difference "
            f"{'none' if first is None else 'chunk %s pos %s' % first}")


def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in out.splitlines() if line.startswith("GPU "))


def gpu_ids(args: list[str]) -> int:
    """How many cards `args` put the model on (--gpu 0,1 -> 2)."""
    return len(args[args.index("--gpu") + 1].split(",")) if "--gpu" in args else 1


def set_args(args: list[str], extra: list[str]) -> list[str]:
    """`args` with each flag of `extra` replacing the same flag (and its value) in `args`, or added."""
    out = list(args)
    i = 0
    while i < len(extra):
        flag = extra[i]
        has_value = i + 1 < len(extra) and not extra[i + 1].startswith("--")
        value = [extra[i + 1]] if has_value else []
        if flag in out:
            j = out.index(flag)
            old_has_value = j + 1 < len(out) and not out[j + 1].startswith("--")
            out[j:j + 1 + old_has_value] = [flag, *value]
        else:
            out += [flag, *value]
        i += 1 + has_value
    return out


def run(a) -> int:
    n_ctx, n_vocab, n_chunk, chunks = read_base(a.base)
    first = n_ctx // 2
    model_path = Path(a.model)
    args = ["--model", str(model_path.resolve()), "--spec", "5"]
    if a.architecture == "qwen4exp":
        args += ["--adapt-swaps", "0", "--peer-adapt-swaps", "0"]
    args += ["--max-context", "4096"]
    # No conversation caching: each chunk must start from nothing (a reused prompt prefix would score rows
    # the engine did not read in this request), and a model whose session keeps no running state cannot
    # checkpoint at all - with caching on, serve fails the request with "this model keeps no checkpoint".
    args += ["--prompt-cache", "0"]
    if gpu_count() > 1:
        args += ["--gpu", "0,1"]   # a model that only splits: add --gpu-mode split
    args = set_args(args, a.engine_args + ["--ppl-base", str(Path(a.base).resolve())])
    if "--gpu-mode" not in args and gpu_ids(args) > 1:
        args += ["--gpu-mode", "peer"]      # one card refuses a mode: "a mode needs two or more"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = out / "rows.tsv"
    rows.unlink(missing_ok=True)
    args = set_args(args, ["--ppl-out", str(rows)])
    with open(out / "engine.log", "w") as log:
        log.write(f"ppl_gate: engine={a.engine_bin} T={a.t} chunks={a.chunks} args={' '.join(args)}\n")
        log.flush()
        t0 = time.time()
        eng = subprocess.Popen([a.engine_bin, "--serve", *args], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        quit_sent = False
        try:
            for line in eng.stdout:
                if line.startswith("READY"):
                    break
            else:
                print("the engine exited before READY; see", out / "engine.log")
                return 1
            t_load = time.time() - t0
            hits = look = off = 0
            pp_ms = dec_ms = 0.0
            nc = min(a.chunks, n_chunk) if a.chunks > 0 else n_chunk
            passes = max(1, a.repeat)
            t1 = time.time()
            sent = 0
            for p in range(passes):
                for k in range(nc):
                    ids = ",".join(str(t) for t in chunks[k][:first + 1])
                    eng.stdin.write(f"GEN {n_ctx - 1 - first} ppl={k} ppl_t={a.t} {ids}\n")
                    eng.stdin.flush()
                    for line in eng.stdout:
                        if line.startswith("ERR"):
                            print(f"pass {p + 1} chunk {k}: {line.strip()}")
                            return 1
                        if line.startswith("DONE"):
                            f = line.split()
                            pp_ms += float(f[3]); dec_ms += float(f[4])
                            if f[8] != "0":
                                print(f"pass {p + 1} chunk {k}: {f[8]} prompt tokens reused from an earlier request")
                            if len(f) > 10:
                                hits += int(f[9]); look += int(f[10])
                            if len(f) > 15:
                                off += int(f[15])
                            break
                    else:
                        print(f"pass {p + 1} chunk {k}: the engine exited; see", out / "engine.log")
                        return 1
                    sent += 1
                    if a.progress and sent % a.progress == 0:
                        print(f"  {sent}/{passes * nc} windows, {time.time() - t1:.0f} s", flush=True)
            t_run = time.time() - t1
            eng.stdin.write("QUIT\n")
            eng.stdin.flush()
            quit_sent = True
        finally:
            if eng.poll() is None and not quit_sent and eng.stdin is not None and not eng.stdin.closed:
                try:
                    eng.stdin.write("QUIT\n")
                    eng.stdin.flush()
                except OSError:
                    pass
            for pipe in (eng.stdin, eng.stdout):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass
            try:
                eng.wait(timeout=SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                eng.kill()
                eng.wait()
    extra = {"T": a.t, "load s": f"{t_load:.0f}", "run s": f"{t_run:.1f}", "passes": f"{passes}",
             "prompt ms/chunk": f"{pp_ms / max(1, sent):.0f}", "decode ms/chunk": f"{dec_ms / max(1, sent):.0f}",
             "decode routed experts": f"{look + off}: VRAM hits {100.0 * hits / max(1, look + off):.1f}%, "
                                      f"peer GPU/PCIe {100.0 * off / max(1, look + off):.1f}%, "
                                      f"CPU {100.0 * (look - hits) / max(1, look + off):.1f}%"}
    rep = summary(str(rows), extra)
    if passes > 1:
        rep += "\n" + carryover(str(rows), passes)
    (out / "report.txt").write_text(f"engine {a.engine_bin}\nargs {' '.join(args)}\n{rep}\n")
    print(rep)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", help="the .basalt file")
    ap.add_argument("--architecture", choices=("qwen4exp", "inkling2", "other"),
                    help="model architecture; selects architecture-specific measurement controls")
    ap.add_argument("--engine-bin", default=str(ENGINE_BIN) if ENGINE_BIN.is_file() else None,
                    help="the engine binary (default: bin/basalt in the repo)")
    ap.add_argument("--base")
    ap.add_argument("--out")
    ap.add_argument("--chunks", type=int, default=0, help="the first N chunks (0: all)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="send each chunk this many times in the same engine process, and report the spread "
                         "between the passes: 2 answers in a minute whether a reused session restarts its state "
                         "for every request")
    ap.add_argument("--t", type=int, default=3, help="tokens per verify window")
    ap.add_argument("--progress", type=int, default=10, help="a progress line every N chunks (0: none)")
    ap.add_argument("--report", help="print the summary of an existing rows.tsv and exit")
    ap.add_argument("--base-stats", metavar="BASE", help="print what the base file alone says (llama.cpp's "
                                                         "perplexity over the scored rows) and exit")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"), help="compare two rows.tsv files row by row and exit")
    a, extra = ap.parse_known_args()
    a.engine_args = extra
    if a.diff:
        print(diff(*a.diff))
        return 0
    if a.report:
        print(summary(a.report))
        return 0
    if a.base_stats:
        print(base_stats(a.base_stats))
        return 0
    if not a.architecture:
        ap.error("--architecture is required when running the gate")
    if not (a.model and a.base and a.out):
        ap.error("--model, --base and --out are required")
    if not a.engine_bin:
        ap.error("no engine binary: pass --engine-bin")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
