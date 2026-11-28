#!/usr/bin/env python3
"""Compare cached prompt resume against full recompute (`--serve`).

    prompt_cache_resume_drift_test.py ENGINE MODEL.basalt [ENGINE ARGS...]

Runs a suite of short (`<= 8` tokens, decode path on resume) and long (`> 8` tokens, prefill path on resume)
suffixes appended to a shared prefix (> 2048 tokens by default) under greedy decoding (`temperature=0`,
`logprobs=5`) using the model's own tokenizer via `GENT`:
  1. Full recompute (`--prompt-cache 0`): warms expert placement on `prefix`, then runs each `prefix + suffix`
     from position 0 through the prefill path.
  2. Cached resume (`--prompt-cache 2`): caches `prefix`, then resumes each `prefix + suffix` from the cached
     prefix checkpoint (`REUSED >= min_reused`).

Reports per-prompt greedy prefix match and first-token logprob difference, plus the summary mean/max drift,
so any numerical drift between prefill (>8 rows) and decode/resume is directly measured and bounded.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def build_prefix(target_blocks: int) -> str:
    parts = [
        "Technical architecture notes for a sparse mixture-of-experts transformer runtime.\n"
        "The attention stack combines sliding-window compressed layers with sparse latent attention layers "
        "whose Lightning Indexer selects top-k historical positions beyond the local window.\n\n"
    ]
    for i in range(target_blocks):
        parts.append(
            f"Section {i + 1}: In step {i * 17 + 3}, the scheduler batches {8 + (i % 7)} prompt rows across "
            f"stream lanes {i % 4} and verifies quantized key-value row stride {608 + (i % 3) * 16} bytes "
            f"against the host checkpoint buffer before advancing position {(i + 1) * 32}.\n"
        )
    parts.append("\n")
    return "".join(parts)


SUFFIXES = [
    ("short_a (<=8)", "Summary:"),
    ("short_b (<=8)", "Answer:"),
    ("short_c (<=8)", "Conclusion:\n"),
    ("long_a (>8)", (
        "Explain in one concise paragraph why a hybrid runtime might use a batched BF16 or FP16 "
        "path when processing large prompt chunks and a specialized single-row or small-window path "
        "when decoding individual tokens, and what numerical drift that introduces when resuming "
        "from a cached prefix checkpoint:\n"
    )),
    ("long_b (>8)", (
        "List the three most important architectural components described above, explain how each "
        "affects memory usage, and state the main trade-off:\n"
    )),
]


class Engine:
    def __init__(self, cmd: list[str], log_path: Path):
        self.log = open(log_path, "w")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
        while True:
            raw = self.p.stdout.readline()
            if not raw:
                raise RuntimeError(f"engine exited before READY (see {log_path})")
            if raw.decode("utf-8", errors="replace").startswith("READY"):
                break

    def run_text(self, text: str, max_new: int, logprobs: int = 5) -> tuple[int, list[int], list[float]]:
        """Run one greedy GENT and return (reused, token_ids, top1_logprobs)."""
        payload = text.encode("utf-8")
        hdr = f"GENT {max_new} temperature=0 logprobs={logprobs} {len(payload)}\n".encode("utf-8")
        self.p.stdin.write(hdr)
        self.p.stdin.write(payload)
        self.p.stdin.flush()
        reused = -1
        toks: list[int] = []
        lps: list[float] = []
        while True:
            raw = self.p.stdout.readline()
            if not raw:
                raise RuntimeError("engine exited unexpectedly")
            line = raw.decode("utf-8", errors="replace").strip()
            f = line.split()
            if not f:
                continue
            if f[0] == "ERR":
                raise RuntimeError(line)
            if f[0] == "REUSED":
                reused = int(f[1])
            elif f[0] == "P" and len(f) >= 3:
                lps.append(float(f[2]))
            elif f[0] == "T" and len(f) >= 2:
                toks.append(int(f[1]))
            elif f[0] == "DONE":
                return reused, toks, lps

    def stop(self) -> None:
        try:
            self.p.stdin.write(b"QUIT\n")
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
    ap.add_argument("--prefix-blocks", type=int, default=64,
                    help="repeated paragraphs in prefix (~43 tokens/block; 64 blocks ~= 2765 tokens)")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--max-context", type=int, default=8192)
    ap.add_argument("--min-tok0-match", type=int, default=4)
    ap.add_argument("--max-mean-lp-diff", type=float, default=0.35)
    args, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    prefix = build_prefix(args.prefix_blocks)
    min_reused = max(64, args.prefix_blocks * 20)

    with tempfile.TemporaryDirectory(prefix="basalt_resume_drift_") as tmp:
        tmp_p = Path(tmp)
        cold = Engine([args.engine, "--model", args.model, "--serve", "--prompt-cache", "0",
                       "--max-context", str(args.max_context), *extra], tmp_p / "cold.log")
        try:
            # Warm expert placement once on prefix so cold vs warm isolates prefill-vs-resume drift.
            cold.run_text(prefix, 2)
            full_res = {}
            for name, suf in SUFFIXES:
                _, t, lp = cold.run_text(prefix + suf, args.max_new)
                full_res[name] = (t, lp)
        finally:
            cold.stop()

        warm = Engine([args.engine, "--model", args.model, "--serve", "--prompt-cache", "2",
                       "--max-context", str(args.max_context), *extra], tmp_p / "warm.log")
        try:
            warm.run_text(prefix, 2)
            warm_res = {}
            for name, suf in SUFFIXES:
                warm_res[name] = warm.run_text(prefix + suf, args.max_new)
        finally:
            warm.stop()

    ok = True
    tok0_matches = 0
    matched_toks = 0
    total_toks = 0
    lp_diffs: list[float] = []
    for name, _ in SUFFIXES:
        ft, flp = full_res[name]
        reused, wt, wlp = warm_res[name]
        n = min(len(ft), len(wt))
        m = 0
        while m < n and ft[m] == wt[m]:
            m += 1
        t0 = bool(ft and wt and ft[0] == wt[0])
        tok0_matches += int(t0)
        matched_toks += m
        total_toks += n
        d0 = abs(flp[0] - wlp[0]) if flp and wlp else float("inf")
        lp_diffs.append(d0)
        print(f"[{name:14s}] reused={reused} tok0_match={t0} greedy_prefix_match={m:2d}/{n} "
              f"first_lp_diff={d0:.6f} (cold={flp[0]:.4f} warm={wlp[0]:.4f})")
        if reused < min_reused:
            print(f"  FAIL: expected reused >= {min_reused}, got {reused}", file=sys.stderr)
            ok = False

    mean_lp = sum(lp_diffs) / len(lp_diffs)
    max_lp = max(lp_diffs)
    print(f"summary: tok0_match={tok0_matches}/{len(SUFFIXES)} greedy_prefix={matched_toks}/{total_toks} "
          f"mean_first_lp_diff={mean_lp:.6f} max_first_lp_diff={max_lp:.6f}")
    if tok0_matches < args.min_tok0_match:
        print(f"FAIL: tok0_match {tok0_matches} < {args.min_tok0_match}", file=sys.stderr)
        ok = False
    if mean_lp > args.max_mean_lp_diff:
        print(f"FAIL: mean_first_lp_diff {mean_lp:.6f} > {args.max_mean_lp_diff:.6f}", file=sys.stderr)
        ok = False
    if not ok:
        return 1
    print("prompt_cache_resume_drift_test: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
