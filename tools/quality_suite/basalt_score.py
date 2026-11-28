"""Score a Basalt pack against the reference through the engine's `score=N score_top=K` serve mode.
  basalt_score.py --pack FILE.basalt --ref ref.safetensors --out DIR [--smoke] [--top 256] [--engine BIN] [--gpu 0]
Starts `basalt --serve --ppl-out DIR/rows.txt`, sends one `GEN 1 score=0 score_top=K <ids>` per conversation (the
whole stream is teacher-forced, so every reference row exists), splits the rows file per conversation by byte offset
and writes DIR/conv_NNNN.txt. Then: score.py REF --basalt DIR/conv_*.txt [--smoke]."""
import argparse, os, subprocess, sys, time
from pathlib import Path
from safetensors.numpy import load_file
from common import smoke_convs
ap = argparse.ArgumentParser()
ap.add_argument("--pack", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--smoke", action="store_true"); ap.add_argument("--top", type=int, default=256)
ap.add_argument("--engine", default=str(Path(__file__).resolve().parents[2] / "bin" / "basalt"), help="the basalt binary (default: bin/basalt in the repo)"); ap.add_argument("--gpu", default="0")
ap.add_argument("--max-context", type=int, default=5120); ap.add_argument("--ppl-t", type=int, default=16); ap.add_argument("--limit", type=int)
a = ap.parse_args()
for what, path in (("--pack", a.pack), ("--ref", a.ref), ("--engine", a.engine)):
    if not os.path.isfile(path): ap.error(f"{what} {path}: no such file")
t = load_file(a.ref); off, ro, toks = t["conv_off"], t["row_off"], t["tokens"]
convs = smoke_convs(off) if a.smoke else [c for c in range(len(off) - 1) if off[c + 1] - off[c] <= a.max_context]
if a.limit: convs = convs[:a.limit]
os.makedirs(a.out, exist_ok=True)
rows = os.path.join(a.out, "rows.txt")
if os.path.exists(rows): os.remove(rows)
log = open(os.path.join(a.out, "engine.log"), "w")
eng = subprocess.Popen([a.engine, "--serve", "--model", a.pack, "--gpu", a.gpu, "--max-context", str(a.max_context),
                        "--prompt-cache", "0", "--ppl-out", rows], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
for line in eng.stdout:
    if line.startswith("READY"): break
else: sys.exit("engine exited before READY; see " + log.name)
t0 = time.time(); prev = 0; used = []
for n, c in enumerate(convs):
    c0, c1 = int(off[c]), int(off[c + 1])
    nz = [i - c0 for i in range(c0, c1) if ro[i + 1] > ro[i]]
    if not nz:
        open(os.path.join(a.out, f"conv_{c:04d}.txt"), "wb").write(b""); used.append(c); continue
    first_pos = nz[0]
    ids = ",".join(map(str, toks[c0:c1].tolist()))
    eng.stdin.write(f"GEN 1 score={first_pos} score_top={a.top} ppl_t={a.ppl_t} {ids}\n"); eng.stdin.flush()
    for line in eng.stdout:
        if line.startswith("ERR"): sys.exit(f"conv {c}: {line.strip()}")
        if line.startswith("DONE"): break
    else: sys.exit("engine exited; see " + log.name)
    size = os.path.getsize(rows)
    with open(rows, "rb") as f: f.seek(prev); chunk = f.read(size - prev)
    open(os.path.join(a.out, f"conv_{c:04d}.txt"), "wb").write(chunk); prev = size; used.append(c)
    print(f"{n+1}/{len(convs)} conv {c} {len(chunk.splitlines())} rows {time.time()-t0:.0f}s", flush=True)
eng.stdin.write("QUIT\n"); eng.stdin.flush()
try: eng.wait(timeout=60)
except subprocess.TimeoutExpired: eng.kill()
open(os.path.join(a.out, "convs.txt"), "w").write(" ".join(map(str, used)))
print("SCORE_RUN_DONE", len(used), "conversations", f"{time.time()-t0:.0f}s")
