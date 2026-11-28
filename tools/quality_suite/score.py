"""Scorer for quality suite v2.
  score.py ref.safetensors test.bin [--role assistant|other|all] [--test2 other.bin] [--boot 2000]
test.bin: llama_project output (per scored row: float lp[k] at the reference ids, float leftover, int32 top1), in
reference order. Everything is computed over scored rows; error bars bootstrap whole conversations."""
import argparse, json, sys
import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file

def load(ref, test, sel=None):
    t = load_file(ref); ro = t["row_off"]; roles = t["roles"]; n = len(roles)
    rl = np.diff(ro); rows = np.nonzero(rl > 0)[0]            # rows with data, in order
    if sel is not None: rows = rows[np.isin(np.searchsorted(t["conv_off"], rows, side="right") - 1, sel)]
    k = rl[rows]; starts = ro[rows]
    # test.bin: for each row in order: k floats lp, 1 float leftover, 1 int32 top1
    raw = np.fromfile(test, dtype=np.uint8)
    stride = (k + 2) * 4; offs = np.concatenate([[0], np.cumsum(stride)])
    assert offs[-1] == len(raw), f"test file size {len(raw)} != expected {offs[-1]}"
    u32 = raw.view(np.float32)  # words
    wo = offs[:-1] // 4
    flat_idx = np.concatenate([np.arange(a, a + kk) for a, kk in zip(wo, k)]) if len(rows) else np.array([], int)
    tlp = u32[flat_idx]; tleft = u32[wo + k]; ttop1 = raw.view(np.int32)[wo + k + 1]
    conv = np.searchsorted(t["conv_off"], rows, side="right") - 1
    return dict(t=t, rows=rows, k=k, starts=starts, tlp=tlp, tleft=tleft, ttop1=ttop1, conv=conv, role=roles[rows + 1])

def load_basalt(ref, files, binary=0, sel=None):
    """Test input from Basalt `--ppl-out` files (one file per conversation, in reference conversation order).
    Text rows (score_top=K>=1): `pos target lp_target lse id:lp id:lp ...`; binary rows (score_top=0, pass
    --basalt-vocab N): int64 pos, int32 target, N float32 log-probs. `pos` is matched either as the source
    position (target == tokens[pos+1]) or as the target index (target == tokens[pos]), detected on the first
    row. Reference ids missing from a text row's top-K get that row's smallest listed log-prob (an upper
    bound) and are counted; use K large enough to cover the reference ids."""
    t = load_file(ref); ro = t["row_off"]; conv_off = t["conv_off"]; tok = t["tokens"]
    rl = np.diff(ro); rows = np.nonzero(rl > 0)[0]
    convs = list(range(len(conv_off) - 1)) if sel is None else list(sel)
    if sel is not None: rows = rows[np.isin(np.searchsorted(conv_off, rows, side="right") - 1, sel)]
    assert len(files) == len(convs), "need one file per selected reference conversation"
    out_lp, out_left, out_top1, missing = [], [], [], 0
    shift = None
    for c, fn in zip(convs, files):
        a = int(conv_off[c]); crow = rows[(rows >= a) & (rows < conv_off[c + 1])]
        recs = {}
        if not binary:
            for line in open(fn):
                w = line.split()
                if not w:
                    continue
                pos, target = int(w[0]), int(w[1])
                recs[pos] = (target, {int(x.split(":")[0]): float(x.split(":")[1]) for x in w[4:]})
        else:
            rs = 8 + 4 + binary * 4
            raw = np.memmap(fn, dtype=np.uint8, mode="r")
            for r in range(len(raw) // rs):
                b = np.asarray(raw[r * rs:(r + 1) * rs])
                recs[int(b[:8].view(np.int64)[0])] = (int(b[8:12].view(np.int32)[0]), b[12:].view(np.float32))
        if shift is None:
            p0 = min(recs); tg = recs[p0][0]
            shift = 1 if tok[a + p0 + 1] == tg else 0
            assert shift or tok[a + p0] == tg, "cannot align Basalt pos/target with the reference tokens"
        for i in crow:
            key = i - a if shift else i - a + 1
            target, d = recs[key]
            ids = t["topk_ids"][ro[i]:ro[i + 1]]
            if binary:
                lp = d[ids].astype(np.float32); top1 = int(np.argmax(d))
            else:
                mn = min(d.values()); lp = np.array([d.get(int(x), mn) for x in ids], np.float32)
                missing += sum(int(x) not in d for x in ids); top1 = max(d, key=d.get)
            out_lp.append(lp); out_left.append(max(0.0, 1.0 - float(np.exp(lp.astype(np.float64)).sum()))); out_top1.append(top1)
    if missing:
        print(f"WARNING: {missing} reference ids were not in the Basalt top-K rows (bounded by the row minimum)", file=sys.stderr)
    conv = np.searchsorted(conv_off, rows, side="right") - 1
    return dict(t=t, rows=rows, k=rl[rows], starts=ro[rows], tlp=np.concatenate(out_lp), tleft=np.array(out_left, np.float32),
                ttop1=np.array(out_top1, np.int32), conv=conv, role=t["roles"][rows + 1])


def per_row(d):
    t, rows, k, st = d["t"], d["rows"], d["k"], d["starts"]
    flat = np.concatenate([np.arange(a, a + kk) for a, kk in zip(st, k)]) if len(rows) else np.array([], int)   # entries of the selected rows
    rlp, ids = t["topk_lp"][flat].astype(np.float64), t["topk_ids"][flat]
    tlp = d["tlp"].astype(np.float64)
    seg = np.repeat(np.arange(len(rows)), k)
    p = np.exp(rlp)
    Lp = t["leftover"][rows].astype(np.float64)
    Lq = np.clip(d["tleft"].astype(np.float64), 1e-12, None)
    kl = np.bincount(seg, weights=p * (rlp - tlp), minlength=len(rows))
    m = Lp > 0
    kl[m] += Lp[m] * np.log(Lp[m] / Lq[m])
    first = np.concatenate([[0], np.cumsum(k)[:-1]])          # ref top-1 is the first entry of each row
    top1_ref = ids[first]; same = (top1_ref == d["ttop1"]).astype(np.float64)
    dp = np.exp(tlp[first]) - p[first]                         # change in probability of the reference top token
    # actual next token: nll under ref and test, where it is in the stored top-K
    tgt = t["tokens"][rows + 1]
    hit = (ids == np.repeat(tgt, k)); pos = np.nonzero(hit)[0]
    has = np.zeros(len(rows), bool); has[seg[pos]] = True
    nllb = np.zeros(len(rows)); nllq = np.zeros(len(rows))
    nllb[seg[pos]] = -rlp[pos]; nllq[seg[pos]] = -tlp[pos]
    return dict(kld=kl, same=same, dp=dp, has=has, nllb=nllb, nllq=nllq)

def boot_sums(conv, cols, nconv, B, seed=0):
    """cols: dict name -> per-row value; returns per-conv sums and counts, bootstrap indices."""
    cnt = np.bincount(conv, minlength=nconv).astype(np.float64)
    sums = {n: np.bincount(conv, weights=v, minlength=nconv) for n, v in cols.items()}
    return cnt, sums

def summarize(d, pr, mask, B, seed=0):
    conv = d["conv"][mask]; nconv = len(d["t"]["conv_off"]) - 1
    keep = np.unique(conv); remap = -np.ones(nconv, int); remap[keep] = np.arange(len(keep)); c = remap[conv]
    kl, same, dp = pr["kld"][mask], pr["same"][mask], pr["dp"][mask]; has = pr["has"][mask].astype(np.float64)
    dn = (pr["nllq"] - pr["nllb"])[mask]
    cols = dict(kl=kl, same=same, dp=dp, dp2=dp * dp, has=has, dn=dn, nq=pr["nllq"][mask], nb=pr["nllb"][mask], one=np.ones(len(kl)))
    cnt, S = boot_sums(c, cols, len(keep), B)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(keep), size=(B, len(keep)))
    def stat(f):
        full = f({n: v.sum() for n, v in S.items()}, cnt.sum())
        bs = f({n: v[idx].sum(1) for n, v in S.items()}, cnt[idx].sum(1))
        return full, bs.std(ddof=1)
    out = {}
    out["KLD"] = stat(lambda s, n: s["kl"] / s["one"])
    out["top1"] = stat(lambda s, n: s["same"] / s["one"])
    out["dp_mean"] = stat(lambda s, n: s["dp"] / s["one"])
    out["dp_rms"] = stat(lambda s, n: np.sqrt(s["dp2"] / s["one"]))
    out["ln_ppl_ratio"] = stat(lambda s, n: s["dn"] / s["has"])
    out["ppl_ratio"] = stat(lambda s, n: np.exp(s["dn"] / s["has"]))
    out["ppl_q"] = stat(lambda s, n: np.exp(s["nq"] / s["has"]))
    out["ppl_base"] = stat(lambda s, n: np.exp(s["nb"] / s["has"]))
    naive = kl.std(ddof=1) / np.sqrt(len(kl))
    q = np.percentile(kl, [50, 90, 95, 99, 99.9, 100])
    return out, dict(n_rows=int(len(kl)), n_convs=int(len(keep)), kld_naive_se=float(naive),
                     kld_pct=dict(zip(["median", "p90", "p95", "p99", "p99.9", "max"], map(float, q))),
                     rows_target_outside_topk=int((has == 0).sum()))

def paired(dA, pA, dB, pB, mask, B, seed=0):
    conv = dA["conv"][mask]; nconv = len(dA["t"]["conv_off"]) - 1
    keep = np.unique(conv); remap = -np.ones(nconv, int); remap[keep] = np.arange(len(keep)); c = remap[conv]
    res = {}
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(keep), size=(B, len(keep)))
    one = np.ones(mask.sum())
    for name, a, b in [("KLD", pA["kld"], pB["kld"]), ("top1", pA["same"], pB["same"])]:
        a, b = a[mask], b[mask]
        sa, sb, sd = [np.bincount(c, weights=v, minlength=len(keep)) for v in (a, b, a - b)]
        cnt = np.bincount(c, minlength=len(keep)).astype(float)
        f = lambda s: s[idx].sum(1) / cnt[idx].sum(1)
        res[name] = dict(A=sa.sum() / cnt.sum(), A_se=f(sa).std(ddof=1), B=sb.sum() / cnt.sum(), B_se=f(sb).std(ddof=1),
                         diff=sd.sum() / cnt.sum(), diff_paired_se=f(sd).std(ddof=1),
                         diff_unpaired_se=float(np.hypot(f(sa).std(ddof=1), f(sb).std(ddof=1))))
    return res

def fmt(name, v):
    return f"{name:14s} {v[0]:12.6f} +- {v[1]:.6f}"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ref"); ap.add_argument("test", nargs="?"); ap.add_argument("--test2")
    ap.add_argument("--role", default="assistant", choices=["assistant", "other", "all"]); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true", help="only the fixed smoke slice of conversations (common.smoke_convs); test files must cover exactly those"); ap.add_argument("--json"); ap.add_argument("--basalt", nargs="+", help="Basalt --ppl-out files, one per conversation, instead of test.bin"); ap.add_argument("--basalt-vocab", type=int, default=0, help="n_vocab for binary (score_top=0) files"); a = ap.parse_args()
    from common import smoke_convs
    sel = smoke_convs(load_file(a.ref)["conv_off"]) if a.smoke else None
    d = load_basalt(a.ref, a.basalt, a.basalt_vocab, sel) if a.basalt else load(a.ref, a.test, sel); pr = per_row(d)
    mask = {"assistant": d["role"] == 1, "other": d["role"] == 2, "all": d["role"] > 0}[a.role]
    out, info = summarize(d, pr, mask, a.boot)
    print(f"rows={info["n_rows"]} conversations={info["n_convs"]} role={a.role}  (+- = conversation bootstrap, {a.boot} resamples)")
    print("====== KL divergence statistics (top-K + leftover bucket) ======")
    print(fmt("Mean KLD", out["KLD"]), f"   (per-token naive +- {info["kld_naive_se"]:.6f})")
    for k, v in info["kld_pct"].items(): print(f"  {k:8s} KLD {v:12.6f}")
    print("====== Token statistics ======")
    print(fmt("Same top p", (out["top1"][0] * 100, out["top1"][1] * 100)), "%")
    print(fmt("Mean dp (top)", (out["dp_mean"][0] * 100, out["dp_mean"][1] * 100)), "%")
    print(fmt("RMS dp (top)", (out["dp_rms"][0] * 100, out["dp_rms"][1] * 100)), "%")
    print("====== Perplexity (rows whose next token is in the stored top-K) ======")
    print(fmt("PPL(Q)", out["ppl_q"])); print(fmt("PPL(base)", out["ppl_base"]))
    print(fmt("ln(PPL Q/base)", out["ln_ppl_ratio"])); print(fmt("PPL(Q)/PPL(base)", out["ppl_ratio"]))
    print("rows with next token outside top-K (excluded from PPL):", info["rows_target_outside_topk"])
    res = dict(single=dict(stats={k: list(map(float, v)) for k, v in out.items()}, info=info))
    if a.test2:
        d2 = load(a.ref, a.test2); pr2 = per_row(d2)
        assert (d2["rows"] == d["rows"]).all()
        pp = paired(d, pr, d2, pr2, mask, a.boot); res["paired"] = pp
        print("====== Paired A (test) vs B (test2), same positions ======")
        for k, v in pp.items():
            print(f"{k}: A={v["A"]:.6f}+-{v["A_se"]:.6f}  B={v["B"]:.6f}+-{v["B_se"]:.6f}  A-B={v["diff"]:.6f}  "
                  f"paired +-{v["diff_paired_se"]:.6f}  unpaired +-{v["diff_unpaired_se"]:.6f}  (ratio {v["diff_unpaired_se"]/max(v["diff_paired_se"],1e-12):.2f}x)")
    if a.json: json.dump(res, open(a.json, "w"), indent=1)
