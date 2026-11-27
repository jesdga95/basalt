"""Offline analyses on the stored reference: (1) error bars at a 25.6K-token subsample vs the full set,
(2) size/accuracy trade-off of the stored truncation (threshold, cap)."""
import sys, numpy as np
import score as S
if len(sys.argv) != 3: sys.exit("usage: analysis.py REF.safetensors TEST.bin")
ref, test = sys.argv[1], sys.argv[2]
d = S.load(ref, test); t = d["t"]; pr = S.per_row(d)
asst = d["role"] == 1
# (1) subsample conversations (excluding long ones) until >= 25,600 assistant tokens; conversation-bootstrap +- on each
rng = np.random.default_rng(7)
nconv = len(t["conv_off"]) - 1
clen = np.diff(t["conv_off"]); short = np.nonzero(clen < 5000)[0]
cnt_asst = np.bincount(d["conv"][asst], minlength=nconv)
res = []
for rep in range(20):
    perm = rng.permutation(short); cs = np.cumsum(cnt_asst[perm]); sel = perm[:np.searchsorted(cs, 25600) + 1]
    m = asst & np.isin(d["conv"], sel)
    out, info = S.summarize(d, pr, m, 500, seed=rep)
    res.append((info["n_rows"], len(sel), out["KLD"][0], out["KLD"][1], info["kld_naive_se"], out["top1"][0], out["top1"][1]))
r = np.array(res)
print("25.6K-token subsamples (20 draws): rows %.0f convs %.0f | KLD %.5f +- %.5f (naive per-token +- %.5f) | top1 %.3f%% +- %.3f%%" % (
    r[:, 0].mean(), r[:, 1].mean(), r[:, 2].mean(), r[:, 3].mean(), r[:, 4].mean(), 100 * r[:, 5].mean(), 100 * r[:, 6].mean()))
out, info = S.summarize(d, pr, asst, 1000)
print("full assistant set: rows %d convs %d | KLD %.5f +- %.5f (naive +- %.5f) | top1 %.3f%% +- %.3f%%" % (
    info["n_rows"], info["n_convs"], out["KLD"][0], out["KLD"][1], info["kld_naive_se"], 100 * out["top1"][0], 100 * out["top1"][1]))
# (2) truncation sweep on assistant rows: keep first k entries with cum p >= 1 - tau (cap), recompute both leftovers
rows, k = d["rows"], d["k"]
sel_rows = np.nonzero(asst)[0]
rlp_all, tlp_all = t["topk_lp"].astype(np.float64), d["tlp"].astype(np.float64)
starts = np.concatenate([[0], np.cumsum(k)])[:-1]
base = None
print("\ntruncation sweep (assistant rows, same test run; KLD with leftover bucket):")
for tau, cap in [(1e-6, 256), (1e-5, 256), (1e-4, 256), (1e-3, 256), (1e-4, 64), (1e-3, 64), (1e-2, 32)]:
    kl = np.zeros(len(sel_rows)); ent = 0; kk_all = []
    for n_i, ri in enumerate(sel_rows):
        a, kk = starts[ri], k[ri]
        p = np.exp(rlp_all[a:a + kk]); c = np.cumsum(p)
        # stored leftover counts as mass beyond the stored ids
        kt = min(int(np.searchsorted(c, 1 - tau)) + 1, kk, cap)
        pr_, rl_, tl_ = p[:kt], rlp_all[a:a + kt], tlp_all[a:a + kt]
        Lp = max(0.0, 1 - pr_.sum()); Lq = max(1 - np.exp(tl_).sum(), 1e-12)
        kl[n_i] = (pr_ * (rl_ - tl_)).sum() + (Lp * np.log(Lp / Lq) if Lp > 0 else 0); ent += kt
    if base is None: base = kl.copy()
    # bytes per entry: int32 id + fp32 lp = 8
    print(f"  tau={tau:g} cap={cap:3d}: KLD {kl.mean():.6f} (bias vs stored {kl.mean()-base.mean():+.2e}) mean k {ent/len(sel_rows):6.1f}  assistant-only size ~{ent*8/1e6:6.1f} MB")
