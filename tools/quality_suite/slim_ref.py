"""Shrink a reference file: keep only rows whose target token has the given role(s) and optionally tighten the top-K cut.
  slim_ref.py in.safetensors out.safetensors [--roles 1] [--tau 1e-6] [--cap 256]
Rows of other roles become empty (tokens and roles stay, so conversations and positions are unchanged)."""
import argparse, json, numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file
ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--roles", default="1"); ap.add_argument("--tau", type=float, default=1e-6); ap.add_argument("--cap", type=int, default=256)
a = ap.parse_args(); keep_roles = {int(x) for x in a.roles.split(",")}
t = load_file(a.inp); meta = dict(safe_open(a.inp, "np").metadata())
ro, roles = t["row_off"], t["roles"]; n = len(roles)
ids_o, lp_o, left_o, lens = [], [], np.zeros(n, np.float32), np.zeros(n, np.int64)
for i in range(n):
    if ro[i + 1] == ro[i] or i + 1 >= n or roles[i + 1] not in keep_roles: continue
    lp = t["topk_lp"][ro[i]:ro[i + 1]].astype(np.float64); p = np.exp(lp); c = np.cumsum(p)
    k = min(int(np.searchsorted(c, 1 - a.tau)) + 1, len(lp), a.cap)
    ids_o.append(t["topk_ids"][ro[i]:ro[i] + k]); lp_o.append(t["topk_lp"][ro[i]:ro[i] + k]); lens[i] = k
    left_o[i] = max(0.0, 1.0 - c[k - 1])
t["row_off"] = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
t["topk_ids"] = np.concatenate(ids_o); t["topk_lp"] = np.concatenate(lp_o); t["leftover"] = left_o
meta["slimmed"] = json.dumps(dict(roles=sorted(keep_roles), tau=a.tau, cap=a.cap))
save_file(t, a.out, metadata=meta)
print("rows", int((lens > 0).sum()), "entries", len(t["topk_ids"]), "mean k %.1f" % (len(t["topk_ids"]) / max(1, (lens > 0).sum())))
