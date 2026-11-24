"""The logits of a Nemotron-H NVFP4 checkpoint (Model Optimizer's export: FP8 and NVFP4 weights dequantized to their
exact values) in float64, layer by layer over the tokens, with NVIDIA's fast path: dt = softplus(dt_raw + dt_bias) unclamped, the
norm after the gate, a sigmoid router with a selection bias, squared-ReLU experts in a latent, no positional encoding.
The engine's end-to-end parity test (model_parity.cpp) compares its logits with these.

    python3 tests/models/nemotronh/reference.py CHECKPOINT_DIR --tokens 1,2,3 --out REF.bin

REF.bin: int32 n_tokens, int32 n_vocab, the tokens (int32), then the logits after each token (float32 [n][vocab]).

With an MTP head in the checkpoint, REF.bin.mtp: int32 n - 1, then per position p < n - 1 the head's two greedy drafts
after the token at p + 1 (float32 each: the first draft's id, its probability, the second's id, its probability).  The
first comes from the pair (the model's normed hidden row at p, the token at p + 1), its cache the true pairs before; the
second from the head's normed output and the first draft at p + 1, attending to the true pairs through p and itself."""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tools" / "convert"))
import checkpoint as C  # noqa: E402
import checkpoint_nemotronh as N  # noqa: E402
import model_sections as S  # noqa: E402

E4M3 = S.fp8_e4m3_table().astype(np.float64)
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float64)


class Weights:
    def __init__(self, d: pathlib.Path):
        self.ck = C.Checkpoint(d)
        self.tc = N.text_config(self.ck.config)
        self.g = N.Geometry(self.tc)
        vl = any(k.startswith("language_model.") for k in self.ck.t)
        self.p = "language_model.backbone." if vl else "backbone."
        self.mtp = ("language_model.mtp." if vl else "mtp.") if self.g.mtp_kinds else None
        self.head = ("language_model." if vl else "") + "lm_head"
        self.cache = {}

    def f(self, name) -> np.ndarray:   # BF16 / F32 values
        return self.ck.values(name).astype(np.float64)

    def mat(self, base) -> np.ndarray:   # an FP8, NVFP4 or BF16 matrix's values [rows][cols]
        if base in self.cache:
            return self.cache[base]
        _, _, dt, shape, _ = self.ck.t[base + ".weight"]
        if dt == "F8_E4M3":
            w = E4M3[self.ck.raw(base + ".weight")].reshape(shape) * float(np.ravel(self.ck.values(base + ".weight_scale"))[0])
        elif dt == "U8":
            rows, half = shape
            c = self.ck.raw(base + ".weight").reshape(rows, half)
            v = np.empty((rows, 2 * half), np.float64)
            v[:, 0::2], v[:, 1::2] = E2M1[c & 15], E2M1[c >> 4]
            sc = E4M3[self.ck.raw(base + ".weight_scale")].reshape(rows, -1)
            w = v * np.repeat(sc, 16, axis=1) * float(np.ravel(self.ck.values(base + ".weight_scale_2"))[0])
        else:
            w = self.f(base + ".weight")
        if w.size <= 1 << 24:
            self.cache[base] = w
        return w


def rms(x, w, eps):
    return x / np.sqrt(np.mean(x * x) + eps) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def attention(W: Weights, m: str, h: np.ndarray) -> np.ndarray:
    """A causal attention layer's output over the rows h (normed), positions 0.. in order."""
    g = W.g
    n = len(h)
    q = (h @ W.mat(m + "q_proj").T).reshape(n, g.n_head, g.head_dim)
    k = (h @ W.mat(m + "k_proj").T).reshape(n, g.n_kv, g.head_dim)
    vv = (h @ W.mat(m + "v_proj").T).reshape(n, g.n_kv, g.head_dim)
    att = np.empty((n, g.n_head, g.head_dim))
    causal = np.triu(np.full((n, n), -np.inf), 1)
    for hh in range(g.n_head):
        kh = hh // (g.n_head // g.n_kv)
        s = q[:, hh, :] @ k[:, kh, :].T / np.sqrt(g.head_dim) + causal
        s = np.exp(s - s.max(axis=1, keepdims=True))
        att[:, hh, :] = (s / s.sum(axis=1, keepdims=True)) @ vv[:, kh, :]
    return att.reshape(n, -1) @ W.mat(m + "o_proj").T


def moe(W: Weights, m: str, h: np.ndarray) -> np.ndarray:
    """A latent MoE layer's output over the rows h (normed)."""
    g = W.g
    logits = h @ W.f(m + "gate.weight").T
    sc = 1.0 / (1.0 + np.exp(-logits))
    choice = sc + W.f(m + "gate.e_score_correction_bias")
    lat = h @ W.mat(m + "fc1_latent_proj").T if g.has_latent else h
    acc = np.zeros_like(lat)
    ids = np.argsort(-choice, axis=1, kind="stable")[:, :g.top_k]
    for e in np.unique(ids):
        rows, slot = np.nonzero(ids == e)
        wt = sc[rows, e] / (np.take_along_axis(sc, ids, 1)[rows].sum(axis=1) + 1e-20) * g.scale
        u = lat[rows] @ W.mat(m + f"experts.{e}.up_proj").T
        acc[rows] += wt[:, None] * (np.maximum(u, 0.0) ** 2 @ W.mat(m + f"experts.{e}.down_proj").T)
        W.cache.pop(m + f"experts.{e}.up_proj", None)
        W.cache.pop(m + f"experts.{e}.down_proj", None)
    o = acc @ W.mat(m + "fc2_latent_proj").T if g.has_latent else acc
    su = h @ W.mat(m + "shared_experts.up_proj").T
    return o + np.maximum(su, 0.0) ** 2 @ W.mat(m + "shared_experts.down_proj").T


def mtp_drafts(W: Weights, tokens, hn: np.ndarray) -> np.ndarray:
    """The MTP head's two greedy drafts after each token from the second on: [n - 1][4] (id, p, id, p)."""
    g, eps, P = W.g, W.g.eps, W.mtp
    m0, m1 = P + "layers.0.", P + "layers.1."
    emb = W.f(W.p + "embeddings.weight")
    eh = W.f(m0 + "eh_proj.weight")
    head = W.mat(W.head)

    def fuse(tok, hid):
        e = np.stack([rms(emb[t], W.f(m0 + "enorm.weight"), eps) for t in tok])
        hh = np.stack([rms(r, W.f(m0 + "hnorm.weight"), eps) for r in hid])
        return np.concatenate([e, hh], axis=1) @ eh.T

    def after_attention(x, o):   # the MoE layer, the final norm, the logits
        x = x + o
        x = x + moe(W, m1 + "mixer.", np.stack([rms(r, W.f(m1 + "norm.weight"), eps) for r in x]))
        out = np.stack([rms(r, W.f(m1 + "final_layernorm.weight"), eps) for r in x])
        return out, out @ head.T

    def top(lg):
        z = np.exp(lg - lg.max(axis=1, keepdims=True))
        return lg.argmax(axis=1), (z / z.sum(axis=1, keepdims=True)).max(axis=1)

    n = len(tokens) - 1
    x = fuse(tokens[1:], hn[:n])                                   # the true pairs, positions 0 .. n - 1
    a = np.stack([rms(r, W.f(m0 + "norm.weight"), eps) for r in x])
    out, lg = after_attention(x, attention(W, m0 + "mixer.", a))
    d0, p0 = top(lg)
    # the second draft: one row a position, attending to the true pairs through p and itself
    y = fuse(d0, out)
    b = np.stack([rms(r, W.f(m0 + "norm.weight"), eps) for r in y])
    mm = m0 + "mixer."
    kt = (a @ W.mat(mm + "k_proj").T).reshape(n, g.n_kv, g.head_dim)
    vt = (a @ W.mat(mm + "v_proj").T).reshape(n, g.n_kv, g.head_dim)
    q = (b @ W.mat(mm + "q_proj").T).reshape(n, g.n_head, g.head_dim)
    ks = (b @ W.mat(mm + "k_proj").T).reshape(n, g.n_kv, g.head_dim)
    vs = (b @ W.mat(mm + "v_proj").T).reshape(n, g.n_kv, g.head_dim)
    att = np.empty((n, g.n_head, g.head_dim))
    for p in range(n):
        for hh in range(g.n_head):
            kh = hh // (g.n_head // g.n_kv)
            K = np.concatenate([kt[:p + 1, kh], ks[p:p + 1, kh]])
            Vv = np.concatenate([vt[:p + 1, kh], vs[p:p + 1, kh]])
            sc = K @ q[p, hh] / np.sqrt(g.head_dim)
            sc = np.exp(sc - sc.max())
            att[p, hh] = (sc / sc.sum()) @ Vv
    _, lg2 = after_attention(y, att.reshape(n, -1) @ W.mat(mm + "o_proj").T)
    d1, p1 = top(lg2)
    return np.stack([d0, p0, d1, p1], axis=1)


def run(W: Weights, tokens, verbose: bool = False):
    """The logits, and the normed hidden rows the output head reads."""
    """Layer by layer over all the tokens (each matrix dequantized once); the recurrence and the attention in order."""
    g, p = W.g, W.p
    eps = g.eps
    n = len(tokens)
    x = W.f(p + "embeddings.weight")[np.array(tokens)]      # [n][h]
    hpg = g.m_heads // g.n_groups
    G = g.d_inner // g.n_groups
    for l, kind in enumerate(g.kinds):
        b = f"{p}layers.{l}."
        h = np.stack([rms(r, W.f(b + "norm.weight"), eps) for r in x])
        m = b + "mixer."
        if kind == "mamba":
            zxbcdt = h @ W.mat(m + "in_proj").T                  # [n][pw]
            z = zxbcdt[:, :g.d_inner]
            xbc_raw = zxbcdt[:, g.d_inner:g.d_inner + g.conv_dim]
            dtr = zxbcdt[:, g.d_inner + g.conv_dim:]
            cw = W.f(m + "conv1d.weight")[:, 0, :]               # [c][k], tap k-1 the current token
            padded = np.concatenate([np.zeros((g.d_conv - 1, g.conv_dim)), xbc_raw], axis=0)
            conv = sum(cw[:, k] * padded[k:k + n] for k in range(g.d_conv)) + W.f(m + "conv1d.bias")
            xbc = silu(conv)
            dt = softplus(dtr + W.f(m + "dt_bias"))              # [n][heads]
            A = -np.exp(W.f(m + "A_log"))
            D = W.f(m + "D")
            gi = np.arange(g.m_heads) // hpg
            S_ = np.zeros((g.m_heads, g.m_head_dim, g.d_state))
            ys = np.empty((n, g.d_inner))
            for t in range(n):
                xs = xbc[t, :g.d_inner].reshape(g.m_heads, g.m_head_dim)
                Bm = xbc[t, g.d_inner:g.d_inner + g.n_groups * g.d_state].reshape(g.n_groups, g.d_state)[gi]
                Cm = xbc[t, g.d_inner + g.n_groups * g.d_state:].reshape(g.n_groups, g.d_state)[gi]
                dA = np.exp(dt[t] * A)
                S_ = S_ * dA[:, None, None] + (dt[t][:, None] * xs)[:, :, None] * Bm[:, None, :]
                ys[t] = (np.einsum("hpn,hn->hp", S_, Cm) + D[:, None] * xs).reshape(-1)
            v = (ys * silu(z)).reshape(n, g.n_groups, G)
            v = v / np.sqrt(np.mean(v * v, axis=2, keepdims=True) + eps)
            v = v.reshape(n, -1) * W.f(m + "norm.weight")
            o = v @ W.mat(m + "out_proj").T
        elif kind == "attention":
            o = attention(W, m, h)
        else:
            o = moe(W, m, h)
        x = x + o
        W.cache.clear()
        if verbose:
            print(f"layer {l} ({kind}) done", flush=True)
    hn = np.stack([rms(r, W.f(p + "norm_f.weight"), eps) for r in x])
    return hn @ W.mat(W.head).T, hn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verbose", action="store_true", help="a line per layer")
    a = ap.parse_args()
    tokens = [int(t) for t in a.tokens.split(",")]
    W = Weights(pathlib.Path(a.checkpoint))
    lg, hn = run(W, tokens, a.verbose)
    with open(a.out, "wb") as f:
        f.write(struct.pack("<ii", len(tokens), lg.shape[1]))
        f.write(np.array(tokens, "<i4").tobytes())
        f.write(lg.astype("<f4").tobytes())
    if W.mtp:
        d = mtp_drafts(W, tokens, hn)
        with open(a.out + ".mtp", "wb") as f:
            f.write(struct.pack("<i", len(d)))
            f.write(d.astype("<f4").tobytes())
        print(f"MTP head: first drafts agree with the next token at {int((d[:-1, 0] == np.array(tokens[2:])).sum())} of {len(d) - 1}")
    top = lg.argmax(axis=1)
    print(f"{len(tokens)} tokens, vocab {lg.shape[1]}; logits std {lg.std():.3f}; top-1 of the last rows: {top[-8:].tolist()}")


if __name__ == "__main__":
    main()
