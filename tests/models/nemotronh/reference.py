"""The logits of a Nemotron-H NVFP4 checkpoint (Model Optimizer's export: FP8 and NVFP4 weights dequantized to their
exact values) in float64, token by token, with NVIDIA's fast path: dt = softplus(dt_raw + dt_bias) unclamped, the
norm after the gate, a sigmoid router with a selection bias, squared-ReLU experts in a latent, no positional encoding.
The engine's end-to-end parity test (model_parity.cpp) compares its logits with these.

    python3 tests/models/nemotronh/reference.py CHECKPOINT_DIR --tokens 1,2,3 --out REF.bin

REF.bin: int32 n_tokens, int32 n_vocab, the tokens (int32), then the logits after each token (float32 [n][vocab])."""
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
        self.head = ("language_model." if vl else "") + "lm_head"
        self.cache = {}

    def f(self, name) -> np.ndarray:   # BF16 / F32 values
        return self.ck.values(name).astype(np.float64)

    def mat(self, base) -> np.ndarray:   # an FP8, NVFP4 or BF16 matrix's values [rows][cols]
        if base in self.cache:
            return self.cache[base]
        _, _, dt, shape, _ = self.ck.t[base + ".weight"]
        if dt == "F8_E4M3":
            w = E4M3[self.ck.raw(base + ".weight")].reshape(shape) * float(self.ck.values(base + ".weight_scale"))
        elif dt == "U8":
            rows, half = shape
            c = self.ck.raw(base + ".weight").reshape(rows, half)
            v = np.empty((rows, 2 * half), np.float64)
            v[:, 0::2], v[:, 1::2] = E2M1[c & 15], E2M1[c >> 4]
            sc = E4M3[self.ck.raw(base + ".weight_scale")].reshape(rows, -1)
            w = v * np.repeat(sc, 16, axis=1) * float(self.ck.values(base + ".weight_scale_2"))
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


def run(W: Weights, tokens) -> np.ndarray:
    g, p = W.g, W.p
    eps = g.eps
    emb = W.f(p + "embeddings.weight")
    hpg = g.m_heads // g.n_groups
    G = g.d_inner // g.n_groups
    state = {}
    kv = {}
    out = []
    for tok in tokens:
        x = emb[tok].copy()
        for l, kind in enumerate(g.kinds):
            b = f"{p}layers.{l}."
            h = rms(x, W.f(b + "norm.weight"), eps)
            m = b + "mixer."
            if kind == "mamba":
                if l not in state:
                    state[l] = (np.zeros((g.m_heads, g.m_head_dim, g.d_state)), np.zeros((g.conv_dim, g.d_conv - 1)))
                S_, hist = state[l]
                zxbcdt = W.mat(m + "in_proj") @ h
                z, xbc_raw, dtr = zxbcdt[:g.d_inner], zxbcdt[g.d_inner:g.d_inner + g.conv_dim], zxbcdt[g.d_inner + g.conv_dim:]
                cw = W.f(m + "conv1d.weight")[:, 0, :]
                taps = np.concatenate([hist, xbc_raw[:, None]], axis=1)
                xbc = silu((cw * taps).sum(axis=1) + W.f(m + "conv1d.bias"))
                hist[:] = taps[:, 1:]
                xs = xbc[:g.d_inner].reshape(g.m_heads, g.m_head_dim)
                Bm = xbc[g.d_inner:g.d_inner + g.n_groups * g.d_state].reshape(g.n_groups, g.d_state)
                Cm = xbc[g.d_inner + g.n_groups * g.d_state:].reshape(g.n_groups, g.d_state)
                dt = softplus(dtr + W.f(m + "dt_bias"))
                A = -np.exp(W.f(m + "A_log"))
                dA = np.exp(dt * A)
                gi = np.arange(g.m_heads) // hpg
                S_[:] = S_ * dA[:, None, None] + (dt[:, None] * xs)[:, :, None] * Bm[gi][:, None, :]
                y = np.einsum("hpn,hn->hp", S_, Cm[gi]) + W.f(m + "D")[:, None] * xs
                v = y.reshape(-1) * silu(z)
                v = v.reshape(g.n_groups, G)
                v = v / np.sqrt(np.mean(v * v, axis=1, keepdims=True) + eps)
                v = v.reshape(-1) * W.f(m + "norm.weight")
                o = W.mat(m + "out_proj") @ v
            elif kind == "attention":
                q = (W.mat(m + "q_proj") @ h).reshape(g.n_head, g.head_dim)
                k = (W.mat(m + "k_proj") @ h).reshape(g.n_kv, g.head_dim)
                vv = (W.mat(m + "v_proj") @ h).reshape(g.n_kv, g.head_dim)
                kv.setdefault(l, ([], []))
                kv[l][0].append(k)
                kv[l][1].append(vv)
                Ks, Vs = np.stack(kv[l][0]), np.stack(kv[l][1])      # [cells][n_kv][hd]
                att = np.empty((g.n_head, g.head_dim))
                for hh in range(g.n_head):
                    kh = hh // (g.n_head // g.n_kv)
                    s = Ks[:, kh, :] @ q[hh] / np.sqrt(g.head_dim)
                    s = np.exp(s - s.max())
                    att[hh] = (s / s.sum()) @ Vs[:, kh, :]
                o = W.mat(m + "o_proj") @ att.reshape(-1)
            else:
                logits = W.f(m + "gate.weight") @ h
                sc = 1.0 / (1.0 + np.exp(-logits))
                choice = sc + W.f(m + "gate.e_score_correction_bias")
                ids = np.argsort(-choice, kind="stable")[:g.top_k]
                wts = sc[ids] / (sc[ids].sum() + 1e-20) * g.scale
                lat = W.mat(m + "fc1_latent_proj") @ h if g.has_latent else h
                acc = np.zeros_like(lat)
                for e, wt in zip(ids, wts):
                    u = W.mat(m + f"experts.{e}.up_proj") @ lat
                    acc += wt * (W.mat(m + f"experts.{e}.down_proj") @ np.maximum(u, 0.0) ** 2)
                o = W.mat(m + "fc2_latent_proj") @ acc if g.has_latent else acc
                su = W.mat(m + "shared_experts.up_proj") @ h
                o = o + W.mat(m + "shared_experts.down_proj") @ np.maximum(su, 0.0) ** 2
            x = x + o
        out.append(W.mat(W.head) @ rms(x, W.f(p + "norm_f.weight"), eps))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tokens = [int(t) for t in a.tokens.split(",")]
    W = Weights(pathlib.Path(a.checkpoint))
    lg = run(W, tokens)
    with open(a.out, "wb") as f:
        f.write(struct.pack("<ii", len(tokens), lg.shape[1]))
        f.write(np.array(tokens, "<i4").tobytes())
        f.write(lg.astype("<f4").tobytes())
    top = lg.argmax(axis=1)
    print(f"{len(tokens)} tokens, vocab {lg.shape[1]}; logits std {lg.std():.3f}; top-1 of the last rows: {top[-8:].tolist()}")


if __name__ == "__main__":
    main()
