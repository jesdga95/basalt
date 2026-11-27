"""A whole qwen35moe pack (Qwen3.5-family hybrid: GDN and gated attention layers, a routed-expert SwiGLU FFN with a gated
shared expert) from a BF16 checkpoint, for tools/convert/convert.py. The layers, norms, head and embedding follow
checkpoint_qwen35.py; what is added:

  experts / experts.layout       the routed experts rounded once to NVFP4 (ggml type 40), the 27B pack's rounding: per
                                 matrix the tensor scale amax / (6 x 448), per 16 values the UE4M3 scale of least
                                 squared error near amax / 6; blobs [gate | up | down] plus the FP32 tail
                                 {s_gate, s_up, s_down, 0}, zero-padded to a multiple of 144 bytes, layers in order
  mtp/experts / mtp/experts.layout   the same for the next-token block's experts (layout layer id = block_count)
  gguf                           also ffn_gate_inp (the router, BF16), ffn_{gate,up,down}_shexp (Q8_0) and
                                 ffn_gate_inp_shexp (the shared expert's gate, BF16)

The checkpoint stores a layer's experts fused: `mlp.experts.gate_up_proj` [experts, 2 ff, hidden] with the gate rows first,
`mlp.experts.down_proj` [experts, hidden, ff]. Needs numpy.
"""
from __future__ import annotations

import numpy as np

import checkpoint as C
import checkpoint_qwen35 as Q
import fp4q_cuda
import model_sections as S

ARCH = "qwen35moe"
EXPERTS_PER_JOB = 4
BLOB_ALIGN = 144   # lcm(36, 16): MMQ strides experts in whole 36-byte blocks, and reads them 16-byte aligned
MTP_PREFIX = "mtp.layers.0."
LAYER_PREFIX = "model.language_model.layers."

# the dense plan's rules without its FFN, plus the MoE FFN's tensors (the routed experts are not a plan tensor: see Plan.experts)
MOE_RULES = {k: v for k, v in Q.LAYER.items() if not k.startswith("mlp.")}
MOE_RULES.update({
    "mlp.gate.weight": ("ffn_gate_inp.weight", "BF16", "id"),
    "mlp.shared_expert.gate_proj.weight": ("ffn_gate_shexp.weight", "Q8", "id"),
    "mlp.shared_expert.up_proj.weight": ("ffn_up_shexp.weight", "Q8", "id"),
    "mlp.shared_expert.down_proj.weight": ("ffn_down_shexp.weight", "Q8", "id"),
    "mlp.shared_expert_gate.weight": ("ffn_gate_inp_shexp.weight", "BF16", "id"),
})
EXPERT_KEYS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")
# This checkpoint's tokenizer.json splits letters as \p{L}+, the official Qwen3.5 files as [\p{L}\p{M}]+. The engine runs the
# `qwen35` split (llama.cpp does too for this model), so the two differ only on combining marks.
CHECKPOINT_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
                    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def _expert_job(args):
    """Blobs of experts [e0, e1) of one layer: each [gate | up | down] NVFP4 blocks and the FP32 scale tail."""
    gu_path, gu_off, dn_path, dn_off, ff, hidden, e0, e1, pad = args
    out = []
    for e in range(e0, e1):
        mats = ((gu_path, gu_off + (2 * e) * ff * hidden * 2, ff, hidden),
                (gu_path, gu_off + (2 * e + 1) * ff * hidden * 2, ff, hidden),
                (dn_path, dn_off + e * hidden * ff * 2, hidden, ff))
        tail = []
        for path, off, rows, cols in mats:
            u = np.fromfile(path, dtype="<u2", count=rows * cols, offset=off)
            x = S._bf16_to_f32(u).reshape(rows, cols)
            if not np.isfinite(x).all():
                return f"expert {e}: a non-finite value"
            gs = Q.fp4q_global(float(np.abs(x).max()))
            done = fp4q_cuda.rows_bf16(u, rows, cols, gs)
            codes, sc = done if done is not None else Q.fp4q_rows(x, gs)
            out.append(C.nvfp4_rows(codes, sc).tobytes())
            tail.append(gs)
        out.append(np.array(tail + [0.0], dtype="<f4").tobytes())
        out.append(bytes(pad))
    return b"".join(out)


class MoePlan(Q.Plan):
    ARCH = ARCH
    TEXT_TYPE = "qwen3_5_moe_text"
    RULES = MOE_RULES
    V_HEADS = 32
    DENSE = False

    def keys(self, ck):
        """Every language-model tensor but the fused experts, the next-token block's (in its own file) after the layers."""
        mtp = [k for k in ck.t if k.startswith("mtp.")]
        return [k for k in ck.order + mtp if not k.rpartition(".mlp.")[2].startswith("experts")]

    def __init__(self, ck, head=None):
        tc = ck.text
        self.experts = {}      # layer id (block_count for the next-token block) -> (gate_up_proj, down_proj) tensor names
        self.n_expert, self.ff, self.hidden = int(tc["num_experts"]), int(tc["moe_intermediate_size"]), int(tc["hidden_size"])
        if int(tc["num_experts_per_tok"]) > self.n_expert or int(tc.get("shared_expert_intermediate_size", 0)) != self.ff:
            raise S.ConvertError("config.json: the routed and shared experts differ in width, which the pack does not hold")
        n_layers = int(tc["num_hidden_layers"])
        for layer, prefix in [(l, f"{LAYER_PREFIX}{l}.") for l in range(n_layers)] + \
                ([(n_layers, MTP_PREFIX)] if int(tc.get("mtp_num_hidden_layers", 0)) == 1 else []):
            names = tuple(prefix + k for k in EXPERT_KEYS)
            ck.need(names[0], "BF16", (self.n_expert, 2 * self.ff, self.hidden))
            ck.need(names[1], "BF16", (self.n_expert, self.hidden, self.ff))
            self.experts[layer] = names
        super().__init__(ck, head)
        if self.ff % 64 or self.hidden % 64:
            raise S.ConvertError("NVFP4 expert rows must be whole 64-value blocks")
        mats = (self.ff, self.hidden), (self.ff, self.hidden), (self.hidden, self.ff)
        raw = sum(r * S.nvfp4_row_bytes(c) for r, c in mats) + S.NVFP4_TAIL
        self.blob = -(-raw // BLOB_ALIGN) * BLOB_ALIGN
        self.pad = self.blob - raw
        self.n_layers = n_layers

    def make_kv(self, name):
        tc = self.ck.text
        kv = Q.model_kv(self.ck, name, ARCH, CHECKPOINT_SPLIT)
        A = ARCH + "."
        kv.pop(A + "feed_forward_length", None)
        kv.update({
            A + "expert_count": self.n_expert,
            A + "expert_used_count": int(tc["num_experts_per_tok"]),
            A + "expert_feed_forward_length": self.ff,
            A + "expert_shared_feed_forward_length": int(tc["shared_expert_intermediate_size"]),
        })
        return kv

    def _chunks(self, pool, layers):
        def jobs():
            for l in layers:
                gu, dn = self.experts[l]
                gp, goff = self.ck.t[gu][:2]
                dp, doff = self.ck.t[dn][:2]
                for e in range(0, self.n_expert, EXPERTS_PER_JOB):
                    yield (str(gp), goff, str(dp), doff, self.ff, self.hidden, e, min(e + EXPERTS_PER_JOB, self.n_expert), self.pad)
        for b in S.bounded_imap(pool, _expert_job, jobs(), 4 * 24):
            if isinstance(b, str):
                raise S.ConvertError(b)
            yield b

    def _layout(self, layers):
        total = len(layers) * self.n_expert * self.blob
        rows = [(l, S.NVFP4, S.NVFP4, i * self.n_expert * self.blob, self.blob) for i, l in enumerate(layers)]
        return total, S.nvfp4_layout_text(self.n_expert, total, rows)

    def write_sections(self, job, w, pool):
        main, mtp = list(range(self.n_layers)), [l for l in self.experts if l >= self.n_layers]
        for name, layers in (("experts", main), ("mtp/experts", mtp)):
            if not layers:
                continue
            total, layout = self._layout(layers)
            job.chunks(name, self._chunks(pool, layers), total)
            if w.entries[-1][2] != total:
                raise S.ConvertError(f"{name}: {w.entries[-1][2]} bytes written, {total} planned")
            job.bytes(name + ".layout", layout)
        return {"experts": {
            "format": "nvfp4", "count": self.n_expert, "used": int(self.ck.text["num_experts_per_tok"]), "ff": self.ff,
            "blob_bytes": self.blob, "layers": len(main), "mtp_layers": len(mtp),
            "source": "the checkpoint's BF16 mlp.experts.gate_up_proj (gate rows, then up) and down_proj, rounded once: per "
                      "matrix the tensor scale amax / (6 x 448), per 16 values the UE4M3 scale of least squared error among "
                      "half to four times amax / 6, E2M1 to nearest"}}


def build(a, Job, fingerprint, die) -> None:
    Q.build(a, Job, fingerprint, die, MoePlan)
