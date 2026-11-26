"""GLM-5.3-Flash (GGUF architecture `glm5-next`), built from Unsloth's GGUFs.

Every tensor keeps the GGUF's type: the float tensors in `dense`, the quantized non-expert tensors in `gguf`, the routed
experts of the sparse layers in `experts` for the layers that have them (the first `leading_dense_block_count` layers are
dense, so the layout lists the sparse layers by layer id), and the GGUF's last block (the next-n layer) in `mtp/dense`,
`mtp/dense.index`, `mtp/experts` and `mtp/experts.layout`, laid out like the main sections. There are no side sections:
the vision tower comes as `--mmproj` (llama.cpp's `glm5v` projector GGUF, the side file of the release) as `vision/mmproj`.

A layer is linear-attention (KDA: `attn_q/k/v` plus the `ssm_*` parts, whose decay is per channel through the low-rank
`ssm_f_a/f_b`) when `attention.head_count_kv` gives it 0 heads, and sparse MLA with an indexer (`attn_q_a` and `attn_q_b`,
`attn_kv_a_mqa`, `attn_k_b` and `attn_v_b`, `indexer.*`, and the `indexer_compressor_*` tensors of the k-pool) when it
gives it 1. Layers at or past `leading_dense_block_count` hold routed experts. Every layer but the drafter's carries the
hyper-connection tensors (`hc_attn_*`, `hc_ffn_*`, four streams).
"""
from __future__ import annotations

from . import Arch, gguf_stem, register

# The LICENSE file of the official model repository, with LF line endings.
MIT = """MIT License

Copyright (c) 2026 Z.AI Co., Ltd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

P = "glm5-next."
# GGUF keys not copied to meta.geometry: none, the whole prefixed block is the model's geometry
NOT_GEOMETRY = frozenset()

# what every layer carries, whatever its attention is
COMMON_TENSORS = ("attn_norm.weight", "ffn_norm.weight", "attn_output.weight")
# what a linear-attention layer of this model carries beyond its projections
KDA_TENSORS = ("attn_q.weight", "attn_k.weight", "attn_v.weight",
               "ssm_a", "ssm_beta.weight", "ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight",
               "ssm_dt.bias", "ssm_f_a.weight", "ssm_f_b.weight", "ssm_g_a.weight", "ssm_g_b.weight", "ssm_norm.weight")
# what a sparse-MLA layer carries beyond its output projection: the low-rank query, the compressed key/value, the indexer
DSA_TENSORS = ("attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight", "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight",
               "attn_k_b.weight", "attn_v_b.weight", "indexer.attn_q_b.weight", "indexer.attn_k.weight", "indexer.proj.weight",
               "indexer.k_norm.weight", "indexer_compressor_gate.weight", "indexer_compressor_ape.weight")
# the hyper-connection tensors, on every layer but the drafter's
HC_TENSORS = ("hc_attn_base.weight", "hc_attn_fn.weight", "hc_attn_scale.weight",
              "hc_ffn_base.weight", "hc_ffn_fn.weight", "hc_ffn_scale.weight")
# a layer with routed experts also carries its shared experts, which the routed path adds to whatever the router picks
SHEXP_TENSORS = ("ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight")


class Glm5Next(Arch):
    name = "glm5next"
    side_sections = False     # no hc8, no --mtp-source: the drafter is the GGUF's last block
    mmproj_only = True        # `--mmproj` adds llama.cpp's glm5v projector GGUF (the side file of the release) as vision/mmproj
    projector_type = "glm5v"
    sampling_kv = {}          # the GGUF carries general.sampling.temp and general.sampling.top_p

    def mtp_layers(self, kv):
        n = int(kv[P + "block_count"])
        k = int(kv.get(P + "nextn_predict_layers", 0))
        return list(range(n - k, n)) if k else []

    def license(self, kv):
        lic = {"name": "MIT", "text": MIT}
        if kv.get("general.quantized_by"):
            lic["quantized_by"] = kv["general.quantized_by"]
        return lic

    def default_name(self, first_gguf, kv, sections_from=False):
        stem = gguf_stem(first_gguf)
        base = "GLM-5.3-Flash"
        tail = stem[len(base):].strip("-_ ") if stem.startswith(base) else ""
        return f"{base} {tail}" if tail else base

    def check(self, kv, own):
        """The invariants the GGUF's own metadata can prove; the tensors each layer kind needs are checked in `meta_extra`,
        the first hook that sees them, so a source of the wrong shape is refused rather than converted into a bad pack."""
        from model_sections import ConvertError
        n = int(kv[P + "block_count"])
        kinds = kv.get(P + "attention.head_count_kv")
        if not isinstance(kinds, list) or len(kinds) != n:
            raise ConvertError(f"attention.head_count_kv lists {len(kinds) if kinds else 0} layers, the model has {n}")
        if any(int(x) not in (0, 1) for x in kinds):
            raise ConvertError(f"attention.head_count_kv holds {sorted({int(x) for x in kinds})}; a layer is "
                               f"linear-attention (0) or sparse MLA (1)")
        dense = int(kv.get(P + "leading_dense_block_count", 0))
        if not 0 <= dense < n:
            raise ConvertError(f"leading_dense_block_count is {dense} of {n} layers")
        mtp = self.mtp_layers(kv)
        if not mtp:
            raise ConvertError("the GGUF has no next-n block (the drafter)")
        if len(mtp) != 1 or mtp[0] != n - 1:
            raise ConvertError(f"the drafter is block(s) {mtp}; expected the last block, {n - 1}")
        if int(kv.get(P + "kda.head_dim", 0)) == 0 or int(kv.get(P + "ssm.conv_kernel", 0)) == 0:
            raise ConvertError("the GGUF lacks kda.head_dim or ssm.conv_kernel")
        if int(kv[P + "expert_used_count"]) > int(kv[P + "expert_count"]):
            raise ConvertError(f"the router keeps {kv[P + 'expert_used_count']} of {kv[P + 'expert_count']} experts")
        if int(kv.get(P + "attention.indexer.kpool", 0)) == 0:
            raise ConvertError("the GGUF lacks attention.indexer.kpool (the indexer's key pooling)")
        if int(kv.get(P + "hyper_connection.count", 0)) == 0:
            raise ConvertError("the GGUF lacks hyper_connection.count (the number of hyper-connection streams)")

    def meta_extra(self, kv, tensors, own):
        from model_sections import ConvertError
        n = int(kv[P + "block_count"])
        kinds = [int(x) for x in kv[P + "attention.head_count_kv"]]
        dense = int(kv.get(P + "leading_dense_block_count", 0))
        shared = int(kv.get(P + "expert_shared_count", 0))
        mtp = set(self.mtp_layers(kv))
        names = set(tensors)
        m = lambda l, s: f"blk.{l}.{s}"
        for l in range(n):
            if l in mtp:
                # The drafter block is not a model layer. Its own tensors go to the mtp sections, and the ones the
                # converter stores there never reach the set this hook is given, so its contract is the mtp writer's
                # and the tests' - asking for them here would refuse a source that converts fine.
                continue
            kind = "linear-attention" if kinds[l] == 0 else "sparse MLA"
            for s in (*COMMON_TENSORS, *(KDA_TENSORS if kinds[l] == 0 else DSA_TENSORS)):
                if m(l, s) not in names:
                    raise ConvertError(f"layer {l} is {kind} but has no {m(l, s)}")
            if l >= dense and shared > 0:
                for s in SHEXP_TENSORS:
                    if m(l, s) not in names:
                        raise ConvertError(f"layer {l} routes to experts and the model has {shared} shared expert(s), "
                                           f"but it has no {m(l, s)}")
            for s in HC_TENSORS:
                if m(l, s) not in names:
                    raise ConvertError(f"layer {l} has no {m(l, s)} (the hyper-connection)")
        return {
            "arch": self.name,
            "geometry": {k[len(P):]: v for k, v in kv.items() if k.startswith(P) and k[len(P):] not in NOT_GEOMETRY},
        }


register(Glm5Next(), "glm5-next")
