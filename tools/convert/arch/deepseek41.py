"""DeepSeek-V4.1-Flash (GGUF architecture `deepseek41`).

Every tensor is stored as published: the float tensors in `dense`, the quantized non-expert tensors in `gguf`, the routed
experts in `experts`, and the Engram table (`per_layer_token_embd.weight`, rows of 256 IQ4_NL values) in a section of its
own so it can be read by row with O_DIRECT. There is no hc8 section. `--mmproj` adds the vision
projector GGUF as the `vision/mmproj` section, `--dspark` the DSpark drafter GGUF as `dspark/gguf` (meta.dspark).
"""
from __future__ import annotations

import re
from pathlib import Path

from . import Arch, register

ENGRAM_TENSOR = "per_layer_token_embd.weight"
ENGRAM_SECTION = "engram"
ENGRAM_ROW_VALUES = 256
IQ4_NL = 20
IQ4_NL_BLOCK = (32, 18)       # values, bytes

# The LICENSE file of the official model repository, with LF line endings.
MIT = """MIT License

Copyright (c) 2023 DeepSeek

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

P = "deepseek41."
# GGUF keys of the geometry, copied to meta.geometry under the key without the prefix
GEOMETRY_KEYS = (
    "block_count", "context_length", "embedding_length", "attention.head_count", "attention.head_count_kv",
    "attention.key_length", "attention.value_length", "attention.q_lora_rank", "attention.sliding_window",
    "attention.output_group_count", "attention.output_lora_rank", "rope.dimension_count", "rope.freq_base",
    "attention.compress_rope_freq_base", "rope.scaling.factor", "rope.scaling.original_context_length",
    "attention.layer_norm_rms_epsilon",
    "expert_count", "expert_used_count", "expert_shared_count", "expert_feed_forward_length", "expert_gating_func",
    "expert_weights_scale", "expert_weights_norm", "swiglu_clamp_exp", "swiglu_clamp_shexp",
    "attention.indexer.head_count", "attention.indexer.key_length", "attention.indexer.top_k",
    "attention.compress_ratios", "hyper_connection.count", "hyper_connection.sinkhorn_iterations",
    "hyper_connection.epsilon", "embedding_length_out", "hash_layer_count",
)
# the hash and layout constants of the Engram row lookup, copied to meta.engram (the token map stays in gguf_kv)
ENGRAM_KEYS = (
    "ple.layers", "ple.ngram_size", "ple.heads_per_ngram", "ple.layer_multipliers", "ple.head_offsets",
    "ple.head_vocab_sizes", "embedding_length_per_layer_input", "engram.vocab_size", "engram.pad_token_id",
    "engram.compressed_vocab_size",
)


class DeepSeek41(Arch):
    name = "deepseek41"
    not_in_dense = frozenset({ENGRAM_TENSOR})
    own_sections = {ENGRAM_TENSOR: ENGRAM_SECTION}
    side_sections = False
    mmproj_only = True
    dspark_ok = True
    projector_type = "deepseek41v"
    sampling_kv = {"general.sampling.temp": 1.0, "general.sampling.top_p": 1.0, "general.sampling.top_k": 0}

    def license(self, kv):
        lic = {"name": "MIT", "text": MIT}
        if kv.get("general.quantized_by"):
            lic["quantized_by"] = kv["general.quantized_by"]
        return lic

    def default_name(self, first_gguf, kv, sections_from=False):
        base = kv.get("general.name") or "DeepSeek-V4.1-Flash"
        stem = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", Path(first_gguf).name)
        tail = stem[len(base):].strip("-_ ") if stem.startswith(base) else ""
        return f"{base} {tail}" if tail else base

    def check(self, kv, own):
        from model_sections import ConvertError
        t = own.get(ENGRAM_TENSOR)
        if t is None:
            raise ConvertError(f"the model has no {ENGRAM_TENSOR} (the Engram table)")
        if t.type_id != IQ4_NL or len(t.shape) != 2 or int(t.shape[0]) != ENGRAM_ROW_VALUES:
            raise ConvertError(f"{ENGRAM_TENSOR} is {t.type_name} {list(t.shape)}; expected rows of "
                               f"{ENGRAM_ROW_VALUES} IQ4_NL values")
        want = kv.get(P + "embedding_length_per_layer_input")
        if want is not None and int(want) != ENGRAM_ROW_VALUES:
            raise ConvertError(f"embedding_length_per_layer_input is {want}, the Engram rows hold {ENGRAM_ROW_VALUES}")
        sizes = kv.get(P + "ple.head_vocab_sizes")
        if sizes and sum(int(x) for x in sizes) != int(t.shape[1]):
            raise ConvertError(f"the Engram heads hold {sum(int(x) for x in sizes)} rows, the table {int(t.shape[1])}")

    def meta_extra(self, kv, tensors, own):
        t = own[ENGRAM_TENSOR]
        vals, nbytes = IQ4_NL_BLOCK
        row_bytes = ENGRAM_ROW_VALUES // vals * nbytes
        return {
            "arch": self.name,
            "geometry": {k: kv[P + k] for k in GEOMETRY_KEYS if P + k in kv},
            "engram": {"section": ENGRAM_SECTION, "tensor": ENGRAM_TENSOR, "type": t.type_name, "type_id": t.type_id,
                       "rows": int(t.shape[1]), "row_values": ENGRAM_ROW_VALUES, "row_bytes": row_bytes,
                       "bytes": int(t.shape[1]) * row_bytes,
                       **{k: kv[P + k] for k in ENGRAM_KEYS if P + k in kv},
                       "token_map": "gguf_kv:" + P + "engram.token_map"},
        }


register(DeepSeek41())
