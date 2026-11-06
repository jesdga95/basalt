"""Flash-Next (and any architecture without its own entry)."""
from __future__ import annotations

from pathlib import Path

from . import Arch, register

LICENSE_TEXT = (Path(__file__).resolve().parent.parent / "qwen-community-license-1.0.txt").read_text(encoding="utf-8")

# float form the engine reads each dense tensor in; other forms are converted and recorded in meta.conversions:
#   F32 -> BF16 when exactly BF16 (low 16 bits zero)              exact, by default
#   other F32, F16 or quantized -> BF16                           rounds (nearest-even): only with --compat-bf16
#   F32 -> F16 (ple_conv1d)                                       index kind 3, narrowed by the loader
#   F16 / BF16 -> F32                                             exact widening
# tensors not listed are written as stored (floats) or served from the `gguf` section (quantized)
FORM = {
    "ffn_gate_inp.weight": "BF16", "ffn_gate_inp_shexp.weight": "BF16",
    "hc_attn_down.weight": "BF16", "hc_attn_up.weight": "BF16", "hc_attn_inject.weight": "BF16",
    "hc_ffn_down.weight": "BF16", "hc_ffn_up.weight": "BF16", "hc_ffn_inject.weight": "BF16",
    "output_hc_down.weight": "BF16", "output_hc_up.weight": "BF16",
    "indexer.k_proj.weight": "BF16", "indexer.q_proj.weight": "BF16",
    "ssm_alpha.weight": "BF16", "ssm_beta.weight": "BF16",
    "ple_key.weight": "BF16", "ple_value.weight": "BF16", "ple_conv1d.weight": "F16",
    "attn_q_norm.weight": "F32", "attn_k_norm.weight": "F32", "hc_attn_norm.weight": "F32",
    "hc_ffn_norm.weight": "F32", "output_hc_norm.weight": "F32", "indexer.q_norm.weight": "F32",
    "indexer.k_norm.weight": "F32", "ple_norm_conv.weight": "F32", "ple_norm_key.weight": "F32",
    "ple_norm_query.weight": "F32", "ssm_a": "F32", "ssm_conv1d.weight": "F32", "ssm_dt.bias": "F32",
    "ssm_norm.weight": "F32",
}
# PLE key encodings the engine reads natively; other quantized keys need --compat-bf16
NATIVE_PLE_KEY = {"Q2_0", "Q8_0"}


class Qwen4Exp(Arch):
    name = "qwen4exp"
    form_table = FORM
    # the PLE table comes from the `gguf` section, never the dense pool
    not_in_dense = frozenset({"per_layer_token_embd.weight"})
    sampling_kv = {"general.sampling.temp": 1.0, "general.sampling.top_p": 0.95, "general.sampling.top_k": 20}

    def native(self, name, type_name):
        return self.form_of(name) is None or (name == "blk.1.ple_key.weight" and type_name in NATIVE_PLE_KEY)

    def license(self, kv):
        """The model's license, with its full text (the license requires it in every copy)."""
        lic = {"name": "Qwen Community License 1.0", "text": LICENSE_TEXT}
        if kv.get("general.quantized_by"):
            lic["quantized_by"] = kv["general.quantized_by"]
        return lic

    def default_name(self, first_gguf, kv, sections_from=False):
        if sections_from:
            return "Qwen3.8-Flash-Next " + Path(first_gguf).name.split("Next-")[-1].split("-0000")[0]
        return "Qwen3.8-Flash-Next IQ3_XXS"


register(Qwen4Exp())
