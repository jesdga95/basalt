"""Hy4-flash (architecture `hy4flash`; GGUF architecture `hy_v4_flash`), the test pack from AngelSlim's Q4_K_M GGUF.

Every tensor keeps the GGUF's type: the float tensors in `dense`, the Q8_0 projections in `gguf`, the routed experts (Q4_K
gate and up, Q6_K or Q4_K down) in `experts` for the layers that have them (layer 0 is dense, so the layout lists layers
1 .. N-1 by layer id), and the Engram table (`per_layer_token_embd.weight`, rows of 96 F8_E4M3 values, llama.cpp's type 43)
in a section of its own so it can be read by row with O_DIRECT. The GGUF's next-n block (the last layer, BF16) is the
drafter: `mtp/dense` and `mtp/experts`, laid out like the main sections. The vision tower comes as `--mmproj` (the section `vision/mmproj`, read in place by src/models/hy4flash/vision_encoder.cpp);
the DFly2 drafter comes from the Hugging Face checkpoint later; meta.dfly2 is null until then. The Hugging Face directory
(`--hf-dir`) supplies the LICENSE text and config.json (meta.hf_config), which carries what the GGUF omits (the MRoPE
sections, the vision tower's geometry, the DFly2 settings).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import Arch, register

ENGRAM_TENSOR = "per_layer_token_embd.weight"
ENGRAM_SECTION = "engram"
ENGRAM_LAYERS = [1, 11, 21, 31]
F8_E4M3_RAW = 43              # llama.cpp's F8_E4M3: one E4M3 code per value, no scale
P = "hy_v4_flash."
# GGUF keys of the Engram hash and row layout, copied to meta.engram (the token map stays in gguf_kv)
ENGRAM_KEYS = (
    "ple.layers", "ple.ngram_size", "ple.heads_per_ngram", "ple.layer_multipliers", "ple.head_offsets",
    "ple.head_vocab_sizes", "embedding_length_per_layer_input", "engram.pad_token_id",
)
# keys not copied to meta.geometry: the Engram ones above and the token map (all stay in gguf_kv)
NOT_GEOMETRY = frozenset(ENGRAM_KEYS) | {"engram.token_map"}


class Hy4Flash(Arch):
    name = "hy4flash"
    gguf_architecture = "hy_v4_flash"
    not_in_dense = frozenset({ENGRAM_TENSOR})
    own_sections = {ENGRAM_TENSOR: ENGRAM_SECTION}
    side_sections = False
    mmproj_only = True            # `--mmproj` adds llama.cpp's hy_v4_flash projector GGUF (the side file of the BF16 checkpoint) as vision/mmproj
    projector_type = "hy_v4_flash"
    needs_hf_dir = True
    sampling_kv = {}
    engram_type = F8_E4M3_RAW     # the source table's type; the BF16 GGUF's is BF16 (hy4flash_bf16.py rounds it)
    needs_nextn = True            # the Q4_K_M GGUF keeps llama.cpp's next-n block (the BF16 GGUF has none)

    def mtp_layers(self, kv):
        n = int(kv[P + "block_count"])
        k = int(kv.get(P + "nextn_predict_layers", 0))
        return list(range(n - k, n))

    def license(self, kv):
        lic = {"name": "Tencent Hy License, Version 1.0"}
        if kv.get("general.quantized_by"):
            lic["quantized_by"] = kv["general.quantized_by"]
        return lic

    def hf_meta(self, hf_dir: Path) -> dict:
        """License text and config.json from the Hugging Face directory; `sources` entries for them."""
        from model_sections import ConvertError
        lic, cfg = hf_dir / "LICENSE", hf_dir / "config.json"
        for p in (lic, cfg):
            if not p.is_file():
                raise ConvertError(f"missing {p} (--hf-dir is a tencent/Hy4-flash directory)")
        conf = json.loads(cfg.read_text(encoding="utf-8"))
        if conf.get("model_type") != "hy_v4_flash":
            raise ConvertError(f"{cfg} is model_type {conf.get('model_type')!r}, not hy_v4_flash")
        rp = conf["text_config"]["rope_parameters"]
        secs = {json.dumps([v["mrope_section"], v["mrope_interleaved"]]) for v in rp.values()}
        if len(secs) != 1:
            raise ConvertError("the config's layer kinds disagree on the MRoPE sections")
        section, interleaved = json.loads(secs.pop())
        return {"license": {"name": "Tencent Hy License, Version 1.0", "text": lic.read_text(encoding="utf-8")},
                "hf_config": conf, "rope": {"mrope_section": section, "interleaved": interleaved}}

    def default_name(self, first_gguf, kv, sections_from=False):
        stem = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", Path(first_gguf).name)
        base = "Hy4-flash"
        tail = stem[len(base):].strip("-_ ") if stem.startswith(base) else ""
        return f"{base} {tail}" if tail else base

    def check(self, kv, own):
        from model_sections import ConvertError
        t = own.get(ENGRAM_TENSOR)
        if t is None:
            raise ConvertError(f"the model has no {ENGRAM_TENSOR} (the Engram table)")
        width = int(kv.get(P + "embedding_length_per_layer_input", 0))
        if t.type_id != self.engram_type or len(t.shape) != 2 or int(t.shape[0]) != width:
            raise ConvertError(f"{ENGRAM_TENSOR} is {t.type_name} {list(t.shape)}; expected rows of {width} values of type {self.engram_type}")
        sizes = kv.get(P + "ple.head_vocab_sizes")
        offs = kv.get(P + "ple.head_offsets")
        if not sizes or not offs or len(sizes) != len(offs):
            raise ConvertError("the GGUF lacks the Engram head offsets and sizes")
        if int(offs[-1]) + int(sizes[-1]) != int(t.shape[1]):
            raise ConvertError(f"the Engram heads end at row {int(offs[-1]) + int(sizes[-1])}, the table has {int(t.shape[1])}")
        n_head = int(kv[P + "ple.heads_per_ngram"]) * (int(kv[P + "ple.ngram_size"]) - 1)
        if len(sizes) != n_head * len(kv[P + "ple.layers"]):
            raise ConvertError(f"{len(sizes)} Engram heads for {len(kv[P + 'ple.layers'])} layers of {n_head}")
        if [int(x) for x in kv[P + "ple.layers"]] != ENGRAM_LAYERS:
            raise ConvertError(f"the Engram layers are {kv[P + 'ple.layers']}, expected {ENGRAM_LAYERS}")
        if self.needs_nextn and not self.mtp_layers(kv):
            raise ConvertError("the GGUF has no next-n block")

    def meta_extra(self, kv, tensors, own):
        t = own[ENGRAM_TENSOR]
        width = int(t.shape[0])
        return {
            "arch": self.name,
            "geometry": {k[len(P):]: v for k, v in kv.items() if k.startswith(P) and k[len(P):] not in NOT_GEOMETRY},
            "engram": {"section": ENGRAM_SECTION, "tensor": ENGRAM_TENSOR, "type": "F8_E4M3_RAW", "type_id": F8_E4M3_RAW,
                       "rows": int(t.shape[1]), "row_values": width, "row_bytes": width, "bytes": int(t.shape[1]) * width,
                       **{k: kv[P + k] for k in ENGRAM_KEYS if P + k in kv},
                       "token_map": "gguf_kv:" + P + "engram.token_map"},
            "vision": None,
            "dfly2": None,
        }


register(Hy4Flash(), "hy_v4_flash")
