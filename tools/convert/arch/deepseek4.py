"""DeepSeek V4 Flash Vision GGUF architecture and container rules."""
from __future__ import annotations

import math
import json
import hashlib
from pathlib import Path

from . import Arch, register
from .deepseek41 import MIT

_REQUIRED = (
    "block_count", "context_length", "embedding_length", "attention.head_count", "attention.head_count_kv",
    "attention.key_length", "attention.q_lora_rank", "attention.sliding_window", "attention.output_group_count",
    "attention.output_lora_rank",
    "attention.layer_norm_rms_epsilon", "attention.compress_rope_freq_base", "attention.compress_ratios",
    "rope.dimension_count", "rope.freq_base", "rope.scaling.factor", "rope.scaling.original_context_length",
    "rope.scaling.yarn_beta_fast", "rope.scaling.yarn_beta_slow", "expert_count", "expert_used_count",
    "expert_shared_count", "expert_feed_forward_length", "expert_gating_func", "expert_weights_scale",
    "expert_weights_norm", "hyper_connection.count", "hyper_connection.sinkhorn_iterations",
    "hyper_connection.epsilon", "embedding_length_out",
    "hash_layer_count", "attention.indexer.head_count", "attention.indexer.key_length", "attention.indexer.top_k",
)
_SUPPORTED_GEOMETRY = {
    "block_count": 43,
    "context_length": 1048576,
    "embedding_length": 4096,
    "attention.head_count": 64,
    "attention.head_count_kv": 1,
    "attention.key_length": 512,
    "attention.q_lora_rank": 1024,
    "attention.sliding_window": 128,
    "attention.output_group_count": 8,
    "attention.output_lora_rank": 1024,
    "rope.dimension_count": 64,
    "rope.freq_base": 10000,
    "rope.scaling.factor": 16,
    "rope.scaling.original_context_length": 65536,
    "rope.scaling.yarn_beta_fast": 32,
    "rope.scaling.yarn_beta_slow": 1,
    "expert_count": 256,
    "expert_used_count": 6,
    "expert_shared_count": 1,
    "expert_feed_forward_length": 2048,
    "expert_gating_func": 4,
    "expert_weights_scale": 1.5,
    "hyper_connection.count": 4,
    "hyper_connection.sinkhorn_iterations": 20,
    "embedding_length_out": 16384,
    "hash_layer_count": 3,
    "attention.indexer.head_count": 64,
    "attention.indexer.key_length": 128,
    "attention.indexer.top_k": 512,
}


class DeepSeek4(Arch):
    name = "deepseek4"
    needs_hf_dir = True
    side_sections = False
    mmproj_only = True
    projector_type = "deepseek4v"
    sampling_kv = {"general.sampling.temp": 1.0, "general.sampling.top_p": 1.0, "general.sampling.top_k": 0}

    def hf_sections(self, hf_dir):
        from model_sections import ConvertError, read_safetensors_header

        root = Path(hf_dir)
        try:
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ConvertError(f"{root}: invalid DeepSeek V4 checkpoint config or safetensors index: {exc}") from exc
        if (config.get("model_type") != "deepseek_v4" or config.get("hidden_size") != 4096 or
                config.get("hc_mult") != 4 or config.get("num_hidden_layers") != 43):
            raise ConvertError(f"{root}: checkpoint is not the supported DeepSeek V4-0731 geometry")

        expected = {
            "hc_head_fn": ("hc_head_fn.weight", [4, 16384], [16384, 4]),
            "hc_head_base": ("hc_head_base.weight", [4], [4]),
            "hc_head_scale": ("hc_head_scale.weight", [1], [1]),
        }
        records = []
        for source_name, (name, source_shape, pack_shape) in expected.items():
            try:
                shard_name = weight_map[source_name]
                shard = root / shard_name
                header, data_start = read_safetensors_header(shard)
                tensor = header[source_name]
                start, end = tensor["data_offsets"]
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise ConvertError(f"{root}: missing or invalid checkpoint tensor {source_name}: {exc}") from exc
            size = 4 * math.prod(source_shape)
            if tensor.get("dtype") != "F32" or tensor.get("shape") != source_shape or end - start != size:
                raise ConvertError(f"{shard}: {source_name} is {tensor.get('dtype')} {tensor.get('shape')}, "
                                   f"expected F32 {source_shape}")
            with shard.open("rb") as f:
                f.seek(data_start + start)
                raw = f.read(size)
            if len(raw) != size:
                raise ConvertError(f"{shard}: truncated checkpoint tensor {source_name}")
            records.append({"name": name, "source": str(shard), "offset": data_start + start, "bytes": size,
                            "type": "F32", "type_id": 0, "shape": pack_shape,
                            "source_sha256": hashlib.sha256(raw).hexdigest()})
        return {"deepseek4/head": records}

    def license(self, kv):
        lic = {"name": "MIT", "text": MIT}
        if kv.get("general.quantized_by"):
            lic["quantized_by"] = kv["general.quantized_by"]
        return lic

    def default_name(self, first_gguf, kv, sections_from=False):
        return kv.get("general.name") or "DeepSeek-V4-Flash-Vision"

    def check(self, kv, own):
        from model_sections import ConvertError
        prefix = self.name + "."
        for key in _REQUIRED:
            if prefix + key not in kv:
                raise ConvertError(f"missing required {prefix}{key} metadata")

        def number(key):
            value = kv[prefix + key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConvertError(f"{prefix}{key} must be numeric")
            if not math.isfinite(value):
                raise ConvertError(f"{prefix}{key} must be finite")
            return float(value)

        def positive(key):
            value = number(key)
            if not value > 0:
                raise ConvertError(f"{prefix}{key} must be positive")
            return value

        for key, expected in _SUPPORTED_GEOMETRY.items():
            if number(key) != expected:
                raise ConvertError(f"unsupported DeepSeek V4 geometry: {prefix}{key} must be {expected}")
        layers = _SUPPORTED_GEOMETRY["block_count"]
        for key in (
            "context_length", "embedding_length", "attention.head_count", "attention.key_length",
            "attention.q_lora_rank", "attention.sliding_window", "attention.output_group_count",
            "attention.output_lora_rank", "attention.layer_norm_rms_epsilon",
            "rope.dimension_count", "rope.freq_base", "attention.compress_rope_freq_base",
            "rope.scaling.factor", "rope.scaling.original_context_length", "rope.scaling.yarn_beta_fast",
            "rope.scaling.yarn_beta_slow",
            "expert_count", "expert_shared_count", "expert_feed_forward_length", "attention.indexer.head_count",
            "attention.indexer.key_length", "attention.indexer.top_k", "hyper_connection.epsilon",
        ):
            positive(key)

        if number("attention.head_count_kv") != 1 or number("hyper_connection.count") != 4:
            raise ConvertError("deepseek4 metadata is outside the supported MLA/mHC geometry")
        if number("expert_used_count") != 6 or number("hash_layer_count") != 3 or number("expert_gating_func") != 4 \
                or number("expert_weights_scale") != 1.5 or kv[prefix + "expert_weights_norm"] is not True:
            raise ConvertError("deepseek4 routing metadata is outside the top-6 hash-routing contract")
        if number("expert_used_count") > number("expert_count") or number("hash_layer_count") > layers:
            raise ConvertError("deepseek4 metadata has an invalid expert or hash-layer count")
        if number("rope.scaling.factor") < 1 or number("embedding_length_out") != 4 * number("embedding_length"):
            raise ConvertError("deepseek4 metadata is outside the supported RoPE or mHC output geometry")

        ratios = kv[prefix + "attention.compress_ratios"]
        if not isinstance(ratios, (list, tuple)) or len(ratios) != layers + 3:
            raise ConvertError("deepseek4.attention.compress_ratios must cover transformer layers and three MTP slots")
        if any(type(ratio) is not int or ratio not in (0, 4, 128) for ratio in ratios[:layers]):
            raise ConvertError("deepseek4.attention.compress_ratios contains an unsupported layer ratio")
        if any(type(ratio) is not int or ratio != 0 for ratio in ratios[layers:]):
            raise ConvertError("deepseek4 MTP compression ratios must be zero")

    def meta_extra(self, kv, tensors, own):
        prefix = self.name + "."
        return {
            "arch": self.name,
            "geometry": {k[len(prefix):]: v for k, v in kv.items() if k.startswith(prefix)},
        }


register(DeepSeek4())
