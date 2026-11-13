"""A whole qwen35 pack (Qwen3.5-family dense hybrid: GDN and gated attention layers, dense SwiGLU) straight from an
NVIDIA Model Optimizer checkpoint (as nvidia/Qwen3.8-27B-NVFP4), for tools/convert/convert.py.

  gguf                every language-model and drafter tensor in GGUF names:
                        NVFP4 MLP and head: E2M1 codes and UE4M3 scales moved into Basalt's 64-value blocks
                          (type 40, unchanged values), the FP32 tensor scale as `<name>.scale` (F32 [1])
                        FP8 projections: code x per-tensor scale in f32, then Q8_0 (one rounding)
                        BF16 projections (the drafter's): Q8_0
                        norms F32 with the offset "+1" folded in (exact), the GDN gated norm as stored
                        GDN value heads in the GGUF order (value head h meets key head h % 16)
                        a / b projections, token embedding: BF16 as stored
  vision/mmproj       the vision tower as llama.cpp's clip GGUF (projector qwen3vl_merger), BF16 / F32 as stored
  vision/text_vocab   the text model's metadata alone, for the vision sidecar's tokenizer
  tokenizer/*, meta   from config.json, generation_config.json, the tokenizer files and the model card

Needs numpy.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import pathlib
import re
import sys
import tempfile
import time

import numpy as np

import checkpoint as C
import model_sections as S

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from gguf_writer import GGUFWriter  # noqa: E402
from model_file import TENSOR_ALIGN, ModelWriter, Stream  # noqa: E402

ARCH = "qwen35"
TYPE_ID = {"F32": 0, "F16": 1, "Q8_0": 8, "BF16": 30, "NVFP4": 40}
ROWS_PER_JOB = 1 << 22
E4M3 = S.fp8_e4m3_table()

# checkpoint tensor (after `model.language_model.layers.N.` or `mtp.layers.0.`) -> (GGUF name, form, transform)
#   forms: Q8 (Q8_0 from FP8 or BF16), FP4 (NVFP4 kept), BF16, F32
#   transforms as checkpoint.py names them: id, norm, heads, negexp, gconv, vrows, zrows, vcols
LAYER = {
    "input_layernorm.weight": ("attn_norm.weight", "F32", "norm"),
    "post_attention_layernorm.weight": ("post_attention_norm.weight", "F32", "norm"),
    "linear_attn.in_proj_qkv.weight": ("attn_qkv.weight", "Q8", "vrows"),
    "linear_attn.in_proj_z.weight": ("attn_gate.weight", "Q8", "zrows"),
    "linear_attn.out_proj.weight": ("ssm_out.weight", "Q8", "vcols"),
    "linear_attn.in_proj_a.weight": ("ssm_alpha.weight", "BF16", "heads"),
    "linear_attn.in_proj_b.weight": ("ssm_beta.weight", "BF16", "heads"),
    "linear_attn.A_log": ("ssm_a", "F32", "negexp"),
    "linear_attn.dt_bias": ("ssm_dt.bias", "F32", "heads"),
    "linear_attn.conv1d.weight": ("ssm_conv1d.weight", "F32", "gconv"),
    "linear_attn.norm.weight": ("ssm_norm.weight", "F32", "id"),
    "self_attn.q_proj.weight": ("attn_q.weight", "Q8", "id"),
    "self_attn.k_proj.weight": ("attn_k.weight", "Q8", "id"),
    "self_attn.v_proj.weight": ("attn_v.weight", "Q8", "id"),
    "self_attn.o_proj.weight": ("attn_output.weight", "Q8", "id"),
    "self_attn.q_norm.weight": ("attn_q_norm.weight", "F32", "norm"),
    "self_attn.k_norm.weight": ("attn_k_norm.weight", "F32", "norm"),
    "mlp.gate_proj.weight": ("ffn_gate.weight", "FP4", "id"),
    "mlp.up_proj.weight": ("ffn_up.weight", "FP4", "id"),
    "mlp.down_proj.weight": ("ffn_down.weight", "FP4", "id"),
}
GLOBAL = {
    "model.language_model.embed_tokens.weight": ("token_embd.weight", "BF16", "id"),
    "model.language_model.norm.weight": ("output_norm.weight", "F32", "norm"),
    "lm_head.weight": ("output.weight", "FP4", "id"),
    "mtp.fc.weight": ("mtp.fc.weight", "Q8", "id"),
    "mtp.norm.weight": ("mtp.norm.weight", "F32", "norm"),
    "mtp.pre_fc_norm_embedding.weight": ("mtp.pre_fc_norm_embedding.weight", "F32", "norm"),
    "mtp.pre_fc_norm_hidden.weight": ("mtp.pre_fc_norm_hidden.weight", "F32", "norm"),
}
QUANT_SIDE = re.compile(r"\.(weight_scale|weight_scale_2|input_scale)$")
METHODS = dict(C.METHODS, id="as stored")


class Tensor:
    def __init__(self, name, form, src, how, ne):
        self.name, self.form, self.src, self.how, self.ne = name, form, src, how, ne
        self.type_name = {"Q8": "Q8_0", "FP4": "NVFP4"}.get(form, form)
        self.type_id = TYPE_ID[self.type_name]

    def expected_bytes(self) -> int:
        n = int(np.prod(self.ne))
        return {"F32": 4 * n, "BF16": 2 * n, "Q8_0": n // 32 * 34, "NVFP4": n // 64 * 36}[self.type_name]


class Plan:
    """The pack's `gguf` tensors in checkpoint order, each with how it is made."""

    def __init__(self, ck: C.Checkpoint):
        tc = ck.text
        self.ck = ck
        if tc.get("model_type") != "qwen3_5_text":
            raise S.ConvertError(f"config.json: text model_type {tc.get('model_type')}, expected qwen3_5_text")
        for k in ("num_experts", "hc_count"):
            if k in tc:
                raise S.ConvertError(f"config.json: {k} is not a dense qwen3_5 field")
        if int(tc["linear_num_key_heads"]) != S.GDN_K_HEADS or int(tc["linear_num_value_heads"]) != S.GDN_V_HEADS \
                or int(tc["linear_key_head_dim"]) != S.GDN_HEAD or int(tc["linear_value_head_dim"]) != S.GDN_HEAD:
            raise S.ConvertError("the GDN head geometry differs from the GGUF head order this converter knows")
        self.tensors: dict[str, Tensor] = {}
        self.scales: dict[str, str] = {}      # NVFP4 tensor -> its weight_scale_2 source
        unknown = []
        for k in ck.order:
            if k.startswith("model.visual.") or QUANT_SIDE.search(k):
                continue
            m = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.(.+)", k)
            mm = re.fullmatch(r"mtp\.layers\.0\.(.+)", k)
            if m:
                rule, name = LAYER.get(m.group(2)), lambda r: f"blk.{m.group(1)}.{r}"
            elif mm:
                rule, name = LAYER.get(mm.group(1)), lambda r: f"mtp.{r}"
            else:
                rule, name = GLOBAL.get(k), lambda r: r
            if rule is None:
                unknown.append(k)
                continue
            gname, form, how = rule[0], rule[1], rule[2]
            gname = name(gname)
            _, _, dt, shape, _ = ck.t[k]
            if mm and form == "FP4":
                form = "Q8"           # the drafter's MLP is BF16 in the checkpoint
            if form == "FP4" and dt != "U8":
                raise S.ConvertError(f"{k} is {dt}; NVFP4 codes are U8")
            if how == "gconv":
                if len(shape) != 3 or shape[1] != 1:
                    raise S.ConvertError(f"{k}: a conv weight of shape {shape}")
                ne = [shape[2], shape[0]]
            elif form == "FP4":
                ne = [shape[1] * 2, shape[0]]
            else:
                ne = list(reversed(shape))
            if form == "Q8" and (len(ne) != 2 or ne[0] % 32):
                raise S.ConvertError(f"{k}: rows of {ne[0]} values cannot be Q8_0")
            if form == "Q8" and dt not in ("F8_E4M3", "BF16"):
                raise S.ConvertError(f"{k} is {dt}; projections are read from FP8 or BF16")
            if form in ("F32", "BF16") and dt != "BF16":
                raise S.ConvertError(f"{k} is {dt}, expected BF16")
            if gname in self.tensors:
                raise S.ConvertError(f"two checkpoint tensors map to {gname}")
            self.tensors[gname] = Tensor(gname, form, k, how, ne)
            if form == "FP4":
                base = k[:-len(".weight")]
                ck.need(base + ".weight_scale", "F8_E4M3", (shape[0], shape[1] * 2 // 16))
                ck.need(base + ".weight_scale_2", "F32", ())
                self.scales[gname] = base + ".weight_scale_2"
            if dt == "F8_E4M3":
                ck.need(k[:-len(".weight")] + ".weight_scale", "F32", ())
        if unknown:
            raise S.ConvertError("checkpoint tensors with no GGUF counterpart: " + ", ".join(unknown[:8]))
        self._check_complete(tc)

    def _check_complete(self, tc):
        n_layers, kinds = int(tc["num_hidden_layers"]), tc["layer_types"]
        need = [r[0] for r in GLOBAL.values()]
        for l in range(n_layers):
            if kinds[l] not in ("linear_attention", "full_attention"):
                raise S.ConvertError(f"config.json: layer {l} is {kinds[l]}")
            for key, rule in LAYER.items():
                part = key.split(".")[0]
                if part == "linear_attn" and kinds[l] != "linear_attention" or part == "self_attn" and kinds[l] != "full_attention":
                    continue
                need.append(f"blk.{l}.{rule[0]}")
        if int(tc.get("mtp_num_hidden_layers", 0)) == 1:
            need += [f"mtp.{r[0]}" for k, r in LAYER.items() if not k.startswith("linear_attn")]
        missing = [n for n in need if n not in self.tensors]
        if missing:
            raise S.ConvertError(f"the checkpoint lacks {len(missing)} tensors the engine reads: " + ", ".join(missing[:8]))

    # ---- float tensors
    def values(self, name) -> np.ndarray:
        t = self.tensors[name]
        x = self.ck.values(t.src)
        if t.how == "gconv":
            x = x[:, 0, :]
        if t.how == "norm":
            x = x + np.float32(1)
        if t.how == "negexp":
            x = (-np.exp(x.astype(np.float64))).astype(np.float32)
        idx = C._index(t.how, x.shape[0], S.GDN_QK_ROWS)
        if idx is not None:
            x = x[idx]
        return np.ascontiguousarray(x, dtype=np.float32)

    def float_bytes(self, name) -> bytes:
        t = self.tensors[name]
        x = self.values(name)
        if not np.isfinite(x).all():
            raise S.ConvertError(f"{name} has a non-finite value")
        if t.type_name == "F32":
            return x.astype("<f4").tobytes()
        u = x.view(np.uint32)
        if np.count_nonzero(u & 0xFFFF):
            raise S.ConvertError(f"{name}: a value that is not BF16")
        return (u >> 16).astype("<u2").tobytes()

    def scale_value(self, name) -> np.ndarray:
        v = self.ck.values(self.scales[name]).astype("<f4").reshape(1)
        if not (np.isfinite(v).all() and v[0] > 0):
            raise S.ConvertError(f"{self.scales[name]} is not finite and positive")
        return v

    # ---- quantized tensors, streamed
    def jobs(self, name):
        t = self.tensors[name]
        p, off, dt, shape, _ = self.ck.t[t.src]
        ne0, ne1 = t.ne
        if t.form == "FP4":
            sp, soff = self.ck.t[t.src[:-len(".weight")] + ".weight_scale"][:2]
            step = max(1, ROWS_PER_JOB // ne0)
            for r in range(0, ne1, step):
                yield ("fp4", str(p), off, str(sp), soff, ne0, r, min(step, ne1 - r))
            return
        scale = float(self.ck.values(t.src[:-len(".weight")] + ".weight_scale")) if dt == "F8_E4M3" else None
        kind = {"vrows": "rows", "zrows": "rows", "vcols": "cols"}.get(t.how)
        first = S.GDN_QK_ROWS if t.how == "vrows" else 0
        step = max(1, ROWS_PER_JOB // ne0)
        for r in range(0, ne1, step):
            yield ("q8", str(p), off, dt, scale, ne0, ne1, r, min(step, ne1 - r), kind, first)

    def chunks(self, name, pool, depth: int = 64):
        for b in S.bounded_imap(pool, _job, self.jobs(name), depth):
            if isinstance(b, str):
                raise S.ConvertError(f"{name}: {b}")
            yield b


def _job(args):
    if args[0] == "fp4":
        _, path, off, spath, soff, ne0, row0, rows = args
        codes = np.fromfile(path, dtype=np.uint8, count=rows * ne0 // 2, offset=off + row0 * ne0 // 2).reshape(rows, ne0 // 2)
        sc = np.fromfile(spath, dtype=np.uint8, count=rows * ne0 // 16, offset=soff + row0 * ne0 // 16).reshape(rows, ne0 // 16)
        if (sc & 0x80).any() or (sc == 0x7F).any():
            return "a UE4M3 scale with the sign bit set or NaN"
        return C.nvfp4_rows(codes, sc).tobytes()
    _, path, off, dt, scale, ne0, ne1, row0, rows, kind, first = args
    if dt == "F8_E4M3":
        mm = np.memmap(path, dtype=np.uint8, mode="r", offset=off, shape=(ne1, ne0))
    else:
        mm = np.memmap(path, dtype="<u2", mode="r", offset=off, shape=(ne1, ne0))
    if kind == "rows":
        u = mm[S.gdn_index(ne1, first)[row0:row0 + rows]]
    elif kind == "cols":
        u = mm[row0:row0 + rows][:, S.gdn_index(ne0, 0)]
    else:
        u = mm[row0:row0 + rows]
    u = np.ascontiguousarray(u)
    x = E4M3[u.reshape(-1)] * np.float32(scale) if dt == "F8_E4M3" else S._bf16_to_f32(u).reshape(-1)
    if not np.isfinite(x).all():
        return "a non-finite value"
    return S.q8_0(x)


# ------------------------------------------------------------------ metadata
def model_kv(ck: C.Checkpoint, name: str | None) -> dict:
    c, tc = ck.config, ck.text
    A = ARCH + "."
    d = ck.dir
    gen = json.loads((d / "generation_config.json").read_text(encoding="utf-8")) if (d / "generation_config.json").is_file() else {}
    card = C.front_matter(d / "README.md")
    kinds = tc["layer_types"]
    rope = tc.get("rope_parameters", {})
    head = int(tc["head_dim"])
    kv = {"general.architecture": ARCH, "general.type": "model"}
    if "top_k" in gen:
        kv["general.sampling.top_k"] = int(gen["top_k"])
    if "top_p" in gen:
        kv["general.sampling.top_p"] = C.f32(gen["top_p"])
    if "temperature" in gen:
        kv["general.sampling.temp"] = C.f32(gen["temperature"])
    if name:
        kv["general.name"] = name
    if "license" in card:
        kv["general.license"] = card["license"]
    bases = card.get("base_model") or []
    bases = [bases] if isinstance(bases, str) else bases
    if bases:
        kv["general.base_model.count"] = len(bases)
        for i, b in enumerate(bases):
            org, _, repo = b.rpartition("/")
            kv[f"general.base_model.{i}.name"] = repo.replace("-", " ")
            if org:
                kv[f"general.base_model.{i}.organization"] = org
            kv[f"general.base_model.{i}.repo_url"] = f"https://huggingface.co/{b}"
    if tc.get("attn_output_gate") is not True:
        raise S.ConvertError("config.json: attention without the output gate")
    kv.update({
        A + "block_count": int(tc["num_hidden_layers"]),
        A + "context_length": int(tc["max_position_embeddings"]),
        A + "embedding_length": int(tc["hidden_size"]),
        A + "feed_forward_length": int(tc["intermediate_size"]),
        A + "attention.head_count": int(tc["num_attention_heads"]),
        A + "attention.head_count_kv": int(tc["num_key_value_heads"]),
        A + "attention.key_length": head,
        A + "attention.value_length": head,
        A + "attention.layer_norm_rms_epsilon": C.f32(tc["rms_norm_eps"]),
        A + "rope.dimension_count": int(head * float(rope.get("partial_rotary_factor", tc.get("partial_rotary_factor")))),
        A + "rope.dimension_sections": (list(rope.get("mrope_section", [])) + [0, 0, 0, 0])[:4],
        A + "rope.freq_base": C.f32(rope.get("rope_theta", tc.get("rope_theta"))),
        A + "ssm.conv_kernel": int(tc["linear_conv_kernel_dim"]),
        A + "ssm.state_size": int(tc["linear_key_head_dim"]),
        A + "ssm.group_count": int(tc["linear_num_key_heads"]),
        A + "ssm.time_step_rank": int(tc["linear_num_value_heads"]),
        A + "ssm.inner_size": int(tc["linear_num_value_heads"]) * int(tc["linear_value_head_dim"]),
        A + "attention.recurrent_layers": [k == "linear_attention" for k in kinds],
        A + "full_attention_interval": int(tc["full_attention_interval"]),
        A + "nextn_predict_layers": int(tc.get("mtp_num_hidden_layers", 0)),
        A + "image_token_id": int(c["image_token_id"]),
        A + "video_token_id": int(c["video_token_id"]),
        A + "vision_start_token_id": int(c["vision_start_token_id"]),
        "general.file_type": 40,
        "general.quantization_version": 2,
    })
    kv.update(C.tokenizer_kv(ck))
    return kv


# ------------------------------------------------------------------ vision
VISION_BLOCK = {
    "attn.qkv.weight": ("attn_qkv.weight", "BF16"), "attn.qkv.bias": ("attn_qkv.bias", "F32"),
    "attn.proj.weight": ("attn_out.weight", "BF16"), "attn.proj.bias": ("attn_out.bias", "F32"),
    "mlp.linear_fc1.weight": ("ffn_up.weight", "BF16"), "mlp.linear_fc1.bias": ("ffn_up.bias", "F32"),
    "mlp.linear_fc2.weight": ("ffn_down.weight", "BF16"), "mlp.linear_fc2.bias": ("ffn_down.bias", "F32"),
    "norm1.weight": ("ln1.weight", "F32"), "norm1.bias": ("ln1.bias", "F32"),
    "norm2.weight": ("ln2.weight", "F32"), "norm2.bias": ("ln2.bias", "F32"),
}
VISION_GLOBAL = {
    "model.visual.merger.norm.weight": ("v.post_ln.weight", "F32"), "model.visual.merger.norm.bias": ("v.post_ln.bias", "F32"),
    "model.visual.merger.linear_fc1.weight": ("mm.0.weight", "BF16"), "model.visual.merger.linear_fc1.bias": ("mm.0.bias", "F32"),
    "model.visual.merger.linear_fc2.weight": ("mm.2.weight", "BF16"), "model.visual.merger.linear_fc2.bias": ("mm.2.bias", "F32"),
    "model.visual.patch_embed.proj.bias": ("v.patch_embd.bias", "F32"),
    "model.visual.pos_embed.weight": ("v.position_embd.weight", "F32"),
}


def mmproj(ck: C.Checkpoint, out: pathlib.Path, name: str) -> None:
    """The vision tower as the clip GGUF the engine's encoder reads (llama.cpp's qwen3vl_merger layout)."""
    vc = ck.config["vision_config"]
    if vc.get("model_type") not in ("qwen3_5_vision",) or vc.get("deepstack_visual_indexes"):
        raise S.ConvertError("config.json: the vision tower is not the Qwen3-VL one without deepstack layers")
    pp = json.loads((ck.dir / "preprocessor_config.json").read_text(encoding="utf-8"))
    w = GGUFWriter()
    for k, v, t in (("general.architecture", "clip", "string"), ("general.type", "mmproj", "string"),
                    ("general.name", name + " (mmproj)", "string"), ("clip.has_vision_encoder", True, "bool"),
                    ("clip.projector_type", "qwen3vl_merger", "string"), ("clip.use_gelu", True, "bool"),
                    ("clip.vision.projection_dim", int(vc["out_hidden_size"]), "u32"),
                    ("clip.vision.image_size", 768, "u32"), ("clip.vision.patch_size", int(vc["patch_size"]), "u32"),
                    ("clip.vision.embedding_length", int(vc["hidden_size"]), "u32"),
                    ("clip.vision.feed_forward_length", int(vc["intermediate_size"]), "u32"),
                    ("clip.vision.block_count", int(vc["depth"]), "u32"),
                    ("clip.vision.attention.head_count", int(vc["num_heads"]), "u32"),
                    ("clip.vision.attention.layer_norm_epsilon", 1e-6, "f32"),
                    ("clip.vision.spatial_merge_size", int(vc["spatial_merge_size"]), "u32")):
        w.add(k, v, t)
    w.add("clip.vision.image_mean", [float(x) for x in pp["image_mean"]], "array:f32")
    w.add("clip.vision.image_std", [float(x) for x in pp["image_std"]], "array:f32")
    if vc.get("hidden_act") != "gelu_pytorch_tanh" or int(vc["temporal_patch_size"]) != 2:
        raise S.ConvertError("config.json: the vision tower's activation or temporal patch differs from Qwen3-VL's")

    def put(gname, src, ty):
        x = ck.values(src)
        if ty == "BF16":
            u = x.astype(np.float32).view(np.uint32) >> 16
            w.add_raw(gname, list(reversed(x.shape)), "BF16", u.astype("<u2").tobytes())
        else:
            a = np.ascontiguousarray(x, dtype="<f4")
            w.add_raw(gname, list(reversed(a.shape)), "F32", a.tobytes())

    for l in range(int(vc["depth"])):
        for k, (g, ty) in VISION_BLOCK.items():
            put(f"v.blk.{l}.{g}", f"model.visual.blocks.{l}.{k}", ty)
    for k, (g, ty) in VISION_GLOBAL.items():
        put(g, k, ty)
    pw = ck.values("model.visual.patch_embed.proj.weight")     # [out, 3, t=2, 16, 16]
    if pw.ndim != 5 or pw.shape[2] != 2:
        raise S.ConvertError(f"model.visual.patch_embed.proj.weight has shape {list(pw.shape)}")
    for t, g in ((0, "v.patch_embd.weight"), (1, "v.patch_embd.weight.1")):
        a = np.ascontiguousarray(pw[:, :, t], dtype="<f4")      # [out, 3, 16, 16] -> ne [16, 16, 3, out]
        w.add_raw(g, list(reversed(a.shape)), "F32", a.tobytes())
    w.write(out)


def text_vocab(kv: dict, out: pathlib.Path) -> None:
    """The text model's metadata alone (no tensors), as vision/text_vocab."""
    w = GGUFWriter()
    for k, v in kv.items():
        if isinstance(v, bool):
            w.add(k, v, "bool")
        elif isinstance(v, int):
            w.add(k, v, "u32" if v >= 0 else "i32")
        elif isinstance(v, float):
            w.add(k, v, "f32")
        elif isinstance(v, str):
            w.add(k, v, "string")
        elif isinstance(v, list) and v:
            w.add(k, v, {bool: "array:bool", int: "array:i32", float: "array:f32", str: "array:string"}[type(v[0])])
    w.write(out)


# ------------------------------------------------------------------ the pack
LICENSE = {"apache-2.0": "Apache License 2.0"}


def build(a, Job, fingerprint, die) -> None:
    import os
    import tokenizer
    out = pathlib.Path(a.out)
    partial = pathlib.Path(str(out) + ".partial")
    if out.exists() and not a.force:
        die(f"{out} exists (use --force to overwrite)")
    try:
        ck = C.Checkpoint(a.checkpoint)
        plan = Plan(ck)
        name = a.name or C.front_matter(ck.dir / "README.md").get("base_model", ["Qwen3.5"])[0].split("/")[-1] + " NVFP4"
        kv = model_kv(ck, name)
    except S.ConvertError as e:
        die(str(e))
    partial.unlink(missing_ok=True)
    w = ModelWriter(str(partial))
    job = Job(w)
    ok = False
    try:
        with tempfile.TemporaryDirectory() as tmp, multiprocessing.Pool(a.workers) as pool:
            tokenizer.extract_metadata(kv, pathlib.Path(tmp))
            print("writing sections:", flush=True)
            for p in sorted(q for q in (pathlib.Path(tmp) / "tokenizer").iterdir() if q.is_file()):
                job.file(f"tokenizer/{p.name}", p)
            t = time.time()
            parts, names = [], []
            for n, info in plan.tensors.items():
                if info.type_name in ("Q8_0", "NVFP4"):
                    parts.append(Stream(info.expected_bytes(), plan.chunks(n, pool)))
                else:
                    parts.append(plan.float_bytes(n))
                names.append(n)
                if info.type_name == "NVFP4":
                    parts.append(plan.scale_value(n).tobytes())
                    names.append(n[:-len(".weight")] + ".scale")
            offs = w.add_parts("gguf", parts, TENSOR_ALIGN)
            job._report("gguf", w.entries[-1][2], time.time() - t)
            mp = pathlib.Path(tmp) / "mmproj.gguf"
            mmproj(ck, mp, name)
            job.file("vision/mmproj", mp)
            tv = pathlib.Path(tmp) / "text_vocab.gguf"
            text_vocab(kv, tv)
            job.file("vision/text_vocab", tv)
            tens = {}
            for n, off in zip(names, offs):
                if n in plan.tensors:
                    t_ = plan.tensors[n]
                    tens[n] = {"type": t_.type_name, "type_id": t_.type_id, "shape": [int(d) for d in t_.ne],
                               "bytes": t_.expected_bytes(), "offset": off}
                else:
                    tens[n] = {"type": "F32", "type_id": 0, "shape": [1], "bytes": 4, "offset": off}
            used = sorted({t_.how for t_ in plan.tensors.values()})
            lic = kv.get("general.license", "")
            meta = {
                "format": "basalt", "version": 2, "name": name, "arch": ARCH,
                "converter": {"tool": "tools/convert/convert.py"},
                "sources": [{"path": p.name, "size": p.stat().st_size,
                             "fingerprint_sha256_first_last_64mib_and_size": fingerprint(p)} for p in ck.shards],
                "gguf_kv": kv,
                "tensors": tens,
                "license": {"name": LICENSE.get(lic, lic)},
                "vision": {"projector": "qwen3vl_merger", "source": "the checkpoint's BF16 vision tower"},
                "mtp": {"layers": int(ck.text.get("mtp_num_hidden_layers", 0)), "source": "the checkpoint's BF16 mtp.*, Q8_0"},
                "checkpoint": {
                    "dir": pathlib.Path(a.checkpoint).name,
                    "config_sha256": hashlib.sha256((ck.dir / "config.json").read_bytes()).hexdigest(),
                    "nvfp4": "weight (E2M1 pairs) and weight_scale (UE4M3 per 16) moved into 64-value blocks unchanged; "
                             "weight_scale_2 as <name>.scale; input_scale unused",
                    "fp8": "code x weight_scale in f32, Q8_0 blocks of 32 (ggml reference rounding); input_scale unused",
                    "bf16_projections": "Q8_0 (ggml reference rounding)",
                    "transforms": {k: METHODS[k] for k in used}},
            }
            job.bytes("meta", json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        w.close()
        ok = True
    except S.ConvertError as e:
        die(str(e))
    finally:
        if not ok:
            try:
                w.f.close()
            except OSError:
                pass
            partial.unlink(missing_ok=True)
    os.replace(partial, out)
    print(f"done: {len(job.rows)} sections, {out.stat().st_size / (1 << 30):.3f} GiB, {time.time() - job.t0:.1f} s -> {out}")
