"""Tests for checkpoint_inkling2.py (convert.py --checkpoint DIR on an inkling2_loop checkpoint) on a tiny synthetic
checkpoint shaped as Thinking Machines' release: three layers (a sliding-window one, a full-attention loop core layer,
a sliding-window coda), NVFP4 experts with an amax and a denominator, MXFP8 experts in the last layer, BF16 everything
else, the DSpark drafter beside it in mtp.safetensors, a one-block vision tower and tokenizer files.

    python3 tools/convert/test_checkpoint_inkling2.py
    python3 -m unittest discover -s tools/convert -p 'test_checkpoint_inkling2.py'"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import checkpoint_inkling2 as I  # noqa: E402
import model_sections as S  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402
from model_file import ModelFile  # noqa: E402
from test_checkpoint import basalt_values, modelopt_values, write_safetensors  # noqa: E402

H, NH, NKV, HD, DREL, FF, NE = 128, 2, 1, 32, 16, 128, 4
LAYERS, LOCAL, CORE = 3, [0, 2], (1, 2)
VH, VPATCH, VGRID = 64, 20, 2
MH, MFF, MRANK = 128, 64, 8
TOKENS = list("abcdefghijklmnop") + ["<|endoftext|>", "<|message_user|>", "<|message_model|>", "<|content_text|>",
                                     "<|end_message|>", "<|content_model_end_sampling|>", "<|unused_200054|>"]
VOCAB = 32                                   # the padded rows of the embedding and the head
E4M3 = S.fp8_e4m3_table()


def e2m1(c):
    return [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6][int(c)]


def q8_values(blob: bytes, rows: int, cols: int) -> np.ndarray:
    b = np.frombuffer(blob, np.uint8).reshape(rows * cols // 32, 34)
    d = np.ascontiguousarray(b[:, :2]).view("<f2").astype(np.float64).reshape(-1)
    return (b[:, 2:].view(np.int8).astype(np.float64) * d[:, None]).reshape(rows, cols)


class Fixture:
    def __init__(self, d: Path):
        self.rng = np.random.default_rng(5)
        self.v, self.tens, self.mtp = {}, {}, {}
        self.nv, self.mx = {}, {}
        L = "model.llm.layers."
        self.bf("model.llm.embed.weight", (VOCAB, H))
        self.bf("model.llm.embed_norm.weight", (H,))
        self.bf("model.llm.norm.weight", (H,))
        self.bf("model.llm.unembed.weight", (VOCAB, H))
        self.f32("model.llm.attn_res.output_query", (H,))
        for i in range(LAYERS - 1):   # attn_queries.i feeds layer i + 1: three queries for the full layer 1
            self.f32(f"model.llm.attn_res.attn_queries.{i}", (3, H) if i + 1 not in LOCAL else (H,))
        for l in range(LAYERS):
            self.f32(f"model.llm.attn_res.mlp_queries.{l}", (H,))
        self.f32("model.llm.middle_cycle_cross_pass_queries", (2 * (CORE[1] - CORE[0]), H))
        self.f32("model.llm.middle_cycle_cross_pass_kv_queries", (2 * (CORE[1] - CORE[0]), H))
        for l in range(LAYERS):
            p = f"{L}{l}."
            for n, shape in (("wq_du", (NH * HD, H)), ("wk_dv", (NKV * HD, H)), ("wv_dv", (NKV * HD, H)), ("wr_du", (NH * DREL, H)),
                             ("gate_attn", (NH * HD, H)), ("attn_sink_proj", (NH, H)), ("wo_ud", (H, NH * HD))):
                self.bf(p + f"attn.{n}.weight", shape)
            self.bf(p + "attn.q_l2_clip.scale", (NH, HD))
            self.bf(p + "attn.rel_l2_clip.scale", (NH, DREL))
            self.bf(p + "attn.rel_logits_proj.proj", (DREL, 1024))
            if l in LOCAL:
                self.bf(p + "attn_norm.weight", (H,))
                self.bf(p + "attn.k_sconv.weight", (NKV * HD, 1, 4))
                self.bf(p + "attn.v_sconv.weight", (NKV * HD, 1, 4))
            else:
                for n in ("q_input_norm", "k_input_norm", "v_input_norm"):
                    self.bf(p + f"{n}.weight", (H,))
                for n in ("k_sconv", "v_sconv"):
                    self.bf(p + f"attn.{n}.static_w", (4, NKV * HD))
                    self.bf(p + f"attn.{n}.weight_proj", (NKV * 4, H))
            self.bf(p + "attn_sconv.weight", (H, 1, 4))
            self.bf(p + "mlp_sconv.weight", (H, 1, 4))
            self.bf(p + "mlp_norm.weight", (H,))
            self.bf(p + "mlp.gate.weight", (NE, H))
            self.f32(p + "mlp.gate.bias", (NE,))
            if CORE[0] <= l < CORE[1]:
                for s in ("attn", "k", "v", "mlp"):
                    self.bf(p + f"controllers.{s}.down_weight", (64, 2 * H))
                    self.bf(p + f"controllers.{s}.forget_weight", (H, 64))
                    self.bf(p + f"controllers.{s}.gain_weight", (H, 64))
            if l < LAYERS - 1:
                self.nvfp4_experts(p + "mlp.experts.")
            else:
                self.mxfp8_experts(p + "mlp.experts.")
        V = "model.visual."
        self.bf(V + "backbone.embeddings.class_embedding", (VH,))
        self.bf(V + "backbone.embeddings.cls_position_embedding", (VH,))
        self.bf(V + "backbone.embeddings.patch_embedding.weight", (VH, 3 * VPATCH * VPATCH))
        self.bf(V + "backbone.embeddings.position_embedding", (VGRID * VGRID, VH))
        self.bf(V + "backbone.ln_pre.weight", (VH,))
        self.bf(V + "backbone.ln_pre.bias", (VH,))
        B = V + "backbone.layers.0."
        for n in ("ln_1", "ln_2"):
            self.bf(B + n + ".weight", (VH,))
            self.bf(B + n + ".bias", (VH,))
        self.bf(B + "ls_1.gamma", (VH,))
        self.bf(B + "ls_2.gamma", (VH,))
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            self.bf(B + f"self_attn.{n}.weight", (VH, VH))
            self.bf(B + f"self_attn.{n}.bias", (VH,))
        self.bf(B + "mlp.c_fc.weight", (2 * VH, VH))
        self.bf(B + "mlp.c_fc.bias", (2 * VH,))
        self.bf(B + "mlp.c_proj.weight", (VH, 2 * VH))
        self.bf(B + "mlp.c_proj.bias", (VH,))
        self.bf(V + "proj_in.weight", (128, 4 * VH))
        self.bf(V + "proj_out.weight", (H, 128))
        write_safetensors(d / "model-00001-of-00001.safetensors", self.tens)
        # the drafter: no prefix of its own
        for n, shape in (("fuse_weight", (2, 2)), ("mask_embed", (MH,)), ("norm.weight", (MH,)),
                         ("markov_head.markov_w1.weight", (VOCAB, MRANK)), ("markov_head.rnn_proj.weight", (3 * MRANK, 2 * MRANK + 2 * MH)),
                         ("markov_head.rnn_proj.bias", (3 * MRANK,)), ("markov_head.hidden_out.weight", (MH, MRANK)),
                         ("markov_head.hidden_out.bias", (MH,)), ("markov_head.dflare_norm.weight", (MH,)),
                         ("layers.0.input_layernorm.weight", (MH,)), ("layers.0.post_attention_layernorm.weight", (MH,)),
                         ("layers.0.dflare_norm.weight", (MH,)), ("layers.0.self_attn.q_proj.weight", (NH * HD, MH)),
                         ("layers.0.self_attn.k_proj.weight", (NKV * HD, MH)), ("layers.0.self_attn.v_proj.weight", (NKV * HD, MH)),
                         ("layers.0.self_attn.o_proj.weight", (MH, NH * HD)), ("layers.0.self_attn.k_proj_ctx.weight", (NKV * HD, MH)),
                         ("layers.0.self_attn.v_proj_ctx.weight", (NKV * HD, MH)), ("layers.0.self_attn.q_norm.weight", (HD,)),
                         ("layers.0.self_attn.k_norm.weight", (HD,)), ("layers.0.self_attn.k_norm_ctx.weight", (HD,)),
                         ("layers.0.mlp.gate_proj.weight", (MFF, MH)), ("layers.0.mlp.up_proj.weight", (MFF, MH)),
                         ("layers.0.mlp.down_proj.weight", (MH, MFF))):
            x = (self.rng.integers(-64, 64, size=shape) / 64).astype(np.float32)
            self.v["mtp." + n] = x
            self.mtp[n] = ("BF16", shape, (x.view(np.uint32) >> 16).astype("<u2").tobytes())
        write_safetensors(d / "mtp.safetensors", self.mtp)
        text = {"model_type": "inkling2_loop_text", "model_max_length": 65536, "hidden_size": H, "num_hidden_layers": LAYERS,
                "vocab_size": VOCAB, "unpadded_vocab_size": len(TOKENS), "num_attention_heads": NH, "num_key_value_heads": NKV,
                "head_dim": HD, "d_rel": DREL, "rel_extent": 1024, "q_bias": False, "o_bias": False, "log_scaling_n_floor": 8192,
                "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6, "use_value_norm_no_gamma": True, "qk_l2_clip_threshold": 22.6,
                "rel_l2_clip_threshold": 8.0, "local_layer_ids": LOCAL, "sconv_kernel_size": 4, "final_logit_softcapping": 30.0,
                "swa_head_dim": HD, "swa_num_attention_heads": NH, "swa_num_key_value_heads": NKV, "sliding_window_size": 1024,
                "n_layers_per_attn_res_block": 1, "attn_res_normalize_values": True, "n_routed_experts": NE,
                "num_experts_per_tok": 2, "n_shared_experts": 0, "intermediate_size": FF, "mlp_gate_softcap": 4.0,
                "mlp_up_softcap": 25.0, "route_scale": 8.0, "use_gate_bias": True,
                "loop_config": {"core_start": CORE[0], "core_end": CORE[1], "num_passes": 2, "gate_rank": 64, "is_varloop": False}}
        vision = {"model_type": "inkling2_vision", "text_hidden_size": H, "temporal_patch_size": 2, "native_patch_size": VPATCH,
                  "adapter_merge_size": 2, "projection_hidden_dim": 128, "projector_bias": False, "hidden_size": VH,
                  "num_hidden_layers": 1, "num_attention_heads": 2, "mlp_ratio": 2.0, "posemb_grid_size": VGRID, "layer_norm_eps": 1e-5}
        mtp = {"hidden_size": MH, "num_hidden_layers": 1, "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD,
               "intermediate_size": MFF, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "local_layer_ids": [0],
               "sliding_window_size": 1024, "block_size": 8, "markov_rank": MRANK, "markov_head_type": "rnn_hidden",
               "target_layer_ids": [1, 4]}
        (d / "config.json").write_text(json.dumps({"model_type": "inkling2_loop", "architectures": ["Inkling2LoopForConditionalGeneration"],
                                                   "text_config": text, "vision_config": vision, "mtp_config": mtp}))
        groups = [{"weight_transform": "nvfp4_nearest", "weight_scale_format": "amax_denominator",
                   "tensors": [f"{L}{l}.mlp.experts.{w}_weight" for l in range(LAYERS - 1) for w in ("w13", "w2")]},
                  {"weight_transform": "mxfp8_nearest", "tensors": [f"{L}{LAYERS - 1}.mlp.experts.{w}_weight" for w in ("w13", "w2")]}]
        (d / "hf_quant_config.json").write_text(json.dumps({"model": {"parameter_groups": groups}}))
        (d / "generation_config.json").write_text(json.dumps({"temperature": 1.0}))
        (d / "processor_config.json").write_text(json.dumps({"image_processor": {"image_mean": [0.5] * 3, "image_std": [0.5] * 3}}))
        (d / "README.md").write_text("---\nlicense: apache-2.0\n---\n# card\n")
        (d / "chat_template.jinja").write_text("{{ messages }}")
        vocab = {t: i for i, t in enumerate(TOKENS[:16])}
        (d / "tokenizer.json").write_text(json.dumps({
            "model": {"type": "BPE", "vocab": vocab, "merges": [["a", "b"], ["c", "d"]]},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": I.INKLING2_HF_SPLIT}}, {"type": "ByteLevel"}]},
            "added_tokens": [{"id": 16 + i, "content": t, "special": True} for i, t in enumerate(TOKENS[16:])]}))
        (d / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|content_model_end_sampling|>", "pad_token": "<|endoftext|>"}))

    def bf(self, name, shape):
        x = (self.rng.integers(-64, 64, size=shape) / 64).astype(np.float32)
        self.v[name] = x
        self.tens[name] = ("BF16", shape, (x.view(np.uint32) >> 16).astype("<u2").tobytes())

    def f32(self, name, shape):
        x = self.rng.standard_normal(shape).astype(np.float32)
        self.v[name] = x
        self.tens[name] = ("F32", shape, x.astype("<f4").tobytes())

    def nvfp4_experts(self, p):
        for w, rows, cols in (("w13", 2 * FF, H), ("w2", H, FF)):
            codes = self.rng.integers(0, 256, size=(NE, rows, cols // 2), dtype=np.uint8)
            scales = self.rng.integers(32, 120, size=(NE, rows, cols // 16), dtype=np.uint8)
            amax = self.rng.uniform(0.1, 0.5, size=(NE,)).astype(np.float32)
            den = np.array([2688.0], np.float32)
            self.nv[p + w] = (codes, scales, amax, den)
            self.tens[p + w + "_weight"] = ("U8", (NE, rows, cols // 2), codes.tobytes())
            self.tens[p + w + "_weight_scale"] = ("F8_E4M3", (NE, rows, cols // 16), scales.tobytes())
            self.tens[p + w + "_weight_amax"] = ("F32", (NE,), amax.tobytes())
            self.tens[p + w + "_weight_outer_scale_denominator"] = ("F32", (1,), den.tobytes())

    def mxfp8_experts(self, p):
        for w, rows, cols in (("w13", 2 * FF, H), ("w2", H, FF)):
            codes = self.rng.integers(0, 127, size=(NE, rows, cols), dtype=np.uint8) | (self.rng.integers(0, 2, size=(NE, rows, cols), dtype=np.uint8) << 7)
            scales = self.rng.integers(112, 120, size=(NE, rows, cols // 32), dtype=np.uint8)
            self.mx[p + w] = (codes, scales)
            self.tens[p + w + "_weight"] = ("F8_E4M3", (NE, rows, cols), codes.tobytes())
            self.tens[p + w + "_weight_scale"] = ("U8", (NE, rows, cols // 32), scales.tobytes())


class ConvertTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name) / "ckpt"
        d.mkdir()
        cls.fx = Fixture(d)
        cls.out = Path(cls.tmp.name) / "tiny.basalt"
        r = subprocess.run([sys.executable, str(HERE / "convert.py"), "--checkpoint", str(d), "--out", str(cls.out), "--workers", "2"],
                           capture_output=True, text=True)
        if r.returncode:
            raise AssertionError(r.stdout + r.stderr)
        cls.mf = ModelFile(str(cls.out))
        cls.meta = json.loads(cls.mf.read("meta"))
        cls.gguf = cls.mf.read("gguf")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def tensor(self, name):
        t = self.meta["tensors"][name]
        return t, self.gguf[t["offset"]:t["offset"] + t["bytes"]]

    def test_the_projections_are_q8_0_of_their_values(self):
        for name, src in (("blk.0.attn_q.weight", "model.llm.layers.0.attn.wq_du.weight"),
                          ("blk.1.attn_k_conv_proj.weight", "model.llm.layers.1.attn.k_sconv.weight_proj"),
                          ("blk.2.attn_output.weight", "model.llm.layers.2.attn.wo_ud.weight")):
            t, raw = self.tensor(name)
            self.assertEqual(t["type"], "Q8_0", name)
            x = self.fx.v[src]
            got = q8_values(raw, x.shape[0], x.shape[1])
            self.assertLess(np.abs(got - x).max(), np.abs(x).max() / 127 + 1e-6, name)

    def test_the_float_tensors_are_exact(self):
        for name, src in (("blk.0.attn_norm.weight", "model.llm.layers.0.attn_norm.weight"),
                          ("blk.1.ffn_gate_inp.weight", "model.llm.layers.1.mlp.gate.weight"),
                          ("blk.1.exp_probs_b.bias", "model.llm.layers.1.mlp.gate.bias"),
                          ("blk.0.attn_rel_bias.weight", "model.llm.layers.0.attn.rel_logits_proj.proj"),
                          ("blk.2.attn_res_query.weight", "model.llm.attn_res.attn_queries.1"),
                          ("blk.1.attn_res_query.weight", "model.llm.attn_res.attn_queries.0"),
                          ("output_res_query.weight", "model.llm.attn_res.output_query")):
            t, raw = self.tensor(name)
            self.assertEqual(t["type"], "F32", name)
            np.testing.assert_array_equal(np.frombuffer(raw, "<f4").reshape(self.fx.v[src].shape), self.fx.v[src], name)
        _, raw = self.tensor("blk.0.attn_k_conv.weight")      # [C, 1, K] -> [C, K]
        np.testing.assert_array_equal(np.frombuffer(raw, "<f4").reshape(NKV * HD, 4), self.fx.v["model.llm.layers.0.attn.k_sconv.weight"][:, 0])
        _, raw = self.tensor("blk.1.attn_k_conv.weight")      # static taps, row 0 the current token -> the conv order
        np.testing.assert_array_equal(np.frombuffer(raw, "<f4").reshape(NKV * HD, 4), self.fx.v["model.llm.layers.1.attn.k_sconv.static_w"][::-1].T)

    def test_the_message_queries_are_stacked(self):
        _, raw = self.tensor("blk.1.loop_msg_query.weight")
        q, kv = self.fx.v["model.llm.middle_cycle_cross_pass_queries"], self.fx.v["model.llm.middle_cycle_cross_pass_kv_queries"]
        np.testing.assert_array_equal(np.frombuffer(raw, "<f4").reshape(4, H), np.stack([q[0], kv[0], kv[1], q[1]]))

    def test_the_embedding_and_head_drop_the_padded_rows(self):
        for name, src in (("token_embd.weight", "model.llm.embed.weight"), ("output.weight", "model.llm.unembed.weight")):
            t, raw = self.tensor(name)
            self.assertEqual((t["type"], t["shape"]), ("BF16", [H, len(TOKENS)]))
            got = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(len(TOKENS), H)
            np.testing.assert_array_equal(got, self.fx.v[src][:len(TOKENS)])

    def test_the_nvfp4_experts_move_bit_for_bit(self):
        lay = [l.split() for l in self.mf.read("experts.layout").decode().splitlines() if l and l[0] != "#"]
        self.assertEqual([int(x[1]) for x in lay], [40, 40, 46])
        experts = self.mf.read("experts")
        l, _, _, off, blob = map(int, lay[1])
        codes, scales, amax, den = self.fx.nv[f"model.llm.layers.{l}.mlp.experts.w13"]
        codes2, scales2, amax2, den2 = self.fx.nv[f"model.llm.layers.{l}.mlp.experts.w2"]
        gu_bytes = FF * S.nvfp4_row_bytes(H)
        for e in range(NE):
            b = np.frombuffer(experts[off + e * blob:off + (e + 1) * blob], np.uint8)
            want = modelopt_values(codes[e], scales[e])
            np.testing.assert_array_equal(basalt_values(b[:gu_bytes], H), want[0::2])          # gate: the even rows
            np.testing.assert_array_equal(basalt_values(b[gu_bytes:2 * gu_bytes], H), want[1::2])
            np.testing.assert_array_equal(basalt_values(b[2 * gu_bytes:blob - 16], FF), modelopt_values(codes2[e], scales2[e]))
            tail = np.frombuffer(b[blob - 16:].tobytes(), "<f4")
            self.assertEqual(tail.tolist(), [np.float32(amax[e]) / np.float32(den[0])] * 2 + [np.float32(amax2[e]) / np.float32(den2[0]), 0.0])

    def test_the_mxfp8_experts_are_native_mxfp8_blocks(self):
        lay = [l.split() for l in self.mf.read("experts.layout").decode().splitlines() if l and l[0] != "#"]
        l, gu, dt, off, blob = map(int, lay[2])
        self.assertEqual((gu, dt), (46, 46))
        experts = self.mf.read("experts")
        codes, scales = self.fx.mx[f"model.llm.layers.{l}.mlp.experts.w13"]
        c2, s2 = self.fx.mx[f"model.llm.layers.{l}.mlp.experts.w2"]
        e = 1
        def pack(c, s):
            nb = c.shape[1] // 128
            return np.concatenate([s.reshape(c.shape[0], nb, 4), c.reshape(c.shape[0], nb, 128)], axis=2).tobytes()
        b = experts[off + e * blob:off + (e + 1) * blob]
        n = FF * (H // 128) * 132
        self.assertEqual(b[:n], pack(codes[e, 0::2], scales[e, 0::2]))
        self.assertEqual(b[n:2 * n], pack(codes[e, 1::2], scales[e, 1::2]))
        self.assertEqual(b[2 * n:], pack(c2[e], s2[e]))

    def test_the_metadata(self):
        kv = self.meta["gguf_kv"]
        self.assertEqual(self.meta["arch"], "inkling2")
        self.assertEqual(kv["general.architecture"], "inkling2")
        self.assertEqual(kv["inkling2.loop.start"], 1)
        self.assertEqual(kv["inkling2.attention.sliding_window_pattern"], [True, False, True])
        self.assertEqual(kv["inkling2.dspark.target_layers"], [1, 4])
        self.assertEqual(kv["tokenizer.ggml.pre"], "inkling2")
        self.assertEqual(len(kv["tokenizer.ggml.tokens"]), len(TOKENS))
        self.assertIn("vision/mmproj", self.mf.names())
        self.assertIn("dspark.markov.w1.weight", self.meta["tensors"])

    def test_the_drafter_is_kept(self):
        t, raw = self.tensor("dspark.blk.0.attn_k_ctx.weight")
        self.assertEqual(t["type"], "BF16")
        got = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(NKV * HD, MH)
        np.testing.assert_array_equal(got, self.fx.v["mtp.layers.0.self_attn.k_proj_ctx.weight"])

    def test_the_vision_tower_is_llama_cpps_mmproj(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "mmproj.gguf"
            p.write_bytes(self.mf.read("vision/mmproj"))
            g = GGUFFile(p)
            raw = p.read_bytes()
        md = g.metadata
        self.assertEqual((md["general.architecture"], md["clip.vision.projector_type"]), ("clip", "inkling2v"))
        self.assertEqual([md["clip.vision." + k] for k in ("patch_size", "embedding_length", "feed_forward_length", "block_count",
                                                           "spatial_merge_size", "projection_dim", "image_size")],
                         [VPATCH, VH, 2 * VH, 1, 2, H, VGRID * VPATCH])
        self.assertEqual(md["clip.vision.image_mean"], [0.5] * 3)
        t = {x.name: x for x in g.tensors}
        self.assertEqual(len(t), 6 + 18 + 2)        # the embeddings and the pre-norm, one block, the two projector products

        def values(name):
            x = t[name]
            at = g.data_start + x.offset
            if x.type_name == "F32":
                return np.frombuffer(raw[at:at + 4 * x.elements], "<f4")
            return (np.frombuffer(raw[at:at + 2 * x.elements], "<u2").astype(np.uint32) << 16).view(np.float32)

        # the linear patch embedding [out][(row, column, channel)] as llama.cpp's convolution [out][channel][row][column]
        pw = self.fx.v["model.visual.backbone.embeddings.patch_embedding.weight"].reshape(VH, VPATCH, VPATCH, 3).transpose(0, 3, 1, 2)
        self.assertEqual((t["v.patch_embd.weight"].type_name, t["v.patch_embd.weight"].shape), ("F32", [VPATCH, VPATCH, 3, VH]))
        np.testing.assert_array_equal(values("v.patch_embd.weight"), pw.reshape(-1))
        V, B = "model.visual.backbone.", "model.visual.backbone.layers.0."
        for name, src, ty in (("v.class_embd", V + "embeddings.class_embedding", "F32"),
                              ("v.class_position_embd.weight", V + "embeddings.cls_position_embedding", "F32"),
                              ("v.position_embd.weight", V + "embeddings.position_embedding", "F32"),
                              ("v.pre_ln.bias", V + "ln_pre.bias", "F32"), ("v.blk.0.ls1.weight", B + "ls_1.gamma", "F32"),
                              ("v.blk.0.ls2.weight", B + "ls_2.gamma", "F32"), ("v.blk.0.attn_q.weight", B + "self_attn.q_proj.weight", "BF16"),
                              ("v.blk.0.attn_out.bias", B + "self_attn.out_proj.bias", "F32"), ("v.blk.0.ffn_up.weight", B + "mlp.c_fc.weight", "BF16"),
                              ("v.blk.0.ffn_down.weight", B + "mlp.c_proj.weight", "BF16"), ("mm.1.weight", "model.visual.proj_in.weight", "BF16"),
                              ("mm.2.weight", "model.visual.proj_out.weight", "BF16")):
            with self.subTest(tensor=name):
                self.assertEqual(t[name].type_name, ty)
                np.testing.assert_array_equal(values(name), self.fx.v[src].reshape(-1))
        self.assertEqual(t["mm.2.weight"].shape, [128, H])   # [in, out] in ggml's order: the text width out


class DFlash2ConvertTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name) / "ckpt"
        df = Path(cls.tmp.name) / "dflash2"
        d.mkdir()
        df.mkdir()
        cls.fx = Fixture(d)
        rng = np.random.default_rng(11)
        cls.dv, dtens = {}, {}
        groups = H // 16
        for n, shape in (("fc.weight", (H, 2 * H)), ("hidden_norm.weight", (H,)), ("norm.weight", (H,)),
                         ("candidate_selector.hidden_projection.weight", (MRANK, H)),
                         ("candidate_selector.predecessor_codebook", (VOCAB, MRANK)),
                         ("candidate_selector.successor_codebook", (VOCAB, MRANK)),
                         ("layers.0.input_layernorm.weight", (H,)), ("layers.0.post_attention_layernorm.weight", (H,)),
                         ("layers.0.self_attn.q_proj.weight", (4 * HD, H)), ("layers.0.self_attn.k_proj.weight", (NKV * HD, H)),
                         ("layers.0.self_attn.v_proj.weight", (NKV * HD, H)), ("layers.0.self_attn.o_proj.weight", (H, 4 * HD)),
                         ("layers.0.self_attn.q_norm.weight", (HD,)), ("layers.0.self_attn.k_norm.weight", (HD,)),
                         ("layers.0.mlp.gate_proj.weight", (MFF, H)), ("layers.0.mlp.up_proj.weight", (MFF, H)),
                         ("layers.0.mlp.down_proj.weight", (H, MFF)),
                         ("layers.0.attention_conv.base_kernel", (2, 2, H)),
                         ("layers.0.attention_conv.kernel_projection.weight", (4 * groups, H)),
                         ("layers.0.mlp_conv.base_kernel", (2, 2, H)),
                         ("layers.0.mlp_conv.kernel_projection.weight", (4 * groups, H))):
            x = (rng.integers(-64, 64, size=shape) / 64).astype(np.float32)
            cls.dv[n] = x
            dtens[n] = ("BF16", shape, (x.view(np.uint32) >> 16).astype("<u2").tobytes())
        write_safetensors(df / "model.safetensors", dtens)
        (df / "config.json").write_text(json.dumps({
            "architectures": ["DFlash2DraftModel"], "hidden_size": H, "intermediate_size": MFF, "num_hidden_layers": 1,
            "num_attention_heads": 4, "num_key_value_heads": NKV, "head_dim": HD, "rms_norm_eps": 1e-6, "is_causal": False,
            "layer_types": ["sliding_attention"], "sliding_window": 2048, "vocab_size": VOCAB, "attention_bias": False,
            "hidden_act": "silu", "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
            "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2, "final_logit_softcapping": 30.0,
                              "mask_token_id": len(TOKENS) - 1, "selector_rank": MRANK,
                              "selector_softcap_mode": "after_correction", "selector_top_k": 16, "target_layer_ids": [1, 4]},
        }))
        cls.out = Path(cls.tmp.name) / "dflash2.basalt"
        r = subprocess.run([sys.executable, str(HERE / "checkpoint_inkling2.py"), "--checkpoint", str(d),
                            "--dflash2", str(df), "--out", str(cls.out), "--workers", "2"], capture_output=True, text=True)
        if r.returncode:
            raise AssertionError(r.stdout + r.stderr)
        cls.mf = ModelFile(str(cls.out))
        cls.meta = json.loads(cls.mf.read("meta"))
        cls.gguf = cls.mf.read("gguf")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def tensor(self, name):
        t = self.meta["tensors"][name]
        return t, self.gguf[t["offset"]:t["offset"] + t["bytes"]]

    def test_dflash2_replaces_dspark(self):
        kv = self.meta["gguf_kv"]
        self.assertIsNone(self.meta["dspark"])
        self.assertIn("dflash2", self.meta)
        self.assertFalse(any(k.startswith("inkling2.dspark.") for k in kv))
        self.assertFalse(any(k.startswith("dspark.") for k in self.meta["tensors"]))
        self.assertNotIn("mtp.safetensors", [s["path"] for s in self.meta["sources"]])
        self.assertEqual(kv["inkling2.dflash2.block_count"], 1)
        self.assertEqual(kv["inkling2.dflash2.target_layers"], [1, 4])
        self.assertEqual(kv["inkling2.dflash2.sliding_window"], 2048)
        self.assertEqual(kv["inkling2.dflash2.attention.head_count"], 4)
        self.assertEqual(kv["inkling2.dflash2.conv_group"], 16)

    def test_dflash2_tensors_and_vocab_trim(self):
        t, raw = self.tensor("dflash2.fc.weight")
        self.assertEqual((t["type"], t["shape"]), ("Q8_0", [2 * H, H]))
        x = self.dv["fc.weight"]
        self.assertLess(np.abs(q8_values(raw, x.shape[0], x.shape[1]) - x).max(), np.abs(x).max() / 127 + 1e-6)
        t, raw = self.tensor("dflash2.blk.0.attn_conv.base")
        self.assertEqual((t["type"], t["shape"]), ("F32", [H, 2, 2]))
        np.testing.assert_array_equal(np.frombuffer(raw, "<f4").reshape(2, 2, H), self.dv["layers.0.attention_conv.base_kernel"])
        for name, src in (("dflash2.selector.pred", "candidate_selector.predecessor_codebook"),
                          ("dflash2.selector.succ", "candidate_selector.successor_codebook")):
            t, raw = self.tensor(name)
            self.assertEqual((t["type"], t["shape"]), ("BF16", [MRANK, len(TOKENS)]))
            got = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(len(TOKENS), MRANK)
            np.testing.assert_array_equal(got, self.dv[src][:len(TOKENS)])


if __name__ == "__main__":
    unittest.main()
