"""--model FILE.basalt: the config derived from the file, the tokenizer unpacked once, the peer card, vision."""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from serve.engine import modelfile
from serve.server import engine_args, gpus_in_play
from model_file import ModelFile, ModelWriter  # noqa: E402

META = {"name": "tiny-basalt",
        "sampling": {"temperature": 0.7, "top_p": 0.9}}
TOKENIZER = {"vocab.json": b'{"a": 0}', "merges.txt": b"a a", "token_type.json": b"[1]",
             "chat_template.jinja": b"{{ messages }}"}


class Tmp(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.dir = Path(self.td.name)
        patch = mock.patch.dict(os.environ, {"BASALT_CACHE": str(self.dir / "cache")})
        patch.start()
        self.addCleanup(patch.stop)
        self.engine_bin = self.dir / "bin" / "basalt"
        self.engine_bin.parent.mkdir()
        self.engine_bin.write_text("")

    def build(self, vision=False, meta=META):
        path = self.dir / "tiny.basalt"
        w = ModelWriter(str(path))
        w.add_bytes("meta", json.dumps(meta).encode())
        for n, b in TOKENIZER.items():
            w.add_bytes("tokenizer/" + n, b)
        if vision:
            w.add_bytes("vision/mmproj", b"MMPROJ" * 100)
            w.add_bytes("vision/text_vocab", b"VOCAB")
        w.close()
        return str(path)


class Derivation(Tmp):
    def test_config(self):
        path = self.build()
        cfg = modelfile.model_config(path, str(self.engine_bin), gpus=1)
        self.assertEqual(cfg["engine_bin"], str(self.engine_bin))
        self.assertEqual(cfg["args"], ["--model", path])
        self.assertEqual(cfg["model_name"], "tiny-basalt")
        self.assertEqual(cfg["sampling"], META["sampling"])
        self.assertTrue(cfg["log"].startswith(str(self.dir / "cache")))
        self.assertNotIn("vision", cfg)
        self.assertEqual(engine_args(cfg), cfg["args"])

    def test_cache_key_is_the_meta_hash(self):
        path = self.build()
        key = ModelFile(path).sections["meta"][2].hex()[:16]
        tok = Path(modelfile.model_config(path, str(self.engine_bin), gpus=1)["tokenizer"])
        self.assertEqual(tok, self.dir / "cache" / key / "tokenizer")

    def test_tokenizer_extracted_once_and_reused(self):
        path = self.build()
        tok = Path(modelfile.model_config(path, str(self.engine_bin), gpus=1)["tokenizer"])
        for n, b in TOKENIZER.items():
            self.assertEqual((tok / n).read_bytes(), b)
        self.assertEqual([p.name for p in tok.parent.iterdir()], ["tokenizer"])
        (tok / "vocab.json").write_bytes(b"{}")                # a present directory is not rewritten
        modelfile.model_config(path, str(self.engine_bin), gpus=1)
        self.assertEqual((tok / "vocab.json").read_bytes(), b"{}")

    def test_missing_tokenizer_and_bad_file(self):
        w = ModelWriter(str(self.dir / "bare.basalt"))
        w.add_bytes("meta", b"{}")
        w.close()
        with self.assertRaises(ValueError):
            modelfile.model_config(str(self.dir / "bare.basalt"), str(self.engine_bin), gpus=1)
        (self.dir / "junk").write_bytes(b"x" * 5000)
        with self.assertRaises(ValueError):
            modelfile.model_config(str(self.dir / "junk"), str(self.engine_bin), gpus=1)


class Peer(Tmp):
    def test_second_gpu_becomes_the_peer(self):
        path = self.build()
        cfg = modelfile.model_config(path, str(self.engine_bin), gpus=2)
        if not modelfile.peer_tier_supported():      # the engine refuses a peer tier here, so nothing is offered
            self.assertNotIn("--peer-device", cfg["args"])
            return
        self.assertEqual(cfg["args"][-2:], ["--peer-device", "1"])
        self.assertEqual(gpus_in_play(cfg, cfg["args"]), [0, 1])
        self.assertNotIn("--peer-device", modelfile.model_config(path, str(self.engine_bin), gpus=1)["args"])

    def test_vision_tells_the_engine(self):
        path = self.build(vision=True)
        (self.engine_bin.parent / "basalt-vision").write_text("")
        cfg = modelfile.model_config(path, str(self.engine_bin), gpus=1, vision=True)
        self.assertIn("vision", cfg)
        self.assertIn("--vision", cfg["args"])

    def test_engine_words_follow_the_model_as_given(self):
        path = self.build()
        args = modelfile.model_config(path, str(self.engine_bin), gpus=1,
                                      engine_words=["--spec", "0", "--mtp-max-t", "4", "--peer-exclusive"])["args"]
        self.assertEqual(args, ["--model", path, "--spec", "0", "--mtp-max-t", "4", "--peer-exclusive"])
        args = modelfile.model_config(path, str(self.engine_bin), gpus=1,
                                      engine_words=["--control-vector-layer-range", "4", "44"])["args"]
        self.assertEqual(args, ["--model", path, "--control-vector-layer-range", "4", "44"])

    def test_a_given_peer_device_replaces_the_default(self):
        path = self.build()
        args = modelfile.model_config(path, str(self.engine_bin), gpus=2, engine_words=["--peer-device", "-1"])["args"]
        self.assertEqual(args, ["--model", path, "--peer-device", "-1"])

    def test_plain_pack_gives_only_the_model(self):
        path = self.build()
        self.assertEqual(modelfile.model_config(path, str(self.engine_bin), gpus=1)["args"], ["--model", path])

    def test_engine_args_in_an_old_pack_are_ignored(self):
        path = self.build(meta={**META, "engine_args": ["--spec", "7", "--peer-device", "-1"]})
        cfg = modelfile.model_config(path, str(self.engine_bin), gpus=2)
        peer = ["--peer-device", "1"] if modelfile.peer_tier_supported() else []
        self.assertEqual(cfg["args"], ["--model", path] + peer)
        self.assertEqual(cfg["sampling"], META["sampling"])

    def test_count_respects_visible_devices_and_nvidia_smi(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            self.assertEqual(modelfile.gpu_count(), 1)
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
            self.assertEqual(modelfile.gpu_count(), 2)
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}):
            self.assertEqual(modelfile.gpu_count(), 0)
        env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        out = mock.Mock(stdout="GPU 0: RTX (UUID: a)\nGPU 1: RTX (UUID: b)\n")
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("subprocess.run", return_value=out):
            self.assertEqual(modelfile.gpu_count(), 2)
            path = self.build()
            peer = ["--peer-device", "1"] if modelfile.peer_tier_supported() else []
            self.assertEqual(modelfile.model_config(path, str(self.engine_bin))["args"], ["--model", path] + peer)
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("subprocess.run", side_effect=OSError):
            self.assertEqual(modelfile.gpu_count(), 0)


class Vision(Tmp):
    def test_sidecar_beside_the_engine_bin_and_sections_unpacked(self):
        path = self.build(vision=True)
        side = self.engine_bin.parent / "basalt-vision"
        side.write_text("")
        v = modelfile.model_config(path, str(self.engine_bin), gpus=1, vision=True)["vision"]
        self.assertEqual(v["vision_bin"], str(side.resolve()))
        self.assertEqual(Path(v["mmproj"]).read_bytes(), b"MMPROJ" * 100)
        self.assertEqual(Path(v["model"]).read_bytes(), b"VOCAB")
        self.assertEqual([p.name for p in Path(v["mmproj"]).parent.iterdir() if ".gguf." in p.name], [])

    def test_off_by_default(self):
        path = self.build(vision=True)
        (self.engine_bin.parent / "basalt-vision").write_text("")
        cfg = modelfile.model_config(path, str(self.engine_bin), gpus=1)
        self.assertNotIn("vision", cfg)
        self.assertNotIn("--vision", cfg["args"])

    def test_gpu_or_cpu_encoder(self):
        path = self.build(vision=True)
        (self.engine_bin.parent / "basalt-vision").write_text("")
        self.assertTrue(modelfile.model_config(path, str(self.engine_bin), gpus=1, vision=True)["vision"]["gpu"])
        self.assertFalse(modelfile.model_config(path, str(self.engine_bin), gpus=1, vision="cpu")["vision"]["gpu"])

    def test_modes_name_the_encoder(self):
        path = self.build(vision=True)
        (self.engine_bin.parent / "basalt-vision").write_text("")
        for mode, name in (("on", "on"), (True, "on"), ("peer", "peer"), ("cpu", "cpu"), ("dynamic", "dynamic")):
            self.assertEqual(modelfile.model_config(path, str(self.engine_bin), gpus=1, vision=mode)["vision"]["mode"], name)

    def test_dynamic_gives_the_engine_the_projector_and_needs_no_sidecar(self):
        path = self.build(vision=True)                      # no basalt-vision next to the engine
        cfg = modelfile.model_config(path, str(self.engine_bin), gpus=1, vision="dynamic")
        at = cfg["args"].index("--vision-mmproj")
        self.assertEqual(cfg["args"][at + 1], cfg["vision"]["mmproj"])
        self.assertIn("--vision", cfg["args"])
        self.assertNotIn("--vision-mmproj", modelfile.model_config(path, str(self.engine_bin), gpus=1)["args"])
        with self.assertRaisesRegex(ValueError, "basalt-vision"):
            modelfile.model_config(path, str(self.engine_bin), gpus=1, vision="peer")

    def test_error_when_the_sidecar_is_missing(self):
        path = self.build(vision=True)
        with self.assertRaisesRegex(ValueError, "basalt-vision"):
            modelfile.model_config(path, str(self.engine_bin), gpus=1, vision=True)

    def test_error_when_the_model_has_no_projector(self):
        path = self.build(vision=False)
        (self.engine_bin.parent / "basalt-vision").write_text("")
        with self.assertRaisesRegex(ValueError, "projector"):
            modelfile.model_config(path, str(self.engine_bin), gpus=1, vision="cpu")


class Cli(Tmp):
    def run_main(self, *argv):
        from serve import server
        with mock.patch.object(sys, "argv", ["server", *argv]), mock.patch.object(server, "Server") as srv:
            srv.side_effect = RuntimeError("stop")
            with self.assertRaises((SystemExit, RuntimeError)) as cm:
                server.main()
        return cm.exception

    def test_model_needs_an_engine_binary_and_a_valid_gpu(self):
        path = self.build()
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            e = self.run_main("--model", path)
        self.assertEqual(getattr(e, "code", None), 2)
        self.assertIn("engine binary", err.getvalue())
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            e = self.run_main("--model", path, "--engine-bin", str(self.engine_bin), "--gpu", "x")
        self.assertEqual(getattr(e, "code", None), 2)

    def test_vision_flags_reach_model_config(self):
        from serve import server
        path = self.build(vision=True)
        base = ["--model", path, "--engine-bin", str(self.engine_bin), "--gpu", "0"]
        for flags, want in (([], False), (["--vision", "on"], "on"), (["--vision", "peer"], "peer"),
                            (["--vision", "dynamic"], "dynamic"), (["--vision", "cpu"], "cpu"), (["--vision", "off"], False)):
            with mock.patch.object(modelfile, "model_config", side_effect=ValueError("stop")) as mc, \
                    mock.patch("sys.stderr", io.StringIO()):
                self.run_main(*base, *flags)
            self.assertEqual(mc.call_args.kwargs["vision"], want, flags)

    def test_vision_without_sidecar_is_a_startup_error(self):
        path = self.build(vision=True)
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            e = self.run_main("--model", path, "--engine-bin", str(self.engine_bin), "--gpu", "0", "--vision", "on")
        self.assertEqual(getattr(e, "code", None), 2)
        self.assertIn("basalt-vision", err.getvalue())


if __name__ == "__main__":
    unittest.main()
