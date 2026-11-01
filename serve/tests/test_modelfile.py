"""--model FILE.basalt: the config derived from the file, the tokenizer unpacked once, the peer card, vision."""
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from serve.engine import modelfile
from serve.server import engine_args, gpus_in_play
from model_file import ModelFile, ModelWriter  # noqa: E402
from model_adapter import write_adapter  # noqa: E402

META = {"name": "tiny-basalt",
        "sampling": {"temperature": 0.7, "top_p": 0.9}}
TOKENIZER = {"vocab.json": b'{"a": 0}', "merges.txt": b"a a", "token_type.json": b"[1]",
             "chat_template.jinja": b"{{ messages }}"}


class Tmp(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.dir = Path(self.td.name)
        patch = mock.patch.object(modelfile, "cache_root", lambda: self.dir / "cache")
        patch.start()
        self.addCleanup(patch.stop)
        exe = ".exe" if sys.platform == "win32" else ""
        self.engine_bin = self.dir / "bin" / ("basalt" + exe)
        self.vision_bin = self.engine_bin.with_name("basalt-vision" + exe)
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
        cfg = modelfile.model_config(path, str(self.engine_bin))
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
        tok = Path(modelfile.model_config(path, str(self.engine_bin))["tokenizer"])
        self.assertEqual(tok, self.dir / "cache" / key / "tokenizer")

    def test_conversation_dir_follows_the_budget(self):
        path = self.build()
        key = ModelFile(path).sections["meta"][2].hex()[:16]
        conv = str(self.dir / "cache" / key / "conversations")

        def args(*words):
            return modelfile.model_config(path, str(self.engine_bin), engine_words=list(words))["args"]
        self.assertEqual(args(), ["--model", path])
        self.assertEqual(args("--conversation-cache-disk-mib", "0"), ["--model", path, "--conversation-cache-disk-mib", "0"])
        self.assertEqual(args("--conversation-cache-dir", "/x"), ["--model", path, "--conversation-cache-dir", "/x"])
        self.assertEqual(args("--conversation-cache-disk-mib", "32768"),
                         ["--model", path, "--conversation-cache-disk-mib", "32768", "--conversation-cache-dir", conv])
        self.assertEqual(args("--conversation-cache-disk-mib", "5", "--conversation-cache-dir", "/x"),
                         ["--model", path, "--conversation-cache-disk-mib", "5", "--conversation-cache-dir", "/x"])

    def test_tokenizer_extracted_once_and_reused(self):
        path = self.build()
        tok = Path(modelfile.model_config(path, str(self.engine_bin))["tokenizer"])
        for n, b in TOKENIZER.items():
            self.assertEqual((tok / n).read_bytes(), b)
        self.assertEqual([p.name for p in tok.parent.iterdir()], ["tokenizer"])
        (tok / "vocab.json").write_bytes(b"{}")                # a present directory is not rewritten
        modelfile.model_config(path, str(self.engine_bin))
        self.assertEqual((tok / "vocab.json").read_bytes(), b"{}")

    def test_missing_tokenizer_and_bad_file(self):
        w = ModelWriter(str(self.dir / "bare.basalt"))
        w.add_bytes("meta", b"{}")
        w.close()
        with self.assertRaises(ValueError):
            modelfile.model_config(str(self.dir / "bare.basalt"), str(self.engine_bin))
        (self.dir / "junk").write_bytes(b"x" * 5000)
        with self.assertRaises(ValueError):
            modelfile.model_config(str(self.dir / "junk"), str(self.engine_bin))

    def test_adapter_is_not_a_standalone_model(self):
        path = self.build(meta={**META, "kind": "adapter"})
        with self.assertRaisesRegex(ValueError, "not a model"):
            modelfile.model_config(path, str(self.engine_bin))
        self.assertFalse((self.dir / "cache").exists())


class AdapterCatalog(Tmp):
    def adapter(self, filename, name=None):
        path = self.dir / filename
        write_adapter(path, {"some.native.target": ([1], 0, b"\0" * 4)}, name=name)
        return str(path)

    def test_ordered_compositions_and_alias_leave_base_first(self):
        path = self.build()
        a = self.adapter("a.basalt", "alpha")
        b = self.adapter("b.basalt", "beta")
        cfg = modelfile.model_config(path, str(self.engine_bin),
                                     adapters=[[a], [b, a], [a, b]], alias="base")
        self.assertEqual(cfg["model_name"], "base")
        self.assertEqual(cfg["model_names"], ["base", "alpha", "beta+alpha", "alpha+beta"])

    def test_missing_or_empty_names_use_file_stems(self):
        path = self.build(meta={"name": ""})
        a = self.adapter("unnamed.basalt")
        b = self.adapter("empty.basalt", "")
        cfg = modelfile.model_config(path, str(self.engine_bin), adapters=[[a], [b]])
        self.assertEqual(cfg["model_names"], ["tiny", "unnamed", "empty"])

    def test_base_only_catalog_and_alias(self):
        path = self.build()
        self.assertEqual(modelfile.model_config(path, str(self.engine_bin))["model_names"], ["tiny-basalt"])
        cfg = modelfile.model_config(path, str(self.engine_bin), alias="named-base")
        self.assertEqual((cfg["model_name"], cfg["model_names"]), ("named-base", ["named-base"]))

    def test_duplicate_names_are_rejected_before_extraction(self):
        path = self.build()
        a = self.adapter("a.basalt", "same")
        b = self.adapter("b.basalt", "same")
        for groups, alias in (([[a], [b]], None), ([[a]], "same"), ([[a], [a]], None)):
            with self.subTest(groups=groups, alias=alias), self.assertRaises(ValueError):
                modelfile.model_config(path, str(self.engine_bin), adapters=groups, alias=alias)
            self.assertFalse((self.dir / "cache").exists())

    def test_composite_name_collision_is_rejected(self):
        path = self.build()
        a = self.adapter("a.basalt", "a")
        b = self.adapter("b.basalt", "b")
        ab = self.adapter("ab.basalt", "a+b")
        with self.assertRaises(ValueError):
            modelfile.model_config(path, str(self.engine_bin), adapters=[[a, b], [ab]])
        self.assertFalse((self.dir / "cache").exists())

    def test_base_filename_alias_is_reserved_even_with_alias(self):
        path = self.build()
        a = self.adapter("a.basalt", Path(path).name)
        with self.assertRaises(ValueError):
            modelfile.model_config(path, str(self.engine_bin), adapters=[[a]], alias="base")
        self.assertFalse((self.dir / "cache").exists())

    def test_malformed_groups_and_non_adapters_are_rejected_before_extraction(self):
        path = self.build()
        for groups in ([], [None], [[]], ["adapter.basalt"], [[None]], [[1]], [[""]], [[" "]], [[path]]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                modelfile.model_config(path, str(self.engine_bin), adapters=groups)
            self.assertFalse((self.dir / "cache").exists())

    def test_invalid_adapter_metadata_is_rejected(self):
        path = self.build()
        adapter = self.dir / "bad.basalt"
        w = ModelWriter(str(adapter))
        w.add_bytes("meta", b'{"format":"basalt","kind":"adapter","version":1}')
        w.add_bytes("adapter/data", b"\0" * 4)
        w.close()
        with self.assertRaises(ValueError):
            modelfile.model_config(path, str(self.engine_bin), adapters=[[str(adapter)]])
        self.assertFalse((self.dir / "cache").exists())


class Peer(Tmp):
    def test_no_gpu_flag_is_added(self):
        # the engine picks each card's role from --gpu, which the server passes after resolving it
        path = self.build()
        cfg = modelfile.model_config(path, str(self.engine_bin))
        self.assertEqual(cfg["args"], ["--model", path])
        self.assertEqual(gpus_in_play(cfg, cfg["args"]), [0])

    def test_vision_tells_the_engine(self):
        path = self.build(vision=True)
        self.vision_bin.write_text("")
        cfg = modelfile.model_config(path, str(self.engine_bin), vision=True)
        self.assertIn("vision", cfg)
        self.assertIn("--vision", cfg["args"])

    def test_engine_words_follow_the_model_as_given(self):
        path = self.build()
        args = modelfile.model_config(path, str(self.engine_bin),
                                      engine_words=["--spec", "0", "--mtp-max-t", "4", "--peer-exclusive"])["args"]
        self.assertEqual(args, ["--model", path, "--spec", "0", "--mtp-max-t", "4", "--peer-exclusive"])
        args = modelfile.model_config(path, str(self.engine_bin),
                                      engine_words=["--control-vector-layer-range", "4", "44"])["args"]
        self.assertEqual(args, ["--model", path, "--control-vector-layer-range", "4", "44"])

    def test_plain_pack_gives_only_the_model(self):
        path = self.build()
        self.assertEqual(modelfile.model_config(path, str(self.engine_bin))["args"], ["--model", path])

    def test_engine_args_in_an_old_pack_are_ignored(self):
        path = self.build(meta={**META, "engine_args": ["--spec", "7", "--kv", "int8"]})
        cfg = modelfile.model_config(path, str(self.engine_bin))
        self.assertEqual(cfg["args"], ["--model", path])
        self.assertEqual(cfg["sampling"], META["sampling"])


class Vision(Tmp):
    def test_sidecar_beside_the_engine_bin_and_sections_unpacked(self):
        path = self.build(vision=True)
        side = self.vision_bin
        side.write_text("")
        v = modelfile.model_config(path, str(self.engine_bin), vision=True)["vision"]
        self.assertEqual(v["vision_bin"], str(side.resolve()))
        self.assertEqual(Path(v["mmproj"]).read_bytes(), b"MMPROJ" * 100)
        self.assertEqual(Path(v["model"]).read_bytes(), b"VOCAB")
        self.assertEqual([p.name for p in Path(v["mmproj"]).parent.iterdir() if ".gguf." in p.name], [])

    def test_off_by_default(self):
        path = self.build(vision=True)
        self.vision_bin.write_text("")
        cfg = modelfile.model_config(path, str(self.engine_bin))
        self.assertNotIn("vision", cfg)
        self.assertNotIn("--vision", cfg["args"])

    def test_gpu_or_cpu_encoder(self):
        path = self.build(vision=True)
        self.vision_bin.write_text("")
        self.assertTrue(modelfile.model_config(path, str(self.engine_bin), vision="1")["vision"]["gpu"])
        self.assertFalse(modelfile.model_config(path, str(self.engine_bin), vision="cpu")["vision"]["gpu"])

    def test_modes_name_the_encoder(self):
        path = self.build(vision=True)
        self.vision_bin.write_text("")
        for mode, name in (("1", "1"), ("0", "0"), ("cpu", "cpu"), ("dynamic", "dynamic")):
            self.assertEqual(modelfile.model_config(path, str(self.engine_bin), vision=mode)["vision"]["mode"], name)

    def test_dynamic_gives_the_engine_the_projector_and_needs_no_sidecar(self):
        path = self.build(vision=True)                      # no basalt-vision next to the engine
        cfg = modelfile.model_config(path, str(self.engine_bin), vision="dynamic")
        at = cfg["args"].index("--vision-mmproj")
        self.assertEqual(cfg["args"][at + 1], cfg["vision"]["mmproj"])
        self.assertIn("--vision", cfg["args"])
        self.assertNotIn("--vision-mmproj", modelfile.model_config(path, str(self.engine_bin))["args"])
        with self.assertRaisesRegex(ValueError, "basalt-vision"):
            modelfile.model_config(path, str(self.engine_bin), vision="1")

    def test_error_when_the_sidecar_is_missing(self):
        path = self.build(vision=True)
        with self.assertRaisesRegex(ValueError, "basalt-vision"):
            modelfile.model_config(path, str(self.engine_bin), vision="1")

    def test_error_when_the_model_has_no_projector(self):
        path = self.build(vision=False)
        self.vision_bin.write_text("")
        with self.assertRaisesRegex(ValueError, "projector"):
            modelfile.model_config(path, str(self.engine_bin), vision="cpu")


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
        for flags, want in (([], False), (["--vision", "1"], "1"),
                            (["--vision", "dynamic"], "dynamic"), (["--vision", "cpu"], "cpu"), (["--vision", "off"], False)):
            with mock.patch.object(modelfile, "model_config", side_effect=ValueError("stop")) as mc, \
                    mock.patch("sys.stderr", io.StringIO()):
                self.run_main(*base, *flags)
            self.assertEqual(mc.call_args.kwargs["vision"], want, flags)

    def test_vision_without_sidecar_is_a_startup_error(self):
        path = self.build(vision=True)
        err = io.StringIO()
        with mock.patch("serve.server.gpu_card_name", return_value="GeForce RTX 4090"), mock.patch("sys.stderr", err):
            e = self.run_main("--model", path, "--engine-bin", str(self.engine_bin), "--gpu", "0", "--vision", "1")
        self.assertEqual(getattr(e, "code", None), 2)
        self.assertIn("basalt-vision", err.getvalue())

    def test_vision_numbered_card_needs_only_to_exist(self):
        path = self.build(vision=True)
        self.vision_bin.write_text("")
        base = ["--model", path, "--engine-bin", str(self.engine_bin), "--gpu", "0"]

        def plan(engine_bin, value):                     # the engine's own choice for --gpu 0
            return {"devices": [0], "cards": [{"smi": 0}], "notes": [], "warning": None}
        def name(index):                                 # nvidia-smi knows card 9 (an older card), not card 8
            return None if index == 8 else "GeForce RTX 3070"
        with mock.patch("serve.server.gpu_plan", side_effect=plan), \
                mock.patch("serve.server.gpu_card_name", side_effect=name):
            e = self.run_main(*base, "--vision", "9")   # a non-Blackwell card gets through to the Server
        self.assertIsInstance(e, RuntimeError)
        with mock.patch("serve.server.gpu_plan", side_effect=plan), \
                mock.patch("serve.server.gpu_card_name", side_effect=name), \
                mock.patch("sys.stderr", io.StringIO()) as err:
            e = self.run_main(*base, "--vision", "8")   # only a card that is not there fails at startup
        self.assertEqual(getattr(e, "code", None), 2)
        self.assertIn("vision: GPU 8 is not found by nvidia-smi", err.getvalue())


if __name__ == "__main__":
    unittest.main()
