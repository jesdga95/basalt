"""--dfly --into: a synthetic safetensors drafter joins a synthetic hy4flash pack; the record, the offsets, roll-back."""
import json
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import convert  # noqa: E402
import dfly_attach  # noqa: E402
from model_file import ModelFile, ModelWriter  # noqa: E402

CONFIG = {"text_config": {"dfly_num_hidden_layers": 2, "dfly_mask_token_id": 7, "dfly_target_kv_layers": [3, 5],
                          "dfly_target_index_layer": [3, 5], "dfly_target_index_topk": 2048,
                          "dfly_context_local_window": None, "dfly_mtp_fusion": "slot0",
                          "dfly_enable_hidden_correction": True}}


def bf16(n, seed):
    r = random.Random(seed)
    return b"".join(struct.pack("<f", r.gauss(0, 1))[2:] for _ in range(n))


def write_safetensors(path, names, extra=(), dtype="BF16"):
    head, data = {}, b""
    for i, n in enumerate(list(names) + list(extra)):
        raw = bf16(6, i)
        head[n] = {"dtype": dtype, "shape": [2, 3], "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    h = json.dumps(head).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + data)


class DflyAttachTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.pack = self.t / "p.basalt"
        w = ModelWriter(str(self.pack))
        w.add_bytes("dense", b"d" * 100)
        w.add_bytes("meta", json.dumps({"arch": "hy4flash", "hf_config": CONFIG, "sources": []}).encode())
        w.close()
        self.names = [dfly_attach.PREFIX + n for n in sorted(dfly_attach.wanted(2))]
        self.st = self.t / "dfly.safetensors"

    def args(self, **kw):
        d = dict(into=str(self.pack), dfly=str(self.st), undo_dir=str(self.t / "undo"), roll_back=False)
        d.update(kw)
        return type("A", (), d)()

    def test_the_file_is_stored_whole_and_each_tensor_is_found_by_its_offset(self):
        write_safetensors(self.st, self.names, extra=["model.layers.0.other"])
        before = ModelFile(str(self.pack))
        dense = before.read("dense")
        dfly_attach.add_dfly(self.args(), convert.source_record)
        m = ModelFile(str(self.pack))
        self.assertTrue(m.verify("dfly"))
        self.assertEqual(m.read("dense"), dense)
        self.assertEqual(m.read("dfly"), self.st.read_bytes())
        meta = json.loads(m.read("meta"))
        rec = meta["dfly2"]
        self.assertEqual((rec["layers"], rec["mask_token_id"], rec["target_kv_layers"]), (2, 7, [3, 5]))
        self.assertEqual(len(rec["tensors"]), len(self.names))
        self.assertNotIn("other", " ".join(rec["tensors"]))
        sec = m.read("dfly")
        for name, t in rec["tensors"].items():
            self.assertEqual(sec[t["offset"]:t["offset"] + t["bytes"]], bf16(6, self.names.index(dfly_attach.PREFIX + name)))
        convert.add_dspark(type("A", (), dict(into=str(self.pack), dspark=None, undo_dir=str(self.t / "undo"), roll_back=True))())
        self.assertEqual(sorted(ModelFile(str(self.pack)).names()), sorted(before.names()))

    def test_the_released_config_keeps_the_dfly_keys_at_its_top_level(self):
        write_safetensors(self.st, self.names)
        rec = dfly_attach.dfly_record(self.st, {"text_config": {}, **CONFIG["text_config"]})
        self.assertEqual((rec["layers"], rec["mask_token_id"], rec["target_index_layers"]), (2, 7, [3, 5]))

    def test_a_missing_tensor_a_wrong_dtype_and_a_second_drafter_are_refused(self):
        write_safetensors(self.st, self.names[:-1])
        with self.assertRaises(SystemExit):
            dfly_attach.add_dfly(self.args())
        write_safetensors(self.st, self.names, dtype="F32")
        with self.assertRaises(SystemExit):
            dfly_attach.add_dfly(self.args())
        self.assertFalse((self.t / "undo").exists())
        write_safetensors(self.st, self.names)
        dfly_attach.add_dfly(self.args())
        with self.assertRaises(SystemExit):
            dfly_attach.add_dfly(self.args(undo_dir=str(self.t / "undo2")))


if __name__ == "__main__":
    unittest.main()
