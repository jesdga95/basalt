"""Tests for tools/quality_suite: the scorer's KLD, top-1 and PPL on a toy reference with known values, the Basalt rows
reader, the role masks and the smoke slice. CPU only; vLLM is not needed."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import score  # noqa: E402

# Two conversations: BOS plus assistant tokens (roles 0 1 1 1 and 0 1 1). Rows exist where the next token has a role.
TOKENS = np.array([2, 10, 11, 12, 2, 10, 13], dtype=np.int32)
ROLES = np.array([0, 1, 1, 1, 0, 1, 1], dtype=np.uint8)
CONV_OFF = np.array([0, 4, 7], dtype=np.int64)
ROWS = [0, 1, 2, 4, 5]          # global positions of the rows (distribution over token i+1)
REF_P = (0.6, 0.3)              # stored probabilities of ids 10 and 11; leftover 0.1
REF_ID = (10, 11)


def write_reference(path):
    n = len(TOKENS)
    row_off = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        row_off[i + 1] = row_off[i] + (2 if i in ROWS else 0)
    topk_ids = np.array(REF_ID * len(ROWS), dtype=np.int32)
    topk_lp = np.log(np.array(REF_P * len(ROWS))).astype(np.float32)
    leftover = np.zeros(n, dtype=np.float32)
    leftover[ROWS] = 0.1
    save_file(dict(tokens=TOKENS, roles=ROLES, conv_off=CONV_OFF, row_off=row_off, topk_ids=topk_ids, topk_lp=topk_lp,
                   leftover=leftover), str(path))


def write_test_bin(path, test_p, test_left, top1):
    """llama_project's output: per row k float log-probs, the leftover, the argmax (int32)."""
    out = bytearray()
    for p, left, t in zip(test_p, test_left, top1):
        out += np.log(np.array(p, dtype=np.float64)).astype(np.float32).tobytes()
        out += np.float32(left).tobytes() + np.int32(t).tobytes()
    Path(path).write_bytes(bytes(out))


def kld(p, left_p, q, left_q):
    p, q = np.array(p), np.array(q)
    return float((p * np.log(p / q)).sum() + left_p * np.log(left_p / left_q))


class ScorerTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.ref = Path(self.dir.name) / "ref.safetensors"
        write_reference(self.ref)

    def tearDown(self):
        self.dir.cleanup()

    def stats(self, d, boot=50):
        pr = score.per_row(d)
        out, info = score.summarize(d, pr, d["role"] == 1, boot)
        return pr, out, info

    def test_identical_model_has_zero_kld_and_full_top1(self):
        test = Path(self.dir.name) / "same.bin"
        write_test_bin(test, [REF_P] * 5, [0.1] * 5, [10] * 5)
        pr, out, info = self.stats(score.load(str(self.ref), str(test)))
        self.assertEqual(info["n_rows"], 5)
        self.assertEqual(info["n_convs"], 2)
        self.assertAlmostEqual(out["KLD"][0], 0.0, places=6)
        self.assertAlmostEqual(out["top1"][0], 1.0)
        self.assertAlmostEqual(out["ppl_ratio"][0], 1.0, places=6)

    def test_known_kld_top1_and_ppl_ratio(self):
        q = [(0.5, 0.3)] * 3 + [(0.6, 0.25)] * 2   # leftovers 0.2 and 0.15
        left = [0.2] * 3 + [0.15] * 2
        top1 = [10, 10, 11, 10, 10]                 # reference top is id 10: 4 of 5 agree
        test = Path(self.dir.name) / "q.bin"
        write_test_bin(test, q, left, top1)
        pr, out, info = self.stats(score.load(str(self.ref), str(test)))
        want = [kld(REF_P, 0.1, a, b) for a, b in zip(q, left)]
        np.testing.assert_allclose(pr["kld"], want, rtol=1e-5)
        self.assertAlmostEqual(out["KLD"][0], float(np.mean(want)), places=6)
        self.assertAlmostEqual(out["top1"][0], 0.8)
        # targets are 10, 11, 12, 10, 13: rows 0 and 3 (id 10) and row 1 (id 11) are inside the stored ids
        self.assertEqual(list(pr["has"]), [True, True, False, True, False])
        want_ln = (np.log(0.6 / 0.5) + np.log(0.3 / 0.3) + np.log(0.6 / 0.6)) / 3
        self.assertAlmostEqual(out["ln_ppl_ratio"][0], float(want_ln), places=6)
        self.assertAlmostEqual(out["ppl_ratio"][0], float(np.exp(want_ln)), places=6)
        self.assertAlmostEqual(pr["dp"][0], 0.5 - 0.6, places=6)

    def test_paired_difference_is_exact(self):
        a, b = Path(self.dir.name) / "a.bin", Path(self.dir.name) / "b.bin"
        write_test_bin(a, [(0.5, 0.3)] * 5, [0.2] * 5, [10] * 5)
        write_test_bin(b, [(0.55, 0.3)] * 5, [0.15] * 5, [10, 10, 10, 11, 11])
        dA, dB = score.load(str(self.ref), str(a)), score.load(str(self.ref), str(b))
        mask = dA["role"] == 1
        res = score.paired(dA, score.per_row(dA), dB, score.per_row(dB), mask, 50)
        self.assertAlmostEqual(res["KLD"]["diff"], res["KLD"]["A"] - res["KLD"]["B"], places=9)
        self.assertAlmostEqual(res["top1"]["A"], 1.0)
        self.assertAlmostEqual(res["top1"]["B"], 0.6)

    def test_basalt_rows_match_the_binary_path(self):
        q = [(0.5, 0.3)] * 3 + [(0.6, 0.25)] * 2
        left = [0.2] * 3 + [0.15] * 2
        test = Path(self.dir.name) / "q.bin"
        write_test_bin(test, q, left, [10] * 5)
        want = score.per_row(score.load(str(self.ref), str(test)))["kld"]
        files = []
        for c, rows in enumerate([[0, 1, 2], [3, 4]]):
            f = Path(self.dir.name) / f"conv_{c}.txt"
            lines = []
            for r in rows:
                pos = ROWS[r] - int(CONV_OFF[c])
                target = int(TOKENS[ROWS[r] + 1])
                lp = np.log(np.array(q[r]))
                # pos target lp_target lse id:lp ... (the likeliest first)
                lines.append(f"{pos} {target} 0.0 0.0 10:{lp[0]:.9f} 11:{lp[1]:.9f} 99:-9.0")
            f.write_text("\n".join(lines) + "\n")
            files.append(str(f))
        d = score.load_basalt(str(self.ref), files)
        got = score.per_row(d)
        # a Basalt row's leftover is 1 - sum p over the reference ids, so it equals the test's leftover above
        np.testing.assert_allclose(got["kld"], want, rtol=1e-5)
        self.assertEqual(list(d["ttop1"]), [10] * 5)


class RolesTest(unittest.TestCase):
    S = {"<|turn>": 1, "<turn|>": 2, "<|channel>": 3, "<channel|>": 4, "<|tool_response>": 5, "<tool_response|>": 6, "<bos>": 7}
    MODEL = 50

    def test_assistant_span_and_tool_response(self):
        S = self.S
        # bos, user turn, model turn with the empty thought header, text, tool stop token, injected response, text, end of turn
        ids = [S["<bos>"], S["<|turn>"], 40, 41, 42, S["<turn|>"], 43, S["<|turn>"], self.MODEL, 43, S["<|channel>"], 44, S["<channel|>"],
               60, 61, S["<|tool_response>"], 70, 71, S["<tool_response|>"], 62, S["<turn|>"]]
        roles = common.token_roles(ids, S, self.MODEL).tolist()
        self.assertEqual(roles[0], 0)
        self.assertTrue(all(r == 2 for r in roles[1:13]))   # user turn, template and the empty thought header
        self.assertEqual(roles[13:16], [1, 1, 1])             # text and the generated tool-response stop token
        self.assertEqual(roles[16:19], [2, 2, 2])             # the injected response
        self.assertEqual(roles[19:], [1, 1])                  # text and the end of turn

    def test_smoke_slice(self):
        off = np.cumsum([0] + [100] * 41 + [6000])             # 42 conversations, the last one long
        self.assertEqual(common.smoke_convs(off), [0, 20, 40])
        self.assertEqual(common.smoke_convs(off, stride=20, max_len=50), [])


if __name__ == "__main__":
    unittest.main()
