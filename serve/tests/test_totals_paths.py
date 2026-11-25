import math
import unittest
from pathlib import Path

from serve.totals import totals_add_saved, totals_path_for_model, totals_path_of


class TotalsPaths(unittest.TestCase):
    def test_each_model_gets_a_lowercase_file_without_spaces(self):
        base = totals_path_of("", "/m/basalt_config.json")
        # Compared as paths, not as strings: the totals file is a real file name and the platform decides its
        # separator (pathlib compares by parts, so this holds for / and \ alike).
        for model, slug in (("Qwen 3.8 FN", "qwen3.8fn"), ("DeepSeek V4.1 Flash", "deepseekv4.1flash"),
                            ("Inkling 2 Mini", "inkling2mini")):
            self.assertEqual(Path(totals_path_for_model(base, model)),
                             Path(f"/m/basalt_config.totals.{slug}.json"))

    def test_a_name_with_nothing_to_keep_fails(self):
        with self.assertRaises(ValueError):
            totals_path_for_model("/m/c.totals.json", " / ")

    def test_totals_add_saved_ignores_bool_and_non_finite_values(self):
        base = {"since": 1000, "requests": 5, "prompt": 100, "reused": 20, "written": 30,
                "prompt_ms": 10.0, "decode_ms": 20.0}
        saved = {
            "since": float("-inf"),
            "requests": True,
            "prompt": float("nan"),
            "reused": float("inf"),
            "written": False,
            "prompt_ms": 2.5,
            "decode_ms": float("-inf"),
        }
        out = totals_add_saved(dict(base), saved)
        self.assertEqual(out["since"], 1000)
        self.assertEqual(out["requests"], 5)
        self.assertEqual(out["prompt"], 100)
        self.assertEqual(out["reused"], 20)
        self.assertEqual(out["written"], 30)
        self.assertEqual(out["prompt_ms"], 12.5)
        self.assertEqual(out["decode_ms"], 20.0)
        # bool since is ignored; finite earlier since is applied
        self.assertEqual(totals_add_saved({"since": 1000}, {"since": False})["since"], 1000)
        self.assertEqual(totals_add_saved({"since": 1000}, {"since": float("nan")})["since"], 1000)
        self.assertEqual(totals_add_saved({"since": 1000}, {"since": 500})["since"], 500)
        self.assertTrue(all(math.isfinite(v) for v in out.values()))


if __name__ == "__main__":
    unittest.main()
