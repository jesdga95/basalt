import unittest
from pathlib import Path

from serve.totals import totals_path_for_model, totals_path_of


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


if __name__ == "__main__":
    unittest.main()
