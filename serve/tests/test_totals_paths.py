import unittest

from serve.totals import totals_path_for_model, totals_path_of


class TotalsPaths(unittest.TestCase):
    def test_each_model_gets_a_lowercase_file_without_spaces(self):
        base = totals_path_of("", "/m/basalt_config.json")
        self.assertEqual(totals_path_for_model(base, "Qwen 3.8 FN"), "/m/basalt_config.totals.qwen3.8fn.json")
        self.assertEqual(totals_path_for_model(base, "DeepSeek V4.1 Flash"), "/m/basalt_config.totals.deepseekv4.1flash.json")
        self.assertEqual(totals_path_for_model(base, "Inkling 2 Mini"), "/m/basalt_config.totals.inkling2mini.json")

    def test_a_name_with_nothing_to_keep_fails(self):
        with self.assertRaises(ValueError):
            totals_path_for_model("/m/c.totals.json", " / ")


if __name__ == "__main__":
    unittest.main()
