"""The decode fixtures are one prompt each, ending where the model starts answering.

    python3 -m unittest tools.test_bench_fixtures
"""
import unittest
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent / "data" / "bench"
# DeepSeek-V4.1 chat mode (thinking off): BOS, <User>, the text, <Assistant>, </think>; no end-of-sentence token
DEEPSEEK_BOS, DEEPSEEK_USER, DEEPSEEK_ASSISTANT, DEEPSEEK_THINK_END, DEEPSEEK_EOS = 0, 128803, 128804, 128822, 1


def ids(path: Path) -> list[int]:
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1, f"{path}: one prompt per file"
    return [int(x) for x in lines[0].split(",")]


class Fixtures(unittest.TestCase):
    def test_every_decode_fixture_is_one_prompt_of_token_ids(self):
        for d in [BENCH, *sorted(p for p in BENCH.iterdir() if p.is_dir())]:
            for name in ("conv", "struct", "prose"):
                with self.subTest(dir=d.name, fixture=name):
                    self.assertTrue(all(i >= 0 for i in ids(d / f"{name}.ids")))

    def test_deepseek_fixtures_end_at_the_generation_header(self):
        for name in ("conv", "struct", "prose"):
            with self.subTest(fixture=name):
                t = ids(BENCH / "deepseek41" / f"{name}.ids")
                self.assertEqual(t[:2], [DEEPSEEK_BOS, DEEPSEEK_USER])
                self.assertEqual(t[-2:], [DEEPSEEK_ASSISTANT, DEEPSEEK_THINK_END])
                self.assertNotIn(DEEPSEEK_EOS, t)


if __name__ == "__main__":
    unittest.main()
