"""The model cache: the budget and LRU books."""
from __future__ import annotations

import unittest

from serve.parking import ModelCache


class Books(unittest.TestCase):
    def test_a_budget_below_one_mib_is_refused(self):
        with self.assertRaises(ValueError):
            ModelCache(0)

    def test_what_fits_evicts_nothing(self):
        c = ModelCache(1000)
        c.add("a", 400)
        self.assertEqual(c.eviction_for(600), [])
        self.assertEqual(c.total_mib(), 400)

    def test_the_least_recently_used_goes_first_and_only_as_many_as_needed(self):
        c = ModelCache(1000)
        for k, mib in (("a", 300), ("b", 300), ("c", 300)):
            c.add(k, mib)
        self.assertEqual(c.eviction_for(200), ["a"])
        self.assertEqual(c.eviction_for(500), ["a", "b"])
        self.assertEqual(c.eviction_for(1000), ["a", "b", "c"])

    def test_parking_again_makes_it_the_most_recent(self):
        c = ModelCache(1000)
        c.add("a", 300)
        c.add("b", 300)
        c.add("a", 350)
        self.assertEqual(c.keys(), ["b", "a"])
        self.assertEqual(c.total_mib(), 650)

    def test_more_than_the_budget_never_fits(self):
        c = ModelCache(1000)
        c.add("a", 100)
        self.assertIsNone(c.eviction_for(1001))
        self.assertEqual(c.eviction_for(1000), ["a"])

    def test_taking_a_model_returns_what_it_held(self):
        c = ModelCache(1000)
        c.add("a", 300)
        self.assertEqual(c.take("a"), 300)
        self.assertNotIn("a", c)
        self.assertEqual(c.total_mib(), 0)

    def test_over_budget_names_the_oldest_until_it_fits(self):
        c = ModelCache(1000)
        for k, mib in (("a", 500), ("b", 500), ("c", 500)):
            c.add(k, mib)
        self.assertEqual(c.over_budget(), ["a"])
        c.add("d", 900)
        self.assertEqual(c.over_budget(), ["a", "b", "c"])

    def test_low_ram_names_the_oldest_only_when_it_is_low_and_known(self):
        c = ModelCache(1000)
        self.assertIsNone(c.short_of_ram(10, 100))            # nothing parked
        c.add("a", 300)
        c.add("b", 300)
        self.assertEqual(c.short_of_ram(99, 100), "a")
        self.assertIsNone(c.short_of_ram(100, 100))
        self.assertIsNone(c.short_of_ram(None, 100))          # unreadable: nothing is guessed


if __name__ == "__main__":
    unittest.main()
