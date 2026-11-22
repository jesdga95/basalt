"""Service.reply_cap: a thinking budget's first pass must not become the reply's output cap (#396, #420).

The arm that matters is `fit-max-tokens` ON: the service then builds every text prompt with clip=True, so a guard that
keys on `ids.clip` (the variant suggested in #396) is a no-op there, and a test with fit-max-tokens off goes green while
the bug is live. Every case below runs with clip on AND off.
"""
from __future__ import annotations

import types
import unittest

from serve.prompt import CTX_SLACK, PromptText
from serve.service import Service

CTX = 262144


def counted(n: int, settled: int | None, clip: bool) -> PromptText:
    """A text prompt the engine has counted: n prompt tokens, `settled` = the cap it settled on for the pass it was sent."""
    p = PromptText(tokenizer=None, text="x", clip=clip)
    p.n, p.max_new = n, settled
    return p


def cap(ids: PromptText, max_new: int, asked: int | None, ctx: int = CTX) -> int:
    svc = types.SimpleNamespace(engine=types.SimpleNamespace(max_context=ctx, known_ctx=0))
    return Service.reply_cap(svc, ids, max_new, asked)


class ReplyCap(unittest.TestCase):
    def test_a_budget_capped_first_pass_does_not_cap_the_reply(self):
        # start line --reasoning-budget-tokens 4096, request max_tokens 16384: the first pass asks 4096 and the engine
        # settles exactly that; the reply's cap must stay 16384 (patw's measurement: 4096 + finish=length as released)
        for clip in (True, False):
            with self.subTest(fit_max_tokens=clip):
                self.assertEqual(cap(counted(1000, 4096, clip), 16384, asked=4096), 16384)

    def test_a_tiny_budget_leaves_room_to_answer(self):
        # request reasoning_budget_tokens 8, max_tokens 512: as released the reply ended at 8 tokens, empty
        for clip in (True, False):
            for budget in (8, 16, 32, 128):
                with self.subTest(fit_max_tokens=clip, budget=budget):
                    self.assertEqual(cap(counted(200, budget, clip), 512, asked=budget), 512)

    def test_an_engine_that_cut_the_pass_to_its_room_is_kept(self):
        # the engine settled BELOW what the pass asked: that is its own context room, not the budget
        for clip in (True, False):
            with self.subTest(fit_max_tokens=clip):
                self.assertEqual(cap(counted(1000, 2000, clip), 16384, asked=4096), 2000)

    def test_fit_max_tokens_cuts_an_overshooting_reply_to_the_room(self):
        n = 200000
        room = CTX - CTX_SLACK - n
        self.assertEqual(cap(counted(n, 4096, True), 131072, asked=4096), room)

    def test_without_fit_max_tokens_an_overshooting_reply_is_the_overflow_error(self):
        with self.assertRaises(ValueError) as e:
            cap(counted(200000, 4096, False), 131072, asked=4096)
        self.assertIn("exceeds the context window", str(e.exception))

    def test_context_unknown_keeps_the_old_behaviour(self):
        self.assertEqual(cap(counted(1000, 4096, True), 16384, asked=4096, ctx=0), 4096)
        self.assertEqual(cap(counted(1000, None, True), 16384, asked=4096, ctx=0), 16384)


if __name__ == "__main__":
    unittest.main()
