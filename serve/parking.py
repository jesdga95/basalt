"""The books of the model cache: which models are parked in RAM, what each holds, and who goes first.

A parked model is an engine process asleep (see serve/engine/suspend.py): its VRAM is free, its RAM stays. The cache
holds no memory itself; it decides what the budget allows. Pure bookkeeping, so it is tested without an engine.
"""
from __future__ import annotations

import collections


class ModelCache:
    """Parked models with the MiB each holds, least recently used first, against one budget."""

    def __init__(self, budget_mib: int):
        if budget_mib < 1:
            raise ValueError("the model cache budget must be at least 1 MiB")
        self.budget_mib = budget_mib
        self._held: collections.OrderedDict[object, int] = collections.OrderedDict()

    def __contains__(self, key) -> bool:
        return key in self._held

    def __len__(self) -> int:
        return len(self._held)

    def total_mib(self) -> int:
        return sum(self._held.values())

    def held_mib(self, key) -> int:
        return self._held[key]

    def keys(self) -> list:
        """The parked models, least recently used first."""
        return list(self._held)

    def eviction_for(self, need_mib: int) -> list | None:
        """The models to evict, least recently used first, so that `need_mib` more fits the budget; [] when it fits as it
        is; None when it cannot fit even with the cache empty."""
        if need_mib > self.budget_mib:
            return None
        out, total = [], self.total_mib()
        for key in self._held:
            if total + need_mib <= self.budget_mib:
                break
            out.append(key)
            total -= self._held[key]
        return out

    def add(self, key, mib: int) -> None:
        """`key` is parked now, holding `mib`; it is the most recently used."""
        self._held.pop(key, None)
        self._held[key] = max(0, int(mib))

    def take(self, key) -> int:
        """`key` leaves the cache (woken or evicted); the MiB it held."""
        return self._held.pop(key)

    def over_budget(self) -> list:
        """The models to evict, least recently used first, when the parked total exceeds the budget (a model held more
        than was estimated)."""
        out, total = [], self.total_mib()
        for key in self._held:
            if total <= self.budget_mib:
                break
            out.append(key)
            total -= self._held[key]
        return out

    def short_of_ram(self, available_mib: int | None, floor_mib: int) -> object | None:
        """The model to evict when the machine's available RAM is under `floor_mib`: the least recently used one. None
        when RAM is fine, unreadable, or nothing is parked."""
        if available_mib is None or available_mib >= floor_mib or not self._held:
            return None
        return next(iter(self._held))
