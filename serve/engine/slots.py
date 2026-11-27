"""First-come admission to the batch slots, with bounded overtaking."""
from __future__ import annotations

import itertools

OVERTAKE_MAX = 4        # times a waiting request may be passed by later ones before it must go next


class Waiter:
    __slots__ = ("ticket", "skipped")

    def __init__(self, ticket: int):
        self.ticket, self.skipped = ticket, 0


class RoomWaiter:
    """A request's place in the line for KV room: `arrival` orders it, `ends` is the slot-end count when it last asked."""
    __slots__ = ("arrival", "ends")

    def __init__(self, arrival: int):
        self.arrival, self.ends = arrival, None


class AdmissionQueue:
    """Requests waiting for a free slot, oldest first. The oldest takes the next free slot; a later one may go
    first only when `hit` says its prompt resumes a slot's held prefix, and not past the oldest's overtake budget.
    All calls are made under the caller's lock."""

    def __init__(self, overtake_max: int = OVERTAKE_MAX):
        self.overtake_max, self.waiters, self._tickets = overtake_max, [], itertools.count()
        self.room, self._arrivals = [], itertools.count()

    def join(self) -> Waiter:
        w = Waiter(next(self._tickets))
        self.waiters.append(w)
        return w

    def leave(self, w: Waiter) -> None:
        if w in self.waiters:
            self.waiters.remove(w)

    def may_go(self, w: Waiter, hit: bool = False) -> bool:
        """Whether `w` may take a free slot now; a granted overtake is charged to the oldest waiter."""
        head = self.waiters[0] if self.waiters else w
        if w is head:
            return True
        if hit and head.skipped < self.overtake_max:
            head.skipped += 1
            return True
        return False

    def arrive(self) -> RoomWaiter:
        """A request's place for KV room, fixed for its life: older requests are served first."""
        return RoomWaiter(next(self._arrivals))

    def _room_enter(self, w: RoomWaiter) -> None:
        if w not in self.room:
            self.room.append(w)
            self.room.sort(key=lambda x: x.arrival)

    def room_may_ask(self, w: RoomWaiter, ends: int, idle: bool) -> bool:
        """Whether `w` may ask the engine for room now. With nobody waiting for room any request may; once one waits,
        only the oldest asks, and only after a slot ended since its last ask (or when no slot runs)."""
        if not self.room:
            return True
        self._room_enter(w)
        return self.room[0] is w and (w.ends is None or w.ends != ends or idle)

    def room_refused(self, w: RoomWaiter, ends: int) -> None:
        """The engine had no room for `w`: it keeps its place and asks again after the next slot end."""
        self._room_enter(w)
        w.ends = ends

    def room_leave(self, w: RoomWaiter) -> None:
        """`w` was admitted, or gave up; the next oldest asks at once."""
        if w not in self.room:
            return
        head = self.room[0] is w
        self.room.remove(w)
        if head and self.room:
            self.room[0].ends = None

    def __len__(self) -> int:
        return len(self.waiters)
