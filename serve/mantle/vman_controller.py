"""ModelController backed by `vman mode <profile>`, the only supported way to switch models.

Nothing here touches containers directly and `--force` is never passed. The subprocess runner is injectable so the
controller is testable without a GPU; the real runner is used only when a deployment constructs it on purpose."""
from __future__ import annotations

import json
import subprocess
import threading
from typing import Callable, Mapping, Optional, Sequence

VMAN = "/data/tools/bin/vman"
STATE_FILE = "/data/tools/sre/vram_current_mode.json"   # what `vman mode` writes atomically after a verified switch
PROFILES = frozenset({"strata", "llm", "image", "video", "voice", "quant", "off"})

Runner = Callable[[Sequence[str], float], "tuple[int, str]"]


class VmanError(RuntimeError):
    pass


def vman_state_mode(path: str = STATE_FILE) -> Optional[str]:
    """The mode vman last switched to, or None when the state file is missing or unreadable. vman itself assumes
    `llm` then; a controller must not switch (or restore) on a guess, so it gets None and refuses."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            mode = json.load(f).get("mode")
    except (OSError, ValueError, AttributeError):
        return None
    return mode if isinstance(mode, str) and mode in PROFILES else None


def subprocess_runner(argv: Sequence[str], timeout: float) -> "tuple[int, str]":
    try:
        p = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise VmanError(f"{argv[1:]} did not finish in {timeout:.0f}s; check `vman status` before retrying") from None
    return p.returncode, (p.stdout + p.stderr)[-2000:]


class VmanModelController:
    """load(model) switches to the model's profile; release(model) puts back the profile that was active before the
    first load, but only if this controller's profile is still the active one (another job may have taken over)."""

    def __init__(self, profiles: Mapping[str, str], current_mode: Callable[[], Optional[str]],
                 runner: Runner = subprocess_runner, timeout_s: float = 600.0, caller: str = "COO"):
        bad = {m: p for m, p in profiles.items() if p not in PROFILES}
        if bad:
            raise ValueError(f"unknown vman profile(s): {bad}")
        self.profiles, self.current_mode, self.runner = dict(profiles), current_mode, runner
        self.timeout_s, self.caller = timeout_s, caller
        self._previous: Optional[str] = None
        self._active: Optional[str] = None
        self._lock = threading.Lock()

    def _switch(self, profile: str) -> None:
        code, out = self.runner([VMAN, "mode", "--caller", self.caller, profile], self.timeout_s)
        if code != 0:
            raise VmanError(f"vman mode {profile} failed ({code}): {out.strip()[-300:]}")

    def load(self, model: str) -> None:
        profile = self.profiles.get(model)
        if profile is None:
            raise VmanError(f"no vman profile for model {model!r}")
        with self._lock:
            now = self.current_mode()
            if now == profile:
                self._active = profile
                return
            if now is None or now not in PROFILES:
                raise VmanError(f"current vman mode is unknown ({now!r}); not switching on a guess")
            self._switch(profile)                 # a failed switch leaves nothing recorded to restore
            if self._previous is None:
                self._previous = now
            self._active = profile

    def release(self, model: str) -> None:
        with self._lock:
            if self._active is None:
                return
            if self._previous is not None and self.current_mode() == self._active:
                self._switch(self._previous)
            self._active, self._previous = None, None


def default_controller(profiles: Mapping[str, str], **kw) -> VmanModelController:
    """The controller wired to the real `vman` and its state file."""
    return VmanModelController(profiles, vman_state_mode, **kw)
