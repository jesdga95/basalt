"""The level of a log line, read from the marker its writer put on it. No message text is searched for words.

Markers, one per kind of writer:
  engine   basalt <area>: <text>            info
           basalt <area>: warning: <text>   warning
           basalt <area>: error: <text>     error      (older engines: "basalt warning: <text>", "basalt error: <text>")
  server   [server] <text>                  info
           [server] warning: <text>         warning
           [server] error: <text>           error
  python   a "Traceback (most recent call last):" block, and the exception line that ends it   error
  other    anything else (driver and library output, engine tables)                             info
"""
from __future__ import annotations

import re
from typing import NamedTuple

INFO, WARN, ERR = "info", "warn", "err"
_NAMES = {"warning": WARN, "error": ERR}
ENGINE = re.compile(r"(?P<tag>basalt(?:-[a-z]+)?(?: [\w-]+)*):(?: (?P<lvl>warning|error):)? (?P<text>.*)", re.S)
LEGACY = re.compile(r"basalt (?P<lvl>warning|error): (?P<text>.*)", re.S)
SERVER = re.compile(r"\[server\] (?:(?P<lvl>warning|error): )?")
TRACEBACK = "Traceback (most recent call last):"
CHAINED = ("During handling of the above exception", "The above exception was the direct cause")


class EngineLine(NamedTuple):
    tag: str        # "basalt generate"; "basalt" for a line of no area
    level: str      # INFO, WARN or ERR
    text: str       # the message, without the tag and the marker


def engine_line(line: str) -> EngineLine | None:
    """The parts of an engine log line, or None when the line is not one."""
    m = LEGACY.fullmatch(line)
    if m:
        return EngineLine("basalt", _NAMES[m["lvl"]], m["text"])
    m = ENGINE.fullmatch(line)
    if m:
        return EngineLine(m["tag"], _NAMES.get(m["lvl"], INFO), m["text"])
    return None


def server_level(line: str) -> str | None:
    """The level a `[server] ` line states, or None when the line is not a server line."""
    m = SERVER.match(line)
    return None if m is None else _NAMES.get(m["lvl"], INFO)


class LevelReader:
    """Reads the level of the lines of one stream, in order (a traceback spans lines)."""

    def __init__(self):
        self._in_traceback = False

    def read(self, line: str) -> str:
        if line.startswith(TRACEBACK):
            self._in_traceback = True
            return ERR
        if line.startswith(CHAINED):
            self._in_traceback = False
            return ERR
        if self._in_traceback:
            if line[:1] not in (" ", "\t") and line.strip():     # the exception line ends the block
                self._in_traceback = False
            return ERR
        e = engine_line(line)
        if e is not None:
            return e.level
        s = server_level(line)
        return INFO if s is None else s
