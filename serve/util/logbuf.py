"""Recent server and engine output for GET /api/logs: LogBuffer, TeeStream and tail_file."""
from __future__ import annotations

import collections
import os
import re
import threading
import time

from serve.util.loglevel import ERR, WARN, LevelReader

CAPACITY = 2000
MAX_LINE = 2000
ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")
SERVER_PREFIX = "[server] "


class LogBuffer:
    """A thread-safe ring of the last CAPACITY lines with rising ids."""

    def __init__(self, capacity: int = CAPACITY):
        self.lock = threading.Lock()
        self.lines: collections.deque = collections.deque(maxlen=capacity)
        self.last = 0
        self.readers: dict[str, LevelReader] = {}      # per source: a traceback spans lines

    def add(self, src: str, text: str) -> None:
        """Add each non-empty line of `text` (ANSI codes removed, long lines cut) from source `src`. A line's level is
        read from its marker (serve/util/loglevel.py); a `[server] ` prefix is dropped after that."""
        now = round(time.time(), 3)
        for raw in ANSI.sub("", text or "").splitlines():
            msg = raw.rstrip()[:MAX_LINE]
            if not msg.strip():
                continue
            with self.lock:
                lvl = self.readers.setdefault(src, LevelReader()).read(msg)
                self.last += 1
                self.lines.append({"id": self.last, "t": now, "src": src, "lvl": lvl,
                                   "msg": msg.removeprefix(SERVER_PREFIX)})

    def since(self, after_id: int = 0, limit: int = 500) -> tuple[list[dict], int]:
        """Return (the newest `limit` lines above `after_id`, oldest first; the newest id)."""
        with self.lock:
            fresh = [dict(x) for x in self.lines if x["id"] > after_id]
            last = self.last
        return (fresh[-limit:] if limit > 0 else []), last


class TeeStream:
    """Wraps sys.stdout/stderr: passes everything through and adds each complete line to the buffer as `src`."""

    def __init__(self, stream, buffer: LogBuffer, src: str):
        self._stream, self._buffer, self._src = stream, buffer, src
        self._partial = ""
        self._lock = threading.Lock()

    def write(self, text):
        n = self._stream.write(text)
        if isinstance(text, str):
            with self._lock:
                self._partial += text
                *done, self._partial = self._partial.split("\n")
            for line in done:
                self._buffer.add(self._src, line)
        return n

    def __getattr__(self, name):
        return getattr(self._stream, name)


class ColorStream:
    """Wraps a terminal stream: error lines are red and warning lines yellow, by the markers of serve/util/loglevel.py.
    Off a terminal, or with NO_COLOR set, the stream is returned unwrapped. Text that moves the cursor (escape
    codes, carriage returns) is passed through at once so in-place redraws keep working."""

    RED, YELLOW, RESET = "\x1b[31m", "\x1b[33m", "\x1b[0m"

    @classmethod
    def wrap(cls, stream):
        try:
            tty = stream.isatty()
        except (AttributeError, ValueError):
            tty = False
        return cls(stream) if tty and not os.environ.get("NO_COLOR") else stream

    def __init__(self, stream):
        self._stream = stream
        self._partial = ""
        self._lock = threading.Lock()
        self._reader = LevelReader()

    def paint(self, line: str) -> str:
        lvl = self._reader.read(ANSI.sub("", line))
        color = self.RED if lvl == ERR else self.YELLOW if lvl == WARN else ""
        return color + line + self.RESET if color else line

    def write(self, text):
        if not isinstance(text, str):
            return self._stream.write(text)
        with self._lock:
            *done, rest = (self._partial + text).split("\n")
            self._partial = ""
            if "\x1b" in rest or "\r" in rest:        # a redraw in progress: no waiting for its newline
                done.append(rest)
                out = "\n".join(self.paint(x) if i < len(done) - 1 else x for i, x in enumerate(done))
            else:
                self._partial = rest
                out = "".join(self.paint(x) + "\n" for x in done)
            if out:
                self._stream.write(out)
        return len(text)

    def flush(self):
        with self._lock:
            if self._partial:
                self._stream.write(self._partial)
                self._partial = ""
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def tail_file(buffer: LogBuffer, path: str, src: str = "eng", poll_s: float = 0.25) -> None:
    """Follow a file from its end, adding complete lines to the buffer; waits for a missing file and rereads a
    shrunk one from the start."""
    try:
        pos = os.path.getsize(path)
    except OSError:
        pos = 0
    partial = b""
    while True:
        try:
            size = os.path.getsize(path)
            if size < pos:
                pos, partial = 0, b""
            if size > pos:
                with open(path, "rb") as f:
                    f.seek(pos)
                    data = f.read(min(size - pos, 1 << 20))
                pos += len(data)
                *done, partial = (partial + data).split(b"\n")
                for line in done:
                    buffer.add(src, line.decode("utf-8", "replace"))
                continue
        except OSError:
            pass
        time.sleep(poll_s)
