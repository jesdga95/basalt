"""Where a request's images may come from, and the bounded read of each one.

A request from this machine may name any path or URL. Any other request gets `data:` URLs plus the hosts and
directories the operator allowed. `ImageSources()` is that remote policy with nothing allowed, so a caller that
passes no policy reads no file and fetches no URL. Which requests count as local: security.remote_reason.
"""
from __future__ import annotations

import base64
import contextlib
import errno
import http.client
import ipaddress
import ntpath
import os
import queue
import re
import socket
import ssl
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from serve.security import is_ip

IMAGE_MAX_BYTES = 32 << 20                      # one image's bytes, from any source
IMAGE_READ_S = 60.0                             # one source's whole read, name lookup and connect included
USER_AGENT = "local-server"
USAGE = "an image must be a data: URL, an http(s) URL or a local file path"
POLICY = "image source refused by server policy"
DATA_WORKS = "A data: URL always works."
WHY_REMOTE = {"peer": "this client is not on the server's machine",
              "no-peer": "the client's address is unknown",
              "forwarded": "the request came through a proxy",
              "headers": "the request's headers are malformed",
              "host": "the request's Host is not a loopback name",
              "origin": "the request came from a web page",
              "remote-only": "the server treats every client as remote (--image-remote-only)"}
_SCHEME = re.compile(r"[a-z][a-z0-9+.-]+://")     # two letters at least: `c://x` is a drive
_ENTRY = re.compile(r"(\[[0-9a-f:.]+\]|[a-z0-9._-]+)(?::([0-9]{1,5}))?")
_UNSAFE_URL = re.compile(r"[^\x21-\x7e]|\\")
_WS = {ord(c): None for c in " \t\r\n\f\v"}
# not `ipaddress`'s is_global and friends: their answers differ between Python releases
_V4_NOT_PUBLIC = tuple(ipaddress.ip_network(n) for n in (
    "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16", "172.16.0.0/12", "192.0.0.0/24",
    "192.0.2.0/24", "192.31.196.0/24", "192.52.193.0/24", "192.88.99.0/24", "192.168.0.0/16", "192.175.48.0/24",
    "198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24", "224.0.0.0/4", "240.0.0.0/4"))
_V6_UNICAST = ipaddress.ip_network("2000::/3")          # everything else is special: mapped, link- and site-local, NAT64...
_V6_NOT_PUBLIC = tuple(ipaddress.ip_network(n) for n in (
    "2001::/23", "2001:db8::/32", "2002::/16", "2620:4f:8000::/48", "3fff::/20"))   # IETF, docs, 6to4, AS112, docs
_NT_RESERVED = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
                *(f"{d}{n}" for d in ("COM", "LPT") for n in (*"0123456789", "\u00b9", "\u00b2", "\u00b3"))}


@dataclass(frozen=True)
class ImageSources:
    """What one request may name. `url_hosts`: (host, port or None for the scheme's default). `dirs`: real paths.
    `remote_only`: no request counts as local. `local`, `why` and `peer` come from the request's connection."""
    url_hosts: tuple[tuple[str, int | None], ...] = ()
    dirs: tuple[str, ...] = ()
    remote_only: bool = False
    local: bool = False
    why: str = "no-peer"
    peer: str = ""


class ImageRefused(ValueError):
    """A refused or failed source: the text is for the client, `log_line` for the operator."""

    def __init__(self, message: str, log_line: str):
        super().__init__(message)
        self.log_line = log_line


def shown(text) -> str:
    """Client-named text as one safe, short line: no line ends or terminal escapes get through."""
    return ascii(str(text)[:200])


def parse_url_host(text: str) -> tuple[str, int | None]:
    """A --image-url-host entry, HOST or HOST:PORT ([ADDR] for IPv6), as (host, port). No wildcards. ValueError."""
    m = _ENTRY.fullmatch(text.strip().lower())
    port = int(m.group(2)) if m and m.group(2) else None
    host = m.group(1) if m else ""
    if not host or host[0] == "." or host[-1] == "." or ".." in host or (port is not None and not 0 < port < 65536):
        raise ValueError(f"expected HOST or HOST:PORT, not {text!r}")
    if host.startswith("["):
        try:
            host = str(ipaddress.IPv6Address(host[1:-1]))
        except ValueError:
            raise ValueError(f"expected HOST or HOST:PORT, not {text!r}") from None
    return _canonical_host(host), port


def image_sources_of(hosts=(), dirs=(), remote_only: bool = False) -> ImageSources:
    """The operator's settings: --image-url-host, --image-dir, --image-remote-only. ValueError on a bad entry."""
    out = []
    for d in dirs:
        real = os.path.realpath(os.path.expanduser(d))
        if not os.path.isdir(real):
            raise ValueError(f"--image-dir {d}: not a directory")
        out.append(real)
    return ImageSources(url_hosts=tuple(dict.fromkeys(parse_url_host(h) for h in hosts)), dirs=tuple(dict.fromkeys(out)),
                        remote_only=remote_only)


def _canonical_host(host: str) -> str:
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        return host


def is_public_address(text: str) -> bool:
    """An address on the public internet, by this module's own tables, so the answer is the same on every Python."""
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return False
    if ip.version == 4:
        return not any(ip in net for net in _V4_NOT_PUBLIC)
    return not ip.scope_id and ip in _V6_UNICAST and not any(ip in net for net in _V6_NOT_PUBLIC)


def _refused(allowed: ImageSources, what: str, detail: str = "") -> ImageRefused:
    why = WHY_REMOTE.get(allowed.why, WHY_REMOTE["no-peer"])
    return ImageRefused(f"{POLICY}: {why}, {what}. {DATA_WORKS}",
                        f"{POLICY}: peer={shown(allowed.peer)} why={allowed.why} {detail}".rstrip())


def _failed(allowed: ImageSources, message: str, detail: str = "") -> ImageRefused:
    return ImageRefused(message, f"{message}: peer={shown(allowed.peer)} {detail}".rstrip())


def _local_failed(allowed: ImageSources, message: str, source: str, reason: str = "") -> ImageRefused:
    """A local client's specific error: it names the source and the reason. The log, which other clients may read, does
    not."""
    said = f"{message} ({reason})" if reason else message
    return ImageRefused(f"{said}: {shown(source)}", f"{message}: peer={shown(allowed.peer)}")


def _url_host(source: str) -> str:
    """The host of a URL, for the log: never its credentials, path or query."""
    try:
        return shown(urllib.parse.urlsplit(source).hostname or "")
    except ValueError:
        return "''"


# ------------------------------------------------------------------------------------------------ data: URLs
def _read_data(source: str) -> bytes:
    head, comma, payload = source.partition(",")
    if not comma:
        raise ValueError("a data: URL needs a comma before its payload")
    if ";base64" not in head.lower():
        raise ValueError("a data: URL must be base64")
    payload = payload.translate(_WS)                    # line-wrapped base64 is common
    if not payload:
        raise ValueError("a data: URL's payload is empty")
    if len(payload) > (IMAGE_MAX_BYTES + 2) // 3 * 4:   # checked before anything is decoded
        raise ValueError(f"the image is larger than {IMAGE_MAX_BYTES >> 20} MiB")
    try:
        return base64.b64decode(payload, validate=True)
    except ValueError:
        raise ValueError("a data: URL's payload is not valid base64") from None


# ------------------------------------------------------------------------------------------------ files
def file_url_path(source: str, windows: bool = os.name == "nt", relative: bool = False, decode: bool = True) -> str:
    """The path of a file: URL. `file://C:/x`, which harnesses build by joining strings, reads as the drive path.
    Anything else after `file://` is a host, a ValueError, unless `relative` reads it as a relative path."""
    rest = source[len("file:"):]
    if rest.startswith("//"):
        rest = rest[2:]
        drive = re.match(r"[A-Za-z]:[\\/]", rest)
        if rest[:10].lower() == "localhost/":
            rest = rest[9:]
        elif not rest.startswith("/") and not (windows and drive) and not relative:
            raise ValueError("a file: URL may not name another host")
    path = urllib.parse.unquote(rest) if decode else rest
    if windows and re.match(r"/[A-Za-z]:[\\/]", path):
        path = path[1:]
    return path


def windows_path_refused(path: str) -> bool:
    """Forms a remote request may not name on Windows: anything but a plain `C:\\...` path. UNC and device paths,
    drive-relative paths, alternate data streams, reserved device names and names Windows would silently rewrite."""
    if not re.match(r"[A-Za-z]:[\\/]", path) or ntpath.splitdrive(path)[0][1:] != ":":
        return True
    for part in re.split(r"[\\/]", path[3:]):
        if part in ("", ".", ".."):
            continue
        if re.search(r'[:*?"<>|\x00-\x1f]', part) or part != part.rstrip(". "):
            return True
        if part.split(".")[0].rstrip(" ").upper() in _NT_RESERVED:
            return True
    return False


def _read_handle(fd: int) -> bytes:
    """The bytes of an open regular file, at most the cap; the handle is closed. OverflowError past the cap, OSError
    for anything but a regular file."""
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise OSError(errno.EINVAL, "not a regular file")
        if info.st_size > IMAGE_MAX_BYTES:
            raise OverflowError
    except BaseException:
        os.close(fd)
        raise
    with os.fdopen(fd, "rb") as f:
        data = f.read(IMAGE_MAX_BYTES + 1)              # also for a file that grew, or that reports no size
    if len(data) > IMAGE_MAX_BYTES:
        raise OverflowError
    return data


def _open_flags() -> int:
    # a FIFO's open would block without O_NONBLOCK; regular files ignore it
    return (os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOINHERIT", 0))


def _read_local_file(source: str, allowed: ImageSources) -> bytes:
    path = source
    if source[:6].lower() == "file:/":
        # percent-decoded, as a URL is; a client that joined "file://" and a path means a `%` in a name literally
        path = file_url_path(source, relative=True)
        if "%" in source:
            literal = file_url_path(source, relative=True, decode=False)
            path = path if os.path.exists(path) or not os.path.exists(literal) else literal
    if not path:
        raise ValueError(USAGE)
    try:
        if not stat.S_ISREG(os.stat(path).st_mode):     # before an open, which can have side effects on a device
            raise _local_failed(allowed, "the image path is not a regular file", path)
        return _read_handle(os.open(path, _open_flags()))
    except FileNotFoundError:
        raise _local_failed(allowed, "the image file does not exist", path) from None
    except OverflowError:
        raise _local_failed(allowed, f"the image is larger than {IMAGE_MAX_BYTES >> 20} MiB", path) from None
    except OSError as e:
        raise _local_failed(allowed, "the image file could not be read", path, e.strerror or "error") from None


def _within(path: str, root: str) -> bool:
    path, root = os.path.normcase(path), os.path.normcase(root)
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _open_under(root: str, real: str) -> int:
    """Open `real`, a path with no symlinks below `root`, so that a component swapped for a symlink after the check
    is refused. Fd-relative where the platform has it. Elsewhere the opened file must be the one the path still
    names, with the path still resolving under `root`."""
    if os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY"):
        parts = [p for p in os.path.relpath(real, root).split(os.sep) if p != "."]
        # O_PATH: a directory that may be traversed but not listed is fine
        through = getattr(os, "O_PATH", os.O_RDONLY) | os.O_DIRECTORY | os.O_CLOEXEC
        walk = through | os.O_NOFOLLOW
        at = os.open(root, through)
        try:
            for part in parts[:-1]:
                below = os.open(part, walk, dir_fd=at)
                os.close(at)
                at = below
            return os.open(parts[-1], _open_flags() | os.O_NOFOLLOW, dir_fd=at)
        finally:
            os.close(at)
    fd = os.open(real, _open_flags())
    try:
        if os.path.samestat(os.fstat(fd), os.stat(real)) and _within(os.path.realpath(real), root):
            return fd
    except OSError:
        pass
    os.close(fd)
    raise PermissionError(13, "the path changed while it was opened")


def _read_allowed_file(source: str, allowed: ImageSources, windows: bool = os.name == "nt") -> bytes:
    """A file a remote request names: it must resolve under an --image-dir. Every refusal reads the same, whether the
    file exists or not, and nothing outside the directories is opened. The check is on the resolved path: the path as
    named may reach the directory through a symlink."""
    refused = _refused(allowed, "and the path is not a file under the directories allowed with --image-dir"
                       if allowed.dirs else "so a file path needs --image-dir on the server", "kind=path")
    if not allowed.dirs:
        raise refused
    try:
        path = file_url_path(source, windows) if source[:6].lower() == "file:/" else source
        if "\x00" in path or not os.path.isabs(path) or (windows and windows_path_refused(path)):
            raise refused
        real = os.path.realpath(path)
        root = next((r for r in allowed.dirs if _within(real, r)), None)
        if root is None or real == root:
            raise refused
        return _read_handle(_open_under(root, real))
    except OverflowError:
        raise _failed(allowed, f"the image is larger than {IMAGE_MAX_BYTES >> 20} MiB", "kind=path") from None
    except (OSError, ValueError) as e:
        if e is refused:
            raise
        raise refused from None


# ------------------------------------------------------------------------------------------------ URLs
class _Watch:
    """The connections of one read. `close` ends them, also from another thread, so a read blocked on one returns.
    A duplicate of each socket does it: wrapping a socket in TLS takes the original's descriptor away."""

    def __init__(self):
        self.lock, self.socks, self.closed = threading.Lock(), [], False

    def add(self, sock: socket.socket) -> None:
        dup = sock.dup()
        with self.lock:
            if not self.closed:
                self.socks.append(dup)
                return
        self._end(dup)

    def close(self) -> None:
        with self.lock:
            self.closed = True
            socks, self.socks = self.socks, []
        for sock in socks:
            self._end(sock)

    @staticmethod
    def _end(sock: socket.socket) -> None:
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)
        sock.close()


class _Watched:
    """A urllib HTTP handler whose connections join a _Watch from the moment they exist."""
    watch: _Watch

    def do_open(self, http_class, req, **kw):
        def make(host, **args):
            conn = http_class(host, **args)
            create = conn._create_connection

            def watched(*a, **k):
                sock = create(*a, **k)
                self.watch.add(sock)
                return sock
            conn._create_connection = watched
            return conn
        return super().do_open(make, req, **kw)


class _WatchedHTTP(_Watched, urllib.request.HTTPHandler):
    pass


class _WatchedHTTPS(_Watched, urllib.request.HTTPSHandler):
    pass


def _read_response(r) -> bytes:
    """The response's bytes up to the cap. At the deadline _bounded ends the connection, which ends this loop."""
    parts, total = [], 0
    while True:
        piece = r.read1(1 << 16)
        if not piece:
            if getattr(r, "length", None):              # read1 does not report a body cut short
                raise http.client.IncompleteRead(b"", r.length)
            return b"".join(parts)
        total += len(piece)
        if total > IMAGE_MAX_BYTES:
            raise OverflowError
        parts.append(piece)


def _read_local_url(source: str, allowed: ImageSources, watch: _Watch) -> bytes:
    req = urllib.request.Request(source, headers={"User-Agent": USER_AGENT})
    handlers = _WatchedHTTP(), _WatchedHTTPS()
    for handler in handlers:
        handler.watch = watch
    try:
        with urllib.request.build_opener(*handlers).open(req, timeout=IMAGE_READ_S) as r:
            return _read_response(r)
    except OverflowError:
        raise _local_failed(allowed, f"the image is larger than {IMAGE_MAX_BYTES >> 20} MiB", source) from None
    except urllib.error.HTTPError as e:
        e.close()
        raise _local_failed(allowed, f"the image URL answered HTTP {e.code}", source) from None
    except (OSError, http.client.HTTPException, ValueError) as e:
        reason = getattr(e, "reason", None) or e
        raise _local_failed(allowed, "the image URL could not be read", source, shown(reason)) from None


def _resolve(host: str, port: int) -> list[str]:
    """The name's addresses, each once, in the resolver's order."""
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(info[4][0] for info in infos))


def _connect(ip: str, port: int, timeout: float) -> socket.socket:
    """A TCP connection to exactly this address; no name lookup happens here."""
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        sock.connect((ip, port))
        return sock
    except BaseException:
        sock.close()
        raise


def _tls_context() -> ssl.SSLContext:
    return ssl.create_default_context()


class _PinnedConnection(http.client.HTTPConnection):
    """Connects to the addresses that were checked, all of them within the timeout. The Host header and the TLS name
    and certificate check stay on the original host."""

    def __init__(self, host: str, port: int, addresses: list[str], tls: ssl.SSLContext | None, timeout: float,
                 watch: _Watch):
        super().__init__(host, port=port, timeout=timeout)
        self.addresses, self.tls, self.watch = addresses, tls, watch

    def connect(self):
        last = OSError("no address")
        deadline = time.monotonic() + self.timeout
        for ip in self.addresses:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("the connect timed out")
            try:
                sock = _connect(ip, self.port, left)
                break
            except OSError as e:
                last = e
        else:
            raise last
        try:
            self.watch.add(sock)
            self.sock = self.tls.wrap_socket(sock, server_hostname=self.host) if self.tls else sock
        except BaseException:
            sock.close()
            raise


class _PinnedHandler(urllib.request.AbstractHTTPHandler):
    def __init__(self, host: str, port: int, addresses: list[str], watch: _Watch):
        super().__init__()
        self.target = (host, port, addresses, watch)

    def _pinned(self, req, tls):
        host, port, addresses, watch = self.target
        return self.do_open(lambda _, timeout: _PinnedConnection(host, port, addresses, tls, timeout, watch), req)

    def http_open(self, req):
        return self._pinned(req, None)

    def https_open(self, req):
        return self._pinned(req, _tls_context())

    http_request = urllib.request.AbstractHTTPHandler.do_request_
    https_request = urllib.request.AbstractHTTPHandler.do_request_


def _read_allowed_url(source: str, allowed: ImageSources, watch: _Watch) -> bytes:
    """A URL a remote request names: exactly an allowed host and port. A name must resolve to public addresses only,
    and the connection goes to those; an IP literal the operator named is taken as written. No proxy, no redirect."""
    refused = _refused(allowed, "and the URL's host and port are not among those allowed with --image-url-host"
                       if allowed.url_hosts else "so an http(s) URL needs --image-url-host HOST on the server",
                       f"kind=url host={_url_host(source)}")
    if _UNSAFE_URL.search(source):
        raise refused
    try:
        parts = urllib.parse.urlsplit(source)
        host, port = _canonical_host((parts.hostname or "").lower()), parts.port
    except ValueError:
        raise refused from None
    if parts.username is not None or parts.password is not None or not host:
        raise refused
    default = 443 if parts.scheme == "https" else 80
    if not any(h == host and (p or default) == (port or default) for h, p in allowed.url_hosts):
        raise refused
    port = port or default
    fetch_failed = _failed(allowed, "the image URL could not be fetched", f"host={shown(host)}")
    if is_ip(host):
        addresses = [host]
    else:
        try:
            addresses = _resolve(host, port)
        except (OSError, UnicodeError) as e:
            fetch_failed.log_line += f" reason={shown(e)}"
            raise fetch_failed from None
        if not addresses or not all(is_public_address(a) for a in addresses):
            raise ImageRefused(f"{POLICY}: the allowed host {shown(host)} resolves to a non-public address; name the "
                               f"address itself in --image-url-host to allow it. {DATA_WORKS}",
                               f"{POLICY}: peer={shown(allowed.peer)} host={shown(host)} non-public")
    opener = urllib.request.OpenerDirector()            # no proxy, redirect, ftp or file handler
    for handler in (_PinnedHandler(host, port, addresses, watch), urllib.request.HTTPDefaultErrorHandler(),
                    urllib.request.HTTPErrorProcessor()):
        opener.add_handler(handler)
    req = urllib.request.Request(source, headers={"User-Agent": USER_AGENT})
    try:
        with opener.open(req, timeout=IMAGE_READ_S) as r:
            return _read_response(r)
    except OverflowError:
        raise _failed(allowed, f"the image is larger than {IMAGE_MAX_BYTES >> 20} MiB", f"host={shown(host)}") from None
    except urllib.error.HTTPError as e:                 # with no redirect handler, every 3xx lands here too
        e.close()
        if 300 <= e.code < 400:
            raise ImageRefused(f"{POLICY}: the image URL redirected, and redirects are not followed for this client; "
                               f"name the final URL's host with --image-url-host on the server. {DATA_WORKS}",
                               f"{POLICY}: peer={shown(allowed.peer)} redirect={e.code} "
                               f"host={shown(host)}") from None
        raise _failed(allowed, f"the image URL answered HTTP {e.code}", f"host={shown(host)}") from None
    except (OSError, http.client.HTTPException, ValueError) as e:
        fetch_failed.log_line += f" reason={shown(e)}"
        raise fetch_failed from None


# ------------------------------------------------------------------------------------------------ the one entry
def _bounded(read, allowed: ImageSources, source: str) -> bytes:
    """`read(watch)` with a deadline, whatever blocks it. A slow host's connection is ended at the deadline, so its
    thread ends too. A name lookup or a dead network mount cannot be ended: that thread stays until the system gives
    up, as a daemon, and the request goes on."""
    done: queue.Queue = queue.Queue(1)
    watch = _Watch()

    def run():
        try:
            done.put((read(watch), None))
        except BaseException as e:  # noqa: BLE001 - raised again on the request's thread
            done.put((None, e))
        finally:
            watch.close()

    threading.Thread(target=run, daemon=True, name="image-read").start()
    try:
        data, error = done.get(timeout=IMAGE_READ_S)
    except queue.Empty:
        watch.close()
        message = f"the image source did not answer within {IMAGE_READ_S:.0f} s"
        raise ImageRefused(message if not allowed.local else f"{message}: {shown(source)}",
                           f"{message}: peer={shown(allowed.peer)}") from None
    if error is not None:
        raise error
    return data


def load_image(source, allowed: ImageSources | None = None) -> bytes:
    """One request image's bytes, under the policy; no policy is the remote one with nothing allowed."""
    allowed = allowed or ImageSources()
    if not isinstance(source, str) or not source:
        raise ValueError(USAGE)
    low = source[:32].lower()
    if low.startswith("data:"):
        return _read_data(source)
    if low.startswith(("http://", "https://")):
        fetch = _read_local_url if allowed.local else _read_allowed_url
        return _bounded(lambda watch: fetch(source, allowed, watch), allowed, source)
    if _SCHEME.match(low) and not low.startswith("file://"):
        raise ValueError(USAGE)
    read = _read_local_file if allowed.local else _read_allowed_file
    return _bounded(lambda watch: read(source, allowed), allowed, source)
