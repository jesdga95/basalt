"""Host and origin checks, the allowed names, and the exposure test for a server without an API key."""
from __future__ import annotations

import hmac
import ipaddress
import socket


LOOPBACK_NAMES = ("localhost", "127.0.0.1", "::1")


# proxies announce themselves with these; ordinary clients send none of them
FORWARDING_HEADERS = frozenset((
    "forwarded", "via", "x-real-ip", "x-client-ip", "client-ip", "true-client-ip", "x-cluster-client-ip",
    "cf-connecting-ip", "fastly-client-ip", "x-originating-ip", "x-remote-ip", "x-remote-addr", "x-host",
    "x-proxyuser-ip", "wl-proxy-client-ip", "proxy-connection"))
FORWARDING_PREFIXES = ("x-forwarded-", "x-original-", "x-envoy-", "x-amzn-", "x-azure-", "x-appengine-", "cf-",
                       "tailscale-")


def exposed_without_key(host: str, api_key: str) -> bool:
    """True when the server listens beyond this PC without an API key."""
    return host not in LOOPBACK_NAMES and not api_key


def api_key_of(value) -> str:
    """The key as a client can send it. An HTTP header loses the spaces and line ends around its value, so a key kept
    with them would match no request. A value that is only such characters raises ValueError: it never means "no key"."""
    key = "" if value is None else str(value)
    if key and not key.strip():
        raise ValueError("an API key was given but it is empty")
    return key.strip()


def key_matches(given: str, key: str) -> bool:
    """`given` is a header's value as http.server read it, each byte one character. A key with characters outside
    ASCII arrives as UTF-8 from most clients and as Latin-1 from some; the right key passes in both forms.
    Constant-time."""
    raw = given.encode("latin-1", "replace")
    ok = hmac.compare_digest(raw, key.encode())
    try:
        ok |= hmac.compare_digest(raw, key.encode("latin-1"))
    except UnicodeEncodeError:
        pass
    return ok


def lan_addresses() -> list[str]:
    """This PC's non-loopback, non-link-local IPv4 addresses."""
    import socket
    first, ips = None, set()
    try:                                                # default-route address; UDP connect sends nothing
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            first = s.getsockname()[0]
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    ok = lambda ip: ip and not ip.startswith(("127.", "169.254.", "0."))
    return ([first] if ok(first) else []) + sorted(ip for ip in ips if ok(ip) and ip != first)


def host_name(value) -> str:
    """The lowercase name of a Host header or origin host[:port] ("[::1]:8095" -> "::1"); "" when malformed."""
    v = (value or "").strip().lower()
    if v.startswith("["):
        name, sep, rest = v[1:].partition("]")
        return name if sep and (not rest or (rest[:1] == ":" and rest[1:].isdigit())) else ""
    if v.count(":") == 1:
        v, port = v.split(":")
        if not port.isdigit():
            return ""
    elif ":" in v:                                       # a bare IPv6 address
        return v
    v = v.rstrip(".")
    return v if v and all(c.isalnum() or c in "-._" for c in v) else ""


def is_ip(name: str) -> bool:
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return False


def _name_in(name: str, names) -> bool:
    """`name` is one of `names`, or below an entry that starts with a dot (".example.com")."""
    return name in names or any(n.startswith(".") and (name.endswith(n) or name == n[1:]) for n in names)


def allowed_hosts_of(value) -> list[str]:
    """The --allowed-host names: host names, "*" or ".example.com" (it and below). Scheme, port and path are
    dropped; a bad entry raises ValueError."""
    items = [] if value in (None, "") else [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        raise ValueError("allowed_hosts: expected a host name or a list of them")
    out = []
    for raw in items:
        x = raw.strip().lower()
        if x == "*":
            out.append(x)
            continue
        x = x.split("://", 1)[-1].split("/", 1)[0]
        dot = x.startswith(".")
        name = host_name(x[1:] if dot else x)
        if not name:
            raise ValueError(f"allowed_hosts: {raw!r} is not a host name")
        out.append("." + name if dot else name)
    return list(dict.fromkeys(out))


def host_names_for(bind_host: str, allowed_hosts=(), trusted_origins=()) -> set[str]:
    """Every non-IP name this server answers to: localhost, the bind address, allowed hosts, trusted origins' hosts
    and, when listening beyond this PC, its name and LAN addresses (the Origin check trusts no IP)."""
    names = set(LOOPBACK_NAMES)
    bind = host_name(bind_host)
    if bind and bind not in ("0.0.0.0", "::"):
        names.add(bind)
    if bind not in LOOPBACK_NAMES:
        try:
            pc = socket.gethostname().lower()
            names.update((pc, pc + ".local"))
        except OSError:
            pass
        names.update(("host.docker.internal", *lan_addresses()))
    names.update(x for x in allowed_hosts if x != "*")
    names.update(filter(None, (host_name(o.split("://", 1)[-1]) for o in trusted_origins)))
    return names


def host_allowed(host, names, any_host=False) -> bool:
    """DNS-rebinding check: only known names pass, since a rebinding page carries its own site's name in Host.
    IPs, "*.localhost" and a missing Host (browsers always send one) pass too."""
    if any_host or not (host or "").strip():
        return True
    name = host_name(host)
    return bool(name) and (is_ip(name) or name == "localhost" or name.endswith(".localhost")
                           or _name_in(name, names))


def origin_allowed(origin: str, host, names, origins=()) -> bool:
    """Whether an Origin may use the model without an API key: this server's own page, a known name on any port,
    or a listed origin. Unlike the Host check no IP passes on trust, and "null" fails (any site can send it);
    non-http(s) schemes pass since no web site can send them."""
    origin = (origin or "").strip().rstrip("/")
    if origin in origins or "*" in origins:             # "*": every page
        return True
    scheme, sep, rest = origin.lower().partition("://")
    if not sep or not scheme:
        return False
    if scheme not in ("http", "https"):
        return True
    if host and rest == host.strip().lower():
        return True
    name = host_name(rest)
    return bool(name) and (name in LOOPBACK_NAMES or name.endswith(".localhost") or _name_in(name, names))


def peer_of(client_address) -> str:
    """The socket peer's address as text; "" when there is none."""
    try:
        return str(client_address[0])
    except (TypeError, IndexError, KeyError):
        return ""


def loopback_peer(address) -> bool:
    """127.0.0.0/8, ::1 or an IPv4-mapped ::ffff:127.x; anything else, this machine's LAN address too, is not."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return (getattr(ip, "ipv4_mapped", None) or ip).is_loopback


def _loopback_host(host) -> bool:
    """A Host value only a client of this machine sends. Any other name fails, which also covers a rebinding page."""
    name = host_name(host)
    return name == "localhost" or loopback_peer(name) or name in ("0.0.0.0", "::")


def remote_reason(client_address, headers) -> str:
    """"" when a request is provably from this machine, else why not. Only the socket peer can make a request local;
    headers can only make it remote: a proxy's forwarding headers, a Host that is not a loopback name, any Origin or a
    Fetch Metadata value other than `none`, which a desktop app's main process sends. No web page counts as local, the
    server's own neither. Unparsable headers count too: the parser drops whatever follows a malformed line, which can
    hide a proxy's own."""
    if not peer_of(client_address):
        return "no-peer"
    if not loopback_peer(peer_of(client_address)):
        return "peer"
    if headers is None or headers.defects or headers.get_payload():
        return "headers"
    names = [k.lower().replace("_", "-") for k in headers.keys()]
    if any(n in FORWARDING_HEADERS or n.startswith(FORWARDING_PREFIXES) for n in names):
        return "forwarded"
    hosts = headers.get_all("Host") or []
    if len(hosts) != 1 or not _loopback_host(hosts[0]):
        return "host"
    if headers.get_all("Origin") or any(v.strip().lower() != "none" for v in headers.get_all("Sec-Fetch-Site") or []):
        return "origin"
    return ""


def origins_of(value, key: str, wildcard: bool) -> list[str]:
    """An origin or list of origins (scheme://host[:port], no path); "*" only when `wildcard`; a bad entry raises."""
    if value is None or value == "" or value == []:
        return []
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        raise SystemExit(f"[server] error: {key}: expected an origin or a list of origins")
    out = []
    for x in items:
        x = x.strip().rstrip("/")
        if x == "*" and wildcard:
            out.append(x)
            continue
        scheme, sep, rest = x.partition("://")
        if scheme not in ("http", "https") or not sep or not rest or "/" in rest or "*" in rest:
            raise SystemExit(f"[server] error: {key}: {x!r} is not an origin like https://chat.example.com"
                             + ("" if wildcard else " (no wildcards)"))
        out.append(x)
    return out
