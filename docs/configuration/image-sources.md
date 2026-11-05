# Image sources

A request names an image as a `data:` URL, an `http(s)` URL, a `file:` URL or a path. What the server reads for it
depends on where the request comes from. This holds for an image anywhere in a request, a tool result included.

| Source | From this machine | From anywhere else |
| --- | --- | --- |
| `data:` URL | yes | yes |
| `http(s)` URL | yes, any host, redirects followed | only a host named with `--image-url-host`, redirects refused |
| path or `file:` URL | yes, any file the server's user can read | only under a directory named with `--image-dir` |

Every source, from every client: one image is at most 32 MiB of bytes and 64 megapixels, a read takes at most 60 s, and
only regular files are read (no FIFOs or devices). A request's images together are at most 1 GiB as files, after any
conversion to PNG. Beyond 2 GiB of read images waiting for their turn to encode, a request gets a 409 and may try again.

## What counts as this machine

The socket's peer address decides: it must be a loopback address. The machine's own LAN address is another machine.
Headers can only take this away. A loopback request still counts as remote when it has:

- a forwarding header (`Forwarded`, `X-Forwarded-*`, `X-Real-IP`, `Via`, `CF-*`, `Tailscale-*` and similar),
- a `Host` that is not `localhost`, a loopback address or `0.0.0.0`, as when a client reaches this machine by name,
- any `Origin`, or a `Sec-Fetch-Site` other than `none`: every web page, also the server's own, one on another local
  port, one let in with `--cors-origin` and a desktop app's page (`app://`, `file://`). A desktop app's main process
  counts as local; its pages send `data:` URLs,
- headers the server cannot parse.

`--image-remote-only` makes every request remote, whatever its peer.

A request refused for this reason gets a 400 that says it is a policy refusal, why, and which flag allows it.

## `--image-url-host HOST[:PORT]`

Exactly this host: no subdomains, no wildcards, and `http`/`https` on their default ports unless a port is given.

- A **name** must resolve only to public addresses. Loopback, private, link-local (cloud metadata too), CGNAT,
  multicast and reserved addresses are refused, and of IPv6 only global unicast passes, not IPv4-mapped, NAT64 or 6to4
  forms. The connection
  goes to the address that was checked, with `Host` and the TLS name and certificate check on the original name.
- An **IP address** is taken as written, private ones too: `--image-url-host 192.168.1.20:8000`. This is how to allow a
  machine on the network. A name such as `localhost` or `nas.lan` that resolves to a private address does not pass.

Environment proxy settings are not used for these fetches. They are for requests from this machine.

## `--image-dir DIR`

The path must be absolute and resolve under `DIR`, after symlinks and `..`, however it reaches there, and the resolved
file is the one opened. Nothing outside `DIR` is opened, though resolving a path reads the links along it.
On Linux it is opened component by component, so a swap to a symlink during the open is refused. On Windows the opened
file is checked against the path afterwards, and UNC and device paths, `C:relative` paths, alternate data streams and
reserved names (`CON`, `NUL`, ...) are refused. A hard link or bind mount inside `DIR` is inside `DIR`.

Outside `DIR`, and for a name inside it that is missing, unreadable or not a regular file, the refusal reads the same.
A readable file inside `DIR` can answer differently: too large, or not an image.

## Limits

- **Forwarders on this machine.** Anything on this machine that passes on other people's requests makes them look
  local unless it adds a forwarding header, and `Host` is no help: `proxy_set_header Host $host` passes on whatever the
  client sent, such as `localhost`. This covers:
  - nginx by default (it adds no forwarding header) and HAProxy without `option forwardfor`,
  - plain TCP forwarders: `ssh -L`, `socat`, `kubectl port-forward`, rootless `pasta` from the host's loopback,
  - app gateways that pass a user's `image_url` along: LiteLLM, Open WebUI's API.

  Whoever can reach one of these can read the server user's files and its network. Set `--image-remote-only` on such a
  server. Caddy, Apache, Traefik and `tailscale serve` do add forwarding headers.
- **Shared machines.** A request from this machine reads files as the server's user, not as the client's. Another
  local user reaches the same power, so set `--api-key` there; a loopback request needs the key like any other. Anyone
  holding the key, on this machine, has that power. Nothing tells local users apart beyond the key.
- **Containers.** A published port's peer is the bridge's gateway, so clients of a container are remote, also those on
  the host; use `data:` URLs or the two flags. Paths name files in the container. With `--network host` the peer is real.
