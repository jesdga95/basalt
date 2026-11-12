# Image sources

A request names an image as a `data:` URL, `http(s)` URL, `file:` URL or path. What the server reads depends on whether the request is
from this machine. A [document](documents.md) named by URL or path follows the same rules.

| Source | From this machine | From anywhere else |
|---|---|---|
| `data:` URL | yes | yes |
| `http(s)` URL | yes, any host | only a host named with `--image-url-host` |
| path or `file:` URL | yes, any file the server's user can read | only under a directory named with `--image-dir` |

Limits for every source: 32 MiB and 64 megapixels per image, 60 s per read, regular files only, 1 GiB of files per request.
Videos follow the same rules, with 1 GiB per video and per request.

**This machine** means a loopback peer with no forwarding header, no non-local `Host`, and no `Origin` or `Sec-Fetch-Site` other than
`none` (any web page counts as remote, including the server's own page). `--image-remote-only` makes every request remote.

- `--image-url-host HOST[:PORT]`: exactly that host. A name must resolve only to public addresses. An IP address is taken as written, so
  `192.168.1.20:8000` works for a machine on the network.
- `--image-dir DIR`: the resolved path must stay under `DIR`, after symlinks and `..`.

**Forwarders:** anything on this machine that passes on other people's requests (nginx by default, `ssh -L`, `socat`, LiteLLM and similar)
makes them look local. Set `--image-remote-only` there. Caddy, Apache, Traefik and `tailscale serve` add forwarding headers.

**Shared machines:** local requests read files as the server's user. Set `--api-key` there.

**Containers:** a published port's clients look remote. Use `data:` URLs, or the two flags.
