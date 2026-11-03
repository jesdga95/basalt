"""The image-source policy through real handlers and sockets: any source from this machine, allowed ones from elsewhere."""
from __future__ import annotations

import base64
import contextlib
import http.client
import http.server
import io
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.config import build_parser, model_settings, parse_settings  # noqa: E402
from serve.engine import image_source  # noqa: E402
from serve.engine.image_source import (ImageRefused, ImageSources, file_url_path, image_sources_of,  # noqa: E402
                                       is_public_address, load_image, parse_url_host, windows_path_refused)
from serve.engine.mock import MockEngine  # noqa: E402
from serve.engine.vision import Vision  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.handler import Server, make_handler, serve  # noqa: E402
from serve.security import lan_addresses, loopback_peer, remote_reason  # noqa: E402
from serve.service import Service  # noqa: E402
from serve.util.text import ByteTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PNG = b"\x89PNG\r\n\x1a\n" + b"pixels"
DATA = "data:image/png;base64," + base64.b64encode(PNG).decode()
MODEL = "basalt"
LOCAL = ImageSources(local=True, why="")
POLICY = "refused by server policy"
posix_only = unittest.skipUnless(os.name == "posix", "POSIX paths and special files")


class RecordingVision(Vision):
    """The real read path, with the encoder replaced by a record of the bytes it was given."""

    def __init__(self):
        self.dir = Path(tempfile.mkdtemp(prefix="basalt-vision-test-"))
        self.got = []

    def encode(self, images):
        self.got.extend(images)
        out = self.dir / f"req-{len(self.got)}.sve"
        out.write_bytes(b"rows")
        return out, [2] * len(images)

    def alive(self):
        return True


class ImageServer(unittest.TestCase):
    """A real server over the mock engine. `peer` replaces the socket peer the handler sees (FORCED_* below)."""
    peer = None
    host = "127.0.0.1"
    sources = ImageSources()

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="image-sources-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.image = self.tmp / "a b.png"
        self.image.write_bytes(PNG)
        self.vision = RecordingVision()
        self.addCleanup(shutil.rmtree, self.vision.dir, ignore_errors=True)
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "</think>\n\nok", max_context=4096), tok,
                           ChatTemplate.from_file(ROOT / "serve/chat_template.jinja"), vision=self.vision)
        self.svc.image_sources = self.sources
        self.httpd = None

    def start(self, host=None, peer=None, **attrs):
        for k, v in attrs.items():
            setattr(self.svc, k, v)
        peer = peer if peer is not None else self.peer
        if peer is None:
            self.httpd = serve(self.svc, host=host or self.host, port=0)
        else:
            base = make_handler(self.svc)
            self.svc.host_names = {"localhost", "127.0.0.1", "::1"}

            class Faked(base):
                def setup(self):
                    super().setup()
                    self.client_address = peer[0]

            self.httpd = Server((host or self.host, 0), Faked)
            threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.port = self.httpd.server_address[1]

    def post(self, path, body, headers=None, host="127.0.0.1", source_address=None):
        c = http.client.HTTPConnection(host, self.port, timeout=30, source_address=source_address)
        h = {"Content-Type": "application/json"} | (headers or {})
        c.request("POST", path, json.dumps(body), h)
        r = c.getresponse()
        raw = r.read()
        c.close()
        return r.status, raw.decode()

    def chat(self, source, headers=None, **kw):
        return self.post("/v1/chat/completions", openai_body(source), headers, **kw)

    def assertReadIt(self, reply, data=PNG):
        self.assertEqual(reply[0], 200, reply[1])
        self.assertEqual(self.vision.got[-1], data)

    def assertRefused(self, reply, *flags):
        self.assertEqual(reply[0], 400, reply[1])
        self.assertIn(POLICY, reply[1])
        self.assertIn("data: URL always works", reply[1])
        for flag in flags:
            self.assertIn(flag, reply[1])
        self.assertEqual(self.vision.got, [])


def openai_body(source, **kw):
    return {"model": MODEL, "max_tokens": 2, **kw, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": source}}]}]}


def frontends(source):
    """Every API and part shape that names a source: (route, body)."""
    user = lambda *parts: {"role": "user", "content": [{"type": "text", "text": "look"}, *parts]}
    base = {"model": MODEL, "max_tokens": 2}
    return {
        "openai object": ("/v1/chat/completions", {**base, "messages": [user(
            {"type": "image_url", "image_url": {"url": source}})]}),
        "openai string": ("/v1/chat/completions", {**base, "messages": [user(
            {"type": "image_url", "image_url": source})]}),
        "openai stream": ("/v1/chat/completions", {**base, "stream": True, "messages": [user(
            {"type": "image_url", "image_url": {"url": source}})]}),
        "openai old turn": ("/v1/chat/completions", {**base, "messages": [
            user({"type": "image_url", "image_url": {"url": source}}),
            {"role": "assistant", "content": "seen"}, {"role": "user", "content": "and now?"}]}),
        "responses string": ("/v1/responses", {"model": MODEL, "max_output_tokens": 2, "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "look"},
                                         {"type": "input_image", "image_url": source}]}]}),
        "responses object": ("/v1/responses", {"model": MODEL, "max_output_tokens": 2, "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "look"},
                                         {"type": "input_image", "image_url": {"url": source}}]}]}),
        "responses stream": ("/v1/responses", {"model": MODEL, "max_output_tokens": 2, "stream": True, "input": [
            {"role": "user", "content": [{"type": "input_image", "image_url": source}]}]}),
        "anthropic url": ("/v1/messages", {**base, "messages": [user(
            {"type": "image", "source": {"type": "url", "url": source}})]}),
        "anthropic path": ("/v1/messages", {**base, "messages": [user(
            {"type": "image", "source": {"type": "path", "path": source}})]}),
        "anthropic stream": ("/v1/messages", {**base, "stream": True, "messages": [user(
            {"type": "image", "source": {"type": "url", "url": source}})]}),
        "anthropic old turn": ("/v1/messages", {**base, "messages": [
            user({"type": "image", "source": {"type": "url", "url": source}}),
            {"role": "assistant", "content": "seen"}, {"role": "user", "content": "and now?"}]}),
    }


class PeerClasses(unittest.TestCase):
    def test_table(self):
        for address in ("127.0.0.1", "127.0.0.2", "127.255.255.254", "::1", "0:0:0:0:0:0:0:1", "::ffff:127.0.0.1",
                        "::ffff:7f00:1", "::ffff:127.9.9.9"):
            self.assertTrue(loopback_peer(address), address)
        for address in ("10.0.0.1", "192.168.2.205", "172.17.0.1", "169.254.169.254", "::ffff:10.0.0.1", "0.0.0.0", "::",
                        "::127.0.0.1", "fe80::1", "2001:db8::1", "", "localhost", "127.0.0.1.evil", "127.1"):
            self.assertFalse(loopback_peer(address), address)

    def test_no_peer_is_remote(self):
        msg = http.client.parse_headers(io.BytesIO(b"Host: 127.0.0.1:1\r\n\r\n"))
        for address in (None, (), "", ("",), [None]):
            self.assertIn(remote_reason(address, msg), ("no-peer", "peer"), address)
        self.assertEqual(remote_reason(("127.0.0.1", 1), None), "headers")
        self.assertEqual(remote_reason(("127.0.0.1", 1), msg), "")

    def test_host_values(self):
        for host in ("127.0.0.1:8080", "localhost:8080", "LOCALHOST", "[::1]:8080", "127.0.0.2", "0.0.0.0:8080",
                     "[::ffff:127.0.0.1]:1"):
            msg = http.client.parse_headers(io.BytesIO(f"Host: {host}\r\n\r\n".encode()))
            self.assertEqual(remote_reason(("127.0.0.1", 1), msg), "", host)
        for host in ("box.tailnet.ts.net", "192.168.2.205:8080", "evil.example", "foo.localhost", "localhost.evil.example",
                     "127.0.0.1.nip.io", "", "a b"):
            msg = http.client.parse_headers(io.BytesIO(f"Host: {host}\r\n\r\n".encode()))
            self.assertEqual(remote_reason(("127.0.0.1", 1), msg), "host", host)
        for raw in (b"\r\n", b"Host: 127.0.0.1\r\nHost: 127.0.0.1\r\n\r\n"):
            self.assertEqual(remote_reason(("127.0.0.1", 1), http.client.parse_headers(io.BytesIO(raw))), "host")


class RealPeers(ImageServer):
    """The peer as the kernel reports it, over real sockets and each address family."""

    def path_reply(self, *a, **kw):
        return self.chat(str(self.image), *a, **kw)

    def test_v4_loopback(self):
        self.start()
        self.assertReadIt(self.path_reply())

    def test_another_address_of_127_8(self):
        self.start(host="0.0.0.0")
        if sys.platform == "darwin":
            self.skipTest("only 127.0.0.1 is configured here")
        self.assertReadIt(self.path_reply(source_address=("127.0.0.9", 0)))

    def _family(self, bind, connect, v6only):
        if not socket.has_ipv6:
            self.skipTest("no IPv6")

        class Six(Server):
            address_family = socket.AF_INET6

            def server_bind(self):
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, v6only)
                super().server_bind()

        try:
            self.httpd = Six((bind, 0), make_handler(self.svc))
        except OSError:
            self.skipTest("no IPv6 here")
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.port = self.httpd.server_address[1]
        return self.chat(str(self.image), host=connect)

    def test_v6_loopback(self):
        self.assertReadIt(self._family("::1", "::1", 1))

    def test_v4_mapped_loopback(self):
        # a dual-stack listener: the kernel reports ::ffff:127.0.0.1
        self.assertReadIt(self._family("::", "127.0.0.1", 0))

    def test_this_machines_lan_address_is_remote(self):
        addresses = lan_addresses()
        if not addresses:
            self.skipTest("no LAN address here")
        self.start(host="0.0.0.0")
        self.assertRefused(self.chat(str(self.image), host=addresses[0]), "--image-dir")
        self.assertReadIt(self.post("/v1/chat/completions", openai_body(DATA), host=addresses[0]))

    def test_lan_url_from_lan_address_is_remote(self):
        addresses = lan_addresses()
        if not addresses:
            self.skipTest("no LAN address here")
        self.start(host="0.0.0.0")
        self.assertRefused(self.chat("https://images.example/a.png", host=addresses[0]), "--image-url-host")


class Headers(ImageServer):
    """Headers only ever take locality away."""

    FORWARDING = ("Forwarded", "X-Forwarded-For", "X-Forwarded-Host", "X-Forwarded-Proto", "X-Forwarded-Port",
                  "X-Forwarded-Server", "X-Real-IP", "Via", "True-Client-IP", "CF-Connecting-IP", "CF-Ray",
                  "Fastly-Client-IP", "X-Client-IP", "X-Cluster-Client-IP", "Client-IP", "X-Original-Forwarded-For",
                  "X-Envoy-External-Address", "X-Amzn-Trace-Id", "Tailscale-User-Login", "x-forwarded-for",
                  "X_Forwarded_For", "Proxy-Connection")

    def test_each_forwarding_header_makes_a_request_remote(self):
        self.start()
        for name in self.FORWARDING:
            for value in ("203.0.113.9", "127.0.0.1", "localhost", ""):
                with self.subTest(name=name, value=value):
                    self.assertRefused(self.chat(str(self.image), {name: value}), "--image-dir", "proxy")

    def test_header_cannot_grant_local(self):
        self.start(peer=(("203.0.113.9", 1),))
        for name in ("X-Forwarded-For", "X-Real-IP", "Forwarded"):
            self.assertRefused(self.chat(str(self.image), {name: "127.0.0.1"}), "--image-dir")
        self.assertRefused(self.chat(str(self.image), {"Host": "127.0.0.1", "Origin": "http://127.0.0.1",
                                                       "Sec-Fetch-Site": "same-origin"}), "--image-dir")

    def test_a_malformed_line_cannot_hide_a_proxys_header(self):
        self.start()
        body = json.dumps(openai_body(str(self.image))).encode()
        for bad in (b"no colon here\r\n", b"X-Forwarded-For : 1.2.3.4\r\n", b" leading fold\r\n", b"\x00bad: x\r\n"):
            with self.subTest(bad=bad), socket.create_connection(("127.0.0.1", self.port), timeout=30) as s:
                s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n" + bad +
                          b"X-Forwarded-For: 203.0.113.9\r\nContent-Type: application/json\r\n" +
                          b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
                s.shutdown(socket.SHUT_WR)
                raw = b""
                while chunk := s.recv(65536):
                    raw += chunk
                self.assertEqual(self.vision.got, [], raw[:200])

    def test_a_non_loopback_host_is_remote(self):
        self.start(allowed_hosts=["*"])
        self.assertRefused(self.chat(str(self.image), {"Host": "box.tailnet.ts.net:443"}), "--image-dir", "Host")
        self.assertRefused(self.chat(str(self.image), {"Host": "192.168.2.205:8080"}), "--image-dir")

    def test_ordinary_clients_stay_local(self):
        self.start()
        sdk = {"User-Agent": "OpenAI/Python 2.1.0", "Accept": "application/json", "Authorization": "Bearer unused",
               "X-Stainless-Lang": "python", "X-Stainless-Package-Version": "2.1.0", "X-Stainless-OS": "Linux",
               "X-Stainless-Arch": "x64", "X-Stainless-Runtime": "CPython", "X-Stainless-Runtime-Version": "3.12",
               "X-Stainless-Retry-Count": "0", "Connection": "keep-alive", "Accept-Encoding": "gzip, deflate"}
        anthropic = {"User-Agent": "Anthropic/Python 0.5", "anthropic-version": "2023-06-01", "x-api-key": "k"}
        curl = {"User-Agent": "curl/8.5", "Accept": "*/*"}
        for headers in (sdk, anthropic, curl):
            self.assertReadIt(self.chat(str(self.image), headers))

    def test_the_real_openai_sdk(self):
        try:
            import openai
        except ImportError:
            self.skipTest("the openai package is not installed")
        self.start()
        client = openai.OpenAI(base_url=f"http://127.0.0.1:{self.port}/v1", api_key="unused")
        client.chat.completions.create(model=MODEL, max_tokens=2, messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": str(self.image)}}]}])
        self.assertEqual(self.vision.got, [PNG])


class BrowserPages(ImageServer):
    EVIL = {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "cors"}

    def test_another_sites_page(self):
        # with --cors-origin * the page is let in, and it still reads nothing of this machine's
        self.start(cors_origins=["*"])
        self.assertRefused(self.chat(str(self.image), self.EVIL), "--image-dir", "web page")
        self.assertRefused(self.chat("http://169.254.169.254/latest/meta-data/", self.EVIL), "--image-url-host")
        self.assertReadIt(self.post("/v1/chat/completions", openai_body(DATA), self.EVIL))

    def test_origin_forms(self):
        self.start(cors_origins=["*"])
        for origin in ("https://evil.example", "http://localhost:3000", "http://127.0.0.1:9999", "null",
                       "chrome-extension://abcdef", "app://.", f"https://127.0.0.1:{{port}}", f"http://127.0.0.2:{{port}}",
                       f"http://LOCALHOST.evil:{{port}}", ""):
            with self.subTest(origin=origin):
                self.assertRefused(self.chat(str(self.image), {"Origin": origin.format(port=self.port)}), "--image-dir")

    def test_fetch_metadata_alone(self):
        self.start()
        for value in ("cross-site", "same-site", "same-origin", "weird", ""):
            with self.subTest(value=value):
                self.assertRefused(self.chat(str(self.image), {"Sec-Fetch-Site": value}), "--image-dir", "web page")

    def test_a_desktop_apps_main_process_is_local(self):
        # Electron's net.fetch sends no Origin and this value
        self.start()
        self.assertReadIt(self.chat(str(self.image), {"Sec-Fetch-Site": "none", "Sec-Fetch-Mode": "no-cors",
                                                      "Sec-Fetch-Dest": "empty"}))

    def test_a_desktop_apps_pages_are_remote(self):
        self.start(cors_origins=["*"])
        for headers in ({"Origin": "app://bundle", "Sec-Fetch-Site": "none"},        # any origin, whatever else it says
                        {"Origin": "app://bundle", "Sec-Fetch-Site": "cross-site"},
                        {"Origin": "null", "Sec-Fetch-Site": "cross-site"},
                        {"Sec-Fetch-Site": "cross-site"},                            # a file:// page
                        {"Sec-Fetch-Site": "none", "Origin": ""}):
            with self.subTest(headers=headers):
                self.assertRefused(self.chat(str(self.image), headers), "--image-dir", "web page")

    def test_fetch_metadata_disagrees_with_origin(self):
        self.start()
        own = {"Origin": "https://evil.example", "Sec-Fetch-Site": "none"}
        self.start(cors_origins=["*"])
        self.assertRefused(self.chat(str(self.image), own), "--image-dir")

    def test_the_servers_own_page_is_remote_too(self):
        # it has no image client: an XSS in it must not become a file read
        self.start()
        own = {"Origin": f"http://127.0.0.1:{self.port}", "Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors",
               "Referer": f"http://127.0.0.1:{self.port}/"}
        self.assertRefused(self.chat(str(self.image), own), "--image-dir", "web page")
        self.assertReadIt(self.post("/v1/chat/completions", openai_body(DATA), own))

    def test_text_plain_post_without_a_preflight(self):
        # a form or no-cors fetch: no key, foreign origin is 403, a let-in origin is 415; nothing is read
        evil = {"Content-Type": "text/plain", **self.EVIL}
        self.start()
        self.assertEqual(self.chat(str(self.image), evil)[0], 403)
        self.assertEqual(self.vision.got, [])
        self.start(cors_origins=["*"])
        self.assertEqual(self.chat(str(self.image), evil)[0], 415)
        self.assertEqual(self.vision.got, [])

    def test_text_plain_with_a_key_is_remote(self):
        self.start(api_key="k")
        evil = {"Content-Type": "text/plain", "Authorization": "Bearer k", **self.EVIL}
        self.assertRefused(self.chat(str(self.image), evil), "--image-dir")
        self.assertEqual(self.chat(str(self.image), {"Content-Type": "text/plain", **self.EVIL})[0], 401)

    def test_a_rebinding_page_with_the_host_check_off(self):
        # the page's name resolves here: its Origin equals its Host, and only the Host's name gives it away
        self.start(allowed_hosts=["*"])
        page = {"Host": "evil.example:8080", "Origin": "http://evil.example:8080", "Sec-Fetch-Site": "same-origin"}
        self.assertRefused(self.chat(str(self.image), page), "--image-dir")

    def test_cors_preflight_gives_no_allowance(self):
        self.start()
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        c.request("OPTIONS", "/v1/chat/completions", headers={"Origin": "https://evil.example",
                                                              "Access-Control-Request-Method": "POST"})
        r = c.getresponse()
        r.read()
        self.assertIsNone(r.getheader("Access-Control-Allow-Origin"))
        c.close()


class RemoteOnly(ImageServer):
    """--image-remote-only: for a server behind a same-machine forwarder that adds no header, which looks like this."""

    def test_a_plain_loopback_request_is_remote(self):
        self.svc.image_sources = image_sources_of(remote_only=True)
        self.start()
        self.assertRefused(self.chat(str(self.image)), "--image-dir", "--image-remote-only")
        self.assertRefused(self.chat("http://127.0.0.1:1/a.png"), "--image-url-host")
        self.assertRefused(self.chat(self.image.as_uri()), "--image-dir")
        self.assertReadIt(self.post("/v1/chat/completions", openai_body(DATA)))

    def test_what_the_operator_allowed_still_works(self):
        with Web() as web:
            self.svc.image_sources = image_sources_of((f"127.0.0.1:{web.port}",), (str(self.tmp),), remote_only=True)
            self.start()
            self.assertReadIt(self.chat(str(self.image)))
            self.assertReadIt(self.chat(web.url("/a.png")))
            self.vision.got.clear()
            self.assertRefused(self.chat(web.url("/redirect")), "redirect")     # the remote rules, not the local ones

    def test_off_by_default(self):
        self.assertFalse(parse_settings(build_parser(), ["--mock-engine"])[0].image_remote_only)
        self.assertFalse(image_sources_of().remote_only)
        self.start()
        self.assertReadIt(self.chat(str(self.image)))


class SharedMachine(ImageServer):
    def test_a_loopback_request_still_needs_the_key(self):
        self.start(api_key="k")
        reply = self.chat(str(self.image))
        self.assertEqual(reply[0], 401)
        self.assertEqual(self.vision.got, [])
        self.assertReadIt(self.chat(str(self.image), {"Authorization": "Bearer k"}))

    def test_a_key_does_not_make_a_remote_request_local(self):
        self.start(peer=(("203.0.113.9", 1),), api_key="k")
        self.assertRefused(self.chat(str(self.image), {"Authorization": "Bearer k"}), "--image-dir")


class EveryFrontend(ImageServer):
    """Each API and part shape, from each peer class."""

    def test_local_path_and_file_url(self):
        self.start()
        for source in (str(self.image), self.image.as_uri()):
            for name, (route, body) in frontends(source).items():
                with self.subTest(name=name, source=source):
                    self.vision.got.clear()
                    self.assertEqual(self.post(route, body)[0], 200)
                    self.assertEqual(self.vision.got, [PNG])

    def _refused_everywhere(self, headers=None):
        for source, flag in ((str(self.image), "--image-dir"), (self.image.as_uri(), "--image-dir"),
                             ("http://169.254.169.254/latest/meta-data/", "--image-url-host"),
                             ("https://images.example/a.png", "--image-url-host")):
            for name, (route, body) in frontends(source).items():
                with self.subTest(name=name, source=source):
                    self.assertRefused(self.post(route, body, headers), flag)

    def test_refused_from_a_forwarded_loopback_request(self):
        self.start()
        self._refused_everywhere({"X-Forwarded-For": "203.0.113.9"})

    def test_refused_from_a_gateway_peer(self):
        # a published container port: the peer is the bridge's gateway
        self.start(peer=(("172.17.0.1", 40000),))
        self._refused_everywhere()

    def test_refused_with_a_missing_peer(self):
        for peer in (None, "", ("",), ()):
            with self.subTest(peer=peer):
                self.start(peer=(peer,))
                self._refused_everywhere()

    def test_data_urls_from_everywhere(self):
        for peer in (None, (("203.0.113.9", 1),), (None,)):
            self.start(peer=peer)
            for name, (route, body) in frontends(DATA).items():
                with self.subTest(name=name, peer=peer):
                    self.vision.got.clear()
                    self.assertEqual(self.post(route, body)[0], 200)
                    self.assertEqual(self.vision.got, [PNG])

    def test_anthropic_base64(self):
        self.start(peer=(("203.0.113.9", 1),))
        body = {"model": MODEL, "max_tokens": 2, "messages": [{"role": "user", "content": [{"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": base64.b64encode(PNG).decode()}}]}]}
        self.assertReadIt(self.post("/v1/messages", body))

    def test_one_bad_source_among_good_ones(self):
        self.start(peer=(("203.0.113.9", 1),))
        body = {"model": MODEL, "max_tokens": 2, "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": DATA}},
                                         {"type": "image_url", "image_url": {"url": str(self.image)}}]}]}
        self.assertRefused(self.post("/v1/chat/completions", body), "--image-dir")

    def test_count_tokens_reads_nothing(self):
        self.start(peer=(("203.0.113.9", 1),))
        with mock.patch.object(image_source, "_read_allowed_file") as read:
            body = frontends(str(self.image))["anthropic path"][1]
            self.assertEqual(self.post("/v1/messages/count_tokens", body)[0], 200)
        read.assert_not_called()

    def test_service_default_is_remote(self):
        # a caller that names no access gets the remote policy
        self.start()
        msgs = [{"role": "user", "content": [{"type": "image", "source": str(self.image)}]}]
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, POLICY):
            self.svc.prepare(msgs, None, {})
        self.assertEqual(self.vision.got, [])

    def test_vision_read_default_is_remote(self):
        with self.assertRaisesRegex(ValueError, POLICY):
            self.vision.read([str(self.image)])
        with self.assertRaisesRegex(ValueError, POLICY):
            load_image(str(self.image))

    def test_a_cached_image_is_read_again_under_the_policy(self):
        # the cache is keyed by the bytes: a known image gives a remote client nothing by naming its path
        self.start()
        self.assertReadIt(self.chat(str(self.image)))
        self.vision.got.clear()
        self.assertRefused(self.chat(str(self.image), {"Via": "1.1 proxy"}), "--image-dir")

    def test_requests_on_one_server_do_not_share_a_class(self):
        self.start()
        out = []

        def one(i):
            local = i % 2 == 0
            reply = self.chat(str(self.image), None if local else {"X-Real-IP": "203.0.113.9"})
            out.append((local, reply[0]))
        threads = [threading.Thread(target=one, args=(i,)) for i in range(24)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        self.assertEqual(sorted(out), sorted((i % 2 == 0, 200 if i % 2 == 0 else 400) for i in range(24)))

    def test_odd_source_values(self):
        self.start()
        bodies = [{"type": "image", "source": "/etc/passwd"}, {"type": "image", "source": 5}, {"type": "image"},
                  {"type": "image", "source": {"type": "url", "url": 5}},
                  {"type": "image", "source": {"type": "file", "file_id": "x"}},
                  {"type": "image_url", "image_url": {"url": ["/etc/passwd"]}}, {"type": "image_url", "image_url": 5}]
        for part in bodies:
            with self.subTest(part=part):
                reply = self.post("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": [part]}]})
                self.assertEqual(reply[0], 400, reply[1])
        self.assertEqual(self.vision.got, [])


class LocalClients(ImageServer):
    """What must keep working with no flags."""

    def test_app_data_and_temp_paths(self):
        self.start()
        for sub in (".local/share/harness/images", "AppData/Roaming/Harness/cache", "tmp/xyz"):
            f = self.tmp / sub / "shot.png"
            f.parent.mkdir(parents=True)
            f.write_bytes(PNG + sub.encode())
            self.assertReadIt(self.chat(str(f)), PNG + sub.encode())

    @posix_only
    def test_file_url_forms(self):
        self.start()
        for source in (self.image.as_uri(), f"file:{self.image.parent}/a%20b.png",
                       f"file://{self.image.parent}/a%20b.png", f"file://localhost{self.image.parent}/a%20b.png",
                       "FILE://" + str(self.image)):
            with self.subTest(source=source):
                self.assertReadIt(self.chat(source))

    def test_a_relative_file_url_reads_as_before_for_a_local_client(self):
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(self.tmp)
        (self.tmp / "sub").mkdir()
        (self.tmp / "sub" / "x.png").write_bytes(PNG)
        self.start()
        self.assertReadIt(self.chat("file://sub/x.png"))
        self.assertIn("does not exist", self.chat("file://sub/none.png")[1])
        self.vision.got.clear()
        self.assertRefused(self.chat("file://sub/x.png", {"Via": "1.1 proxy"}), "--image-dir")

    @posix_only
    def test_a_percent_in_a_file_url(self):
        # decoded as a URL when that names a file, else literal: a client that joined "file://" and a path means it so
        self.start()
        cases = {"100%.png": ("file://{d}/100%.png",), "a%20c.png": ("file://{d}/a%20c.png",),
                 "b d.png": ("file://{d}/b%20d.png",), "p%41q.png": ("file://{d}/p%41q.png",)}
        for name, (template,) in cases.items():
            f = self.tmp / name
            f.write_bytes(PNG + name.encode())
            with self.subTest(name=name):
                self.assertReadIt(self.chat(template.format(d=self.tmp)), PNG + name.encode())
        (self.tmp / "both%20.png").write_bytes(PNG + b"literal")
        (self.tmp / "both .png").write_bytes(PNG + b"decoded")
        self.assertReadIt(self.chat(f"file://{self.tmp}/both%20.png"), PNG + b"decoded")

    def test_a_percent_is_decoded_only_for_a_remote_client(self):
        (self.tmp / "a%20c.png").write_bytes(PNG)
        allowed = remote(dirs=(str(self.tmp),))
        with self.assertRaisesRegex(ImageRefused, POLICY):
            load_image(f"file://{self.tmp}/a%20c.png", allowed)
        self.assertEqual(load_image(f"file://{self.tmp}/a%2520c.png", allowed), PNG)

    def test_an_env_proxy_still_carries_a_local_fetch(self):
        with Web() as proxy, mock.patch.dict(os.environ, {"http_proxy": f"http://127.0.0.1:{proxy.port}", "no_proxy": ""}):
            self.assertEqual(load_image("http://images.example/a.png", LOCAL), PNG)
            self.assertEqual(proxy.requests[0][0], "http://images.example/a.png")

    def test_a_relative_path(self):
        self.start()
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(self.tmp)
        self.assertReadIt(self.chat("a b.png"))

    def test_http_url_on_localhost_and_with_a_redirect(self):
        with Web() as web:
            self.start()
            self.assertReadIt(self.chat(web.url("/a.png")))
            self.assertReadIt(self.chat(web.url("/redirect")))
            self.assertReadIt(self.chat(f"http://localhost:{web.port}/a.png"))

    def test_http_url_on_the_lan(self):
        addresses = lan_addresses()
        if not addresses:
            self.skipTest("no LAN address here")
        with Web("0.0.0.0") as web:
            self.start()
            self.assertReadIt(self.chat(f"http://{addresses[0]}:{web.port}/a.png"))

    def test_specific_errors(self):
        self.start()
        missing = self.tmp / "missing.png"
        reply = self.chat(str(missing))
        self.assertEqual(reply[0], 400)
        self.assertIn("does not exist", reply[1])
        self.assertIn("missing.png", reply[1])
        self.assertNotIn(POLICY, reply[1])
        self.assertIn("not a regular file", self.chat(str(self.tmp))[1])
        with Web() as web:
            self.assertIn("HTTP 404", self.chat(web.url("/nothing"))[1])
        self.assertIn("could not be read", self.chat("http://127.0.0.1:1/a.png")[1])
        self.assertIn("an image must be", self.chat("ftp://127.0.0.1/a.png")[1])
        self.assertIn("an image must be", self.chat("gopher://127.0.0.1/a.png")[1])
        self.assertIn("an image must be", self.chat("")[1])


class Web:
    """A real image host; `requests` keeps what it was sent."""

    def __init__(self, host="127.0.0.1", tls=None):
        self.requests = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                outer.requests.append((self.path, self.headers.get("Host"), self.headers.get("User-Agent")))
                path = self.path.split("?")[0]
                if path.startswith("/code/"):
                    self.send_response(int(path[6:]))
                    self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", "/a.png")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if path == "/nothing":
                    self.send_error(404)
                    return
                if path == "/drip":
                    self.send_response(200)
                    self.end_headers()
                    with contextlib.suppress(OSError):
                        for _ in range(600):
                            self.wfile.write(b"x")
                            self.wfile.flush()
                            time.sleep(0.05)
                    return
                if path in ("/big", "/big-unsized"):
                    self.send_response(200)
                    if path == "/big":
                        self.send_header("Content-Length", str(1 << 30))
                    self.end_headers()
                    with contextlib.suppress(OSError):
                        for _ in range(64):
                            self.wfile.write(b"x" * (1 << 20))
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(PNG)))
                self.end_headers()
                self.wfile.write(PNG)

            def log_message(self, *a):
                pass

        self.httpd = http.server.ThreadingHTTPServer((host, 0), Handler)
        self.httpd.daemon_threads = True
        self.sni = []
        if tls:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(*tls)
            ctx.sni_callback = lambda sock, name, c: self.sni.append(name)
            self.httpd.socket = ctx.wrap_socket(self.httpd.socket, server_side=True)
        self.port = self.httpd.server_address[1]

    def url(self, path, name="127.0.0.1", scheme="http"):
        return f"{scheme}://{name}:{self.port}{path}"

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


def remote(hosts=(), dirs=()):
    allowed = image_sources_of(hosts, dirs)
    return ImageSources(url_hosts=allowed.url_hosts, dirs=allowed.dirs, local=False, why="peer", peer="203.0.113.9")


REAL_GETADDRINFO = socket.getaddrinfo


def resolver(*answers):
    """A stand-in for getaddrinfo for names under .example: each call gives the next answer list, or raises it."""
    calls = []

    def fake(host, port, *a, **kw):
        if not str(host).lower().endswith(".example"):
            return REAL_GETADDRINFO(host, port, *a, **kw)
        calls.append(host)
        if answers and isinstance(answers[0], Exception):
            raise answers[0]
        ips = answers[min(len(calls) - 1, len(answers) - 1)]
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in ips]
    fake.calls = calls
    return fake


class AllowedHosts(unittest.TestCase):
    def test_public_addresses(self):
        for ip in ("8.8.8.8", "93.184.216.34", "2606:4700:4700::1111", "2606:4700::1"):
            self.assertTrue(is_public_address(ip), ip)
        for ip in ("127.0.0.1", "10.0.0.1", "172.16.0.1", "192.168.1.1", "169.254.169.254", "100.64.0.1", "100.127.255.255",
                   "224.0.0.1", "239.255.255.250", "0.0.0.0", "240.0.0.1", "255.255.255.255", "192.0.2.1", "198.18.0.1",
                   "::1", "::", "fe80::1", "fe80::1%eth0", "fc00::1", "fd00:ec2::254", "ff02::1", "2001:db8::1",
                   "::ffff:127.0.0.1", "::ffff:10.0.0.1", "::ffff:169.254.169.254", "::ffff:8.8.8.8", "::7f00:1", "::808:808",
                   "fec0::1", "fec0:ffff::1", "feff::1", "2002:808:808::", "3fff::1", "3fff:fff::1", "2001:1::1", "2001:3::1",
                   "2001:4:112::1", "2001:20::1", "2001:0:4136:e378:8000:63bf:3fff:fdd2", "2620:4f:8000::1", "100::1",
                   "5f00::1", "64:ff9b:1::1", "192.31.196.1", "192.52.193.1", "192.175.48.1", "192.88.99.1",
                   "198.51.100.1", "203.0.113.1", "2606:4700::1%eth0",
                   "64:ff9b::808:808", "64:ff9b::a9fe:a9fe", "64:ff9b::7f00:1", "2002:7f00:1::", "2002:a9fe:a9fe::", "not an ip", ""):
            self.assertFalse(is_public_address(ip), ip)

    def test_entries(self):
        self.assertEqual(parse_url_host("Images.Example"), ("images.example", None))
        self.assertEqual(parse_url_host("images.example:8443"), ("images.example", 8443))
        self.assertEqual(parse_url_host("10.0.0.5:81"), ("10.0.0.5", 81))
        self.assertEqual(parse_url_host("[2001:DB8::1]:81"), ("2001:db8::1", 81))
        for bad in ("", "*", "*.example.com", ".example.com", "https://a.example", "a.example/x", "a.example:http",
                    "a.example:0", "a.example:70000", "::1", "[::1", "[nope]", "a b", "user@a.example"):
            with self.assertRaises(ValueError, msg=bad):
                parse_url_host(bad)

    def refused_before_any_lookup(self, url, hosts):
        with mock.patch.object(image_source.socket, "getaddrinfo", side_effect=AssertionError("looked up")), \
                mock.patch.object(image_source, "_connect", side_effect=AssertionError("connected")):
            with self.assertRaisesRegex(ImageRefused, POLICY) as cm:
                load_image(url, remote(hosts))
        self.assertIn("--image-url-host", str(cm.exception))

    def test_nothing_allowed(self):
        for url in ("http://169.254.169.254/latest/meta-data/", "http://[fd00:ec2::254]/", "http://127.0.0.1:8080/",
                    "https://images.example/a.png"):
            self.refused_before_any_lookup(url, ())

    def test_exactly_that_host(self):
        for url in ("http://images.example.evil.test/", "http://evil.test/images.example", "http://sub.images.example/",
                    "http://images.example./", "http://images.example:8080/", "https://images.example:8443/",
                    "http://images.example@evil.test/", "http://evil.test#@images.example/",
                    "http://evil.test?@images.example/", "http://images.example:80@evil.test/",
                    "http://user:pw@images.example/", "http://evil.test\\@images.example/",
                    "http://images.example\\.evil.test/", "http://images.exam\tple/", "http://images.example\r\nX: y/",
                    "http://images.example/ x", "http://[::ffff:127.0.0.1]/", "http://2130706433/", "http://0x7f.1/",
                    "http://127.1/", "http://0177.0.0.1/", "http://images.example:99999/", "http://\u0456mages.example/"):
            with self.subTest(url=url):
                self.refused_before_any_lookup(url, ("images.example",))

    def test_a_bare_entry_means_the_default_ports(self):
        with Web() as web:
            self.refused_before_any_lookup(web.url("/a.png"), ("127.0.0.1",))

    def test_a_literal_entry_is_taken_as_written(self):
        with Web() as web:
            self.assertEqual(load_image(web.url("/a.png"), remote((f"127.0.0.1:{web.port}",))), PNG)
            self.refused_before_any_lookup(f"http://[::ffff:127.0.0.1]:{web.port}/a.png", (f"127.0.0.1:{web.port}",))

    def test_more_url_spellings(self):
        for url in ("http://@images.example/", "http://:@images.example/", "http:///images.example/", "http://",
                    "http:// images.example/", "http://images.example%2f.evil.test/", "http://images%2eexample/",
                    "http://[images.example]/", "http://[::1", "http://images.example:80:80/",
                    "http://images.example:-1/", "http://images.example:0x50/", "http://%69mages.example/"):
            with self.subTest(url=url):
                self.refused_before_any_lookup(url, ("images.example",))

    def test_other_schemes_stay_out(self):
        for source in ("ftp://images.example/a.png", "gopher://images.example/", "dict://images.example/",
                       "ldap://images.example/", "jar:file:///etc/passwd!/a", "HTTPS+x://images.example/"):
            for allowed in (remote(("images.example",)), LOCAL):
                with self.subTest(source=source), \
                        mock.patch.object(image_source.socket, "getaddrinfo", side_effect=AssertionError("looked up")):
                    with self.assertRaises(ValueError):
                        load_image(source, allowed)

    def test_case_of_the_host_does_not_matter(self):
        with Web() as web, mock.patch.object(image_source.socket, "getaddrinfo", resolver(["93.184.216.34"])), \
                mock.patch.object(image_source, "_connect", side_effect=lambda ip, port, t: socket.create_connection(
                    ("127.0.0.1", web.port), t)):
            self.assertEqual(load_image(web.url("/a.png", "IMAGES.example"),
                                        remote((f"images.example:{web.port}",))), PNG)

    def test_a_name_with_a_private_answer(self):
        for answer in (["127.0.0.1"], ["10.0.0.1"], ["169.254.169.254"], ["100.64.0.1"], ["224.0.0.1"], ["::1"],
                       ["::ffff:127.0.0.1"], ["fe80::1%eth0"], ["64:ff9b::a9fe:a9fe"], ["93.184.216.34", "10.0.0.1"], []):
            with self.subTest(answer=answer), mock.patch.object(image_source.socket, "getaddrinfo", resolver(answer)), \
                    mock.patch.object(image_source, "_connect", side_effect=AssertionError("connected")):
                with self.assertRaises(ImageRefused) as cm:
                    load_image("http://images.example/a.png", remote(("images.example",)))
                self.assertIn(POLICY, str(cm.exception))
                self.assertIn("non-public", str(cm.exception))
                for leak in answer:                    # neither the answer nor the log, which /api/logs serves
                    self.assertNotIn(leak, str(cm.exception))
                    self.assertNotIn(leak, cm.exception.log_line)

    def test_localhost_named_as_a_host_is_not_enough(self):
        with Web() as web, self.assertRaisesRegex(ImageRefused, "non-public"):
            load_image(web.url("/a.png", "localhost"), remote((f"localhost:{web.port}",)))


class Rebinding(unittest.TestCase):
    """A name answering public at the check and private at the fetch."""

    def test_the_connection_goes_to_the_first_answer(self):
        seen = []
        with Web() as web:
            def connect(ip, port, timeout):
                seen.append(ip)
                return socket.create_connection(("127.0.0.1", web.port), timeout)
            fake = resolver(["93.184.216.34"], ["127.0.0.1"], ["169.254.169.254"])
            with mock.patch.object(image_source.socket, "getaddrinfo", fake), \
                    mock.patch.object(image_source, "_connect", connect):
                data = load_image(f"http://images.example:{web.port}/a.png", remote((f"images.example:{web.port}",)))
        self.assertEqual(data, PNG)
        self.assertEqual(fake.calls, ["images.example"])         # resolved once
        self.assertEqual(seen, ["93.184.216.34"])                # the checked address
        # the original host, not the address, on the wire
        self.assertEqual(web.requests, [("/a.png", f"images.example:{web.port}", "local-server")])

    def test_real_sockets_reach_only_the_checked_address(self):
        if sys.platform == "darwin":
            self.skipTest("only 127.0.0.1 is configured here")
        with Web("127.0.0.3") as checked:
            fake = resolver(["127.0.0.3"], ["127.0.0.1"])
            with mock.patch.object(image_source.socket, "getaddrinfo", fake), \
                    mock.patch.object(image_source, "is_public_address", return_value=True):
                data = load_image(f"http://images.example:{checked.port}/a.png",
                                  remote((f"images.example:{checked.port}",)))
        self.assertEqual(data, PNG)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(checked.requests[0][1], f"images.example:{checked.port}")

    def test_the_connect_is_bounded_across_all_addresses(self):
        given = []

        def slow(ip, port, timeout):
            given.append(timeout)
            time.sleep(0.15)
            raise TimeoutError("no answer")
        addresses = [f"93.184.216.{n}" for n in range(1, 50)]
        with mock.patch.object(image_source.socket, "getaddrinfo", resolver(addresses)), \
                mock.patch.object(image_source, "_connect", slow), mock.patch.object(image_source, "IMAGE_READ_S", 0.4):
            t0 = time.monotonic()
            with self.assertRaises(ImageRefused):
                load_image("http://images.example/a.png", remote(("images.example",)))
            waited = time.monotonic() - t0
            time.sleep(0.5)
        self.assertLess(waited, 2)
        self.assertLessEqual(len(given), 4)                     # the budget ran out, not the list
        self.assertLess(given[-1], given[0])                    # each try gets what is left

    def test_a_dead_first_answer_moves_on_only_to_checked_ones(self):
        seen = []
        with Web() as web:
            def connect(ip, port, timeout):
                seen.append(ip)
                if ip == "2606:4700::1":
                    raise OSError("no route")
                return socket.create_connection(("127.0.0.1", web.port), timeout)
            with mock.patch.object(image_source.socket, "getaddrinfo", resolver(["2606:4700::1", "93.184.216.34"])), \
                    mock.patch.object(image_source, "_connect", connect):
                load_image(f"http://images.example:{web.port}/a.png", remote((f"images.example:{web.port}",)))
        self.assertEqual(seen, ["2606:4700::1", "93.184.216.34"])

    def test_proxy_settings_cannot_carry_a_remote_fetch(self):
        with Web() as web, mock.patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:1", "HTTP_PROXY": "http://127.0.0.1:1"}):
            self.assertEqual(load_image(web.url("/a.png"), remote((f"127.0.0.1:{web.port}",))), PNG)


@unittest.skipUnless(shutil.which("openssl"), "needs the openssl command")
class PinnedTls(unittest.TestCase):
    """TLS over the checked address: the name sent and the certificate check stay on the original host."""

    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp(prefix="image-tls-test-"))
        cls.addClassCleanup(shutil.rmtree, cls.dir, ignore_errors=True)
        for name in ("images.example", "other.example"):
            subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "2", "-subj", f"/CN={name}",
                            "-addext", f"subjectAltName=DNS:{name}", "-keyout", cls.dir / f"{name}.key",
                            "-out", cls.dir / f"{name}.crt"], check=True, capture_output=True)

    def fetch(self, cert, trust=None):
        tls = (self.dir / f"{cert}.crt", self.dir / f"{cert}.key")
        seen = []
        with Web(tls=tls) as web:
            def connect(ip, port, timeout):
                seen.append(ip)
                return socket.create_connection(("127.0.0.1", web.port), timeout)
            context = ssl.create_default_context(cafile=self.dir / f"{trust or cert}.crt")
            with mock.patch.object(image_source.socket, "getaddrinfo", resolver(["93.184.216.34"], ["127.0.0.1"])), \
                    mock.patch.object(image_source, "_connect", connect), \
                    mock.patch.object(image_source, "_tls_context", return_value=context):
                data = load_image(f"https://images.example:{web.port}/a.png", remote((f"images.example:{web.port}",)))
            return data, web, seen

    def test_the_certificate_is_checked_against_the_original_host(self):
        data, web, seen = self.fetch("images.example")
        self.assertEqual(data, PNG)
        self.assertEqual(seen, ["93.184.216.34"])
        self.assertEqual(web.sni, ["images.example"])
        self.assertEqual(web.requests[0][1], f"images.example:{web.port}")

    def test_a_certificate_for_another_name_is_refused(self):
        with self.assertRaises(ImageRefused) as cm:
            self.fetch("other.example")
        self.assertEqual(str(cm.exception), "the image URL could not be fetched")
        self.assertIn("reason=", cm.exception.log_line)

    def test_an_untrusted_certificate_is_refused(self):
        with self.assertRaises(ImageRefused):
            self.fetch("images.example", trust="other.example")


class Redirects(unittest.TestCase):
    def test_every_3xx_from_a_real_handler_is_refused(self):
        with Web() as web:
            allowed = remote((f"127.0.0.1:{web.port}",))
            for code in (300, 301, 302, 303, 304, 305, 307, 308):
                with self.subTest(code=code), self.assertRaises(ImageRefused) as cm:
                    load_image(web.url(f"/code/{code}"), allowed)
                self.assertIn(POLICY, str(cm.exception))
                self.assertIn("redirect", str(cm.exception))
                self.assertIn("--image-url-host", str(cm.exception))
                self.assertNotIn("169.254", str(cm.exception))   # the target stays out of the answer
            # one request each: the Location was never followed
            self.assertEqual([p for p, _, _ in web.requests], [f"/code/{c}" for c in (300, 301, 302, 303, 304, 305, 307, 308)])

    def test_a_local_request_follows_a_redirect(self):
        with Web() as web:
            self.assertEqual(load_image(web.url("/redirect"), LOCAL), PNG)

    def test_a_redirect_is_not_a_way_to_a_file(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(302)
                self.send_header("Location", "file:///etc/passwd")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *a):
                pass
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        for allowed in (LOCAL, remote((f"127.0.0.1:{httpd.server_address[1]}",))):
            with self.assertRaises(ValueError):
                load_image(f"http://127.0.0.1:{httpd.server_address[1]}/", allowed)


@posix_only
class AllowedDirs(unittest.TestCase):
    def setUp(self):
        base = Path(tempfile.mkdtemp(prefix="image-dirs-test-"))
        self.addCleanup(shutil.rmtree, base, ignore_errors=True)
        self.base = base
        self.root, self.outside = base / "allowed", base / "outside"
        self.root.mkdir()
        self.outside.mkdir()
        (self.root / "a.png").write_bytes(PNG)
        (self.outside / "secret").write_bytes(b"TOP-SECRET")
        self.allowed = remote(dirs=(str(self.root),))

    def refused(self, source):
        with self.assertRaisesRegex(ImageRefused, POLICY) as cm:
            load_image(source, self.allowed)
        return str(cm.exception)

    def test_inside(self):
        f = self.root / "a.png"
        for source in (str(f), f.as_uri(), f"{self.root}/./a.png", f"{self.root}/sub/../a.png"):
            self.assertEqual(load_image(source, self.allowed), PNG, source)

    def test_a_symlink_inside_that_stays_inside(self):
        (self.root / "link.png").symlink_to(self.root / "a.png")
        (self.root / "d").mkdir()
        (self.root / "d" / "up.png").symlink_to("../a.png")
        self.assertEqual(load_image(str(self.root / "link.png"), self.allowed), PNG)
        self.assertEqual(load_image(str(self.root / "d" / "up.png"), self.allowed), PNG)

    def test_an_allowed_dir_reached_through_a_symlink(self):
        alias = self.base / "alias"
        alias.symlink_to(self.root)
        allowed = remote(dirs=(str(alias),))
        self.assertEqual(load_image(str(alias / "a.png"), allowed), PNG)
        self.assertEqual(load_image(str(self.root / "a.png"), allowed), PNG)

    def test_escapes_and_their_text_is_one(self):
        (self.root / "link").symlink_to(self.outside / "secret")
        (self.root / "dirlink").symlink_to(self.outside)
        (self.root / "dangling").symlink_to(self.base / "nothing")
        forms = [str(self.outside / "secret"), f"{self.root}/../outside/secret", f"{self.root}/link",
                 f"{self.root}/dirlink/secret", f"{self.root}/dangling", f"{self.root}/missing.png", str(self.root),
                 f"file://{self.outside}/secret", f"file://{self.root}/%2e%2e/outside/secret",
                 f"file://other-host{self.root}/a.png", "relative.png", "a.png", "../a.png", "//host/share/a.png",
                 f"{self.root}-sibling/a.png", f"{self.root}/a.png\x00.txt", "/etc/passwd", "/proc/self/environ",
                 "/does/not/exist", "\\\\host\\share\\a.png", "C:\\Windows\\win.ini"]
        (self.base / "allowed-sibling").mkdir()
        (self.base / "allowed-sibling" / "a.png").write_bytes(PNG)
        texts = {self.refused(source) for source in forms}
        self.assertEqual(len(texts), 1, texts)
        text = texts.pop()
        self.assertIn("--image-dir", text)
        for leak in (str(self.root), "secret", "exist", "TOP"):
            self.assertNotIn(leak, text)

    def test_no_directories_allowed(self):
        self.allowed = remote()
        with mock.patch.object(os, "open", side_effect=AssertionError("opened")), \
                mock.patch.object(os.path, "realpath", side_effect=AssertionError("resolved")):
            self.refused(str(self.root / "a.png"))

    def test_nothing_outside_is_opened(self):
        with mock.patch.object(os, "open", side_effect=AssertionError("opened")):
            self.refused("/etc/passwd")
            self.refused(f"{self.root}/../outside/secret")

    def test_any_alias_that_resolves_under_the_directory(self):
        alias = self.base / "alias"
        alias.symlink_to(self.root)
        (self.base / "via").mkdir()
        (self.base / "via" / "one.png").symlink_to(self.root / "a.png")
        for path in (alias / "a.png", self.base / "via" / "one.png", f"{self.outside}/../allowed/a.png"):
            self.assertEqual(load_image(str(path), self.allowed), PNG, path)

    def test_a_traverse_only_directory(self):
        if not hasattr(os, "O_PATH"):
            self.skipTest("needs O_PATH")
        locked = self.root / "locked"
        locked.mkdir()
        (locked / "a.png").write_bytes(PNG)
        locked.chmod(0o111)
        self.addCleanup(locked.chmod, 0o755)
        self.assertEqual(load_image(str(locked / "a.png"), self.allowed), PNG)
        locked.chmod(0o711)
        self.assertEqual(load_image(str(locked / "a.png"), self.allowed), PNG)
        self.root.chmod(0o111)                                  # the directory itself too
        self.addCleanup(self.root.chmod, 0o755)
        self.assertEqual(load_image(str(locked / "a.png"), self.allowed), PNG)

    def test_an_unreadable_file_inside_reads_like_a_missing_one(self):
        if os.geteuid() == 0:
            self.skipTest("root reads everything")
        f = self.root / "noperm.png"
        f.write_bytes(PNG)
        f.chmod(0)
        self.assertEqual(self.refused(str(f)), self.refused(str(self.root / "missing.png")))

    def test_inside_the_directory_a_readable_file_can_answer_differently(self):
        # the docs' exact claim: too large, or not an image, is told; the operator allowed this directory
        f = self.root / "big.png"
        with f.open("wb") as out:
            out.truncate(image_source.IMAGE_MAX_BYTES + 1)
        with self.assertRaises(ImageRefused) as cm:
            load_image(str(f), self.allowed)
        self.assertNotIn(POLICY, str(cm.exception))

    def test_non_regular_files_do_not_hang(self):
        fifo = self.root / "fifo"
        os.mkfifo(fifo)
        zero = self.root / "zero"
        zero.symlink_to("/dev/zero")
        t0 = time.monotonic()
        for source in (str(fifo), str(zero), str(self.root)):
            self.refused(source)
        self.assertLess(time.monotonic() - t0, 5)

    def test_a_swap_to_a_symlink_after_the_check(self):
        # the race: the path resolves clean, then a component becomes a symlink out of the directory
        (self.root / "d").mkdir()
        (self.root / "d" / "f.png").write_bytes(PNG)
        real_realpath = os.path.realpath

        def swapping(component):
            def fn(p, *a, **kw):
                out = real_realpath(p, *a, **kw)
                if str(p).startswith(str(self.root)):
                    victim = self.root / component
                    if victim.is_dir() and not victim.is_symlink():
                        shutil.rmtree(victim)
                    elif victim.exists():
                        victim.unlink()
                    victim.symlink_to(self.outside / "secret" if component.endswith("file.png") else self.outside)
                return out
            return fn
        (self.root / "file.png").write_bytes(PNG)
        with mock.patch.object(os.path, "realpath", swapping("file.png")):
            self.refused(str(self.root / "file.png"))
        (self.outside / "f.png").write_bytes(b"TOP-SECRET")
        with mock.patch.object(os.path, "realpath", swapping("d")):
            self.refused(str(self.root / "d" / "f.png"))

    def test_more_escape_spellings(self):
        forms = [f"file://{self.root}/%2e%2e%2foutside%2fsecret", f"file://{self.root}/%252e%252e/outside/secret",
                 f"{self.root}/" + "x" * 5000, f"{self.root}/" + "d/" * 3000 + "a.png", f"{self.root}/\ud800.png",
                 f"{self.root}/a.png/", f"{self.root}/a.png/.", f"{self.root}/a.png/..", f"{self.root}/link/",
                 f" {self.root}/a.png", f"file://localhost.evil{self.root}/a.png", f"file:{self.root}/../outside/secret",
                 "c://windows/win.ini", "file:", "file:/", "file://", "file:///"]
        (self.root / "link").symlink_to(self.outside)
        for source in forms:
            with self.subTest(source=source[:80]):
                try:
                    data = load_image(source, self.allowed)
                except ValueError as e:
                    self.assertIn(POLICY, str(e))
                    continue
                self.assertEqual(data, PNG)                      # inside: a spelling of the allowed file

    def test_the_fallback_open_without_dir_fds(self):
        # the portable path Windows takes
        with mock.patch.object(os, "supports_dir_fd", set()):
            self.assertEqual(load_image(str(self.root / "a.png"), self.allowed), PNG)
            (self.root / "file.png").write_bytes(PNG)
            real_realpath = os.path.realpath
            done = []

            def swapping(p, *a, **kw):
                out = real_realpath(p, *a, **kw)
                if str(p).endswith("file.png") and not done:
                    done.append(1)
                    (self.root / "file.png").unlink()
                    (self.root / "file.png").symlink_to(self.outside / "secret")
                return out
            with mock.patch.object(os.path, "realpath", swapping):
                self.refused(str(self.root / "file.png"))

    def test_oversize(self):
        f = self.root / "big.png"
        with mock.patch.object(image_source, "IMAGE_MAX_BYTES", len(PNG) - 1):
            with self.assertRaisesRegex(ValueError, "larger than"):
                load_image(str(self.root / "a.png"), self.allowed)
        with f.open("wb") as out:
            out.truncate(image_source.IMAGE_MAX_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "larger than"):
            load_image(str(f), self.allowed)

    def test_no_handle_is_left_open(self):
        (self.root / "d").mkdir()
        before = len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else None
        if before is None:
            self.skipTest("needs /proc")
        for _ in range(20):
            with contextlib.suppress(ValueError):
                load_image(str(self.root / "d"), self.allowed)
            load_image(str(self.root / "a.png"), self.allowed)
        time.sleep(0.1)
        self.assertLessEqual(len(os.listdir("/proc/self/fd")), before + 2)


@posix_only
class LocalBounds(unittest.TestCase):
    """The size cap and the deadline hold for a local client too."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="image-bounds-test-"))
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)

    def test_special_files(self):
        fifo = self.d / "fifo"
        os.mkfifo(fifo)
        t0 = time.monotonic()
        for path in (fifo, "/dev/zero", "/dev/null", self.d):
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                load_image(str(path), LOCAL)
        self.assertLess(time.monotonic() - t0, 5)

    def test_a_file_swapped_for_a_fifo_at_the_open_does_not_hang(self):
        fifo = self.d / "fifo"
        os.mkfifo(fifo)
        regular = os.stat_result((0o100644,) + (0,) * 9)
        t0 = time.monotonic()
        with mock.patch.object(image_source.os, "stat", return_value=regular), \
                self.assertRaisesRegex(ValueError, "not a regular file"):
            load_image(str(fifo), LOCAL)
        self.assertLess(time.monotonic() - t0, 5)

    def test_oversize_file(self):
        f = self.d / "big.png"
        with f.open("wb") as out:
            out.truncate(image_source.IMAGE_MAX_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "larger than"):
            load_image(str(f), LOCAL)

    def test_a_file_that_reports_no_size(self):
        with mock.patch.object(image_source, "IMAGE_MAX_BYTES", 8):
            f = self.d / "x"
            f.write_bytes(b"123456789")
            real = os.fstat

            class Zero:
                def __init__(self, st):
                    self.st_mode, self.st_size = st.st_mode, 0

            with mock.patch.object(image_source.os, "fstat", side_effect=lambda fd: Zero(real(fd))), \
                    self.assertRaisesRegex(ValueError, "larger than"):
                load_image(str(f), LOCAL)

    def test_a_blocked_read_hits_the_deadline(self):
        release = threading.Event()
        self.addCleanup(release.set)
        with mock.patch.object(image_source, "IMAGE_READ_S", 0.3), \
                mock.patch.object(image_source, "_read_local_file", side_effect=lambda *a: release.wait(30)):
            t0 = time.monotonic()
            with self.assertRaisesRegex(ValueError, "did not answer within"):
                load_image("/some/dead/mount/a.png", LOCAL)
        self.assertLess(time.monotonic() - t0, 3)

    def test_urls_are_bounded_from_every_peer(self):
        with Web() as web:
            for allowed in (LOCAL, remote((f"127.0.0.1:{web.port}",))):
                with self.subTest(local=allowed.local):
                    for path in ("/big", "/big-unsized"):
                        with mock.patch.object(image_source, "IMAGE_MAX_BYTES", 1 << 20):
                            with self.assertRaisesRegex(ValueError, "larger than"):
                                load_image(web.url(path), allowed)
                    with mock.patch.object(image_source, "IMAGE_READ_S", 0.3):
                        t0 = time.monotonic()
                        with self.assertRaisesRegex(ValueError, "did not answer within"):
                            load_image(web.url("/drip"), allowed)
                        self.assertLess(time.monotonic() - t0, 3)

    def _dripping_listener(self, first=b"HTTP/1.1 200 OK\r\n", read=True):
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(16)
        self.addCleanup(srv.close)

        def one(c):
            with contextlib.suppress(OSError), c:
                if read:
                    c.recv(4096)
                c.sendall(first)
                while True:
                    c.sendall(b"X")
                    time.sleep(0.05)

        def accept():
            with contextlib.suppress(OSError):
                while True:
                    threading.Thread(target=one, args=(srv.accept()[0],), daemon=True).start()
        threading.Thread(target=accept, daemon=True).start()
        return srv.getsockname()[1]

    def _no_read_threads_left(self, base):
        end = time.monotonic() + 5
        while time.monotonic() < end and sum(t.name == "image-read" for t in threading.enumerate()) > base:
            time.sleep(0.05)
        self.assertEqual(sum(t.name == "image-read" for t in threading.enumerate()), base)

    def test_a_header_drip_keeps_no_thread(self):
        port = self._dripping_listener()
        for allowed in (LOCAL, remote((f"127.0.0.1:{port}",))):
            with self.subTest(local=allowed.local), mock.patch.object(image_source, "IMAGE_READ_S", 0.3):
                base = sum(t.name == "image-read" for t in threading.enumerate())
                t0 = time.monotonic()
                with self.assertRaisesRegex(ValueError, "did not answer within"):
                    load_image(f"http://127.0.0.1:{port}/x", allowed)
                self.assertLess(time.monotonic() - t0, 3)
                self._no_read_threads_left(base)

    def test_a_body_drip_keeps_no_thread(self):
        with Web() as web:
            for allowed in (LOCAL, remote((f"127.0.0.1:{web.port}",))):
                with self.subTest(local=allowed.local), mock.patch.object(image_source, "IMAGE_READ_S", 0.3):
                    base = sum(t.name == "image-read" for t in threading.enumerate())
                    with self.assertRaisesRegex(ValueError, "did not answer within"):
                        load_image(web.url("/drip"), allowed)
                    self._no_read_threads_left(base)

    def test_a_tls_handshake_drip_keeps_no_thread(self):
        port = self._dripping_listener(first=b"\x16\x03\x03\x40\x00")
        for allowed in (LOCAL, remote((f"127.0.0.1:{port}",))):
            with self.subTest(local=allowed.local), mock.patch.object(image_source, "IMAGE_READ_S", 0.3):
                base = sum(t.name == "image-read" for t in threading.enumerate())
                with self.assertRaisesRegex(ValueError, "did not answer within"):
                    load_image(f"https://127.0.0.1:{port}/x", allowed)
                self._no_read_threads_left(base)

    def test_no_descriptor_is_left_by_a_fetch(self):
        if not os.path.isdir("/proc/self/fd"):
            self.skipTest("needs /proc")
        with Web() as web:
            before = len(os.listdir("/proc/self/fd"))
            for _ in range(30):
                load_image(web.url("/a.png"), LOCAL)
                load_image(web.url("/a.png"), remote((f"127.0.0.1:{web.port}",)))
            time.sleep(0.2)
            self.assertLessEqual(len(os.listdir("/proc/self/fd")), before + 4)

    def test_a_body_cut_short_is_an_error(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(8)
        self.addCleanup(srv.close)

        def serve_short():
            with contextlib.suppress(OSError):
                while True:
                    c = srv.accept()[0]
                    c.recv(4096)
                    c.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nshort")
                    c.close()
        threading.Thread(target=serve_short, daemon=True).start()
        port = srv.getsockname()[1]
        for allowed in (LOCAL, remote((f"127.0.0.1:{port}",))):
            with self.subTest(local=allowed.local), self.assertRaisesRegex(ValueError, "could not be"):
                load_image(f"http://127.0.0.1:{port}/x", allowed)

    def test_a_slow_source_does_not_hold_the_request_turn(self):
        with Web() as web:
            tok = ByteTokenizer()
            vision = RecordingVision()
            self.addCleanup(shutil.rmtree, vision.dir, ignore_errors=True)
            svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate.from_file(ROOT / "serve/chat_template.jinja"),
                          vision=vision)
            turn_free = []

            def spying(source, allowed=None):
                turn_free.append(svc.fifo.acquire(blocking=False))
                if turn_free[-1]:
                    svc.fifo.release()
                return load_image(source, allowed)
            msgs = [{"role": "user", "content": [{"type": "image", "source": web.url("/a.png")}]}]
            with mock.patch("serve.engine.vision.load_image", spying):
                svc.prepare(msgs, None, {}, image_access=LOCAL)
            self.assertEqual(turn_free, [True])
            self.assertEqual(vision.got, [PNG])
            svc.embeddings.path.unlink(missing_ok=True)

    def test_a_request_total(self):
        v = RecordingVision()
        self.addCleanup(shutil.rmtree, v.dir, ignore_errors=True)
        f, g = self.d / "a.png", self.d / "b.png"
        f.write_bytes(PNG)
        g.write_bytes(PNG + b"2")
        from serve.engine import vision
        with mock.patch.object(vision, "IMAGE_REQUEST_BYTES", len(PNG) + 1):
            self.assertEqual(v.read([str(f), str(f), str(f)], LOCAL), [PNG] * 3)   # one source, read once
            with self.assertRaisesRegex(ValueError, "request's images are larger"):
                v.read([str(f), str(g)], LOCAL)


class DataUrls(unittest.TestCase):
    def test_accepted_from_any_policy(self):
        b64 = base64.b64encode(PNG).decode()
        wrapped = "\r\n".join(b64[i:i + 4] for i in range(0, len(b64), 4))
        for source in (DATA, f"DATA:image/png;BASE64,{b64}", f"data:;base64,{b64}", f"data:image/png;base64,{wrapped}",
                       f"data:image/png;charset=x;base64, {b64}"):
            for allowed in (None, LOCAL):
                self.assertEqual(load_image(source, allowed), PNG, source)

    def test_invalid(self):
        for source, text in (("data:", "comma"), ("data:image/png;base64", "comma"), ("data:image/png;base64,", "empty"),
                             ("data:image/png,AAAA", "must be base64"), ("data:,hello", "must be base64"),
                             ("data:;base64,@@@@", "not valid base64"), ("data:;base64,AAA", "not valid base64"),
                             ("data:;base64,A", "not valid base64"), ("data:;base64,caf\u00e9", "not valid base64"),
                             ("data:;base64,a-_b", "not valid base64"), ("data:;base64,%41%41", "not valid base64")):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, text):
                load_image(source, LOCAL)

    def test_oversize_is_refused_before_it_is_decoded(self):
        huge = "data:;base64," + "A" * (((image_source.IMAGE_MAX_BYTES + 2) // 3 * 4) + 4)
        with mock.patch.object(image_source.base64, "b64decode", side_effect=AssertionError("decoded")), \
                self.assertRaisesRegex(ValueError, "larger than 32 MiB"):
            load_image(huge)

    def test_exactly_the_cap(self):
        with mock.patch.object(image_source, "IMAGE_MAX_BYTES", 6):
            self.assertEqual(load_image("data:;base64," + base64.b64encode(b"123456").decode()), b"123456")
            with self.assertRaisesRegex(ValueError, "larger than"):
                load_image("data:;base64," + base64.b64encode(b"1234567").decode())


class WaitingImages(ImageServer):
    def test_a_slot_only_for_pillow(self):
        from serve.engine import vision
        with mock.patch.object(vision, "_decodes", threading.Semaphore(0)):
            self.assertEqual(Vision.normalize(PNG), PNG)         # no slot free, and no wait
            for data in (b"BM" + b"x" * 20, b"GIF89a" + b"x" * 20):
                self.assertEqual(Vision.normalize(data), data)
            done = threading.Event()

            def converting():
                with contextlib.suppress(ValueError):
                    Vision.normalize(b"not a passthrough format")
                done.set()
            threading.Thread(target=converting, daemon=True).start()
            self.assertFalse(done.wait(0.2))                     # Pillow's work does wait
            vision._decodes.release()
            self.assertTrue(done.wait(5))

    def test_a_request_past_the_waiting_budget_is_told_to_retry(self):
        import serve.service as service
        self.start()
        with mock.patch.object(service, "IMAGE_WAITING_BYTES", len(PNG) - 1):
            reply = self.post("/v1/chat/completions", openai_body(DATA))
        self.assertEqual(reply[0], 409, reply[1])
        self.assertIn("try again", reply[1])
        self.assertEqual(self.vision.got, [])
        self.assertEqual(self.svc.images_waiting, 0)
        self.assertReadIt(self.post("/v1/chat/completions", openai_body(DATA)))
        self.assertEqual(self.svc.images_waiting, 0)

    def test_requests_waiting_for_the_turn_count_together(self):
        import serve.service as service
        self.start()
        with mock.patch.object(service, "IMAGE_WAITING_BYTES", len(PNG) + 1):
            with self.svc.fifo:                                  # a request is encoding: the next ones wait
                out = []
                first = threading.Thread(target=lambda: out.append(self.post("/v1/chat/completions", openai_body(DATA))))
                first.start()
                end = time.monotonic() + 10
                while not self.svc.images_waiting and time.monotonic() < end:
                    time.sleep(0.01)
                self.assertEqual(self.svc.images_waiting, len(PNG))
                self.assertEqual(self.post("/v1/chat/completions", openai_body(DATA))[0], 409)
            first.join(30)
        self.assertEqual(out[0][0], 200)
        self.assertEqual(self.svc.images_waiting, 0)

    def test_a_failed_encode_gives_its_budget_back(self):
        self.start()
        self.vision.encode = mock.Mock(side_effect=ValueError("the image could not be read"))
        self.assertEqual(self.post("/v1/chat/completions", openai_body(DATA))[0], 400)
        self.assertEqual(self.svc.images_waiting, 0)


class Decoder(unittest.TestCase):
    def setUp(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("needs Pillow")

    def test_a_bomb_is_refused_before_its_pixels(self):
        # a TIFF header claiming far more pixels than its bytes hold
        import struct
        tags = [(256, 4, 1, 40000), (257, 4, 1, 40000), (258, 3, 1, 8), (259, 3, 1, 1), (262, 3, 1, 1), (273, 4, 1, 8),
                (277, 3, 1, 1), (278, 4, 1, 40000), (279, 4, 1, 16)]
        tiff = b"II*\x00\x08\x00\x00\x00" + struct.pack("<H", len(tags))
        tiff += b"".join(struct.pack("<HHII", *t) for t in tags) + b"\x00\x00\x00\x00"
        with self.assertRaisesRegex(ValueError, "megapixels"):
            Vision.normalize(tiff)

    def test_ghostscript_is_out_of_reach(self):
        eps = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\nshowpage\n"
        with mock.patch("subprocess.check_call", side_effect=AssertionError("ran ghostscript")), \
                mock.patch("subprocess.Popen", side_effect=AssertionError("ran ghostscript")):
            with self.assertRaisesRegex(ValueError, "not a format this server decodes"):
                Vision.normalize(eps)

    def test_formats_out_of_reach(self):
        import struct
        from PIL import Image
        out = io.BytesIO()
        Image.new("RGBA", (8, 8)).save(out, format="PNG")
        png = out.getvalue()
        ico = struct.pack("<HHH", 0, 1, 1) + struct.pack("<BBBBHHII", 1, 1, 0, 0, 1, 32, len(png), 22) + png
        with self.assertRaisesRegex(ValueError, "not a format this server decodes"):
            Vision.normalize(ico)                                # its header claims 1x1, its PNG holds what it likes
        jp2 = io.BytesIO()
        try:
            Image.new("RGB", (8, 8)).save(jp2, format="JPEG2000")
        except (KeyError, OSError):
            self.skipTest("this Pillow has no JPEG 2000")
        with self.assertRaisesRegex(ValueError, "not a format this server decodes"):
            Vision.normalize(jp2.getvalue())

    def test_the_rotation_check_opens_jpeg_only(self):
        from PIL import Image
        seen = []
        real = Image.open

        def spying(fp, *a, **kw):
            seen.append(kw.get("formats"))
            return real(fp, *a, **kw)
        with mock.patch.object(Image, "open", spying):
            self.assertFalse(Vision._exif_rotated(b"\xff\xd8\xff" + b"\0" * 200))
        self.assertEqual(seen, [["JPEG"]])

    def test_a_non_image_gives_no_content_back(self):
        with self.assertRaises(ValueError) as cm:
            Vision.normalize(b"root:x:0:0:SECRET:/root:/bin/bash\n")
        self.assertNotIn("SECRET", str(cm.exception))
        self.assertNotIn("0x", str(cm.exception))

    def test_webp_still_reads(self):
        from PIL import Image
        out = io.BytesIO()
        try:
            Image.new("RGB", (4, 4)).save(out, format="WEBP")
        except (KeyError, OSError):
            self.skipTest("this Pillow has no WebP")
        self.assertEqual(Vision.normalize(out.getvalue())[:4], b"\x89PNG")


class DriveLikeSources(unittest.TestCase):
    def test_a_double_slash_drive_is_a_path_not_a_scheme(self):
        with self.assertRaisesRegex(ImageRefused, "--image-dir"):
            load_image("c://windows/win.ini")
        with self.assertRaisesRegex(ValueError, "does not exist|could not be read"):
            load_image("c://no/such/file.png", LOCAL)


class WindowsForms(unittest.TestCase):
    """Parsed without touching a file, so they run anywhere."""

    def test_refused(self):
        for path in ("\\\\host\\share\\a.png", "//host/share/a.png", "\\\\.\\PhysicalDrive0", "\\\\?\\C:\\a.png",
                     "\\??\\C:\\a.png", "\\\\.\\C:\\a.png", "\\/host\\share", "\\Windows\\win.ini", "/etc/passwd", "C:a.png",
                     "C:", "a.png", "CON", "C:\\img\\CON", "C:\\img\\con.png", "C:\\img\\NUL", "C:\\img\\nul.tar.gz",
                     "C:\\img\\AUX.jpg", "C:\\img\\PRN", "C:\\img\\COM1", "C:\\img\\com9.png", "C:\\img\\LPT1.png",
                     "C:\\img\\COM\u00b9", "C:\\img\\CONIN$", "C:\\img\\CONOUT$", "C:\\img\\a.png:stream",
                     "C:\\img\\a.png::$DATA", "C:\\img\\a.png:stream:$DATA", "C:\\img\\d:x\\a.png", "C:\\img\\a.png.",
                     "C:\\img\\a.png ", "C:\\img\\a.png...", "C:\\img\\dir \\a.png", "C:\\img\\a*.png", "C:\\img\\a?.png",
                     "C:\\img\\a\x01.png", "1:\\img\\a.png", "C:\\img\\NUL .png"):
            self.assertTrue(windows_path_refused(path), path)

    def test_accepted(self):
        for path in ("C:\\img\\a.png", "c:/img/a.png", "D:\\img\\sub dir\\a b.png", "C:\\img\\console.png",
                     "C:\\img\\nullable.png", "C:\\img\\COM10.png", "C:\\img\\..\\img\\a.png"):
            self.assertFalse(windows_path_refused(path), path)

    def test_file_urls(self):
        for url, path in (("file:///C:/img/a.png", "C:/img/a.png"), ("file://C:/img/a.png", "C:/img/a.png"),
                          ("file://C:\\img\\a.png", "C:\\img\\a.png"), ("file:///C:/a%20b.png", "C:/a b.png"),
                          ("file://localhost/C:/a.png", "C:/a.png"), ("file:/C:/a.png", "C:/a.png")):
            self.assertEqual(file_url_path(url, windows=True), path, url)
        for url in ("file://host/share/a.png", "file://1.2.3.4/a.png"):
            with self.assertRaisesRegex(ValueError, "another host"):
                file_url_path(url, windows=True)

    def test_posix_file_urls(self):
        for url, path in (("file:///tmp/a.png", "/tmp/a.png"), ("file:/tmp/a.png", "/tmp/a.png"),
                          ("file://localhost/tmp/a.png", "/tmp/a.png"), ("file:///tmp/a%20b.png", "/tmp/a b.png"),
                          ("file:///tmp/100%.png", "/tmp/100%.png")):
            self.assertEqual(file_url_path(url, windows=False), path, url)
        for url in ("file://host/a.png", "file://C:/a.png"):
            with self.assertRaises(ValueError):
                file_url_path(url, windows=False)

    def test_a_windows_form_never_reaches_the_read(self):
        allowed = remote(dirs=(tempfile.gettempdir(),))
        with mock.patch.object(os.path, "isabs", return_value=True), \
                mock.patch.object(image_source, "_open_under", side_effect=AssertionError("opened")):
            for path in ("\\\\host\\share\\a.png", "C:\\img\\a.png:stream", "C:\\img\\CON"):
                with self.assertRaisesRegex(ImageRefused, POLICY):
                    image_source._read_allowed_file(path, allowed, windows=True)


class Leaks(ImageServer):
    """What a refused client learns, and what reaches the log."""
    peer = (("203.0.113.9", 1),)
    SIGNED = "user:pw@images.example/private/a.png?X-Amz-Signature=SIGSECRET&token=SIGSECRET#SIGSECRET"

    def logged(self, *a, **kw):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            reply = self.chat(*a, **kw)
        return reply, out.getvalue()

    def test_the_log_holds_no_query_credentials_or_path(self):
        self.start()
        for source in (f"https://{self.SIGNED}", "https://images.example/private/a.png?sig=SIGSECRET"):
            reply, text = self.logged(source)
            self.assertEqual(reply[0], 400)
            for leak in ("SIGSECRET", "user:pw", "private", "a.png"):
                self.assertNotIn(leak, text)
                self.assertNotIn(leak, reply[1])
            self.assertIn("kind=url", text)
        self.assertIn("images.example", text)
        _, text = self.logged("/home/someone/private/SIGSECRET.png")
        self.assertNotIn("SIGSECRET", text)
        self.assertNotIn("private", text)
        self.assertIn("kind=path", text)

    def test_an_allowed_hosts_failures_log_the_host_alone(self):
        with Web() as web:
            self.svc.image_sources = image_sources_of((f"127.0.0.1:{web.port}",))
            self.start()
            for path in ("/nothing?sig=SIGSECRET", "/code/302?sig=SIGSECRET", "/big?sig=SIGSECRET"):
                with mock.patch.object(image_source, "IMAGE_MAX_BYTES", 1 << 20):
                    reply, text = self.logged(web.url(path))
                self.assertEqual(reply[0], 400)
                self.assertNotIn("SIGSECRET", text)
                self.assertNotIn("SIGSECRET", reply[1])
                self.assertIn("127.0.0.1", text)

    def test_a_local_clients_failures_tell_it_but_not_the_log(self):
        self.start(peer=(("127.0.0.1", 1),))
        with Web() as web:
            for source in (web.url("/nothing?sig=SIGSECRET"), "/no/such/SIGSECRET.png",
                           f"http://u:SIGSECRET@127.0.0.1:{web.port}/nothing"):
                reply, text = self.logged(source)
                self.assertEqual(reply[0], 400)
                self.assertIn("SIGSECRET", reply[1])             # its own request
                self.assertNotIn("SIGSECRET", text)
                self.assertEqual(text.count("\n"), 1)

    def test_a_remote_answer_says_nothing_about_the_file(self):
        self.start()
        secret = self.tmp / "secret-name"
        secret.write_bytes(b"TOP-SECRET")
        a = self.chat(str(secret))
        b = self.chat(str(self.tmp / "no-such-name"))
        c = self.chat("/etc/shadow")
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(a[0], 400)
        self.assertRefused(a, "--image-dir")
        for leak in ("secret-name", "TOP-SECRET", str(self.tmp), "exist"):
            self.assertNotIn(leak, a[1])

    def test_with_a_directory_set_the_answer_does_not_ask_to_set_one(self):
        secret = self.tmp / "elsewhere"
        secret.mkdir()
        (secret / "real.png").write_bytes(PNG)
        allowed = self.tmp / "allowed"
        allowed.mkdir()
        self.svc.image_sources = image_sources_of((), (str(allowed),))
        self.start()
        reply = self.chat(str(secret / "real.png"))
        self.assertRefused(reply, "not a file under the directories allowed with --image-dir")
        self.assertNotIn("needs --image-dir", reply[1])
        self.assertNotIn(str(allowed), reply[1])
        self.assertEqual(reply, self.chat(str(secret / "missing.png")))     # exists or not
        self.assertEqual(reply, self.chat(str(allowed / "missing.png")))    # inside or not

    def test_with_a_host_set_the_answer_does_not_ask_to_set_one(self):
        self.svc.image_sources = image_sources_of(("images.example",))
        self.start()
        reply = self.chat("https://other.example/a.png")
        self.assertRefused(reply, "not among those allowed with --image-url-host")
        self.assertNotIn("needs --image-url-host", reply[1])
        self.assertNotIn("images.example", reply[1])
        self.assertEqual(reply, self.chat("http://10.1.2.3:8000/x.png"))

    def test_with_nothing_set_the_answer_names_the_flag_to_set(self):
        self.start()
        self.assertRefused(self.chat("/etc/passwd"), "a file path needs --image-dir on the server")
        self.assertRefused(self.chat("https://other.example/a.png"), "needs --image-url-host HOST on the server")

    def test_a_remote_answer_says_nothing_about_a_network(self):
        self.start()
        a = self.chat("http://10.1.2.3:8000/x.png")
        self.assertEqual(a, self.chat("https://images.example/a.png"))
        self.assertRefused(a, "--image-url-host")
        self.assertNotIn("10.1.2.3", a[1])

    def test_an_allowed_host_that_is_down(self):
        with mock.patch.object(image_source.socket, "getaddrinfo", resolver(OSError("internal-dns.corp says no"))):
            self.svc.image_sources = image_sources_of(("images.example",))
            self.start()
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                reply = self.chat("http://images.example/a.png")
        self.assertEqual(reply[0], 400)
        self.assertIn("could not be fetched", reply[1])
        self.assertNotIn("internal-dns", reply[1])
        self.assertIn("internal-dns", out.getvalue())           # the operator's log has the reason

    def test_a_private_answer_is_not_told(self):
        with mock.patch.object(image_source.socket, "getaddrinfo", resolver(["10.9.8.7"])):
            self.svc.image_sources = image_sources_of(("images.example",))
            self.start()
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                reply = self.chat("http://images.example/a.png")
        self.assertIn("non-public", reply[1])
        self.assertNotIn("10.9.8.7", reply[1])
        self.assertNotIn("10.9.8.7", out.getvalue())
        self.assertIn("non-public", out.getvalue())

    def test_the_log_line_is_one_escaped_line(self):
        self.start()
        evil = "/x\n[server] error: forged line\x1b[31m\r\u2028"
        for source in (evil, "http://evil.test/\n[server] forged\x1b[31m", "file:///x\n[server] forged", "http://h\x1b[0m/"):
            with self.subTest(source=source):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    reply = self.chat(source)
                self.assertEqual(reply[0], 400)
                text = out.getvalue()
                self.assertEqual(text.count("\n"), 1, text)
                self.assertEqual(sum(line.startswith("[server]") for line in text.splitlines()), 1)
                self.assertNotIn("\x1b", text)
                self.assertNotIn("\r", text)
                self.assertNotIn("\u2028", text)
                self.assertIn("peer='203.0.113.9'", text)

    @posix_only
    def test_a_local_error_is_escaped_too(self):
        out = io.StringIO()
        self.start(peer=(("127.0.0.1", 1),))
        with contextlib.redirect_stdout(out):
            reply = self.chat("/no/such\n[server] forged\x1b[31m")
        self.assertIn("does not exist", reply[1])
        self.assertEqual(out.getvalue().count("\n"), 1)
        self.assertNotIn("\x1b", out.getvalue())

    def test_a_lone_surrogate_does_not_break_the_answer(self):
        self.start()
        for source in ("/x/\ud800.png", "http://images.example/\ud800", "file:///\ud800"):
            reply = self.chat(source)
            self.assertRefused(reply)
        self.start(peer=(("127.0.0.1", 1),))
        self.assertEqual(self.chat("/x/\ud800.png")[0], 400)

    def test_the_responses_api_says_it_is_policy(self):
        self.start()
        reply = self.post("/v1/responses", frontends("/etc/passwd")["responses string"][1])
        self.assertEqual(reply[0], 400)
        error = json.loads(reply[1])["error"]
        self.assertIn(POLICY, error["message"])
        self.assertNotEqual(error.get("code"), "context_length_exceeded")

    def test_allowed_dir_flows_through_a_real_handler(self):
        self.svc.image_sources = image_sources_of((), (str(self.tmp),))
        self.start()
        self.assertReadIt(self.chat(str(self.image)))
        self.vision.got.clear()
        self.assertRefused(self.chat("/etc/passwd"), "--image-dir")

    def test_allowed_host_flows_through_a_real_handler(self):
        with Web() as web:
            self.svc.image_sources = image_sources_of((f"127.0.0.1:{web.port}",))
            self.start()
            self.assertReadIt(self.chat(web.url("/a.png")))
            self.vision.got.clear()
            self.assertRefused(self.chat(web.url("/code/302")), "--image-url-host", "redirect")


class Options(unittest.TestCase):
    def parse(self, *words):
        return model_settings(build_parser(), ["--mock-engine", *words])[0][0]

    def test_flags(self):
        a = self.parse("--image-url-host", "a.example", "--image-url-host", "b.example:81", "--image-dir", "/tmp")
        self.assertEqual(a.image_url_host, ["a.example", "b.example:81"])
        self.assertEqual(a.image_dir, ["/tmp"])
        self.assertEqual(self.parse().image_url_host, [])
        self.assertEqual(self.parse().image_dir, [])

    def test_bad_flags(self):
        for words in (("--image-url-host", "https://a.example/x"), ("--image-url-host", "*.a.example"),
                      ("--image-url-host", "a.example:http")):
            with self.subTest(words=words), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                self.parse(*words)

    @posix_only
    def test_config_keys(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d, "c.json")
            cfg.write_text(json.dumps({"models": [{"model": "m"}], "server": {"image-url-host": ["a.example", "b.example:81"], "image_dir": ["pics", "/abs"]}}))
            a = self.parse("--config", str(cfg))
            self.assertEqual(a.image_url_host, ["a.example", "b.example:81"])
            self.assertEqual(a.image_dir, [str(Path(d).resolve() / "pics"), "/abs"])

    def test_a_missing_directory(self):
        with self.assertRaisesRegex(ValueError, "--image-dir no-such-dir-here: not a directory"):
            image_sources_of((), ("no-such-dir-here",))

    def test_directory_is_resolved(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(image_sources_of((), (d,)).dirs, (os.path.realpath(d),))

    def test_remote_only_flag_and_key(self):
        self.assertTrue(self.parse("--image-remote-only").image_remote_only)
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d, "c.json")
            cfg.write_text(json.dumps({"models": [{"model": "m"}], "server": {"image-remote-only": True}}))
            self.assertTrue(self.parse("--config", str(cfg)).image_remote_only)
            cfg.write_text(json.dumps({"models": [{"model": "m"}], "server": {"image_remote_only": False}}))
            self.assertFalse(self.parse("--config", str(cfg)).image_remote_only)

    def test_a_listener_is_ipv4(self):
        # the docs promise no IPv6 listener, and a peer of any family is judged the same
        self.assertEqual(Server.address_family, socket.AF_INET)
        page = (ROOT / "docs/configuration/image-sources.md").read_text()
        self.assertNotIn("::1", page)

    def test_help_and_docs_name_them(self):
        text = build_parser().format_help()
        doc = (ROOT / "docs/configuration/index.md").read_text()
        page = (ROOT / "docs/configuration/image-sources.md").read_text()
        for name in ("image-url-host", "image-dir", "image-remote-only"):
            self.assertIn("--" + name, text)
            self.assertIn("`" + name + "`", doc)
            self.assertIn("--" + name, page)

    def test_the_default_policy_allows_nothing(self):
        s = ImageSources()
        self.assertFalse(s.local)
        self.assertEqual((s.url_hosts, s.dirs), ((), ()))


if __name__ == "__main__":
    unittest.main()
