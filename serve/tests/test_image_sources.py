"""The image-source policy: inline `data:` URLs by default, URL hosts and directories only when allowed."""
import base64
import contextlib
import http.server
import io
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from serve.config import build_parser
from serve.engine.vision import ImageSources, load_image

PNG = b"\x89PNG\r\n\x1a\n" + b"body"
DATA = "data:image/png;base64," + base64.b64encode(PNG).decode()


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/a.png")
            self.end_headers()
            return
        body = b"x" * 4096 if self.path == "/big" else PNG
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


class ImageSourcesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_data_url_is_always_allowed(self):
        self.assertEqual(load_image(DATA), PNG)

    def test_data_url_over_the_cap(self):
        with self.assertRaisesRegex(ValueError, "larger than"):
            load_image(DATA, ImageSources(max_bytes=4))

    def test_data_url_without_a_comma(self):
        with self.assertRaisesRegex(ValueError, "comma"):
            load_image("data:image/png;base64")

    def test_a_url_is_refused_by_default(self):
        with self.assertRaisesRegex(ValueError, "not allowed"):
            load_image(f"http://127.0.0.1:{self.port}/a.png")

    def test_a_file_is_refused_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, "a.png")
            path.write_bytes(PNG)
            with self.assertRaisesRegex(ValueError, "disabled"):
                load_image(str(path))

    def test_file_under_an_allowed_directory(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, "a.png")
            path.write_bytes(PNG)
            sources = ImageSources(dirs=(Path(d).resolve(),))
            self.assertEqual(load_image(str(path), sources), PNG)
            self.assertEqual(load_image(path.as_uri(), sources), PNG)

    def test_file_outside_the_allowed_directories(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as other:
            path = Path(other, "a.png")
            path.write_bytes(PNG)
            with self.assertRaisesRegex(ValueError, "outside"):
                load_image(str(path), ImageSources(dirs=(Path(d).resolve(),)))

    def test_a_symlink_out_of_the_allowed_directory(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as other:
            target = Path(other, "a.png")
            target.write_bytes(PNG)
            link = Path(d, "link.png")
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are not available here")
            with self.assertRaisesRegex(ValueError, "outside"):
                load_image(str(link), ImageSources(dirs=(Path(d).resolve(),)))

    def test_allowed_url_host(self):
        sources = ImageSources(url_hosts=(f"127.0.0.1:{self.port}",))
        self.assertEqual(load_image(f"http://127.0.0.1:{self.port}/a.png", sources), PNG)

    def test_another_port_of_an_allowed_host(self):
        sources = ImageSources(url_hosts=(f"127.0.0.1:{self.port + 1}",))
        with self.assertRaisesRegex(ValueError, "not allowed"):
            load_image(f"http://127.0.0.1:{self.port}/a.png", sources)

    def test_a_url_redirect_is_refused(self):
        sources = ImageSources(url_hosts=(f"127.0.0.1:{self.port}",))
        with self.assertRaisesRegex(ValueError, "redirected"):
            load_image(f"http://127.0.0.1:{self.port}/redirect", sources)

    def test_a_url_body_over_the_cap(self):
        sources = ImageSources(url_hosts=(f"127.0.0.1:{self.port}",), max_bytes=1024)
        with self.assertRaisesRegex(ValueError, "larger than"):
            load_image(f"http://127.0.0.1:{self.port}/big", sources)

    def test_an_allowed_name_that_resolves_to_this_machine(self):
        # a host name, unlike an IP literal, must resolve to a public address: an allowed name cannot be
        # pointed at a private one behind the operator's back
        sources = ImageSources(url_hosts=(f"localhost:{self.port}",))
        with self.assertRaises(ValueError):
            load_image(f"http://localhost:{self.port}/a.png", sources)

    def test_a_url_with_credentials(self):
        with self.assertRaisesRegex(ValueError, "user name or password"):
            load_image(f"http://user:pass@127.0.0.1:{self.port}/a.png",
                       ImageSources(url_hosts=(f"127.0.0.1:{self.port}",)))

    def test_a_non_http_url(self):
        with self.assertRaises(ValueError):
            load_image("ftp://127.0.0.1/a.png", ImageSources(url_hosts=("127.0.0.1",)))


class ImageDirStartupTest(unittest.TestCase):
    def test_a_missing_directory_stops_the_server(self):
        from serve.server import image_sources
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            image_sources(SimpleNamespace(image_dir=["no-such-directory-here"], image_url_host=[]))

    def test_a_directory_is_resolved(self):
        from serve.server import image_sources
        with tempfile.TemporaryDirectory() as d:
            sources = image_sources(SimpleNamespace(image_dir=[d], image_url_host=["images.example"]))
            self.assertEqual(sources.dirs, (Path(d).resolve(),))
            self.assertEqual(sources.url_hosts, ("images.example",))


class ImageHostOptionTest(unittest.TestCase):
    def parse(self, *words):
        return build_parser().parse_args(["--model", "m.basalt", *words])

    def test_host_and_port(self):
        self.assertEqual(self.parse("--image-url-host", "images.example:8443").image_url_host,
                         ["images.example:8443"])

    def test_a_bare_host(self):
        self.assertEqual(self.parse("--image-url-host", "images.example").image_url_host, ["images.example"])

    def test_a_bad_port(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.parse("--image-url-host", "images.example:http")

    def test_a_url_instead_of_a_host(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.parse("--image-url-host", "https://images.example/x")

    def test_repeatable(self):
        args = self.parse("--image-url-host", "a.example", "--image-url-host", "b.example:81",
                          "--image-dir", "images")
        self.assertEqual(args.image_url_host, ["a.example", "b.example:81"])
        self.assertEqual(args.image_dir, ["images"])


if __name__ == "__main__":
    unittest.main()
