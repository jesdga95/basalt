"""The image encoder process, and the GPU helpers the engine and encoder environments are built from."""
from __future__ import annotations

import base64
import hashlib
import os
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
import queue
from pathlib import Path

from serve.engine.flags import flag_value
from serve.engine.process import frame_pipes

# Reads of the encoder's pipe have a timeout: a read that never returns would hold the request turn for good.
# Generous: a cold model load or a slow encode is not a hang. A silent encoder is killed and the next request
# starts a fresh one.
VISION_READY_S = 300.0
VISION_ENCODE_S = 300.0
VISION_CACHE_BYTES = 1 << 30                    # room for about 64 images


class Vision:
    """The resident `basalt-vision` encoder: `ENC <image> <out>` lines in, embeddings out, cached by image hash."""

    def __init__(self, cfg: dict, log=None, env: dict | None = None):
        args = [cfg["vision_bin"], "--mmproj", cfg["mmproj"], "--model", cfg["model"]] + (["--gpu"] if cfg.get("gpu") else [])
        self.dir = Path(tempfile.mkdtemp(prefix="basalt-vision-"))
        self.spawn = (args, log, env)                   # to start again
        self.stopped = False
        self.ready_s = 0.0                              # last start's time to READY
        self._start()
        self.lock = threading.Lock()
        self.cache: dict[str, tuple[Path, int, int]] = {}  # LRU cache: file, image tokens, bytes
        self.cache_bytes = 0
        self.pinned: dict[str, int] = {}                # image -> requests using it

    def _start(self):
        args, log, env = self.spawn
        t0 = time.time()
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log or subprocess.DEVNULL,
                                     text=True, encoding="utf-8", bufsize=1, env=env)
        frame_pipes(self.proc, False)
        try:
            line = self._readline(VISION_READY_S, "the vision encoder did not start")
        except RuntimeError:                            # silent: it was ended, and is not left half-started
            self.stopped = True
            raise
        if not line.startswith("READY"):
            raise RuntimeError("the vision encoder did not start: " + line.strip())
        self.ready_s = time.time() - t0
        self.stopped = False

    def _readline(self, timeout: float, what: str) -> str:
        """One line from the encoder, waiting at most `timeout` s. On a timeout the encoder is killed, so the next
        request starts a fresh one, and the read raises RuntimeError instead of holding the request turn."""
        proc = self.proc
        got: queue.Queue = queue.Queue()

        def read():
            try:
                got.put(proc.stdout.readline())
            except (OSError, ValueError):               # killed under the read
                got.put("")

        threading.Thread(target=read, daemon=True).start()
        try:
            return got.get(timeout=timeout)
        except queue.Empty:
            try:
                proc.kill()
            except OSError:
                pass
            raise RuntimeError(f"{what}: it said nothing for {timeout:.0f} s") from None

    def alive(self) -> bool:
        return not self.stopped and self.proc.poll() is None

    def unload(self):
        """Stop the encoder; encoded images stay cached on disk."""
        self.close()
        self.stopped = True

    def restart(self):
        """Start the encoder again, keeping the cache."""
        try:
            self.proc.kill()
        except OSError:
            pass
        self._start()

    @staticmethod
    def load(source: str) -> bytes:
        if source.startswith("data:"):
            return base64.b64decode(source.split(",", 1)[1])
        if source.startswith(("http://", "https://")):
            req = urllib.request.Request(source, headers={"User-Agent": "local-server"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        path = source[7:] if source.startswith("file://") else source
        if path and os.path.isfile(path):
            return Path(path).read_bytes()
        raise ValueError("an image must be a data: URL, an http(s) URL or a local file path")

    @staticmethod
    def _exif_rotated(data: bytes) -> bool:
        """A JPEG whose EXIF Orientation says it is stored turned (phones do this for portrait photos): the decoder
        ignores the tag. False without Pillow, or when the tag is missing, 1 or unreadable."""
        try:
            import io
            from PIL import Image
            with Image.open(io.BytesIO(data)) as im:
                return im.getexif().get(0x0112, 1) not in (None, 0, 1)
        except Exception:  # noqa: BLE001 - any failure leaves the image as it was sent
            return False

    @staticmethod
    def normalize(data: bytes) -> bytes:
        """Formats the decoder reads pass through; others are converted to PNG. A JPEG with an EXIF Orientation
        other than 1 is turned upright first (needs Pillow)."""
        jpeg = data[:3] == b"\xff\xd8\xff"
        if (jpeg and not Vision._exif_rotated(data)) or data[:8] == b"\x89PNG\r\n\x1a\n" or data[:2] == b"BM" or \
                data[:6] in (b"GIF87a", b"GIF89a"):
            return data
        try:
            import io
            from PIL import Image
        except ImportError:
            raise ValueError("this image format needs Pillow (python -m pip install pillow); JPEG, PNG, BMP and "
                             "GIF work without it") from None
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
            from PIL import ImageOps
            im = ImageOps.exif_transpose(im)             # the camera's orientation tag, applied
        except Exception as e:
            raise ValueError(f"the image could not be read ({e})") from None
        if im.mode in ("RGBA", "LA", "P") and "transparency" in im.info or im.mode in ("RGBA", "LA"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))   # transparency becomes white
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")
        out = io.BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()

    def encode(self, sources: list[str]) -> tuple[Path, list[int]]:
        """Encode a request's images into one embeddings file; return it and each image's token count."""
        pins = []
        try:
            encoded = self._encode_pinned(sources, pins)
            combined = self.dir / f"req-{uuid.uuid4().hex[:12]}.sve"
            try:
                with open(combined, "wb") as f:
                    for path, _, _ in encoded:
                        f.write(path.read_bytes())
            except BaseException:                       # no partial file is left behind
                combined.unlink(missing_ok=True)
                raise
            return combined, [n for _, n, _ in encoded]
        finally:
            with self.lock:
                for k in pins:
                    self.pinned[k] -= 1
                    if not self.pinned[k]:
                        del self.pinned[k]
                self._evict()

    def _pin(self, key: str, pins: list[str]) -> None:
        self.pinned[key] = self.pinned.get(key, 0) + 1
        pins.append(key)
        if key in self.cache:
            self.cache[key] = self.cache.pop(key)

    def _encode_pinned(self, sources: list[str], pins: list[str]) -> list[tuple[Path, int, int]]:
        return [self._encode_one(src, pins) for src in sources]

    def _encode_one(self, source: str, pins: list[str]) -> tuple[Path, int, int]:
        data = self.normalize(self.load(source))
        key = hashlib.sha256(data).hexdigest()[:32]
        with self.lock:
            self._pin(key, pins)
            if key in self.cache:
                return self.cache[key]
            img, out = self.dir / f"{key}.img", self.dir / f"{key}.sve"
            img.write_bytes(data)
            try:
                line = self._enc(img, out)
            finally:                                                   # also when the pipe is gone
                img.unlink(missing_ok=True)
            if not line.startswith("OK"):
                raise ValueError("the image could not be read: " + (line[4:] if line.startswith("ERR") else
                                                                     "the vision encoder stopped"))
            return self._remember(key, out, int(line.split()[1]))

    def _enc(self, img: Path, out: Path) -> str:
        """One ENC round trip with the sidecar: its `OK <n> ...` or `ERR ...` line."""
        self.proc.stdin.write(f"ENC {img} {out}\n")
        self.proc.stdin.flush()
        try:
            return self._readline(VISION_ENCODE_S, "the image could not be read").strip()
        except RuntimeError as e:                       # silent: the encoder was ended
            self.stopped = True                         # alive() is False, so the next request starts it again
            raise ValueError(str(e)) from None

    def _remember(self, key: str, out: Path, n: int) -> tuple[Path, int, int]:
        size = out.stat().st_size
        self.cache[key] = (out, n, size)
        self.cache_bytes += size
        self._evict()
        return self.cache[key]

    def _evict(self) -> None:
        for k in [k for k in self.cache if k not in self.pinned]:
            if self.cache_bytes <= VISION_CACHE_BYTES:
                break
            path, _, size = self.cache.pop(k)
            path.unlink(missing_ok=True)
            self.cache_bytes -= size

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


class EngineVision(Vision):
    """The encoder inside the engine process (`--vision dynamic`): `ENC` lines go over the engine's own pipe, the engine
    borrows expert-cache slots for the encode and gives them back. Same cache, same files as the sidecar's."""
    in_engine = True

    def __init__(self, cfg: dict):
        self.dir = Path(tempfile.mkdtemp(prefix="basalt-vision-"))
        self.engine = None                              # set by attach() once the engine exists
        self.stopped = False
        self.ready_s = 0.0
        self.lock = threading.Lock()
        self.cache: dict[str, tuple[Path, int, int]] = {}
        self.cache_bytes = 0
        self.pinned: dict[str, int] = {}

    def attach(self, engine) -> None:
        self.engine = engine

    def alive(self) -> bool:                            # the engine's life is the engine's business
        return True

    def unload(self):
        """The engine unloads itself; encoded images stay cached on disk."""

    def restart(self):
        pass

    def close(self):
        pass

    def _encode_pinned(self, sources: list[str], pins: list[str]) -> list[tuple[Path, int, int]]:
        """Every image not cached yet goes to the engine in one ENC, so it borrows its slots once."""
        keys, datas = [], {}
        for src in sources:
            data = self.normalize(self.load(src))
            key = hashlib.sha256(data).hexdigest()[:32]
            keys.append(key)
            datas[key] = data
        with self.lock:
            for k in keys:
                self._pin(k, pins)
            todo = [k for k in dict.fromkeys(keys) if k not in self.cache]
            jobs = [(self.dir / f"{k}.img", self.dir / f"{k}.sve") for k in todo]
            try:
                for (img, _), k in zip(jobs, todo):
                    img.write_bytes(datas[k])
                if jobs:
                    lines = self.engine.encode_images(jobs)
                    for k, (_, out), line in zip(todo, jobs, lines):
                        self._remember(k, out, int(line.split()[1]))
            finally:
                for img, _ in jobs:
                    img.unlink(missing_ok=True)
            return [self.cache[k] for k in keys]


def gpu_list(cfg: dict) -> list[int]:
    """The nvidia-smi numbers of the cards --gpu lists (first: model, second: peer tier); [] when none."""
    g = cfg.get("gpu")
    if g is None or g == "":
        return []
    items = g if isinstance(g, (list, tuple)) else str(g).split(",")
    return [int(str(x).strip()) for x in items if str(x).strip() != ""]


def gpus_in_play(cfg: dict, args: list[str]) -> list[int]:
    """The nvidia-smi numbers of the GPUs the engine uses, its own card first, then the --peer-device card; [0] if none."""
    visible = gpu_list(cfg)
    out = visible[:1] or [0]
    peer = next((a.split("=", 1)[1] for a in reversed(args) if a.startswith("--peer-device=")),
                flag_value(args, "--peer-device"))
    if peer is not None and str(peer).strip().lstrip("-").isdigit() and int(peer) >= 0:
        ordinal = int(peer)
        out = out + [visible[ordinal] if ordinal < len(visible) else ordinal]
    return list(dict.fromkeys(out))


def engine_args(cfg: dict) -> list[str]:
    """The engine arguments the model file carries."""
    return list(cfg["args"])


def child_env(cfg: dict) -> dict:
    """The engine environment: only the --gpu cards are visible."""
    env = dict(os.environ)
    if gpu_list(cfg):                              # number cards as nvidia-smi does
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"      # CUDA's default order differs
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_list(cfg))
    return env


VISION_MODES = ("on", "off", "peer", "dynamic", "cpu")
VISION_USAGE = "vision: use one of " + ", ".join(VISION_MODES)


def vision_mode(value) -> str:
    """The --vision setting as one of VISION_MODES (None: "off"); booleans and anything else are a ValueError."""
    if value is None:
        return "off"
    text = str(value).strip().lower()
    if isinstance(value, bool) or text not in VISION_MODES:
        raise ValueError(f"{VISION_USAGE} (got {value!r})")
    return text


def peer_card(cfg: dict) -> int | None:
    """The card the peer tier runs on (the second of --gpu); None without a second card."""
    gpus = gpu_list(cfg)
    if len(gpus) >= 2:
        return gpus[1]
    if not gpus:
        from serve.engine.modelfile import gpu_count
        if gpu_count() > 1:
            return 1
    return None


def vision_env(cfg: dict, env: dict, mode: str = "peer") -> dict:
    """The encoder process's environment: `peer` puts it on the second card alone (the engine's when there is none),
    `on` on the model's own card alone."""
    gpus = gpu_list(cfg)
    if mode == "on":
        if len(gpus) < 2:
            return env
        card = gpus[0]
    else:
        if len(gpus) < 2:
            if gpus or peer_card(cfg) is None:
                return env
            env = dict(env)
            env["CUDA_VISIBLE_DEVICES"] = "1"
            return env
        card = gpus[1]
    env = dict(env)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(card)
    return env


def gpu_card_name(index: int) -> str | None:
    """A card's name from one nvidia-smi query; None if unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(index), "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        name = out.stdout.strip().splitlines()[0].strip() if out.returncode == 0 and out.stdout.strip() else ""
    except (OSError, subprocess.SubprocessError):
        return None
    return name.removeprefix("NVIDIA GeForce ").removeprefix("NVIDIA ") or None


def vision_line(vision: dict | None, cfg: dict, ready_s: float = 0.0, name=gpu_card_name) -> str:
    """The startup line about the encoder: off, or where it runs and its ready time."""
    if not vision:
        return "[server] vision mode: off"
    took = f"ready in {ready_s:.1f} s"
    mode = vision.get("mode") or ("peer" if vision.get("gpu") else "cpu")
    listed = gpu_list(cfg)
    main = listed[0] if listed else 0

    def card(i):
        n = name(i)
        return f"GPU {i}{f' ({n})' if n else ''}"
    if mode == "cpu" or not vision.get("gpu"):
        return f"[server] vision mode: cpu, {max(1, (os.cpu_count() or 2) // 2)} threads, {took}"
    if mode == "dynamic":
        return f"[server] vision mode: dynamic, {card(main)}, loads per image"
    if mode == "on":
        return f"[server] vision mode: on, {card(main)}, {took}"
    other = peer_card(cfg)
    return f"[server] vision mode: peer, {card(other if other is not None else main)}, {took}"
