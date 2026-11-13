"""The image and video encoder process, and the GPU helpers the engine and encoder environments are built from."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
import queue
from pathlib import Path
from typing import NamedTuple

from serve.engine import lifeline
from serve.engine.image_source import VIDEO, ImageSources, load_image
from serve.engine.process import frame_pipes
from serve.util.console import tagged

# Reads of the encoder's pipe have a timeout: a read that never returns would hold the request turn for good.
# Generous: a cold model load or a slow encode is not a hang. A silent encoder is killed and the next request
# starts a fresh one.
VISION_READY_S = 300.0
VISION_ENCODE_S = 300.0
VISION_CACHE_BYTES = 1 << 30                    # room for about 64 images
IMAGE_MAX_PIXELS = 64 << 20                     # matches the decoder's kMaxImagePixels
IMAGE_REQUEST_BYTES = 1 << 30                   # one request's images, as the encoder will get them
IMAGE_WAITING_BYTES = 2 << 30                   # all requests' read images waiting for their turn to encode
VIDEO_REQUEST_BYTES = 1 << 30                   # one request's videos, as read
VIDEO_MAX_TOKENS = 8192                         # one video's merged tokens over all its groups (--video-max-tokens)
# what Pillow decodes for a client: formats whose header gives the size that decoding will produce, and whose cost is
# in proportion. Out: EPS (runs Ghostscript), ICO (the size checked is not the size decoded), JPEG 2000 (seconds of
# work that stall the server).
PILLOW_FORMATS = ("JPEG", "PNG", "BMP", "GIF", "WEBP", "TIFF", "AVIF", "HEIF", "TGA", "PPM", "QOI")
_decodes = threading.Semaphore(4)               # reads run outside the request turn: this bounds Pillow's memory


def _unlink_quietly(path: Path) -> None:
    """Drop a scratch file; on Windows the encoder can still hold it open, and a request must not fail for that."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def model_plan(engine_bin: str, path: str, value, words: list[str]) -> dict:
    """The engine's choice for the gpu setting (`basalt --gpu-plan`) for a model with its flags: the cards, main first, as
    "devices" (their CUDA_VISIBLE_DEVICES spelling) and "cards", plus the lines to print once. The engine parses the flags
    by the model's own rules first and loads nothing. ValueError with the engine's reason."""
    text = gpu_text(value)
    cmd = [engine_bin, "--model", path, *words, "--gpu-plan"] + (["--gpu", text] if text else [])
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError):
        raise ValueError(f"{engine_bin} could not check this model's flags") from None
    errors = [ln.split("error: ", 1)[1].strip() for ln in run.stderr.splitlines() if "error: " in ln]
    try:
        plan = json.loads(run.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        plan = {}
    if plan.get("error"):
        raise ValueError(plan["error"])
    if run.returncode != 0 or not plan:
        raise ValueError("; ".join(errors) or f"{engine_bin} could not check this model's flags (exit code {run.returncode})")
    return plan


def flag_refused(engine_bin: str, path: str, group: list[str]) -> bool:
    """Whether the model in the file `path` has no such flag; `group` is the flag and its values."""
    try:
        run = subprocess.run([engine_bin, "--model", path, *group, "--gpu-plan"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return f"unknown argument: {group[0]}" in run.stderr


class VideoRows(NamedTuple):
    """An encoded video: `groups` records of `rows` rows each, and each group's time in seconds."""
    groups: int
    rows: int
    times: list


def video_rows(line: str) -> VideoRows:
    """A video reply's `OK <groups> <nx> <ny> <ms> <time,...>` as VideoRows."""
    w = line.split()
    return VideoRows(int(w[1]), int(w[2]) * int(w[3]), [float(t) for t in w[5].split(",")])


class Vision:
    """The resident `basalt-vision` encoder: `ENC <image> <out>` and `ENCV <video> <out>` lines in, embeddings out,
    cached by content hash."""

    def __init__(self, cfg: dict, log=None, env: dict | None = None, start: bool = True):
        """`start` false: down until the service restarts it, as an unloaded encoder is."""
        args = [cfg["vision_bin"], "--model", cfg["model"]] + (["--gpu"] if cfg.get("gpu") else [])
        self.dir = Path(tempfile.mkdtemp(prefix="basalt-vision-"))
        self.spawn = (args, log, env)                   # to start again
        self.stopped = False
        self.ready_s = 0.0                              # last start's time to READY
        self.proc = None
        if start:
            self._start()
        else:
            self.stopped = True
        self.lock = threading.Lock()
        self.cache: dict[str, tuple[Path, int, int]] = {}  # LRU cache: file, image tokens, bytes
        self.cache_bytes = 0
        self.pinned: dict[str, int] = {}                # image -> requests using it
        self.videos: dict[str, VideoRows] = {}          # a cached video's groups, beside its cache entry

    def _start(self):
        args, log, env = self.spawn
        t0 = time.time()
        self.proc = lifeline.popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log or subprocess.DEVNULL,
                                   text=True, encoding="utf-8", bufsize=1, env=env)
        frame_pipes(self.proc, False)
        try:
            line = self._readline(VISION_READY_S, "the vision encoder did not start")
        except RuntimeError:                            # silent: it was ended, and is not left half-started
            self.stopped = True
            raise
        if line.startswith("VIDEO "):                    # whether it reads video, before READY
            self.video = line[6:].strip()
            line = self._readline(VISION_READY_S, "the vision encoder did not start")
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

    video = None                                        # "on <release> ..." or "off <reason>", from the encoder

    def alive(self) -> bool:
        return not self.stopped and self.proc.poll() is None

    def video_state(self):
        """Whether the encoder reads video ("on ..." / "off <reason>"); None before it has said."""
        return self.video

    def read_video(self, source, allowed: ImageSources | None = None) -> bytes:
        """A video's bytes, read under the request's policy (VIDEO_MAX_BYTES at most). Decoding is the encoder's."""
        return load_image(source, allowed, VIDEO)

    def unload(self):
        """Stop the encoder; encoded images stay cached on disk."""
        self.close()
        self.stopped = True

    def restart(self):
        """Start the encoder again, keeping the cache."""
        try:
            if self.proc is not None:
                self.proc.kill()
        except OSError:
            pass
        self._start()

    def read(self, sources: list, allowed: ImageSources | None = None) -> list[bytes]:
        """The request's images, read under its policy and normalized, each source once. Called outside the request
        turn, so a slow source holds up only its own request."""
        seen: dict[str, bytes] = {}
        out, total = [], 0
        for source in sources:
            if not isinstance(source, str) or source not in seen:
                data = self.normalize(load_image(source, allowed))
                total += len(data)
                if total > IMAGE_REQUEST_BYTES:
                    raise ValueError(f"the request's images are larger than {IMAGE_REQUEST_BYTES >> 20} MiB")
                seen[source] = data
            out.append(seen[source])
        return out

    @staticmethod
    def _exif_rotated(data: bytes) -> bool:
        """A JPEG whose EXIF Orientation says it is stored turned (phones do this for portrait photos): the decoder
        ignores the tag. False without Pillow, or when the tag is missing, 1 or unreadable."""
        try:
            import io
            from PIL import Image
            with Image.open(io.BytesIO(data), formats=["JPEG"]) as im:
                return im.getexif().get(0x0112, 1) not in (None, 0, 1)
        except Exception:  # noqa: BLE001 - any failure leaves the image as it was sent
            return False

    @staticmethod
    def normalize(data: bytes) -> bytes:
        """Formats the decoder reads pass through; others are converted to PNG. A JPEG with an EXIF Orientation
        other than 1 is turned upright first (needs Pillow). An image past IMAGE_MAX_PIXELS is refused before its
        pixels are loaded; the decoder enforces the same budget for the formats it reads itself."""
        jpeg = data[:3] == b"\xff\xd8\xff"
        if (jpeg and not Vision._exif_rotated(data)) or data[:8] == b"\x89PNG\r\n\x1a\n" or data[:2] == b"BM" or \
                data[:6] in (b"GIF87a", b"GIF89a"):
            return data
        with _decodes:
            return Vision._converted(data)

    @staticmethod
    def _converted(data: bytes) -> bytes:
        """The image through Pillow, as a PNG."""
        try:
            import io
            from PIL import Image, UnidentifiedImageError
        except ImportError:
            raise ValueError("this image format needs Pillow (python -m pip install pillow); JPEG, PNG, BMP and "
                             "GIF work without it") from None
        try:
            Image.init()                                 # plugin sets differ between builds
            im = Image.open(io.BytesIO(data), formats=[f for f in PILLOW_FORMATS if f in Image.OPEN])
            width, height = im.size
            if width * height > IMAGE_MAX_PIXELS:
                raise ValueError(f"the image is {width}x{height} pixels; the limit is "
                                 f"{IMAGE_MAX_PIXELS >> 20} megapixels")
            im.load()
            from PIL import ImageOps
            im = ImageOps.exif_transpose(im)             # the camera's orientation tag, applied
        except ValueError:
            raise                                        # the pixel budget's own message
        except Image.DecompressionBombError:
            raise ValueError(f"the image is larger than {IMAGE_MAX_PIXELS >> 20} megapixels") from None
        except UnidentifiedImageError:
            raise ValueError("the image could not be read: not a format this server decodes") from None
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

    def encode(self, images: list[bytes]) -> tuple[Path, list[int]]:
        """Encode a request's images (from read) into one embeddings file; return it and each image's token count."""
        pins = []
        try:
            encoded = self._encode_pinned(images, pins)
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

    def encode_media(self, jobs: list) -> tuple[Path, list]:
        """Encode a request's images and videos, (kind, bytes, video options) in prompt order, into one embeddings
        file; return it and per item an image's token count or a video's VideoRows."""
        pins = []
        try:
            images = self._encode_pinned([data for kind, data, _ in jobs if kind == "image"], pins)
            parts, k = [], 0
            for kind, data, opts in jobs:
                if kind == "image":
                    parts.append((images[k][0], images[k][1]))
                    k += 1
                else:
                    parts.append(self._encode_video(data, opts or {}, pins))
            combined = self.dir / f"req-{uuid.uuid4().hex[:12]}.sve"
            try:
                with open(combined, "wb") as f:
                    for path, _ in parts:
                        f.write(path.read_bytes())
            except BaseException:
                combined.unlink(missing_ok=True)
                raise
            return combined, [r for _, r in parts]
        finally:
            with self.lock:
                for k in pins:
                    self.pinned[k] -= 1
                    if not self.pinned[k]:
                        del self.pinned[k]
                self._evict()

    def _encode_video(self, data: bytes, opts: dict, pins: list[str]) -> tuple[Path, VideoRows]:
        """One video through ENCV, cached by its bytes and the sampling asked for."""
        words = " ".join(f"{k}={opts[k]!r}" if isinstance(opts[k], float) else f"{k}={opts[k]}" for k in sorted(opts))
        key = hashlib.sha256(data + b"\0" + words.encode()).hexdigest()[:32]
        with self.lock:
            self._pin(key, pins)
            if key in self.cache and key in self.videos:
                return self.cache[key][0], self.videos[key]
            tag = uuid.uuid4().hex[:8]                                 # per attempt, as for an image
            vid, part, out = self.dir / f"{key}-{tag}.vid", self.dir / f"{key}-{tag}.part", self.dir / f"{key}.sve"
            vid.write_bytes(data)
            try:
                line = self._encv(vid, part, words)
            finally:
                _unlink_quietly(vid)
            if not line.startswith("OK"):
                _unlink_quietly(part)
                raise ValueError("the video could not be read: " + (line[4:] if line.startswith("ERR") else
                                                                     "the vision encoder stopped"))
            os.replace(part, out)                                      # published whole
            rows = video_rows(line)
            self.videos[key] = rows
            return self._remember(key, out, rows.groups * rows.rows)[0], rows

    def _encv(self, video: Path, out: Path, words: str) -> str:
        """One ENCV round trip with the sidecar: its `OK <groups> ...` or `ERR ...` line."""
        self.proc.stdin.write(f"ENCV {video} {out} {words}\n".rstrip() + "\n")
        self.proc.stdin.flush()
        try:
            return self._readline(VISION_ENCODE_S, "the video could not be read").strip()
        except RuntimeError as e:
            self.stopped = True
            raise ValueError(str(e)) from None

    def _pin(self, key: str, pins: list[str]) -> None:
        self.pinned[key] = self.pinned.get(key, 0) + 1
        pins.append(key)
        if key in self.cache:
            self.cache[key] = self.cache.pop(key)

    def _encode_pinned(self, images: list[bytes], pins: list[str]) -> list[tuple[Path, int, int]]:
        return [self._encode_one(data, pins) for data in images]

    def _encode_one(self, data: bytes, pins: list[str]) -> tuple[Path, int, int]:
        key = hashlib.sha256(data).hexdigest()[:32]
        with self.lock:
            self._pin(key, pins)
            if key in self.cache:
                return self.cache[key]
            # the image and the encoder's answer are written under names of this attempt: the encoder holds the image
            # while it reads it, so a second attempt at the same image must not write under that open handle
            tag = uuid.uuid4().hex[:8]
            img, part, out = self.dir / f"{key}-{tag}.img", self.dir / f"{key}-{tag}.part", self.dir / f"{key}.sve"
            img.write_bytes(data)
            try:
                line = self._enc(img, part)
            finally:                                                   # also when the pipe is gone
                _unlink_quietly(img)
            if not line.startswith("OK"):
                _unlink_quietly(part)
                raise ValueError("the image could not be read: " + (line[4:] if line.startswith("ERR") else
                                                                     "the vision encoder stopped"))
            os.replace(part, out)                                      # published whole: a reader never sees half of it
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
            self.videos.pop(k, None)

    def close(self):
        if self.proc is None:
            return
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
        self.videos: dict[str, VideoRows] = {}

    def attach(self, engine) -> None:
        self.engine = engine

    def video_state(self):
        """Whether the engine's encoder reads video, from its start lines; None before the engine has run."""
        return getattr(self.engine, "video", None)

    def _encv(self, video: Path, out: Path, words: str) -> str:
        """ENCV over the engine's own pipe: the engine borrows its slots once for the video's groups."""
        return self.engine.encode_video(video, out, words)

    def alive(self) -> bool:                            # the engine's life is the engine's business
        return True

    def unload(self):
        """The engine unloads itself; encoded images stay cached on disk."""

    def restart(self):
        pass

    def close(self):
        pass

    def _encode_pinned(self, images: list[bytes], pins: list[str]) -> list[tuple[Path, int, int]]:
        """Every image not cached yet goes to the engine in one ENC, so it borrows its slots once."""
        keys, datas = [], {}
        for data in images:
            key = hashlib.sha256(data).hexdigest()[:32]
            keys.append(key)
            datas[key] = data
        with self.lock:
            for k in keys:
                self._pin(k, pins)
            todo = [k for k in dict.fromkeys(keys) if k not in self.cache]
            # per attempt, as on the sidecar path: the engine holds an image while it reads it
            tag = uuid.uuid4().hex[:8]
            jobs = [(self.dir / f"{k}-{tag}.img", self.dir / f"{k}-{tag}.part") for k in todo]
            try:
                for (img, _), k in zip(jobs, todo):
                    img.write_bytes(datas[k])
                if jobs:
                    lines = self.engine.encode_images(jobs)
                    for k, (_, part), line in zip(todo, jobs, lines):
                        n = int(line.split()[1])              # an ERR line raises here, as it did before
                        out = self.dir / f"{k}.sve"
                        os.replace(part, out)                 # published whole: a reader never sees half of it
                        self._remember(k, out, n)
            finally:
                for img, part in jobs:
                    _unlink_quietly(img)
                    _unlink_quietly(part)
            return [self.cache[k] for k in keys]


def gpu_text(value) -> str:
    """The gpu setting as the engine reads it: "" (unset), "auto" or "A,B,..."; a config may give a number or a list."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(x).strip() for x in value)
    return str(value).strip()


def check_gpu(value) -> None:
    """ValueError unless the gpu setting reads as auto or distinct GPU numbers."""
    text = gpu_text(value)
    if text in ("", "auto"):
        return
    items = [x.strip() for x in text.split(",")]
    if not all(x.isdigit() for x in items) or len(set(int(x) for x in items)) != len(items):
        raise ValueError(text)


def gpu_list(cfg: dict) -> list:
    """The cards the engine runs on, main first, as CUDA_VISIBLE_DEVICES names them (nvidia-smi numbers, or a UUID
    where the environment gave one); [] when none was chosen (the mock engine)."""
    g = gpu_text(cfg.get("gpu"))
    if g in ("", "auto"):
        return []
    return [int(x) if x.isdigit() else x for x in (w.strip() for w in g.split(",")) if x]


def gpus_in_play(cfg: dict, args: list[str] | None = None) -> list[int]:
    """The nvidia-smi numbers of the GPUs the engine uses, its main card first; [0] if none."""
    smi = [i for i in cfg.get("gpu_smi") or [] if isinstance(i, int) and i >= 0]
    if smi:
        return smi
    return [i for i in gpu_list(cfg) if isinstance(i, int)] or [0]


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


VISION_MODES = ("off", "dynamic", "cpu")
VISION_USAGE = "vision: use one of " + ", ".join(VISION_MODES) + " or a GPU number, as nvidia-smi numbers them"


def vision_mode(value) -> str:
    """The --vision setting: one of VISION_MODES (None: "off") or a GPU number as a string (the sidecar's own
    card); booleans and anything else are a ValueError."""
    if value is None:
        return "off"
    text = str(value).strip().lower()
    if text in VISION_MODES:
        return text
    if text.isascii() and text.isdigit():                 # odd Unicode digits are the error, not int()'s guess
        return str(int(text))
    raise ValueError(f"{VISION_USAGE} (got {value!r})")


def vision_card(mode) -> int | None:
    """The GPU number a vision mode names (the sidecar's own card); None for off, dynamic and cpu."""
    text = str(mode).strip()
    return int(text) if text.isascii() and text.isdigit() else None


def vision_env(env: dict, mode) -> dict:
    """The encoder process's environment: a GPU-numbered vision mode runs the sidecar on that card alone; the
    other modes inherit the engine's environment."""
    card = vision_card(mode)
    if card is None:
        return env
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


def vision_item(vision: dict | None, cfg: dict) -> str | None:
    """The startup item for image input: where the encoder runs; None when it is off."""
    if not vision:
        return None
    gpu = vision_card(vision.get("mode") or "cpu")
    if gpu is not None:
        return f"vision=gpu{gpu}"
    if vision.get("mode") == "dynamic":
        listed = gpu_list(cfg)
        return f"vision=gpu{listed[0] if listed else 0}"
    return "vision=cpu"