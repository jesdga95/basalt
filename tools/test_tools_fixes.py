"""Unit tests for community_calibrate, serve_soak, greedy_identity, ppl_gate, check_isa_isolation, and CMake invariants."""
import argparse
import contextlib
import io
import json
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

import check_benchmarks as cb  # noqa: E402
import check_isa_isolation as isa  # noqa: E402
import community_calibrate as cal  # noqa: E402
import community_gate as gate  # noqa: E402
import greedy_identity as gi  # noqa: E402
import model_file as mf  # noqa: E402
import ppl_gate as ppl  # noqa: E402
import serve_soak as soak  # noqa: E402


class TestCommunityCalibrate(unittest.TestCase):
    def test_pr_filter_and_status_normalization(self):
        self.assertTrue(cal.is_community_pr([{"path": "docs/community/runs/m/0123456789ab.json"}]))
        self.assertTrue(cal.is_community_pr([{"path": gate.LEGACY_FILE}]))
        self.assertFalse(cal.is_community_pr([{"path": "README.md"}]))
        changed = cal.normalize_changed([
            {"path": "a", "changeType": "ADDED"},
            {"path": "b", "changeType": "MODIFIED"},
            {"path": "c", "changeType": "DELETED"},
            {"path": "d", "changeType": "RENAMED"},
        ])
        self.assertEqual(changed, [("a", "added"), ("b", "modified"), ("c", "removed"), ("d", "renamed")])

    def test_base_tree_and_head_files(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
            (repo / "README.md").write_text("hi\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "empty"], cwd=repo, check=True)
            sha0 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
            self.assertEqual(cal.base_tree(sha0, str(repo)), {})

            run_rel = "docs/community/runs/hy4-flash/0123456789ab.json"
            (repo / run_rel).parent.mkdir(parents=True)
            (repo / run_rel).write_text('{"ok": 1}\n', encoding="utf-8")
            subprocess.run(["git", "add", run_rel], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "add run"], cwd=repo, check=True)
            sha1 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
            self.assertEqual(cal.base_tree(sha1, str(repo)), {run_rel: '{"ok": 1}\n'})

        fetched = []
        changed = [
            ("docs/community/runs/hy4-flash/0123456789ab.json", "added"),
            ("docs/community/runs/hy4-flash/fedcba987654.json", "removed"),
            ("README.md", "modified"),
        ]
        got = cal.head_files_at("abc123", changed, fetch_fn=lambda p, s: fetched.append((p, s)) or "{}")
        self.assertEqual(got, {"docs/community/runs/hy4-flash/0123456789ab.json": "{}"})
        self.assertEqual(fetched, [("docs/community/runs/hy4-flash/0123456789ab.json", "abc123")])

    def test_main_uses_current_run_gate_signature(self):
        base = gate.read_tree(ROOT / "docs" / "community", "runs")
        base = {"docs/community/" + k: v for k, v in base.items()}
        sample = json.loads(next(v for v in base.values() if json.loads(v)["model"] == "Qwen3.8-Flash-Next"
                                 and json.loads(v)["gpus"][0]["name"] == "RTX 5090" and len(json.loads(v)["gpus"]) == 1))
        sample["submitter"] = "caluser"
        sample["commit"] = "0123456789"
        sample["gpus"][0]["power_w"] = 455
        rel = gate.RUNS_PREFIX + cb.run_path(sample)
        text = cb.canonical_row(sample)

        def fake_gh(*args):
            if args[:2] == ("pr", "list"):
                return json.dumps([{"number": 42, "title": "community run", "state": "OPEN", "author": {"login": "caluser"}}])
            if args[:2] == ("pr", "view"):
                return json.dumps({"files": [{"path": rel, "changeType": "ADDED"}], "baseRefOid": "b1", "headRefOid": "h1"})
            if args[0] == "api" and "compare/" in args[1]:
                return json.dumps({"m": "mb1"})
            if args[0] == "api" and "contents/" in args[1]:
                return text
            raise AssertionError(f"unexpected gh call: {args}")

        out = io.StringIO()
        with mock.patch.object(cal, "gh", side_effect=fake_gh), \
                mock.patch.object(cal, "base_tree", return_value=base), \
                mock.patch.object(gate, "git_known_commit", return_value=lambda s: True), \
                contextlib.redirect_stdout(out):
            rc = cal.main(["community_calibrate.py", "--limit", "1"])
        self.assertEqual(rc, 0)
        self.assertIn("   42 open    caluser            pass", out.getvalue())


class TestServeSoak(unittest.TestCase):
    def _start_server(self, hang_chat=False):
        unblock = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/v1/models":
                    body = json.dumps({"data": [{"id": "test-model"}]}).encode()
                elif self.path == "/slots":
                    body = json.dumps([{"n_ctx": 8192, "is_processing": False}]).encode()
                else:
                    body = b"{}"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0"))
                _ = self.rfile.read(n)
                if self.path == "/v1/messages/count_tokens":
                    body = json.dumps({"input_tokens": 100}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if hang_chat:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    unblock.wait(5)
                    return
                if self.path == "/v1/chat/completions":
                    payload = (
                        'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                        'data: {"usage":{"completion_tokens":1},"choices":[]}\n\n'
                        'data: [DONE]\n\n'
                    ).encode()
                else:
                    payload = (
                        'data: {"type":"content_block_delta","delta":{"text":"ok"}}\n\n'
                        'data: {"type":"message_delta","usage":{"output_tokens":1}}\n\n'
                    ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        return srv, unblock, f"http://127.0.0.1:{srv.server_port}"

    def test_is_bad_flags_hung_and_unsuccessful_requests(self):
        q = soak.Req("openai", "chat")
        self.assertTrue(soak.is_bad(q))  # not done -> bad
        q.done = True
        self.assertTrue(soak.is_bad(q))  # status None != 200 -> bad
        q.status = 200
        self.assertFalse(soak.is_bad(q))
        q.err = "stream error"
        self.assertTrue(soak.is_bad(q))
        q.cancelled = True
        self.assertTrue(soak.is_bad(q))  # cancelled with error -> still bad
        q.err = None
        self.assertFalse(soak.is_bad(q))  # cleanly cancelled -> ok
        q400 = soak.Req("openai", "over")
        q400.expect = 400
        q400.status = 400
        q400.err = "too long"
        q400.done = True
        self.assertFalse(soak.is_bad(q400))

    def test_run_topics_returns_nonzero_on_hang_and_zero_on_ok(self):
        srv_ok, _, base_ok = self._start_server(hang_chat=False)
        try:
            a = argparse.Namespace(base=base_ok, secs=0.2, streams=1, max_tokens=16, bin=10)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(soak.run_topics(a, join_timeout=2), 0)
        finally:
            soak._SERVED.pop(base_ok, None)
            srv_ok.shutdown()
            srv_ok.server_close()

        srv_hang, unblock, base_hang = self._start_server(hang_chat=True)
        try:
            a = argparse.Namespace(base=base_hang, secs=0.1, streams=1, max_tokens=16, bin=10)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(soak.run_topics(a, join_timeout=0.2), 1)
        finally:
            unblock.set()
            soak._SERVED.pop(base_hang, None)
            srv_hang.shutdown()
            srv_hang.server_close()

    def test_run_soak_returns_nonzero_on_hang_and_closes_log(self):
        srv_hang, unblock, base_hang = self._start_server(hang_chat=True)
        try:
            with tempfile.TemporaryDirectory() as d:
                out_log = Path(d) / "soak.jsonl"
                a = argparse.Namespace(base=base_hang, secs=0.25, pgid=0, out=str(out_log),
                                       images=False, seed=1, sample_s=1, stuck_s=90)
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = soak.run_soak(a, grace_s=0.1, post_stop_s=0.05, final_sleep_s=0)
                self.assertEqual(rc, 1)
                self.assertTrue(out_log.is_file())
                out_log.unlink()  # fails on Windows if log handle leaked
        finally:
            unblock.set()
            soak._SERVED.pop(base_hang, None)
            srv_hang.shutdown()
            srv_hang.server_close()


class TestGreedyIdentity(unittest.TestCase):
    def _make_pack(self, path: Path, arch: str) -> None:
        w = mf.ModelWriter(str(path))
        w.add_bytes("meta", json.dumps({"version": 2, "gguf_kv": {"general.architecture": arch}}).encode())
        w.close()

    def test_model_aware_flags(self):
        with tempfile.TemporaryDirectory() as d:
            for arch, expect_pinned in (("qwen4exp", True), ("deepseek41", False), ("hy_v4_flash", False),
                                        ("inkling2", False), ("qwen35", False)):
                p = Path(d) / f"{arch}.basalt"
                self._make_pack(p, arch)
                args = gi.build_args(str(p), [])
                if expect_pinned:
                    self.assertIn("--adapt-swaps", args, arch)
                    self.assertIn("--peer-adapt-swaps", args, arch)
                else:
                    self.assertNotIn("--adapt-swaps", args, arch)
                    self.assertNotIn("--peer-adapt-swaps", args, arch)
            # Unknown/unreadable path omits qwen4exp-only flags and still honors explicit extra overrides.
            other = gi.build_args(str(Path(d) / "missing.basalt"), ["--max-context", "4096"])
            self.assertNotIn("--adapt-swaps", other)
            self.assertEqual(other[other.index("--max-context") + 1], "4096")


class TestPplGate(unittest.TestCase):
    @staticmethod
    def _row(chunk, pos, nll, nllb, kld, same=1, top="10", dlp=0.01):
        return f"{chunk}\t{pos}\t1\t{nll}\t{nllb}\t{kld}\t{same}\t{top}\t{top}\t{dlp}\n"

    def test_summary_and_diff_do_not_mask_nan(self):
        with tempfile.TemporaryDirectory() as d:
            p1 = Path(d) / "a.tsv"
            p2 = Path(d) / "b.tsv"
            p1.write_text(self._row(0, 0, 1.0, 1.0, 0.01) + self._row(0, 1, 1.2, 1.2, "nan") + self._row(0, 2, 1.1, 1.1, 0.02))
            p2.write_text(self._row(0, 0, 1.0, 1.0, 0.01) + self._row(0, 1, 1.2, 1.2, 0.02) + self._row(0, 2, 1.1, 1.1, 0.02))
            rep = ppl.summary(str(p1))
            self.assertIn("KLD mean nan  median nan  p99 nan  max nan", rep)
            self.assertIn("WARNING: 1 row(s) contain non-finite values", rep)

            # Row 0 is finite and identical (0.0); row 1 has NaN so max |dKLD| must be nan, not 0.000000.
            df = ppl.diff(str(p1), str(p2))
            self.assertIn("max |dKLD| nan", df)
            self.assertIn("first difference chunk 0 pos 1", df)

            # Two matching NaN rows must not be counted as identical either.
            df_both = ppl.diff(str(p1), str(p1))
            self.assertIn("identical 2 (66.67%)", df_both)
            self.assertIn("max |dKLD| nan", df_both)

            # NaN in NLL must propagate to se (never masked to 0.0 by max(0.0, nan)).
            p3 = Path(d) / "nll_nan.tsv"
            p3.write_text(self._row(0, 0, 1.0, 1.0, 0.01) + self._row(0, 1, "nan", 1.0, 0.01))
            rep3 = ppl.summary(str(p3))
            self.assertIn("PPL nan +/- nan", rep3)

    def test_run_early_err_exits_promptly_and_closes_log(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            base = tmp / "base.kld"
            # _logits_ + (n_ctx=4, n_vocab=16, n_chunk=1) + 4 int32 tokens
            base.write_bytes(b"_logits_" + struct.pack("<Iii4i", 4, 16, 1, 1, 2, 3, 4))
            fake_engine = tmp / "fake_engine.py"
            fake_engine.write_text(
                "import sys\n"
                "print('READY', flush=True)\n"
                "for line in sys.stdin:\n"
                "    if line.startswith('GEN '):\n"
                "        print('ERR synthetic failure', flush=True)\n"
                "    elif line.startswith('QUIT'):\n"
                "        break\n",
                encoding="utf-8",
            )
            out_dir = tmp / "out"
            a = argparse.Namespace(
                model=str(tmp / "m.basalt"),
                architecture="other",
                engine_bin=sys.executable,
                base=str(base),
                out=str(out_dir),
                chunks=1,
                t=2,
                progress=0,
                engine_args=[],
            )
            real_popen = subprocess.Popen

            def wrap_popen(cmd, **kw):
                return real_popen([sys.executable, str(fake_engine)], **kw)

            t0 = time.monotonic()
            with mock.patch.object(ppl, "gpu_count", return_value=1), \
                    mock.patch.object(subprocess, "Popen", side_effect=wrap_popen), \
                    contextlib.redirect_stdout(io.StringIO()):
                rc = ppl.run(a)
            elapsed = time.monotonic() - t0
            self.assertEqual(rc, 1)
            self.assertLess(elapsed, 5.0)
            (out_dir / "engine.log").unlink()  # fails on Windows if log handle leaked


class TestCheckIsaIsolation(unittest.TestCase):
    def test_dedicated_avx512_matches_msvc_and_ninja_paths(self):
        for ok in (
            Path("build/CMakeFiles/kernels_cpu.dir/src/kernels/cpu/expert.cpp.o"),
            Path("build/CMakeFiles/kernels_cpu.dir/src/kernels/cpu/expert.cpp.obj"),
            Path("build/kernels_cpu.dir/Release/expert.obj"),
            Path("build/kernels_cpu.dir/Debug/expert.obj"),
            Path("build/kernels_cpu.dir/Release/iq_avx512.obj"),
            Path("build/ggml_dots_avx512.dir/Release/ggml_cpu_dots.obj"),
            Path("build/tools/vision/vision_core.dir/Release/gemm_cpu_avx512.obj"),
        ):
            self.assertTrue(isa.is_dedicated_avx512(ok), str(ok))
        for bad in (
            Path("build/kernels_cpu.dir/Release/native_expert.obj"),
            Path("build/model_qwen4exp_prefill.dir/Release/shared_expert.obj"),
            Path("build/other.dir/Release/expert.obj"),
            Path("build/CMakeFiles/kernels_cpu.dir/src/kernels/cpu/pool.cpp.o"),
        ):
            self.assertFalse(isa.is_dedicated_avx512(bad), str(bad))

    def test_empty_dir_and_objdump_failure_return_nonzero(self):
        with tempfile.TemporaryDirectory() as d:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(isa.main(["check_isa_isolation.py", d]), 1)
            self.assertIn("no object files", err.getvalue())

            obj = Path(d) / "kernels_cpu.dir" / "Release" / "expert.obj"
            obj.parent.mkdir(parents=True)
            obj.write_bytes(b"fake")

            fake_fail = subprocess.CompletedProcess([], 1, stdout="", stderr="invalid object")
            err = io.StringIO()
            with mock.patch.object(isa, "disassembler", return_value="fake-objdump"), \
                    mock.patch.object(subprocess, "run", return_value=fake_fail), \
                    contextlib.redirect_stderr(err):
                self.assertEqual(isa.main(["check_isa_isolation.py", d]), 1)
            self.assertIn("invalid object", err.getvalue())

            # EVEX in dedicated MSVC Release/expert.obj passes; EVEX in non-dedicated object fails.
            disasm = "0000000000000000 <my_avx512_kernel>:\n  10:\t62 f1 7d 48 6f 00 \tvmovdqa64 (%rax),%zmm0\n"
            fake_ok = subprocess.CompletedProcess([], 0, stdout=disasm, stderr="")
            with mock.patch.object(isa, "disassembler", return_value="fake-objdump"), \
                    mock.patch.object(subprocess, "run", return_value=fake_ok), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(isa.main(["check_isa_isolation.py", d]), 0)

            bad_obj = Path(d) / "kernels_cpu.dir" / "Release" / "native_expert.obj"
            bad_obj.write_bytes(b"fake")
            with mock.patch.object(isa, "disassembler", return_value="fake-objdump"), \
                    mock.patch.object(subprocess, "run", return_value=fake_ok), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(isa.main(["check_isa_isolation.py", d]), 1)


class TestCMakeInvariants(unittest.TestCase):
    def test_cmake_ordering_and_windows_settings(self):
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertLess(cmake.index("add_subdirectory(tools/vision)"),
                        cmake.index("basalt_gpu_test(NAME hy4_encoder_parity COMMAND hy4_encoder_parity)"))
        self.assertLess(cmake.index("find_package(Python3 COMPONENTS Interpreter QUIET)"),
                        cmake.index("if(BASALT_STDIN_STOP_MODEL AND Python3_Interpreter_FOUND)"))
        self.assertIn(".venv/Scripts/python.exe", cmake)
        self.assertIn('BASALT_BENCH_BUILD_TYPE="$<CONFIG>"', cmake)
        # Every standalone add_test outside _basalt_labelled_test must carry a cpu/gpu label.
        raw_add_tests = re.findall(r"^\s*add_test\((.*)\)$", cmake, re.M)
        self.assertEqual(len(raw_add_tests), 2)  # _basalt_labelled_test(${ARGN}) and model_layering (labelled cpu)
        self.assertIn("NAME model_layering", raw_add_tests[1])

        vision = (ROOT / "tools" / "vision" / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("/Zc:__cplusplus", vision)
        self.assertIn("/Zc:preprocessor", vision)


if __name__ == "__main__":
    unittest.main()
