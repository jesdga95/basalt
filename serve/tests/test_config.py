"""The --config file: parsing, precedence, server options and engine flags, switches, API key sources."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.engine.modelfile import with_arg  # noqa: E402
from serve.config import ENGINE_PASSTHROUGH, build_parser, check_engine_flags, engine_flags_of, engine_line, engine_settings, load_config_file, parallel_count, parse_settings, server_options, split_engine_extras  # noqa: E402
from serve.security import exposed_without_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TMP = tempfile.TemporaryDirectory()


def write(text: str, name: str | None = None) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=TMP.name) if name is None else \
        open(Path(TMP.name) / name, "w")
    f.write(text)
    f.close()
    return f.name


def conf(obj) -> str:
    return write(json.dumps(obj))


def parse(cli: list[str], cfg=None):
    return parse_settings(build_parser(), (["--config", conf(cfg)] if cfg is not None else []) + cli)


def engine_parser_flags() -> dict:
    """{flag: takes a value} for every flag the engine's argument parser in generate.cpp accepts."""
    src = (ROOT / "src/program/generate.cpp").read_text()
    body = src[src.index('if (a == "--help"'):]
    body = body[:body.index("--gpu-stages")]
    out = {}
    for chunk in re.split(r"\belse if \(|\bif \(", body):
        names = re.findall(r'a == "(--[a-z0-9-]+)"', chunk)
        for n in names:
            out[n] = "next(" in chunk or "value(" in chunk
    return out


class ConfigFile(unittest.TestCase):
    def test_server_and_engine_sections(self):
        ap = build_parser()
        argv, engine = load_config_file(conf({"server": {"port": 9000, "vision": "on", "lazy": False,
                                                         "allowed-host": ["a", "b"], "--gpu": "0,1"},
                                              "engine": {"max-context": 32768, "--kv": "int8",
                                                         "resident-experts": True}}), ap)
        self.assertEqual(argv, ["--port", "9000", "--vision", "on", "--allowed-host", "a", "--allowed-host", "b",
                                "--gpu", "0,1"])
        self.assertEqual(engine, {"--max-context": 32768, "--kv": "int8", "--resident-experts": True})

    def test_paths_are_relative_to_the_file(self):
        p = conf({"server": {"model": "m/x.basalt", "tokenizer": "tok"}})
        argv, _ = load_config_file(p, build_parser())
        here = Path(p).resolve().parent
        self.assertEqual(argv, ["--model", str(here / "m" / "x.basalt"), "--tokenizer", str(here / "tok")])

    def test_errors(self):
        ap = build_parser()
        for obj, msg in (({"spec": 7}, "not the key"),
                         ({"server": {"nope": 1}}, "not a server option"),
                         ({"server": {"spec": 7}}, "under \"engine\""),
                         ({"server": {"engine-bin": "/x/basalt"}}, "engine-bin is set by basalt serve"),
                         ({"server": {"vision": "yes"}}, "vision: use one of on, off, peer, dynamic, cpu"),
                         ({"server": {"vision": True}}, "vision: use one of on, off, peer, dynamic, cpu"),
                         ({"server": {"vision": False}}, "vision: use one of on, off, peer, dynamic, cpu"),
                         ({"server": {"port": [1]}}, "one value"),
                         ({"server": {"config": "x"}}, "not a server option"),
                         ({"engine": {"port": 1}}, "a server option"),
                         ({"engine": {"model": "x"}}, "a server option"),
                         ({"engine": {"spec": None}}, "expected a string or number"),
                         ({"engine": {"spec": ""}}, "expected a string or number"),
                         ({"engine": []}, "must be an object")):
            with self.assertRaisesRegex(ValueError, msg, msg=str(obj)):
                load_config_file(conf(obj), ap)
        for text in ("[1]", "{"):
            with self.assertRaises(ValueError):
                load_config_file(write(text), ap)
        with self.assertRaises(ValueError):
            load_config_file("/nonexistent/config.json", ap)

    def test_unknown_engine_key(self):
        known = {"--spec", "--kv"}
        _, engine = load_config_file(conf({"engine": {"spc": 1}}), build_parser())
        with self.assertRaisesRegex(ValueError, "--spc is not an engine flag"):
            check_engine_flags(engine, known, "--config c.json")
        check_engine_flags({"--kv": "int8"}, known, "x")
        check_engine_flags({"--anything": 1}, None, "x")       # no readable binary: unchecked


class Precedence(unittest.TestCase):
    def test_server_options(self):
        a, *_ = parse([], {"server": {"port": 9000, "alias": "cfg"}})
        self.assertEqual((a.port, a.alias), (9000, "cfg"))
        a, *_ = parse(["--port", "9100"], {"server": {"port": 9000, "alias": "cfg"}})
        self.assertEqual((a.port, a.alias), (9100, "cfg"))
        a, *_ = parse([], None)
        self.assertEqual(a.port, 8095)

    def test_repeatable_options_replace(self):
        a, *_ = parse([], {"server": {"cors-origin": ["http://a"]}})
        self.assertEqual(a.cors_origin, ["http://a"])
        a, *_ = parse(["--cors-origin", "http://b"], {"server": {"cors-origin": ["http://a"]}})
        self.assertEqual(a.cors_origin, ["http://b"])

    def test_switch_true_and_false(self):
        self.assertTrue(parse([], {"server": {"lazy": True}})[0].lazy)
        self.assertFalse(parse([], {"server": {"lazy": False}})[0].lazy)
        self.assertTrue(parse(["--lazy"], {"server": {"lazy": False}})[0].lazy)

    def test_engine_flags(self):
        _, _, file_engine, cli_engine = parse(["--spec", "5", "--kv", "fp16", "--resident-experts"],
                                              {"engine": {"spec": 7, "max-context": 32768, "kv": "int8"}})
        flags, src = engine_settings(file_engine, cli_engine)
        self.assertEqual(flags, {"--spec": 5, "--max-context": 32768, "--kv": "fp16", "--resident-experts": True})
        self.assertEqual(src, {"--spec": "cli", "--max-context": "config", "--kv": "cli", "--resident-experts": "cli"})

    def test_switch_in_args(self):
        base = ["--model", "m"]
        self.assertEqual(with_arg(base, "--resident-experts", True), base + ["--resident-experts"])
        self.assertEqual(with_arg(base + ["--kv", "int8"], "--kv", "fp16"), base + ["--kv", "fp16"])

    def test_extras(self):
        self.assertEqual(split_engine_extras(["--kv", "int8", "--resident-experts", "--seed=3", "--x", "-1"]),
                         {"--kv": "int8", "--resident-experts": True, "--seed": "3", "--x": "-1"})
        with self.assertRaises(ValueError):
            split_engine_extras(["stray"])

    def test_startup_line(self):
        flags, src = engine_settings({"--max-context": 32768, "--resident-experts": True, "--kv": "int8"}, {"--spec": 7})
        flags["--kv"] = False
        self.assertEqual(engine_line(flags, src),
                         "[server] engine: --max-context 32768 (config), --resident-experts (config), "
                         "no --kv (config), --spec 7 (cli)")
        self.assertEqual(engine_line({}, {}, config=False),
                         "[server] engine: built-in defaults (no --config): spec off, 4096 context")
        self.assertEqual(engine_line({"--spec": 7}, {"--spec": "cli"}, config=False),
                         "[server] engine: built-in defaults (no --config): spec 7, 4096 context; --spec 7 (cli)")


class SlotContext(unittest.TestCase):
    """--max-context is shared by the --parallel slots unless --slot-full-context."""

    def flags(self, cli, cfg=None):
        _, _, file_engine, cli_engine = parse(cli, cfg)
        return {**file_engine, **cli_engine}

    def test_each_slot_gets_an_equal_part(self):
        self.assertEqual(self.flags(["--parallel", "4", "--max-context", "262144"])["--max-context"], 65536)
        self.assertEqual(self.flags(["--parallel", "2", "--max-context", "262144"])["--max-context"], 131072)
        self.assertEqual(self.flags(["--max-context", "262144"])["--max-context"], 262144)                  # solo: whole
        self.assertEqual(self.flags(["--parallel", "4"], {"engine": {"max-context": 8192}})["--max-context"], 2048)

    def test_engine_line_shows_the_division(self):
        ap = build_parser()
        a, _, file_engine, cli_engine = parse_settings(ap, ["--parallel", "4", "--max-context", "65536"])
        flags, src = engine_settings(file_engine, cli_engine)
        line = engine_line(flags, src, False, (a.shared_context, a.parallel))
        self.assertEqual(flags["--max-context"], 16384)
        self.assertIn("--max-context 16384 (cli: 65536 / 4 slots)", line)
        self.assertNotIn("slots", engine_line(flags, src, False))

    def test_slot_full_context_keeps_the_whole(self):
        f = self.flags(["--parallel", "4", "--max-context", "262144", "--slot-full-context"])
        self.assertEqual(f["--max-context"], 262144)
        self.assertEqual(self.flags(["--parallel", "4", "--max-context", "262144"],
                                    {"server": {"slot-full-context": True}})["--max-context"], 262144)


class EveryOption(unittest.TestCase):
    def sample(self, act):
        if act.nargs == 0:
            return True
        if act.choices:
            return [c for c in act.choices][-1]
        if act.type is int or act.type is parallel_count:
            return 7
        if act.type is float:
            return 0.25
        # a path the config file resolves against its own directory: absolute in this platform's own form, so the
        # resolution leaves it as it is
        return str(Path(TMP.name) / "x") if act.dest in ("model", "tokenizer") else "x"

    def test_every_server_option_by_both_paths(self):
        ap = build_parser()
        for name, act in server_options(ap).items():
            if name in ENGINE_PASSTHROUGH or name in ("api-key", "engine-bin"):
                continue
            v = "localhost" if name == "host" else self.sample(act)
            if isinstance(act, argparse._AppendAction):
                cli, cfgv = ["--" + name, "x"], ["x"]
            elif v is True:
                cli, cfgv = ["--" + name], True
            else:
                cli, cfgv = ["--" + name, str(v)], v
            by_cli = parse(cli)[0]
            by_cfg = parse([], {"server": {name: cfgv}})[0]
            dest = act.dest
            self.assertEqual(getattr(by_cfg, dest), getattr(by_cli, dest), name)
            self.assertNotEqual(getattr(by_cli, dest), ap.get_default(dest), name)

    def test_every_engine_flag_by_both_paths(self):
        flags = engine_parser_flags()
        self.assertGreater(len(flags), 100)
        server = set(server_options(build_parser())) - set(ENGINE_PASSTHROUGH)
        engine_bin = write("\0".join(flags) + "\0", "fake-engine")
        known = engine_flags_of(engine_bin)
        self.assertEqual(known, set(flags))
        for flag, takes_value in flags.items():
            if flag == "--help":                           # the server answers its own help
                continue
            if flag[2:] in server:                         # the server's own: --model, --vision, ...
                with self.assertRaises(ValueError, msg=flag):
                    load_config_file(conf({"engine": {flag: 1}}), build_parser())
                continue
            cli = [flag, "5"] if takes_value else [flag]
            cfgv = 5 if takes_value else True
            _, _, _, cli_engine = parse(cli)
            _, _, file_engine, _ = parse([], {"engine": {flag: cfgv}})
            self.assertEqual(list(cli_engine), [flag])
            self.assertEqual(float(cli_engine[flag]) if takes_value else cli_engine[flag], 5.0 if takes_value else True, flag)
            self.assertEqual(file_engine, {flag: cfgv}, flag)
            check_engine_flags(cli_engine, known, "cli")
            check_engine_flags(file_engine, known, "config")
            args = with_arg(["--model", "m"], flag, {**file_engine, **cli_engine}[flag])
            self.assertIn(flag, args)

    def test_the_built_engine_accepts_every_flag(self):
        engine_bin = ROOT / "bin/basalt"
        if not engine_bin.exists():
            self.skipTest("no built bin/basalt")
        missing = set(engine_parser_flags()) - (engine_flags_of(str(engine_bin)) or set())
        self.assertEqual(missing, set())


class ApiKey(unittest.TestCase):
    def test_the_config_cannot_hold_a_key(self):
        for obj in ({"server": {"api-key": "k"}}, {"server": {"--api_key": "k"}}, {"server": {"API-KEY": "k"}},
                    {"engine": {"api-key": "k"}}):
            with self.assertRaisesRegex(ValueError, "api-key can't be set in a config file"):
                load_config_file(conf(obj), build_parser())

    def test_a_network_host_from_the_config_starts_without_a_key(self):
        cfg = {"server": {"host": "0.0.0.0"}}
        self.assertEqual(parse([], cfg)[0].host, "0.0.0.0")
        a = parse(["--api-key", "k"], cfg)[0]
        self.assertEqual((a.host, a.api_key), ("0.0.0.0", "k"))
        self.assertEqual(parse([], {"server": {"host": "127.0.0.1"}})[0].host, "127.0.0.1")

    def test_exposed_without_key(self):
        self.assertTrue(exposed_without_key("0.0.0.0", ""))
        self.assertFalse(exposed_without_key("0.0.0.0", "k"))
        self.assertFalse(exposed_without_key("127.0.0.1", ""))


class Startup(unittest.TestCase):
    """The server process itself: the checks of main() apply to values from the file."""

    def run_server(self, cfg, *argv):
        r = subprocess.run([sys.executable, "-m", "serve.server", "--config", conf(cfg), *argv], cwd=ROOT,
                           capture_output=True, text=True, timeout=60, env=os.environ)
        return r.returncode, r.stderr

    def test_requirements_apply_to_the_file(self):
        code, err = self.run_server({})
        self.assertEqual(code, 2)
        self.assertIn("the server needs --model and the engine binary", err)
        code, err = self.run_server({"server": {"model": "m.basalt"}})
        self.assertIn("the server needs --model and the engine binary", err)
        code, err = self.run_server({"server": {"gpu": "a,b", "mock-engine": True}})
        self.assertIn("--gpu a,b: expected numbers", err)
        code, err = self.run_server({"server": {"mock-engine": True}, "engine": {"spec": 7}})
        self.assertIn("engine flags need --model", err)
        code, err = self.run_server({}, "--mock-engine", "--api-key", "")
        self.assertEqual(code, 2)
        self.assertIn("--api-key is empty", err)
        code, err = self.run_server({"server": {"api-key": "k"}})
        self.assertEqual(code, 2)
        self.assertIn("api-key can't be set in a config file", err)
        code, err = self.run_server({"server": {"nope": 1}})
        self.assertIn("not a server option", err)
        code, err = self.run_server({"server": {"port": "abc"}})
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
