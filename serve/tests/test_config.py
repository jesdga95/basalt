"""The --config file: parsing, precedence, server options and engine flags, switches, API key sources."""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.config import (  # noqa: E402
    CONFIG_VERSION,
    ENTRY_OPTIONS,
    MODEL_OPTIONS,
    SAMPLING_KEYS,
    budget_count,
    build_parser,
    check_engine_flags,
    engine_flags_of,
    load_config_file,
    model_settings,
    parallel_count,
    parse_settings,
    sampling_defaults_from_config,
    sampling_items,
    start_items,
    model_line,
    server_options,
    split_engine_extras,
    vision_setting,
)
from serve.engine.flags import flag_value  # noqa: E402
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


def v2(cfg: dict) -> dict:
    """`cfg`'s settings as one model entry and the blocks they belong in: {"server": ..., "engine": ...} by kind."""
    server, entry, shared = {}, {"model": "m.basalt"}, dict(cfg.get("engine", {}))
    for key, value in cfg.get("server", {}).items():
        name = key.replace("_", "-").removeprefix("--")
        if name in ENTRY_OPTIONS or name in MODEL_OPTIONS:
            entry[key] = value
        else:
            server[key] = value
    return {"version": CONFIG_VERSION, "models": [entry], "server": server, "engine": shared}


def parse(cli: list[str], cfg=None):
    if cfg is None:
        return parse_settings(build_parser(), cli)
    return model_settings(build_parser(), ["--config", conf(v2(cfg))] + cli)[0]


def engine_parser_flags() -> dict:
    """{flag: takes a value} for every flag the engine's argument parser accepts."""
    src = (ROOT / "src/program/cli/arguments.cpp").read_text() + (ROOT / "src/models/qwen4exp/flags.cpp").read_text()
    body = src[src.index('if (a == "--help"'):]
    body = body[:body.index("--gpu-stages")]
    out = {}
    for chunk in re.split(r"\belse if \(|\bif \(", body):
        names = re.findall(r'a == "(--[a-z0-9-]+)"', chunk)
        for n in names:
            out[n] = "next(" in chunk or "value(" in chunk
    return out


class ConfigFile(unittest.TestCase):
    def test_blocks(self):
        cfg = load_config_file(conf({"version": 2, "server": {"port": 9000, "lazy": False, "allowed-host": ["a", "b"]},
                                     "engine": {"--kv": "int8", "resident-experts": True, "vision": 2},
                                     "models": [{"model": "m.basalt", "max-context": 32768, "--gpu": "0,1"}]}), build_parser())
        self.assertEqual(cfg.server_argv, ["--port", "9000", "--allowed-host", "a", "--allowed-host", "b"])
        entry = cfg.models[0]
        self.assertEqual(entry.argv[:2], ["--vision", "2"])
        self.assertEqual(entry.argv[-2:], ["--gpu", "0,1"])
        self.assertEqual(entry.shared_words, ["--kv", "int8", "--resident-experts"])
        self.assertEqual(entry.own_words, ["--max-context", "32768"])

    def test_paths_are_relative_to_the_file(self):
        p = conf({"server": {"tokenizer": "tok"}, "models": [{"model": "m/x.basalt"}]})
        cfg = load_config_file(p, build_parser())
        here = Path(p).resolve().parent
        self.assertEqual(cfg.server_argv, ["--tokenizer", str(here / "tok")])
        self.assertEqual(cfg.models[0].argv, ["--model", str(here / "m" / "x.basalt")])
        self.assertEqual(cfg.models[0].path, str(here / "m" / "x.basalt"))

    def test_the_exact_shape(self):
        p = conf({"engine": {"conversation-cache-disk-mib": 32768, "idle-sleep": 300, "pool-workers": 8, "vram-reserve-mib": 500},
                  "models": [{"alias": "Qwen 3.8 FN", "gpu": "auto", "kv": "int8", "max-context": 262144,
                              "model": "a.basalt", "parallel": 4, "slot-full-context": True, "vision": "dynamic"},
                             {"alias": "DeepSeek V4.1 Flash", "gpu": "auto", "max-context": 1048576,
                              "model": "b.basalt", "vision": "dynamic"}],
                  "server": {"host": "0.0.0.0", "port": 8080}})
        (qwen, _, qfile, _), (ds, _, dfile, _) = model_settings(build_parser(), ["--config", p])
        self.assertEqual((qwen.alias, qwen.parallel, qwen.gpu, qwen.vision, qwen.idle_sleep, qwen.host, qwen.port),
                         ("Qwen 3.8 FN", 4, "auto", "dynamic", 300, "0.0.0.0", 8080))
        self.assertEqual((ds.alias, ds.parallel, ds.gpu, ds.vision, ds.idle_sleep, ds.host, ds.port),
                         ("DeepSeek V4.1 Flash", None, "auto", "dynamic", 300, "0.0.0.0", 8080))
        for words in (qfile, dfile):
            self.assertEqual(flag_value(words, "--pool-workers"), "8")
            self.assertEqual(flag_value(words, "--vram-reserve-mib"), "500")
            self.assertEqual(flag_value(words, "--conversation-cache-disk-mib"), "32768")
        self.assertEqual(flag_value(qfile, "--kv"), "int8")
        self.assertIsNone(flag_value(dfile, "--kv"))
        self.assertEqual(flag_value(qfile, "--batch"), "4")
        self.assertEqual(flag_value(qfile, "--max-context"), "262144")     # slot-full-context
        self.assertEqual(flag_value(dfile, "--max-context"), "1048576")

    def test_the_ram_reserve(self):
        p = conf({"engine": {"ram-reserve-mib": 4096}, "models": [{"model": "a.basalt"}, {"model": "b.basalt"}]})
        for _, _, words, _ in model_settings(build_parser(), ["--config", p]):
            self.assertEqual(flag_value(words, "--ram-reserve-mib"), "4096")

    def test_version(self):
        ap = build_parser()
        entry = [{"model": "m.basalt"}]
        load_config_file(conf({"models": entry}), ap)                   # no key: this format
        load_config_file(conf({"version": 2, "models": entry}), ap)
        for old in ({"server": {"model": "m.basalt"}, "engine": {"kv": "int8"}}, {"v": 1, "models": entry},
                    {"version": 1, "models": entry}, {}):
            with self.assertRaisesRegex(ValueError, "Basalt requires v2 config, please upgrade", msg=str(old)):
                load_config_file(conf(old), ap)
        p = conf({"server": {"model": "m.basalt"}})
        with self.assertRaises(ValueError) as cm:
            load_config_file(p, ap)
        words = str(cm.exception).split("upgrade: ", 1)[1].split()       # a command that runs as printed
        self.assertEqual((words[0], words[2], words[3]), ("python3", str(Path(p).resolve()), "-o"))
        self.assertTrue(Path(words[1]).is_file())
        self.assertTrue(words[4].endswith(".v2.json"))
        with self.assertRaisesRegex(ValueError, "this is a v3 config; this Basalt reads v2"):
            load_config_file(conf({"version": 3, "models": entry}), ap)
        for bad in ("2", True, 2.5):
            with self.assertRaisesRegex(ValueError, "whole number"):
                load_config_file(conf({"version": bad, "models": entry}), ap)

    def test_top_level_settings_a_model_lacks_are_left_out_for_it(self):
        p = conf({"engine": {"kv": "int8", "spec": 7, "pool-workers": 8},
                  "models": [{"model": "a.basalt"}, {"model": "b.basalt", "route-prior": 1}]})

        def refused(path, group):
            return path.endswith("b.basalt") and group[0] in ("--spec", "--route-prior")
        (a, _, afile, _), (b, _, bfile, _) = model_settings(build_parser(), ["--config", p], refused)
        self.assertEqual(afile, ["--kv", "int8", "--spec", "7", "--pool-workers", "8"])
        self.assertEqual(a.skipped, [])
        # an entry's own flag is not asked here: the engine's check refuses it at start
        self.assertEqual(bfile, ["--kv", "int8", "--pool-workers", "8", "--route-prior", "1"])
        self.assertEqual(b.skipped, ["--spec"])

    def test_model_and_alias_do_not_go_with_a_config(self):
        p = conf({"models": [{"model": "a.basalt"}]})
        for extra, text in ((["--model", "b.basalt"], "^--model is not supported when using a config file$"),
                            (["--alias", "x"], "^--alias is not supported when using a config file$"),
                            (["--model", "b.basalt", "--alias", "x"],
                             "^--model and --alias are not supported when using a config file$")):
            with self.assertRaisesRegex(ValueError, text):
                model_settings(build_parser(), ["--config", p, *extra])

    def test_the_command_line_applies_to_every_model(self):
        p = conf({"models": [{"model": "a.basalt", "parallel": 2, "max-context": 8192}, {"model": "b.basalt"}]})
        (a, _, afile, acli), (b, _, _, bcli) = model_settings(build_parser(), ["--config", p, "--vision", "cpu", "--kv", "int8"])
        self.assertEqual((a.vision, b.vision), ("cpu", "cpu"))
        self.assertEqual(acli, ["--kv", "int8"])
        self.assertEqual(bcli, ["--kv", "int8"])
        self.assertEqual(flag_value(afile, "--max-context"), "4096")

    def test_errors(self):
        ap = build_parser()
        entry = {"model": "m.basalt"}
        for obj, msg in (({"spec": 7, "models": [entry]}, "unexpected key"),
                         ({"models": []}, "at least one"),
                         ({"models": [1]}, "at least one"),
                         ({"models": {}}, "at least one"),
                         ({"models": [{"alias": "x"}]}, 'models\\[0\\]: "model" names'),
                         ({"models": [{"model": 3}]}, 'models\\[0\\]: "model" names'),
                         ({"models": [entry, {"model": "b", "port": 1}]}, "models\\[1\\].port: a server option"),
                         ({"models": [{"model": "b", "vision": True}]}, "vision: use one of"),
                         ({"models": [{"model": "b", "parallel": [1]}]}, "one value"),
                         ({"models": [{"model": "b", "slot-full-context": 1}]}, "true or false"),
                         ({"models": [entry], "server": {"nope": 1}}, "not a server option"),
                         ({"models": [entry], "server": {"spec": 7}}, "not a server option \\(engine flags go under"),
                         ({"models": [entry], "server": {"engine-bin": "/x/basalt"}}, "engine-bin is set by basalt serve"),
                         ({"models": [entry], "server": {"parallel": 2}}, "a model setting"),
                         ({"models": [entry], "server": {"model": "m"}}, "a model setting"),
                         ({"models": [entry], "server": {"port": [1]}}, "one value"),
                         ({"models": [entry], "server": {"config": "x"}}, "not a server option"),
                         ({"models": [entry], "engine": {"port": 1}}, "a server option"),
                         ({"models": [entry], "engine": {"model": "x"}}, "in a model entry"),
                         ({"models": [entry], "engine": {"spec": None}}, "expected a string or number"),
                         ({"models": [entry], "engine": {"spec": ""}}, "expected a string or number"),
                         ({"models": [entry], "engine": {"control-vector-layer-range": []}}, "empty list"),
                         ({"models": [entry], "engine": {"control-vector-layer-range": [4, [44]]}}, "expected a string or number"),
                         ({"models": [entry], "engine": {"control-vector-layer-range": [4, True]}}, "expected a string or number"),
                         ({"models": [entry], "engine": {"kv": {"a": 1}}}, "expected a string or number"),
                         ({"models": [entry], "engine": []}, "must be an object")):
            with self.assertRaisesRegex(ValueError, msg, msg=str(obj)):
                load_config_file(conf(obj), ap)
        for text in ("[1]", "{"):
            with self.assertRaises(ValueError):
                load_config_file(write(text), ap)
        with self.assertRaises(ValueError):
            load_config_file("/nonexistent/config.json", ap)

    def test_unknown_engine_key(self):
        known = {"--spec", "--kv"}
        engine = load_config_file(conf({"engine": {"spc": 1}, "models": [{"model": "m"}]}), build_parser()).models[0].shared_words
        with self.assertRaisesRegex(ValueError, "--spc is not an engine flag"):
            check_engine_flags(engine, known, "--config c.json")
        check_engine_flags(["--kv", "int8"], known, "x")
        check_engine_flags(["--spec", "-1", "--kv", "int8"], known, "x")      # value words are not names
        check_engine_flags(["--anything", "1"], None, "x")      # no readable binary: unchecked
        with self.assertRaisesRegex(ValueError, "command line: --kvv is not an engine flag"):
            check_engine_flags(["--kv", "int8", "--kvv", "1"], known, "command line")

    def test_list_values_become_words(self):
        engine = load_config_file(conf({"models": [{"model": "m"}], "engine": {
            "control-vector-layer-range": [4, 44], "kv": "int8", "resident_experts": True, "no-x": False, "rate": 0.5,
            "offset": ["-1", -2]}}), build_parser()).models[0].shared_words
        self.assertEqual(engine, ["--control-vector-layer-range", "4", "44", "--kv", "int8", "--resident-experts",
                                  "--rate", "0.5", "--offset", "-1", "-2"])


class Precedence(unittest.TestCase):
    def test_server_options(self):
        a, *_ = parse([], {"server": {"port": 9000, "alias": "cfg"}})
        self.assertEqual((a.port, a.alias), (9000, "cfg"))
        a, *_ = parse(["--port", "9100"], {"server": {"port": 9000, "alias": "cfg"}})
        self.assertEqual((a.port, a.alias), (9100, "cfg"))
        a, *_ = parse([], None)
        self.assertEqual(a.port, 8080)

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
        _, _, file_words, cli_words = parse(["--spec", "5", "--kv", "fp16", "--resident-experts"],
                                            {"engine": {"spec": 7, "max-context": 32768, "kv": "int8"}})
        self.assertEqual(file_words, ["--spec", "7", "--max-context", "32768", "--kv", "int8"])
        self.assertEqual(cli_words, ["--spec", "5", "--kv", "fp16", "--resident-experts"])

    def test_route_tail_skip_config_words(self):
        # the optional-value flag: a number is its value, true is the bare flag, false leaves it off
        for value, words in ((7, ["--route-tail-skip", "7"]), (True, ["--route-tail-skip"]), (False, [])):
            _, _, file_words, _ = parse([], {"engine": {"route-tail-skip": value}})
            self.assertEqual(file_words, words)
        _, _, _, cli_words = parse(["--route-tail-skip", "--resident-experts"])
        self.assertEqual(cli_words, ["--route-tail-skip", "--resident-experts"])

    def test_route_prior_config_words(self):
        _, _, file_words, _ = parse([], {"engine": {"route-prior": 1.5}})
        self.assertEqual(file_words, ["--route-prior", "1.5"])

    def test_the_command_line_wins_over_the_file(self):
        # the engine reads its words in order and a later flag wins
        _, _, file_words, cli_words = parse(["--kv", "fp16"], {"engine": {"kv": "int8"}})
        words = file_words + cli_words
        self.assertEqual(words, ["--kv", "int8", "--kv", "fp16"])
        self.assertEqual(flag_value(words, "--kv"), "fp16")

    def test_switches(self):
        _, _, file_words, cli_words = parse(["--resident-experts"], {"engine": {"on": True, "off": False}})
        self.assertEqual(file_words, ["--on"])
        self.assertEqual(cli_words, ["--resident-experts"])

    def test_cli_passes_a_two_value_flag_as_typed(self):
        _, _, _, cli_words = parse(["--control-vector-layer-range", "4", "44", "--resident-experts"])
        self.assertEqual(cli_words, ["--control-vector-layer-range", "4", "44", "--resident-experts"])
        _, _, file_words, cli_words = parse(["--control-vector-layer-range", "1", "2"],
                                            {"engine": {"control-vector-layer-range": [4, 44]}})
        self.assertEqual(file_words + cli_words, ["--control-vector-layer-range", "4", "44",
                                                  "--control-vector-layer-range", "1", "2"])

    def test_negative_value_words(self):
        _, _, _, cli_words = parse(["--vram-reserve-mib", "-1", "--x=-2"])
        self.assertEqual(cli_words, ["--vram-reserve-mib", "-1", "--x", "-2"])
        check_engine_flags(cli_words, {"--vram-reserve-mib", "--x"}, "cli")

    def test_gpu_is_a_server_setting(self):
        with self.assertRaisesRegex(ValueError, "cli: --peer-device is not an engine flag"):
            check_engine_flags(["--peer-device", "1"], {"--gpu"}, "cli")
        with self.assertRaisesRegex(ValueError, "gpu is a server setting"):
            check_engine_flags(["--gpu", "0,1"], {"--gpu"}, "--config c.json")

    def test_extras(self):
        self.assertEqual(split_engine_extras(["--kv", "int8", "--resident-experts", "--seed=3", "--x", "-1"]),
                         ["--kv", "int8", "--resident-experts", "--seed", "3", "--x", "-1"])
        self.assertEqual(split_engine_extras([]), [])
        for bad in (["stray"], ["-1", "--x"], ["--"]):
            with self.assertRaises(ValueError):
                split_engine_extras(bad)

    def test_former_server_options_are_engine_flags(self):
        for name in ("spec", "mtp-max-t", "max-context", "peer-exclusive"):
            self.assertNotIn(name, server_options(build_parser()))
        _, _, _, cli_words = parse(["--spec", "5", "--peer-exclusive", "--max-context=8192"])
        self.assertEqual(cli_words, ["--spec", "5", "--peer-exclusive", "--max-context", "8192"])

    def test_server_read_flags_are_found_last_wins(self):
        self.assertEqual(flag_value(["--spec", "3", "--kv", "x", "--spec", "5"], "--spec"), "5")
        self.assertIsNone(flag_value(["--spec"], "--spec"))
        self.assertIsNone(flag_value([], "--spec"))

    def test_startup_items(self):
        self.assertEqual(start_items(["--max-context", "32768", "--resident-experts", "--kv", "int8",
                                      "--control-vector-layer-range", "4", "44", "--pool-workers", "8",
                                      "--vram-reserve-mib", "500", "--conversation-cache-disk-mib", "32768"],
                                     ["--spec", "7", "--pool-workers", "12"]),
                         ["ctx=32768", "resident-experts", "kv=int8", "control-vector-layer-range=4,44", "reserve=500MiB",
                          "disk-cache=32768MiB", "spec=7", "pool=12"])
        self.assertEqual(start_items([], [], grammar=(False, "cli")), ["grammar=off"])
        self.assertEqual(start_items([], []), [])
        self.assertEqual(start_items(["--conversation-cache-dir", "/x"], []), [])
        self.assertEqual(start_items(["--conversation-cache-dir", "/x"], [], debug=True), ["conversation-cache-dir=/x"])

    def test_model_line_pads_the_name(self):
        self.assertEqual(model_line("Qwen", 12, ["kv=int8", "ctx=8192"]), "[server] Qwen         kv=int8 ctx=8192")
        self.assertEqual(model_line("Qwen", 4, []), "[server] Qwen")


class SamplingOptions(unittest.TestCase):
    """The sampling keys are server options: consumed here, never passed to the engine."""

    def test_command_line_values(self):
        a, _, _, cli_words = parse(["--temperature", "0.8", "--top-k", "20", "--top-p=0.95", "--min-p", "0.05",
                                    "--presence-penalty", "0.5", "--frequency-penalty", "0.1",
                                    "--repetition-penalty", "1.1", "--penalty-last-n", "128", "--seed", "7"])
        self.assertEqual(a.sampling_launch, {"temperature": 0.8, "top_k": 20, "top_p": 0.95, "min_p": 0.05,
                                             "presence_penalty": 0.5, "frequency_penalty": 0.1,
                                             "repetition_penalty": 1.1, "penalty_last_n": 128, "seed": 7})
        self.assertIsInstance(a.sampling_launch["top_k"], int)
        self.assertEqual(cli_words, [])

    def test_none_given_sets_nothing(self):
        self.assertEqual(parse([])[0].sampling_launch, {})

    def test_config_server_section_and_command_line_wins(self):
        cfg = {"server": {"temperature": 0.3, "top-k": 10, "min_p": 0.1}}
        a, _, file_words, cli_words = parse([], cfg)
        self.assertEqual(a.sampling_launch, {"temperature": 0.3, "top_k": 10, "min_p": 0.1})
        self.assertEqual((file_words, cli_words), ([], []))
        a = parse(["--temperature", "0.9"], cfg)[0]
        self.assertEqual(a.sampling_launch, {"temperature": 0.9, "top_k": 10, "min_p": 0.1})
        self.assertEqual(a.sampling_cli, {"temperature"})

    def test_under_engine_in_the_config_applies_to_every_model(self):
        self.assertEqual(parse([], {"engine": {"temperature": 0.5}})[0].temperature, 0.5)

    def test_bad_values(self):
        for flag, bad in (("--temperature", "-1"), ("--temperature", "hot"), ("--temperature", "nan"),
                          ("--top-p", "0"), ("--top-p", "1.5"), ("--min-p", "2"), ("--top-k", "-2"),
                          ("--top-k", "2.5"), ("--presence-penalty", "inf"), ("--repetition-penalty", "0"),
                          ("--penalty-last-n", "-1"), ("--penalty-last-n", "4097"), ("--seed", "1.5")):
            with self.assertRaises(SystemExit, msg=(flag, bad)):
                parse([flag, bad])
        with self.assertRaises(SystemExit):
            parse([], {"server": {"top_k": -5}})

    def test_values_the_engine_honours(self):
        a = parse(["--top-k", "0", "--seed", "0", "--presence-penalty", "-0.5", "--penalty-last-n", "4096"])[0]
        self.assertEqual(a.sampling_launch, {"top_k": 0, "seed": 0, "presence_penalty": -0.5, "penalty_last_n": 4096})
        self.assertEqual(parse(["--top-k", "1000"])[0].sampling_launch, {"top_k": 1000})

    def test_the_model_file_check_is_the_same(self):
        with self.assertRaisesRegex(SystemExit, "sampling.top_k=-5"):
            sampling_defaults_from_config({"sampling": {"top_k": -5}})
        self.assertEqual(sampling_defaults_from_config({"sampling": {"top_k": 100}}), {"top_k": 100})

    def test_top_k_zero_in_the_model_file_means_no_cut(self):
        self.assertEqual(sampling_defaults_from_config({"sampling": {"top_k": 0, "temperature": 1.0}}), {"temperature": 1.0})

    def test_startup_items_leave_out_what_changes_nothing(self):
        self.assertEqual(sampling_items({"temperature": 0.8, "top_k": 20, "top_p": 0.9, "min_p": 0.0,
                                         "presence_penalty": 0.0, "repetition_penalty": 1.0}),
                         ["temp=0.8", "top_k=20", "top_p=0.9"])
        self.assertEqual(sampling_items({"top_k": 0, "presence_penalty": 1.5}), ["pres=1.5"])


class ThinkingBudget(unittest.TestCase):
    """reasoning-budget-tokens is the launch default for the request field of the same name; the field itself and
    what a budget does to a reply are covered in test_server's ThinkingBudget (Service level)."""

    def test_left_out_the_request_decides(self):
        self.assertIsNone(parse([])[0].reasoning_budget_tokens)

    def test_a_bad_budget_is_refused(self):
        for bad in ("-1", "-3072", "lots", "1.5", ""):
            with self.subTest(value=bad), self.assertRaises(SystemExit):
                parse(["--reasoning-budget-tokens", bad])


class SlotContext(unittest.TestCase):
    """--max-context is shared by the --parallel slots unless --slot-full-context."""

    def context(self, cli, cfg=None):
        _, _, file_words, cli_words = parse(cli, cfg)
        return int(flag_value(file_words + cli_words, "--max-context"))

    def test_each_slot_gets_an_equal_part(self):
        self.assertEqual(self.context(["--parallel", "4", "--max-context", "262144"]), 65536)
        self.assertEqual(self.context(["--parallel", "2", "--max-context", "262144"]), 131072)
        self.assertEqual(self.context(["--max-context", "262144"]), 262144)                  # solo: whole
        self.assertEqual(self.context(["--parallel", "4"], {"engine": {"max-context": 8192}}), 2048)
        self.assertEqual(self.context(["--parallel", "2", "--max-context", "65536"],
                                      {"engine": {"max-context": 8192}}), 32768)

    def test_the_default_is_shared_once_the_engine_names_it(self):
        from serve.config import slot_share
        a, _, file_words, cli_words = parse(["--parallel", "4"])
        self.assertIsNone(flag_value(file_words + cli_words, "--max-context"))      # the engine's plan gives the total
        self.assertTrue(a.split_default)
        self.assertEqual(slot_share(8192, 4), 2048)
        self.assertEqual(slot_share(512, 4), 256)                                     # never under the least slot

    def test_parallel_conflicts_with_batch_and_slots(self):
        for flag in ("--batch", "--slots"):
            with self.assertRaisesRegex(ValueError, f"parallel and engine {flag} both set"):
                parse(["--parallel", "2", flag, "3"])
            with self.assertRaisesRegex(ValueError, f"parallel and engine {flag} both set"):
                parse(["--parallel", "2"], {"engine": {flag: 3}})
        _, _, file_words, cli_words = parse(["--parallel", "2"])
        self.assertEqual(flag_value(cli_words, "--batch"), "2")

    def test_engine_line_shows_the_division(self):
        ap = build_parser()
        a, _, file_words, cli_words = parse_settings(ap, ["--parallel", "4", "--max-context", "65536"])
        items = start_items(file_words, cli_words, (a.shared_context, a.parallel))
        self.assertEqual(flag_value(cli_words, "--max-context"), "16384")
        self.assertIn("ctx=16384 (65536/4 slots)", items)
        self.assertNotIn("slots", " ".join(start_items(file_words, cli_words)))

    def test_slot_full_context_keeps_the_whole(self):
        self.assertEqual(self.context(["--parallel", "4", "--max-context", "262144", "--slot-full-context"]), 262144)
        self.assertEqual(self.context(["--parallel", "4", "--max-context", "262144"],
                                      {"server": {"slot-full-context": True}}), 262144)

    def test_answer_room_switch(self):
        """On by default; off by --no-answer-room or the config's "answer-room": false (a switch on by default)."""
        self.assertTrue(parse([])[0].answer_room)
        self.assertFalse(parse(["--no-answer-room"])[0].answer_room)
        self.assertFalse(parse([], {"server": {"answer-room": False}})[0].answer_room)
        self.assertTrue(parse([], {"server": {"answer-room": True}})[0].answer_room)


class EveryOption(unittest.TestCase):
    def sample(self, act):
        if act.nargs == 0:
            return True
        if act.choices:
            return [c for c in act.choices][-1]
        if act.type is int or act.type is parallel_count or act.type is budget_count:
            return 7
        if act.type is vision_setting:                      # --vision: a GPU number is a valid sample
            return "1"
        if act.type is float:
            return 0.25
        if act.dest in SAMPLING_KEYS:                      # the model file's checks apply
            return 0.5 if SAMPLING_KEYS[act.dest][0] is float else 7
        # a path the config file resolves against its own directory: absolute in this platform's own form, so the
        # resolution leaves it as it is
        return str(Path(TMP.name) / "x") if act.dest in ("model", "tokenizer", "image_dir") else "x"

    def test_every_server_option_by_both_paths(self):
        ap = build_parser()
        for name, act in server_options(ap).items():
            # grammar and answer-room are on by default: their no- forms differ (test_grammar, test_answer_room_switch)
            if name in ("api-key", "engine-bin", "grammar", "answer-room"):
                continue
            v = "localhost" if name == "host" else self.sample(act)
            if isinstance(act, argparse._AppendAction):
                cli, cfgv = ["--" + name, str(v)], [v]
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
        server = set(server_options(build_parser()))
        # A parse that came back empty or cut short would make the loop below vacuous.
        self.assertIn("--help", flags)
        self.assertTrue(any(f[2:] not in server for f in flags), "no engine-only flags parsed")
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
            _, _, _, cli_words = parse(cli)
            _, _, file_words, _ = parse([], {"engine": {flag: cfgv}})
            self.assertEqual(cli_words[:1], [flag], flag)
            self.assertEqual(cli_words[1:], ["5"] * takes_value, flag)
            self.assertEqual(file_words, [flag, "5"] if takes_value else [flag], flag)
            check_engine_flags(cli_words, known, "cli")
            check_engine_flags(file_words, known, "config")

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
                load_config_file(conf({**obj, "models": [{"model": "m"}]}), build_parser())
        with self.assertRaisesRegex(ValueError, "api-key can't be set in a config file"):
            load_config_file(conf({"models": [{"model": "m", "api-key": "k"}]}), build_parser())

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


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Startup(unittest.TestCase):
    """The server process itself: the checks of main() apply to values from the file."""

    def run_server(self, cfg, *argv):
        own_port = [] if "port" in cfg.get("server", {}) or "--port" in argv else ["--port", str(free_port())]
        r = subprocess.run([sys.executable, "-m", "serve.server", "--config", conf(cfg), *own_port, *argv], cwd=ROOT,
                           capture_output=True, text=True, timeout=60, env=os.environ)
        return r.returncode, r.stderr

    def test_requirements_apply_to_the_file(self):
        entry = {"model": "m.basalt"}
        code, err = self.run_server({})
        self.assertEqual(code, 2)
        self.assertIn("Basalt requires v2 config", err)
        code, err = self.run_server({"models": [entry]})
        self.assertIn("the server needs --model and the engine binary", err)
        code, err = self.run_server({"models": [{**entry, "gpu": "a,b"}], "server": {"mock-engine": True}})
        self.assertIn("--gpu a,b: expected numbers", err)
        code, err = self.run_server({"models": [entry]}, "--mock-engine", "--api-key", "")
        self.assertEqual(code, 2)
        self.assertIn("--api-key is empty", err)
        code, err = self.run_server({"models": [entry], "server": {"api-key": "k"}})
        self.assertEqual(code, 2)
        self.assertIn("api-key can't be set in a config file", err)
        code, err = self.run_server({"models": [entry], "server": {"nope": 1}})
        self.assertIn("not a server option", err)
        code, err = self.run_server({"models": [entry], "server": {"port": "abc"}})
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
