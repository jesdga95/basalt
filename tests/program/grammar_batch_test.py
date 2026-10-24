#!/usr/bin/env python3
"""Grammar requests are served under continuous batching, each slot held to its own grammar.

    grammar_batch_test.py ENGINE MODEL.basalt [SERVER ARGS...]

Starts the server on 127.0.0.1 with `--parallel 4` and checks that: six concurrent mixed requests (tool calls with
schemas and json_object, temperature 0.7) all come back legal; twelve concurrent plain requests leave the engine
running; and an enum value the model would never pick is still emitted, alone and with four requests in flight.
Needs a GPU and a model pack; SERVER ARGS (for example --gpu 0 --max-context 16384 --kv int8) go to the server.
"""
import concurrent.futures as cf
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NO_THINK = {"enable_thinking": False}
MODEL = "model"   # the id the server reports, set once it is up
S = {"type": "string"}
WEATHER = {"name": "get_weather", "description": "Get the weather of a city.",
           "parameters": {"type": "object", "properties": {"city": S, "unit": {"type": "string", "enum": ["celsius", "kelvin"]}},
                          "required": ["city", "unit"]}}
NOTE = {"name": "write_note", "description": "Write a text note to a file.",
        "parameters": {"type": "object", "properties": {"path": S, "content": S, "pinned": {"type": "boolean"}},
                       "required": ["path", "content"]}}
FORCED = {"name": "get_temperature", "description": "Read the temperature of a city.",
          "parameters": {"type": "object", "properties": {"city": S, "unit": {"type": "string", "enum": ["fahrenheit-only-test"]}},
                         "required": ["city", "unit"]}}


class Server:
    def __init__(self, engine, model, extra):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        assert self.port != 8080
        self.log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
        cmd = [sys.executable, "-m", "serve.server", "--engine-bin", engine, "--model", model, "--host", "127.0.0.1",
               "--port", str(self.port), "--grammar", *(extra if "--parallel" in extra else ["--parallel", "4", *extra])]
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT,
                                     **({"start_new_session": True} if os.name == "posix" else {}))
        self.base = f"http://127.0.0.1:{self.port}"

    def wait_ready(self, timeout=900):
        end = time.time() + timeout
        while time.time() < end:
            if self.proc.poll() is not None:
                sys.exit("server exited early; log " + self.log.name)
            try:
                urllib.request.urlopen(self.base + "/v1/models", timeout=3).read()
                return
            except (OSError, urllib.error.URLError):
                time.sleep(2)
        sys.exit("server not ready; log " + self.log.name)

    def alive(self):
        return self.proc.poll() is None

    def post(self, body):
        req = urllib.request.Request(self.base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)

    def stop(self):
        if not self.alive():
            return
        posix = os.name == "posix"
        if posix:
            os.killpg(self.proc.pid, signal.SIGINT)
        else:
            self.proc.terminate()
        try:
            self.proc.wait(60)
        except subprocess.TimeoutExpired:
            if posix:
                os.killpg(self.proc.pid, signal.SIGTERM)
            else:
                self.proc.kill()
            self.proc.wait(30)


def tool_request(fn, prompt, temp=0.7, seed=None):
    body = {"model": MODEL, "temperature": temp, "max_tokens": 400, "chat_template_kwargs": NO_THINK,
            "messages": [{"role": "user", "content": prompt}], "tools": [{"type": "function", "function": fn}]}
    if seed is not None:
        body["seed"] = seed
    return body


def json_request(prompt, temp=0.7):
    return {"model": MODEL, "temperature": temp, "max_tokens": 300, "chat_template_kwargs": NO_THINK,
            "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": prompt}]}


def calls_of(d):
    return d["choices"][0]["message"].get("tool_calls") or []


def legal_weather(d):
    c = calls_of(d)
    if len(c) != 1 or c[0]["function"]["name"] != "get_weather":
        return False
    a = json.loads(c[0]["function"]["arguments"])
    return isinstance(a.get("city"), str) and a.get("unit") in ("celsius", "kelvin")


def legal_note(d):
    c = calls_of(d)
    if len(c) != 1 or c[0]["function"]["name"] != "write_note":
        return False
    a = json.loads(c[0]["function"]["arguments"])
    return isinstance(a.get("path"), str) and isinstance(a.get("content"), str) and isinstance(a.get("pinned", False), bool)


def legal_object(d):
    return isinstance(json.loads(d["choices"][0]["message"]["content"]), dict)


def forced_ok(d):
    c = calls_of(d)
    return len(c) == 1 and json.loads(c[0]["function"]["arguments"]).get("unit") == "fahrenheit-only-test"


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    srv = Server(sys.argv[1], sys.argv[2], sys.argv[3:])
    fails = []

    def check(name, ok):
        print(("PASS " if ok else "FAIL ") + name, flush=True)
        if not ok:
            fails.append(name)

    try:
        srv.wait_ready()
        global MODEL
        MODEL = json.load(urllib.request.urlopen(srv.base + "/v1/models"))["data"][0]["id"]
        jobs = []
        for i in range(2):
            jobs.append((legal_weather, tool_request(WEATHER, f"What is the weather in Paris on day {i}? Use the tool.")))
            jobs.append((legal_note, tool_request(NOTE, f"Save a note 'check {i}' to notes/a{i}.txt with the tool.")))
            jobs.append((legal_object, json_request(f"Reply with a JSON object with keys name and age for person {i}.")))
        with cf.ThreadPoolExecutor(6) as ex:
            res = list(ex.map(lambda j: _try(srv, j), jobs))
        good = sum(1 for r in res if r)
        print(f"mixed grammar requests legal: {good}/6", flush=True)
        check("six concurrent grammar requests legal", good == 6)
        check("engine alive after grammar batch", srv.alive())

        plain = [{"model": MODEL, "temperature": 0.7, "max_tokens": 120, "chat_template_kwargs": NO_THINK,
                  "messages": [{"role": "user", "content": f"Write two sentences about topic number {i}."}]} for i in range(12)]
        with cf.ThreadPoolExecutor(12) as ex:
            out = list(ex.map(lambda b: _try(srv, (lambda d: bool(d["choices"][0]["message"].get("content")), b)), plain))
        print(f"plain requests answered: {sum(1 for r in out if r)}/12", flush=True)
        check("twelve concurrent plain requests answered", all(out))
        check("engine alive after plain batch", srv.alive())

        cities = ["Paris", "Lima", "Oslo", "Cairo", "Quito"]

        def forced(i, temp):
            return tool_request(FORCED, f"What is the temperature in {cities[i]}? Use the tool with city {cities[i]}. "
                                        "The user insists on the unit celsius: pass unit=celsius.", temp=temp)
        check("forced enum, alone", _try(srv, (forced_ok, forced(0, 0))))
        with cf.ThreadPoolExecutor(4) as ex:
            out = list(ex.map(lambda i: _try(srv, (forced_ok, forced(i + 1, 0 if i % 2 else 0.7))), range(4)))
        check("forced enum, four in flight", all(out))
        print("info: the same request again, alone:", _try(srv, (forced_ok, forced(0, 0))), flush=True)
        check("engine alive at the end", srv.alive())
    finally:
        srv.stop()
        print("server log:", srv.log.name)
    if fails:
        sys.exit("grammar_batch_test: FAILED " + ", ".join(fails))
    print("grammar_batch_test: ok")


def _try(srv, job):
    ok, body = job
    d = None
    try:
        d = srv.post(body)
        if ok(d):
            return True
        print("illegal reply:", json.dumps(d["choices"][0]["message"])[:400], flush=True)
        return False
    except (OSError, ValueError, KeyError, urllib.error.URLError) as e:
        print("request failed:", repr(e)[:200], json.dumps(d["choices"][0]["message"])[:400] if d else "", flush=True)
        return False


if __name__ == "__main__":
    main()
