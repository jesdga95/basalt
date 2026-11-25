"""The page contract: what the console's app.js reads of /metrics is what the server sends."""
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate
from serve.service import Service
from serve.util.telemetry import GPU_KEYS, Telemetry
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "serve" / "chat_template.jinja"
APP_JS = (ROOT / "serve" / "web" / "app.js").read_text(encoding="utf-8")


class PageContract(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = MockEngine(tok, "Hi.")
        self.engine.conversation = {"parked": 2, "bytes": 5, "evictions": 1, "restored": 9, "restored_ms": 1.5}
        self.svc = Service(self.engine, tok, ChatTemplate.from_file(TEMPLATE))
        self.m = self.svc.metrics()

    def reads(self, var, text=APP_JS):
        return set(re.findall(rf"\b{var}\.([a-z_]+)", text))

    def test_the_top_level_fields(self):
        top = self.reads("M") - {"slots", "models", "swap"}          # /slots, and the router's
        self.assertTrue({"live", "engine", "history", "requests", "best", "hardware", "totals", "conversation"} <= top)
        self.assertEqual(top - set(self.m), set())

    def test_the_live_fields(self):
        live = self.reads("live")
        self.assertTrue({"state", "phase", "slots"} <= live)
        self.assertEqual(live - set(self.m["live"]), set())

    def test_the_conversation_and_totals_fields(self):
        self.assertEqual(self.reads("conv") - set(self.m["conversation"]), set())
        totals = APP_JS.split("function drawTotals")[1].split("\n}\n")[0]
        self.assertEqual(self.reads("n", totals) - set(self.m["totals"]), set())
        self.assertEqual(self.reads("t", totals) - set(self.m["totals"]), set())


class PendingFields(unittest.TestCase):
    """Fields the page draws once the server sends them: read only through ahead(), listed in PENDING, and not sent yet."""

    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "Hi."), tok, ChatTemplate.from_file(TEMPLATE))

    def hardware(self):
        """The fields of /metrics' `hardware`: the telemetry's reading."""
        tel = Telemetry(gpu_factory=lambda i: type("NoCard", (), {"ok": lambda self: False})())
        self.addCleanup(tel.stop)
        return set(tel.now) | set(tel.sample())

    def test_every_pending_field_is_listed_and_not_sent_yet(self):
        block = APP_JS.split("const PENDING = {")[1].split("\n};\n")[0]
        listed = {scope: set(re.findall(r'"(\w+)"', names)) for scope, names in re.findall(r"(\w+): \[([^\]]*)\]", block)}
        read = set(re.findall(r'ahead\(\w+, "(\w+)"\)', APP_JS))
        self.assertEqual(read, set().union(*listed.values()))             # each read is listed, each listed field is read
        m = self.svc.metrics()
        sent = {"metrics": set(m), "engine": set(m["engine"]), "slot": set(self.svc.slot_table()[0][0]),
                "gpu": set(GPU_KEYS), "hardware": self.hardware()}
        self.assertLessEqual(set(listed), set(sent))
        for scope, names in listed.items():
            # a field the server now sends is read directly and leaves PENDING
            self.assertEqual(names & sent[scope], set(), scope)


NODE = shutil.which("node")
# app.js run in node over a stub DOM, returning what its pane builders draw for the inputs on stdin
HARNESS = r"""
const fs = require("fs"), vm = require("vm");
const el = () => new Proxy(function () {}, {
  get: (t, k) => k === "classList" ? { toggle() {}, add() {}, remove() {} } : ["style", "dataset"].includes(k) ? {}
    : k === "querySelectorAll" ? () => [] : k === "querySelector" ? () => null : typeof k === "symbol" ? undefined : el(),
  set: () => true, apply: () => el() });
const ctx = { document: { querySelector: () => el(), addEventListener() {}, createElement: () => el(), hidden: false },
  sessionStorage: { getItem: () => null, setItem() {} }, location: { hash: "", host: "" }, fetch: () => new Promise(() => {}),
  ResizeObserver: class { observe() {} }, matchMedia: () => ({ matches: false }), requestAnimationFrame() {}, setTimeout() {},
  clearTimeout() {}, performance: { now: () => 0 }, devicePixelRatio: 1, console };
const input = JSON.parse(fs.readFileSync(0, "utf8"));
vm.runInNewContext(fs.readFileSync(process.argv[2], "utf8") + `
  globalThis.out = {
    slot: Object.values(slotSections(input.slot, [], input.reqs, "full")).join(""),
    card: slotCard(input.slot, [], input.reqs, 100, "full", { draft: true, hit: true, cols: "1fr" }),
    gpu: gpuDetail(input.gpu),
    split: splitCells([["VRAM", "", 0.9, "t0"], ["CPU", "", 0.1, "t2"]], 10),
    empty: splitCells([["VRAM", "", null, "t0"], ["CPU", "", null, "t2"]], 4),
    down: slotsDown(2),
    idle: Object.values(slotSections({ id: 0, n_ctx: 262144, is_processing: false, state: "idle", api: null, tok_s: 0, generated: 0,
      prompt_tokens: 0, reused: null, ttft_s: null, draft_len: null, ctx_used: 18173, drafts_offered: 154, drafts_accepted: 108 }, [],
      [{ id: "r_0007", slot: 0, prompt_tokens: 18005, reused: 18000, output_tokens: 169, decode_tok_s: 209.7, duration_s: 0.9,
         prompt_ms: 30, ttft_s: 0.04, draft_len: 2.5 }], "quarter")).join(""),
    lone: mine([{ id: "a", slot: null }, { id: "b", slot: 1 }], { id: 0 }).map(r => r.id),
    reads: [{ id: "c", prompt_tokens: 25797, reused: 25761, prompt_ms: 31 }, { id: "d", prompt_tokens: 24950, reused: 0, prompt_ms: 2166 }],
  };
  out.headline = out.reads.find(realRead).id;
  out.best = bestRead({}, out.reads).id;
  out.serverBest = bestRead({ prefill: { id: "s", tok_s: 1 } }, out.reads).id;`, Object.assign(ctx, { input }));
process.stdout.write(JSON.stringify(ctx.out));
"""


@unittest.skipUnless(NODE, "needs node")
class MissingFieldsHide(unittest.TestCase):
    """What the server does not send yet is left out of the panels, and drawn once it arrives."""

    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "Hi."), tok, ChatTemplate.from_file(TEMPLATE))
        self.slot = {**self.svc.slot_table()[0][0], "is_processing": True, "state": "generation", "generated": 10}
        self.gpu = {k: None for k in GPU_KEYS} | {"index": 0, "mem_used": 1, "mem_total": 2, "pcie_gen": 5, "pcie_width": 16}

    def draw(self, slot, gpu):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(HARNESS)
        self.addCleanup(os.unlink, f.name)
        r = subprocess.run([NODE, f.name, str(ROOT / "serve" / "web" / "app.js")], input=json.dumps({"slot": slot, "gpu": gpu, "reqs": []}),
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def test_the_fields_the_server_sends_draw_and_the_rest_are_left_out(self):
        out = self.draw(self.slot, self.gpu)
        self.assertIn("Current request", out["slot"])
        self.assertIn("Context", out["slot"])
        for gone in ("Expert lookups", "Elapsed", "Endpoint", "Read time", "Drafting", 'class="pos"'):
            self.assertNotIn(gone, out["slot"])                         # no per-slot split, start, path, read time or drafts
        self.assertIn("PCIe link", out["gpu"])
        self.assertIn("Traffic in", out["gpu"])
        for gone in ("SM clock", "Memory clock", "Host → GPU", "Experts held", "Throttling"):
            self.assertNotIn(gone, out["gpu"])

    def test_each_lights_up_when_its_field_arrives(self):
        slot = {**self.slot, "started": 1, "path": "/v1/messages", "prompt_ms": 310.0, "accept_pos": [0.9, 0.5],
                "drafts_offered": 40, "drafts_accepted": 30, "draft_len": 2.0,
                "hit": {"vram": 0.9, "peer": None, "cpu": 0.1, "lookups_s": 1, "peer_s": 0, "ram_s": 1}}
        out = self.draw(slot, {**self.gpu, "sm_clock": 2680, "throttle": "", "expert_slots": 9})
        for shown in ("Expert lookups", "Elapsed", "Read time", "Drafting", 'class="pos"'):
            self.assertIn(shown, out["slot"])
        self.assertIn("messages", out["card"].split("</header>")[0])          # the endpoint, in the card's header
        # the panel never scrolls: at most four rows under each section's figure
        for sec in out["card"].split("<section>")[1:]:
            self.assertLessEqual(sec.count("<dt>"), 4)
        for shown in ("SM clock", "Experts held", "Throttling"):
            self.assertIn(shown, out["gpu"])

    def test_the_expert_map_without_a_per_expert_map_is_the_tier_split(self):
        out = self.draw(self.slot, self.gpu)
        self.assertEqual(out["split"], ["t0 flat"] * 9 + ["t2 flat"])
        self.assertEqual(out["empty"], ["off"] * 4)                      # no request yet: outlines, no made-up split

    def test_with_no_slot_up_the_cards_stay_quiet(self):
        out = self.draw(self.slot, self.gpu)
        self.assertIn("Slot 1", out["down"])
        self.assertIn("Slot 2", out["down"])
        self.assertNotIn("No model", out["down"])                       # no instructional line
        self.assertNotIn("No model loaded", APP_JS)
        self.assertEqual(out["lone"], ["a"])                             # a request in no slot ran in the only one

    def test_an_idle_slot_shows_its_last_request(self):
        out = self.draw(self.slot, self.gpu)["idle"]                     # the live shape of an idle slot in a 4-slot server
        for shown in ("209.7", "18,005", "100%", "18.2K", "70%", "2.5", "108"):
            self.assertIn(shown, out)

    def test_a_prompt_read_needs_real_reading_for_the_headline_and_the_best(self):
        out = self.draw(self.slot, self.gpu)
        self.assertEqual(out["headline"], "d")                           # a follow-up that read 36 tokens has no read rate
        self.assertEqual(out["best"], "d")                               # no best from the server: the fastest real read listed
        self.assertEqual(out["serverBest"], "s")

# app.js in node over the stub DOM, with fetch answering each post as `answer` says: the load and unload controls step by step
CONTROLS = HARNESS.split("const input =")[0] + r"""
const posts = [];
let answer = null;
ctx.fetch = (path, opts) => {           // the page's polls never answer; each post waits for the test
  if (!opts || opts.method !== "POST") return new Promise(() => {});
  posts.push(path); return new Promise(res => { answer = res; });
};
vm.runInNewContext(fs.readFileSync(process.argv[2], "utf8") + `
  const models = (a, b) => [{ id: "A", status: { value: a } }, { id: "B", status: { value: b } }];
  const idle = { state: "idle", queued: 0 };
  const settleAll = async () => { for (let i = 0; i < 20; i++) await null; };     // let the post's answer run through
  globalThis.run = async (answer, posts) => {
    const out = {};
    control("v1/unload", "A"); control("v1/unload", "A"); control("v1/load", "B");
    out.unloadPosts = posts.length;
    out.during = unloadView(models("loaded", "unloaded"), idle);
    settle(models("loaded", "unloaded"), true); out.stillPending = !!pending;     // no answer yet: still in flight
    answer()({ ok: true, json: async () => ({}) }); await settleAll();
    settle(models("unloaded", "unloaded"), true); out.after = unloadView(models("unloaded", "unloaded"), idle); out.cleared = !pending;
    control("v1/unload", "A"); answer()({ ok: false, status: 409, json: async () => ({ error: { message: "a request is running" } }) });
    await settleAll();
    out.failedPending = !!pending; out.failed = failed && failed.text; out.restored = unloadView(models("loaded", "unloaded"), idle);
    failed = null;
    control("v1/load", "B"); out.loadPending = pending && pending.path;
    settle(models("unloaded", "loading"), true); out.loadCleared = !pending;
    out.posts = posts.slice();
    return out;
  };`, ctx);
ctx.run(() => answer, posts).then(o => process.stdout.write(JSON.stringify(o)));
"""


@unittest.skipUnless(NODE, "needs node")
class LoadControls(unittest.TestCase):
    """Unload and Load: one post per click, a word while in flight, settled by the models list, restored on a refusal."""

    def test_the_steps(self):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(CONTROLS)
        self.addCleanup(os.unlink, f.name)
        r = subprocess.run([NODE, f.name, str(ROOT / "serve" / "web" / "app.js")], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        o = json.loads(r.stdout)
        self.assertEqual(o["unloadPosts"], 1)                            # repeated clicks and a load meanwhile post nothing
        self.assertEqual((o["during"]["disabled"], o["during"]["label"]), (True, "Unloading…"))
        self.assertTrue(o["stillPending"])
        self.assertTrue(o["after"]["hidden"])                            # unloaded: the button goes, as with nothing loaded
        self.assertTrue(o["cleared"])
        self.assertFalse(o["failedPending"])                             # refused: the button is back, the error in the header
        self.assertEqual(o["failed"], "Unload failed: a request is running")
        self.assertEqual((o["restored"]["disabled"], o["restored"]["label"]), (False, "Unload"))
        self.assertEqual(o["loadPending"], "v1/load")
        self.assertTrue(o["loadCleared"])                                # loading: the load bar takes over
        self.assertEqual(o["posts"], ["v1/unload", "v1/unload", "v1/load"])
