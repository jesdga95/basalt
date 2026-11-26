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
from serve.util.loglevel import ERR, WARN
from serve.util.telemetry import GPU_KEYS, Telemetry
from serve.util.text import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "serve" / "chat_template.jinja"
APP_JS = (ROOT / "serve" / "web" / "app.js").read_text(encoding="utf-8")
APP_CSS = (ROOT / "serve" / "web" / "app.css").read_text(encoding="utf-8")


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
const fs = require("fs"), vm = require("vm"), { TextDecoder } = require("util");
const el = () => new Proxy(function () {}, {
  get: (t, k) => k === "classList" ? { toggle() {}, add() {}, remove() {} } : ["style", "dataset"].includes(k) ? {}
    : k === "querySelectorAll" ? () => [] : k === "querySelector" ? () => null : typeof k === "symbol" ? undefined : el(),
  set: () => true, apply: () => el() });
const ctx = { document: { querySelector: () => el(), addEventListener() {}, createElement: () => el(), hidden: false },
  sessionStorage: { getItem: () => null, setItem() {} }, location: { hash: "", host: "" }, fetch: () => new Promise(() => {}),
  ResizeObserver: class { observe() {} }, matchMedia: () => ({ matches: false }), requestAnimationFrame() {}, setTimeout() {},
  clearTimeout() {}, performance: { now: () => 0 }, devicePixelRatio: 1, window: { addEventListener() {} }, console,
  AbortController: class { constructor() { this.signal = { aborted: false }; } abort() { this.signal.aborted = true; } }, TextDecoder };
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

    def test_asleep_and_parked_models(self):
        script = r"""
const fs = require("fs"), vm = require("vm"), { TextDecoder } = require("util");
function makeEl(tag = "div") {
  const classes = new Set(), attrs = {};
  const node = {
    tagName: tag.toUpperCase(), children: [], dataset: {}, style: {}, textContent: "", innerHTML: "", title: "", hidden: false, disabled: false,
    get className() { return [...classes].join(" "); },
    set className(v) { classes.clear(); String(v || "").split(/\s+/).filter(Boolean).forEach(c => classes.add(c)); },
    classList: {
      add: (...cs) => cs.forEach(c => classes.add(c)),
      remove: (...cs) => cs.forEach(c => classes.delete(c)),
      contains: c => classes.has(c),
      toggle: (c, f) => { const on = f === undefined ? !classes.has(c) : !!f; on ? classes.add(c) : classes.delete(c); return on; },
    },
    get firstElementChild() { return this.children[0] || null; },
    get lastElementChild() { return this.children[this.children.length - 1] || null; },
    get childElementCount() { return this.children.length; },
    get firstChild() { return this.children[0] || null; },
    append(...xs) { xs.forEach(x => { x.parentElement = this; this.children.push(x); }); },
    replaceChildren(...xs) { this.children = []; this.append(...xs); },
    remove() { if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(x => x !== this); },
    setAttribute(k, v) { attrs[k] = String(v); },
    getAttribute(k) { return attrs[k] ?? null; },
    hasAttribute(k) { return Object.prototype.hasOwnProperty.call(attrs, k); },
    removeAttribute(k) { delete attrs[k]; },
    contains(x) { return x === this || this.children.some(c => c.contains(x)); },
    closest(sel) {
      if (sel.startsWith(".") && classes.has(sel.slice(1))) return this;
      return this.parentElement ? this.parentElement.closest(sel) : null;
    },
    querySelector(sel) {
      for (const c of this.children) {
        if ((sel.startsWith(".") && c.classList.contains(sel.slice(1))) || (sel.startsWith("#") && c.id === sel.slice(1))) return c;
        const hit = c.querySelector(sel);
        if (hit) return hit;
      }
      return null;
    },
    querySelectorAll() { return []; },
    addEventListener() {}, focus() {}, getBoundingClientRect: () => ({ width: 0, height: 0 }),
    getContext: () => ({ setTransform() {}, clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, closePath() {}, fill() {} }),
  };
  return node;
}
const els = new Map();
const getEl = sel => {
  if (!els.has(sel)) {
    const n = makeEl();
    if (sel.startsWith("#")) n.id = sel.slice(1);
    if (sel === "#pbar") n.append(makeEl("i"));
    if (sel === "#tabs") n.append(makeEl("button"));
    els.set(sel, n);
  }
  return els.get(sel);
};
const posts = [];
const ctx = {
  document: { querySelector: getEl, addEventListener() {}, createElement: makeEl, hidden: false, activeElement: null },
  sessionStorage: { getItem: () => null, setItem() {} }, location: { hash: "", host: "localhost" },
  fetch: (path, opts) => {
    if (!opts || opts.method !== "POST") return new Promise(() => {});
    posts.push({ path, body: JSON.parse(opts.body) });
    return new Promise(() => {});
  },
  ResizeObserver: class { observe() {} }, matchMedia: () => ({ matches: true }), requestAnimationFrame() {},
  setTimeout() { return 1; }, clearTimeout() {}, performance: { now: () => 0 }, devicePixelRatio: 1,
  window: { addEventListener() {} }, console,
  AbortController: class { constructor() { this.signal = { aborted: false }; } abort() { this.signal.aborted = true; } },
  TextDecoder,
};
vm.runInNewContext(fs.readFileSync(process.argv[2], "utf8") + "\n" + fs.readFileSync(process.argv[3], "utf8") + `
  const idle = { state: "idle", queued: 0 }, loading = { state: "loading", queued: 0 }, asleep = { state: "asleep", queued: 0 };
  const parkedAndActive = [
    { id: "A", status: { value: "loaded" }, asleep: true, parked_mib: 3000 },
    { id: "B", status: { value: "loaded" }, asleep: false, parked_mib: null },
  ];
  const out = {};
  out.unloadActive = unloadView(parkedAndActive, idle);
  out.unloadDuringSwap = unloadView(parkedAndActive, loading);
  drawModels(parkedAndActive, "B");
  const pop = $("#models").querySelector(".pop");
  out.rows = pop.children.map(b => ({ id: b.dataset.v, cls: b.className, checked: b.getAttribute("aria-checked"), word: b.children[1].textContent }));
  pop.onclick({ target: pop.children[1] });                    // active B: not clickable
  out.postsAfterActiveClick = posts.length;
  pop.onclick({ target: pop.children[0] });                    // parked A: wakes via v1/load
  out.postsAfterParkedClick = posts.slice();
  settle(parkedAndActive, true);                               // still asleep: stays pending
  out.pendingWhileAsleep = !!pending;
  settle([
    { id: "A", status: { value: "loading" }, asleep: true, parked_mib: null },
    { id: "B", status: { value: "loaded" }, asleep: true, parked_mib: 3000 },
  ], true);
  out.clearedOnLoading = !pending;
  control("v1/load", "A");
  settle([
    { id: "A", status: { value: "loaded" }, asleep: false, parked_mib: null },
    { id: "B", status: { value: "loaded" }, asleep: true, parked_mib: 3000 },
  ], true);
  out.clearedOnWake = !pending;
  const onlyParked = [
    { id: "A", status: { value: "loaded" }, asleep: true, parked_mib: 3000 },
    { id: "B", status: { value: "unloaded" }, asleep: false, parked_mib: null },
  ];
  out.unloadOnlyParked = unloadView(onlyParked, idle);
  M = { live: { state: "asleep", queued: 0, slots: { total: 1, active: 0 } }, engine: { model: "A" }, models: onlyParked };
  draw();
  out.onlyParkedHeroEmpty = $(".hero").classList.contains("empty");
  out.onlyParkedState = $("#state").textContent;
  const idleAsleep = [
    { id: "A", status: { value: "loaded" }, asleep: true },
    { id: "B", status: { value: "unloaded" }, asleep: false },
  ];
  out.unloadIdleAsleep = unloadView(idleAsleep, asleep);
  drawModels(idleAsleep, "A");
  out.idleAsleepRowCls = $("#models").querySelector(".pop").children[0].className;
  globalThis.out = out;`, Object.assign(ctx, { posts }));
process.stdout.write(JSON.stringify(ctx.out));
"""
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(script)
        self.addCleanup(os.unlink, f.name)
        r = subprocess.run([NODE, f.name, str(ROOT / "serve" / "web" / "logs.js"), str(ROOT / "serve" / "web" / "app.js")],
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        o = json.loads(r.stdout)
        self.assertEqual((o["unloadActive"]["hidden"], o["unloadActive"]["id"], o["unloadActive"]["disabled"]), (False, "B", False))
        self.assertTrue(o["unloadDuringSwap"]["hidden"])
        self.assertEqual(o["rows"], [
            {"id": "A", "cls": "opt go", "checked": "false", "word": "Asleep"},
            {"id": "B", "cls": "opt", "checked": "true", "word": "Loaded"},
        ])
        self.assertEqual(o["postsAfterActiveClick"], 0)
        self.assertEqual(o["postsAfterParkedClick"], [{"path": "v1/load", "body": {"model": "A"}}])
        self.assertTrue(o["pendingWhileAsleep"])
        self.assertTrue(o["clearedOnLoading"])
        self.assertTrue(o["clearedOnWake"])
        self.assertTrue(o["unloadOnlyParked"]["hidden"])
        self.assertTrue(o["onlyParkedHeroEmpty"])
        self.assertEqual(o["onlyParkedState"], "No model")
        self.assertEqual((o["unloadIdleAsleep"]["hidden"], o["unloadIdleAsleep"]["id"], o["unloadIdleAsleep"]["disabled"]),
                         (False, "A", False))
        self.assertEqual(o["idleAsleepRowCls"], "opt go")


class LogStreamAndLevels(unittest.TestCase):
    """Log SSE empty frames do not trigger reconnect loops, and server 'err' levels style, count and filter."""

    def test_css_matches_server_log_levels(self):
        self.assertIn(f"#log .{ERR} span, #log .{ERR} u", APP_CSS)
        self.assertIn(f"#log.warnonly > div:not(.{WARN}):not(.{ERR}):not(.empty)", APP_CSS)
        self.assertIn(f"#log > div:last-child:not(.{WARN}):not(.{ERR}):not(.empty) span", APP_CSS)
        self.assertIn(f'.ticker[data-lvl="{ERR}"] span', APP_CSS)
        self.assertNotIn("#log .error", APP_CSS)
        self.assertNotIn(":not(.error)", APP_CSS)
        self.assertNotIn('.ticker[data-lvl="error"]', APP_CSS)

    @unittest.skipUnless(NODE, "needs node")
    def test_empty_log_frames_and_err_level_handling(self):
        script = r"""
const fs = require("fs"), vm = require("vm"), { TextDecoder, TextEncoder } = require("util");
function makeEl(tag = "div") {
  const classes = new Set();
  const node = {
    tagName: tag.toUpperCase(), children: [], dataset: {}, style: {}, textContent: "", innerHTML: "", title: "", hidden: false,
    scrollTop: 0, scrollHeight: 0, clientHeight: 0,
    get className() { return [...classes].join(" "); },
    set className(v) { classes.clear(); String(v || "").split(/\s+/).filter(Boolean).forEach(c => classes.add(c)); },
    classList: {
      add: (...cs) => cs.forEach(c => classes.add(c)),
      remove: (...cs) => cs.forEach(c => classes.delete(c)),
      contains: c => classes.has(c),
      toggle: (c, f) => { const on = f === undefined ? !classes.has(c) : !!f; on ? classes.add(c) : classes.delete(c); return on; },
    },
    get childElementCount() { return this.children.length; },
    get firstChild() { return this.children[0] || null; },
    get firstElementChild() { return this.children[0] || null; },
    get lastElementChild() { return this.children[this.children.length - 1] || null; },
    append(...xs) { xs.forEach(x => { x.parentElement = this; this.children.push(x); }); },
    replaceChildren(...xs) { this.children = []; this.append(...xs); },
    remove() { if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(x => x !== this); },
    querySelector(sel) {
      for (const c of this.children) {
        if (sel.startsWith(".") && c.classList.contains(sel.slice(1))) return c;
        const hit = c.querySelector(sel);
        if (hit) return hit;
      }
      return null;
    },
    querySelectorAll() { return []; },
    addEventListener() {}, setAttribute() {}, removeAttribute() {}, hasAttribute: () => false,
  };
  return node;
}
const els = new Map();
const getEl = sel => { if (!els.has(sel)) els.set(sel, makeEl()); return els.get(sel); };
const enc = new TextEncoder();
const streams = [], logPaths = [], timers = [];
function makeStream() {
  const q = []; let waiter = null, closed = false;
  const push = s => {
    const item = { value: enc.encode(s), done: false };
    if (waiter) { const w = waiter; waiter = null; w(item); } else q.push(item);
  };
  const close = () => {
    closed = true;
    if (waiter) { const w = waiter; waiter = null; w({ value: undefined, done: true }); }
  };
  const reader = {
    read() {
      if (q.length) return Promise.resolve(q.shift());
      if (closed) return Promise.resolve({ value: undefined, done: true });
      return new Promise(res => { waiter = res; });
    },
    cancel: () => Promise.resolve(),
    releaseLock() {},
  };
  return { push, close, response: { ok: true, status: 200, body: { getReader: () => reader } } };
}
const ctx = {
  document: { querySelector: getEl, addEventListener() {}, createElement: makeEl, hidden: false },
  sessionStorage: { getItem: () => null, setItem() {} }, location: { hash: "", host: "localhost" },
  fetch: path => {
    if (path.startsWith("api/logs/stream")) {
      logPaths.push(path);
      const st = makeStream();
      streams.push(st);
      return Promise.resolve(st.response);
    }
    return new Promise(() => {});
  },
  ResizeObserver: class { observe() {} }, matchMedia: () => ({ matches: true }), requestAnimationFrame() {},
  setTimeout: (fn, ms) => { timers.push({ fn, ms }); return timers.length; },
  clearTimeout() {}, performance: { now: () => 0 }, devicePixelRatio: 1,
  window: { addEventListener() {} }, console,
  AbortController: class { constructor() { this.signal = { aborted: false }; } abort() { this.signal.aborted = true; } },
  TextDecoder,
};
vm.runInNewContext(fs.readFileSync(process.argv[2], "utf8") + "\n" + fs.readFileSync(process.argv[3], "utf8") + `
  const flush = async () => { for (let i = 0; i < 20; i++) await null; };
  globalThis.run = async () => {
    const out = {};
    await flush();
    streams[0].push('event: logs\\ndata: {"lines":[],"next":0,"instance_id":"s1","reset":false,"dropped":false}\\n\\n');
    await flush();
    out.emptyInitial = {
      status: $("#log-status").textContent, retries: timers.length, lastLog, logInstance,
      placeholder: $("#log").children.map(c => c.textContent),
    };
    streams[0].push('event: logs\\ndata: ' + JSON.stringify({
      lines: [
        { id: 1, t: 1, src: "srv", lvl: "info", msg: "ready" },
        { id: 2, t: 2, src: "eng", lvl: "warn", msg: "low vram" },
        { id: 3, t: 3, src: "eng", lvl: "err", msg: "boom" },
      ],
      next: 3, instance_id: "s1", reset: false, dropped: false,
    }) + '\\n\\n');
    await flush();
    out.afterLines = {
      status: $("#log-status").textContent, retries: timers.length, logCount, warnCount,
      classes: $("#log").children.map(c => c.className),
      tickerLvl: $("#ticker").dataset.lvl,
      tickerHasBoom: $("#ticker-t").textContent.includes("boom"),
    };
    // Simulate disconnect + scheduled reconnect receiving an empty frame (no new lines)
    streams[0].close();
    await flush();
    out.afterClose = { status: $("#log-status").textContent, retries: timers.length };
    timers[timers.length - 1].fn();
    await flush();
    streams[1].push('event: logs\\ndata: {"lines":[],"next":3,"instance_id":"s1","reset":false,"dropped":false}\\n\\n');
    await flush();
    out.afterEmptyReconnect = {
      path: logPaths[1], status: $("#log-status").textContent, retries: timers.length,
      logCount, warnCount, tickerLvl: $("#ticker").dataset.lvl, lineCount: $("#log").children.length,
    };
    // Server restart with empty buffer
    streams[1].push('event: logs\\ndata: {"lines":[],"next":0,"instance_id":"s2","reset":true,"dropped":false}\\n\\n');
    await flush();
    out.afterEmptyReset = {
      status: $("#log-status").textContent, retries: timers.length, logCount, warnCount,
      lastLog, logInstance, tickerText: $("#ticker-t").textContent, tickerLvl: $("#ticker").dataset.lvl,
      placeholder: $("#log").children.map(c => c.textContent),
    };
    return out;
  };`, Object.assign(ctx, { streams, logPaths, timers }));
ctx.run().then(o => process.stdout.write(JSON.stringify(o)));
"""
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(script)
        self.addCleanup(os.unlink, f.name)
        r = subprocess.run([NODE, f.name, str(ROOT / "serve" / "web" / "logs.js"), str(ROOT / "serve" / "web" / "app.js")],
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        o = json.loads(r.stdout)
        self.assertEqual(o["emptyInitial"], {
            "status": "Live", "retries": 0, "lastLog": 0, "logInstance": "s1",
            "placeholder": ["No log lines yet."],
        })
        self.assertEqual(o["afterLines"], {
            "status": "Live", "retries": 0, "logCount": 3, "warnCount": 2,
            "classes": ["", "warn", "err"], "tickerLvl": "err", "tickerHasBoom": True,
        })
        self.assertEqual(o["afterClose"], {"status": "Disconnected; retrying", "retries": 1})
        self.assertEqual(o["afterEmptyReconnect"], {
            "path": "api/logs/stream?after=3&instance=s1", "status": "Live", "retries": 1,
            "logCount": 3, "warnCount": 2, "tickerLvl": "err", "lineCount": 3,
        })
        self.assertEqual(o["afterEmptyReset"], {
            "status": "Live", "retries": 1, "logCount": 0, "warnCount": 0,
            "lastLog": 0, "logInstance": "s2", "tickerText": "", "tickerLvl": "",
            "placeholder": ["No log lines yet."],
        })
