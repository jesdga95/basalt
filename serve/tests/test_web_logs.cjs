"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { createLogParser } = require("../web/logs.js");

const received = [], parse = createLogParser(x => received.push(x));
const frame = 'event: logs\ndata: {"lines":[{"msg":"Привет <script>"}],"next":1}\n\n';
const bytes = new TextEncoder().encode(frame), decoder = new TextDecoder();
for (const byte of bytes) parse(decoder.decode(new Uint8Array([byte]), { stream: true }));
parse(': heartbeat\n\nevent: logs\ndata: {"lines":[],"next":2}\n\n');
assert.equal(received.length, 2);
assert.equal(received[0].lines[0].msg, "Привет <script>");
assert.equal(received[1].next, 2);
assert.throws(() => createLogParser(() => {})("x".repeat(8 * 1024 * 1024 + 1)), /Oversized/);

// Verify app.js append() tolerates empty frames and tracks "err" lines without throwing.
function makeEl(tag = "div") {
  const classes = new Set();
  return {
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
}
const els = new Map();
const getEl = sel => { if (!els.has(sel)) els.set(sel, makeEl()); return els.get(sel); };
const ctx = {
  document: { querySelector: getEl, addEventListener() {}, createElement: makeEl, hidden: false },
  sessionStorage: { getItem: () => null, setItem() {} }, location: { hash: "", host: "localhost" },
  fetch: () => new Promise(() => {}), ResizeObserver: class { observe() {} }, matchMedia: () => ({ matches: true }),
  requestAnimationFrame() {}, setTimeout() { return 1; }, clearTimeout() {}, performance: { now: () => 0 },
  devicePixelRatio: 1, window: { addEventListener() {} }, console,
  AbortController: class { constructor() { this.signal = { aborted: false }; } abort() { this.signal.aborted = true; } },
  TextDecoder, createLogParser,
};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../web/app.js"), "utf8") + `
  clearLog("No log lines yet.");
  append([]);
  append(null);
  globalThis.emptyState = { logCount, warnCount, placeholder: $("#log").children.map(c => c.textContent) };
  append([
    { id: 1, t: 1, src: "srv", lvl: "info", msg: "started" },
    { id: 2, t: 2, src: "eng", lvl: "warn", msg: "fallback" },
    { id: 3, t: 3, src: "eng", lvl: "err", msg: "oom" },
  ]);
  append([]);
  globalThis.loadedState = {
    logCount, warnCount, classes: $("#log").children.map(c => c.className),
    tickerLvl: $("#ticker").dataset.lvl,
  };`, ctx);
assert.deepEqual(JSON.parse(JSON.stringify(ctx.emptyState)), { logCount: 0, warnCount: 0, placeholder: ["No log lines yet."] });
assert.deepEqual(JSON.parse(JSON.stringify(ctx.loadedState)), { logCount: 3, warnCount: 2, classes: ["", "warn", "err"], tickerLvl: "err" });
console.log("Log SSE parser and web log handler: passed");
