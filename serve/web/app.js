"use strict";
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const num = v => typeof v === "number" && Number.isFinite(v);
const fmt = (v, d = 0) => num(v) ? v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }) : "—";
const p2 = n => String(n).padStart(2, "0");
const clock = ms => { const d = new Date(ms); return `${p2(d.getHours())}:${p2(d.getMinutes())}:${p2(d.getSeconds())}.${String(d.getMilliseconds()).padStart(3, "0")}`; };
const gib = n => num(n) ? (n / 1024 ** 3).toFixed(1) : "—";
const HIST = 60;
let perStream = Array(HIST).fill(0);   // mean tok/s per decoding slot, one sample per poll

let apiKey = sessionStorage.getItem("basalt.key") || "";
let M = null;          // the last /metrics reading

async function get(path) {
  const r = await fetch(path, { cache: "no-store", headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {} });
  if (r.status === 401) throw Object.assign(new Error("locked"), { locked: true });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

/* ---- views ---- */
function view(v) {
  $("#view-stats").hidden = v !== "stats";
  $("#view-hardware").hidden = v !== "hardware";
  $("#top").hidden = v !== "stats";
  $$(".tabs .tab").forEach(t => t.setAttribute("aria-selected", String(t.dataset.view === v)));
  if (location.hash.slice(1) !== v) history.replaceState(null, "", "#" + v);
  if (v === "stats") scrollLog();
}
$$(".tabs .tab").forEach(t => t.onclick = () => view(t.dataset.view));

/* ---- charts ---- */
const CHARTS = [
  { id: "decode", name: "Decode", steps: [50, 100, 200, 250, 300, 400, 500, 600, 700, 800, 1000, 1200, 1500, 2000], hover: null, v: Array(HIST).fill(0), max: 400 },
  { id: "prefill", name: "Prefill", steps: [2000, 4000, 8000, 16000, 32000], hover: null, v: Array(HIST).fill(0), max: 8000 }];
const axis = n => n >= 1000 ? `${n / 1000}k` : String(n);
// the fastest whole request this tab has seen (kept across reloads, cleared on a server restart or a click): a
// request's own mean, not a burst, and only one big enough that fixed costs do not decide it
const BEST_MIN_OUT = 256, BEST_MIN_READ = 4096;
const ktok = n => n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}K` : String(n);
let best = (() => { try { return JSON.parse(sessionStorage.getItem("basalt.best")) || {}; } catch (e) { return {}; } })();
const saveBest = () => { try { sessionStorage.setItem("basalt.best", JSON.stringify(best)); } catch (e) { /* private mode */ } };
function noteBest(reqs) {
  const keep = (id, v, r, extra) => {
    if (num(v) && v > 0 && !(best[id] && best[id].v >= v)) { best[id] = { v, t: r.time * 1000, id: r.id, ...extra }; saveBest(); }
  };
  for (const r of reqs) {
    if (r.output_tokens >= BEST_MIN_OUT)
      keep("decode", r.decode_tok_s, r, { out: r.output_tokens, ctx: (r.prompt_tokens || 0) + r.output_tokens,
        acc: r.drafts_offered > 0 ? r.drafts_accepted / r.drafts_offered : null });
    const read = (r.prompt_tokens || 0) - (r.reused || 0);
    if (read >= BEST_MIN_READ && r.prompt_ms > 0) keep("prefill", read / (r.prompt_ms / 1000), r, { read });
  }
}
function drawBest(id) {
  const b = best[id], el = $("#pk-" + id);
  el.hidden = !b;
  if (!b) return;
  text2(el.querySelector("i"), id === "decode"
    ? `${fmt(b.v, 1)} tok/s · ${fmt(b.out)} tok${num(b.acc) ? ` · ${Math.round(b.acc * 100)}% accept` : ""} · ${ktok(b.ctx)} ctx`
    : `${fmt(b.v)} tok/s · ${ktok(b.read)} read`);
  el.title = `Fastest whole request this session (${b.id}, ${clock(b.t).slice(0, 8)}): `
    + (id === "decode" ? `at least ${BEST_MIN_OUT} output tokens` : `at least ${fmt(BEST_MIN_READ)} newly read prompt tokens`)
    + ". Kept in this tab until the server restarts. Click to reset.";
}
["decode", "prefill"].forEach(id => $("#pk-" + id).onclick = () => { delete best[id]; saveBest(); draw(); });
const tail = a => { const v = (a || []).slice(-HIST).map(x => num(x) ? x : 0); while (v.length < HIST) v.unshift(0); return v; };

function drawChart(c, now, last, lastNote) {
  const el = $("#chart-" + c.id), [cur, dot, tip] = el.querySelectorAll(".cur, .cdot, .tip");
  const label = $("#rd-" + c.id + "-l"), read = $("#rd-" + c.id), unit = "<small>tok/s</small>";
  // the axis follows every plotted line (aggregate and per stream), with headroom, so nothing clips
  const per = c.id === "decode" ? perStream : [];
  const top = Math.max(...c.v, ...per, 0) * 1.05;
  c.max = c.steps.find(s => s >= top) || Math.ceil(top / 500) * 500;
  const pts = a => a.map((x, i) => `${i},${(100 - clamp(x, 0, c.max) / c.max * 100).toFixed(2)}`).join(" ");
  $("#line-" + c.id).setAttribute("points", pts(c.v));
  if (c.id === "decode") $("#line-per").setAttribute("points", pts(per));
  el.querySelector(".y0").textContent = axis(c.max);
  el.querySelector(".y1").textContent = axis(c.max / 2);
  const h = c.hover;
  [cur, dot, tip].forEach(e => e.hidden = h == null);
  if (h == null) {
    const live = num(now) && now > 0, shown = live ? now : last;
    label.textContent = `${c.name}, ${live ? "now" : `last request${lastNote ? ` · ${lastNote}` : ""}`}`;
    read.innerHTML = num(shown) ? `${fmt(shown, c.id === "decode" ? 1 : 0)}${unit}` : "—";
    return;
  }
  const v = c.v[h], x = h / (HIST - 1) * 100, y = 100 - clamp(v, 0, c.max) / c.max * 100, ago = HIST - 1 - h;
  const when = ago ? `${ago} s ago` : "Now", val = v > 0 ? `${fmt(v, c.id === "decode" ? 1 : 0)} tok/s` : "Idle";
  cur.style.left = x + "%";
  dot.style.left = x + "%"; dot.style.top = y + "%";
  tip.style.left = x + "%"; tip.style.top = clamp(y, 16, 84) + "%";
  tip.classList.toggle("flip", h > 34);
  tip.innerHTML = `<b>${val}</b><span>${when}</span>`;
  label.textContent = `${c.name}, ${when.toLowerCase()}`;
  read.innerHTML = v > 0 ? `${fmt(v, c.id === "decode" ? 1 : 0)}${unit}` : "Idle";
}
CHARTS.forEach(c => {
  const el = $("#chart-" + c.id);
  el.addEventListener("pointermove", e => {
    const r = el.getBoundingClientRect();
    c.hover = clamp(Math.round((e.clientX - r.left) / r.width * (HIST - 1)), 0, HIST - 1);
    draw();
  });
  el.addEventListener("pointerleave", () => { c.hover = null; draw(); });
});

/* ---- drawing the metrics ---- */
function setState(s, label) { $("#state").dataset.s = s; text("#state", label); }
const share = r => {      // the tier split of the routed lookups: [vram, peer, cpu] as fractions, or null
  if (!r || !num(r.hit_rate)) return null;
  const peer = num(r.pcie_share) ? r.pcie_share : 0;
  return [r.hit_rate * (1 - peer), peer, (1 - r.hit_rate) * (1 - peer)];
};
const pct = f => num(f) ? `${(f * 100).toFixed(1)}%` : "—";
const text = (id, v) => { const e = $(id); if (e.textContent !== v) e.textContent = v; };

function draw() {
  if (!M) return;
  const live = M.live || {}, eng = M.engine || {}, hw = M.hardware || {}, st = M.hardware_static || {};
  const gpus = hw.gpus || [], g0 = gpus[0] || {}, names = Object.fromEntries((st.gpus || []).map(g => [g.index, g.name]));
  const reqs = M.requests || [], r0 = reqs[0], busy = live.state === "reading" || live.state === "generating";
  const generating = live.state === "generating";
  const sl = live.slots || {}, multi = sl.total > 1, slotRows = M.slots || [], dec = slotRows.filter(s => s.state === "decoding").length;

  const name = (live.state || "idle")[0].toUpperCase() + (live.state || "idle").slice(1);
  setState(live.state || "idle", live.queued > 0 ? `${name} · ${live.queued} queued` : name);
  text("#model", eng.model || "");

  const last = r0 && num(r0.decode_tok_s) ? r0.decode_tok_s : null, tok = generating && num(live.tok_s) ? live.tok_s : last;
  $("#v-tok").textContent = num(tok) ? tok.toFixed(1) : "—";
  $("#v-tok").classList.toggle("idle", !generating);
  $("#heat").style.left = clamp((tok || 0) / 800, 0, 1) * 100 + "%";
  CHARTS[0].name = multi ? "Tok/s (aggregate)" : "Decode";
  text("#k-tok", multi ? "Tok/s (aggregate)" : "Tokens/s");
  text("#speed-note", multi ? "Last 60 s · dashed: per stream" : "Last 60 s");
  $("#line-per").style.display = multi ? "" : "none";
  $("#s-busy").hidden = !multi;
  $("#s-busy").innerHTML = `<b>${sl.active || 0}</b> of ${sl.total} slots busy${dec && num(live.tok_s) ? ` · ${fmt(live.tok_s / dec)} per stream` : ""}`;
  text("#limit", num(g0.power_limit) ? `@ ${Math.round(g0.power_limit)} W` : "");
  $("#v-acc").innerHTML = r0 && r0.drafts_offered > 0 ? `${(r0.drafts_accepted / r0.drafts_offered * 100).toFixed(0)}<small>%</small>` : "—";
  const ctx = multi ? slotRows.reduce((a, s) => a + (s.is_processing ? s.ctx_used : 0), 0)
    : busy ? (live.prompt_tokens || 0) + (live.generated || 0) : r0 ? (r0.prompt_tokens || 0) + (r0.output_tokens || 0) : null;
  text("#v-ctx", num(ctx) ? fmt(ctx) : "—");
  text("#s-ctx", num(eng.max_context) ? `of ${fmt(slotRows.length > 1 && num(slotRows[0].n_ctx) ? slotRows[0].n_ctx : eng.max_context)} tokens${slotRows.length > 1 ? " (per slot)" : ""}` : "");

  const h = M.history || {};
  CHARTS[0].v = tail(h.tok_s); CHARTS[1].v = tail(h.prefill_tok_s_mean);
  const lastPrefill = r0 && r0.prompt_ms > 0 && r0.prompt_tokens > (r0.reused || 0) ? (r0.prompt_tokens - (r0.reused || 0)) / (r0.prompt_ms / 1000) : null;
  drawChart(CHARTS[0], generating ? live.tok_s : null, last);
  const lastRead = r0 ? (r0.prompt_tokens || 0) - (r0.reused || 0) : 0;   // short prompts are dominated by fixed cost, so show the size
  drawChart(CHARTS[1], live.state === "reading" ? live.prefill_tok_s_mean : null, lastPrefill, lastPrefill != null ? `${fmt(lastRead)} tokens` : "");
  noteBest(reqs); drawBest("decode"); drawBest("prefill");

  text("#s-load", num(g0.util) ? `${Math.round(g0.util)}%` : "—");
  text("#s-vram", num(g0.mem_used) ? `${gib(g0.mem_used)} / ${gib(g0.mem_total)} GiB` : "—");
  text("#s-temp", num(g0.temp) ? `${Math.round(g0.temp)} °C` : "—");
  text("#s-pow", num(g0.power) ? `${Math.round(g0.power)}${num(g0.power_limit) ? ` / ${Math.round(g0.power_limit)}` : ""} W` : "—");
  text("#s-cpu", num(hw.cpu) ? `${Math.round(hw.cpu)}%` : "—");
  text("#s-ram", num(hw.ram_total) ? `${gib(hw.ram_used)} / ${gib(hw.ram_total)} GiB` : "—");
  text("#s-disk", num(hw.disk_read_mb) ? `${hw.disk_read_mb.toFixed(1)} MB/s` : "—");

  const sh = share(r0);
  [["vram", 0], ["peer", 1], ["cpu", 2]].forEach(([k, i]) => {
    $("#tr-" + k).style.width = sh ? sh[i] * 100 + "%" : "0";
    text("#tl-" + k, sh ? pct(sh[i]) : "—");
  });

  drawRequests(live, reqs, busy, multi);
  drawSlots(M, multi);
  drawHardware(eng, st, hw, gpus, names, sh);
}

function drawRequests(live, reqs, busy, multi) {
  const rows = [];
  if (busy && !multi) rows.push({ t: Date.now() / 1000 - (live.elapsed_s || 0), id: "Running", path: "—", out: live.generated, tps: live.tok_s, dur: live.elapsed_s, live: true });
  reqs.forEach(r => rows.push({ t: r.time, id: r.id, path: (r.path || "").replace("/v1/", "").replace("chat/completions", "chat"), out: r.output_tokens, tps: r.decode_tok_s, dur: r.duration_s, finish: r.finish }));
  const total = rows.length; rows.length = Math.min(rows.length, 6);   // with slots, the lanes show what is running
  const tb = $("#t-req"); tb.replaceChildren();
  rows.forEach(r => {
    const tr = document.createElement("tr"); if (r.live) tr.className = "live";
    // a request that did not simply stop is marked on its id, with the engine's reason as the tooltip
    const odd = r.finish && r.finish !== "stop" ? (r.finish === "error" ? "err" : "warn") : "";
    const id = `${r.id || "—"}${odd ? ` · ${r.finish}` : ""}`;
    [[clock(r.t * 1000).slice(0, 8), ""], [id, "f " + odd], [r.path || "—", ""], [num(r.out) ? fmt(r.out) : "—", "n"],
     [num(r.tps) ? r.tps.toFixed(1) : "—", r.live ? "n f live" : "n f"], [num(r.dur) ? `${r.dur.toFixed(1)} s` : "—", "n"]].forEach(([t, c]) => {
      const td = document.createElement("td"); td.textContent = t; if (c) td.className = c.trim(); tr.append(td);
    });
    tb.append(tr);
  });
  $("#req-empty").hidden = rows.length > 0;
  PANE.req = String(total); text("#req-count", PANE.req); if (curPane() === "req") text("#pane-count", PANE.req);
}

function drawHardware(eng, st, hw, gpus, names, sh) {
  const meter = f => `<div class="meter"><i style="width:${(clamp(f || 0, 0, 1) * 100).toFixed(1)}%"></i></div>`;
  const val = (x, unit) => num(x) ? `<div class="dv">${Math.round(x)}${unit ? `<small>${unit}</small>` : ""}</div>` : `<div class="dv none">—</div>`;
  const mem = (used, total, share) => {
    const ok = num(used) && num(total) && total > 0;
    return `<div class="dm"><span>${ok ? `${gib(used)} / ${gib(total)} GiB` : "—"}</span><i>${ok ? `${(used / total * 100).toFixed(0)}%` : ""}</i>${meter(ok ? used / total : 0)}
      <div class="lk">${meter(share)}<i>${num(share) ? `${pct(share)} of lookups` : "No lookups yet"}</i></div></div>`;
  };
  const experts = (slots, mib, note) => num(mib) && mib > 0
    ? `<div class="dx"><b>${num(slots) ? `${fmt(slots)} slots` : note}</b><span>${(mib / 1024).toFixed(1)} GiB${num(slots) ? "" : " pinned"}</span></div>`
    : `<div class="dv none">—</div>`;
  const peerSlots = (eng.expert_slots || 0) - (eng.expert_slots_primary || 0), peerMib = (eng.expert_cache_mib || 0) - (eng.expert_cache_primary_mib || 0);
  const tier = [[eng.expert_slots_primary, eng.expert_cache_primary_mib, sh && sh[0]], [peerSlots, peerMib, sh && sh[1]]];
  const label = [...gpus.map((g, i) => [names[g.index] || `CUDA${g.index}`, `CUDA${g.index} · ${i === 0 ? "Primary" : "Peer experts"}`]), ["System RAM", "Host · CPU load"]];
  const row = cells => `<div class="dl"><div class="dn"><b></b><span></span></div>${cells}</div>`;
  const rows = gpus.map((g, i) => row(
      mem(g.mem_used, g.mem_total, tier[i] && tier[i][2]) + val(g.util, "%")
      + (num(g.power) ? `<div class="dv">${Math.round(g.power)}<small>/ ${num(g.power_limit) ? Math.round(g.power_limit) : "—"} W</small></div>` : val(null))
      + val(g.temp, "°C")
      + (num(g.pcie_gen) ? `<div class="dv">Gen${g.pcie_gen}<small>x${g.pcie_width}</small></div>` : val(null))
      + (tier[i] ? experts(tier[i][0], tier[i][1]) : val(null))));
  rows.push(row(mem(hw.ram_used, hw.ram_total, sh && sh[2]) + val(hw.cpu, "%")
      + val(null) + val(null) + val(null) + experts(null, eng.arena_mib, "Remainder")));
  $("#h-dev").innerHTML = `<div class="dl head"><span>Device</span><span>Memory · routed lookups</span><span>Load</span><span>Power</span><span>Temperature</span><span>PCIe</span><span>Experts</span></div>` + rows.join("");
  // device names come from the driver, so they are set as text
  $$("#h-dev .dl:not(.head) .dn").forEach((d, i) => { d.querySelector("b").textContent = label[i][0]; d.querySelector("span").textContent = label[i][1]; });

  text("#h-cpu", st.cpu_name || "—");
  text("#h-cores", st.cores && st.threads ? `${st.cores} cores · ${st.threads} threads` : num(st.threads) ? `${st.threads} threads` : "—");
  text("#h-load", num(hw.cpu) ? `${Math.round(hw.cpu)}%` : "—");
  text("#h-driver", st.driver ? `${st.driver}${st.cuda ? ` · CUDA ${st.cuda}` : ""}` : "—");
  text("#h-model", eng.model || "—");
  text("#h-ctx", num(eng.max_context) ? `${fmt(eng.max_context)} tokens${eng.kv ? ` · kv ${eng.kv}` : ""}` : "—");
  text("#h-draft", num(eng.spec) ? (eng.spec > 0 ? `Up to ${eng.spec} tokens` : "Off") : "—");
  text("#h-build", eng.version || "—");
}

const PANE = { req: "0", log: "0 lines" };
let paneOn = "req";
// tabbed (several slots): one panel shows; split (one slot): both show, stacked
function pane(p = paneOn) {
  paneOn = p;
  const split = $(".col-r").classList.contains("single");
  $$("#reqs .tab").forEach(t => t.setAttribute("aria-selected", String(t.dataset.pane === p)));
  $("#pane-req").hidden = !split && p !== "req"; $("#pane-log").hidden = !split && p !== "log";
  text("#pane-count", PANE[p]);
  scrollLog();
}
$$("#reqs .tab").forEach(t => t.onclick = () => pane(t.dataset.pane));
const curPane = () => paneOn;

/* ---- log ---- */
const logs = [];
const F = { lvl: "all", q: "", follow: true };
const logEl = $("#log");
const MAXDOM = matchMedia("(min-width: 981px)").matches ? 12 : 14;   // a desktop log shows the last few lines
const passes = e => (F.lvl === "all" || (F.lvl === "warn" && e.lvl !== "info") || (F.lvl === "err" && e.lvl === "err"))
  && (!F.q || e.msg.toLowerCase().includes(F.q.toLowerCase()));

const NUM = /(?<![\w.])\d[\d,]*(?:\.\d+)?(?:\s?(?:tok\/s|tok|ms|MiB|GiB|MB\/s|%|s|W|slots))?(?![\w])/g;
function numbers(str, into) {
  let i = 0, m; NUM.lastIndex = 0;
  while ((m = NUM.exec(str))) {
    if (m.index > i) into.append(str.slice(i, m.index));
    const s = document.createElement("span"); s.className = "n"; s.textContent = m[0]; into.append(s);
    i = m.index + m[0].length;
  }
  if (i < str.length) into.append(str.slice(i));
}
function lineEl(e) {
  const d = document.createElement("div"); d.className = `ln ${e.lvl}`;
  const mk = (c, t) => { const s = document.createElement("span"); s.className = c; s.textContent = t; return s; };
  const msg = mk("msg", ""), q = F.q.toLowerCase(), low = e.msg.toLowerCase();
  let from = 0, at;
  while (q && (at = low.indexOf(q, from)) !== -1) {
    numbers(e.msg.slice(from, at), msg);
    const k = document.createElement("mark"); k.textContent = e.msg.slice(at, at + q.length); msg.append(k);
    from = at + q.length;
  }
  numbers(e.msg.slice(from), msg);
  d.append(mk("t", clock(e.t * 1000)), mk("s", e.src), mk("lv", e.lvl === "info" ? "INF" : e.lvl === "warn" ? "WRN" : "ERR"), msg);
  return d;
}
const counts = () => { PANE.log = `${fmt(logs.length)} lines`; text("#log-count", PANE.log); if (curPane() === "log") text("#pane-count", PANE.log); };
function rebuild() {
  const out = [];
  for (let i = logs.length - 1; i >= 0 && out.length < MAXDOM; i--) if (passes(logs[i])) out.push(logs[i]);
  logEl.replaceChildren(...out.reverse().map(lineEl));
  if (!out.length) { const d = document.createElement("div"); d.className = "empty"; d.textContent = logs.length ? "No lines match the filter." : "No log lines yet."; logEl.append(d); }
  counts(); scrollLog();
}
function append(lines) {
  logs.push(...lines); if (logs.length > 3000) logs.splice(0, logs.length - 3000);
  const shown = lines.filter(passes);
  if (shown.length) {
    logEl.querySelector(".empty")?.remove();
    logEl.append(...shown.map(lineEl));
    while (logEl.childElementCount > MAXDOM) logEl.firstChild.remove();
  }
  counts(); if (F.follow) scrollLog();
}
function scrollLog() { logEl.scrollTop = logEl.scrollHeight; }
logEl.addEventListener("scroll", () => {
  const near = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 12;
  if (!near && F.follow) setFollow(false);
  if (near && !F.follow) setFollow(true);
});
function setFollow(on) { F.follow = on; $("#b-follow").setAttribute("aria-pressed", String(on)); $("#jump").hidden = on; if (on) scrollLog(); }
$("#jump").onclick = () => setFollow(true);
$("#b-follow").onclick = () => setFollow(!F.follow);
$$("#lvl button").forEach(b => b.onclick = () => {
  F.lvl = b.dataset.l; $$("#lvl button").forEach(x => x.setAttribute("aria-pressed", String(x === b))); rebuild();
});
$("#q").oninput = e => { F.q = e.target.value.trim(); rebuild(); };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") { if (e.key === "Escape") e.target.blur(); return; }
  if (e.key === "/") { e.preventDefault(); view("stats"); pane("log"); $("#q").focus(); }
  else if (e.key === "f") setFollow(!F.follow);
});

/* ---- slots and queue (only with more than one slot) ---- */
const API = { openai: "OpenAI", anthropic: "Anthropic" };
const kfmt = n => n >= 10000 ? (n / 1000).toFixed(n >= 100000 ? 0 : 1) + "k" : fmt(n);
const STATE = { idle: "Idle", prefill: "Prefill", decoding: "Decoding" };
const cell = (cls, text) => { const e = document.createElement("span"); e.className = cls; if (text != null) e.textContent = text; return e; };

function drawSlots(M, multi) {
  $("#slots").hidden = $("#queue").hidden = !multi;
  const col = $(".col-r"); if (col.classList.contains("single") === multi) { col.classList.toggle("single", !multi); pane(); }
  if (!multi) return;
  const sl = M.slots || [], host = $("#lanes"), live = M.live, queue = live.queue || [];
  if (host.dataset.n !== String(sl.length)) {
    host.dataset.n = String(sl.length);
    const head = document.createElement("div"); head.className = "lane head";
    ["Slot", "State", "Tokens/s", "Output", "Context", "Accept", "Request"].forEach(t => head.append(cell(t === "Accept" ? "n" : "", t)));
    host.replaceChildren(head, ...sl.map(s => {
      const d = document.createElement("div"); d.className = "lane"; d.dataset.s = "idle";
      const ctxc = cell("l-ctx"), meter = cell("meter"); meter.append(document.createElement("i")); ctxc.append(meter, cell(""));
      d.append(cell("l-id", `Slot ${s.id}`), cell("l-st"), cell("l-tps"), cell("l-tok"), ctxc, cell("l-acc"), cell("l-req"));
      return d;
    }));
  }
  const rows = host.querySelectorAll(".lane:not(.head)");
  sl.forEach((s, i) => {
    const el = rows[i]; if (!el) return;
    const q = c => el.querySelector(c);
    el.dataset.s = s.state;
    const pre = s.state === "prefill" && num(s.prefill_done) && s.prompt_tokens;
    text2(q(".l-st"), (STATE[s.state] || s.state) + (pre ? ` ${Math.min(100, Math.round(s.prefill_done / s.prompt_tokens * 100))}%` : ""));
    q(".l-st").title = pre ? `${kfmt(s.prefill_done)} of ${kfmt(s.prompt_tokens)} prompt tokens read` : "";
    const tps = q(".l-tps"); tps.replaceChildren();
    if (s.state === "decoding") tps.append(fmt(s.tok_s), Object.assign(document.createElement("small"), { textContent: "tok/s" }));
    else tps.append("—");
    text2(q(".l-tok"), s.is_processing ? `${fmt(s.generated)} tokens` : "—");
    const f = s.n_ctx ? s.ctx_used / s.n_ctx : 0, m = q(".meter");
    m.firstChild.style.width = clamp(f, 0, 1) * 100 + "%"; m.classList.toggle("hot", f > .8);
    text2(q(".l-ctx").lastChild, `${kfmt(s.ctx_used || 0)} / ${kfmt(s.n_ctx || 0)}`);
    text2(q(".l-acc"), s.drafts_offered > 20 ? `${(s.drafts_accepted / s.drafts_offered * 100).toFixed(0)}%` : "—");
    const rq = q(".l-req"); rq.replaceChildren();
    if (s.is_processing) { const b = document.createElement("b"); b.textContent = API[s.api] || s.api || "—"; rq.append(b, ` · ${s.model || "—"}`); }
    else rq.append("—");
  });
  text("#slot-count", `${(live.slots || {}).active || 0} of ${sl.length} busy`);
  text("#slot-note", sl.length ? `${kfmt(sl[0].n_ctx)} context each` : "");

  text("#q-count", String(live.queued || 0));
  const tb = $("#t-queue"); tb.replaceChildren();
  queue.slice(0, 3).forEach(r => {
    const tr = document.createElement("tr");
    [[`${(r.waited_s || 0).toFixed(1)} s`, "f"], [API[r.api] || r.api || "—", ""], [r.model || "—", ""], [num(r.prompt_tokens) ? fmt(r.prompt_tokens) : "—", "n"]].forEach(([t, c]) => {
      const td = document.createElement("td"); td.textContent = t; if (c) td.className = c; tr.append(td);
    });
    tb.append(tr);
  });
  const more = (live.queued || 0) - Math.min(3, queue.length);
  if (more > 0) { const tr = document.createElement("tr"), td = document.createElement("td"); td.colSpan = 4; td.textContent = `+${more} more`; tr.append(td); tb.append(tr); }
  $("#q-empty").hidden = (live.queued || 0) > 0;
  text2($("#q-empty"), "No request is waiting.");
}
const text2 = (e, v) => { if (e.textContent !== v) e.textContent = v; };

/* ---- polling ---- */
let lastLog = 0;
async function pollMetrics() {
  try {
    const m = await get("metrics");
    m.slots = m.live && m.live.slots && m.live.slots.total > 1 ? await get("slots").catch(() => []) : [];
    const dec = m.slots.filter(s => s.state === "decoding").length;
    perStream = perStream.slice(1).concat(dec ? (m.live.tok_s || 0) / dec : 0);
    M = m; $("#auth").hidden = true; draw();
  } catch (e) {
    $("#auth").hidden = !e.locked; M = null;
    setState("offline", e.locked ? "Locked" : "Offline");
  }
  setTimeout(pollMetrics, 1000);
}
async function pollLog() {
  try {
    const r = await get(`api/logs?after=${lastLog}`);
    if (r.last < lastLog) { logs.length = 0; lastLog = 0; best = {}; saveBest(); rebuild(); }       // the server restarted
    else if (r.lines.length) { lastLog = r.last; append(r.lines); }
    else if (!logs.length && !logEl.firstChild) rebuild();
  } catch (e) { /* the metrics poll reports the connection */ }
  setTimeout(pollLog, 1000);
}
$("#auth").onsubmit = e => {
  e.preventDefault(); apiKey = $("#key").value.trim(); sessionStorage.setItem("basalt.key", apiKey);
};

$("#host").textContent = location.host;
fetch("health").then(r => r.json()).then(h => { $("#keyed").hidden = !h.api_key; }).catch(() => {});
view(location.hash === "#hardware" ? "hardware" : "stats");
pollMetrics(); pollLog();
