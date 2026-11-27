"use strict";
/* The request monitor: recent requests read as transcripts (the prompt, the thinking, the answer), from /api/monitor. */
const $ = s => document.querySelector(s);
const VIEWS = [["read", "Read"], ["tools", "Tools"], ["input", "Request"], ["response", "Response"]];
const FOLD = 900;                      // turns longer than this many characters start folded
let apiKey = sessionStorage.getItem("basalt.key") || "";
let picked = false;                    // the newest request opens once, on a wide screen
let records = [], capacity = 0, inFlight = 0, selected = null, detail = null, view = "read", filter = "all";
const unfolded = new Set();            // "id:turn" of folded turns the reader opened

const live = r => r.state === "queued" || r.state === "generating";
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const int = n => typeof n === "number" ? Math.round(n).toLocaleString("en-US") : null;
const sec = s => typeof s !== "number" ? null : s < 60 ? `${s.toFixed(2)} s` : `${Math.floor(s / 60)} m ${String(Math.round(s % 60)).padStart(2, "0")} s`;
const ms = s => typeof s !== "number" ? null : s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(2)} s`;
const clock = t => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
const wall = r => r.wallclock_s ?? Date.now() / 1000 - r.started_at;
const set = (el, html) => { if (el._h !== html) { el.innerHTML = html; el._h = html; } };
const headers = () => apiKey ? { Authorization: `Bearer ${apiKey}` } : {};
const phone = () => matchMedia("(max-width: 760px)").matches;

/* token counts under either API's names */
function counts(r) {
  const u = r.usage || {}, t = r.timings || {};
  return {
    prompt: u.prompt_tokens ?? u.input_tokens,
    cached: t.cache_n || u.prompt_tokens_details?.cached_tokens || u.cache_read_input_tokens || 0,
    out: u.completion_tokens ?? u.output_tokens,
    thinking: u.completion_tokens_details?.reasoning_tokens,
    rate: t.predicted_per_second,
    read: typeof t.prompt_ms === "number" ? t.prompt_ms / 1000 : null,
  };
}

async function get(path) {
  const r = await fetch(path, { cache: "no-store", headers: headers() });
  if (r.status === 401) { $("#auth").hidden = false; throw new Error("This server needs an API key"); }
  const body = await r.json();
  if (!r.ok) throw new Error(body.error?.message ?? `HTTP ${r.status}`);
  $("#auth").hidden = true;
  return body;
}

/* ---------- the list ---------- */
function shown() {
  const q = $("#search").value.trim().toLowerCase();
  return records.filter(r => `${r.id} ${r.path} ${r.model}`.toLowerCase().includes(q)
    && (filter === "all" || (filter === "active" ? live(r) : r.state === "error")));
}

function drawList() {
  $("#n-all").textContent = records.length || "";
  $("#n-active").textContent = records.filter(live).length || "";
  $("#n-error").textContent = records.filter(r => r.state === "error").length || "";
  const rows = $("#rows"), list = shown();
  if (!list.length) {
    set(rows, `<p class="mon-none">${records.length ? "No request matches." : "No requests yet. Calls to the API show up here as they happen."}</p>`);
    return;
  }
  set(rows, list.map(r => {
    const c = counts(r), right = r.state === "completed" ? esc(sec(wall(r))) : `<span class="state" data-s="${r.state}">${r.state === "error" ? `${r.http_status ?? ""} error` : r.state}</span>`;
    return `<button type="button" class="mon-row" role="option" data-id="${r.id}" aria-selected="${r.id === selected}">
      <span class="ep">${esc(r.path)}</span><span class="r">${right}</span>
      <span class="meta">${[clock(r.started_at), r.state === "error" ? `HTTP ${r.http_status ?? "–"}` : c.rate ? `${Math.round(c.rate)} tok/s` : ""].filter(Boolean).join(" · ")}</span></button>`;
  }).join(""));
}

/* ---------- the transcript ---------- */
function plain(c) {
  if (typeof c === "string") return c;
  if (c == null) return "";
  if (Array.isArray(c)) return c.map(p => p?.type === "text" || p?.type === "input_text" || p?.type === "output_text" ? p.text
    : p?.type === "tool_result" || p?.type === "function_call_output" ? plain(p.content ?? p.output)
    : p?.type === "tool_use" ? `${p.name}(${JSON.stringify(p.input, null, 2)})`
    : /image/.test(p?.type || "") ? "[image]" : JSON.stringify(p, null, 2)).join("\n\n");
  return JSON.stringify(c, null, 2);
}

/* a tool call's arguments as text: JSON is laid out, anything else (or a cut-off stream) stays as sent */
function args(raw) {
  if (typeof raw !== "string") return plain(raw);
  try { return JSON.stringify(JSON.parse(raw), null, 2); } catch (err) { return raw; }
}
const call = (name, raw) => [name || "tool", args(raw)].filter(Boolean).join("\n");

/* the conversation the client sent, as [role, text, asked, kind]; asked marks a plain-text question; kind "tool" is a call or a result */
function turns(d) {
  let req;
  try { req = JSON.parse(d.input); } catch (err) { return null; }
  const out = [];
  const tool = (role, text) => out.push([role, text, false, "tool"]);
  if (req.system) out.push(["system", plain(req.system)]);
  if (req.instructions) out.push(["system", plain(req.instructions)]);
  const msgs = Array.isArray(req.messages) ? req.messages : typeof req.input === "string" ? [{ role: "user", content: req.input }] : Array.isArray(req.input) ? req.input : [];
  if (req.messages != null && !Array.isArray(req.messages)) out.push(["messages", plain(req.messages)]);
  for (const m of msgs) {
    if (!m || typeof m !== "object") { out.push(["message", plain(m)]); continue; }
    if (m.type === "function_call") { tool("Tool call", call(m.name, m.arguments)); continue; }
    if (m.type === "function_call_output") { tool("Tool result", plain(m.output)); continue; }
    if (m.role === "tool") { tool("Tool result", plain(m.content)); continue; }
    const role = m.role || m.type || "message";
    if (Array.isArray(m.content)) {                    // Anthropic and Responses blocks: text stays one turn, each tool block its own
      let text = [];
      const flush = () => { if (text.length) out.push([role, text.join("\n\n")]); text = []; };
      for (const p of m.content) {
        if (p?.type === "tool_use") { flush(); tool("Tool call", call(p.name, p.input)); }
        else if (p?.type === "tool_result") { flush(); tool("Tool result", plain(p.content)); }
        else text.push(plain([p]));
      }
      flush();
    } else out.push([role, plain(m.content ?? m.output ?? (m.type ? m : "")), m.role === "user" && typeof m.content === "string"]);
    for (const t of m.tool_calls || []) tool("Tool call", call(t.function?.name, t.function?.arguments));
  }
  if (typeof req.prompt === "string") out.push(["prompt", req.prompt, true]);
  return out.filter(([, text]) => text);
}

function turn(role, text, key, opts = {}) {
  const fold = !opts.keep && text.length > FOLD && !unfolded.has(key);
  return `<div class="mon-turn"><div class="role${opts.cls ? " " + opts.cls : ""}">${esc(role)}${opts.sub ? `<small>${esc(opts.sub)}</small>` : ""}</div>
    <div><div class="txt${fold ? " fold" : ""}">${esc(text)}${opts.caret ? '<span class="mon-caret"></span>' : ""}</div>
    ${fold ? `<button type="button" class="more" data-open="${key}">Show all ${int(text.length)} characters</button>` : ""}</div></div>`;
}

function readView(d) {
  const t = turns(d), c = counts(d), generating = d.state === "generating";
  let h = "";
  if (!t) h += `<p class="mon-note">The request body was cut short, so it can't be read as a conversation; see Request.</p>`;
  else {
    const lastAsk = t.length - 1;
    t.forEach(([role, text, asked, kind], i) => { h += turn(role, text, `${d.id}:${i}`, { keep: i === lastAsk, cls: kind === "tool" ? "tool" : "" }); });
  }
  if (d.error) h += turn("error", d.error.message, `${d.id}:err`, { cls: "err", keep: true });
  const answering = generating && !!d.output;
  if (d.reasoning) h += turn("thinking", d.reasoning, `${d.id}:think`, { cls: generating && !d.output ? "hot" : "", caret: generating && !d.output, keep: generating && !d.output,
    sub: c.thinking ? `${int(c.thinking)} tok` : "" });
  if (d.output) {
    let out = d.output;
    if (!generating) { try { out = JSON.stringify(JSON.parse(out), null, 2); } catch (err) { /* plain text */ } }
    h += turn("answer", out, `${d.id}:out`, { cls: answering ? "hot" : "", caret: answering, keep: true });
  }
  const calls = d.tool_calls || [];
  calls.forEach((c, i) => {
    const last = i === calls.length - 1 && generating && !d.output;
    h += turn("Tool call", call(c.name, c.arguments), `${d.id}:call${i}`, { cls: "tool" + (last ? " hot" : ""), caret: last, keep: last, sub: c.id });
  });
  if (!d.output && !calls.length && live(d) && !d.reasoning) h += `<div class="mon-turn"><div class="role hot">${d.state}</div><p class="mon-note">Waiting for the first token.</p></div>`;
  else if (!d.output && !calls.length && !live(d) && !d.error) h += `<div class="mon-turn"><div class="role">answer</div><p class="mon-note">No text recorded.</p></div>`;
  if (d.output_truncated || d.reasoning_truncated || d.tool_calls_truncated) h += `<p class="mon-note">Cut short in this view; the API's reply was not.</p>`;
  return `<div class="mon-tx">${h}</div>`;
}

/* the tools the request offered, parsed from their schemas */
function toolsView(d) {
  const tools = d.tools || [];
  if (!tools.length) return `<p class="mon-note">This request offered no tools.</p>`;
  return `<div class="mon-tools">${tools.map(t => `<section><h3>${esc(t.name)}</h3>${t.description ? `<p>${esc(t.description)}</p>` : ""}
    ${t.params.length ? `<dl>${t.params.map(p => `<div><dt>${esc(p.name)}</dt><dd><span>${esc([p.type, p.required ? "required" : "optional"].filter(Boolean).join(" · "))}</span>${p.description ? esc(p.description) : ""}</dd></div>`).join("")}</dl>` : `<p class="none">No parameters.</p>`}</section>`).join("")}</div>`;
}

function rawView(d) {
  if (view === "input") return `<p class="mon-note">The body as it arrived${d.input_truncated ? ", cut short in this view" : ""}.</p><pre class="mon-raw">${esc(d.input)}</pre>`;
  if (!d.response) return `<p class="mon-note">${d.stream ? "Streamed: the reply went out as events and has no single body. The text is under Read." : "Written when the request finishes."}</p>`;
  let body = d.response;
  try { body = JSON.stringify(JSON.parse(body), null, 2); } catch (err) { /* as sent */ }
  return `<p class="mon-note">The JSON body the server replied with${d.response_truncated ? ", cut short in this view" : ""}.</p><pre class="mon-raw">${esc(body)}</pre>`;
}

function colophon(d) {
  const c = counts(d), parts = [];
  if (d.state === "error") return `Refused with HTTP <b>${d.http_status ?? "–"}</b> after <b>${esc(ms(wall(d)))}</b>.`;
  parts.push(live(d) ? `Running for <b>${esc(sec(wall(d)))}</b>.` : `Answered in <b>${esc(sec(wall(d)))}</b>.`);
  if (d.first_token_s != null) parts.push(`The first token came after <b>${esc(ms(d.first_token_s))}</b>${c.read != null ? `, <b>${esc(ms(c.read))}</b> of it reading the prompt` : ""}${c.cached ? ` with <b>${int(c.cached)}</b> tokens from cache` : ""}.`);
  if (c.out != null && c.rate) parts.push(`Then <b>${int(c.out)}</b> tokens at <b>${Math.round(c.rate)} tok/s</b>.`);
  if (!live(d)) parts.push(`${d.stream ? "Streamed" : "Sent as one JSON body"}, HTTP <b>${d.http_status ?? "–"}</b>.`);
  return parts.join(" ");
}

function side(d) {
  const c = counts(d), row = (k, v, cls = "") => `<div><dt>${k}</dt><dd class="${cls}">${v ?? "–"}</dd></div>`;
  return row("First token", esc(ms(d.first_token_s) ?? "–"))
    + row("Decode", c.rate ? `${Math.round(c.rate)}<small>tok/s</small>` : "–", live(d) && c.rate ? "hot" : "")
    + row("Prompt", c.prompt != null ? `${int(c.prompt)}${c.cached ? `<small>${int(c.cached)} cached</small>` : ""}` : "–")
    + row("Output", c.out != null ? `${int(c.out)}${c.thinking ? `<small>${int(c.thinking)} thinking</small>` : ""}` : "–")
    + row("Wall clock", esc(sec(wall(d))))
    + row("Status", d.http_status ?? (live(d) ? "pending" : "–"), d.state === "error" ? "err" : "")
    + row("Format", esc(d.response_format || "text"))
    + (d.tools?.length ? row("Tools", `${esc(d.tools.map(t => t.name).join(", "))} (${d.tools.length})`) : "")
    + row("Request", esc(d.id));
}

function copyable(d) { return view === "input" ? d.input : view === "response" ? d.response : d.output || d.reasoning; }

function drawDetail() {
  const d = detail;
  $("#empty").hidden = !!d; $("#sel").hidden = !d;
  if (!d) { set($("#empty"), records.length && !phone() ? "Pick a request to read its prompt, thinking and answer. j and k step through the list." : ""); return; }
  set($("#d-kick"), [`<b>POST ${esc(d.path)}</b>`, esc(d.model), `${new Date(d.started_at * 1000).toLocaleDateString([], { day: "numeric", month: "short" })} ${clock(d.started_at)}`,
    `<span class="state" data-s="${d.state}">${d.state}</span>`].map(s => `<span>${s}</span>`).join(""));
  set($("#mtabs"), VIEWS.map(([k, label]) => `<button type="button" role="tab" data-t="${k}" aria-selected="${k === view}">${label}${k === "tools" && d.tools?.length ? ` (${d.tools.length})` : ""}</button>`).join(""));
  set($("#d-body"), view === "read" ? readView(d) : view === "tools" ? toolsView(d) : rawView(d));
  set($("#d-colo"), colophon(d));
  set($("#d-side"), side(d));
}

function drawFoot(err) {
  $("#foot").textContent = err ? err.message : `${records.length} of ${capacity} kept in memory until the server restarts${inFlight ? ` · ${inFlight} in flight` : ""}`;
}

/* ---------- data ---------- */
async function pull(id) {
  try { detail = await get(`api/monitor?id=${encodeURIComponent(id)}`); } catch (err) { detail = null; selected = null; }
  if (detail && detail.id !== selected) return;          // the reader moved on while this was in flight
  drawDetail(); drawList();
}

let timer = 0;
async function refresh() {
  clearTimeout(timer);
  try {
    const m = await get("api/monitor");
    records = m.requests; capacity = m.capacity; inFlight = m.in_flight;
    if (selected && !records.some(r => r.id === selected)) { selected = null; detail = null; }
    if (!picked && records.length) { picked = true; if (!phone()) selected = records[0].id; }
    drawList(); drawFoot();
    if (selected && (!detail || live(detail))) await pull(selected); else drawDetail();
  } catch (err) { drawFoot(err); }
  timer = setTimeout(refresh, records.some(live) ? 700 : 2000);
}

function select(id, open) {
  if (id !== selected) { selected = id; detail = null; drawDetail(); pull(id); }
  drawList();
  $(`.mon-row[data-id="${id}"]`)?.scrollIntoView({ block: "nearest" });
  if (open) { $("#read").classList.add("open"); $("#read").scrollTop = 0; }
}
function move(step) {
  const list = shown(); if (!list.length) return;
  const i = list.findIndex(r => r.id === selected);
  select(list[Math.max(0, Math.min(list.length - 1, i < 0 ? 0 : i + step))].id, false);
}

async function copy(b, text) {
  let ok = false;
  try { if (navigator.clipboard && isSecureContext) { await navigator.clipboard.writeText(text); ok = true; } } catch (err) { /* below */ }
  if (!ok) {                                       // plain http on a LAN address has no clipboard API
    const ta = document.createElement("textarea"); ta.value = text; ta.style.cssText = "position:fixed;opacity:0"; document.body.append(ta); ta.select();
    try { ok = document.execCommand("copy"); } catch (err) { ok = false; }
    ta.remove();
  }
  b.textContent = ok ? "Copied" : "Select the text to copy";
  setTimeout(() => { b.textContent = "Copy"; }, 1500);
}

$("#rows").addEventListener("click", e => { const b = e.target.closest(".mon-row"); if (b) select(b.dataset.id, true); });
$("#mtabs").addEventListener("click", e => { const b = e.target.closest("button[data-t]"); if (b) { view = b.dataset.t; drawDetail(); } });
$("#d-body").addEventListener("click", e => { const b = e.target.closest("[data-open]"); if (b) { unfolded.add(b.dataset.open); drawDetail(); } });
$("#mfilter").addEventListener("click", e => {
  const b = e.target.closest("button[data-f]");
  if (!b) return;
  filter = b.dataset.f;
  document.querySelectorAll("#mfilter button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
  drawList();
});
$("#search").addEventListener("input", drawList);
$("#copy").addEventListener("click", e => { if (detail) copy(e.currentTarget, copyable(detail) || ""); });
$("#clear").addEventListener("click", async () => {
  try {
    const r = await fetch("api/monitor/clear", { method: "POST", headers: { ...headers(), "Content-Type": "application/json" }, body: "{}" });
    if (r.status === 401) { $("#auth").hidden = false; return; }
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
  } catch (err) { drawFoot(err); return; }
  records = []; selected = null; detail = null; unfolded.clear();
  $("#read").classList.remove("open");
  drawList(); drawDetail(); refresh();                 // requests still running come straight back
});
$("#back").addEventListener("click", () => $("#read").classList.remove("open"));
$("#auth").addEventListener("submit", e => {
  e.preventDefault(); apiKey = $("#key").value.trim(); sessionStorage.setItem("basalt.key", apiKey); refresh();
});
document.addEventListener("keydown", e => {
  if (e.target.matches("input")) { if (e.key === "Escape") e.target.blur(); return; }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "j" || e.key === "ArrowDown") { e.preventDefault(); move(1); }
  else if (e.key === "k" || e.key === "ArrowUp") { e.preventDefault(); move(-1); }
  else if (/^[1-4]$/.test(e.key)) { view = VIEWS[+e.key - 1][0]; drawDetail(); }
  else if (e.key === "c" && detail) copy($("#copy"), copyable(detail) || "");
  else if (e.key === "/") { e.preventDefault(); $("#search").focus(); }
  else if (e.key === "Escape") $("#read").classList.remove("open");
});

$("#host").textContent = location.host;
drawDetail();
refresh();
