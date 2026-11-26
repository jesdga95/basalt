"use strict";
const $ = s => document.querySelector(s);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const num = v => typeof v === "number" && Number.isFinite(v);
const fmt = (v, d = 0) => num(v) ? v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }) : "-";
const p2 = n => String(n).padStart(2, "0");
const clock = ms => { const d = new Date(ms); return `${p2(d.getHours())}:${p2(d.getMinutes())}:${p2(d.getSeconds())}`; };
const gib = n => num(n) ? (n / 1024 ** 3).toFixed(1) : "-";
const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ESC[c]);
const drawn = new WeakMap();
const html = (el, h) => { if (drawn.get(el) !== h) { el.innerHTML = h; drawn.set(el, h); } };   // skip unchanged redraws
const text = (el, v) => { if (el.textContent !== v) el.textContent = v; };
const HIST = 60;
const kt = n => n >= 100000 ? `${Math.round(n / 1000)}k` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);   // 41.2k, 262k
const lim = n => n >= 10000 ? kt(n) : fmt(n);   // token limit: 8,192, 41.2k, 262k
const tick = n => n >= 1000 ? `${+(n / 1000).toFixed(1)}k` : String(Math.round(n));
const API = { openai: "OpenAI", anthropic: "Anthropic" };
const shortGpu = s => String(s || "").replace(/^NVIDIA\s+/i, "").replace(/^GeForce\s+/i, "");
const shortCpu = s => String(s || "").replace(/\((R|TM)\)/gi, "").replace(/\s+\d+-Core Processor$/i, "")
  .replace(/\s+CPU\s+@.*$/i, "").replace(/\s+Processor$/i, "").replace(/\s+/g, " ").trim();

let apiKey = sessionStorage.getItem("basalt.key") || "";
let M = null;          // the last combined state snapshot

/* ---- fields the page draws once the server sends them. Each is read only through ahead(); until it arrives, its panel is
   hidden or shows what the server sends now, never a made-up figure. Keyed by where the field lives. ---- */
const PENDING = {
  hardware: ["ram_bw_gbs"],                              // RAM bandwidth, measured while serving
};
const ahead = (o, k) => o == null ? undefined : o[k];

/* ---- series and charts ---- */
// the last HIST samples, padded at the start with null (no line) so the newest sample sits at the right edge
const tail = a => { const v = (a || []).slice(-HIST).map(x => num(x) ? x : null); while (v.length < HIST) v.unshift(null); return v; };
const stats = a => {
  const v = a.filter(num);
  return v.length ? { min: Math.min(...v), max: Math.max(...v), avg: v.reduce((t, x) => t + x, 0) / v.length, last: v[v.length - 1] } : null;
};
const busyStats = a => stats(a.map(x => x > 0 ? x : null));     // over the samples that ran
const STEPS = [10, 20, 50, 100, 120, 150, 160, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000, 3000, 4000, 5000,
  8000, 10000, 16000, 20000, 32000, 50000];
const nice = v => STEPS.find(s => s >= v * 1.15) || Math.ceil(v * 1.15 / 10000) * 10000;
// polyline points in a 0..W x 0..100 box; a null sample breaks the line
function lines(a, y, W = 100) {
  const out = []; let cur = [];
  a.forEach((v, i) => {
    if (v == null) { if (cur.length) out.push(cur); cur = []; return; }
    cur.push(`${(i / (a.length - 1) * W).toFixed(2)},${y(v).toFixed(2)}`);
  });
  if (cur.length) out.push(cur);
  return out.map(c => c.length === 1 ? [c[0], c[0].replace(/^[\d.]+/, x => String(+x + .01))] : c);
}
const poly = (pts, attrs) => `<polyline points="${pts.join(" ")}" fill="none" vector-effect="non-scaling-stroke" stroke-linejoin="round" ${attrs}/>`;
const area = (pts, fill) => `<polygon points="${pts[0].split(",")[0]},100 ${pts.join(" ")} ${pts[pts.length - 1].split(",")[0]},100" fill="${fill}"/>`;
const baseLine = `<line x1="0" y1="100" x2="100" y2="100" stroke="var(--line)" stroke-width="1" vector-effect="non-scaling-stroke"/>`;
let gid = 0;
// a hardware chart: filled line between lo and hi; `limit` draws a dashed line at the top
function sbChart(a, lo, hi, accent, limit) {
  const id = `sbg${++gid}`, tint = accent ? "255,122,61" : "154,164,173";
  const segs = lines(a, v => clamp(100 - (v - lo) / (hi - lo) * 100, 1, 99));
  return `<div class="sb-ch"><svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1">`
    + `<stop offset="0" stop-color="rgba(${tint},.16)"/><stop offset="1" stop-color="rgba(${tint},.02)"/></linearGradient></defs>${baseLine}`
    + (limit ? `<line x1="0" y1="1" x2="100" y2="1" stroke="var(--basalt-500)" stroke-width="1" stroke-dasharray="2 3" vector-effect="non-scaling-stroke"/>` : "")
    + segs.map(p => area(p, `url(#${id})`) + poly(p, `stroke="${accent ? "var(--accent)" : "var(--fg-2)"}" stroke-width="1.5"`)).join("")
    + `</svg></div>`;
}
function spark(a, max) {
  const segs = lines(a, v => 100 - Math.min(v, max) / max * 96 - 2);
  return `<div class="spark"><svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">${baseLine}`
    + segs.map(p => area(p, "rgba(154,164,173,.08)") + poly(p, `stroke="var(--fg-2)" stroke-width="1.4"`)).join("") + `</svg></div>`;
}
// a hero chart: one 60 s line with its fill, and the top of the axis as the only tick
// `shown`: the number above the chart, so an idle chart's axis is not smaller than its headline
function miniChart(a, color, fill, shown) {
  const max = nice(Math.max(40, num(shown) ? shown : 0, ...a.filter(num)));
  const y = v => 100 - 3 - Math.min(v, max) / max * 90;
  return `<svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">${baseLine}`
    + lines(a, y).map(p => area(p, fill) + poly(p, `stroke="${color}" stroke-width="1.6"`)).join("") + `</svg><span class="tk">${tick(max)}</span>`;
}

/* ---- top bar ---- */
function setState(s, label) { $("#state").dataset.s = s; text($("#state"), label); }

/* ---- session figures ---- */
const big = n => !num(n) ? "-" : n >= 1e6 ? `${+(n / 1e6).toFixed(1)}M` : kt(n);
const dur = ms => !num(ms) || ms <= 0 ? "-" : ms < 60000 ? `${Math.round(ms / 1000)} s` : ms < 3600000 ? `${Math.round(ms / 60000)} min` : `${(ms / 3600000).toFixed(1)} h`;
const uptime = () => M && M.totals && num(M.totals.since) ? dur(Date.now() - M.totals.since * 1000) : "-";
// the state word, annotated by the phase: exactly one running request (with several, one request's phase would stand for all)
function stateWord(live, name) {
  const p = live.phase;
  if (live.state !== "generating" || (live.slots || {}).active !== 1 || !p) return [name, ""];
  if (p === "thinking") return ["Thinking", ""];
  if (p === "answering") return ["Answering", ""];
  if (p.startsWith("writing a tool call") || p === "tool call complete") return ["Tool call", p.replace(/^writing a tool call: /, "Tool call: ")];
  return [name, ""];
}
// the conversation cache, only when something is cached and a model is up
const cached = (conv, live) => !["unloaded", "asleep", "loading"].includes(live.state) && conv && num(conv.parked) && conv.parked > 0;
const usd = v => !num(v) || v === 0 ? "$0" : v < 0.01 ? `$${v.toFixed(4)}` : v < 1 ? `$${v.toFixed(3)}` : v < 100 ? `$${v.toFixed(2)}` : `$${fmt(v, 2)}`;
function drawTotals(t) {
  const n = t && num(t.requests) ? t : null;
  const cost = n && num(n.cost_usd) && n.cost_usd > 0 ? `<b>${usd(n.cost_usd)}</b> · ` : "";
  html($("#totals"), n ? `${cost}<b>${fmt(n.requests)}</b> requests · <b>${big(n.prompt)}</b> prompt · <b>${big(n.written)}</b> output` : "");
  const tip = n ? `Since ${new Date((n.since || 0) * 1000).toLocaleDateString()}: ${fmt(n.prompt)} prompt tokens (${fmt(n.reused)} reused), ${dur(n.prompt_ms)} reading, ${dur(n.decode_ms)} generating` : "";
  const ci = n ? (n.cost_input_per_million || 0) : 0;
  const pricing = n && (ci || n.cost_output_per_million) ? `\nPricing: $${ci}/M in · $${n.cost_output_per_million}/M out` +
    ((n.cost_cache_write_per_million != null && n.cost_cache_write_per_million !== ci) || (n.cost_cache_hit_per_million != null && n.cost_cache_hit_per_million !== ci) ?
      ` · $${n.cost_cache_write_per_million ?? ci}/M write · $${n.cost_cache_hit_per_million ?? ci}/M hit` : "") : "";
  $("#totals").title = (tip + pricing) || "Pricing and totals";
  lastTotals = t;
  $("#totals").onclick = () => openPricing();
}

/* ---- the heat scale: 0..400 tok/s and up; only the colour of the live decode number uses it ---- */
const HEAT = [[0, [42, 51, 60]], [.30, [90, 42, 30]], [.48, [184, 64, 26]], [.64, [242, 102, 43]], [.82, [255, 181, 71]], [1, [255, 241, 214]]];
function heatAt(t) {
  t = clamp(t, 0, 1);
  const i = HEAT.findIndex(s => s[0] >= t), b = HEAT[Math.max(1, i)], a = HEAT[Math.max(0, i - 1)];
  const f = b[0] === a[0] ? 0 : (t - a[0]) / (b[0] - a[0]);
  return `rgb(${a[1].map((c, k) => Math.round(c + (b[1][k] - c) * f)).join(",")})`;
}
const heatTop = (best, v) => Math.max(400, Math.ceil((Math.max(best || 0, v || 0) * 1.1) / 100) * 100);

/* ---- prompt reads: a request that read at least MIN_READ new tokens has a read rate worth showing; the headline, the best and
   its fallback all use that rule ---- */
const MIN_READ = 512;
const readOf = x => (x.prompt_tokens || 0) - (x.reused || 0);
const realRead = x => x.prompt_ms > 0 && readOf(x) >= MIN_READ;
const readRate = x => readOf(x) / (x.prompt_ms / 1000);
// the server's best prefill; without one, the fastest real read among the listed requests
function bestRead(best, reqs) {
  if (best.prefill) return best.prefill;
  const top = reqs.filter(realRead).sort((a, b) => readRate(b) - readRate(a))[0];
  return top ? { tok_s: readRate(top), id: top.id, read_tokens: readOf(top), time: top.time, recent: true } : null;
}

/* ---- Speed: each figure is a number and one 60 s chart. Best, the cache, the last prompt and the prefix hit are in the number's
   tooltip and in Requests; the per-slot lines are in Slots ---- */
function drawHero(live, hist, reqs, best, slots, total) {
  const generating = live.state === "generating", reading = live.state === "reading";
  // the live rate while generating, else the newest finished request's that has one
  const r = reqs[0], rd = reqs.find(x => num(x.decode_tok_s)), last = rd ? rd.decode_tok_s : null;
  const liveDec = generating && num(live.tok_s), dec = liveDec ? live.tok_s : last;
  html($("#dec-k"), `Decode${!liveDec && last != null ? `<span>last request</span>` : total > 1 && liveDec ? `<span>all slots</span>` : ""}`);
  html($("#dec-v"), num(dec) ? `${dec.toFixed(1)}<small>tok/s</small>` : "-");
  $("#dec-v").classList.toggle("idle", !liveDec);
  $("#dec-v").style.color = liveDec ? heatAt(Math.max(.56, clamp(dec / heatTop(best.decode && best.decode.tok_s, dec), 0, 1))) : "";
  html($("#dec-chart"), miniChart(tail(hist.tok_s), liveDec ? "var(--accent)" : "var(--fg-2)", liveDec ? "rgba(255,122,61,.14)" : "rgba(154,164,173,.08)", dec));
  const bd = best.decode, bp = bestRead(best, reqs), conv = M.conversation;
  const act = slots.filter(x => x.is_processing).length;
  text($("#dec-sub"), total > 1 ? `${total} slots · ${act ? `${act} busy` : "idle"}` : act ? "1 slot · busy" : "1 slot · idle");
  $("#dec-v").title = [bd ? `Best ${fmt(bd.tok_s, 1)} tok/s (${bd.id}, ${fmt(bd.output_tokens)} tokens, ${clock(bd.time * 1000)})` : "",
    cached(conv, live) ? `${conv.parked} chat${conv.parked === 1 ? "" : "s"} cached · ${gib(conv.bytes)} GiB` : ""].filter(Boolean).join("\n");

  // the live read rate while reading, else the newest finished request's that really read its prompt (a follow-up that read a
  // few tokens past its cache has no meaningful rate)
  const rp = reqs.find(realRead), lastPre = rp ? readRate(rp) : null;
  const livePre = reading && num(live.prefill_tok_s_mean) && live.prefill_tok_s_mean > 0;
  const pre = livePre ? live.prefill_tok_s_mean : lastPre;
  html($("#pre-k"), `Prefill${!livePre && lastPre != null ? `<span>${rp === reqs[0] ? "last request" : "last read"}</span>` : ""}`);
  html($("#pre-v"), num(pre) ? `${fmt(pre)}<small>tok/s</small>` : "-");
  $("#pre-v").classList.toggle("idle", !livePre);
  const pa = tail(hist.prefill_tok_s_mean).map(v => v > 0 ? v : v === 0 ? 0 : null);
  html($("#pre-chart"), miniChart(pa, livePre ? "var(--accent)" : "var(--fg-2)", "rgba(154,164,173,.08)", pre));
  const lp = r && num(r.prompt_tokens) ? `${fmt(r.prompt_tokens)} tok${num(r.prompt_ms) ? ` in ${(r.prompt_ms / 1000).toFixed(2)} s` : ""}` : "";
  const hit = r && r.prompt_tokens > 0 && num(r.reused) ? Math.round(r.reused / r.prompt_tokens * 100) : null;
  text($("#pre-sub"), lp ? `last prompt ${lp}${hit != null ? ` · ${hit}% cached` : ""}` : "");
  $("#pre-v").title = [lp ? `Last prompt ${lp}` : "", hit != null ? `Prefix cache hit ${hit}%` : "",
    bp ? `Best ${fmt(bp.tok_s)} tok/s (${bp.id}, ${fmt(bp.read_tokens)} tokens read)` : ""].filter(Boolean).join("\n");
}

/* ---- the expert map. Its cells fill the column. Without a per-expert map from the server (PENDING) it is the last request's lookup
   split drawn as cells: one cell per 1/N of the lookups, in tier order, static until the next request. A dense model's weights
   are drawn the same way, by bytes. With a map, each cell is one sampled expert: colour for where it lives, brightness for recent
   lookups, an ember edge for a lookup off the GPUs in the last second. Cells change only when the data does. ---- */
const T_VRAM = "var(--t-vram)", T_PEER = "var(--t-peer)", T_CPU = "var(--t-cpu)", T_SSD = "var(--warn)";
const TIERS = ["t0", "t1", "t2", "t3"], SPLIT_CELLS = 640;
let mapKey = "";
function mapLayout(N, W, H, g = 2) {
  let best = null;
  for (let c = 8; c <= 90; c++) {
    const w = W / (c + .5) - g, r = Math.ceil(N / c), h = w * 1.1547, need = r * (h * .75 + g * .87) + h * .25;
    if (w > 3 && need <= H && (!best || w > best.w)) best = { c, r, w, h };
  }
  return best || { c: 40, r: Math.ceil(N / 40), w: 8, h: 9.2 };
}
function buildMap(N) {
  const el = $("#exp-map"), W = el.clientWidth, H = el.clientHeight;
  if (!W || !H) return false;
  const key = `${N}|${W}x${H}`;
  if (key === mapKey) return true;
  mapKey = key;
  const { c, r, w, h } = mapLayout(N, W, H), g = 2, sx = w + g, sy = h * .75 + g * .87;
  const usedW = c * sx + sx / 2 - g, usedH = r * sy + h * .25, ox = (W - usedW) / 2, oy = (H - usedH) / 2;
  const pts = (cx, cy) => [0, 1, 2, 3, 4, 5].map(k => { const a = Math.PI / 3 * k - Math.PI / 2; return `${(cx + Math.cos(a) * h / 2).toFixed(1)},${(cy + Math.sin(a) * h / 2).toFixed(1)}`; }).join(" ");
  let p = "";
  for (let i = 0; i < N; i++) {
    const row = Math.floor(i / c), col = i % c;
    p += `<polygon points="${pts(ox + w / 2 + col * sx + (row % 2 ? sx / 2 : 0), oy + h / 2 + row * sy)}"/>`;
  }
  el.classList.remove("eased");     // a rebuilt map paints at once; only later changes ease
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" aria-hidden="true">${p}</svg>`;
  requestAnimationFrame(() => requestAnimationFrame(() => el.classList.add("eased")));
  return true;
}
// cls[i], op[i]: each cell's class and fill opacity ("" keeps the stylesheet's)
function paintMap(cls, op) {
  if (!buildMap(cls.length)) return;
  $("#exp-map").querySelectorAll("polygon").forEach((el, i) => {
    if (el.getAttribute("class") !== cls[i]) el.setAttribute("class", cls[i]);
    if (el.style.fillOpacity !== op[i]) el.style.fillOpacity = op[i];
  });
}
// a tier split as cells: largest remainder, so the cells add up to n; no split leaves every cell an outline
function splitCells(tiers, n) {
  const tot = tiers.reduce((a, t) => a + (t[2] || 0), 0);
  if (!tot) return Array(n).fill("off");
  const share = tiers.map(t => (t[2] || 0) / tot * n), whole = share.map(Math.floor);
  let left = n - whole.reduce((a, b) => a + b, 0);
  share.map((c, i) => [c - Math.floor(c), i]).sort((a, b) => b[0] - a[0]).forEach(([, i]) => { if (left > 0) { whole[i]++; left--; } });
  return whole.flatMap((k, i) => Array(k).fill(`${tiers[i][3]} flat`));
}
function drawSplit(tiers, tip) {
  const cls = splitCells(tiers, SPLIT_CELLS);
  paintMap(cls, cls.map(() => ""));
  $("#exp-map").title = tip;
}
function drawExpertMap(map) {
  paintMap(map.tier.map((t, i) => `${TIERS[t] || "t2"}${map.routed[i] > 0 && t >= 2 ? " miss" : ""}`),
    map.recent.map(v => (.3 + .7 * (1 - Math.exp(-(v || 0) / 8))).toFixed(2)));
  $("#exp-map").title = `Each cell is one expert: ${fmt(map.tier.length)} of ${fmt(map.total)} sampled. Colour: where it lives. Brightness: lookups in the last seconds. Ember edge: looked up from RAM in the last second.`;
}
new ResizeObserver(() => { mapKey = ""; if (M) draw(); }).observe($("#exp-map"));
const legend = tiers => tiers.map(t => `<div style="--c:${t[1]}"><span>${t[0]}</span><b>${t[2] == null ? "-" : `${(t[2] * 100).toFixed(1)}%`}</b></div>`).join("");
function drawExperts(eng, r, peers) {
  // a dense model has no experts: the column says where its weights live, and flags any that are off the GPU
  const dense = num(eng.host_weight_bytes);
  text($("#exp-h"), dense ? "Weights" : "Experts");
  if (dense) return drawWeights(eng);
  // the tier split of the last request's routed lookups; the peer tier shows only when the engine reports it, the SSD tier and
  // the warning only while experts are read from the SSD. Every expert in VRAM: no lookup ledger exists, and the split is all VRAM
  const resident = !eng.spill && num(eng.expert_slots) && num(eng.expert_total) && eng.expert_total > 0 && eng.expert_slots >= eng.expert_total;
  const peer = r && num(r.pcie_share) ? r.pcie_share : null, hit = resident ? 1 : r && num(r.hit_rate) ? r.hit_rate : null;
  const spill = eng.spill, ssd = spill ? (r && num(r.ssd_share) ? r.ssd_share : hit == null ? null : 0) : 0;
  const tiers = [["VRAM", T_VRAM, hit == null ? null : hit * (1 - (peer || 0)), "t0"]];
  if (peer != null) tiers.push([peers > 1 ? `Peer cards ×${peers}` : "Peer card", T_PEER, peer, "t1"]);
  tiers.push(["CPU · RAM", T_CPU, hit == null ? null : Math.max(0, (1 - hit) * (1 - (peer || 0)) - (ssd || 0)), "t2"]);
  if (spill) tiers.push(["SSD", T_SSD, ssd, "t3"]);
  $(".exp").classList.toggle("spill", !!spill);
  const held = num(eng.expert_slots) && num(eng.expert_total) ? `${fmt(eng.expert_slots)} of ${fmt(eng.expert_total)} experts held in VRAM slots` : "";
  $("#exp-leg").title = [spill ? "SSD in use" : resident ? "All in VRAM" : "Where the last request's expert lookups were served", held].filter(Boolean).join("\n");
  html($("#exp-leg"), legend(tiers));
  const map = M.experts_map;
  const gb = spill ? `${(spill.bytes / 1024 ** 3).toFixed(1)} GiB` : "";
  html($("#exp-key"), spill ? `<span class="warn"><b>${spill.experts} experts on SSD (${gb})</b> · free ${gb} RAM</span>`
    : map ? `<span>${fmt(map.tier.length)} of ${fmt(map.total)} shown</span><span>Bright: recent · <i>Outline</i>: from RAM</span>`
    : `<span>${held ? `${fmt(eng.expert_slots)} of ${fmt(eng.expert_total)} in VRAM` : ""}</span><span>${resident ? "All in VRAM" : hit == null ? (r ? "no lookups reported" : "no request yet") : "last request's lookups"}</span>`);
  if (map) return drawExpertMap(map);
  drawSplit(tiers, hit == null ? "" : `Each cell is 1/${SPLIT_CELLS} of the last request's expert lookups, coloured by where they were served.${held ? `\n${held}` : ""}`);
}
// the weights of a dense model, drawn like the experts: where they sit (VRAM, RAM, SSD), by bytes. The token embedding and the
// vision encoder kept in RAM on purpose count in the RAM tier; weights spilled to RAM or the SSD add a warning
function drawWeights(eng) {
  const B = n => num(n) && n > 0 ? n : 0, G = n => `${(n / 1024 ** 3).toFixed(1)}`;
  const spillRam = B(eng.host_weight_bytes), ssd = eng.spill ? B(eng.spill.bytes) : 0, emb = B(eng.host_embedding_bytes), vis = B(eng.host_vision_bytes);
  const gpu = num(eng.gpu_weight_bytes) ? eng.gpu_weight_bytes : null, ram = spillRam + emb + vis, bad = spillRam > 0 || ssd > 0;
  const tiers = [["VRAM", T_VRAM, gpu || 0, "t0", gpu == null], ["RAM", T_CPU, ram, "t2", false]];
  if (ssd > 0) tiers.push(["SSD", T_SSD, ssd, "t3", false]);
  $(".exp").classList.toggle("spill", bad);
  $("#exp-leg").title = ssd > 0 ? "SSD in use" : spillRam > 0 ? "RAM in use" : "On the GPU";
  const warn = ssd > 0 ? `${G(ssd)} GiB of weights on SSD` : `${G(spillRam)} GiB of weights in RAM`;
  html($("#exp-leg"), tiers.map(t => `<div style="--c:${t[1]}"><span>${t[0]}</span><b>${t[4] ? "-" : `${G(t[2])} GiB`}</b></div>`).join(""));
  html($("#exp-key"), bad ? `<span class="warn"><b>${warn}</b> · slower decode</span>` : "");
  drawSplit(tiers, `Each cell is 1/${SPLIT_CELLS} of the weights' bytes, coloured by where they sit.`);
}

/* ---- tabs: Requests | Slots | Hardware | Logs (keys 1-4 and L; #tab=req|slots|hw|logs opens one) ---- */
const TABS = ["req", "slots", "hw", "logs"];
const fromHash = (/tab=(\w+)/.exec(location.hash) || [])[1];
let tab = (t => TABS.includes(t) ? t : "req")(fromHash || sessionStorage.getItem("basalt.tab3"));
function setTab(t) {
  if (!TABS.includes(t)) return;
  tab = t; try { sessionStorage.setItem("basalt.tab3", tab); } catch (err) { /* private mode */ }
  draw(); drawPanes();
}
$("#tabs").addEventListener("click", e => { const b = e.target.closest("button[data-t]"); if (b) setTab(b.dataset.t); });
document.addEventListener("keydown", e => {
  if (e.target.closest("input, textarea, .ms") || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k >= "1" && k <= "4") setTab(TABS[+k - 1]);
  else if (k === "l") setTab("logs");
});
$("#ticker").addEventListener("click", () => setTab("logs"));
function drawTabs(live, reqs, slots, total, G) {
  const n = live.queued || 0, active = slots.filter(s => s.is_processing).length;
  // while a slot runs, busy over all; otherwise the slot count, in grey
  const T = [["req", "Requests", n ? `${n} queued` : reqs.length ? String(reqs.length) : "", n ? "hot" : ""],
    ["slots", "Slots", active ? `${active}/${total}` : `${total} slot${total === 1 ? "" : "s"}`, active ? "hot" : ""],
    ["hw", "Hardware", G.length ? `${G.length} GPU${G.length === 1 ? "" : "s"}` : "", ""],
    ["logs", "Logs", warnCount ? `${warnCount} warn` : logCount ? fmt(logCount) : "", warnCount ? "warn" : ""]];
  html($("#tabs"), T.map(t => `<button type="button" role="tab" data-t="${t[0]}" aria-selected="${t[0] === tab}">`
    + `<span>${t[1]}</span>${t[2] ? `<em class="${t[3]}">${t[2]}</em>` : ""}</button>`).join(""));
  $("#tabs").lastElementChild.title = `${fmt(logCount)} lines${warnCount ? `, ${warnCount} warning${warnCount === 1 ? "" : "s"}` : ""}`;
}
function drawPanes() {
  TABS.forEach(t => { $(`#p-${t}`).hidden = t !== tab; });
  $("#lfilter").hidden = tab !== "logs"; if (tab !== "logs") $("#jump").hidden = true;
  if (tab === "logs" && follow) logEl.scrollTop = logEl.scrollHeight;   // a hidden log cannot scroll; catch up when shown
}

/* ---- Slots: 1 slot is one card across the panel, 2 side by side, 4 a 2x2 grid; past 4, two columns of compact rows ---- */
const link = g => num(g.pcie_gen_max ?? g.pcie_gen) ? `Gen${g.pcie_gen_max ?? g.pcie_gen}${num(g.pcie_width) ? ` x${g.pcie_width}` : ""}` : "";
const bar = f => `<div class="bar4"><i style="width:${(clamp(f || 0, 0, 1) * 100).toFixed(1)}%"></i></div>`;
const kc = n => n >= 1048576 ? `${n / 1048576 >= 10 ? Math.round(n / 1048576) : +(n / 1048576).toFixed(1)}M` : kt(n).replace("k", "K");
const pctOf = (a, b) => b ? Math.round(a / b * 100) : 0;
const slotWord = s => !s.is_processing ? "Idle" : s.state === "prefill" ? "Reading" : "Decoding";

const accept = s => s.drafts_offered > 20 ? Math.round(s.drafts_accepted / s.drafts_offered * 100) : null;
const prefixHit = s => num(s.reused) && s.prompt_tokens > 0 ? Math.round(s.reused / s.prompt_tokens * 100) : null;
const ACC_NOTE = "Drafts accepted over offered, of the slot's current or last request";
const drafts = s => num(s.drafts_offered) || num(s.draft_len);     // a model with no drafting has no figures to show
// a slot's requests; a request the server did not place in a slot ran in the only one
const mine = (reqs, s) => reqs.filter(r => r.slot === s.id || (r.slot == null && s.id === 0));
function slotRows(rows, series, all = rows) {
  if (all === rows && rows.length > 4) {   // past 4 slots, two columns of compact rows
    const h = Math.ceil(rows.length / 2);
    return `<div class="sl2">${slotRows(rows.slice(0, h), series, all)}${slotRows(rows.slice(h), series, all)}</div>`;
  }
  const dense = all.length > 4, max = nice(Math.max(20, ...rows.flatMap(s => series[s.id].filter(num))));
  let h = `<div class="sl${dense ? " dense" : ""}"><div class="rw h"><span style="grid-column:1" title="Idle slots keep their last figures, dimmed">Slot</span><span class="e" style="grid-column:2">tok/s</span><span class="sk" style="grid-column:3">Last 60 s</span><span class="e" style="grid-column:4">Output tok</span><span class="e cx" style="grid-column:5">Context</span><span class="e" style="grid-column:6" title="${ACC_NOTE}">Accept</span></div>`;
  rows.forEach(s => {
    const a = series[s.id], st = busyStats(a), api = API[s.api] || s.api || "", acc = accept(s), hit = prefixHit(s);
    const pre = s.state === "prefill", run = s.is_processing;
    const rate = !run ? "-" : pre ? (num(s.prefill_done) && s.prompt_tokens ? `${Math.min(100, Math.round(s.prefill_done / s.prompt_tokens * 100))}%` : "…") : fmt(s.tok_s);
    h += `<div class="rw${run ? "" : " idle"}"><span class="p nm" style="grid-column:1" title="Slot ${s.id + 1} · ${slotWord(s)}">Slot ${s.id + 1}${dense ? `<span class="u"> · </span><span class="${run ? "api hot" : "api"}">${esc(api)}</span>` : ""}</span>`
      + `<span class="p rate e" style="grid-column:2">${rate}</span>${spark(a, max)}`
      + `<span class="p e" style="grid-column:4">${run ? `${fmt(s.generated)}${num(s.max_tokens) ? `<span class="u">/${lim(s.max_tokens)}</span>${bar(s.generated / s.max_tokens)}` : ""}` : "-"}</span>`
      + `<span class="p e cx" style="grid-column:5">${kt(s.ctx_used || 0)}<span class="u">/${kt(s.n_ctx || 0)}</span>${bar(s.n_ctx ? s.ctx_used / s.n_ctx : 0)}</span>`
      + `<span class="p e" style="grid-column:6">${acc != null ? `${acc}<span class="u">%</span>` : "-"}</span>`
      + (dense ? "" : `<span class="s" style="grid-column:1">${run ? esc(api) : "Idle"}</span>`
        + `<span class="s e" style="grid-column:2">${pre ? "Read" : st ? `Avg ${Math.round(st.avg)}` : ""}</span>`
        + `<span class="s e" style="grid-column:4">${num(s.ttft_s) ? `TTFT ${s.ttft_s.toFixed(1)} s` : ""}</span>`
        + `<span class="s e cx" style="grid-column:5">${hit != null ? `Hit ${hit}%` : ""}</span>`
        + `<span class="s e" style="grid-column:6">${num(s.draft_len) ? `Draft ${s.draft_len.toFixed(1)}` : ""}</span>`) + `</div>`;
  });
  return h + `</div>`;
}
const seg = parts => `<div class="sbar">${parts.map(p => `<i style="width:${(clamp(p[1], 0, 1) * 100).toFixed(2)}%;background:${p[0]}"></i>`).join("")}</div>`;
const ep = p => String(p || "-").replace("/v1/", "").replace("chat/completions", "chat");
const kv = (k, v) => `<div><dt>${k}</dt><dd>${v}</dd></div>`;
const kvs = rows => `<dl class="kv sm">${rows.filter(Boolean).map(r => kv(r[0], r[1])).join("")}</dl>`;
// each section: heading, the headline figure, its graphic right under it, then the rows that detail it. A row whose field the
// server does not send yet (PENDING) is left out
// an idle slot keeps its context and drafting counts but no request figures; those come from its last request
function settled(s, last) {
  if (s.is_processing || !last) return s;
  return { ...s, prompt_tokens: last.prompt_tokens, reused: last.reused, generated: last.output_tokens, ttft_s: last.ttft_s,
           draft_len: s.draft_len ?? last.draft_len, drafts_offered: s.drafts_offered ?? last.drafts_offered,
           drafts_accepted: s.drafts_accepted ?? last.drafts_accepted };
}
function slotParts(s) {
  const prompt = s.prompt_tokens || 0, cachedTok = clamp(s.reused || 0, 0, prompt), read = prompt - cachedTok, gen = s.generated || 0;
  const n = s.n_ctx || 1, used = s.ctx_used || 0, known = cachedTok + read + gen, k = known > 0 ? Math.min(1, used / known) : 0;
  return { n, used, cached: cachedTok * k, read: read * k, gen: gen * k, other: Math.max(0, used - known * k) };
}
function slotSections(slot, series, reqs, size) {
  const last = mine(reqs, slot)[0] || null, s = settled(slot, last);
  const run = s.is_processing, st = busyStats(series), acc = accept(s), now = Date.now() / 1000;
  const C = slotParts(s), hit = s.hit;
  const started = s.started, path = s.path, readMs = run ? s.prompt_ms : last ? last.prompt_ms : null;
  const pct = v => `${(v * 100).toFixed(1)}%`, per = v => num(v) ? `${big(v)} <span class="u">/ s</span>` : "-";
  const rate = run ? s.tok_s : last && num(last.decode_tok_s) ? last.decode_tok_s : st ? st.avg : null;
  const head = (h, fig) => `<h4>${h}</h4><div class="big">${fig}</div>`;
  const where = run ? path : last ? last.path : null, reading = run && s.state === "prefill";
  const prompt = s.prompt_tokens;
  // a slot reading its prompt shows how much it has read; a decoding one its rate
  const fig = reading ? `${num(s.prefill_done) && s.prompt_tokens ? Math.min(100, Math.round(s.prefill_done / s.prompt_tokens * 100)) + "%" : "-"}<small>read</small>`
    : `${num(rate) ? fmt(rate, 1) : "-"}<small>tok/s</small>`;
  const request = `<section><h4>${run ? "Current request" : "Last request"}</h4><div class="xl${run ? " hot" : ""}">${fig}</div>`
    + kvs([["Output", run && num(s.max_tokens) ? `${fmt(s.generated)} <span class="u">/ ${lim(s.max_tokens)}</span>` : run ? fmt(s.generated) : last ? `${fmt(last.output_tokens)} <span class="u">tok</span>` : "-"],
      ["Prompt", num(prompt) && prompt > 0 ? `${fmt(prompt)} <span class="u">tok</span>` : "-"],
      ["TTFT", num(s.ttft_s) ? `${s.ttft_s.toFixed(1)} s` : "-"],
      run ? (num(started) ? ["Elapsed", dur((now - started) * 1000)] : null) : ["Duration", last && num(last.duration_s) ? `${last.duration_s.toFixed(1)} s` : "-"]])
    + `</section>`;
  const context = `<section>${head("Context", `<b>${kc(C.used)}</b><span class="u"> / ${kc(C.n)} · ${pctOf(C.used, C.n)}%</span>`)}`
    + `<div class="vis">${seg([["var(--fg-2)", C.cached / C.n], ["var(--fg-3)", (C.read + C.other) / C.n], ["var(--accent)", C.gen / C.n]])}</div>`
    + kvs([["Cache hit", prefixHit(s) != null ? `${prefixHit(s)}%` : "-"],
      ["Cached prefix", kc(Math.round(C.cached))], ["Prompt read", kc(Math.round(C.read + C.other))],
      num(readMs) ? ["Read time", `${(readMs / 1000).toFixed(2)} s`] : null]) + `</section>`;
  const pos = s.accept_pos || [];
  const drafting = !drafts(s) ? "" : `<section>${head("Drafting", `<b>${acc != null ? `${acc}%` : "-"}</b><span class="u"> accepted</span>`)}`
    + `<div class="vis grow">${pos.length ? `<div class="pos" title="Acceptance by draft position, 1 to ${pos.length}">${pos.map((p, i) => `<i style="height:${(p * 100).toFixed(0)}%" title="Position ${i + 1}: ${Math.round(p * 100)}%"></i>`).join("")}</div>` : ""}</div>`
    + kvs([["Avg window", num(s.draft_len) ? s.draft_len.toFixed(1) : "-"], ["Drafts", num(s.drafts_offered) ? `${fmt(s.drafts_accepted)} <span class="u">/ ${fmt(s.drafts_offered)}</span>` : "-"],
      ["Tokens / step", num(s.draft_len) && acc != null ? (1 + s.draft_len * acc / 100).toFixed(2) : "-"], ["Rejected", num(s.drafts_offered) ? fmt(s.drafts_offered - s.drafts_accepted) : "-"]]) + `</section>`;
  const tiers = hit ? [["var(--t-vram)", hit.vram], ["var(--t-peer)", hit.peer || 0], ["var(--t-cpu)", hit.cpu]] : null;
  const experts = !hit ? "" : `<section>${head("Expert lookups", `<b>${pct(hit.vram)}</b><span class="u"> in VRAM</span>`)}`
    + `<div class="vis grow">${size === "quarter" ? seg(tiers) : cellBlock(tiers)}</div>`
    + kvs([hit.peer != null ? ["Peer card", pct(hit.peer)] : null, ["Lookups", per(hit.lookups_s)],
      hit.peer != null ? ["Over PCIe", per(hit.peer_s)] : null, ["From RAM", per(hit.ram_s)]]) + `</section>`;
  const H = mine(reqs, s).slice(0, 6);
  const history = `<section><h4>Recent</h4>${H.length ? `<table class="mini"><thead><tr><th>Time</th><th>Endpoint</th><th class="r">Out</th><th class="r">tok/s</th></tr></thead><tbody>${H.map(r => `<tr><td>${clock(r.time * 1000)}</td><td class="dim">${esc(ep(r.path))}</td><td class="r">${fmt(r.output_tokens)}</td><td class="r">${num(r.decode_tok_s) ? r.decode_tok_s.toFixed(1) : "-"}</td></tr>`).join("")}</tbody></table>` : `<div class="dim">None yet</div>`}</section>`;
  return { request, context, drafting, experts, history, where };
}
// a small block of hex cells in the tier colours, echoing the hero map: 60 cells, one per 1/60 of this slot's lookups
function cellBlock(parts) {
  const C = 15, R = 4, n = C * R, tot = parts.reduce((a, p) => a + p[1], 0) || 1;
  const k = parts.map(p => Math.round(p[1] / tot * n)); k[0] += n - k.reduce((a, b) => a + b, 0);
  const col = []; parts.forEach((p, i) => { for (let j = 0; j < k[i]; j++) col.push(p[0]); });
  const w = 10, h = w * 1.1547, sx = w + 1.6, sy = h * .75 + 1.4;
  let g = "";
  for (let r = 0, i = 0; r < R; r++) for (let c = 0; c < C; c++, i++) {
    const cx = w / 2 + c * sx + (r % 2 ? sx / 2 : 0), cy = h / 2 + r * sy;
    g += `<polygon points="${[0, 1, 2, 3, 4, 5].map(q => { const a = Math.PI / 3 * q - Math.PI / 2; return `${(cx + Math.cos(a) * h / 2).toFixed(1)},${(cy + Math.sin(a) * h / 2).toFixed(1)}`; }).join(" ")}" fill="${col[i]}"/>`;
  }
  return `<svg class="cb" viewBox="0 0 ${(C * sx + sx / 2).toFixed(1)} ${((R - 1) * sy + h).toFixed(1)}" preserveAspectRatio="xMinYMax meet" aria-hidden="true">${g}</svg>`;
}
// the one chart that spans a card: the slot's decode rate over 60 s; tokens accepted per window as faint ticks under it (PENDING)
function slotStrip(s, series, max) {
  const A = tail(s.accept_hist || []).map(v => v > 0 ? v : null), amax = Math.max(1, ...A.filter(num));
  const y = v => 100 - 6 - Math.min(v, max) / max * 90;
  const ticks = A.map((v, i) => v == null ? "" : `<line x1="${(i / (A.length - 1) * 100).toFixed(2)}" x2="${(i / (A.length - 1) * 100).toFixed(2)}" y1="100" y2="${(100 - v / amax * 22).toFixed(1)}" stroke="var(--basalt-500)" stroke-width="1" vector-effect="non-scaling-stroke"/>`).join("");
  const run = s.is_processing, line = lines(series, y).map(p => poly(p, `stroke="${run ? "var(--accent)" : "var(--fg-3)"}" stroke-width="1.4"`)).join("");
  const a = stats(A);
  return `<div class="strip"><div class="sh"><span>Decode · last 60 s · ${tick(max)} max</span><span>${a ? `${a.avg.toFixed(2)} tok / window` : ""}</span></div>`
    + `<div class="sc"><svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">${baseLine}${ticks}${line}</svg></div></div>`;
}
function slotCard(s, series, reqs, max, size, want) {
  const run = s.is_processing, X = slotSections(s, series, reqs, size), gap = "<section></section>";
  // "Slot 1 · OpenAI · chat", the API name in orange while the slot runs, then the request's endpoint; the state word is the tooltip
  const head = `<header title="Slot ${s.id + 1} · ${slotWord(s)}"><b>Slot ${s.id + 1}</b><span class="sep">·</span><span class="${run ? "hot" : ""}">${esc(API[s.api] || s.api || slotWord(s))}</span>`
    + `${X.where ? `<span class="sep">·</span><span>${esc(ep(X.where))}</span>` : ""}</header>`;
  // a section another slot shows and this one lacks keeps its column, empty, so the cards line up
  const body = X.request + X.context + (want.draft ? X.drafting || gap : "") + (want.hit ? X.experts || gap : "") + (size === "full" ? X.history : "");
  return `<article class="scard ${size}${run ? "" : " idle"}">${head}<div class="secs" style="--secs:${want.cols}">${body}</div>${size === "quarter" ? "" : slotStrip(s, series, max)}</article>`;
}
// no slot is up (no model, a load, sleep): the cards' layout, quiet
function slotsDown(total) {
  const n = Math.min(Math.max(1, total), 4), size = n === 1 ? "full" : n === 2 ? "half" : "quarter";
  const card = k => `<article class="scard ${size} idle"><header><b>Slot ${k + 1}</b></header><div class="secs" style="--secs:repeat(2, minmax(0, 1fr))">`
    + `<section><h4>Last request</h4><div class="xl">-<small>tok/s</small></div></section>`
    + `<section><h4>Context</h4><div class="big"><b>-</b></div><div class="vis">${seg([])}</div></section></div></article>`;
  return `<div class="sdown"><div class="scards n${n}">${Array.from({ length: n }, (_, k) => card(k)).join("")}</div></div>`;
}
function drawSlots(slots, series, reqs, total) {
  const n = slots.length;
  if (!n) return html($("#p-slots"), slotsDown(total));
  if (n > 4) return html($("#p-slots"), slotRows(slots, series));
  // the strips' axis covers each card's number too, so an idle strip's axis is not under it
  const shown = slots.map(s => s.is_processing ? s.tok_s : (mine(reqs, s)[0] || {}).decode_tok_s).filter(num);
  const max = nice(Math.max(20, ...shown, ...series.flatMap(a => a.filter(num)))), size = n === 1 ? "full" : n === 2 ? "half" : "quarter";
  // one column per section any slot shows
  const draft = slots.some(drafts), hit = slots.some(s => s.hit != null);
  const want = { draft, hit, cols: `repeat(${2 + draft + hit}, minmax(0, 1fr))${size === "full" ? " minmax(0, 1.3fr)" : ""}` };
  html($("#p-slots"), `<div class="scards n${n}">${slots.map(s => slotCard(s, series[s.id], reqs, max, size, want)).join("")}</div>`);
}

/* ---- Requests: queue and history | tokens per request | last and best, cache | totals ---- */
function requestsPane(live, reqs, multi) {
  const rows = [];
  if (!multi && (live.state === "reading" || live.state === "generating"))
    rows.push({ t: Date.now() / 1000 - (live.elapsed_s || 0), id: "Running", path: "", out: live.generated, tps: live.tok_s, dur: live.elapsed_s, live: true });
  reqs.forEach(r => rows.push({ t: r.time, id: r.id, slot: r.slot, path: r.path, out: r.output_tokens, tps: r.decode_tok_s, dur: r.duration_s, finish: r.finish }));
  if (!rows.length) return `<div class="qnone">No requests yet</div>`;
  return `<table><thead><tr><th>Time</th><th>ID</th>${multi ? "<th>Slot</th>" : ""}<th>Endpoint</th><th class="r">Output</th><th class="r">Rate</th><th class="r">Duration</th></tr></thead><tbody>`
    + rows.map(r => {
      // a request that did not simply stop is marked on its id, with the reason
      const odd = r.finish && r.finish !== "stop" ? (r.finish === "error" ? "err" : "warn") : "";
      return `<tr${r.live ? ` class="live"` : ""}><td>${clock(r.t * 1000)}</td><td class="${odd}">${esc(r.id)}${odd ? ` · ${esc(r.finish)}` : ""}</td>`
        + (multi ? `<td class="dim">${num(r.slot) ? `s${r.slot + 1}` : "-"}</td>` : "")
        + `<td>${r.live ? "-" : esc(ep(r.path))}</td><td class="r">${num(r.out) ? fmt(r.out) : "-"}</td><td class="r">${num(r.tps) ? r.tps.toFixed(1) : "-"}</td>`
        + `<td class="r">${num(r.dur) ? `${r.dur.toFixed(1)} s` : "-"}</td></tr>`;
    }).join("") + `</tbody></table>`;
}
const mss = s => `${Math.floor(s / 60)}:${p2(Math.floor(s % 60))}`;
function queuePane(live) {
  const q = live.queue || [], n = live.queued || 0;
  if (!n && !q.length) return `<div class="qnone">Queue empty · no request is waiting</div>`;
  const toks = q.reduce((t, r) => t + (num(r.prompt_tokens) ? r.prompt_tokens : 0), 0);
  return `<div class="qsum"><span><b>${fmt(Math.max(n, q.length))}</b> prompt${Math.max(n, q.length) === 1 ? "" : "s"} waiting</span><span><b>${fmt(toks)}</b> tok to read</span></div>`
    + `<table><thead><tr><th>API</th><th class="r">Prompt tok</th><th class="r">Waiting</th></tr></thead><tbody>`
    + q.map(r => `<tr><td>${esc(API[r.api] || r.api || "-")}</td><td class="r">${num(r.prompt_tokens) ? fmt(r.prompt_tokens) : "-"}</td><td class="r">${mss(r.waited_s || 0)}</td></tr>`).join("")
    + `</tbody></table>`;
}
// stacked columns, one per recent request, oldest first: prompt read (the uncached part) / generated, scaled to the largest
function ctxHistory(reqs) {
  const h = reqs.slice(0, 24).reverse().map(r => ({ p: (r.prompt_tokens || 0) - clamp(r.reused || 0, 0, r.prompt_tokens || 0), g: r.output_tokens || 0 }));
  if (!h.length) return `<h3>Tokens per request</h3><div class="qnone">No requests yet</div>`;
  const W = 480, H = 100, step = W / h.length, bw = step * .62;
  const top = nice(Math.max(1, ...h.map(r => r.p + r.g)) / 1.15);
  let g = "";
  h.forEach((r, i) => {
    let y = H; const x = i * step + (step - bw) / 2;
    [["p", "rgba(154, 164, 173, .5)"], ["g", "var(--accent)"]].forEach(([k, col]) => {
      const hh = r[k] / top * H; y -= hh;
      g += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${hh.toFixed(1)}" fill="${col}"/>`;
    });
  });
  return `<div class="cx-hist"><h3>Tokens per request<span>Last ${h.length}</span></h3>`
    + `<div class="lg"><span style="--c:rgba(154,164,173,.6)">Prompt read</span><span style="--c:var(--accent)">Generated</span></div><div class="cx-chart">`
    + `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Tokens per request">`
    + `<line x1="0" x2="${W}" y1="${H}" y2="${H}" stroke="var(--line)"/><line x1="0" x2="${W}" y1="${H / 2}" y2="${H / 2}" stroke="var(--line-soft)" stroke-dasharray="2 4"/>${g}</svg>`
    + `<span class="tk" style="top:0">${kc(top)}</span><span class="tk" style="top:50%">${kc(top / 2)}</span><span class="tk" style="top:100%">0</span>`
    + `<div class="xl"><span>Oldest</span><span>Latest</span></div></div></div>`;
}
function lastList(reqs, best) {
  const r = reqs[0], bd = best.decode, bp = bestRead(best, reqs);
  return `<h3>Last request</h3><dl class="kv">`
    + kv("Prompt", r && num(r.prompt_tokens) ? `${fmt(r.prompt_tokens)} tok${num(r.prompt_ms) ? ` in ${(r.prompt_ms / 1000).toFixed(2)} s` : ""}` : "-")
    + kv("Prefix cache hit", r && r.prompt_tokens > 0 && num(r.reused) ? `${Math.round(r.reused / r.prompt_tokens * 100)}%` : "-")
    + kv("Decode", r && num(r.decode_tok_s) ? `${r.decode_tok_s.toFixed(1)} tok/s` : "-") + `</dl>`
    + `<h3>Best since the server started</h3><dl class="kv">`
    + (bd ? `<div title="${esc(bd.id)}, ${fmt(bd.output_tokens)} tokens, ${clock(bd.time * 1000)}"><dt>Decode</dt><dd>${fmt(bd.tok_s, 1)} tok/s</dd></div>` : kv("Decode", "-"))
    + (bp ? `<div title="${esc(bp.id)}, ${fmt(bp.read_tokens)} tokens read, ${clock(bp.time * 1000)}${bp.recent ? ", the fastest of the recent requests" : ""}"><dt>Prefill</dt><dd>${fmt(bp.tok_s)} tok/s</dd></div>` : kv("Prefill", "-")) + `</dl>`;
}
function cacheList(conv, live) {
  return cached(conv, live) ? `<h3>Conversation cache</h3><dl class="kv">${kv("Cached", `${conv.parked} chat${conv.parked === 1 ? "" : "s"} · ${gib(conv.bytes)} GiB`)}`
    + (num(conv.evictions) ? kv("Evicted to make room", fmt(conv.evictions)) : "")
    + (num(conv.restored) ? kv("Last restore", `${fmt(conv.restored)} tok${num(conv.restored_ms) ? ` in ${Math.round(conv.restored_ms)} ms` : ""}`) : "") + `</dl>` : "";
}
function sessionList(t) {
  const n = t && num(t.requests) ? t : null;
  return `<h3>Totals</h3><dl class="kv">`
    + (n ? kv("Requests served", fmt(n.requests)) + kv("Tokens written", fmt(n.written)) + kv("Prompt tokens", fmt(n.prompt)) + kv("Reused from cache", fmt(n.reused))
      + kv("Time reading", dur(n.prompt_ms)) + kv("Time generating", dur(n.decode_ms))
      + kv("Average decode", n.decode_ms > 0 ? `${(n.written / (n.decode_ms / 1000)).toFixed(1)} <span class="u">tok/s</span>` : "-")
      + kv("Average read", n.prompt_ms > 0 ? `${fmt((n.prompt - n.reused) / (n.prompt_ms / 1000))} <span class="u">tok/s</span>` : "-")
      + kv("Uptime", uptime()) : kv("Requests served", "-")) + `</dl>`;
}
function drawReq(live, reqs, total) {
  html($("#p-req"), `<div class="rq"><div class="rq-main"><h3>Queue</h3>${queuePane(live)}<h3>Recent requests</h3>${requestsPane(live, reqs, total > 1)}</div>`
    + `<div class="rq-side chart">${ctxHistory(reqs)}</div><div class="rq-side fixed">${lastList(reqs, M.best || {})}${cacheList(M.conversation, live)}</div>`
    + `<div class="rq-side fixed">${sessionList(M.totals)}</div></div>`);
}

/* ---- Hardware: 1-2 cards get the rich columns, 3-4 the compact ones, 5+ a table; the System column at the right ---- */
const sbRow = (k, v, u, st) => `<div class="sb-row"><span class="k">${k}</span>${st ? `<span class="st">${st}</span>` : ""}<span class="num">${v}${u ? `<span class="u">${u}</span>` : ""}</span></div>`;
const r0 = v => num(v) ? String(Math.round(v)) : "-";
function sbLoad(h, now, label = "Load") {
  const s = stats(h);
  return `<div class="sb-met">${sbRow(label, num(now) ? `${Math.round(now)}%` : "-", "", s ? `Min ${Math.round(s.min)} · Avg ${Math.round(s.avg)}%` : "")}${sbChart(h, 0, 100, true)}</div>`;
}
function sbPower(h, now, lim) {
  const s = stats(h), top = num(lim) && lim > 0 ? lim : Math.max(1, s ? s.max : 1);
  return `<div class="sb-met">${sbRow("Power", `${r0(now)}${num(lim) ? ` / ${Math.round(lim)}` : ""}`, "W", s ? `Avg ${Math.round(s.avg)} W` : "")}${sbChart(h, 0, top, false, num(lim))}</div>`;
}
function sbTemp(h, now) {
  const s = stats(h), lo = Math.min(40, s ? Math.floor(s.min) - 5 : 40), hi = Math.max(85, s ? Math.ceil(s.max) + 5 : 85);
  return `<div class="sb-met">${sbRow("Temp", r0(now), "°C", s ? `Min ${Math.round(s.min)} · Max ${Math.round(s.max)} °C` : "")}${sbChart(h, lo, hi, false)}</div>`;
}
const gpuMeta = g => `${num(g.temp) ? `<b>${Math.round(g.temp)}</b> °C` : ""}${num(g.temp) && link(g) ? " · " : ""}${esc(link(g))}`;
const vramRow = g => `<div class="sb-met">${sbRow("VRAM", num(g.mem_used) ? `${gib(g.mem_used)} / ${gib(g.mem_total)}` : "-", "GiB")}${bar(num(g.mem_total) ? g.mem_used / g.mem_total : 0)}</div>`;
function sbGpu(g, h, name, short, foot) {
  return `<div class="gpu"><div class="gh" title="${esc(name)}"><span class="gn">${esc(short || name)}</span><span class="gm">${gpuMeta(g)}</span></div>`
    + vramRow(g) + `<div class="gch">${sbLoad(tail(h.util), g.util)}${sbPower(tail(h.power), g.power, g.power_limit)}${sbTemp(tail(h.temp), g.temp)}</div>`
    + (foot ? `<div class="gfoot" title="SM clock · measured host-to-device bandwidth · experts held">${foot}</div>` : "") + `</div>`;
}
function sbTable(G, H, names) {
  let h = `<div class="sb-tbl"><div class="h l">GPU</div><div class="h">Load</div><div class="h">VRAM GiB</div><div class="h">Power W</div><div class="h">°C</div><div class="h">Link</div>`;
  G.forEach(g => {
    h += `<div class="t">${esc(names(g))}</div><div class="c">${num(g.util) ? `${Math.round(g.util)}%` : "-"}</div>`
      + `<div class="c">${gib(g.mem_used)}<span class="u">/${gib(g.mem_total)}</span></div>`
      + `<div class="c">${r0(g.power)}<span class="u">/${r0(g.power_limit)}</span></div><div class="c">${r0(g.temp)}</div><div class="c dim">${esc(link(g))}</div>`;
  });
  // the mean load and total power of all cards, sample by sample
  const series = k => tail(H[0] ? H[0][k] : []).map((_, i) => {
    const v = H.map(x => tail(x[k])[i]).filter(num);
    return v.length ? v.reduce((t, x) => t + x, 0) / (k === "util" ? v.length : 1) : null;
  });
  const sum = k => G.reduce((t, g) => t + (num(g[k]) ? g[k] : 0), 0);
  const load = G.filter(g => num(g.util)), mean = load.length ? load.reduce((t, g) => t + g.util, 0) / load.length : null;
  return h + `</div><div class="gpu agg"><div class="gh"><span class="gn">All ${G.length} GPUs</span></div><div class="gch">`
    + sbLoad(series("util"), mean, "Mean load") + sbPower(series("power"), sum("power"), sum("power_limit") || null) + `</div></div>`;
}
// a labelled stacked bar: each segment's name and value sit under its start; one too narrow for its label keeps only a tooltip
function labelledBar(parts, T, px) {
  let at = 0, end = -1;
  const segs = parts.map(p => { const w = p[2] / T * 100, c = { ...p, at, w }; at += w; return c; });
  segs.forEach(c => { const need = (Math.max(c[0].length, 4) * 7 + 10) / px * 100; c.show = c.at >= end && c.at + need <= 101; c.right = !c.show && c.at + c.w >= 99.9 && 100 - need >= end; if (c.show) end = c.at + need; });
  return `<div class="lbar"><div class="sbar">${segs.map(c => `<i style="width:${c.w.toFixed(2)}%;background:${c[1]}" title="${c[0]} ${gib(c[2])} GiB"></i>`).join("")}</div>`
    + `<div class="lb">${segs.filter(c => c.show || c.right).map(c => `<span style="${c.right ? "right:0;align-items:flex-end" : `left:${c.at.toFixed(2)}%`}"><em>${c[0]}</em><b>${gib(c[2])}</b></span>`).join("")}</div></div>`;
}
// the expert slots, the KV cache (when the engine says) and the rest in use: weights, workspaces, other processes
const VP = [["experts", "Experts", "var(--t-vram)"], ["kv", "KV", "var(--t-peer)"], ["other", "Other", "var(--fg-2)"]];
function vramSplit(g, px) {
  const v = g.vram_parts, T = g.mem_total || 1;
  if (!v) return bar(g.mem_used / T);
  const parts = VP.filter(p => v[p[0]] > 0).map(p => [p[1], p[2], v[p[0]]]).concat([["Free", "var(--basalt-800)", Math.max(0, T - g.mem_used)]]);
  return labelledBar(parts, T, px);
}
const mb = v => num(v) ? v >= 1000 ? `${(v / 1000).toFixed(1)} GB/s` : `${Math.round(v)} MB/s` : "-";
// stat cells: label above value, in a grid; a cell whose field is missing is left out
const cells = rows => `<dl class="cells">${rows.filter(Boolean).map(r => `<div${r[2] ? ` title="${esc(r[2])}"` : ""}><dt>${r[0]}</dt><dd>${r[1]}</dd></div>`).join("")}</dl>`;
const when = (v, row) => v == null ? null : row;
function gpuDetail(g) {
  const sm = g.sm_clock, mem = g.mem_clock, h2d = g.h2d_gbps, held = g.expert_slots, thr = g.throttle;
  return cells([when(sm, ["SM clock", `${fmt(sm)} <span class="u">MHz</span>`]), when(mem, ["Memory clock", `${fmt(mem)} <span class="u">MHz</span>`]),
    link(g) ? ["PCIe link", esc(link(g))] : null,
    when(h2d, ["Host → GPU", `${h2d} <span class="u">GB/s</span>`, "Measured host-to-device bandwidth"]),
    ["Traffic in", mb(g.pcie_rx_mb)], ["Traffic out", mb(g.pcie_tx_mb)],
    when(held, ["Experts held", fmt(held)]), when(thr, ["Throttling", thr || "none"])]);
}
function gpuRich(g, h, name, cls) {
  return `<div class="gpu ${cls}"><div class="gh"><span class="gn">${esc(name)}</span><span class="gm">${gpuMeta(g)}</span></div>`
    + `<div class="gch3">${sbLoad(tail(h.util), g.util)}${sbPower(tail(h.power), g.power, g.power_limit)}${sbTemp(tail(h.temp), g.temp)}</div>`
    + `<section class="vb"><h4>VRAM <span>${gib(g.mem_used)} / ${gib(g.mem_total)} GiB</span></h4>${vramSplit(g, cls === "one" ? 960 : 455)}</section>${gpuDetail(g)}</div>`;
}
function gpuFoot(g) {
  const sm = g.sm_clock, h2d = g.h2d_gbps, held = g.expert_slots;
  return [num(sm) ? `${fmt(sm)} MHz` : "", num(h2d) ? `${h2d} GB/s` : "", num(held) ? big(held) : ""].filter(Boolean).join(" · ");
}
function drawHw(hw, st, hist, eng) {
  gid = 0;
  const G = hw.gpus || [], H = Object.fromEntries((hist.gpus || []).map(x => [x.index, x]));
  const named = Object.fromEntries((st.gpus || []).map(g => [g.index, shortGpu(g.name)]));
  const title = g => `GPU ${g.index} · ${named[g.index] || "GPU"}`;
  const gpus = !G.length ? `<div class="empty">No GPU readings.</div>`
    : G.length > 4 ? `<div class="gtab">${sbTable(G, G.map(g => H[g.index] || {}), title)}</div>`
    : G.length <= 2 ? `<div class="gpus rich n${G.length}">${G.map(g => gpuRich(g, H[g.index] || {}, title(g), G.length === 1 ? "one" : "two")).join("")}</div>`
    : `<div class="gpus n${G.length}">${G.map(g => sbGpu(g, H[g.index] || {}, title(g), named[g.index], gpuFoot(g))).join("")}</div>`;
  const R = hw.ram_parts, T = hw.ram_total || 1, dh = hw.disk_hist, bw = ahead(hw, "ram_bw_gbs"), commit = eng.commit;
  const ccd = (hw.cpu_ccd || []).map((v, i) => `<div class="ccd"><span>CCD ${i}</span>${bar(v / 100)}<b>${Math.round(v)}%</b></div>`).join("");
  const ram = R ? labelledBar([["Experts", "var(--t-cpu)", R.experts], ["Cache", "var(--fg-3)", R.page_cache], ["Other", "var(--raised)", R.other],
    ["Free", "var(--basalt-800)", Math.max(0, T - hw.ram_used - R.page_cache)]], T, 340) : bar(num(hw.ram_total) ? hw.ram_used / T : 0);
  const disk = dh ? `<div class="dsk">${spark(tail(dh), nice(Math.max(10, ...dh.filter(num))))}</div>` : "<span></span>";
  const held = num(eng.expert_slots) && num(eng.expert_total) && eng.expert_total > 0 ? `${fmt(eng.expert_slots)} <span class="u">of ${fmt(eng.expert_total)}</span>` : null;
  // four blocks spread over the column's height, so a block without its pending rows leaves no hole
  const sys = `<div class="blk"><h4 title="${esc(st.cpu_name || "")}">${esc(shortCpu(st.cpu_name) || "CPU")} <span>${num(hw.cpu) ? `${Math.round(hw.cpu)}%` : ""}</span></h4>${ccd}</div>`
    + `<div class="blk"><h4>RAM <span>${gib(hw.ram_used)} / ${gib(hw.ram_total)} GiB</span></h4>${ram}</div>`
    + `<div class="blk"><div class="srow"><span>Read</span>${disk}<b>${num(hw.disk_read_mb) ? hw.disk_read_mb.toFixed(1) : "-"} <span class="u">MB/s</span></b></div></div>`
    + cells([when(bw, ["RAM bandwidth", `${bw} <span class="u">GB/s</span>`]), held ? ["Experts in VRAM", held, "Experts held in VRAM slots"] : null, ["Uptime", uptime()],
      ["Driver", st.driver ? `${esc(st.driver)} <span class="u">CUDA ${esc(st.cuda || "-")}</span>` : "-"],
      eng.version ? ["Basalt", `${esc(eng.version)}${commit ? ` <span class="u">${esc(commit)}</span>` : ""}`] : null]);
  html($("#p-hw"), `<div class="hw"><div class="hw-g">${gpus}</div><div class="hw-s">${sys}</div></div>`);
}

/* ---- several models: the picker names the model and choosing an unloaded one loads it; Unload beside the name ---- */
const WORD = { loaded: "Loaded", unloaded: "Load", loading: "Loading", failed: "Failed" };
const rowWord = m => m.asleep && m.status.value === "loaded" ? "Asleep" : WORD[m.status.value] || m.status.value;
function setPicker(open) {
  const root = $("#models");
  open ? root.setAttribute("data-open", "") : root.removeAttribute("data-open");
  root.firstElementChild.setAttribute("aria-expanded", String(open));
}
// a load or unload in flight: one post per click, the button or state word says so until the models list shows it took
// (the model loading or loaded, or no longer loaded); a failed post restores the controls and shows the error in the header
let pending = null, failed = null;
const ACT = { "v1/load": ["Loading…", "Load failed"], "v1/unload": ["Unloading…", "Unload failed"] };
function control(path, id) {
  if (pending) return;
  const p = pending = { path, id, ok: false };
  draw();
  fetch(path, { method: "POST", headers: { "Content-Type": "application/json", ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}) },
                body: JSON.stringify({ model: id }) })
    .then(async r => {
      if (!r.ok) throw new Error(await r.json().then(j => (j.error || {}).message, () => null) || `HTTP ${r.status}`);
      p.ok = true;             // the next reading of the models list settles it
    })
    .catch(e => { if (pending === p) pending = null; failed = { text: `${ACT[path][1]}: ${e.message}`, until: Date.now() + 8000 }; draw(); });
}
// whether the models list shows the pending action took; a post that answered ok settles at the next reading anyway
function settle(models, fresh) {
  if (!pending) return;
  const m = models.find(x => x.id === pending.id), v = m ? m.status.value : null;
  if ((pending.path === "v1/unload" ? v !== "loaded" : v === "loading" || (v === "loaded" && !m.asleep)) || (fresh && pending.ok)) pending = null;
}
function unloadView(models, live) {
  if (live.state === "loading") return { hidden: true };
  const m = models.find(x => x.status.value === "loaded" && !x.asleep)
    || models.find(x => x.status.value === "loaded" && x.parked_mib == null);
  if (!m) return { hidden: true };
  if (pending) return { hidden: false, id: m.id, disabled: true, label: pending.path === "v1/unload" ? ACT["v1/unload"][0] : "Unload", title: "" };
  const busy = live.state === "reading" || live.state === "generating" || live.queued > 0;
  return { hidden: false, id: m.id, disabled: busy, label: "Unload", title: busy ? "A request is running or queued" : `Unload ${m.id}; the next request loads it again` };
}
function buildPicker(root) {
  const btn = document.createElement("button"); btn.type = "button"; btn.setAttribute("aria-haspopup", "true"); btn.setAttribute("aria-expanded", "false");
  btn.append(document.createElement("span"));
  const pop = document.createElement("div"); pop.className = "pop"; pop.setAttribute("role", "menu");
  root.append(btn, pop);
  const live = () => [...pop.children].filter(o => !o.disabled);
  btn.onclick = () => {
    const on = !root.hasAttribute("data-open");
    setPicker(on);
    if (on) (pop.querySelector('[aria-checked="true"]:not(:disabled)') || live()[0])?.focus();
  };
  pop.onclick = e => {
    const o = e.target.closest(".opt"); if (!o || o.disabled) return;
    setPicker(false); btn.focus();
    if (o.classList.contains("go")) control("v1/load", o.dataset.v);
  };
  root.onkeydown = e => {
    const opts = live(), k = opts.indexOf(document.activeElement);
    if (e.key === "Escape") { setPicker(false); btn.focus(); }
    else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!root.hasAttribute("data-open")) { setPicker(true); (pop.querySelector('[aria-checked="true"]:not(:disabled)') || opts[0])?.focus(); return; }
      opts[(k + (e.key === "ArrowDown" ? 1 : -1) + opts.length) % opts.length]?.focus();
    }
  };
  document.addEventListener("click", e => { if (!e.target.closest("#models")) setPicker(false); });
}
function drawModels(models, shown) {
  const root = $("#models"), many = models.length > 1;
  root.hidden = !many; $("#model").hidden = many;
  if (!many) return;
  if (!root.firstElementChild) buildPicker(root);
  const busy = models.some(m => m.status.value === "loading") || !!pending;
  const name = (models.find(m => m.status.value === "loading") || {}).id || shown || "";
  text(root.firstElementChild.firstElementChild, name);
  const pop = root.querySelector(".pop"), sig = models.map(m => `${m.id}|${rowWord(m)}`).join() + "|" + name + "|" + busy;
  if (pop.dataset.sig === sig) return;
  pop.dataset.sig = sig;
  const focused = pop.contains(document.activeElement) ? document.activeElement.dataset.v : null;
  pop.replaceChildren(...models.map(m => {
    const b = document.createElement("button"); b.type = "button"; b.setAttribute("role", "menuitemradio");
    const v = m.status.value, go = v === "unloaded" || v === "failed" || (v === "loaded" && !!m.asleep);
    b.className = go ? "opt go" : "opt"; b.dataset.v = m.id; b.disabled = busy;
    b.setAttribute("aria-checked", String(m.id === name));
    const n = document.createElement("b"), w = document.createElement("span");
    n.textContent = m.id; w.textContent = rowWord(m);
    b.append(n, w);
    return b;
  }));
  if (focused) [...pop.children].find(b => b.dataset.v === focused && !b.disabled)?.focus();
}
// the Unload button: for the loaded model; off while a request runs or waits, when the server would refuse it
function drawUnload(models, live) {
  const b = $("#unload"), u = unloadView(models, live);
  b.hidden = u.hidden;
  if (u.hidden) return;
  b.dataset.v = u.id; b.disabled = u.disabled; b.title = u.title; text(b, u.label);
}
$("#unload").addEventListener("click", e => control("v1/unload", e.currentTarget.dataset.v));

/* ---- the no-model screen saver: a field of basalt column tops seen from above, on its own. A soft band of light crosses it every ~40 s,
   and now and then one column flares ember and cools back to stone over ~9 s. The mark's seven columns sit at the centre, its
   middle one ember. Canvas at 8 frames a second, skipped while the tab is hidden and stopped when a model loads;
   prefers-reduced-motion draws one still frame. ---- */
const SS = { on: false, timer: 0, cells: [], hot: [], mark: [], W: 0, H: 0, R: 19, t0: 0 };
const STONE = [[26, 32, 37], [34, 42, 49], [40, 49, 58], [44, 53, 62], [31, 38, 44]];
const mix = (a, b, f) => a.map((x, k) => Math.round(x + (b[k] - x) * f));
function saverLayout() {
  const c = $("#ss"), r = c.getBoundingClientRect(), dpr = Math.min(2, devicePixelRatio || 1);
  SS.W = r.width; SS.H = r.height; c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr);
  c.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
  const R = SS.R, w = R * Math.sqrt(3), sy = R * 1.5;
  SS.cells = [];
  for (let y = 0, row = 0; y < SS.H + R; y += sy, row++)
    for (let x = row % 2 ? w / 2 : 0; x < SS.W + w; x += w) SS.cells.push({ x, y, base: STONE[(row * 7 + Math.round(x / w) * 13) % STONE.length] });
  if (!SS.cells.length) return;
  // the mark: the cell nearest the centre and its six neighbours
  const near = (px, py) => SS.cells.reduce((b, c, i) => Math.hypot(c.x - px, c.y - py) < Math.hypot(SS.cells[b].x - px, SS.cells[b].y - py) ? i : b, 0);
  const m = SS.cells[near(SS.W / 2, SS.H * .44)];
  SS.mark = [near(m.x, m.y)].concat([0, 1, 2, 3, 4, 5].map(k => near(m.x + Math.cos(Math.PI / 3 * k) * w, m.y + Math.sin(Math.PI / 3 * k) * w)));
}
function hexPath(g, x, y, R) {
  g.beginPath();
  for (let k = 0; k < 6; k++) { const a = Math.PI / 3 * k - Math.PI / 2, px = x + Math.cos(a) * (R - 1.2), py = y + Math.sin(a) * (R - 1.2); k ? g.lineTo(px, py) : g.moveTo(px, py); }
  g.closePath();
}
function saverFrame(still) {
  const g = $("#ss").getContext("2d"), t = (performance.now() - SS.t0) / 1000;
  g.clearRect(0, 0, SS.W, SS.H);
  // the light band: along the diagonal, one crossing every 40 s
  const span = SS.W + SS.H, pos = still ? span * .35 : ((t / 40 + .3) % 1) * (span + 600) - 300;
  if (!still && SS.cells.length && Math.random() < .08 && SS.hot.length < 5) {
    const i = Math.floor(Math.random() * SS.cells.length);
    if (!SS.mark.includes(i)) SS.hot.push({ i, t });
  }
  SS.hot = SS.hot.filter(h => t - h.t < 9);
  const heat = new Map(SS.hot.map(h => [h.i, 1 - (t - h.t) / 9]));
  SS.cells.forEach((c, i) => {
    const d = (c.x + c.y) - pos, band = Math.exp(-(d * d) / (2 * 200 * 200));
    let col = mix(c.base, [92, 104, 116], band * .7);
    if (SS.mark.includes(i)) col = mix(col, [70, 82, 94], .75);
    const h = heat.get(i);
    if (h != null) col = h > .7 ? mix([242, 102, 43], [255, 181, 71], (h - .7) / .3) : mix(col, h > .35 ? [242, 102, 43] : [184, 64, 26], Math.min(1, h / .35));
    if (i === SS.mark[0]) col = [242, 102, 43];
    hexPath(g, c.x, c.y, SS.R); g.fillStyle = `rgb(${col})`; g.fill();
  });
}
const still = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
function saverLoop() {
  if (!SS.on) return;
  if (!document.hidden) saverFrame(false);
  SS.timer = setTimeout(saverLoop, 125);   // 8 frames a second is plenty for a 40 s sweep and 9 s fades
}
function saver(on) {
  if (on === SS.on) return;
  SS.on = on; clearTimeout(SS.timer);
  if (!on) return;
  SS.t0 = performance.now(); SS.hot = [];
  saverLayout();
  if (still()) return saverFrame(true);
  saverLoop();
}
new ResizeObserver(() => { if (SS.on) { saverLayout(); saverFrame(true); } }).observe($(".hero"));

/* ---- the whole page ---- */
function draw() {
  if (!M) return;
  const live = M.live || {}, eng = M.engine || {}, hist = M.history || {}, reqs = M.requests || [];
  const total = (live.slots || {}).total || 1, slots = M.slots || [];
  const name = (live.state || "idle")[0].toUpperCase() + (live.state || "idle").slice(1);
  const ld = live.state === "loading" ? live.load || {} : null, weights = ld && ld.kind !== "wake";
  const models = M.models || [];
  const none = models.length > 0 && !models.some(m => (m.status.value === "loaded" && m.parked_mib == null) || m.status.value === "loading");
  const [sw, swTip] = stateWord(live, name);
  $("#state").title = swTip;
  if (failed && Date.now() > failed.until) failed = null;
  if (failed) { setState("error", failed.text); $("#state").title = failed.text; }
  else if (pending && pending.path === "v1/load") setState("loading", ACT["v1/load"][0]);
  else setState(live.state || "idle", weights ? "Loading" : ld ? "Waking up" : live.queued > 0 ? `${name} · ${live.queued} queued` : none ? models.some(m => m.status.value === "failed") ? "Load failed" : "No model" : sw);
  const pct = weights && num(ld.percent) ? ld.percent : null;
  text($("#pct"), pct === null ? "" : `${pct}%`); $("#pct").hidden = pct === null;
  $("#pbar").hidden = !weights; $("#pbar").firstElementChild.style.width = `${pct || 0}%`;
  text($("#model"), eng.model || "");
  drawModels(models, eng.model);
  drawUnload(models, live);
  drawTotals(M.totals);
  // no model: the screen saver takes the hero, with one line on how to load one
  $(".hero").classList.toggle("empty", none);
  $("#none").hidden = !none;
  if (none && !$("#ss")) html($("#none"), `<canvas id="ss" aria-hidden="true"></canvas>`);
  saver(none);
  drawHero(live, hist, reqs, M.best || {}, slots, total);
  if (!none) drawExperts(eng, reqs[0], Math.max(0, ((M.hardware || {}).gpus || []).length - 1));
  drawTabs(live, reqs, slots, total, (M.hardware || {}).gpus || []);
  // only the open pane is drawn; the others are drawn when opened
  if (tab === "req") drawReq(live, reqs, total);
  else if (tab === "slots") drawSlots(slots, slots.map((_, k) => tail((hist.slot_tok_s || [])[k])), reqs, total);
  else if (tab === "hw") drawHw(M.hardware || {}, M.hardware_static || {}, hist, eng);
}

/* ---- logs: a level class per line, a warning count, the footer ticker, a filter ---- */
// the engine's long conversation cache lines, shortened for the panel (the log and the API keep the raw line)
function shortLog(msg) {
  const k = n => n >= 1000 ? `${Math.round(n / 1000)}k` : String(n), g = b => `${(b / 1024 ** 3).toFixed(1)} GiB`;
  let m = /conversation cache: parked (\d+) tokens in [\d.]+ ms; parked=(\d+) bytes=(\d+)/.exec(msg);
  if (m) return `cache: parked ${k(+m[1])} tokens · ${m[2]} cached · ${g(+m[3])}`;
  m = /conversation cache: restored (\d+) tokens \(\w+\) in ([\d.]+) ms/.exec(msg);
  return m ? `cache: restored ${k(+m[1])} tokens in ${Math.round(+m[2])} ms` : msg;
}
const logEl = $("#log"), MAXLOG = 400;
let logCount = 0, warnCount = 0, follow = true;
const srcTag = s => s === "srv" ? "server" : s === "eng" ? "engine" : s;
function lineEl(e) {
  const d = document.createElement("div");
  d.className = e.lvl === "info" ? "" : e.lvl;
  const t = document.createElement("i"); t.textContent = clock(e.t * 1000);
  const s = document.createElement("u"); s.textContent = srcTag(e.src);
  const m = document.createElement("span"); m.textContent = shortLog(e.msg);
  d.append(t, s, m); d.title = e.msg;
  return d;
}
function append(lines) {
  if (!lines || !lines.length) return;
  logEl.querySelector(".empty")?.remove();
  logEl.append(...lines.map(lineEl));
  while (logEl.childElementCount > MAXLOG) logEl.firstChild.remove();
  logCount += lines.length; warnCount += lines.filter(l => l.lvl === "warn" || l.lvl === "err").length;
  const z = lines[lines.length - 1];
  text($("#ticker-t"), `${clock(z.t * 1000)}  ${srcTag(z.src)}  ${shortLog(z.msg)}`);
  $("#ticker").dataset.lvl = z.lvl;
  if (follow) logEl.scrollTop = logEl.scrollHeight;
  draw();
}
logEl.addEventListener("scroll", () => { follow = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 12; $("#jump").hidden = follow; });
$("#jump").onclick = () => { follow = true; logEl.scrollTop = logEl.scrollHeight; $("#jump").hidden = true; };
$("#lfilter").addEventListener("click", e => {
  const b = e.target.closest("button[data-f]"); if (!b) return;
  $("#lfilter").querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
  logEl.classList.toggle("warnonly", b.dataset.f === "warn");
  if (follow) logEl.scrollTop = logEl.scrollHeight;
});
function clearLog(note) {
  logCount = 0; warnCount = 0;
  const d = document.createElement("div"); d.className = "empty"; d.textContent = note; logEl.replaceChildren(d);
  text($("#ticker-t"), "");
  $("#ticker").dataset.lvl = "";
}

/* ---- streamed state and logs ---- */
function stream(path, receive, status) {
  let controller = null, retry = null, stopped = false;
  function cancel() {
    clearTimeout(retry); retry = null;
    controller?.abort(); controller = null;

  }
  async function connect() {
    cancel();
    if (stopped) return;
    const own = controller = new AbortController();
    let reader;
    try {
      status("connecting");
      const r = await fetch(path(), { signal: own.signal, cache: "no-store",
        headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {} });
      if (own.signal.aborted) return;
      if (r.status === 401) { status("locked"); return; }
      if (!r.ok || !r.body) throw new Error(`HTTP ${r.status}`);
      reader = r.body.getReader();
      const decoder = new TextDecoder(), parse = createLogParser(data => { receive(data); status("live"); });
      while (!own.signal.aborted) {
        const { value, done } = await reader.read();
        if (done) throw new Error("Connection closed");
        if (!own.signal.aborted) parse(decoder.decode(value, { stream: true }));
      }
    } catch (e) {
      if (own.signal.aborted) return;
      status("offline"); retry = setTimeout(connect, 2000);
    } finally {
      if (reader) { try { await reader.cancel(); } catch (_) {} reader.releaseLock(); }
    }
  }
  window.addEventListener("pagehide", () => { stopped = true; cancel(); });
  window.addEventListener("pageshow", () => { if (stopped) { stopped = false; connect(); } });
  return connect;
}
const connectState = stream(() => "api/state/stream", data => {
  M = data; settle(M.models || [], true); $("#auth").hidden = true; draw();
}, state => {
  if (state === "locked" || state === "offline") {
    $("#auth").hidden = state !== "locked"; M = null;
    setState("offline", state === "locked" ? "Locked" : "Offline");
  }
});
let lastLog = null, logInstance = null;
const connectLogs = stream(() => "api/logs/stream" + (lastLog === null ? "" :
  `?after=${lastLog}&instance=${encodeURIComponent(logInstance)}`), data => {
  const reset = data.reset || (logInstance !== null && logInstance !== data.instance_id);
  if (reset) {
    clearLog("The server restarted."); lastLog = null;
  }
  logInstance = data.instance_id;
  if (data.dropped) append([{ t: Date.now() / 1000, src: "srv", lvl: "warn", msg: "Some log lines were dropped from the server buffer." }]);
  const fresh = (data.lines || []).filter(line => lastLog === null || line.id > lastLog);
  append(fresh);
  lastLog = data.next;
  if (!logCount) clearLog("No log lines yet.");
  if (reset && !fresh.length && !data.dropped) draw();
}, state => text($("#log-status"), { connecting: "Connecting", locked: "API key required",
  live: "Live", offline: "Disconnected; retrying" }[state]));
$("#auth").onsubmit = e => {
  e.preventDefault(); apiKey = $("#key").value.trim(); sessionStorage.setItem("basalt.key", apiKey);
  connectState(); connectLogs();
};

/* ---- pricing dialog ---- */
/* ---- support: the footer heart opens a quiet dialog of links; it never opens by itself ---- */
function openSupport() {
  let d = $("#support-dlg");
  if (!d) {
    d = document.createElement("dialog"); d.id = "support-dlg"; d.className = "price";
    const out = `<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M6 3h7v7M13 3 5 11" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
    const link = (href, name, note) => `<a class="support-link" href="${href}" target="_blank" rel="noopener noreferrer"><b>${name}</b><span>${note}</span>${out}</a>`;
    d.innerHTML = `<form method="dialog">
<div class="price-head"><h3>Support Basalt</h3><p>Basalt is free and MIT licensed. Sponsorship pays for the tokens behind its development and for time on more hardware to test on.</p></div>
<div class="support-links">${link("https://github.com/sponsors/jesdga95", "GitHub Sponsors", "monthly or one-time")}${link("https://ko-fi.com/jesdga", "Ko-fi", "one-time")}${link("https://paypal.me/jesdga", "PayPal", "one-time")}</div>
<p class="support-alt">No money needed to help: a <a href="https://github.com/jesdga95/basalt/blob/main/docs/community.md" target="_blank" rel="noopener noreferrer">community benchmark run</a> from your hardware helps too.</p>
<div class="price-foot"><span class="grow"></span><button type="submit" class="price-ghost">Close</button></div></form>`;
    document.body.append(d);
    d.onclick = e => { if (e.target === d) d.close(); };
  }
  d.showModal();
}
$("#support").onclick = openSupport;

let lastTotals = null;
const priceVal = id => { const v = $(id).value.trim(); return v === "" ? null : Math.max(0, parseFloat(v) || 0); };
// the dialog's live figure: the same formula as serve/totals.py, from the counts and the rates as typed
function pricePreview() {
  const t = lastTotals || {}, inp = priceVal("#p-in") ?? 0, out = priceVal("#p-out") ?? 0;
  const cw = priceVal("#p-cw") ?? inp, ch = priceVal("#p-ch") ?? inp;
  const given = t.prompt || 0, held = Math.min(t.reused || 0, given);
  const cost = ((given - held) * cw + held * ch + (t.written || 0) * out) / 1e6;
  $("#p-cost").textContent = usd(cost);
  for (const id of ["#p-cw", "#p-ch"]) $(id).placeholder = String(inp);
}
function openPricing() {
  let d = $("#pricing-dlg");
  if (!d) {
    d = document.createElement("dialog"); d.id = "pricing-dlg"; d.className = "price";
    const field = (id, label, ph = "") => `<label class="price-f"><span>${label}</span><span class="price-in"><i>$</i>`
      + `<input id="${id}" type="number" inputmode="decimal" step="any" min="0" placeholder="${ph}" autocomplete="off"></span></label>`;
    d.innerHTML = `<form method="dialog" id="p-form">
<div class="price-head"><h3>Pricing and totals</h3><p>What this server's tokens would cost at API prices, in US dollars per million tokens.</p></div>
<div class="price-sum"><div class="cost" id="p-cost">$0</div><div class="meta" id="p-meta"></div></div><div class="price-since" id="p-since"></div>
<div class="price-grid">${field("p-in", "Input", "0")}${field("p-out", "Output", "0")}${field("p-cw", "Cache write")}${field("p-ch", "Cache hit")}
<div class="price-note">Leave a cache price empty to charge it like input.</div></div>
<div class="price-foot"><button type="button" class="price-reset" id="p-reset">Reset totals</button><span class="price-msg" id="p-msg"></span>
<span class="grow"></span><button type="button" class="price-ghost" id="p-close">Cancel</button><button type="submit" class="price-save">Save</button></div></form>`;
    document.body.append(d);
    $("#p-form").onsubmit = e => { e.preventDefault(); savePricing(); };
    $("#p-close").onclick = () => d.close();
    $("#p-reset").onclick = resetTotals;
    d.addEventListener("input", pricePreview);
    d.onclick = e => { if (e.target === d) d.close(); };
  }
  const t = lastTotals || {}, ci = t.cost_input_per_million || 0;
  $("#p-in").value = ci || "";
  $("#p-out").value = t.cost_output_per_million || "";
  $("#p-cw").value = t.cost_cache_write_per_million != null && t.cost_cache_write_per_million !== ci ? t.cost_cache_write_per_million : "";
  $("#p-ch").value = t.cost_cache_hit_per_million != null && t.cost_cache_hit_per_million !== ci ? t.cost_cache_hit_per_million : "";
  html($("#p-meta"), priceMeta(t));
  $("#p-since").textContent = t.since ? `Counting since ${new Date(t.since * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" })}` : "";
  const reset = $("#p-reset"); reset.textContent = "Reset totals"; reset.classList.remove("armed");
  priceMsg("");
  pricePreview();
  d.showModal();
  $("#p-in").focus();
}
// the summary card's right side, two lines: requests, then tokens in and out, short (2.4M, 1.28B)
const short = v => v >= 1e9 ? `${(v / 1e9).toFixed(2)}B` : v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `${(v / 1e3).toFixed(1)}K` : String(v);
const priceMeta = t => `<b>${short(t.requests || 0)}</b> requests<br><b>${short(t.prompt || 0)}</b> in · <b>${short(t.written || 0)}</b> out`;
function priceMsg(text, kind = "") { const m = $("#p-msg"); m.textContent = text; m.className = "price-msg" + (kind ? " " + kind : ""); }
const authHeaders = () => ({ "Content-Type": "application/json", ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}) });
async function savePricing() {
  const body = { cost_input_per_million: priceVal("#p-in") ?? 0, cost_output_per_million: priceVal("#p-out") ?? 0,
                 cost_cache_write_per_million: priceVal("#p-cw"), cost_cache_hit_per_million: priceVal("#p-ch") };
  try {
    const r = await fetch("api/pricing", { method: "POST", headers: authHeaders(), body: JSON.stringify(body) });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) return priceMsg(d.error?.message || `Error ${r.status}`, "err");
    $("#pricing-dlg").close();
  } catch (e) { priceMsg(e.message, "err"); }
}
// two clicks: the first arms it (and says what it does), the second resets
async function resetTotals() {
  const b = $("#p-reset");
  if (!b.classList.contains("armed")) {
    b.classList.add("armed"); b.textContent = "Click again to reset";
    priceMsg("the current totals are kept in .prev.json");
    clearTimeout(b._t); b._t = setTimeout(() => { b.classList.remove("armed"); b.textContent = "Reset totals"; priceMsg(""); }, 4000);
    return;
  }
  clearTimeout(b._t); b.classList.remove("armed"); b.textContent = "Reset totals";
  try {
    const r = await fetch("api/totals-reset", { method: "POST", headers: authHeaders(), body: "{}" });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) return priceMsg(d.error?.message || `Error ${r.status}`, "err");
    priceMsg("Totals reset", "ok");
    lastTotals = { ...(lastTotals || {}), requests: 0, prompt: 0, reused: 0, written: 0, since: Date.now() / 1000 };
    html($("#p-meta"), priceMeta(lastTotals)); $("#p-since").textContent = "Counting since now"; pricePreview();
  } catch (e) { priceMsg(e.message, "err"); }
}

$("#host").textContent = location.host;
fetch("health").then(r => r.json()).then(h => { $("#keyed").hidden = !h.api_key; }).catch(() => {});
drawPanes(); connectState(); connectLogs();
