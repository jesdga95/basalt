"use strict";
const assert = require("node:assert/strict");
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
console.log("Log SSE parser: passed");
