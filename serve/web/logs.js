"use strict";

// Streaming fetch keeps the bearer key in headers, never in the URL.
function createLogParser(accept) {
  let pending = "";
  return chunk => {
    pending += chunk;
    if (pending.length > 8 * 1024 * 1024) throw new Error("Oversized log frame");
    let boundary;
    while ((boundary = pending.indexOf("\n\n")) >= 0) {
      const frame = pending.slice(0, boundary);
      pending = pending.slice(boundary + 2);
      const data = frame.split("\n").filter(x => x.startsWith("data: ")).map(x => x.slice(6)).join("\n");
      if (data) accept(JSON.parse(data));
    }
  };
}
if (typeof module !== "undefined") module.exports = { createLogParser };
