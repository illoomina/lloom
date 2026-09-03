// Headless-Chrome CDP driver for the #623 LFM2 oracle check. Opens
// notebook.html (which auto-runs the LFM2 cell in a Web Worker), waits for the
// decoded token + fingerprint, and prints them so they can be compared against
// the native `reference_lfm2` oracle byte-for-byte (§2.3.5a).
//
// Env:
//   CDP   headless-Chrome CDP endpoint (default http://localhost:9876)
//   PAGE  the served notebook URL (default http://localhost:8656/notebook.html)
//   GGUF  a GGUF URL to override the page default (served locally to avoid a
//         219 MB network fetch on every run)
// Uses only node built-ins (global WebSocket, fetch).

const CDP = process.env.CDP || "http://localhost:9876";
const GGUF = process.env.GGUF || "";
let PAGE = process.env.PAGE || "http://localhost:8656/notebook.html";
if (GGUF) PAGE += (PAGE.includes("?") ? "&" : "?") + "gguf=" + encodeURIComponent(GGUF);

const tab = await (await fetch(`${CDP}/json/new?${encodeURIComponent(PAGE)}`, { method: "PUT" })).json()
  .catch(async () => (await fetch(`${CDP}/json/new?${encodeURIComponent(PAGE)}`)).json());

const ws = new WebSocket(tab.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
function send(method, params = {}) {
  const msgId = ++id;
  return new Promise((res) => { pending.set(msgId, res); ws.send(JSON.stringify({ id: msgId, method, params })); });
}

const consoleLines = [];
let done = null;
const finished = new Promise((r) => (done = r));

ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); return; }
  if (m.method === "Runtime.consoleAPICalled") {
    const text = (m.params.args || []).map((a) => a.value ?? a.description ?? "").join(" ");
    consoleLines.push(text);
    if (text.startsWith("LFM2_RESULT")) done("lfm2");
  }
  if (m.method === "Runtime.exceptionThrown") {
    consoleLines.push("EXCEPTION " + JSON.stringify(m.params.exceptionDetails?.exception?.description || m.params.exceptionDetails));
  }
});

await new Promise((r) => ws.addEventListener("open", r));
await send("Runtime.enable");
await send("Page.enable");
await send("Log.enable");

// LFM2 load + forward is slower than the #656 micro-op; give it room.
const deadline = Date.now() + 180000;
await Promise.race([
  finished,
  (async () => {
    while (Date.now() < deadline) {
      const r = await send("Runtime.evaluate", {
        expression: "window.__LFM2__ ? JSON.stringify(window.__LFM2__) : null",
        returnByValue: true,
      });
      const v = r?.result?.value;
      if (v && v !== "null") { done("poll:" + v); break; }
      await new Promise((r) => setTimeout(r, 500));
    }
  })(),
]);

const finalR = await send("Runtime.evaluate", {
  expression: "JSON.stringify({ lfm2: window.__LFM2__ || null, title: document.title })",
  returnByValue: true,
});
console.log("=== CONSOLE ===");
for (const l of consoleLines) console.log(l);
console.log("=== FINAL ===");
console.log(finalR?.result?.value ?? "(none)");

ws.close();
process.exit(0);
