// Headless-Chrome CDP driver for the #656 oracle check. Opens the page (which
// auto-runs smoke() inside a Web Worker), waits for the fingerprint, prints it.
// Uses only node built-ins (global WebSocket, fetch).

const CDP = "http://localhost:9876";
const URL = "http://localhost:8656/index.html";

const ver = await (await fetch(`${CDP}/json/version`)).json();
// Open a fresh tab at the target URL.
const tab = await (await fetch(`${CDP}/json/new?${encodeURIComponent(URL)}`, { method: "PUT" })).json()
  .catch(async () => (await fetch(`${CDP}/json/new?${encodeURIComponent(URL)}`)).json());

const ws = new WebSocket(tab.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
function send(method, params = {}) {
  const msgId = ++id;
  return new Promise((res) => {
    pending.set(msgId, res);
    ws.send(JSON.stringify({ id: msgId, method, params }));
  });
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
    if (text.startsWith("SMOKE_RESULT")) done("smoke");
  }
  if (m.method === "Runtime.exceptionThrown") {
    consoleLines.push("EXCEPTION " + JSON.stringify(m.params.exceptionDetails?.exception?.description || m.params.exceptionDetails));
  }
});

await new Promise((r) => ws.addEventListener("open", r));
await send("Runtime.enable");
await send("Page.enable");
await send("Log.enable");

// Poll window.__SMOKE__ as a fallback in case the console event is missed.
const deadline = Date.now() + 30000;
await Promise.race([
  finished,
  (async () => {
    while (Date.now() < deadline) {
      const r = await send("Runtime.evaluate", {
        expression: "window.__SMOKE__ ? JSON.stringify(window.__SMOKE__) : null",
        returnByValue: true,
      });
      const v = r?.result?.value;
      if (v && v !== "null") { done("poll:" + v); break; }
      await new Promise((r) => setTimeout(r, 300));
    }
  })(),
]);

// Final read of the smoke object + title.
const finalR = await send("Runtime.evaluate", {
  expression: "JSON.stringify({ smoke: window.__SMOKE__ || null, title: document.title })",
  returnByValue: true,
});
console.log("=== CONSOLE ===");
for (const l of consoleLines) console.log(l);
console.log("=== FINAL ===");
console.log(finalR?.result?.value ?? "(none)");

ws.close();
process.exit(0);
