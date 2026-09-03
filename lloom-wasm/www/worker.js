// The Web Worker. It loads the wasm module and runs the CPU-backend
// computation off the main thread -- so a hung decode is a worker.terminate()
// (the interrupt story evcxr cannot offer, #656 step 3).
import init, { smoke, build_info, decode_lfm2 } from "./pkg/lloom_wasm.js";

let ready = false;
// The GGUF bytes, fetched once and reused across decodes so re-running a cell
// does not re-download 219 MB.
let gguf = null;

async function ensureReady() {
  if (!ready) {
    await init(); // fetches and instantiates ./pkg/lloom_wasm_bg.wasm
    ready = true;
  }
}

self.onmessage = async (ev) => {
  const { cmd } = ev.data;
  try {
    await ensureReady();
    if (cmd === "build_info") {
      self.postMessage({ ok: true, cmd, result: build_info() });
    } else if (cmd === "smoke") {
      const t0 = performance.now();
      const result = smoke(); // one CPU forward computation
      const ms = performance.now() - t0;
      // Warm timing: repeat and take the median, so the reported latency is the
      // steady-state cost, not the cold JIT/instantiate hit of the first call.
      const N = 1000, samples = [];
      for (let i = 0; i < N; i++) { const s = performance.now(); smoke(); samples.push(performance.now() - s); }
      samples.sort((a, b) => a - b);
      const warm_median_ms = samples[Math.floor(N / 2)];
      self.postMessage({ ok: true, cmd, result, ms, warm_median_ms });
    } else if (cmd === "lfm2") {
      // A real LFM2 token, decoded in the browser (#623). `ev.data.url` is the
      // GGUF to fetch (once); `ev.data.prompt` is the cell text.
      if (gguf === null) {
        self.postMessage({ ok: true, cmd: "status", result: "fetching weights…" });
        const resp = await fetch(ev.data.url);
        if (!resp.ok) throw new Error(`fetch ${ev.data.url}: HTTP ${resp.status}`);
        gguf = new Uint8Array(await resp.arrayBuffer());
        self.postMessage({ ok: true, cmd: "status", result: `weights loaded (${(gguf.length / 1048576).toFixed(1)} MB)` });
      }
      const t0 = performance.now();
      // decode_lfm2 takes ownership of the byte view; pass a copy so a re-run
      // still has `gguf`.
      const result = decode_lfm2(gguf.slice(), ev.data.prompt);
      const ms = performance.now() - t0;
      self.postMessage({ ok: true, cmd, result, ms });
    } else {
      self.postMessage({ ok: false, cmd, error: "unknown cmd" });
    }
  } catch (e) {
    self.postMessage({ ok: false, cmd, error: String(e) });
  }
};
