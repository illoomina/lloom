// The Web Worker. It loads the wasm module and runs the CPU-backend smoke
// computation off the main thread -- so a hung decode is a worker.terminate()
// (the interrupt story evcxr cannot offer, #656 step 3).
import init, { smoke, build_info } from "./pkg/lloom_wasm.js";

let ready = false;

self.onmessage = async (ev) => {
  const { cmd } = ev.data;
  try {
    if (!ready) {
      await init(); // fetches and instantiates ./pkg/lloom_wasm_bg.wasm
      ready = true;
    }
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
    } else {
      self.postMessage({ ok: false, cmd, error: "unknown cmd" });
    }
  } catch (e) {
    self.postMessage({ ok: false, cmd, error: String(e) });
  }
};
