# Test fixtures

Committed inputs that `lloom`'s tests read with `CARGO_MANIFEST_DIR`-relative
paths. Each file is read as data; nothing here is generated at build time.

| file | read by | what it is |
|---|---|---|
| `axes-manifest.txt` | `axes_registry.rs` | the names `lloom::axes::AXES` collects, one per line. This crate is its authority: regenerate with `LLOOM_WRITE_AXES_MANIFEST=1 cargo test -p lloom --features metal --test axes_registry` (the Metal build carries every axis). |
| `lfm2-trace-arena-shuffle-branch.txt` | `optimizer_derivation.rs` | an `lfm2-dispatch-trace --arena --attn sdpa --kv-append in-place --conv-state shuffle --param-style split --hazard-key pointer` run of the LFM2.5-2.6B decode step: the per-kernel dispatch census the walk of the model must reproduce. |
| `qwen4exp-trace-dev-bfc70b8f4.txt` | `qwen4exp_walk.rs` | `qwen4exp-chat` under `CANDLE_METAL_TRACE=1` on a prose prompt: the `TRACE run=… step=… kernel=… n=…` census of each decode step. |
| `qwen4exp-trace-needle-dev-a5c815f13.txt` | `qwen4exp_walk.rs` | the same trace on a 3 749-token prompt, past the dense attention range. |

The three traces are byte-identical copies of recorded runs, named for the
commit they were taken at. A trace is replaced by a new run, never edited; a
test that needs a re-take reads a new file.
