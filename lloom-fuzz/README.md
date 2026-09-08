# lloom-fuzz — the LibAFL campaign over the engine's stateful fuzz models

Issue #706. One harness binary per target; the models are the ones the
proptests drive (behind the engine crates' `fuzzing` feature), plus two of this
crate's own: the paged KV pool and command-buffer liveness under retained and
unretained command buffers.

| binary | model | device |
|---|---|---|
| `fuzz_token_ring` | `lloom::generation::token_ring_fuzz` (input ring, EOS flag, turns; `INPUT_POISON` on never-staged slots) | CPU + Metal |
| `fuzz_kv_slot` | `lloom::models::lfm2::kv_slot_fuzz` (append / speculate / rewind / grow / reset) | CPU |
| `fuzz_kv_pages` | `lloom_fuzz::kv_pages_model` (free list, refcounts, quarantine epochs, grow, prefix sharing) | none |
| `fuzz_buffer_pool` | `buffer_pool_fuzz` (counters, occupancy, eviction, residual cap) | Metal (allocator only) |
| `fuzz_commands` | `commands_fuzz` (`in_flight` bounded across commit / prune / wait) | Metal |
| `fuzz_scratch` | `scratch_fuzz` (`plan_scratch` over a long-context walk) | none |
| `fuzz_cb_liveness` | `lloom_fuzz::cb_liveness` (copies of pool buffers in flight while the host drops / drains / trims them; `--cb retained\|unretained`) | Metal |

## Build

This crate is its own workspace. SanitizerCoverage goes on the engine crates
and this crate's own code — and on nothing else — through `sancov-wrapper.sh`,
the `rustc-wrapper` in `.cargo/config.toml` keyed on `--crate-name` (rustc has
no allowlist; instrumenting every dependency measured 1.18 M guards for a 2 MiB
map scanned per execution). Two LTO arms of the release codegen:

```sh
cd candle/lloom-fuzz
cargo build --profile fuzz-thin --features metal            # pc-guard arm, thin LTO
cargo build --profile fuzz-fat  --features metal            # pc-guard arm, fat LTO
cargo build --profile fuzz-thin --features metal,cmplog     # cmplog arm (the main)
```

Binaries land in `target/aarch64-apple-darwin/<profile>/`. Every binary prints
its instrumentation on start — `guards=N edges_map=M` — and refuses to fuzz at
`guards=0`:

```sh
target/aarch64-apple-darwin/fuzz-thin/fuzz_kv_pages --instrumentation
```

If `N` approaches `M`, rebuild with `LIBAFL_EDGES_MAP_DEFAULT_SIZE=<bigger>` in
the environment (it is read by `libafl_targets`' build script).

## Run

One main (the cmplog build) and one secondary (the pc-guard build) share a
corpus through one LLMP broker — AFL++'s `-M`/`-S` in substance:

```sh
# terminal 1: the main, spawns the broker
LLOOM_FUZZ_MAIN=target/aarch64-apple-darwin/fuzz-thin-cmplog   # wherever the cmplog build went
$LLOOM_FUZZ_MAIN/fuzz_kv_pages --cores 0 --port 1337 --corpus out/kv_pages/corpus --solutions out/kv_pages/solutions
# terminal 2: the secondary, joins the broker
target/aarch64-apple-darwin/fuzz-thin/fuzz_kv_pages --cores 1 --port 1337 --broker false --corpus out/kv_pages/corpus --solutions out/kv_pages/solutions
```

(Two builds with the same profile name overwrite each other's binaries; give
the cmplog build `--target-dir target-cmplog`. `--client-log FILE` captures the
clients' stdout/stderr, which the launcher otherwise silences.)

Run campaigns under `lloom-arb share` so they never silently perturb a
measurement (`CONTRIBUTING.md` §3.2b).

## The persistent record

Every client keeps the last `--record N` (default 64) inputs it executed. On an
oracle failure, a panic or a signal the whole ring is written to
`--record-dir` as `record-<pid>-<k>/0000.bin … NNNN.bin` (oldest first, the
dying input last) with `reason.txt`. Replay it in one fresh process:

```sh
fuzz_cb_liveness --cb unretained --replay out/records/record-12345-0
fuzz_cb_liveness --cb unretained --once  out/solutions/<id>          # the crashing input alone
```

A failure that reproduces from the record and not from the single input is a
state-carryover failure — the class the record exists for. `record_selftest`
is the mechanism's validation target: a *planted* dependency (an input whose
first byte is `0x43` fails iff the previous input's first byte was `0x42`),
labelled as such because no engine harness carries state across inputs.
Measured: the campaign's solution replays clean under `--once`; the record
(9 inputs, `0006.bin` = `42…`, `0007.bin` = `43…`) fails at input 7.

## Poisons

- `INPUT_POISON` / `TILE_POISON` (token ring): kept from the proptests; the
  `fuzzing` feature turns the tile poison's write on in release codegen.
- The pool's **free poison** (`BufferPool::set_free_poison`, `fuzzing` only):
  a buffer is overwritten with `0xDE` the moment it joins the free list, so a
  command buffer still reading it reads the sentinel; `cb_liveness` verifies
  every copy against what it wrote.
- The paged pool has no bytes to poison; its partition invariant (every page
  in exactly one of free / allocated / quarantined) is the sentinel.
