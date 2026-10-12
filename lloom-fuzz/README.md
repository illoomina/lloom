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

Campaigns and sanitizer runs are heavy workloads: use an exclusive
`lloom-arb measure -- timeout <bound> <reviewed-campaign-driver>` lease, with
explicit worker and memory bounds. A build lease permits concurrent builds and
is not a memory quota. Do not use `lloom-arb share` for these runs.

The main/secondary examples above remain supported, but run the entire bounded
process group under **one** lease, not independent terminal jobs outside it.
Retain a supervisor until its children are stopped and reaped; a timeout of the
launcher alone does not prove that clients or the broker exited. On the shared
NAS, acquire the NAS-local shared resource **outside** the container, wrapping
the complete invocation. See issue #958's shared-machine contract and
`CONTRIBUTING.md` §3.2b. New campaigns, sanitizer arms and native/checkpoint
workloads still require the coordinator's bounded execution-plan review;
these examples are not permission to start them concurrently.

## The ASan arm

The sanitizer goes on the engine crates and the harness — `LLOOM_FUZZ_ASAN=1`
makes `sancov-wrapper.sh` add `-Zsanitizer=address` to the same crates it
instruments — not on LibAFL (a global `-Zsanitizer=address` does not link on
this target: `ld: initializer pointer has no target` in the `ctor` crate).
No LTO, so the guard count is ~1.28 M and the map must be 2 MiB:

```sh
LIBAFL_EDGES_MAP_DEFAULT_SIZE=2097152 LLOOM_FUZZ_ASAN=1 \
  cargo +nightly build --profile fuzz-asan --target-dir target-asan --bin fuzz_kv_slot --bin fuzz_token_ring
nm target-asan/aarch64-apple-darwin/fuzz-asan/fuzz_kv_slot | grep -c __asan     # 3209: the runtime is linked
```

Measured: 3 209 `__asan` symbols, 14 049 shadow-address sites; `kv_slot` 60 s
corpus 109 and `token_ring --device cpu` 60 s corpus 95, **0 reports** on
both (`ASAN_OPTIONS=verbosity=1 … --once` prints `AddressSanitizer Init done`,
so the runtime is live). The Metal harnesses are not built under ASan: the
engine's own `sanitize` profile covers the device path (§15.1g).

## The persistent record

Every client keeps the last `--record N` (default 64) inputs it executed.
Oracle failure, panic and signal paths attempt capture under `--record-dir`.
Capture is not guaranteed: a busy byte-ring snapshot is refused, filesystem
errors can prevent a dump, and the signal writer/handler composition still
needs its separate safety repair. Do not infer successful capture from a crash.

The signal handler chains to the action it replaced, with that action's
handler ABI, mask, `SA_NODEFER`, `SA_ONSTACK`, `SA_RESTART` and default/ignore
disposition — everything a `sigaction` query reports. **Darwin's query does
not report `SA_RESETHAND`**, so a predecessor registered one-shot stays
wrapped after its first delivery where the kernel alone would revert it to
`SIG_DFL` (SIGSEGV, SIGBUS, SIGABRT, SIGFPE; SIGILL ignores the flag). The
handlers actually replaced — LibAFL's and Rust std's — do not set it.

New campaign directories use `lloom-input-record-v3`: numbered `.bin` inputs
(oldest first), `reason.txt`, `record.format`, required `record.identity`, and a
final `record.complete` count/length/SHA-256 manifest. The low-level legacy recorder
still writes v1. Failed writes leave inspectable partial evidence, not a
successful directory result. Filesystem I/O allocates and may block; it is not
async-signal-safe. Replay validates format, completion, numeric order, sizes and all v3 content digests
before executing any input. Missing/oversized v2/v3 identity is an error, including
under the diagnostic override. Legacy numbered directories and v1/v2 records remain readable. Unidentified
records warn about identity; records without content digests separately warn
about content verification.

Campaign identity is prepared once at startup: streaming SHA-256 of the file at
`current_exe()`, target, instrumentation arm, input schema, timeout/max input
length and explicit target options. Comparison is exact serialized bytes, not
semantic JSON equivalence. Replay requires the matching binary and options.
`--replay-unchecked-identity` permits diagnostic replay of an incompatible record
with an explicit warning; it does not bypass format/completion/size checks.

Identity does NOT capture environment tuning, dylibs, hardware, driver or a
controlled schedule. The executable hash is not an immutable loaded-image
attestation. V3 records verify every input digest before the first target and recheck the
actual owned bytes at invocation, detecting same-length input replacements.
This is not authenticated storage or an atomic multi-file snapshot: jointly
rewriting the manifest and inputs is outside this integrity contract. Replay bounds directory traversal, input count,
aggregate metadata and total declared input bytes with configurable limits.
These are logical work limits, not an RSS or allocator quota. Signal-safe capture,
record-ring allocation admission and writer-side archive bounds remain open. Do not modify a record during replay.
Replay in one fresh process:

```sh
fuzz_cb_liveness --cb unretained --replay out/records/record-12345-0
fuzz_cb_liveness --cb unretained --once out/solutions/<id>
# Deliberately different binary/options: diagnostic only, never exact replay.
fuzz_cb_liveness --cb unretained --replay out/records/record-12345-0 --replay-unchecked-identity
```

`--once` and replay share the 64 KiB input-byte limit. Single-input execution
checks the file size before reading and keeps a streaming limit if the file grows;
oversized inputs are refused before recording or target execution. This is not a
process-memory cap or verification of build/schedule identity.

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

### Replay work limits

All existing replay formats remain supported. Defaults are 4096 inputs, 8192
visited directory entries (including ignored non-input files), 8 MiB aggregate
format/manifest/identity/input-path bytes and 256 MiB total declared input bytes.
Each input remains limited to 64 KiB. Validation finishes before the first target;
metadata files are size-checked and streamed within their remaining budget.
Directory enumeration may consume one refusal lookahead entry.

Larger archives can explicitly raise `--replay-max-inputs`, `--replay-max-entries`,
`--replay-max-metadata-bytes` and `--replay-max-total-input-bytes`; these flags
require `--replay`. Programmatic callers use `ReplayLimits` with
`replay_with_identity_and_limits`. Neither larger budgets nor the identity
override bypass completion, per-input size or v3 content integrity. Allocation
overheads are bounded by counts but not included in metadata-byte accounting;
filesystem latency and concurrent file replacement are not bounded by these quotas.

### Recorder setup allocation budget

The campaign installs the existing recorder ring fallibly, with a default
64 MiB **requested ring-layout** quota (`--record-max-bytes`). This includes
`N * MAX_INPUT` slot payload bytes and the slot/length metadata-array layouts.
Checked layouts and the entire quote are validated before any ring allocation;
metadata uses fallible reservation, and each zeroed slot returns a refusal on
allocation failure. `--record-max-bytes <larger>` retains larger history support.
Push/snapshot ordering and their nonblocking atomic permit are unchanged.

This is not a process RSS cap: allocator rounding, concurrent setup attempts,
identity/path storage and allocating dump I/O are excluded. Legacy `Record::install`
remains infallible/first-wins; programmatic identified callers can use
`Record::install_with_identity_and_limits` for explicit recovery. Default identified
installation uses the same quota. Async-signal-safe dump I/O and complete schedule
fuzzing remain unresolved; this repair does not diagnose or fix the GPU stall.
