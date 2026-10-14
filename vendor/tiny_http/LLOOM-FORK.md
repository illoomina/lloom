# lloom's pinned tiny_http transport repair

Base: tiny_http0.12.0, upstream Git212b1c45852fef2093dc1374875a9393c55eb4b9,
registry checksum389915df6413a2e74fb181895f933386023c71110878cd0825588928e64cdc82.
Original Cargo.toml.orig, MIT and Apache2 licenses, source/tests/examples are retained.
The original42-file manifest belongs in the accompanying issue947 evidence.

This owned path is intended for candle's public mirror, not a machine cache path.
Production version and existing HTTP/HTTPS constructors remain unchanged.
The incumbent sequential writer std::Mutex/poison-unwrap contract remains on the
legacy route; configured mode uses bounded try_lock and refuses poison instead
of proceeding with poisoned writes. New disposal metadata uses parking_lot.

Changes: fixed EqualReader destructor workspace; opt-in plain HTTP exact-origin
cancellation, finite per-I/O/handoff wait, explicit unread-body time/work policy.
See TransportPolicy docs. Header/chunk parser storage, thread cardinality,
application producer execution and model/native retirement are NOT bounded here.

Server lifecycle (`src/lifecycle.rs`): `Server::shutdown(deadline) ->
ShutdownReport`, `shutdown_blocking`, a `Drop` that queues the same retirement
on a joined service thread instead of blocking, and `join_retirement` for
process exit. The pool retains every thread's handle; each accepted connection
is registered so it can be read-closed at shutdown and cut at the deadline.
Report and controls: design/measurements/issue-947-http-shutdown.md.

## Offline library controls versus preserved legacy development

Main Cargo.toml preserves the production and optional SSL dependency graph,
using serde_json for the JSON documentation example. Its independent genuine
lock was resolved through an explicitly bounded fork-only metadata operation;
a separately authorized locked Darwin fetch obtained eight missing archives,
5,753,029 archive bytes. Neither operation built SSL or ran network tests.
Exact locks, checksums and commands are retained with the issue947 evidence.
The websocket example and nightly benchmark still need upstream rustc-serialize0.3,
sha1 0.6 and fdlimit0.1. The legacy-dev graph was not resolved, fetched or tested.

The complete preserved development entries are explicitly registered in
legacy-dev/Cargo.toml, using the SAME ../src/lib.rs implementation and original
../examples, ../tests and ../benches paths. Nothing was deleted or replaced with
an empty test. Its independent workspace prevents those old dev-only crates
from entering the production application's offline dependency resolution.

Commands (NOT executed during source spike; require dependencies/toolchain):

    cargo test --manifest-path candle/vendor/tiny_http/Cargo.toml --lib --offline
    cargo test --manifest-path candle/vendor/tiny_http/legacy-dev/Cargo.toml
    cargo run --manifest-path candle/vendor/tiny_http/legacy-dev/Cargo.toml --example websockets
    cargo +nightly bench --manifest-path candle/vendor/tiny_http/legacy-dev/Cargo.toml --bench bench

The latter three resolve legacy development dependencies if available; network
resolution requires a separate explicit permission, not an implicit exception
to the current offline/locked build envelope. Their source preservation is NOT
an executed compatibility gate. SSL remains legacy/uncontrolled, and its
optional feature dependency/toolchain tests remain separately unexecuted.

Consumers can select Server::http_with_policy with explicit TransportPolicy.
No default numeric policy has been invented; unconfigured legacy constructors
retain legacy time/work semantics (but the fixed destructor workspace repair).
