//! The front end's per-token seam, reproduced so it can be priced (lloom #412).
//!
//! # What this is a copy of, and why it is a copy
//!
//! Ambrogio streams tokens out of its decode loop through this shape
//! (`amber/crates/ambrogio-models/src/llm/lfm2.rs`, the `generate_stream` arm):
//!
//! ```ignore
//! let (tx, rx) = tokio::sync::mpsc::channel::<Result<CompletionChunk, _>>(32);
//! tokio::task::spawn_blocking(move || {
//!     inner.generate_raw(&prompt, &opts, |frag| {
//!         let _ = emit_tx.blocking_send(Ok(CompletionChunk::Text {
//!             text: frag.to_string(),
//!         }));
//!     });
//! });
//! ```
//!
//! and `generate_raw`'s own loop calls that callback with a `&str` it produced
//! by `tokenizer.decode(&generated[pending..], false)` — so the per-token cost
//! of "being a server" is **three** terms, not one:
//!
//! | term | what it is |
//! |---|---|
//! | **detokenize** | `Tokenizer::decode` over the not-yet-emitted run |
//! | **allocate** | `frag.to_string()` — a fresh `String` per fragment |
//! | **send** | `blocking_send` into a bounded channel at capacity 32 |
//!
//! **It is a copy rather than a measurement of ambrogio itself**, and that is a
//! decision with a cost. `CONTRIBUTING.md` §4 forbids modifying ambrogio, and
//! the monorepo graft (#402) left `amber/Cargo.toml`'s `[patch.crates-io]`
//! pointing at `deps/candle`, which does not exist under `amber/` — so the
//! assistant does not currently resolve in this tree at all and could not have
//! been instrumented in place. What a copy buys is that every other axis is held
//! at the value the corpus's own figures were taken at: this is
//! `lfm2-decode-profile`, so `wall`, `gpu busy`, the dispatch count and the
//! sample bucket mean exactly what they mean on every recorded row.
//!
//! What a copy costs is stated rather than hidden: it prices **this shape**, and
//! it is evidence about ambrogio's seam only to the extent the shape is the
//! same. The three terms above are quoted from that file at the commit named in
//! the measurements artifact, and the assertion that they are the same shape is
//! a **reading**, not a measurement.
//!
//! # Why the consumer is a real task and not a drain
//!
//! A bounded channel's cost is a function of whether it is **full**. Sending
//! into a channel nobody reads fills 32 slots and then blocks forever; sending
//! into one drained instantly never blocks at all, and neither is the shipped
//! behaviour. Ambrogio's consumer is `turn_streaming`, which runs a
//! `SentenceSplitter` over each delta — so the arm here spawns a real task that
//! receives and runs the same shape of work, and the channel is exercised at
//! its actual occupancy rather than at either extreme.
//!
//! # What is deliberately NOT reproduced
//!
//! The tool-call parse and the `Done` chunk are once-per-generation, not
//! per-token, so they are outside the quantity #412 asks about. The
//! `SentenceSplitter` itself is ambrogio's and lives a crate away; what runs
//! here is a boundary scan of the same order, named `splitter_like` so no reader
//! mistakes it for the real one.

use std::time::Instant;

/// One streamed increment. Field-for-field the `Text` arm of ambrogio's
/// `CompletionChunk`, which is what makes `to_string()`'s allocation the same
/// allocation rather than a similar one.
pub enum Chunk {
    Text { text: String },
}

/// Per-token seam timings, in seconds. Kept as three terms rather than a total
/// because they are three different mechanisms with three different fixes, and
/// a total cannot say which one to attack (§15.2 #11a).
#[derive(Clone, Copy, Default)]
pub struct SeamStep {
    pub detokenize_s: f64,
    pub alloc_s: f64,
    pub send_s: f64,
}

// No `total_s`: summing the three would hide that `detokenize_s` is paid by
// BOTH arms and `alloc_s` is structurally zero (see `emit`), so a total is a
// number no reader could act on — §15.2 #11a's shape, and §15.2 #11's
// consume-it-or-delete-it applied to a convenience method nothing consumed.

/// The producer half, living on the decode thread.
///
/// `None` on the null arm: the arm is the *absence* of the seam, not a seam
/// that does nothing, so nothing is allocated, nothing is sent, and no runtime
/// exists. That distinction is the whole comparison.
pub struct Seam {
    /// `Option` so `finish` can drop the sender — closing the channel — without
    /// moving out of a type that implements `Drop`. The consumer's `recv` only
    /// returns `None` once every sender is gone, so this is the close.
    tx: Option<tokio::sync::mpsc::Sender<Chunk>>,
    /// Held so the runtime outlives the sender. Dropping it would abort the
    /// consumer task mid-generation and turn every later `blocking_send` into
    /// an instant error return — which would read as a very fast seam.
    _rt: tokio::runtime::Runtime,
    consumer: Option<std::thread::JoinHandle<ConsumerStats>>,
}

/// What the consumer side saw. Reported so a reader can tell a seam that was
/// exercised from one whose sends all failed (§2.4, after #69's vacuous run):
/// a `sent` count that does not match the token count means the channel closed
/// early and every figure after that point is a measurement of an error path.
#[derive(Clone, Copy, Default)]
pub struct ConsumerStats {
    pub received: usize,
    pub bytes: usize,
    pub boundaries: usize,
}

impl Seam {
    /// Capacity 32, matching `lfm2.rs`'s `channel::<…>(32)`.
    ///
    /// The capacity is load-bearing rather than incidental: it is what decides
    /// whether `blocking_send` ever blocks, and #412 asks whether a slow
    /// consumer can stall generation. A different capacity would answer a
    /// different question.
    pub fn new(capacity: usize) -> anyhow::Result<Self> {
        // A current-thread runtime: the consumer is one task and the producer
        // never awaits, so a multi-threaded pool would add worker threads whose
        // scheduling is not part of the seam. `spawn_blocking`'s own thread is
        // *this* thread — the decode loop — which is exactly ambrogio's shape.
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .context_msg("building the seam runtime")?;
        let (tx, mut rx) = tokio::sync::mpsc::channel::<Chunk>(capacity);

        // The consumer runs on its own OS thread driving its own runtime, so
        // the decode thread is never the one making progress on the receive
        // side. In ambrogio the receiver is a tokio worker for the same reason.
        let consumer = std::thread::spawn(move || {
            let crt = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("consumer runtime");
            crt.block_on(async move {
                let mut stats = ConsumerStats::default();
                while let Some(chunk) = rx.recv().await {
                    match chunk {
                        Chunk::Text { text } => {
                            stats.received += 1;
                            stats.bytes += text.len();
                            stats.boundaries += splitter_like(&text);
                        }
                    }
                }
                stats
            })
        });

        Ok(Self {
            tx: Some(tx),
            _rt: rt,
            consumer: Some(consumer),
        })
    }

    /// Emit one fragment: allocate the `String` and send it, timed as **one**
    /// window.
    ///
    /// # Why one window and not two, which is a finding rather than a shortcut
    ///
    /// §6.4a's rule is that a measurement of a small term has to be *counts ×
    /// prices, not timers*, because *"an `Instant::now()` pair costs 43 ns
    /// against a probe body of 12–51 ns, so timing in place would have reported
    /// mostly the clock"*. That section applies it to candle's fence probe and
    /// the ratio there is ~1:1. **Here it is worse**: measured on this machine
    /// an `Instant` pair is **51.0 ns** and `"hello".to_string()` is **15.5 ns**,
    /// so a timer around the allocation alone would be **3.3× the thing it
    /// measures** — and would report the clock while reading `0.0000 ms`, which
    /// is the shape that looks like a free operation.
    ///
    /// So the allocation is **not timed separately**. One pair spans both terms,
    /// and the allocation is priced by count × price in the artifact instead.
    /// The send is the term large enough to carry a timer: it is a lock, a
    /// memcpy and possibly a park, and it measures in microseconds.
    ///
    /// **`alloc_s` is therefore reported as 0.0 and the total is in `send_s`.**
    /// Left as a field rather than deleted so the RESULT line's shape does not
    /// change between the arm that has an allocation and the arm that does not,
    /// and named in the report as *not separately resolvable* rather than as
    /// zero — a zero and an unmeasurable quantity are different claims, and only
    /// the second one is true (§15.1a's INEXPRESSIVE class: no input to this
    /// instrument produces the other answer, so the repair is a different
    /// instrument, which for a 15 ns term is arithmetic).
    pub fn emit(&self, frag: &str) -> SeamStep {
        let s = Instant::now();
        let text = frag.to_string();
        // `let _ =` exactly as `lfm2.rs` has it: a closed channel is not an
        // error the decode loop acts on. The `if let` is this harness's own —
        // `finish` takes the sender, and calling `emit` after it is a bug rather
        // than a case to handle.
        if let Some(tx) = self.tx.as_ref() {
            let _ = tx.blocking_send(Chunk::Text { text });
        }
        let send_s = s.elapsed().as_secs_f64();

        SeamStep {
            detokenize_s: 0.0,
            alloc_s: 0.0,
            send_s,
        }
    }

    /// Close the channel and collect what the consumer saw.
    ///
    /// Dropping the sender is what ends the consumer's `recv` loop; joining
    /// before reading the stats is what makes the count complete rather than a
    /// sample of a task still running.
    pub fn finish(mut self) -> ConsumerStats {
        self.tx = None;
        match self.consumer.take() {
            Some(h) => h.join().unwrap_or_default(),
            None => ConsumerStats::default(),
        }
    }
}

impl Drop for Seam {
    fn drop(&mut self) {
        // The sender first, then the join. Reversed, this deadlocks: the
        // consumer's `recv` returns `None` only when every sender is gone, so
        // joining while still holding one waits forever. This path is reached
        // when `finish` was not called — an early `?` out of the decode loop —
        // and a harness that hangs on an error is worse than one that reports it.
        self.tx = None;
        if let Some(h) = self.consumer.take() {
            let _ = h.join();
        }
    }
}

/// A sentence-boundary scan of the same order as ambrogio's `SentenceSplitter`.
///
/// Named `_like` deliberately: this is **not** that splitter, it is a scan over
/// the same bytes doing the same class of work, so the consumer side is not
/// free. Its cost lands on the consumer thread and therefore outside every
/// figure the decode thread reports — which is the point of measuring the two
/// sides separately.
fn splitter_like(s: &str) -> usize {
    s.bytes().filter(|b| matches!(b, b'.' | b'!' | b'?')).count()
}

/// `anyhow::Context` without importing the trait into this module's namespace,
/// which would collide with the harness's own `Context` import at the call site.
trait ContextMsg<T> {
    fn context_msg(self, msg: &'static str) -> anyhow::Result<T>;
}

impl<T, E: std::fmt::Display> ContextMsg<T> for std::result::Result<T, E> {
    fn context_msg(self, msg: &'static str) -> anyhow::Result<T> {
        self.map_err(|e| anyhow::anyhow!("{msg}: {e}"))
    }
}

/// Involuntary and voluntary context switches for **this process**, from
/// `getrusage(RUSAGE_SELF)`.
///
/// # Why this and not `proc_pid_rusage`
///
/// `lloom-sample` reads the child through `proc_pid_rusage(RUSAGE_INFO_V6)`,
/// which is what lets it record a workload it did not spawn. That struct
/// **does not carry a context-switch count at all**. The externally-readable
/// one is `proc_pidinfo(PROC_PIDTASKINFO)`'s `pti_csw`, and measured on this
/// machine it is exactly `ru_nvcsw + ru_nivcsw` — the **sum**, with no split.
///
/// The split is what discriminates: an involuntary switch is the scheduler
/// taking the core away, a voluntary one is the thread giving it up at a
/// syscall. A seam that blocks on a channel produces the second; preemption
/// produces the first. So the counter has to be read **in-process**, which is
/// why it is here rather than in the sampler.
#[cfg(unix)]
pub fn csw() -> (u64, u64) {
    // SAFETY: `getrusage` writes a `struct rusage` through the pointer and
    // reads nothing else; the struct is fully initialized by the call on
    // success, and on failure we return the zeroed value rather than reading
    // uninitialized fields. `RUSAGE_SELF` needs no privilege.
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) == 0 {
            (ru.ru_nvcsw as u64, ru.ru_nivcsw as u64)
        } else {
            (0, 0)
        }
    }
}

#[cfg(not(unix))]
pub fn csw() -> (u64, u64) {
    (0, 0)
}

/// Price the instrument against the terms it is pointed at (§6.4a's rule,
/// §2.4's *"check whether the measurement tool is the cost"*).
///
/// Returns `(instant_pair_s, to_string_s)` — the cost of one `Instant::now()`
/// plus one `elapsed()`, and the cost of one `to_string` of a fragment of the
/// size this seam actually emits. **Run outside every measured window**, after
/// the decode loop, so it costs the measurement nothing.
///
/// This is what makes *"the allocation is below the instrument's resolution"* a
/// measurement rather than an excuse: if the pair is smaller than the term, the
/// timer would have been the right instrument and the arithmetic is unnecessary.
/// Measured here it is not, and the report says so with both numbers.
pub fn self_cost(sample_fragment: &str) -> (f64, f64) {
    let n = 200_000;

    let t = Instant::now();
    let mut sink = 0f64;
    for _ in 0..n {
        let a = Instant::now();
        // The `elapsed` is the second half of the pair and must be inside the
        // loop, since that is the shape `emit` pays.
        sink += a.elapsed().as_secs_f64();
    }
    let pair = t.elapsed().as_secs_f64() / n as f64;
    // Consumed so the loop is not optimized away. `black_box` is unstable on
    // this toolchain; a value the compiler cannot prove unused is enough.
    if sink < 0.0 {
        eprintln!("unreachable: negative elapsed sum");
    }

    // Kept rather than dropped per iteration: a `String` freed immediately is a
    // malloc/free pair the allocator can serve from a hot bin, which is not what
    // the seam does — it hands the `String` to a channel and the consumer frees
    // it on another thread. Holding them prices the allocation alone.
    let t = Instant::now();
    let mut keep: Vec<String> = Vec::with_capacity(n);
    for _ in 0..n {
        keep.push(sample_fragment.to_string());
    }
    let alloc = t.elapsed().as_secs_f64() / n as f64;
    drop(keep);

    (pair, alloc)
}

/// Time `Tokenizer::decode` over the pending run — the seam's first term, and
/// the one that is **not** optional: ambrogio's decode loop calls it whether or
/// not anyone is streaming, because it is how the text is built at all.
///
/// It is timed on **both** arms for that reason, and reported separately from
/// the two that only the streaming arm pays. Folding it into the seam total
/// would attribute a cost the null arm also pays to the seam, which is the
/// shape §15.2 #11a warns about.
pub fn time_detokenize<F, T>(f: F) -> (T, f64)
where
    F: FnOnce() -> T,
{
    let t = Instant::now();
    let out = f();
    (out, t.elapsed().as_secs_f64())
}
