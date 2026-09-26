//! The streamed tier's eviction never drains the device (#749, `DESIGN.md`
//! §20.4a).
//!
//! A slot is reused when the pool clock's completed count is past the epoch
//! of the slot's last GPU reader: one atomic load (`GpuClock::completed_count`,
//! an `Acquire` — the host's write into the slot must not be ordered before
//! it), no wait. A slot whose reader has not run takes the staging ring and a
//! copy ordered behind that reader on the queue, so the host still waits for
//! nothing. The one poll in the path, the `pread` arm's direct read, is on a
//! reader thread and only ever handed a slot the clock already shows settled.
//!
//! **A source-level census, because the property is an absence** — a
//! runtime test cannot observe a wait that is not there, and every result is
//! byte-identical either way (`qwen4exp_streamed_identity.rs` shows the
//! evictions happen and change no bit). Comment lines are stripped and the
//! needles assembled, so neither this file nor a doc comment matches itself
//! (the form of `encoder.rs`'s `every_recorded_set_on_the_bind_path_has_a_reader`).

fn code_only(src: &str) -> String {
    src.lines()
        .filter(|l| {
            let t = l.trim_start();
            !(t.starts_with("//") || t.starts_with("/*") || t.starts_with('*'))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn the_eviction_path_never_waits_for_the_device() {
    let stream = code_only(include_str!("../src/nn/expert_stream.rs"));
    let prefetch = code_only(include_str!("../src/nn/expert_prefetch.rs"));

    // Every host-side form of "wait for the GPU": the command buffer's two
    // waits, in the crate's spelling and in Metal's.
    let waits = [
        ["wait_until_", "completed"].concat(),
        ["wait_until_", "scheduled"].concat(),
        ["waitUntil", "Completed"].concat(),
        ["waitUntil", "Scheduled"].concat(),
    ];
    for w in &waits {
        assert!(
            !stream.contains(w.as_str()),
            "the streamed tier waits for the device (`{w}` in expert_stream.rs): \
             eviction must gate on the pool clock's completed count, never drain \
             (DESIGN.md §20.4a, #749)"
        );
        assert!(
            !prefetch.contains(w.as_str()),
            "the prefetcher waits for the device (`{w}` in expert_prefetch.rs)"
        );
    }

    // The control: the gate the eviction uses instead is still there, so the
    // census is not passing because the mechanism vanished. The victim is
    // chosen against the completed count read once per `ensure` ...
    let gate = ["victim_settled(tick, ", "settled_before)"].concat();
    assert!(
        stream.contains(&gate),
        "the settled-victim choice is gone — this census is now vacuous"
    );
    let read = ["let settled_before = inner.", "completed_epochs();"].concat();
    assert!(
        stream.contains(&read),
        "the completed-count read before the victim choice is gone"
    );
    // ... and that count is one atomic load.
    let pool = code_only(include_str!(
        "../../lloom-tensor/src/kernels/metal/buffer_pool.rs"
    ));
    let load = ["self.completed.", "load(Ordering::Acquire)"].concat();
    assert!(
        pool.contains(&load),
        "GpuClock::completed_count is no longer a single atomic load"
    );
}
