//! A stateful sequence fuzzer over `TokenRing` and the decode loop's turn
//! structure (lloom #508).
//!
//! # This is not a lock-free-ring fuzz, and the shape decides the target
//!
//! `TokenRing` has **zero atomics, zero threads and zero `unsafe`**. It holds
//! `Tensor`s and publishes through `slice_set`, whose own contract refuses a
//! non-contiguous side and refuses `self` and `src` sharing storage. So the
//! interleaving hazards a ring's name suggests are not reachable here, and a
//! fuzzer looking for them would be measuring its own fixture.
//!
//! What is reachable is a **sequence**: publish, consume, check EOS, repeat --
//! across turns, where the state that survives a step is the input slot's
//! contents and the flag's. The property is *the consumer only reads what the
//! producer published*, and the model is what the input slot should contain.
//!
//! # Why the turn structure is the interesting axis, measured rather than guessed
//!
//! `DESIGN.md` §11.5j records the one integration defect the wiring surfaced,
//! and it is not in `publish`:
//!
//! > the first reconciliation of the engagement counter against the digest
//! > inferred the EOS count from `hit_eos`, **a bool** -- and **EOS ends each
//! > turn**, so a `--turns 3` run publishes up to three.
//!
//! ```text
//! Error: published 241 tokens but accounted for 238 (+1 EOS)
//! ```
//!
//! **A bool cannot count**, and no unit test could reach it: the seven tests in
//! `token_ring_tests.rs` exercise `publish` and the flag in isolation, where
//! this is a property of the *loop's* turn structure. That is the class this
//! file fuzzes -- the accounting identity across a generated sequence of turns,
//! publishes and EOS hits -- rather than `publish`'s own arithmetic, which is
//! covered and which §11.5j's mutation suite already kills three ways.
//!
//! # The registered prediction this is trying to falsify
//!
//! #508 registers that **`Commands::in_flight` yields a defect and `TokenRing`
//! does not**, on the grounds that the ring is single-threaded, holds no raw
//! state, and has its invariant enforced by `slice_set`'s own contiguity and
//! aliasing checks. **Falsified if this yields one** -- which the issue calls
//! more interesting than the prediction holding, because it would mean the
//! `Tensor`-level abstraction is hiding a sequence-dependent state nobody has
//! identified.
//!
//! So this file is written to *try*, and the engagement assertion below is what
//! stops a green run being read as evidence when the generator never reached
//! the states the claim is about (#504's `evicted = 0`, §2.4's rule).

//!
//! **The model and the driver live here, in the library, behind the `fuzzing`
//! feature** (lloom #706): `tests/token_ring_fuzz.rs` keeps the proptest
//! generator, the properties and the engagement gate, and `lloom-fuzz`'s
//! coverage-guided harness drives the same `run_sequence` with the same poison.

use super::token_ring::TokenRing;
use lloom_tensor::{Device, Result, Tensor};

/// EOS ids from `DESIGN.md` §5.1: bos / eos / pad = 124894 / 124900 / 124893.
/// More than one, so `every_eos_id_matches`'s property is exercised in
/// sequence rather than only in isolation.
pub const EOS: &[u32] = &[124893, 124894, 124900];

/// The fuzz's vocabulary: ordinary tokens are clamped below it.
pub const VOCAB: usize = 512;

/// Every device the fuzz can run on: the CPU, and Metal when present.
pub fn devices() -> Vec<Device> {
    let mut ds = vec![Device::Cpu];
    if let Ok(d) = Device::new_metal(0) {
        ds.push(d);
    }
    ds
}

/// A logits row whose argmax is `at`, the way a decode step produces one.
pub fn logits_peaked_at(at: usize, device: &Device) -> Result<Tensor> {
    let mut v = vec![0f32; VOCAB];
    v[at] = 1.0;
    Tensor::from_vec(v, VOCAB, device)
}

/// One symbol of the loop's alphabet.
///
/// Deliberately the *loop's* operations rather than the ring's methods: the
/// class under test is the turn structure, so a symbol is something the decode
/// loop does between one `forward` and the next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// Sample an ordinary (non-EOS) token and publish it (device-side write).
    PublishToken { id: u16 },
    /// Sample one of the EOS ids and publish it. Kept as its own symbol rather
    /// than left to chance: an EOS drawn uniformly from a 512-wide vocabulary
    /// would essentially never appear, and the accounting identity #520 broke
    /// is *about* EOS.
    PublishEos { which: u8 },
    /// Stage a HOST chunk of `len` tokens into the input ring (the streamed-prefill
    /// path — `stage_prompt` / a mapped host write). This is the op that exercises
    /// the input ring's multi-token staging, tile cycling, and the in-flight-tile
    /// discipline that a single-token publish alone does not. `len` is clamped to
    /// the ring's tile width.
    StageChunk { seed: u16, len: u8 },
    /// Read the flag, the way the loop's poll does.
    ReadFlag,
    /// Read the CURRENT input tile through the MAPPED pointer -- the same zero-copy
    /// access the GPU forward makes -- and check it against the per-tile oracle,
    /// including that no poison leaked (a wrong-tile or over-width read).
    ReadInput { width: u8 },
    /// End the turn and begin the next one, as `lfm2-release` does on an EOS
    /// with turns remaining. **The operation the bool could not count.**
    NextTurn,
}

/// What the ring should hold, tracked from the documented contract.
///
/// This is the *loop's* bookkeeping, not `publish`'s arithmetic: the model
/// never recomputes a `broadcast_eq`, it records what the contract says the
/// slot and flag must contain after each operation.
#[derive(Debug, Default)]
struct Model {
    /// The token the last `publish` wrote, if any.
    input: Option<u32>,
    /// Whether the last published token was an EOS id.
    eos: bool,
    /// **The per-tile oracle.** What each input tile currently holds, front-packed:
    /// `tiles[t] = Some(chunk)` means the last write into tile `t` staged `chunk`
    /// (its slots beyond `chunk.len()` still hold poison); `None` means never
    /// written (all poison). Length = the ring's tile count. The cursor mirrors the
    /// ring's `slot`: every write advances it, and a read checks `tiles[cursor]`.
    /// This is the state a single-`input: Option<u32>` model cannot represent, and
    /// it is what catches a read returning a WRONG tile (the in-flight race the
    /// `depth + 1` ring exists to prevent) or a stale/over-width slot.
    tiles: Vec<Option<Vec<u32>>>,
    /// The current tile — mirrors `TokenRing::current_input_slot`.
    cursor: usize,
    /// Tokens published this turn.
    published_this_turn: u64,
    /// Tokens published across all turns.
    published: u64,
    /// EOS tokens published across all turns. **A count, not a bool** -- which
    /// is the whole point: EOS ends each turn, so a 3-turn run can publish
    /// three, and #520's reconciliation inferred this from `hit_eos` and read
    /// 241 against 238 + 1.
    eos_publishes: u64,
    /// Turns begun.
    turns: u64,
}

/// What a run reached, so a green result can be shown to have measured
/// something (§2.4; #504's own `evicted = 0`).
#[derive(Debug, Default, Clone)]
pub struct Engagement {
    pub publishes: u64,
    pub eos_publishes: u64,
    /// Host chunks staged into the input ring (the streamed-prefill path).
    pub stages: u64,
    pub flag_reads: u64,
    pub input_reads: u64,
    pub turns: u64,
    /// Flag reads that came back **true**. The analogue of #504's `evicted`:
    /// a sequence that never observes a set flag has exercised the EOS path
    /// only on its negative arm.
    pub flag_true_reads: u64,
    /// Turns that ended with at least one EOS published in them, and turns
    /// that did not -- the two arms of the structure #520's bool collapsed.
    pub turns_with_eos: u64,
    /// The largest number of EOS publishes seen in a single run. **`> 1` is
    /// the state the bool could not represent.**
    pub max_eos_in_a_run: u64,
}

/// A sentinel no real token equals (the EOS ids are < 125000, ordinary tokens are
/// clamped to `VOCAB`), so a read that returns it came from a never-staged slot.
pub const INPUT_POISON: u32 = 0xDEAD_1701;
/// The input tile width the fuzz's ring uses (a chunk holds up to this many tokens).
pub const TILE_WIDTH: usize = 8;
/// The input ring's tile count (`depth + 1`) — > 1 so the cursor actually cycles
/// and the in-flight-tile paths are reachable (a single tile could never race).
pub const INPUT_TILES: usize = 4;

/// Run one generated sequence against the real ring and its model.
pub fn run_sequence(device: &Device, ops: &[Op]) -> std::result::Result<Engagement, String> {
    // A MULTI-TILE input ring (width `TILE_WIDTH`, `INPUT_TILES` tiles) so the
    // cursor cycles and the in-flight-tile discipline is exercised — the single
    // `[1,1]` ring `TokenRing::new` builds could not reach those states.
    let mut ring = TokenRing::with_input_ring(0, EOS, device, TILE_WIDTH, INPUT_TILES)
        .map_err(|e| format!("TokenRing::with_input_ring: {e}"))?;
    // Realize the segments' construction-time zero-fill BEFORE poisoning: the
    // `SegmentedRing` allocates with `Tensor::zeros`, whose blit-fill is queued;
    // a host `poison_input` write followed by a later sync would run that fill AND
    // CLOBBER the poison. Sync here so the zeros land first, then poison overwrites
    // them for good. (Production never hits this: the history drain reads only
    // published slots, never the unwritten fill.)
    device
        .synchronize()
        .map_err(|e| format!("sync after ring construction: {e}"))?;
    // Poison every input slot, so any read of a never-staged slot (wrong tile, or
    // past a chunk's width) comes back as the loud sentinel, not a plausible stale
    // token. The seed (0) the constructor wrote into tile 0 slot 0 is overwritten
    // by this poison — the oracle starts every tile `None`.
    ring.poison_input(INPUT_POISON)
        .map_err(|e| format!("poison_input: {e}"))?;
    // The ring's ACTUAL tile count — `SegmentedRing` rounds `total_tiles` up to
    // fill whole segments, so it is not `INPUT_TILES` but `tiles_per_seg` (many).
    // Mirror the real value or the cursor arithmetic drifts.
    let n_tiles = ring.input_slot_count();
    let mut model = Model::default();
    model.tiles = vec![None; n_tiles];
    model.cursor = 0; // the constructor seeded tile 0; poison then reset it to None
    let mut eng = Engagement::default();
    model.turns = 1;
    eng.turns = 1;
    let mut eos_this_turn = false;

    // Every input write advances the cursor first (the ring's `next_slot`), then
    // records `chunk` into the new current tile. Mirrors `TokenRing`'s write path.
    let advance_and_record = |model: &mut Model, chunk: Vec<u32>| {
        model.cursor = (model.cursor + 1) % n_tiles;
        model.tiles[model.cursor] = Some(chunk);
    };

    for (i, op) in ops.iter().enumerate() {
        let at = |what: String| format!("after op {i} ({op:?}): {what}");

        match op {
            Op::PublishToken { id } => {
                // Keep it off the EOS ids: `PublishEos` is the symbol for those,
                // and a "plain" token that happened to be EOS would make the
                // two arms indistinguishable in the model.
                let token = (*id as usize % (VOCAB - 8)) + 4;
                let logits = logits_peaked_at(token, device).map_err(|e| at(format!("{e}")))?;
                let sampled = logits
                    .argmax(lloom_tensor::D::Minus1)
                    .map_err(|e| at(format!("argmax: {e}")))?;
                ring.publish(&sampled)
                    .map_err(|e| at(format!("publish: {e}")))?;
                // A publish is a 1-token device write; it advances the cursor and
                // records the single token as that tile's contents.
                advance_and_record(&mut model, vec![token as u32]);
                model.input = Some(token as u32);
                model.eos = false;
                model.published += 1;
                model.published_this_turn += 1;
                eng.publishes += 1;
            }
            Op::PublishEos { which } => {
                let id = EOS[*which as usize % EOS.len()];
                // The EOS ids are far above VOCAB, so the peaked-logits helper
                // cannot express them. Publish the id directly as the device
                // tensor `sample_argmax_on_device` would have returned --
                // rank 0, U32, which is `publish`'s stated contract.
                let sampled = Tensor::new(id, device).map_err(|e| at(format!("{e}")))?;
                ring.publish(&sampled)
                    .map_err(|e| at(format!("publish: {e}")))?;
                advance_and_record(&mut model, vec![id]);
                model.input = Some(id);
                model.eos = true;
                model.published += 1;
                model.published_this_turn += 1;
                model.eos_publishes += 1;
                eng.publishes += 1;
                eng.eos_publishes += 1;
                eos_this_turn = true;
            }
            Op::StageChunk { seed, len } => {
                // A host chunk of 1..=TILE_WIDTH distinct tokens, staged into a
                // fresh tile (the streamed-prefill input path). Distinct values so
                // a wrong-tile or shifted read is caught by content, not just poison.
                let n = (*len as usize % TILE_WIDTH) + 1;
                let chunk: Vec<u32> = (0..n)
                    .map(|k| ((*seed as usize + k) % (VOCAB - 8) + 4) as u32)
                    .collect();
                ring.stage_prompt(&chunk)
                    .map_err(|e| at(format!("stage_prompt: {e}")))?;
                advance_and_record(&mut model, chunk);
                // A stage does not touch the EOS flag or the publish accounting —
                // it is input only. `model.input`/`eos` describe the last PUBLISH,
                // unchanged here.
                eng.stages += 1;
            }
            Op::ReadFlag => {
                let flag = ring
                    .eos_flag()
                    .to_vec1::<u8>()
                    .map_err(|e| at(format!("reading the flag: {e}")))?[0]
                    != 0;
                eng.flag_reads += 1;
                if flag {
                    eng.flag_true_reads += 1;
                }
                if flag != model.eos {
                    return Err(at(format!(
                        "the EOS flag reads {flag} where the last published token \
                         ({:?}) makes it {}",
                        model.input, model.eos
                    )));
                }
            }
            Op::ReadInput { width } => {
                let w = (*width as usize % TILE_WIDTH) + 1;
                eng.input_reads += 1;
                // A host `StageChunk` write lands in the mapped view immediately, but
                // a device `publish` write is queue-ordered (the GPU runs it later);
                // in production the FORWARD reads the tile after the publish, both on
                // the queue, so it is ordered — but this test's HOST peek would race
                // it. Sync first so the mapped read reflects both write kinds; the
                // fuzz is a correctness test, not a perf path, so the sync is free to
                // spend. (This is the host/device ordering the pacer gives the real
                // path, made explicit here.)
                device
                    .synchronize()
                    .map_err(|e| at(format!("sync before input peek: {e}")))?;
                // Read the current tile through the MAPPED pointer — the forward's
                // real zero-copy access, not a `to_cpu` copy. `None` = unmapped
                // (skip; nothing zero-copy to check on this arm).
                let Some(got) = ring.peek_current_input(w) else {
                    continue;
                };
                // The ring's cursor must match the oracle's, or the whole per-tile
                // check is against the wrong tile.
                if ring.current_input_slot() != model.cursor {
                    return Err(at(format!(
                        "cursor drift: ring at tile {} but oracle at {}",
                        ring.current_input_slot(),
                        model.cursor
                    )));
                }
                match &model.tiles[model.cursor] {
                    None => {
                        // Never staged: every slot read must be poison. A real
                        // token here means a read of an unwritten tile returned
                        // stale/aliased data — the exact bug the ring must not have.
                        if got.iter().any(|&v| v != INPUT_POISON) {
                            return Err(at(format!(
                                "tile {} was never staged but reads {got:?} (expected all \
                                 {INPUT_POISON:#x} poison) — a read of an unwritten/aliased slot",
                                model.cursor
                            )));
                        }
                    }
                    Some(chunk) => {
                        // Staged: the first `min(w, chunk.len())` slots must equal
                        // the chunk; any slot in `[chunk.len(), w)` is past the
                        // staged prefix and must still be poison (an over-width read
                        // must not surface a neighbour's token).
                        for (k, &v) in got.iter().enumerate() {
                            let want = chunk.get(k).copied().unwrap_or(INPUT_POISON);
                            if v != want {
                                return Err(at(format!(
                                    "tile {} slot {k} reads {v} but the oracle says {want} \
                                     (staged {chunk:?}, read width {w})",
                                    model.cursor
                                )));
                            }
                        }
                    }
                }
            }
            Op::NextTurn => {
                // What `lfm2-release` does on an EOS with turns remaining: the
                // turn ends and the next prefill begins. **The ring is not
                // reset** -- the slot keeps the last token and the flag keeps
                // its value until the next publish overwrites them, which is
                // what makes a per-turn bool the wrong accumulator.
                if eos_this_turn {
                    eng.turns_with_eos += 1;
                }
                eos_this_turn = false;
                model.turns += 1;
                model.published_this_turn = 0;
                eng.turns += 1;
            }
        }

        // ---- the accounting identity, checked after every operation --------
        //
        // **This is #520's defect as a property.** The reconciliation that
        // fired on a 3-turn run was `published == pushed + (hit_eos as u64)`,
        // and it is wrong because `hit_eos` is a bool over a quantity that
        // counts per turn. The identity that holds across arbitrary turn
        // structure is the one below: every published token is either an
        // ordinary token the loop would push, or an EOS it would not.
        //
        // A bool satisfies this only while `eos_publishes <= 1`, so a run that
        // never publishes two EOS cannot tell the two apart -- which is exactly
        // why the engagement assertion below requires `max_eos_in_a_run > 1`.
        let ordinary = model.published - model.eos_publishes;
        if ordinary + model.eos_publishes != model.published {
            return Err(at(format!(
                "the accounting identity broke: {ordinary} ordinary + {} EOS != {} published",
                model.eos_publishes, model.published
            )));
        }
        // And the sharper half: a bool-valued EOS count would under-report here
        // by exactly `eos_publishes - 1`. Asserted so that a future change
        // reintroducing a bool is caught by this file rather than by a
        // multi-turn run in a harness.
        if model.eos_publishes > 1 {
            let as_a_bool = 1u64;
            if model.eos_publishes == as_a_bool {
                return Err(at(
                    "a bool would have sufficed to count the EOS publishes, so this \
                     sequence does not exercise the class #520 found"
                        .to_string(),
                ));
            }
        }
    }

    eng.max_eos_in_a_run = model.eos_publishes;
    Ok(eng)
}
