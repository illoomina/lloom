//! `Session::decode_batched` over a device-only fake model (issue #694,
//! `design/docs/batched-decode-design.md` §10).
//!
//! The target is the `Chain` of `decode_driver_tests.rs` — the next token is
//! a fixed function of the last one, computed on the device — extended with
//! `BatchedCausalLM::forward_rows`, so every row's expected stream is derived
//! from its own start token rather than recorded, and rows starting from
//! different tokens reach EOS at different steps. What is asserted is the
//! driver's contract, not a kernel: each row's ids are its solo chain, a row
//! leaves the batch at its own EOS or cap without disturbing the others, the
//! two lookahead arms agree on the ids and differ only in the overrun the
//! design says they do, the position and token bookkeeping reconcile, and the
//! refusals refuse. On the CPU and Metal backends.

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::nn::{
    BatchedCausalLM, CausalLM, DecodeOptions, IdentityProposer, Model, RaggedRow, Row, Session,
};
use lloom_tensor::{DType, Device, Result, Tensor};

const VOCAB: usize = 64;
const EOS: u32 = 42;

fn next_of(tok: u32) -> u32 {
    (tok * 7 + 3) % VOCAB as u32
}

/// The chain from `start`, stopping before EOS, capped at `max`.
fn chain(start: u32, max: usize) -> (Vec<u32>, bool) {
    let mut ids = Vec::new();
    let mut tok = start;
    while ids.len() < max {
        tok = next_of(tok);
        if tok == EOS {
            return (ids, true);
        }
        ids.push(tok);
    }
    (ids, false)
}

/// A decoder whose logits peak at `next_of(last token)`, device-side, for
/// any number of rows at once.
struct Chain {
    table: Tensor,
    eye: Tensor,
}

impl Chain {
    fn new(device: &Device) -> Result<Self> {
        let table: Vec<f32> = (0..VOCAB as u32).map(|t| next_of(t) as f32).collect();
        Ok(Self {
            table: Tensor::from_vec(table, VOCAB, device)?,
            eye: Tensor::eye(VOCAB, DType::F32, device)?,
        })
    }

    /// `[b, seq]` tokens to `[b, vocab]` one-hot logits of each row's successor.
    fn logits_of(&self, tokens: &Tensor) -> Result<Tensor> {
        let (_b, seq) = tokens.dims2()?;
        let last = tokens.narrow(1, seq - 1, 1)?.flatten_all()?;
        let next = self.table.index_select(&last, 0)?.to_dtype(DType::U32)?;
        self.eye.index_select(&next, 0)
    }
}

impl Model for Chain {
    type Config = ();
    fn from_weights(_vb: lloom::nn::VarBuilder, _cfg: &()) -> Result<Self> {
        lloom_tensor::bail!("Chain has no weights")
    }
}

impl CausalLM for Chain {
    /// Positions seen, so a test can check `pos` against the cache.
    type Cache = usize;

    fn new_cache(&self, _cfg: &(), _dev: &Device) -> Result<usize> {
        Ok(0)
    }

    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        Ok(tokens.clone())
    }

    fn forward_embeds(&self, embeds: &Tensor, pos: usize, cache: &mut usize) -> Result<Tensor> {
        let (_b, seq) = embeds.dims2()?;
        assert_eq!(
            *cache, pos,
            "forward at pos {pos} against a cache of {cache}"
        );
        *cache += seq;
        self.logits_of(embeds)
    }

    fn vocab_size(&self) -> usize {
        VOCAB
    }
}

impl BatchedCausalLM for Chain {
    /// The ragged pass: each row's logits are its last token's successor, and
    /// its position counter advances by its run.
    fn forward_ragged(&self, tokens: &Tensor, rows: &mut [RaggedRow<'_, usize>]) -> Result<Tensor> {
        let (b, total) = tokens.dims2()?;
        assert_eq!(b, 1, "a ragged input is [1, T]");
        assert_eq!(total, rows.iter().map(|r| r.len).sum::<usize>());
        let mut lasts = Vec::with_capacity(rows.len());
        let mut off = 0;
        for (r, row) in rows.iter_mut().enumerate() {
            assert_eq!(
                *row.cache, row.pos,
                "row {r}: forward at pos {} against a cache of {}",
                row.pos, row.cache
            );
            *row.cache += row.len;
            off += row.len;
            lasts.push((off - 1) as u32);
        }
        let last = tokens
            .squeeze(0)?
            .index_select(&Tensor::new(lasts.as_slice(), tokens.device())?, 0)?;
        let next = self.table.index_select(&last, 0)?.to_dtype(DType::U32)?;
        self.eye.index_select(&next, 0)
    }

    fn forward_rows(&self, tokens: &Tensor, rows: &mut [Row<'_, usize>]) -> Result<Tensor> {
        let (b, seq) = tokens.dims2()?;
        assert_eq!(seq, 1, "a batched step is one token per row");
        assert_eq!(b, rows.len());
        for (r, row) in rows.iter_mut().enumerate() {
            assert_eq!(
                *row.cache, row.pos,
                "row {r}: forward at pos {} against a cache of {}",
                row.pos, row.cache
            );
            *row.cache += 1;
        }
        self.logits_of(tokens)
    }
}

fn devices() -> Vec<Device> {
    let mut v = vec![Device::Cpu];
    #[cfg(feature = "metal")]
    if let Ok(m) = Device::new_metal(0) {
        v.push(m);
    }
    v
}

fn greedy() -> LogitsProcessor {
    LogitsProcessor::from_sampling(0, Sampling::ArgMax)
}

/// Three prompts of different lengths whose last tokens are 1, 0 and 5: the
/// first reaches EOS at its ninth token, the second never does, the third at
/// its own step — derived, not recorded.
fn prompts() -> Vec<Vec<u32>> {
    vec![vec![3, 1], vec![0], vec![9, 9, 9, 5]]
}

/// What each prompt's solo chain is at `max` tokens.
fn expected(max: usize) -> Vec<(Vec<u32>, bool)> {
    prompts()
        .iter()
        .map(|p| chain(*p.last().unwrap(), max))
        .collect()
}

fn run(
    device: &Device,
    lookahead: usize,
    max_tokens: usize,
) -> Result<(
    Vec<lloom::nn::DecodeOut>,
    Vec<Session<'static, Chain>>,
    Vec<Vec<u32>>,
)> {
    // Leak the model so the sessions can be returned with a `'static` borrow;
    // a test process ends with them.
    let model: &'static Chain = Box::leak(Box::new(Chain::new(device)?));
    let prompts = prompts();
    let mut sessions: Vec<Session<'static, Chain>> = prompts
        .iter()
        .map(|_| Session::new(model, 0usize))
        .collect();
    let refs: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
    let opts = DecodeOptions {
        lookahead,
        ..DecodeOptions::new(max_tokens)
    };
    let mut streamed: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
    let outs = Session::decode_batched(
        &mut sessions,
        &mut greedy(),
        &refs,
        &opts,
        &[EOS],
        device,
        &mut |i, ids| {
            streamed[i].extend_from_slice(ids);
            Ok(())
        },
    )?;
    Ok((outs, sessions, streamed))
}

#[test]
fn every_row_is_its_own_solo_chain_and_leaves_at_its_own_end() -> Result<()> {
    for device in devices() {
        for lookahead in [0, 1] {
            let max = 12;
            let (outs, sessions, streamed) = run(&device, lookahead, max)?;
            let want = expected(max);
            // Non-vacuity: the rows end differently — one on EOS, one on the cap.
            assert!(
                want[0].1 && !want[1].1,
                "the fixture must mix EOS and cap endings"
            );
            assert!(want[0].0.len() < max);
            for (i, (ids, eos)) in want.iter().enumerate() {
                assert_eq!(
                    &outs[i].ids, ids,
                    "row {i} lookahead {lookahead} on {device:?}: ids differ from the solo chain"
                );
                assert_eq!(outs[i].eos, *eos, "row {i}: EOS flag");
                assert_eq!(
                    streamed[i], outs[i].ids,
                    "row {i}: the streamed ids are the ids"
                );
                assert_eq!(outs[i].stats.produced, ids.len());
                // The batch shrank as rows left: the row that ended first saw
                // three live rows on every step; the cap rows saw fewer later.
                assert_eq!(outs[i].stats.batch_rows.len(), outs[i].stats.step_ns.len());
                assert_eq!(
                    outs[i].stats.batch_rows[0], 3,
                    "row {i}: every row is live at step 0"
                );
            }
            let first_end = want.iter().map(|(ids, _)| ids.len()).min().unwrap();
            let longest = want.iter().map(|(ids, _)| ids.len()).max().unwrap();
            let long_row = want
                .iter()
                .position(|(ids, _)| ids.len() == longest)
                .unwrap();
            assert!(
                outs[long_row].stats.batch_rows.iter().any(|&b| b < 3),
                "the longest row must have run in a shrunken batch after step {first_end}"
            );
            // Bookkeeping from the position: the fed ids are the token log's
            // suffix, the overrun is what the design says per arm.
            for (i, s) in sessions.iter().enumerate() {
                let (ids, eos) = &want[i];
                let prompt_len = prompts()[i].len();
                let fed = s.pos() - prompt_len;
                let expected_overrun = match (lookahead, eos) {
                    // Blocking: the EOS is never fed; the cap's last token is
                    // streamed and not fed.
                    (0, true) => 0,
                    (0, false) => 0,
                    // Lookahead: the EOS token was fed before the host saw it.
                    (1, true) => 1,
                    (1, false) => 0,
                    _ => unreachable!(),
                };
                assert_eq!(outs[i].stats.overrun, expected_overrun, "row {i}: overrun");
                assert_eq!(s.overrun(), expected_overrun);
                let in_cache = fed - expected_overrun;
                assert_eq!(&s.tokens()[..prompt_len], prompts()[i].as_slice());
                assert_eq!(
                    &s.tokens()[prompt_len..],
                    &ids[..in_cache],
                    "row {i}: token log"
                );
                // Blocking arm at the cap: the last token is streamed, not fed.
                if lookahead == 0 && !eos {
                    assert_eq!(in_cache, ids.len() - 1);
                }
                if lookahead == 1 && !eos {
                    assert_eq!(in_cache, ids.len());
                    assert!(
                        s.sequence().tokens().len() == s.pos(),
                        "row {i}: a cap row under the lookahead is resumable"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn the_two_lookahead_arms_agree_on_the_ids_and_a_cap_row_resumes() -> Result<()> {
    for device in devices() {
        let (a, _, _) = run(&device, 0, 7)?;
        let (b, mut sessions, _) = run(&device, 1, 7)?;
        for i in 0..a.len() {
            assert_eq!(a[i].ids, b[i].ids, "row {i}: the arms disagree");
            assert_eq!(a[i].eos, b[i].eos);
        }
        // The cap row (row 1 never reaches EOS) resumes from the held logits:
        // an empty prompt continues its chain exactly.
        let cap_row = 1;
        assert!(!b[cap_row].eos);
        let want_next = chain(*b[cap_row].ids.last().unwrap(), 3).0;
        let s = &mut sessions[cap_row];
        // The token log is the whole cache and the last logits are held, so
        // an empty prompt resumes from them (`Session::decode`'s resume).
        assert_eq!(s.tokens().len(), s.pos());
        let out = s.decode(
            &mut IdentityProposer,
            &mut Blocking::new(&[EOS], &device)?,
            &mut greedy(),
            &[],
            &DecodeOptions::new(3),
            &mut |_| Ok(()),
        )?;
        assert_eq!(out.ids, want_next, "the resumed continuation is the chain");
    }
    Ok(())
}

#[test]
fn a_single_row_batch_is_the_serial_loop() -> Result<()> {
    for device in devices() {
        let model = Chain::new(&device)?;
        let prompt = [3u32, 1];
        let mut solo = Session::new(&model, 0usize);
        let solo_out = solo.decode(
            &mut IdentityProposer,
            &mut Blocking::new(&[EOS], &device)?,
            &mut greedy(),
            &prompt,
            &DecodeOptions::new(20),
            &mut |_| Ok(()),
        )?;
        let mut batched = vec![Session::new(&model, 0usize)];
        let outs = Session::decode_batched(
            &mut batched,
            &mut greedy(),
            &[&prompt],
            &DecodeOptions::new(20),
            &[EOS],
            &device,
            &mut |_, _| Ok(()),
        )?;
        assert_eq!(outs[0].ids, solo_out.ids);
        assert_eq!(outs[0].eos, solo_out.eos);
        assert!(outs[0].eos, "the fixture reaches EOS");
    }
    Ok(())
}

#[test]
fn the_refusals_refuse() -> Result<()> {
    let device = Device::Cpu;
    let model = Chain::new(&device)?;
    let prompt = [3u32, 1];

    // A sampler the batch cannot serve on the device.
    let mut sessions = vec![Session::new(&model, 0usize)];
    let err = Session::decode_batched(
        &mut sessions,
        &mut LogitsProcessor::from_sampling(0, Sampling::All { temperature: 0.7 }),
        &[&prompt],
        &DecodeOptions::new(4),
        &[EOS],
        &device,
        &mut |_, _| Ok(()),
    )
    .err()
    .expect("a temperature sampler is refused");
    assert!(err.to_string().contains("argmax"), "{err}");

    // A lookahead past one.
    let err = Session::decode_batched(
        &mut sessions,
        &mut greedy(),
        &[&prompt],
        &DecodeOptions {
            lookahead: 2,
            ..DecodeOptions::new(4)
        },
        &[EOS],
        &device,
        &mut |_, _| Ok(()),
    )
    .err()
    .expect("lookahead 2 is refused");
    assert!(err.to_string().contains("lookahead"), "{err}");

    // An empty prompt with nothing to resume from.
    let err = Session::decode_batched(
        &mut sessions,
        &mut greedy(),
        &[&[]],
        &DecodeOptions::new(4),
        &[EOS],
        &device,
        &mut |_, _| Ok(()),
    )
    .err()
    .expect("an empty prompt with no held logits is refused");
    assert!(err.to_string().contains("resume"), "{err}");

    // An unreconciled overrun (the lookahead arm's EOS row) is refused until
    // the sequence is reset.
    let (_, mut sessions, _) = run(&device, 1, 12)?;
    let eos_row = 0;
    assert_eq!(sessions[eos_row].overrun(), 1);
    let mut one = vec![std::mem::replace(
        &mut sessions[eos_row],
        Session::new(Box::leak(Box::new(Chain::new(&device)?)), 0usize),
    )];
    let err = Session::decode_batched(
        &mut one,
        &mut greedy(),
        &[&prompt],
        &DecodeOptions::new(4),
        &[EOS],
        &device,
        &mut |_, _| Ok(()),
    )
    .err()
    .expect("an overrun is refused");
    assert!(err.to_string().contains("overrun"), "{err}");
    Ok(())
}

/// The CHUNK under the batched driver (lloom #728): rows admitted a chunk per
/// step, rows arriving mid-run, and a tile bounding each prefill forward all
/// produce their solo chains; the forwards a row's prompt took are counted
/// off the run; and a row joins the batch at the step the design names.
#[test]
fn chunked_admission_and_late_arrivals_produce_the_solo_chains() -> Result<()> {
    // Row 0's chain reaches EOS at its ninth token, row 1's never does (the
    // cap ends it), row 2 arrives at step 4 with a four-token prompt.
    let prompts: Vec<Vec<u32>> = vec![vec![9, 9, 9, 9, 9, 3, 1], vec![0], vec![9, 9, 9, 5]];
    let arrivals = [0u64, 0, 4];
    let max = 12;
    let want: Vec<(Vec<u32>, bool)> = prompts
        .iter()
        .map(|p| chain(*p.last().unwrap(), max))
        .collect();
    assert!(
        want[0].1 && !want[1].1,
        "the fixture must mix EOS and cap endings"
    );
    for device in devices() {
        for lookahead in [0usize, 1] {
            for (chunk, tile) in [(0usize, 0usize), (0, 2), (1, 0), (2, 0), (3, 0), (3, 2)] {
                let model: &'static Chain = Box::leak(Box::new(Chain::new(&device)?));
                let mut sessions: Vec<Session<'static, Chain>> = prompts
                    .iter()
                    .map(|_| Session::new(model, 0usize))
                    .collect();
                let refs: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
                let opts = DecodeOptions {
                    lookahead,
                    prefill_chunk: chunk,
                    prefill_tile: tile,
                    ..DecodeOptions::new(max)
                };
                let mut streamed: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
                let outs = Session::decode_batched_admitted(
                    &mut sessions,
                    &mut greedy(),
                    &refs,
                    &arrivals,
                    &opts,
                    &[EOS],
                    &device,
                    &mut |i, ids| {
                        streamed[i].extend_from_slice(ids);
                        Ok(())
                    },
                )?;
                let tag = format!("{device:?} lookahead {lookahead} chunk {chunk} tile {tile}");
                for (i, (ids, eos)) in want.iter().enumerate() {
                    assert_eq!(&outs[i].ids, ids, "{tag}: row {i} ids");
                    assert_eq!(outs[i].eos, *eos, "{tag}: row {i} eos");
                    assert_eq!(streamed[i], outs[i].ids, "{tag}: row {i} streamed");
                    // The forwards the prompt took: whole (tiled) at chunk 0,
                    // else one ragged pass per piece, a piece being at most
                    // the chunk and at most the tile.
                    let len = prompts[i].len();
                    let want_units: usize = if chunk == 0 {
                        if tile == 0 {
                            1
                        } else {
                            len.div_ceil(tile)
                        }
                    } else {
                        let piece = if tile == 0 { chunk } else { chunk.min(tile) };
                        len.div_ceil(piece)
                    };
                    assert_eq!(
                        outs[i].stats.prefill_units, want_units,
                        "{tag}: row {i} prefill forwards"
                    );
                    // The positions fed past the prompt: every id plus the
                    // overrun, except that the blocking arm streams a cap
                    // row's last token without feeding it.
                    let fed = sessions[i].pos() - len;
                    let want_fed = if lookahead == 0 && !eos {
                        ids.len() - 1
                    } else {
                        ids.len() + outs[i].stats.overrun
                    };
                    assert_eq!(fed, want_fed, "{tag}: row {i} positions fed");
                }
                // Row 2 joins after its arrival; the earliest it can decode is
                // the step after its last chunk. Read off row 1's per-step
                // batch width (row 1 outlives everyone).
                let widths = &outs[1].stats.batch_rows;
                let first3 = widths.iter().position(|&b| b == 3);
                assert!(first3.is_some(), "{tag}: row 2 never joined: {widths:?}");
                // Hand-derived for the two schedules the design names.
                if chunk == 0 && tile == 0 {
                    // Rows 0 and 1 are fed at step 0 and decode from step 1;
                    // row 2 is fed whole at step 4 and decodes from step 5.
                    assert_eq!(first3, Some(4), "{tag}: {widths:?}");
                }
                if chunk == 3 && tile == 0 {
                    // The budget is a token budget over the pending rows in
                    // order: steps 0–1 feed row 0 three tokens each, step 2
                    // feeds its last token and, with two left in the budget,
                    // row 1's one — so both decode from step 3. Row 2 takes
                    // steps 4–5 and decodes from step 6: row 1's fourth step.
                    assert_eq!(first3, Some(3), "{tag}: {widths:?}");
                }
                // The doc's own test-file rule (`CONTRIBUTING.md` §3.1b): a
                // shape needs a control. The pieces are fed INSIDE the step's
                // forward, so a row cannot have joined a step earlier than its
                // last piece was fed — row 2's first live step is after step 4.
                if chunk > 0 {
                    assert!(
                        first3.unwrap() >= 1,
                        "{tag}: row 2 joined before its arrival: {widths:?}"
                    );
                }
            }
        }
    }
    Ok(())
}

// ---- the open decoder (`BatchDecoder`, issue #148) --------------------------

use lloom::nn::{BatchDecoder, RowId};

/// Step the decoder until nothing is live or pending, streaming into `streams`
/// keyed by slot, and return the rows retired along the way.
fn drain<'m, R: std::borrow::BorrowMut<Session<'m, Chain>>>(
    dec: &mut BatchDecoder<'m, Chain, R>,
    streams: &mut Vec<Vec<u32>>,
) -> Result<Vec<(RowId, R, lloom::nn::DecodeOut)>> {
    let mut finished = Vec::new();
    while !dec.is_idle() {
        dec.step(&mut |row, ids| {
            streams[row.slot()].extend_from_slice(ids);
            Ok(())
        })?;
        finished.extend(dec.take_finished());
    }
    Ok(finished)
}

/// A row admitted while others are mid-reply joins the batch at the next
/// step and produces its solo chain; the rows already live are undisturbed;
/// a retired row hands its session back with the closed driver's bookkeeping.
#[test]
fn a_row_admitted_mid_run_joins_the_batch_and_decodes_its_own_chain() -> Result<()> {
    for device in devices() {
        for lookahead in [0usize, 1] {
            for chunk in [0usize, 2] {
                let model = Chain::new(&device)?;
                let opts = DecodeOptions {
                    lookahead,
                    prefill_chunk: chunk,
                    ..DecodeOptions::new(12)
                };
                let mut dec: BatchDecoder<'_, Chain, Session<'_, Chain>> =
                    BatchDecoder::new(opts, &[EOS], device.clone())?;
                let mut streams: Vec<Vec<u32>> = vec![Vec::new(); 8];
                let a = dec.admit(Session::new(&model, 0usize), vec![0], Sampling::ArgMax, 0)?;
                let b = dec.admit(
                    Session::new(&model, 0usize),
                    vec![3, 1],
                    Sampling::ArgMax,
                    0,
                )?;
                // Three steps with two rows live, then a third row arrives.
                let mut finished = Vec::new();
                for _ in 0..3 {
                    dec.step(&mut |row, ids| {
                        streams[row.slot()].extend_from_slice(ids);
                        Ok(())
                    })?;
                    finished.extend(dec.take_finished());
                }
                let live_before = dec.live_rows();
                let c = dec.admit(
                    Session::new(&model, 0usize),
                    vec![9, 9, 9, 5],
                    Sampling::ArgMax,
                    0,
                )?;
                assert_eq!(dec.rows(), 3, "three rows held");
                assert_eq!(dec.pending_rows(), 1, "the new row is pending");
                finished.extend(drain(&mut dec, &mut streams)?);
                let tag = format!("{device:?} lookahead {lookahead} chunk {chunk}");
                assert_eq!(finished.len(), 3, "{tag}: every row retires");
                assert!(
                    dec.is_idle() && dec.rows() == 0,
                    "{tag}: nothing held after"
                );
                // Row b is the EOS row of the fixture (chain from 1 ends at its
                // ninth token); it ran with two live rows at least.
                assert!(live_before >= 1, "{tag}: rows were live at the arrival");
                for (id, session, out) in &finished {
                    let (start, prompt_len) = if *id == a {
                        (0u32, 1usize)
                    } else if *id == b {
                        (1, 2)
                    } else if *id == c {
                        (5, 4)
                    } else {
                        panic!("{tag}: an unknown row retired");
                    };
                    let (want, want_eos) = chain(start, 12);
                    assert_eq!(out.ids, want, "{tag}: row {} ids", id.slot());
                    assert_eq!(out.eos, want_eos, "{tag}: row {} eos", id.slot());
                    assert_eq!(streams[id.slot()], out.ids, "{tag}: streamed = ids");
                    // The handed-back session carries the closed driver's
                    // bookkeeping: prompt then the fed ids, overrun per arm.
                    assert_eq!(&session.tokens()[..prompt_len].len(), &prompt_len);
                    let fed = session.pos() - prompt_len;
                    let want_overrun = if lookahead == 1 && want_eos { 1 } else { 0 };
                    assert_eq!(out.stats.overrun, want_overrun, "{tag}: overrun");
                    assert_eq!(session.overrun(), want_overrun);
                    assert_eq!(&session.tokens()[prompt_len..], &want[..fed - want_overrun]);
                }
                // The late row joined a batch that was already decoding: its
                // first step saw more than one live row (the cap row outlives it
                // or at least shares its first steps).
                let c_out = finished.iter().find(|(id, _, _)| *id == c).unwrap();
                assert!(
                    c_out.2.stats.batch_rows.first().copied().unwrap_or(0) >= 2,
                    "{tag}: the late row's first step had {:?} live rows",
                    c_out.2.stats.batch_rows
                );
            }
        }
    }
    Ok(())
}

/// A cancelled row retires with the stream it produced so far, its session's
/// token log ending where the stream does; a row cancelled while its prompt
/// is still pending retires without ever joining. The other rows finish
/// their chains regardless.
#[test]
fn a_cancelled_row_retires_with_what_it_produced() -> Result<()> {
    for device in devices() {
        for lookahead in [0usize, 1] {
            let model = Chain::new(&device)?;
            let opts = DecodeOptions {
                lookahead,
                prefill_chunk: 1,
                ..DecodeOptions::new(12)
            };
            let mut dec: BatchDecoder<'_, Chain, Session<'_, Chain>> =
                BatchDecoder::new(opts, &[EOS], device.clone())?;
            let mut streams: Vec<Vec<u32>> = vec![Vec::new(); 8];
            let keep = dec.admit(Session::new(&model, 0usize), vec![0], Sampling::ArgMax, 0)?;
            let cut = dec.admit(Session::new(&model, 0usize), vec![0], Sampling::ArgMax, 0)?;
            // A four-token prompt at a one-token chunk is pending for four steps.
            let pending = dec.admit(
                Session::new(&model, 0usize),
                vec![9, 9, 9, 5],
                Sampling::ArgMax,
                0,
            )?;
            let mut finished = Vec::new();
            for _ in 0..4 {
                dec.step(&mut |row, ids| {
                    streams[row.slot()].extend_from_slice(ids);
                    Ok(())
                })?;
                finished.extend(dec.take_finished());
            }
            let produced_at_cancel = streams[cut.slot()].len();
            assert!(
                produced_at_cancel > 0,
                "the cut row produced before the cancel"
            );
            assert!(dec.cancel(cut), "a held row is cancellable");
            assert!(dec.cancel(pending), "a pending row is cancellable");
            assert!(
                !dec.cancel(cut) || dec.holds(cut),
                "cancel is idempotent while held"
            );
            finished.extend(drain(&mut dec, &mut streams)?);
            let tag = format!("{device:?} lookahead {lookahead}");
            assert_eq!(finished.len(), 3, "{tag}: all three retire");
            let (_, cut_session, cut_out) = finished.iter().find(|(id, _, _)| *id == cut).unwrap();
            assert_eq!(
                cut_out.ids.len(),
                produced_at_cancel,
                "{tag}: the cut row's stream"
            );
            assert!(!cut_out.eos);
            // The token log is the prompt plus the stream's fed prefix; the
            // lookahead arm's extra forward is the overrun, reconcilable.
            assert_eq!(&cut_session.tokens()[..1], &[0]);
            assert_eq!(
                cut_session.tokens().len() + cut_session.overrun(),
                cut_session.pos(),
                "{tag}: log + overrun = position"
            );
            let (_, pending_session, pending_out) =
                finished.iter().find(|(id, _, _)| *id == pending).unwrap();
            assert!(
                pending_out.ids.is_empty(),
                "{tag}: a pending row produced nothing"
            );
            assert!(
                pending_session.pos() <= 4
                    && pending_session.tokens().len() == pending_session.pos(),
                "{tag}: the pending row's log is the prompt it fed ({} of 4)",
                pending_session.pos()
            );
            assert!(
                !dec.holds(cut) && !dec.holds(pending),
                "{tag}: stale ids are not held"
            );
            let (_, _, keep_out) = finished.iter().find(|(id, _, _)| *id == keep).unwrap();
            assert_eq!(keep_out.ids, chain(0, 12).0, "{tag}: the kept row's chain");
        }
    }
    Ok(())
}

/// The host arm: a row with a temperature sampler draws from its own seeded
/// stream, so its ids are the same whether it decodes alone or beside other
/// rows; the argmax rows beside it still produce their chains (sampled on
/// the host on that step, which is the same argmax); and a mixed batch is
/// fed the host ids, so no row overruns whatever the lookahead option says.
#[test]
fn a_temperature_row_draws_from_its_own_seed_beside_any_neighbours() -> Result<()> {
    for device in devices() {
        let model = Chain::new(&device)?;
        let sampled = |neighbours: usize, seed: u64, lookahead: usize| -> Result<Vec<u32>> {
            let opts = DecodeOptions {
                lookahead,
                ..DecodeOptions::new(10)
            };
            let mut dec: BatchDecoder<'_, Chain, Session<'_, Chain>> =
                BatchDecoder::new(opts, &[EOS], device.clone())?;
            let mut streams: Vec<Vec<u32>> = vec![Vec::new(); 8];
            let t = dec.admit(
                Session::new(&model, 0usize),
                vec![3, 1],
                Sampling::All { temperature: 1.0 },
                seed,
            )?;
            let mut argmax_rows = Vec::new();
            for k in 0..neighbours {
                argmax_rows.push(dec.admit(
                    Session::new(&model, 0usize),
                    vec![k as u32],
                    Sampling::ArgMax,
                    0,
                )?);
            }
            let finished = drain(&mut dec, &mut streams)?;
            for (id, session, out) in &finished {
                if let Some(k) = argmax_rows.iter().position(|r| r == id) {
                    let (want, _) = chain(k as u32, 10);
                    assert_eq!(out.ids, want, "argmax row {k} beside a temperature row");
                }
                // The host arm feeds the sampled ids, so nothing runs past its end.
                assert_eq!(
                    out.stats.overrun,
                    0,
                    "row {}: no overrun on the host arm",
                    id.slot()
                );
                assert_eq!(session.overrun(), 0);
            }
            Ok(streams[t.slot()].clone())
        };
        let alone = sampled(0, 7, 1)?;
        let with_two = sampled(2, 7, 1)?;
        let with_three_blocking = sampled(3, 7, 0)?;
        assert!(!alone.is_empty());
        assert_eq!(
            alone, with_two,
            "{device:?}: the draw depends on the neighbours"
        );
        assert_eq!(
            alone, with_three_blocking,
            "{device:?}: the draw depends on the arm"
        );
        // Non-vacuity: the seed is what the draw depends on.
        let other_seed = sampled(0, 8, 1)?;
        assert_ne!(
            alone, other_seed,
            "{device:?}: two seeds drew the same stream"
        );
        // And a temperature draw is not the argmax chain (the fixture's
        // one-hot logits put ~4 % on the peak at t = 1).
        assert_ne!(
            alone,
            chain(1, 10).0,
            "{device:?}: the temperature row argmaxed"
        );
    }
    Ok(())
}

#[test]
fn filtered_gumbel_runs_the_existing_per_row_host_batch() -> Result<()> {
    let device = Device::Cpu;
    let model = Chain::new(&device)?;
    let mut decoder: BatchDecoder<'_, Chain, Session<'_, Chain>> =
        BatchDecoder::new(DecodeOptions::new(4), &[EOS], device.clone())?;
    let arm = Sampling::FilteredGumbel {
        temperature: 0.7,
        k: 1,
        p: 0.6,
    };
    let row = decoder.admit(Session::new(&model, 0usize), vec![1], arm.clone(), 7)?;
    let neighbour = decoder.admit(Session::new(&model, 0usize), vec![0], Sampling::ArgMax, 9)?;
    let mut streamed = std::collections::BTreeMap::<usize, Vec<u32>>::new();
    while !decoder.is_idle() {
        decoder.step(&mut |id, ids| {
            streamed
                .entry(id.slot())
                .or_default()
                .extend_from_slice(ids);
            Ok(())
        })?;
    }
    let finished = decoder.take_finished();
    assert_eq!(finished.len(), 2);
    for (id, session, out) in finished {
        let start = if id == row {
            1
        } else {
            assert_eq!(id, neighbour);
            0
        };
        let want = chain(start, 4).0;
        assert_eq!(out.ids, want);
        assert_eq!(streamed.get(&id.slot()).unwrap(), &want);
        assert_eq!(session.overrun(), 0);
        assert_eq!(*session.sequence().cache(), session.pos());
    }
    // Closed legacy batches still promise one device argmax, not per-row
    // sampling: the public capability must not promote that wrapper.
    let mut sessions = [Session::new(&model, 0usize)];
    let mut sampler = LogitsProcessor::from_sampling(7, arm);
    let error = Session::decode_batched(
        &mut sessions,
        &mut sampler,
        &[&[1]],
        &DecodeOptions::new(4),
        &[EOS],
        &device,
        &mut |_, _| panic!("closed-batch refusal reached callback"),
    )
    .unwrap_err();
    assert!(error.to_string().contains("per-row host sample"), "{error}");
    assert_eq!(sessions[0].pos(), 0);
    assert_eq!(*sessions[0].sequence().cache(), 0);
    Ok(())
}

// ---- prefilling the pending prompts together (`batched-decode-design.md` §11.8)

use lloom::nn::{PrefillDecode, StepReport};

/// One step, streaming into `streams` keyed by slot.
fn step_streaming<'m>(
    dec: &mut BatchDecoder<'m, Chain, Session<'m, Chain>>,
    streams: &mut [Vec<u32>],
) -> Result<StepReport> {
    dec.step(&mut |row, ids| {
        streams[row.slot()].extend_from_slice(ids);
        Ok(())
    })
}

/// Step until no row is pending, recording each step's report.
fn steps_until_joined<'m>(
    dec: &mut BatchDecoder<'m, Chain, Session<'m, Chain>>,
    streams: &mut [Vec<u32>],
) -> Result<Vec<StepReport>> {
    let mut reports = Vec::new();
    while dec.pending_rows() > 0 {
        reports.push(step_streaming(dec, streams)?);
    }
    Ok(reports)
}

/// `with_prefill_batch(k)`: a step's prompt budget is `k` chunks over the
/// pending rows in slot order, each row's piece at most one chunk, so the
/// pending prompts are fed together — four six-token prompts at a chunk of
/// two are in after 3 steps at k = 4 against 12 at k = 1 — and every row's
/// stream is its solo chain at every k. The budget is tokens, not rows: a
/// row that needs less than a chunk leaves the rest to the next row.
#[test]
fn a_step_carries_up_to_prefill_batch_rows_chunks() -> Result<()> {
    let prompts: Vec<Vec<u32>> = vec![
        vec![9, 9, 9, 9, 9, 1],
        vec![9, 9, 9, 9, 9, 0],
        vec![9, 9, 9, 9, 9, 5],
        vec![9, 9, 9, 9, 9, 7],
    ];
    let max = 10;
    // (k, the (pieces, tokens) of every step until no row is pending),
    // hand-derived: a group of up to k rows takes a two-token piece each
    // per step, three steps a group.
    let schedules: [(usize, Vec<(usize, usize)>); 5] = [
        (1, vec![(1, 2); 12]),
        (2, vec![(2, 4); 6]),
        (3, [vec![(3, 6); 3], vec![(1, 2); 3]].concat()),
        (4, vec![(4, 8); 3]),
        (8, vec![(4, 8); 3]),
    ];
    for device in devices() {
        let model = Chain::new(&device)?;
        for lookahead in [0usize, 1] {
            for (k, want_steps) in &schedules {
                let tag = format!("{device:?} lookahead {lookahead} prefill_batch {k}");
                let opts = DecodeOptions {
                    lookahead,
                    prefill_chunk: 2,
                    ..DecodeOptions::new(max)
                };
                let mut dec: BatchDecoder<'_, Chain, Session<'_, Chain>> =
                    BatchDecoder::new(opts, &[EOS], device.clone())?.with_prefill_batch(*k);
                assert_eq!(dec.prefill_batch(), *k, "{tag}");
                let ids: Vec<RowId> = prompts
                    .iter()
                    .map(|p| {
                        dec.admit(Session::new(&model, 0usize), p.clone(), Sampling::ArgMax, 0)
                    })
                    .collect::<Result<_>>()?;
                let mut streams: Vec<Vec<u32>> = vec![Vec::new(); 8];
                let reports = steps_until_joined(&mut dec, &mut streams)?;
                let got: Vec<(usize, usize)> =
                    reports.iter().map(|r| (r.pieces, r.piece_tokens)).collect();
                assert_eq!(&got, want_steps, "{tag}: the steps' pieces and tokens");
                let finished = drain(&mut dec, &mut streams)?;
                assert_eq!(finished.len(), prompts.len(), "{tag}: every row retires");
                for (id, _, out) in &finished {
                    let r = ids.iter().position(|x| x == id).expect("an admitted row");
                    let (want, want_eos) = chain(*prompts[r].last().unwrap(), max);
                    assert_eq!(out.ids, want, "{tag}: row {r} ids");
                    assert_eq!(out.eos, want_eos, "{tag}: row {r} eos");
                    assert_eq!(streams[id.slot()], want, "{tag}: row {r} streamed");
                    assert_eq!(out.stats.prefill_units, 3, "{tag}: row {r} pieces");
                }
            }
        }
        // The budget is tokens: at k = 2 (four tokens a step) a one-token
        // prompt leaves three, so a third row is fed in the same step.
        let opts = DecodeOptions {
            prefill_chunk: 2,
            ..DecodeOptions::new(4)
        };
        let mut dec: BatchDecoder<'_, Chain, Session<'_, Chain>> =
            BatchDecoder::new(opts, &[EOS], device.clone())?.with_prefill_batch(2);
        for p in [vec![9u32, 9, 1], vec![0u32], vec![9u32, 5]] {
            dec.admit(Session::new(&model, 0usize), p, Sampling::ArgMax, 0)?;
        }
        let mut streams: Vec<Vec<u32>> = vec![Vec::new(); 8];
        let first = step_streaming(&mut dec, &mut streams)?;
        assert_eq!(
            (first.pieces, first.piece_tokens),
            (3, 4),
            "{device:?}: 2 + 1 + 1 tokens in the first step"
        );
    }
    Ok(())
}

/// Under `PrefillDecode::Pause` a row that has streamed waits while pending
/// prompts' pieces are fed — held, neither sampled nor fed, its logits kept
/// — and a row still owing its first token takes it beside the pieces;
/// every row ends on its solo chain. Under `Ride` the streaming row takes a
/// token every step. Steps 3–7 are hand-derived: `b`'s one piece, then
/// `c`'s three, at a chunk of two and one chunk a step.
#[test]
fn the_pause_arm_holds_streaming_rows_while_pieces_are_fed() -> Result<()> {
    let max = 12;
    for device in devices() {
        let model = Chain::new(&device)?;
        for lookahead in [0usize, 1] {
            for arm in [PrefillDecode::Ride, PrefillDecode::Pause] {
                let tag = format!("{device:?} lookahead {lookahead} {arm}");
                let opts = DecodeOptions {
                    lookahead,
                    prefill_chunk: 2,
                    ..DecodeOptions::new(max)
                };
                let mut dec: BatchDecoder<'_, Chain, Session<'_, Chain>> =
                    BatchDecoder::new(opts, &[EOS], device.clone())?.with_prefill_decode(arm);
                let mut streams: Vec<Vec<u32>> = vec![Vec::new(); 8];
                let a = dec.admit(Session::new(&model, 0usize), vec![0], Sampling::ArgMax, 0)?;
                // Step 0 feeds a's prompt; steps 1 and 2 stream its first two.
                for _ in 0..3 {
                    step_streaming(&mut dec, &mut streams)?;
                }
                assert_eq!(streams[a.slot()].len(), 2, "{tag}: a before the arrivals");
                let b = dec.admit(
                    Session::new(&model, 0usize),
                    vec![3, 1],
                    Sampling::ArgMax,
                    0,
                )?;
                let c = dec.admit(
                    Session::new(&model, 0usize),
                    vec![9, 9, 9, 9, 9, 5],
                    Sampling::ArgMax,
                    0,
                )?;
                let mut reports = Vec::new();
                let mut a_len = Vec::new();
                let mut b_len = Vec::new();
                for _ in 3..8 {
                    reports.push(step_streaming(&mut dec, &mut streams)?);
                    a_len.push(streams[a.slot()].len());
                    b_len.push(streams[b.slot()].len());
                }
                let got: Vec<(usize, usize, usize)> =
                    reports.iter().map(|r| (r.live, r.held, r.pieces)).collect();
                match arm {
                    PrefillDecode::Pause => {
                        assert_eq!(
                            got,
                            vec![(0, 1, 1), (1, 1, 1), (0, 2, 1), (0, 2, 1), (3, 0, 0)],
                            "{tag}: (live, held, pieces) at steps 3-7"
                        );
                        assert_eq!(a_len, vec![2, 2, 2, 2, 3], "{tag}: a waits for the pieces");
                        assert_eq!(
                            b_len,
                            vec![0, 1, 1, 1, 2],
                            "{tag}: b's first token is not held"
                        );
                    }
                    PrefillDecode::Ride => {
                        assert_eq!(
                            got,
                            vec![(1, 0, 1), (2, 0, 1), (2, 0, 1), (2, 0, 1), (3, 0, 0)],
                            "{tag}: (live, held, pieces) at steps 3-7"
                        );
                        assert_eq!(a_len, vec![3, 4, 5, 6, 7], "{tag}: a rides every step");
                        assert_eq!(b_len, vec![0, 1, 2, 3, 4], "{tag}: b rides from its join");
                    }
                }
                let finished = drain(&mut dec, &mut streams)?;
                assert_eq!(finished.len(), 3, "{tag}: every row retires");
                for (id, start) in [(a, 0u32), (b, 1), (c, 5)] {
                    let (want, _) = chain(start, max);
                    assert_eq!(streams[id.slot()], want, "{tag}: row {} stream", id.slot());
                }
            }
        }
    }
    Ok(())
}

// ---- the decode step's scope and its stages (`batched-decode-design.md` §13)

/// [`Chain`] recording, per entry point, how many calls ran with a batched
/// decode step's scope (`tuning::DecodeRows`) open and how many without.
struct ScopeChain {
    inner: Chain,
    /// `(open, closed)` calls: `forward_rows`, `forward_ragged`, `forward`.
    rows: ScopeCount,
    ragged: ScopeCount,
    single: ScopeCount,
}

/// Calls made with the scope open and without it.
#[derive(Default)]
struct ScopeCount(std::sync::atomic::AtomicU32, std::sync::atomic::AtomicU32);

impl ScopeCount {
    fn get(&self) -> (u32, u32) {
        use std::sync::atomic::Ordering::Relaxed;
        (self.0.load(Relaxed), self.1.load(Relaxed))
    }
}

impl ScopeChain {
    fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            inner: Chain::new(device)?,
            rows: Default::default(),
            ragged: Default::default(),
            single: Default::default(),
        })
    }

    fn note(c: &ScopeCount) {
        use std::sync::atomic::Ordering::Relaxed;
        if lloom_tensor::tuning::in_decode_rows() {
            c.0.fetch_add(1, Relaxed);
        } else {
            c.1.fetch_add(1, Relaxed);
        }
    }
}

impl Model for ScopeChain {
    type Config = ();
    fn from_weights(_vb: lloom::nn::VarBuilder, _cfg: &()) -> Result<Self> {
        lloom_tensor::bail!("ScopeChain has no weights")
    }
}

impl CausalLM for ScopeChain {
    type Cache = usize;

    fn new_cache(&self, cfg: &(), dev: &Device) -> Result<usize> {
        self.inner.new_cache(cfg, dev)
    }

    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.inner.embed(tokens)
    }

    fn forward_embeds(&self, embeds: &Tensor, pos: usize, cache: &mut usize) -> Result<Tensor> {
        Self::note(&self.single);
        self.inner.forward_embeds(embeds, pos, cache)
    }

    fn vocab_size(&self) -> usize {
        VOCAB
    }
}

impl BatchedCausalLM for ScopeChain {
    fn forward_ragged(&self, tokens: &Tensor, rows: &mut [RaggedRow<'_, usize>]) -> Result<Tensor> {
        Self::note(&self.ragged);
        self.inner.forward_ragged(tokens, rows)
    }

    fn forward_rows(&self, tokens: &Tensor, rows: &mut [Row<'_, usize>]) -> Result<Tensor> {
        Self::note(&self.rows);
        self.inner.forward_rows(tokens, rows)
    }
}

/// The decode step's forward — and only it — runs with the scope
/// `kernels.gemm.f16_rows_decode` routes open: every `forward_rows` call
/// has it, and no step that carries prompt pieces (`forward_ragged`) and no
/// whole-prompt prefill (`forward`) does; the scope is closed again between
/// steps. On both lookahead arms, with chunked and whole-prompt admission.
#[test]
fn only_the_decode_steps_forward_opens_the_decode_rows_scope() -> Result<()> {
    for device in devices() {
        for lookahead in [0usize, 1] {
            for chunk in [0usize, 2] {
                let tag = format!("{device:?} lookahead {lookahead} chunk {chunk}");
                let model = ScopeChain::new(&device)?;
                let opts = DecodeOptions {
                    lookahead,
                    prefill_chunk: chunk,
                    ..DecodeOptions::new(6)
                };
                let mut dec: BatchDecoder<'_, ScopeChain, Session<'_, ScopeChain>> =
                    BatchDecoder::new(opts, &[EOS], device.clone())?;
                for p in [vec![9u32, 9, 1], vec![9, 0], vec![5]] {
                    dec.admit(Session::new(&model, 0usize), p, Sampling::ArgMax, 0)?;
                }
                let mut n = 0;
                while !dec.is_idle() {
                    dec.step(&mut |_, _| Ok(()))?;
                    assert!(
                        !lloom_tensor::tuning::in_decode_rows(),
                        "{tag}: the scope outlived the step"
                    );
                    n += 1;
                    assert!(n < 100, "{tag}: the batch never drained");
                }
                let (rows, ragged, single) =
                    (model.rows.get(), model.ragged.get(), model.single.get());
                assert!(rows.0 > 0, "{tag}: no decode step ran ({rows:?})");
                assert_eq!(rows.1, 0, "{tag}: a decode step ran without the scope");
                assert_eq!(ragged.0, 0, "{tag}: a step with pieces opened the scope");
                assert_eq!(single.0, 0, "{tag}: a prefill opened the scope");
                if chunk > 0 {
                    assert!(ragged.1 > 0, "{tag}: no piece step ran ({ragged:?})");
                } else {
                    assert!(single.1 > 0, "{tag}: no whole prompt was fed ({single:?})");
                }
            }
        }
    }
    Ok(())
}

/// A step's host stages are read where the build carries the clock (debug
/// and `profile`): a decode step's encode is charged, and the stages sum to
/// no more than the step's own host time. In a release build every stage is
/// zero by construction (the clock reads compiled out).
#[test]
fn a_steps_host_stages_partition_its_time() -> Result<()> {
    let device = Device::Cpu;
    let model = Chain::new(&device)?;
    let opts = DecodeOptions {
        lookahead: 1,
        ..DecodeOptions::new(8)
    };
    let mut dec: BatchDecoder<'_, Chain, Session<'_, Chain>> =
        BatchDecoder::new(opts, &[EOS], device.clone())?;
    dec.admit(
        Session::new(&model, 0usize),
        vec![9, 1],
        Sampling::ArgMax,
        0,
    )?;
    dec.admit(
        Session::new(&model, 0usize),
        vec![9, 0],
        Sampling::ArgMax,
        0,
    )?;
    let mut decode_steps = 0;
    while !dec.is_idle() {
        let t = std::time::Instant::now();
        let report = dec.step(&mut |_, _| Ok(()))?;
        let wall = t.elapsed().as_nanos() as u64;
        let ph = dec.last_step_phases();
        if cfg!(debug_assertions) {
            assert!(
                ph.total_ns() <= wall,
                "the stages sum past the step: {} > {wall} ({ph:?})",
                ph.total_ns()
            );
            if report.live > 0 {
                decode_steps += 1;
                assert!(
                    ph.encode_ns > 0,
                    "a decode step's encode was not charged: {ph:?}"
                );
            }
        } else {
            assert_eq!(ph.total_ns(), 0, "release reads no clock: {ph:?}");
        }
    }
    if cfg!(debug_assertions) {
        assert!(decode_steps > 0, "no decode step ran");
    }
    Ok(())
}
