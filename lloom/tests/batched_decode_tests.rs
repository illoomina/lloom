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
use lloom::nn::{BatchedCausalLM, CausalLM, DecodeOptions, IdentityProposer, Model, Row, Session};
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
                    // The forwards the prompt took: whole, or per chunk, each
                    // bounded by the tile.
                    let len = prompts[i].len();
                    let pieces: Vec<usize> = if chunk == 0 {
                        vec![len]
                    } else {
                        let mut v = Vec::new();
                        let mut left = len;
                        while left > 0 {
                            let t = left.min(chunk);
                            v.push(t);
                            left -= t;
                        }
                        v
                    };
                    let want_units: usize = pieces
                        .iter()
                        .map(|&p| if tile == 0 { 1 } else { p.div_ceil(tile) })
                        .sum();
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
            }
        }
    }
    Ok(())
}
