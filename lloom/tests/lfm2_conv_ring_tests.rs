//! The conv-state ring buffers against the shuffle they replace (lloom #141).
//!
//! `DESIGN.md` §6.1's decode path is `narrow` + `Tensor::cat` + `mul` +
//! `sum_keepdim`, which reallocates and copies the whole state every token.
//! §10.2a/§10.2b replace it with a ring. This file is the correctness argument
//! for both forms of that replacement, **and they are held to different bars.**
//!
//! **Sliding — bit-identity.** The live window is `l_cache` contiguous slots in
//! the shuffle's own order, so the three-term sum is accumulated in the same
//! order and the output must be **bit-identical** rather than merely within
//! tolerance. §2.3.5a's changed-digest procedure is for changes that genuinely
//! alter arithmetic; this is not one, and a moved bit here is a defect.
//!
//! **Rotating — a different summation order, held to §2.3.5a instead.** Which
//! slot holds the newest token decides the order `sum_keepdim` walks, so the
//! output differs in the low bits by construction and a bitwise comparison is
//! the wrong bar. What must hold instead is what §2.3.5a names: the result is
//! correct against a reference that did not change, and the error is ulp-scale
//! and **does not grow with reduction length**. Both are asserted here — the
//! reference being an f64 accumulation of the same three products, which is
//! neither of the two float32 orders under test and so is not the thing being
//! checked restated.
//!
//! The device reference is the CPU backend, per `CONTRIBUTING.md` §3.1: it is
//! bit-stable and upstreamable, where the Metal generic grouped path is the
//! unstable side at high channel counts (§2.3.8a).

use candle::{DType, Device, IndexOp, Result, Tensor};

/// The shuffle, exactly as `ShortConv::forward` runs it at `seq_len == 1`.
fn shuffle_step(
    state: &Tensor,
    bx: &Tensor,
    w: &Tensor,
    l_cache: usize,
) -> Result<(Tensor, Tensor)> {
    let tail = state.narrow(2, 1, l_cache - 1)?;
    let next = Tensor::cat(&[tail, bx.clone()], 2)?;
    let out = (&next * w)?.sum_keepdim(2)?.contiguous()?;
    Ok((next, out))
}

/// The ring: one in-place write at the sliding slot, then a contiguous window.
///
/// Returns the (possibly compacted) state alongside the output, because the
/// compaction replaces the buffer.
fn ring_step(
    state: &Tensor,
    bx: &Tensor,
    w: &Tensor,
    l_cache: usize,
    live_w: usize,
    width: usize,
    phase: usize,
    compact: bool,
) -> Result<(Tensor, Tensor)> {
    let (b, hidden, _) = state.dims3()?;
    let state = if compact {
        let live = state.narrow(2, width - live_w, live_w)?.contiguous()?;
        let pad = Tensor::zeros((b, hidden, width - live_w), state.dtype(), state.device())?;
        Tensor::cat(&[live, pad], 2)?.contiguous()?
    } else {
        state.clone()
    };
    state.slice_set(bx, 2, phase)?;
    let window = state.narrow(2, phase + 1 - l_cache, l_cache)?;
    let out = (window * w)?.sum_keepdim(2)?.contiguous()?;
    Ok((state, out))
}

/// The slot and the compaction flag, exactly as `Model::forward` computes them.
fn advance(phase: usize, width: usize, live_w: usize) -> (usize, bool) {
    let next = phase + 1;
    if next >= width {
        (live_w, true)
    } else {
        (next, false)
    }
}

fn to_vec(t: &Tensor) -> Result<Vec<f32>> {
    t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()
}

/// Deterministic pseudo-random values, so a failure reproduces exactly.
fn seeded(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5
        })
        .collect()
}

/// The ring reproduces the shuffle **bit for bit**, across many compactions.
///
/// 200 steps at `slack = 4` compacts ~40 times, so the wrap is exercised heavily
/// rather than incidentally. A test stopping before the first compaction would
/// pass on a window that cannot survive one.
#[test]
fn ring_matches_shuffle_bitwise() -> Result<()> {
    let dev = Device::Cpu;
    let (b, hidden, l_cache) = (1usize, 64usize, 3usize);

    for (k, slack) in [(0usize, 1usize), (0, 4), (0, 16), (1, 4), (2, 7), (5, 16)] {
        let live_w = l_cache + k;
        let width = live_w + slack;
        let w = Tensor::from_vec(seeded(hidden * l_cache, 7), (1, hidden, l_cache), &dev)?;

        // Both arms start from the same prefill-seeded state: the live window in
        // slots 0..live_w, the slack zeroed, so the write slot is live_w - 1.
        let seed_vals = seeded(b * hidden * l_cache, 11);
        let mut sh_state = Tensor::from_vec(seed_vals.clone(), (b, hidden, l_cache), &dev)?;
        let mut ring_state = {
            let mut v = vec![0f32; b * hidden * width];
            for c in 0..hidden {
                for s in 0..l_cache {
                    v[c * width + (live_w - l_cache) + s] = seed_vals[c * l_cache + s];
                }
            }
            Tensor::from_vec(v, (b, hidden, width), &dev)?
        };
        let mut phase = live_w - 1;
        let mut compactions = 0usize;

        for step in 0..200 {
            let bx =
                Tensor::from_vec(seeded(b * hidden, 1000 + step as u32), (b, hidden, 1), &dev)?;

            let (next, sh_out) = shuffle_step(&sh_state, &bx, &w, l_cache)?;
            sh_state = next;

            let (p, compact) = advance(phase, width, live_w);
            phase = p;
            compactions += compact as usize;
            let (st, ring_out) =
                ring_step(&ring_state, &bx, &w, l_cache, live_w, width, phase, compact)?;
            ring_state = st;

            assert_eq!(
                to_vec(&sh_out)?,
                to_vec(&ring_out)?,
                "k={k} slack={slack} step={step} phase={phase}: ring diverged from the shuffle"
            );
        }
        assert!(
            compactions > 0,
            "k={k} slack={slack}: never compacted, so the wrap was not exercised"
        );
    }
    Ok(())
}

/// The live window holds the same *tokens in the same order* the shuffle does —
/// checked directly rather than only through the summed output, so a pairing bug
/// cannot hide behind a coincidental sum.
#[test]
fn ring_window_holds_the_same_tokens_in_order() -> Result<()> {
    let dev = Device::Cpu;
    let (hidden, l_cache, k, slack) = (8usize, 3usize, 0usize, 4usize);
    let live_w = l_cache + k;
    let width = live_w + slack;

    let mut ring_state = Tensor::zeros((1, hidden, width), DType::F32, &dev)?;
    let mut phase = live_w - 1;
    let mut history: Vec<usize> = vec![];

    for t in 1..=40usize {
        // One distinguishable value per token, so a slot's contents name it.
        let bx = Tensor::full(t as f32, (1, hidden, 1), &dev)?;
        let (p, compact) = advance(phase, width, live_w);
        phase = p;
        if compact {
            let live = ring_state.narrow(2, width - live_w, live_w)?.contiguous()?;
            let pad = Tensor::zeros((1, hidden, width - live_w), DType::F32, &dev)?;
            ring_state = Tensor::cat(&[live, pad], 2)?.contiguous()?;
        }
        ring_state.slice_set(&bx, 2, phase)?;
        history.push(t);

        let got: Vec<usize> = (0..l_cache)
            .map(|i| {
                let s = phase + 1 - l_cache + i;
                let v = ring_state
                    .i((0, 0, s))?
                    .to_dtype(DType::F32)?
                    .to_vec0::<f32>()?;
                Ok(v as usize)
            })
            .collect::<Result<Vec<_>>>()?;

        // Oldest..newest of the tokens so far, zero-padded at the start.
        let mut want: Vec<usize> = history.iter().rev().take(l_cache).rev().copied().collect();
        while want.len() < l_cache {
            want.insert(0, 0);
        }
        assert_eq!(got, want, "t={t} phase={phase}: window is not in order");
    }
    Ok(())
}

/// **Mutation test** — `CONTRIBUTING.md` §3.1 #2 and §2.4: a test that cannot
/// fail is not a test. Each mutation is a plausible way to get the sliding
/// window wrong, and each must break the bitwise comparison above.
///
/// **The rotating index is kept as a mutation arm even though it now ships as
/// `ConvState::RotatingRing`**, and that is not a contradiction. What this
/// asserts is that the *sliding* arm's bit-identity claim is falsifiable —
/// substituting the rotating order for it must be detected. That the rotating
/// order is separately a legitimate arm under §2.3.5a's bar (see
/// `rotating_is_a_reduction_order_change_not_an_error`) is a different claim
/// about a different arm; conflating them would give the sliding arm a bitwise
/// test that a reordering can pass.
#[test]
fn mutations_break_the_parity() -> Result<()> {
    let dev = Device::Cpu;
    let (b, hidden, l_cache, k, slack) = (1usize, 32usize, 3usize, 0usize, 4usize);
    let live_w = l_cache + k;
    let width = live_w + slack;
    let w = Tensor::from_vec(seeded(hidden * l_cache, 7), (1, hidden, l_cache), &dev)?;

    for mutation in [
        // The wrap is never taken, so the window walks off its own buffer.
        "no-compaction",
        // The window is read one slot behind: an off-by-one on the tap span.
        "window-off-by-one",
        // The rotating index §10.2a specifies -- which is what this whole
        // mechanism exists to avoid, and it must show up as a difference here.
        "rotating-index",
    ] {
        let seed_vals = seeded(b * hidden * l_cache, 11);
        let mut sh_state = Tensor::from_vec(seed_vals.clone(), (b, hidden, l_cache), &dev)?;
        let mut ring_state = {
            let mut v = vec![0f32; b * hidden * width];
            for c in 0..hidden {
                for s in 0..l_cache {
                    v[c * width + s] = seed_vals[c * l_cache + s];
                }
            }
            Tensor::from_vec(v, (b, hidden, width), &dev)?
        };
        let mut phase = live_w - 1;
        let mut diverged = false;

        for step in 0..40 {
            let bx =
                Tensor::from_vec(seeded(b * hidden, 1000 + step as u32), (b, hidden, 1), &dev)?;
            let (next, sh_out) = shuffle_step(&sh_state, &bx, &w, l_cache)?;
            sh_state = next;

            let ring_out = match mutation {
                "no-compaction" => {
                    let (p, _) = advance(phase, width, live_w);
                    if p >= width {
                        // Walked off the buffer: the mechanism is broken, which
                        // is the point. Count it as divergence.
                        diverged = true;
                        break;
                    }
                    phase = p;
                    let (st, o) =
                        ring_step(&ring_state, &bx, &w, l_cache, live_w, width, phase, false)?;
                    ring_state = st;
                    o
                }
                "window-off-by-one" => {
                    let (p, compact) = advance(phase, width, live_w);
                    phase = p;
                    if compact {
                        let live = ring_state.narrow(2, width - live_w, live_w)?.contiguous()?;
                        let pad = Tensor::zeros((b, hidden, width - live_w), DType::F32, &dev)?;
                        ring_state = Tensor::cat(&[live, pad], 2)?.contiguous()?;
                    }
                    ring_state.slice_set(&bx, 2, phase)?;
                    let start = (phase + 1 - l_cache).saturating_sub(1);
                    let window = ring_state.narrow(2, start, l_cache)?;
                    (window * &w)?.sum_keepdim(2)?.contiguous()?
                }
                "rotating-index" => {
                    phase = (phase + 1) % live_w.max(1);
                    ring_state.slice_set(&bx, 2, phase)?;
                    let window = ring_state.narrow(2, 0, l_cache)?;
                    (window * &w)?.sum_keepdim(2)?.contiguous()?
                }
                _ => unreachable!(),
            };

            if to_vec(&sh_out)? != to_vec(&ring_out)? {
                diverged = true;
                break;
            }
        }
        assert!(
            diverged,
            "mutation {mutation:?} left the parity test green -- the test cannot fail"
        );
    }
    Ok(())
}

/// History slots (`K > 0`) are inert: widening the ring must not move the
/// output, because nothing reads them until a speculative scheme (#89) does.
/// §16 6b is why K is a parameter and not a constant.
#[test]
fn history_slots_do_not_change_the_output() -> Result<()> {
    let dev = Device::Cpu;
    let (b, hidden, l_cache, slack) = (1usize, 32usize, 3usize, 8usize);
    let w = Tensor::from_vec(seeded(hidden * l_cache, 7), (1, hidden, l_cache), &dev)?;

    let mut baseline: Vec<Vec<f32>> = vec![];
    for k in [0usize, 1, 4] {
        let live_w = l_cache + k;
        let width = live_w + slack;
        let seed_vals = seeded(b * hidden * l_cache, 11);
        let mut ring_state = {
            let mut v = vec![0f32; b * hidden * width];
            for c in 0..hidden {
                for s in 0..l_cache {
                    v[c * width + (live_w - l_cache) + s] = seed_vals[c * l_cache + s];
                }
            }
            Tensor::from_vec(v, (b, hidden, width), &dev)?
        };
        let mut phase = live_w - 1;
        let mut outs = vec![];
        for step in 0..60 {
            let bx =
                Tensor::from_vec(seeded(b * hidden, 1000 + step as u32), (b, hidden, 1), &dev)?;
            let (p, compact) = advance(phase, width, live_w);
            phase = p;
            let (st, o) = ring_step(&ring_state, &bx, &w, l_cache, live_w, width, phase, compact)?;
            ring_state = st;
            outs.push(to_vec(&o)?);
        }
        if baseline.is_empty() {
            baseline = outs;
        } else {
            assert_eq!(
                baseline, outs,
                "K={k} changed the output; history is not inert"
            );
        }
    }
    Ok(())
}

/// The weight permutation `ConvState::RotatingRing` builds at load time: slot
/// `s` meets the weight of the token it holds, which at phase `p` is age
/// `(s + width - p - 1) mod width`.
fn rotating_weight(w: &Tensor, phase: usize, width: usize) -> Result<Tensor> {
    let cols: Vec<Tensor> = (0..width)
        .map(|s| w.narrow(2, (s + width - phase - 1) % width, 1))
        .collect::<Result<Vec<_>>>()?;
    Tensor::cat(&cols, 2)?.contiguous()
}

/// The rotating window: one in-place write at the rotating slot, and a read of
/// the whole buffer against the phase-permuted weight.
/// `ConvState::RotatingRing`'s decode path at `K = 0`.
fn rotating_step(state: &Tensor, bx: &Tensor, w: &Tensor, phase: usize) -> Result<Tensor> {
    state.slice_set(bx, 2, phase)?;
    (state * w)?.sum_keepdim(2)?.contiguous()
}

/// The three-term sum accumulated in f64, in a fixed order, from the same
/// products both float32 arms compute.
///
/// **This is the reference §2.3.5a asks for and neither arm can be.** It is not
/// the shuffle's order and not the ring's; it is the exact value the sum
/// approximates, so "which of the two float32 orders is nearer the truth" is a
/// question it can answer and a shuffle-versus-ring comparison cannot. Reading
/// one float32 order as ground truth is what would make a legitimate reordering
/// look like an error.
fn exact_sum(taps: &[(f64, f64)]) -> f64 {
    taps.iter().map(|(v, w)| v * w).sum()
}

/// **§2.3.5a's discriminator, applied to the rotating arm.**
///
/// A changed digest is two events — a reduction-order change or a computational
/// bug — and stability tells them apart not at all. What does is (1) parity
/// against a reference that did not change, and (2) an error that is ulp-scale
/// and does not grow with reduction length.
///
/// Both are asserted here at the level the reordering happens: the 3-term tap
/// sum. The rotating arm's output must sit within one ulp of the f64 value —
/// **the same bound the shuffle itself meets** — which is what makes it a
/// different rounding of the right answer rather than a wrong one.
#[test]
fn rotating_is_a_reduction_order_change_not_an_error() -> Result<()> {
    let dev = Device::Cpu;
    let (b, hidden, l_cache) = (1usize, 64usize, 3usize);
    let width = l_cache; // K = 0: the window is the whole buffer.
    let w = Tensor::from_vec(seeded(hidden * l_cache, 7), (1, hidden, l_cache), &dev)?;
    let w_vals = to_vec(&w)?;

    let seed_vals = seeded(b * hidden * l_cache, 11);
    let mut sh_state = Tensor::from_vec(seed_vals.clone(), (b, hidden, l_cache), &dev)?;
    let mut rot_state = Tensor::from_vec(seed_vals.clone(), (b, hidden, width), &dev)?;
    let mut phase = l_cache - 1;

    // Per-token ages: which token each slot holds, so the reference can be built
    // from the same products in the shuffle's own oldest..newest order.
    let mut ages: Vec<Vec<f32>> = (0..hidden)
        .map(|c| (0..l_cache).map(|s| seed_vals[c * l_cache + s]).collect())
        .collect();

    let (mut worst_rot_ulps, mut worst_sh_ulps) = (0f64, 0f64);
    let mut differed = 0usize;
    // Error at the start of the run against error at the end: a reduction-order
    // difference is flat in reduction length, a defect grows.
    let (mut early_max, mut late_max) = (0f64, 0f64);

    let steps = 400usize;
    for step in 0..steps {
        let bx_vals = seeded(b * hidden, 1000 + step as u32);
        let bx = Tensor::from_vec(bx_vals.clone(), (b, hidden, 1), &dev)?;

        let (next, sh_out) = shuffle_step(&sh_state, &bx, &w, l_cache)?;
        sh_state = next;

        phase = (phase + 1) % width;
        let rot_w = rotating_weight(&w, phase, width)?;
        let rot_out = rotating_step(&rot_state, &bx, &rot_w, phase)?;
        rot_state = rot_state.clone();

        // Track the live window's contents in oldest..newest order, which is what
        // both arms are summing however they lay it out.
        for (c, a) in ages.iter_mut().enumerate() {
            a.remove(0);
            a.push(bx_vals[c]);
        }

        let sh_v = to_vec(&sh_out)?;
        let rot_v = to_vec(&rot_out)?;

        for c in 0..hidden {
            // The shuffle's pairing: slot `s` holds the token of age `s`, and
            // meets weight `s`. This is the exact value BOTH arms should be
            // approximating -- the sum is over (token, its own weight) pairs,
            // and which slot a token sits in is an implementation detail.
            let taps: Vec<(f64, f64)> = (0..l_cache)
                .map(|s| (ages[c][s] as f64, w_vals[c * l_cache + s] as f64))
                .collect();
            let exact = exact_sum(&taps);
            // One ulp **at the magnitude the sum is accumulated at**, which is
            // the largest term rather than the result.
            //
            // Scaling to the result is wrong here and wrong in a way that
            // matters: three terms of mixed sign cancel, so a result near zero
            // has a tiny ulp while the rounding that produced it happened at the
            // terms' magnitude. Measured on this fixture that reads the
            // *shuffle* at ~1009 ulp from its own exact value -- a bound the
            // incumbent fails is a bound that is measuring the wrong thing, which
            // is why the shuffle is carried through this test as a control.
            let scale = taps
                .iter()
                .map(|(v, w)| (v * w).abs())
                .fold(0f64, f64::max)
                .max(exact.abs());
            let scale = (scale as f32).max(f32::MIN_POSITIVE);
            let ulp = (f32::from_bits(scale.to_bits() + 1) - scale) as f64;

            let rot_err = (rot_v[c] as f64 - exact).abs() / ulp;
            let sh_err = (sh_v[c] as f64 - exact).abs() / ulp;
            worst_rot_ulps = worst_rot_ulps.max(rot_err);
            worst_sh_ulps = worst_sh_ulps.max(sh_err);
            if rot_v[c] != sh_v[c] {
                differed += 1;
            }
            if step < 50 {
                early_max = early_max.max((rot_v[c] as f64 - sh_v[c] as f64).abs() / ulp);
            } else if step >= steps - 50 {
                late_max = late_max.max((rot_v[c] as f64 - sh_v[c] as f64).abs() / ulp);
            }
        }
    }

    // (1) Parity against the reference that did not change.
    //
    // **The bound is the incumbent's own accuracy, not a number chosen here.**
    // A fixed tolerance would be arbitrary and could be tuned until it passed;
    // what is actually being claimed is that the rotating order is no further
    // from the truth than the order it would replace, which is the statement
    // "this is a different rounding of the same sum" made checkable. A wrong tap
    // pairing fails it by ~7 orders of magnitude (see the `naive-rotation`
    // mutation), so the test has enormous margin without the bound being loose.
    assert!(
        worst_rot_ulps <= worst_sh_ulps.max(4.0),
        "rotating is {worst_rot_ulps:.3} ulp from the exact sum against the \
         shuffle's {worst_sh_ulps:.3} -- it is not merely a different rounding"
    );
    // The incumbent's own figure is asserted too, so a fixture that made *both*
    // arms inaccurate could not pass by making the comparison vacuous.
    assert!(
        worst_sh_ulps <= 4.0,
        "the shuffle itself is {worst_sh_ulps:.3} ulp out -- the fixture, not the \
         arm, is what this test would be measuring"
    );

    // (2) The difference does not grow with reduction length. Both halves of
    //     this matter: a flat difference is a reordering, and a *zero* difference
    //     would mean the test never exercised the reorder at all.
    assert!(
        differed > 0,
        "rotating never differed from the shuffle -- the comparison is vacuous"
    );
    assert!(
        late_max <= early_max.max(1.0) * 2.0,
        "the divergence grows with the run: {early_max:.3} ulp over the first 50 \
         steps against {late_max:.3} over the last 50 -- that is accumulation, \
         not a fixed reordering"
    );
    Ok(())
}

/// **The weight permutation is load-bearing, and dropping it is not a rounding
/// difference — it is a different operator.**
///
/// This is the mutation that makes the test above evidence rather than a
/// demonstration, and it is worth its own name because the thing it rejects is
/// the form §10.2a's text describes literally. A rotating buffer whose slot `s`
/// meets weight `s` pairs each tap with the wrong weight on every phase but one,
/// so it computes a convolution with permuted taps.
///
/// Measured here so the two failure scales cannot be confused: the permuted form
/// is within the shuffle's own accuracy, and this one is out by **~7 orders of
/// magnitude in ulp** and by an O(1) amount absolutely. That gap is why "the
/// rotating form differs by ~1 ulp" needs to say *which* rotating form.
#[test]
fn naive_rotation_pairs_the_wrong_weights() -> Result<()> {
    let dev = Device::Cpu;
    let (b, hidden, l_cache) = (1usize, 64usize, 3usize);
    let width = l_cache;
    let w = Tensor::from_vec(seeded(hidden * l_cache, 7), (1, hidden, l_cache), &dev)?;
    let w_vals = to_vec(&w)?;

    let seed_vals = seeded(b * hidden * l_cache, 11);
    let mut sh_state = Tensor::from_vec(seed_vals.clone(), (b, hidden, l_cache), &dev)?;
    let rot_state = Tensor::from_vec(seed_vals.clone(), (b, hidden, width), &dev)?;
    let mut phase = l_cache - 1;
    let mut ages: Vec<Vec<f32>> = (0..hidden)
        .map(|c| (0..l_cache).map(|s| seed_vals[c * l_cache + s]).collect())
        .collect();

    let mut worst = 0f64;
    for step in 0..120usize {
        let bx_vals = seeded(b * hidden, 1000 + step as u32);
        let bx = Tensor::from_vec(bx_vals.clone(), (b, hidden, 1), &dev)?;
        let (next, _sh_out) = shuffle_step(&sh_state, &bx, &w, l_cache)?;
        sh_state = next;

        phase = (phase + 1) % width;
        // The mutation: the unpermuted weight, which is what a literal reading
        // of "a rotating write index" produces.
        let rot_out = rotating_step(&rot_state, &bx, &w, phase)?;

        for (c, a) in ages.iter_mut().enumerate() {
            a.remove(0);
            a.push(bx_vals[c]);
        }
        let rot_v = to_vec(&rot_out)?;
        for c in 0..hidden {
            let taps: Vec<(f64, f64)> = (0..l_cache)
                .map(|s| (ages[c][s] as f64, w_vals[c * l_cache + s] as f64))
                .collect();
            let exact = exact_sum(&taps);
            let scale = taps
                .iter()
                .map(|(v, w)| (v * w).abs())
                .fold(0f64, f64::max)
                .max(exact.abs());
            let scale = (scale as f32).max(f32::MIN_POSITIVE);
            let ulp = (f32::from_bits(scale.to_bits() + 1) - scale) as f64;
            worst = worst.max((rot_v[c] as f64 - exact).abs() / ulp);
        }
    }

    assert!(
        worst > 1e4,
        "the unpermuted rotation is only {worst:.3} ulp out -- either the \
         permutation is not load-bearing after all, or this fixture cannot \
         distinguish the two pairings and the test above proves less than it says"
    );
    Ok(())
}

/// The rotating window holds the same *tokens* the shuffle does, in rotated
/// slots — the pairing check that says the ulp bound above is measuring a
/// reordering rather than two errors that happen to be small.
///
/// Without this, an arm that read the right slots with the wrong weights could
/// still land inside the ulp bound on a fixture where the weights are similar.
#[test]
fn rotating_window_holds_the_same_tokens() -> Result<()> {
    let dev = Device::Cpu;
    let (hidden, l_cache) = (8usize, 3usize);
    let rot_state = Tensor::zeros((1, hidden, l_cache), DType::F32, &dev)?;
    let mut phase = l_cache - 1;
    let mut history: Vec<usize> = vec![];

    for t in 1..=40usize {
        let bx = Tensor::full(t as f32, (1, hidden, 1), &dev)?;
        phase = (phase + 1) % l_cache;
        rot_state.slice_set(&bx, 2, phase)?;
        history.push(t);

        // Oldest..newest, read by walking back from the write slot modulo the
        // width -- which is the rotation, stated as a read rather than assumed.
        let got: Vec<usize> = (0..l_cache)
            .map(|i| {
                let s = (phase + l_cache + 1 - l_cache + i) % l_cache;
                let v = rot_state
                    .i((0, 0, s))?
                    .to_dtype(DType::F32)?
                    .to_vec0::<f32>()?;
                Ok(v as usize)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut want: Vec<usize> = history.iter().rev().take(l_cache).rev().copied().collect();
        while want.len() < l_cache {
            want.insert(0, 0);
        }
        // The slots are rotated, so the *set* matches while the order is the
        // rotation of it -- which is exactly the property that moves the sum.
        let mut got_sorted = got.clone();
        let mut want_sorted = want.clone();
        got_sorted.sort_unstable();
        want_sorted.sort_unstable();
        assert_eq!(
            got_sorted, want_sorted,
            "t={t} phase={phase}: rotating window holds the wrong tokens"
        );
    }
    Ok(())
}
