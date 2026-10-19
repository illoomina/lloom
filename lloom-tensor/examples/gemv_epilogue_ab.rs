//! Does a fused elementwise EPILOGUE on a GEMV beat the two-dispatch pair?
//!
//! Measurement harness for lloom issue #484 / `DESIGN.md` §12.2 #3, and the
//! arm §12.3's counter-pressure is aimed at.
//!
//! # Why a microbenchmark and not the decode path
//!
//! §13.1a: `CANDLE_METAL_PROFILE` records dispatch COUNTS per kernel and GPU
//! INTERVALS per command buffer, and the two series can never be joined --
//! dispatches inside one encoder overlap (`MTLDispatchType::Concurrent`), so
//! the command buffer is the smallest unit Metal reports that does not
//! double-count. Per-kernel GPU time is INEXPRESSIVE at dispatch granularity,
//! and the one route that would give it (`CANDLE_METAL_COMPUTE_PER_BUFFER=1`)
//! is measured at **+9.75 % GPU busy, 67x the null** -- it changes the thing it
//! measures.
//!
//! So the pair is timed HERE, where each arm is its own command buffer and
//! `GPUEndTime - GPUStartTime` attributes cleanly, and the decode-path question
//! ("does the step get faster") is a separate measurement with its own
//! instrument.
//!
//! # What is measured
//!
//! **Six arms** over identical inputs, at LFM2's own `w1` geometry by default.
//! The first two are #488's and are unchanged; the rest were added by #492 to
//! decompose their difference, because attributing it to *"traffic"* does not
//! survive its own arithmetic (the intermediate's round trip is 0.006-3.0 us
//! against a 5-331 us effect).
//!
//! * **unfused** -- `gemv` writes the accumulator, then `silu` transforms it.
//!   Two dispatches, and the intermediate makes a full round trip to memory.
//! * **fused** -- `gemv_silu` applies the transform in registers at the store.
//!   One dispatch, no round trip.
//! * **gemv-only** -- the GEMV alone. The floor the other arms are built on.
//! * **gemv+tiny** -- the GEMV and a second dispatch over ONE element.
//! * **gemv+neg** -- the GEMV and a second dispatch over N elements.
//! * **gemv+neg-indep** -- the same, reading a buffer the GEMV never wrote, so
//!   no hazard and **no barrier**. Against `gemv+neg` it isolates the barrier.
//!
//! All are driven through the real `call_mlx_gemv*` and `call_unary_contiguous`
//! entry points, so the tile selection and the binding order are the shipping
//! path's rather than this file's.
//!
//! # The tile is an axis, and it was not one
//!
//! The GEMV's tile is derived from N (`mlx_gemm.rs`: bm8 at `out_vec_size >=
//! 4096`, bm4 below), and `select_tile_config` is never consulted on this path,
//! so `LLOOM_383_TILE` cannot reach it. A sweep in N therefore crosses a KERNEL
//! BOUNDARY at 4096 -- which is inside #488's own range, between its 2048 and
//! 6144 points, exactly where its curve bends. `LLOOM_492_GEMV_BM=4|8` pins it
//! so N is the variable; the tile is legible in the kernel census either way.
//!
//! # Method, and every clause is a rule from CONTRIBUTING.md
//!
//! * **Interleaved A/B** (§3.2). Run A, B, A, B -- never all of A then all of
//!   B -- so the build is not confounded with thermal state and machine load.
//!   `tools/gpu-pstate/run-ab.sh` already interleaves; this does the same.
//! * **A null control in every rotation** (§1.3). The same arm in two slots.
//!   If the null is larger than the effect, the honest verdict is "no effect
//!   resolvable at this noise level".
//! * **The median, not the mean** (§3.2a #3, §6.6b). Run-to-run cv is 18.89 %
//!   on the mean against 8.75 % on the step median in this corpus's own
//!   population.
//! * **A bootstrap CI and no p-values** (§3.2). Our series are autocorrelated
//!   and `N_eff` here is the rotation count.
//! * **GPU time from `GPUEndTime - GPUStartTime`**, in domain A (§3.4b), which
//!   is the same clock everything else in this project brackets in.
//!
//! # Engagement
//!
//! §2.4: an instrument that cannot be shown to have engaged has measured
//! nothing. Two independent checks, both printed and both non-optional:
//!
//! 1. **The arms disagree numerically.** The fused arm computes SiLU; if the
//!    two arms' outputs matched bit for bit, the epilogue would not be running.
//! 2. **The arms differ in dispatch count**, 1 against 2, reported from the
//!    encode loop rather than assumed.
//!
//! ```bash
//! lloom-arb measure -- cargo run --release --example gemv_epilogue_ab
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use lloom_tensor::kernels::metal::{Commands, Device, ResidencySet};
use lloom_tensor::kernels::{
    call_mlx_gemv_full, call_mlx_gemv_with, call_unary_contiguous, unary, BufferOffset, GemmDType,
    GemvEpilogue, Kernels, ParamStyle,
};
use objc2_metal::MTLResourceOptions;

#[derive(Parser, Debug)]
#[command(about = "gemv + SiLU epilogue against the two-dispatch pair (#484)")]
struct Args {
    /// K, the input vector length. LFM2's hidden_size (`DESIGN.md` §5.2).
    #[arg(long, default_value_t = 2048)]
    k: usize,

    /// N values to sweep, each an output vector length.
    ///
    /// The default brackets LFM2's own sites and spans the cache question the
    /// registered prediction is about: at f16 the intermediate is `2*N` bytes,
    /// so 512 is 1 KiB and 65536 is 128 KiB, against a 48 MB SLC. The point of
    /// the sweep is that the two-dispatch arm's intermediate has to be
    /// *written and read back*, and where that round trip lands changes with
    /// N.
    #[arg(
        long,
        value_delimiter = ',',
        default_values_t = [512usize, 2048, 6144, 10752, 32768, 65536, 131072]
    )]
    n: Vec<usize>,

    /// Interleaved rotations. Each runs **six** slots: unfused, fused, the
    /// three decomposition arms (`gemv-only`, `gemv+tiny`, `gemv+neg`), the
    /// barrier arm (`gemv+neg-indep`), and a null slot last.
    ///
    /// All of them inside one rotation, deliberately: the decomposition
    /// subtracts these arms from each other, so a term measured in its own
    /// batch would carry the drift between batches and report it as a
    /// mechanism (§3.2). The cost is that the null now brackets a wider
    /// window, which is why it is reported per N rather than once.
    #[arg(long, default_value_t = 12)]
    rotations: usize,

    /// Timed repetitions inside one slot; the median is that slot's figure.
    #[arg(long, default_value_t = 50)]
    iters: usize,

    /// Untimed repetitions before each slot is timed.
    #[arg(long, default_value_t = 10)]
    warmup: usize,

    /// Element type. **`f16` is what LFM2 decodes in** (`DESIGN.md` §2.3.3b,
    /// §5.1) and is therefore the arm any decode-path claim belongs to; `f32`
    /// is the precision §2.3.5a's parity discriminator wants and is where the
    /// tests run. Both are measured because they are different questions:
    /// §12.1's traffic argument is about BYTES, and the intermediate is 2 B
    /// per element at f16 against 4 at f32.
    #[arg(long, default_value = "f16")]
    dtype: String,
}

/// Median of a slice.
///
/// Median rather than mean, per §3.2a #3: an occasional scheduler preemption
/// produces an outlier the mean absorbs and reports as typical.
fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n.is_multiple_of(2) {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    }
}

/// A deterministic bootstrap CI on the paired percentage difference.
///
/// Deterministic because `DESIGN.md` §2.3 makes reproducibility an invariant
/// and a harness seeded from the clock cannot be re-run to the same interval.
/// The generator is a plain LCG rather than a crate: the quality bar for a
/// resampling index is low and pinning it here means the interval is a
/// property of the data.
fn bootstrap_ci(pairs: &[(f64, f64)], reps: usize) -> (f64, f64) {
    if pairs.len() < 2 {
        return (f64::NAN, f64::NAN);
    }
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut deltas = Vec::with_capacity(reps);
    for _ in 0..reps {
        let mut a = Vec::with_capacity(pairs.len());
        let mut b = Vec::with_capacity(pairs.len());
        for _ in 0..pairs.len() {
            let i = (next() % pairs.len() as u64) as usize;
            a.push(pairs[i].0);
            b.push(pairs[i].1);
        }
        let (ma, mb) = (median(&a), median(&b));
        deltas.push((mb - ma) / ma * 100.0);
    }
    deltas.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let lo = deltas[(reps as f64 * 0.025) as usize];
    let hi = deltas[((reps as f64 * 0.975) as usize).min(reps - 1)];
    (lo, hi)
}

/// Which arm a slot runs.
///
/// #488 measured the first two and reported their difference as one number.
/// **That difference is a sum of four terms with two signs**, and this issue
/// exists because attributing it to *"traffic"* does not survive its own
/// arithmetic: the intermediate is 1 KiB to 512 KiB across the sweep, so a
/// round trip of it is 0.006–3.0 µs at §3.4's 349.7 GB/s, against savings of
/// 5–300 µs. The account is short by 8× to 2135×, on both dtypes, at **every**
/// point — not only at the crossover.
///
/// So the four arms below decompose it, and the decomposition is **checkable**
/// rather than asserted: the three differences must reconstruct the two-arm
/// effect, and the harness prints the residual (`Solver`'s reconciliation, and
/// §15.1a's rule that a figure must carry a quantity that would differ if the
/// instrument had not run).
///
/// ```text
/// dispatch  = GemvPlusTiny - GemvOnly        one launch, ~no work, ~no traffic
/// traffic   = GemvPlusNeg  - GemvPlusTiny    the same launch over N elements
/// silu-only = Unfused      - GemvPlusNeg     what the transcendental adds
/// epilogue  = Fused        - GemvOnly        what fusing COSTS the gemv (§12.3a)
///
/// (Unfused - Fused)  ==  dispatch + traffic + silu-only - epilogue
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    /// `gemv` then `silu`: two dispatches, one intermediate round trip.
    /// #488's arm A, unchanged.
    Unfused,
    /// `gemv_silu`: one dispatch, transform applied in registers.
    /// #488's arm B, unchanged.
    Fused,
    /// `gemv` alone. The floor both other arms are built on, and the one
    /// #488 did not run — without it, *"what the epilogue costs"* and *"what
    /// the second dispatch costs"* are one number.
    GemvOnly,
    /// `gemv` then `neg` over **one** element.
    ///
    /// The dispatch-cost arm the issue asks for, and it is a real candle
    /// kernel through the real entry point rather than a hand-rolled empty
    /// one: same encode path, same binding work, same pipeline lookup, and a
    /// grid of 1 so neither traffic nor arithmetic is in it. `neg` rather than
    /// `silu` because a transcendental over one element would put a small
    /// amount of the thing being isolated back into the isolator.
    GemvPlusTiny,
    /// `gemv` then `neg` over all N elements.
    ///
    /// The same dispatch as `GemvPlusTiny` doing the same READS and WRITES as
    /// `silu` — `-x` is one instruction against `x/(1+exp(-x))` — so the step
    /// from `GemvPlusTiny` is the intermediate's round trip *as the machine
    /// actually pays for it*, which is the quantity §12.1 predicts and this
    /// issue measures. The step from here to `Unfused` is the transcendental
    /// alone.
    GemvPlusNeg,
    /// `gemv` into `gate`, then `neg` over N elements of an **unrelated**
    /// buffer into a third.
    ///
    /// The barrier arm, and it is the one that says what the "dispatch" term
    /// above actually is. `GemvPlusTiny` and `GemvPlusNeg` both read what the
    /// GEMV just wrote, so candle's hazard tracking emits a **barrier** between
    /// the two dispatches (`encoder.rs`'s RAW detection): the second cannot
    /// start until the first has fully drained. This arm reads a buffer the
    /// GEMV never touched, so there is no hazard, no barrier, and the two
    /// dispatches are free to overlap (§3.5: dispatches within one encoder
    /// overlap; serialization comes from encoder breaks and barriers, not from
    /// the GPU).
    ///
    /// **Same two kernels, same lengths, same bytes moved — one bit of
    /// dependency different.** So `GemvPlusNegIndep` against `GemvPlusNeg`
    /// isolates the barrier from everything else, which is the discriminator
    /// the `+tiny` term needs: a second dispatch over ONE element cannot cost
    /// 7.6→51.2 µs as a *launch*, and this arm is what distinguishes a launch
    /// cost from a lost overlap.
    GemvPlusNegIndep,
}

impl Arm {
    /// The name used in the artifact. Written once so a column heading and a
    /// reconciliation term cannot drift apart.
    fn label(self) -> &'static str {
        match self {
            Arm::Unfused => "unfused",
            Arm::Fused => "fused",
            Arm::GemvOnly => "gemv-only",
            Arm::GemvPlusTiny => "gemv+tiny",
            Arm::GemvPlusNeg => "gemv+neg",
            Arm::GemvPlusNegIndep => "gemv+neg-indep",
        }
    }
}

/// What one timed slot reports.
struct Slot {
    /// GPU-busy union over the slot's timed repetitions, in ms per repetition.
    ///
    /// The UNION rather than the sum, per `profile.rs`'s own note: dispatches
    /// inside one encoder overlap, so a sum over-counts. This is the figure
    /// that module says to trust.
    gpu_ms: f64,
    /// Dispatch counts by pipeline label, which is the ENGAGEMENT evidence
    /// (§2.4): it names the kernels that actually ran rather than the ones the
    /// flag asked for.
    by_label: Vec<(String, u64)>,
    /// The last repetition's final output.
    out: Vec<f32>,
    /// The last repetition's UNTRANSFORMED gemv accumulator.
    ///
    /// Only the unfused arm has one as a separate buffer -- that is the
    /// intermediate whose round trip the fusion removes. It is the engagement
    /// evidence: the fused arm's output must differ from THIS (or the epilogue
    /// is not applied) while agreeing with the unfused arm's final output (or
    /// it is applied wrongly). Comparing the two arms' FINAL outputs proves
    /// nothing -- they are meant to agree, and a first version of this harness
    /// asserted they must differ and failed on a correct kernel.
    raw: Vec<f32>,
}

/// One timed slot: `iters` repetitions, each its own command buffer.
///
/// Each repetition is its own command buffer deliberately. That is what makes
/// `GPUEndTime - GPUStartTime` attributable to the arm rather than to a batch
/// (§13.1a), and it is applied IDENTICALLY to both arms, so the per-command-
/// buffer overhead is a constant added to both and cancels in the paired
/// difference. It does mean the absolute figures carry that overhead and are
/// not a per-kernel time; they are not reported as one.
#[allow(clippy::too_many_arguments)]
fn run_slot(
    device: &Device,
    kernels: &Kernels,
    arm: Arm,
    dtype: GemmDType,
    k: usize,
    n_out: usize,
    iters: usize,
    warmup: usize,
) -> Result<Slot> {
    let options = MTLResourceOptions::StorageModeShared;

    // The element size and the `silu` variant follow the dtype. Both are read
    // from it rather than written twice, so a dtype added here cannot be
    // half-wired -- which is the shape §7.1a records failing five times.
    let elem_size = match dtype {
        GemmDType::F32 => 4usize,
        GemmDType::F16 | GemmDType::BF16 => 2,
    };
    // `contiguous::Kernel` is not `Copy`, so it is built at the call site
    // rather than hoisted; it is a name lookup, not work.
    let silu_kernel = || match dtype {
        GemmDType::F32 => unary::contiguous::silu::FLOAT,
        GemmDType::F16 => unary::contiguous::silu::HALF,
        GemmDType::BF16 => unary::contiguous::silu::BFLOAT,
    };
    // `neg` is the decomposition arms' second dispatch: `-x`, one instruction,
    // against `silu`'s `x/(1+exp(-x))`. Same family, same entry point, same
    // grid rule, same reads and writes — so a `neg` over N and a `silu` over N
    // differ in the transcendental and in nothing else, and a `neg` over 1 and
    // a `neg` over N differ in the traffic and in nothing else. Two steps, one
    // variable each, which is what §9.2c's blindness rule asks of a fixture.
    let neg_kernel = || match dtype {
        GemmDType::F32 => unary::contiguous::neg::FLOAT,
        GemmDType::F16 => unary::contiguous::neg::HALF,
        GemmDType::BF16 => unary::contiguous::neg::BFLOAT,
    };

    // The matrix is [N_out, K] and the vector is [K]. Generated in f32 and
    // narrowed, so both dtype arms see the same NUMBERS and the comparison
    // between them is about precision rather than about a different fixture.
    let mat: Vec<f32> = (0..n_out * k)
        .map(|i| ((i % 17) as f32 - 8.0) / 32.0)
        .collect();
    let vec: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 16.0).collect();

    let mk = |v: &[f32]| -> Result<_> {
        let bytes: Vec<u8> = match dtype {
            GemmDType::F32 => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            GemmDType::F16 => v
                .iter()
                .flat_map(|x| half::f16::from_f32(*x).to_ne_bytes())
                .collect(),
            GemmDType::BF16 => v
                .iter()
                .flat_map(|x| half::bf16::from_f32(*x).to_ne_bytes())
                .collect(),
        };
        device
            .new_buffer_with_data(
                bytes.as_ptr() as *const core::ffi::c_void,
                bytes.len(),
                options,
            )
            .map_err(|e| anyhow::anyhow!("{e:?}"))
    };
    let mat_buf = mk(&mat)?;
    let vec_buf = mk(&vec)?;
    let gate = device
        .new_buffer(n_out * elem_size, options)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let out = device
        .new_buffer(n_out * elem_size, options)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    // The independent arm's source and destination. Neither is ever bound by
    // the GEMV, so a `neg` reading `spare_src` raises no hazard against it and
    // candle emits no barrier -- which is the single bit this arm varies.
    // Allocated unconditionally rather than per arm so the allocation itself
    // is not a difference between arms (§9.2c's blindness rule: a fixture that
    // varies a second axis is measuring its own perturbation, which is what
    // #462 shouted to the pit).
    let spare_src = device
        .new_buffer(n_out * elem_size, options)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let spare_dst = device
        .new_buffer(n_out * elem_size, options)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let mut last = Vec::new();
    let mut raw = Vec::new();

    for rep in 0..(warmup + iters) {
        // The profiling window opens AFTER warmup, so the first-touch cost
        // §2.4 measures at 35x-130x is outside every reported figure.
        if rep == warmup {
            lloom_tensor::kernels::metal::profile::reset();
        }
        let queue = device
            .new_command_queue()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let residency = std::sync::Arc::new(ResidencySet::new(device));
        let commands =
            Commands::new(queue, residency.clone()).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let encoder = commands
            .command_encoder()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        match arm {
            Arm::Unfused => {
                // SAFETY: the example owns every buffer bound or mapped here and reads them
                // only after waiting for the command buffer that writes them.
                unsafe {
                    call_mlx_gemv_with(
                        device,
                        &encoder,
                        kernels,
                        dtype,
                        (1, n_out, 1, k),
                        &[n_out * k, k, 1],
                        0,
                        &mat_buf,
                        &[k, 1, 1],
                        0,
                        &vec_buf,
                        &gate,
                        ParamStyle::Split,
                    )
                }
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                // SAFETY: the example owns every buffer bound or mapped here and reads them
                // only after waiting for the command buffer that writes them.
                unsafe {
                    call_unary_contiguous(
                        device,
                        &encoder,
                        kernels,
                        silu_kernel(),
                        elem_size,
                        n_out,
                        BufferOffset {
                            buffer: &gate,
                            offset_in_bytes: 0,
                        },
                        &out,
                    )
                }
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            }
            Arm::Fused => {
                // SAFETY: the example owns every buffer bound or mapped here and reads them
                // only after waiting for the command buffer that writes them.
                unsafe {
                    call_mlx_gemv_full(
                        device,
                        &encoder,
                        kernels,
                        dtype,
                        (1, n_out, 1, k),
                        &[n_out * k, k, 1],
                        0,
                        &mat_buf,
                        &[k, 1, 1],
                        0,
                        &vec_buf,
                        &out,
                        ParamStyle::Split,
                        None,
                        GemvEpilogue::Silu,
                    )
                }
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            }
            // The three decomposition arms. Each writes the gemv accumulator
            // into `gate` exactly as the unfused arm does, so the gemv half is
            // byte-for-byte the same work in all four and the differences
            // isolate the second dispatch.
            Arm::GemvOnly | Arm::GemvPlusTiny | Arm::GemvPlusNeg | Arm::GemvPlusNegIndep => {
                // SAFETY: the example owns every buffer bound or mapped here and reads them
                // only after waiting for the command buffer that writes them.
                unsafe {
                    call_mlx_gemv_with(
                        device,
                        &encoder,
                        kernels,
                        dtype,
                        (1, n_out, 1, k),
                        &[n_out * k, k, 1],
                        0,
                        &mat_buf,
                        &[k, 1, 1],
                        0,
                        &vec_buf,
                        &gate,
                        ParamStyle::Split,
                    )
                }
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                // `GemvOnly` stops here. The other two add a second dispatch
                // that differs ONLY in its length, which is what makes the
                // step between them the traffic term rather than a comparison
                // of two different kernels.
                // (length, source, destination) for the second dispatch.
                // `GemvPlusNegIndep` differs from `GemvPlusNeg` ONLY in the
                // buffers, which is what makes their difference the barrier.
                let second = match arm {
                    Arm::GemvOnly => None,
                    Arm::GemvPlusTiny => Some((1usize, &gate, &out)),
                    Arm::GemvPlusNeg => Some((n_out, &gate, &out)),
                    Arm::GemvPlusNegIndep => Some((n_out, &spare_src, &spare_dst)),
                    _ => unreachable!("only the four decomposition arms reach here"),
                };
                if let Some((len, src, dst)) = second {
                    // SAFETY: the example owns every buffer bound or mapped here and reads them
                    // only after waiting for the command buffer that writes them.
                    unsafe {
                        call_unary_contiguous(
                            device,
                            &encoder,
                            kernels,
                            neg_kernel(),
                            elem_size,
                            len,
                            BufferOffset {
                                buffer: src,
                                offset_in_bytes: 0,
                            },
                            dst,
                        )
                    }
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                }
            }
        }

        drop(encoder);
        commands
            .wait_until_completed()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        if rep == warmup + iters - 1 {
            last = read_as_f32(&out, n_out, dtype);
            if arm == Arm::Unfused {
                raw = read_as_f32(&gate, n_out, dtype);
            }
        }
    }

    let snap = lloom_tensor::kernels::metal::profile::snapshot();
    // `timed_command_buffers` below `iters` means a completion handler had not
    // run when the window was read, which would silently shorten the union.
    // Refused rather than tolerated: §15.1a's VACUOUS class is a figure taken
    // from an instrument that did not fully report.
    anyhow::ensure!(
        snap.timed_command_buffers >= iters as u64,
        "only {} of {iters} command buffers reported an interval; the GPU-busy \
         union is short and the figure would be wrong low",
        snap.timed_command_buffers
    );

    Ok(Slot {
        gpu_ms: snap.gpu_busy_union_s * 1e3 / iters as f64,
        by_label: snap.by_label,
        out: last,
        raw,
    })
}

/// Read a buffer back and widen to f32, whatever it holds.
///
/// Widening rather than comparing in the storage type: the engagement checks
/// are about VALUES, and comparing f16 bit patterns would make "the epilogue
/// ran" depend on the rounding as well as on the transform.
fn read_as_f32(
    buffer: &lloom_tensor::kernels::metal::Buffer,
    len: usize,
    dtype: GemmDType,
) -> Vec<f32> {
    // SAFETY: the example owns every buffer bound or mapped here and reads them
    // only after waiting for the command buffer that writes them.
    let ptr = unsafe { buffer.contents() };
    // §11.5d / #454: the null check must survive the optimiser, and a bare
    // `assert!` on `contents()` is elided at `opt-level=1` with LTO off.
    assert!(
        !std::hint::black_box(ptr).is_null(),
        "buffer is not CPU-mapped"
    );
    match dtype {
        GemmDType::F32 => unsafe { std::slice::from_raw_parts(ptr as *const f32, len).to_vec() },
        GemmDType::F16 => unsafe {
            std::slice::from_raw_parts(ptr as *const half::f16, len)
                .iter()
                .map(|x| x.to_f32())
                .collect()
        },
        GemmDType::BF16 => unsafe {
            std::slice::from_raw_parts(ptr as *const half::bf16, len)
                .iter()
                .map(|x| x.to_f32())
                .collect()
        },
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    lloom_tensor::tuning::require_profile_recorder().context(
        "this interval benchmark requires cargo run --profile release-with-debug -p lloom-tensor --features metal,profile --example gemv_epilogue_ab -- <original arguments>",
    )?;
    anyhow::ensure!(
        lloom_tensor::tuning::try_tuning()?.instrument.profile,
        "set CANDLE_METAL_PROFILE=1: this benchmark requires command-buffer intervals and per-label census"
    );

    let device = Device::system_default().context("no Metal device")?;
    let kernels = Kernels::new();

    // An unknown dtype PANICS rather than falling back: §2.4 — an arm that
    // silently ran the baseline is #69's vacuous determinism run, and the
    // whole point of a dtype flag is to be compared across.
    let dtype = match args.dtype.as_str() {
        "f32" => GemmDType::F32,
        "f16" => GemmDType::F16,
        "bf16" => GemmDType::BF16,
        other => anyhow::bail!("--dtype must be one of f32|f16|bf16, got {other:?}"),
    };

    println!("# issue #484 -- gemv + SiLU epilogue against the two-dispatch pair");
    println!("# DESIGN.md 12.2 #3, and the arm 12.3's counter-pressure is aimed at");
    println!(
        "# K={} rotations={} iters={} warmup={}",
        args.k, args.rotations, args.iters, args.warmup
    );
    println!(
        "# dtype={} (f16 is what LFM2 decodes in; 5.1, 2.3.3b)",
        args.dtype
    );
    // The GEMV tile is DERIVED from N (`mlx_gemm.rs`: `out_vec_size >= 4096`
    // selects bm8, below it bm4), so a sweep in N alone crosses a kernel
    // boundary at 4096 -- which is inside #488's own sweep, between its 2048
    // and 6144 points. `LLOOM_492_GEMV_BM` pins it so N is the variable.
    // Stated on the header line because a run whose tile was pinned and a run
    // whose tile was derived are not the same measurement, and the artifact
    // must say which (§15.1b's provenance argument, at the level of an arm).
    println!(
        "# LLOOM_492_GEMV_BM={} (unset = derived from N: bm8 at N>=4096, bm4 below)",
        std::env::var("LLOOM_492_GEMV_BM").unwrap_or_else(|_| "unset".to_string())
    );
    println!();

    for &n_out in &args.n {
        // Every rotation runs three slots, so a per-slot bias (3.4a-i #3) is
        // shared and the null bounds it.
        let mut unfused = Vec::new();
        let mut fused = Vec::new();
        let mut null_b = Vec::new();
        let mut labels_u = Vec::new();
        let mut labels_f = Vec::new();
        let mut out_u = Vec::new();
        let mut raw_u = Vec::new();
        let mut out_f = Vec::new();

        let mut gemv_only = Vec::new();
        let mut gemv_tiny = Vec::new();
        let mut gemv_neg = Vec::new();
        let mut gemv_indep = Vec::new();
        let mut labels_o = Vec::new();
        let mut labels_t = Vec::new();
        let mut labels_n = Vec::new();
        let mut labels_i = Vec::new();

        for _ in 0..args.rotations {
            let u = run_slot(
                &device,
                &kernels,
                Arm::Unfused,
                dtype,
                args.k,
                n_out,
                args.iters,
                args.warmup,
            )?;
            let f = run_slot(
                &device,
                &kernels,
                Arm::Fused,
                dtype,
                args.k,
                n_out,
                args.iters,
                args.warmup,
            )?;
            // The three decomposition arms sit INSIDE the same rotation as the
            // two they decompose. That is not tidiness: a term measured in its
            // own batch is measured against a different machine state, and the
            // reconciliation below subtracts these from each other, so a drift
            // between batches would appear as a residual and be read as a
            // missing mechanism. Interleaving is what makes the residual
            // interpretable (§3.2).
            let o = run_slot(
                &device,
                &kernels,
                Arm::GemvOnly,
                dtype,
                args.k,
                n_out,
                args.iters,
                args.warmup,
            )?;
            let t = run_slot(
                &device,
                &kernels,
                Arm::GemvPlusTiny,
                dtype,
                args.k,
                n_out,
                args.iters,
                args.warmup,
            )?;
            let g = run_slot(
                &device,
                &kernels,
                Arm::GemvPlusNeg,
                dtype,
                args.k,
                n_out,
                args.iters,
                args.warmup,
            )?;
            // The barrier arm: identical to `g` except in which buffers the
            // second dispatch touches, so their difference is the barrier and
            // nothing else.
            let i = run_slot(
                &device,
                &kernels,
                Arm::GemvPlusNegIndep,
                dtype,
                args.k,
                n_out,
                args.iters,
                args.warmup,
            )?;
            // The null: the SAME arm in a second slot. Its spread is what any
            // claimed effect has to clear (§1.3). It is run LAST so the
            // unfused arm occupies the first and last slots -- the null
            // therefore brackets every other arm, and a monotone per-slot
            // drift (§3.4a-i #3) shows up in it rather than hiding inside an
            // effect. With five arms between them the null now bounds a wider
            // window than #488's did, which is the honest cost of running more
            // arms in one rotation and is why it is reported per N.
            let u2 = run_slot(
                &device,
                &kernels,
                Arm::Unfused,
                dtype,
                args.k,
                n_out,
                args.iters,
                args.warmup,
            )?;

            unfused.push(u.gpu_ms);
            fused.push(f.gpu_ms);
            gemv_only.push(o.gpu_ms);
            gemv_tiny.push(t.gpu_ms);
            gemv_neg.push(g.gpu_ms);
            gemv_indep.push(i.gpu_ms);
            null_b.push(u2.gpu_ms);
            labels_u = u.by_label;
            labels_f = f.by_label;
            labels_o = o.by_label;
            labels_t = t.by_label;
            labels_n = g.by_label;
            labels_i = i.by_label;
            out_u = u.out;
            raw_u = u.raw;
            out_f = f.out;
        }
        let null_a = unfused.clone();

        let m_u = median(&unfused);
        let m_f = median(&fused);
        let effect = (m_f - m_u) / m_u * 100.0;
        let pairs: Vec<(f64, f64)> = unfused.iter().cloned().zip(fused.iter().cloned()).collect();
        let (lo, hi) = bootstrap_ci(&pairs, 2000);

        let m_na = median(&null_a);
        let m_nb = median(&null_b);
        let null_effect = (m_nb - m_na) / m_na * 100.0;
        let null_pairs: Vec<(f64, f64)> =
            null_a.iter().cloned().zip(null_b.iter().cloned()).collect();
        let (nlo, nhi) = bootstrap_ci(&null_pairs, 2000);

        // ENGAGEMENT, per §2.4 -- proved from a quantity the change should
        // have moved, not from the flag. Printed rather than asserted so a
        // failure is visible in the artifact rather than only in an exit code,
        // and asserted too so it cannot be read past.
        // The fused output must differ from the RAW accumulator: that is what
        // says the epilogue ran.
        let transformed = raw_u
            .iter()
            .zip(out_f.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        // ...and must AGREE with the unfused arm's final output to a tolerance,
        // which is what says it ran correctly. Not bit-equality: the fused arm
        // transforms the f32 accumulator where the pair transforms the stored
        // value, so the two legitimately differ in the low bits (see
        // `EpilogueSilu`'s note in gemv.metal).
        let worst_disagreement = out_u
            .iter()
            .zip(out_f.iter())
            .map(|(a, b)| ((a - b).abs() as f64) / (a.abs().max(1.0) as f64))
            .fold(0.0f64, f64::max);
        let distinct_u = out_u
            .iter()
            .map(|v| v.to_bits())
            .collect::<std::collections::HashSet<_>>()
            .len();
        // A THIRD arm, and it is the one the transform's own shape needs: SiLU
        // approaches the identity for large positive x, so on an all-positive
        // fixture a kernel with no epilogue at all would pass the first check
        // only marginally. Counting negatives says the fixture reaches the
        // region where SiLU is furthest from the identity.
        let negatives = raw_u.iter().filter(|v| **v < 0.0).count();

        let fmt_labels = |ls: &[(String, u64)]| -> String {
            ls.iter()
                .map(|(k, v)| format!("{k}x{v}"))
                .collect::<Vec<_>>()
                .join(" ")
        };

        println!(
            "## N_out = {n_out}   (intermediate = {} B at {})",
            n_out
                * match dtype {
                    GemmDType::F32 => 4,
                    _ => 2,
                },
            args.dtype
        );
        // The kernel census per arm. This is the ENGAGEMENT evidence (§2.4)
        // and it is also what makes the tile visible: the tile is in the
        // `[[host_name]]`, so a reader can see `bm4` against `bm8` here
        // without trusting the flag -- which is how #488's own artifact
        // records the boundary this issue found in it.
        for (arm, ls) in [
            (Arm::Unfused, &labels_u),
            (Arm::Fused, &labels_f),
            (Arm::GemvOnly, &labels_o),
            (Arm::GemvPlusTiny, &labels_t),
            (Arm::GemvPlusNeg, &labels_n),
            (Arm::GemvPlusNegIndep, &labels_i),
        ] {
            println!("   kernels, {:<9}: {}", arm.label(), fmt_labels(ls));
        }
        println!(
            "   ENGAGEMENT:      {transformed} of {} outputs differ from the raw \
             accumulator; {negatives} accumulators are negative; unfused has \
             {distinct_u} distinct values",
            raw_u.len()
        );
        println!(
            "   CROSS-ARM:       worst relative disagreement {worst_disagreement:.3e} \
             (expected small and non-zero: different rounding order)"
        );
        println!("   unfused  median: {m_u:.6} ms");
        println!("   fused    median: {m_f:.6} ms");

        // The decomposition. Reported in MICROSECONDS rather than as
        // percentages, deliberately: a percentage divides by an arm whose own
        // size changes 44x across this sweep, and #488's curve is
        // non-monotone in percent while its absolute savings are flat below
        // N=32768. The terms only add up in absolute time, and the
        // reconciliation below is the check that they do.
        let m_o = median(&gemv_only);
        let m_t = median(&gemv_tiny);
        let m_g = median(&gemv_neg);
        let us = |ms: f64| ms * 1e3;
        let t_dispatch = us(m_t - m_o);
        let t_traffic = us(m_g - m_t);
        let t_transcendental = us(m_u - m_g);
        let t_epilogue = us(m_f - m_o);
        let total = us(m_u - m_f);
        let reconstructed = t_dispatch + t_traffic + t_transcendental - t_epilogue;
        // The intermediate's round trip at the achievable bandwidth (§3.4's
        // 349.7 GB/s at a 5 GiB working set). This is what §12.1's traffic
        // account PREDICTS the fusion collects, and printing it beside the
        // measured term is the whole point of the issue.
        let elem_bytes = match dtype {
            GemmDType::F32 => 4.0f64,
            _ => 2.0,
        };
        let predicted_traffic_us = 2.0 * n_out as f64 * elem_bytes / 349.7e9 * 1e6;

        let m_i = median(&gemv_indep);
        // The barrier: the same two kernels over the same lengths, differing
        // only in whether the second reads what the first wrote. A positive
        // value is time the dependent arm spends that the independent one does
        // not -- lost overlap (§3.5), not launch cost.
        let t_barrier = us(m_g - m_i);
        println!("   gemv-only median: {m_o:.6} ms   gemv+tiny: {m_t:.6} ms   gemv+neg: {m_g:.6} ms   gemv+neg-indep: {m_i:.6} ms");
        println!("      BARRIER       (gemv+neg  - indep    ): {t_barrier:+9.3} us   [same kernels, same bytes, one dependency]");
        println!("   DECOMPOSITION (us, absolute; the two-arm effect is their sum):");
        println!("      dispatch      (gemv+tiny - gemv-only): {t_dispatch:+9.3} us");
        println!("      traffic       (gemv+neg  - gemv+tiny): {t_traffic:+9.3} us   [12.1 predicts {predicted_traffic_us:+9.3}]");
        println!("      transcendental(unfused   - gemv+neg ): {t_transcendental:+9.3} us");
        println!("      epilogue cost (fused     - gemv-only): {t_epilogue:+9.3} us   [12.3a: +5 regs, split wait]");
        println!(
            "      RECONCILIATION: sum {reconstructed:+9.3} us against measured {total:+9.3} us   \
             residual {:+9.3} us",
            reconstructed - total
        );
        println!("   EFFECT (fused vs unfused): {effect:+.3} % [{lo:+.3}, {hi:+.3}]");
        println!("   NULL   (unfused vs unfused): {null_effect:+.3} % [{nlo:+.3}, {nhi:+.3}]");
        let resolvable = effect.abs() > null_effect.abs() && (lo > 0.0 || hi < 0.0);
        println!(
            "   verdict: {}",
            if resolvable {
                if effect < 0.0 {
                    "FUSED WINS"
                } else {
                    "FUSED LOSES"
                }
            } else {
                "unresolved at this noise level"
            }
        );
        println!();

        assert!(
            transformed > 0,
            "N={n_out}: the fused output is bit-identical to the raw gemv \
             accumulator; the epilogue is not engaged and every figure above \
             is vacuous (DESIGN.md 15.1a)"
        );
        assert!(
            worst_disagreement < 1e-4,
            "N={n_out}: the arms disagree by {worst_disagreement:.3e}; the \
             fused arm is not computing what the pair computes and a timing \
             comparison between them is meaningless"
        );
        assert!(
            negatives > 0,
            "N={n_out}: no negative accumulators; SiLU is near-indistinguishable \
             from the identity on this fixture"
        );
        assert!(
            distinct_u >= 2,
            "N={n_out}: the unfused arm wrote {distinct_u} distinct value(s); \
             the comparison is vacuous"
        );

        // ENGAGEMENT for the three decomposition arms, from the kernel census
        // rather than from the code that asked for them (§2.4, and #229's
        // `config_line` case where a flag was believed and the census was
        // right). Each arm's dispatch COUNT is what distinguishes it, so the
        // count is what is asserted.
        let dispatch_total = |ls: &[(String, u64)]| -> u64 { ls.iter().map(|(_, v)| *v).sum() };
        let per_rep = |ls: &[(String, u64)]| -> u64 { dispatch_total(ls) / args.iters as u64 };
        assert_eq!(
            per_rep(&labels_o),
            1,
            "N={n_out}: the gemv-only arm dispatched {} kernels per repetition, \
             not 1; it is not the floor the decomposition subtracts and every \
             term below is wrong (DESIGN.md 15.1a)",
            per_rep(&labels_o)
        );
        assert_eq!(
            per_rep(&labels_t),
            2,
            "N={n_out}: the gemv+tiny arm dispatched {} kernels per repetition, \
             not 2; the dispatch term is not a dispatch",
            per_rep(&labels_t)
        );
        assert_eq!(
            per_rep(&labels_n),
            2,
            "N={n_out}: the gemv+neg arm dispatched {} kernels per repetition, \
             not 2; the traffic term is not isolated to a length change",
            per_rep(&labels_n)
        );
        // `gemv+tiny` and `gemv+neg` must differ ONLY in the second dispatch's
        // length, so they must name the SAME two kernels. If they do not, the
        // step between them is a kernel change and not a traffic change, which
        // is precisely the confound this issue exists to remove from #488's
        // sweep (where N and the TILE moved together across N=4096).
        let names = |ls: &[(String, u64)]| -> Vec<String> {
            let mut v: Vec<String> = ls.iter().map(|(k, _)| k.clone()).collect();
            v.sort();
            v
        };
        assert_eq!(
            names(&labels_t),
            names(&labels_n),
            "N={n_out}: the tiny and full-length arms ran DIFFERENT kernels, so \
             their difference is not the traffic term"
        );
        // The reconciliation is exact by construction -- the four terms are
        // differences of five medians that telescope -- so a non-zero residual
        // means the arithmetic above was edited into something that no longer
        // telescopes, not that a mechanism is missing. Asserted so that edit
        // cannot ship silently.
        assert!(
            (reconstructed - total).abs() < 1e-6,
            "N={n_out}: the decomposition does not telescope (residual \
             {:.6} us); the four terms are differences of five medians and must \
             sum to the two-arm effect by construction",
            reconstructed - total
        );
    }

    Ok(())
}
