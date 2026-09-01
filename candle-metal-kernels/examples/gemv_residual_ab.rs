//! Does a fused RESIDUAL-ADD epilogue on a GEMV beat the two-dispatch pair?
//!
//! Measurement harness for lloom issue #509 / `DESIGN.md` §12.2e's own
//! recommended candidate — the one that catalogue has no row for.
//!
//! # What this measures, and why it is not #484's question
//!
//! §12.2 #7 covers the residual add's **consumer** end and fails §12.2e's G1
//! arity gate: the add's two operands are the layer output and a residual the
//! *consuming* kernel does not hold. Nothing covers its **producer** end,
//! which is an epilogue on `w2` / `out_proj` / `conv_out` — the winning side
//! of §12.2d's partition rule.
//!
//! **The arity argument is the whole reason this is admissible.** G1 requires
//! the functor see ONE value. A residual add has two operands, so it fails as
//! stated — but the second operand is read at `out_row + tm`, the **storing
//! thread's own output position**, so each thread reads exactly the one
//! element it is about to write. `kDoAxpby` is already that shape
//! (`gemv.metal`: `bias[(out_row + tm) * bias_stride]`), already instantiated,
//! and it is the structural difference from §12.2b's `gate * up`, whose second
//! operand is a full vector of the INPUT extent and is therefore multiplied by
//! `n_tgp`: **a residual is indexable from the output position and a gate is
//! not.**
//!
//! # Why a microbenchmark, and what that costs — stated rather than assumed
//!
//! §13.1a: `CANDLE_METAL_PROFILE` records dispatch COUNTS per kernel and GPU
//! INTERVALS per command buffer, and the two series can never be joined —
//! dispatches inside one encoder overlap, so the command buffer is the
//! smallest unit Metal reports that does not double-count. Per-kernel GPU time
//! is INEXPRESSIVE at dispatch granularity, and the one route that would give
//! it (`CANDLE_METAL_COMPUTE_PER_BUFFER=1`) is measured at **+9.75 % GPU busy,
//! 67× the null** — it changes the thing it measures.
//!
//! **So this is a per-kernel figure and §12.2c is the standing warning that a
//! per-kernel figure is not a per-token one.** #509's brief asks for a
//! decode-path measurement; that is a separate arm needing `lfm2.rs` wiring,
//! and this harness does not pretend to be it.
//!
//! # The arms
//!
//! * **unfused** — `gemv` writes the accumulator, then `badd` adds the
//!   residual. Two dispatches; the second reads what the first wrote, so
//!   candle emits a **barrier** between them.
//! * **fused** — `gemv_residual` adds the residual in registers at the store.
//!   One dispatch, no barrier, no intermediate round trip.
//! * **gemv-only** — the GEMV alone. The floor both are built on.
//! * **gemv+badd-indep** — `gemv`, then a `badd` over the same lengths moving
//!   the same bytes but reading a buffer the GEMV **never wrote**. No hazard,
//!   no barrier. Against **unfused** this isolates the barrier, and it is one
//!   bit wide.
//!
//! # The term this is priced against, and it is NOT the one #509 was filed on
//!
//! #509's body prices this candidate at *"§12.2f's measured barrier at f16 /
//! bm4 / N = 2048 — **20.51 µs**"*. **That figure is the DISPATCH term**
//! (`gemv+tiny − gemv-only`), not the barrier term. #503's own artifact prints
//! both four lines apart, and the barrier at that cell is **9.037 µs**. See
//! `measurements/issue-509-raw/reprice.py`, which re-derives it and exits
//! non-zero if it stops reproducing.
//!
//! So the `gemv+badd-indep` arm here is not a re-take of a known number — it
//! is the first measurement of the barrier at the seam this fusion actually
//! removes, with a `badd` consumer rather than #503's `neg`.
//!
//! # Method, and every clause is a rule from `CONTRIBUTING.md`
//!
//! * **Interleaved A/B** (§3.2), null control **inside every rotation** (§1.3),
//!   **median** not mean (§3.2a #3), a **deterministic bootstrap CI** and no
//!   p-values (§3.2) — all as #484's harness, which this deliberately mirrors
//!   rather than reimplements.
//! * **The tile is PINNED** by `LLOOM_492_GEMV_BM`. All three producers of a
//!   residual add sit at `N_out` = 2048 → `bm4`, and #492 measures the tile
//!   worth **up to 28.0 pp** at one N, so a sweep that does not pin it is
//!   measuring two kernels and calling it one curve.
//!
//! # Engagement
//!
//! §2.4: an instrument that cannot be shown to have engaged has measured
//! nothing. Three checks, all printed, none optional:
//!
//! 1. **The arms disagree with the raw accumulator.** The fused output must
//!    differ from the GEMV's untransformed result — or the epilogue did not
//!    run — while agreeing with the unfused arm's final output.
//! 2. **The kernel census names what ran**, so `_residual` appearing is read
//!    off the dispatch labels rather than inferred from the flag.
//! Check 3 used to be **"the metallib must carry the arm at all"** — without
//! `-DLLOOM_509_RESIDUAL_EPILOGUE` the `_residual` name did not exist and this
//! failed at pipeline load. #516 instantiates the arm unconditionally, so that
//! check has no failing input any more and the `-D` is gone from the command
//! below. The remaining two checks are unchanged and are the ones that can
//! still fail.
//!
//! ```bash
//! CANDLE_METAL_PROFILE=1 LLOOM_492_GEMV_BM=4 \
//!   lloom-arb measure -- gemv_residual_ab --n 2048
//! ```

use anyhow::{Context, Result};
use candle_metal_kernels::metal::{Commands, Device, ResidencySet};
use candle_metal_kernels::{
    call_binary_contiguous, call_mlx_gemv_epilogue, call_mlx_gemv_with, BufferOffset, GemmDType,
    GemvEpilogue, Kernels, ParamStyle,
};
use clap::Parser;
use objc2_metal::MTLResourceOptions;

#[derive(Parser, Debug)]
#[command(about = "gemv + residual-add epilogue against the two-dispatch pair (#509)")]
struct Args {
    /// K, the input vector length.
    ///
    /// **2048 is `out_proj`'s and `conv.out_proj`'s** (`DESIGN.md` §5.2's
    /// hidden_size). `w2`, the third producer, is K=10752 at the same N_out —
    /// so `--k 10752` is the other decode site and is swept separately rather
    /// than averaged in.
    #[arg(long, default_value_t = 2048)]
    k: usize,

    /// N values to sweep, each an output vector length.
    ///
    /// **2048 is where all three producers sit**, so the default is a single
    /// point rather than a curve: a curve is what would say whether the win is
    /// the barrier or the bytes, and it is #509's second question rather than
    /// its first. The wider default brackets it either side.
    #[arg(long, value_delimiter = ',', default_values_t = [512usize, 2048, 10752])]
    n: Vec<usize>,

    /// Interleaved rotations. Each runs five slots: unfused, fused, gemv-only,
    /// the barrier arm, and a null slot last.
    #[arg(long, default_value_t = 12)]
    rotations: usize,

    /// Timed repetitions inside one slot; the median is that slot's figure.
    #[arg(long, default_value_t = 50)]
    iters: usize,

    /// Untimed repetitions before each slot is timed.
    #[arg(long, default_value_t = 10)]
    warmup: usize,

    /// Element type. `f16` is what LFM2 decodes in (§5.1, §2.3.3b).
    #[arg(long, default_value = "f16")]
    dtype: String,
}

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
/// Same generator and same reps as #484's, so the two issues' intervals are
/// computed the same way and are comparable.
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    /// `gemv` then `badd`: two dispatches, a barrier, one intermediate round
    /// trip. This is what LFM2 dispatches today, 60 times per token.
    Unfused,
    /// `gemv_residual`: one dispatch, the add applied in registers at the
    /// store.
    Fused,
    /// `gemv` alone — the floor. Without it, *"what the epilogue costs"* and
    /// *"what the second dispatch costs"* are one number.
    GemvOnly,
    /// `gemv` into `acc`, then `badd` over N elements reading two buffers the
    /// GEMV **never wrote**.
    ///
    /// The barrier arm, one bit wide: same two kernels, same lengths, same
    /// bytes moved, only the source differing, so one raises a RAW hazard and
    /// one does not (§3.5 — serialization comes from barriers, not from the
    /// GPU). Against `Unfused` this is the barrier and nothing else.
    GemvPlusBaddIndep,
}

impl Arm {
    /// The name used in the artifact. Written **once** so a column heading and
    /// the census line beside it cannot drift apart -- the failure §7.1a-i
    /// records, where one axis rendered by two separate expressions
    /// mislabelled every `RESULT` line since #116.
    fn label(self) -> &'static str {
        match self {
            Arm::Unfused => "unfused",
            Arm::Fused => "fused",
            Arm::GemvOnly => "gemv-only",
            Arm::GemvPlusBaddIndep => "gemv+badd-indep",
        }
    }
}

struct Slot {
    /// GPU-busy UNION over the slot's timed repetitions, ms per repetition.
    /// The union rather than the sum, per `profile.rs`: dispatches inside one
    /// encoder overlap, so a sum over-counts.
    gpu_ms: f64,
    by_label: Vec<(String, u64)>,
    out: Vec<f32>,
    /// The last repetition's UNTRANSFORMED gemv accumulator, on the unfused
    /// arm only. This is the engagement evidence: the fused output must differ
    /// from THIS (or the epilogue is not applied) while agreeing with the
    /// unfused final output (or it is applied wrongly). Comparing the two
    /// arms' FINAL outputs proves nothing — they are meant to agree.
    raw: Vec<f32>,
}

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

    let elem_size = match dtype {
        GemmDType::F32 => 4usize,
        GemmDType::F16 | GemmDType::BF16 => 2,
    };
    // `badd` is the kernel LFM2 actually dispatches at this seam -- 60
    // `badd_f16` per token (`measurements/issue-249-raw/kernel-census.txt`).
    // Using the real one rather than a stand-in is what makes the unfused arm
    // the shipping path rather than a model of it, and the name is spelled the
    // way the census prints it so the two can be read against each other.
    let badd_kernel = match dtype {
        GemmDType::F32 => "badd_f32",
        GemmDType::F16 => "badd_f16",
        GemmDType::BF16 => "badd_bf16",
    };

    // Generated in f32 and narrowed, so both dtype arms see the same NUMBERS
    // and a comparison between them is about precision rather than about a
    // different fixture.
    let mat: Vec<f32> = (0..n_out * k)
        .map(|i| ((i % 17) as f32 - 8.0) / 32.0)
        .collect();
    let vec: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 16.0).collect();
    // The residual stream. Two properties, both load-bearing and both learned
    // by getting them wrong first.
    //
    // **Non-constant**, because a residual of all ones would make an epilogue
    // that ignored its index produce the right answer -- the failure mode
    // #266's `norm_w` note records a fixture being blind to. This varies with
    // position, so a wrong index gives a wrong number.
    //
    // **And never ZERO.** The first version was `(i % 23 - 11) / 8`, which is
    // zero at 89 of 2048 positions, and the engagement check -- *how many
    // fused outputs differ from the raw accumulator* -- read **1959 of 2048**.
    // That is exactly `2048 - 89`: adding zero changes nothing, so those 89
    // positions are indistinguishable from *the epilogue did not run*. The
    // check was right and 89 of its samples were vacuous (§15.1a's class, in
    // the arm written to detect it). Offsetting by a half step removes every
    // zero while keeping the position dependence.
    let res: Vec<f32> = (0..n_out)
        .map(|i| ((i % 23) as f32 - 11.0) / 8.0 + 0.0625)
        .collect();

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
    let res_buf = mk(&res)?;
    let acc = device
        .new_buffer(n_out * elem_size, options)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let out = device
        .new_buffer(n_out * elem_size, options)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    // The independent arm's sources and destination. Neither source is ever
    // written by the GEMV, so a `badd` reading them raises no hazard and
    // candle emits no barrier -- the single bit this arm varies. Allocated
    // unconditionally rather than per arm so the allocation is not itself a
    // difference between arms (§9.2c's blindness rule).
    let spare_a = mk(&res)?;
    let spare_b = mk(&res)?;
    let spare_dst = device
        .new_buffer(n_out * elem_size, options)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let mut last = Vec::new();
    let mut raw = Vec::new();

    for rep in 0..(warmup + iters) {
        // The profiling window opens AFTER warmup, so §2.4's 35x-130x
        // first-touch cost is outside every reported figure.
        if rep == warmup {
            candle_metal_kernels::metal::profile::reset();
        }
        let queue = device
            .new_command_queue()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let residency = ResidencySet::new(device);
        let commands = Commands::new(queue, &residency).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let encoder = commands
            .command_encoder()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        // The GEMV half, identical in every arm except `Fused` -- so the four
        // arms differ only in what follows it.
        let gemv_dest = if arm == Arm::Fused { &out } else { &acc };
        if arm == Arm::Fused {
            call_mlx_gemv_epilogue(
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
                gemv_dest,
                ParamStyle::Split,
                None,
                GemvEpilogue::Residual,
                Some((&res_buf, 0)),
            )
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        } else {
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
                gemv_dest,
                ParamStyle::Split,
            )
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        }

        // (lhs, rhs, dst) for the second dispatch. `GemvPlusBaddIndep` differs
        // from `Unfused` ONLY in the buffers, which is what makes their
        // difference the barrier.
        let second: Option<(&_, &_, &_)> = match arm {
            Arm::Fused | Arm::GemvOnly => None,
            Arm::Unfused => Some((&acc, &res_buf, &out)),
            Arm::GemvPlusBaddIndep => Some((&spare_a, &spare_b, &spare_dst)),
        };
        if let Some((lhs, rhs, dst)) = second {
            call_binary_contiguous(
                device,
                &encoder,
                kernels,
                badd_kernel,
                elem_size,
                n_out,
                BufferOffset {
                    buffer: lhs,
                    offset_in_bytes: 0,
                },
                BufferOffset {
                    buffer: rhs,
                    offset_in_bytes: 0,
                },
                dst,
            )
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        }

        drop(encoder);
        commands
            .wait_until_completed()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        if rep == warmup + iters - 1 {
            last = read_as_f32(&out, n_out, dtype);
            if arm == Arm::Unfused {
                raw = read_as_f32(&acc, n_out, dtype);
            }
        }
    }

    let snap = candle_metal_kernels::metal::profile::snapshot();
    // `timed_command_buffers` below `iters` means a completion handler had not
    // run when the window was read, which would silently shorten the union.
    // Refused rather than tolerated: §15.1a's VACUOUS class is a figure taken
    // from an instrument that did not fully report.
    //
    // NOTE this requires `CANDLE_METAL_PROFILE=1`; without it the count is 0
    // and this fires, which is the intended failure rather than a wrong number.
    anyhow::ensure!(
        snap.timed_command_buffers >= iters as u64,
        "only {} of {iters} command buffers reported an interval; the GPU-busy \
         union is short and the figure would be wrong low. Is \
         CANDLE_METAL_PROFILE=1 set?",
        snap.timed_command_buffers
    );

    Ok(Slot {
        gpu_ms: snap.gpu_busy_union_s * 1e3 / iters as f64,
        by_label: snap.by_label,
        out: last,
        raw,
    })
}

fn read_as_f32(
    buffer: &candle_metal_kernels::metal::Buffer,
    len: usize,
    dtype: GemmDType,
) -> Vec<f32> {
    let ptr = buffer.contents();
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

fn kernels_line(tag: &str, labels: &[(String, u64)]) -> String {
    let mut s = format!("   kernels, {tag:<16}: ");
    for (name, n) in labels {
        s.push_str(&format!("{name}x{n} "));
    }
    s
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = Device::system_default().context("no Metal device")?;
    let kernels = Kernels::new();

    // An unknown dtype PANICS rather than falling back: §2.4 -- an arm that
    // silently ran the baseline is #69's vacuous determinism run.
    let dtype = match args.dtype.as_str() {
        "f32" => GemmDType::F32,
        "f16" => GemmDType::F16,
        "bf16" => GemmDType::BF16,
        other => anyhow::bail!("--dtype must be one of f32|f16|bf16, got {other:?}"),
    };

    println!("# issue #509 -- gemv + residual-add epilogue against the two-dispatch pair");
    println!("# DESIGN.md 12.2e's own recommended candidate; the catalogue has no row for it");
    println!(
        "# K={} rotations={} iters={} warmup={}",
        args.k, args.rotations, args.iters, args.warmup
    );
    println!(
        "# dtype={} (f16 is what LFM2 decodes in; 5.1, 2.3.3b)",
        args.dtype
    );
    println!(
        "# LLOOM_492_GEMV_BM={} (unset = derived from N: bm8 at N>=4096, bm4 below)",
        std::env::var("LLOOM_492_GEMV_BM").unwrap_or_else(|_| "unset".to_string())
    );
    println!(
        // Still recorded, still not required: since #516 the fused arm is
        // instantiated unconditionally, so this reads `unset` on a normal run.
        // Kept because the string decides how every kernel here was compiled
        // (§14.5a), which a reader comparing two runs needs either way.
        "# LLOOM_434_METAL_CFLAGS={} (compile flags this metallib was built with)",
        std::env::var("LLOOM_434_METAL_CFLAGS").unwrap_or_else(|_| "unset".to_string())
    );
    println!("#");
    println!("# The term #509 was filed on -- '20.51 us at f16/bm4/N=2048' -- is the");
    println!("# DISPATCH term, not the barrier. #503's barrier at that cell is 9.037 us.");
    println!("# See measurements/issue-509-raw/reprice.py.");
    println!();

    for &n_out in &args.n {
        let mut unfused = Vec::new();
        let mut fused = Vec::new();
        let mut gemv_only = Vec::new();
        let mut indep = Vec::new();
        let mut null_b = Vec::new();
        let (mut lu, mut lf, mut lo, mut li) = (vec![], vec![], vec![], vec![]);
        let mut out_u = Vec::new();
        let mut raw_u = Vec::new();
        let mut out_f = Vec::new();

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
            let i = run_slot(
                &device,
                &kernels,
                Arm::GemvPlusBaddIndep,
                dtype,
                args.k,
                n_out,
                args.iters,
                args.warmup,
            )?;
            // The null runs LAST so the unfused arm occupies the first and
            // last slots: the null therefore brackets every other arm, and a
            // monotone per-slot drift (§3.4a-i #3) shows up in it rather than
            // hiding inside an effect.
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
            indep.push(i.gpu_ms);
            null_b.push(u2.gpu_ms);
            lu = u.by_label;
            lf = f.by_label;
            lo = o.by_label;
            li = i.by_label;
            out_u = u.out;
            raw_u = u.raw;
            out_f = f.out;
        }
        let null_a = unfused.clone();

        let m_u = median(&unfused);
        let m_f = median(&fused);
        let m_o = median(&gemv_only);
        let m_i = median(&indep);
        let effect = (m_f - m_u) / m_u * 100.0;
        let pairs: Vec<(f64, f64)> = unfused.iter().cloned().zip(fused.iter().cloned()).collect();
        let (lo_ci, hi_ci) = bootstrap_ci(&pairs, 2000);

        let m_na = median(&null_a);
        let m_nb = median(&null_b);
        let null_effect = (m_nb - m_na) / m_na * 100.0;
        let null_pairs: Vec<(f64, f64)> =
            null_a.iter().cloned().zip(null_b.iter().cloned()).collect();
        let (nlo, nhi) = bootstrap_ci(&null_pairs, 2000);

        let inter = n_out * if dtype == GemmDType::F32 { 4 } else { 2 };
        println!("## N_out = {n_out}   (intermediate = {inter} B)");
        // The labels come from `Arm::label` rather than from four literals, so
        // a renamed arm renames its census line too.
        println!("{}", kernels_line(Arm::Unfused.label(), &lu));
        println!("{}", kernels_line(Arm::Fused.label(), &lf));
        println!("{}", kernels_line(Arm::GemvOnly.label(), &lo));
        println!("{}", kernels_line(Arm::GemvPlusBaddIndep.label(), &li));

        // ------------------------------------------------------- ENGAGEMENT
        // The fused output must differ from the RAW accumulator (or the
        // epilogue did not run) and agree with the unfused final output (or it
        // ran wrongly). Both directions, because either alone passes on a
        // defect: comparing the two FINAL outputs proves nothing -- they are
        // meant to agree.
        let differ = out_f
            .iter()
            .zip(raw_u.iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-6)
            .count();
        let worst = out_f
            .iter()
            .zip(out_u.iter())
            .map(|(a, b)| (a - b).abs() / b.abs().max(1e-3))
            .fold(0.0f32, f32::max);
        println!(
            "   ENGAGEMENT:      {differ} of {} fused outputs differ from the raw accumulator",
            out_f.len()
        );
        // **This reads 0.000e0 on this fixture and that is CORRECT, not a
        // defect** -- recorded because the opposite was predicted and the
        // prediction was wrong. The fused arm adds at the f32 accumulator and
        // `badd` adds at `T` on the rounded store, so the two ARE different
        // computations; but this fixture's values are small dyadic rationals
        // (`mat` in 32nds, `vec` in 16ths, `res` in 16ths) that f16 represents
        // EXACTLY over the range the accumulator reaches, so both rounding
        // orders land on the same number.
        //
        // The consequence is a scope limit rather than a reassurance: **this
        // fixture cannot demonstrate that the two arms differ**, so it is not
        // evidence about what wiring this into `lfm2.rs` would do to the
        // digests. On real weights the two orders will differ and the arm owes
        // §2.3.5a's discriminator. Saying so here rather than letting a
        // convenient zero read as *the fusion is bit-identical* (§2.4: a
        // result that agrees with the design document is not thereby
        // interrogated less).
        println!(
            "   CROSS-ARM:       worst relative disagreement {worst:.3e} \
             (0 is EXPECTED here: this fixture is exactly representable at \
             f16, so both rounding orders agree -- see the source note)"
        );
        if differ == 0 {
            println!("   !! the epilogue did not run: every output equals the raw accumulator");
        }

        println!("   unfused  median: {m_u:.6} ms");
        println!("   fused    median: {m_f:.6} ms");
        println!("   gemv-only median: {m_o:.6} ms   gemv+badd-indep: {m_i:.6} ms");
        println!(
            "      BARRIER       (unfused   - indep    ): {:+9.3} us   \
             [same kernels, same bytes, one dependency]",
            (m_u - m_i) * 1000.0
        );
        println!(
            "      second dispatch, total (unfused - gemv-only): {:+9.3} us",
            (m_u - m_o) * 1000.0
        );
        println!(
            "      epilogue cost (fused     - gemv-only): {:+9.3} us   \
             [12.3a: a fusion costs registers and a split wait]",
            (m_f - m_o) * 1000.0
        );
        println!("   EFFECT (fused vs unfused): {effect:.3} % [{lo_ci:.3}, {hi_ci:.3}]");
        println!("   NULL   (unfused vs unfused): {null_effect:.3} % [{nlo:.3}, {nhi:.3}]");
        // The verdict is MECHANICAL: an effect whose interval overlaps the
        // null's is unresolved, and saying so is the outcome §1.3 names as
        // acceptable rather than as a failed measurement.
        let verdict = if hi_ci < 0.0 && effect.abs() > null_effect.abs() {
            "FUSED WINS"
        } else if lo_ci > 0.0 {
            "FUSED LOSES"
        } else {
            "UNRESOLVED at this noise level"
        };
        println!("   verdict: {verdict}");
        println!();
    }

    Ok(())
}
