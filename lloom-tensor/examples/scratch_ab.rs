//! A/B harness for the scratch class's sizing policies (`DESIGN.md` §9.1, #71).
//!
//! # What this is for
//!
//! Three sizing policies are compiled and **none is chosen**, because the regime
//! that would choose one is long context and the largest `kv_len` this project
//! has ever recorded is 2720 (§13.2; `.bench/` §3 enumerates the gap). This
//! harness is the mechanism that would *generate* the evidence, which the issue
//! names as the deliverable rather than the choice.
//!
//! **The sizing half of that is now stale and the layout half is not (#526).**
//! #234 wired `ScratchSizing` to FlashDecoding's own allocation and #274
//! (§9.1f) swept all three arms across the curve *through the real kernel*, so
//! the sizing evidence exists and did not come from here. The arms still stay
//! unchosen -- they separate at `kv_len` 4034 and at neither neighbour.
//!
//! **What this harness is still the only instrument for is `--layout`.**
//! `flash_decoding.metal` is hardcoded to the planes layout and takes no layout
//! parameter, so §9.1d's **+45.5 % per record** for the interleaved arm was
//! measured here and can be re-derived only here.
//!
//! It reports two things at every `kv_len`, and the separation is the point:
//!
//! - **footprint and waste**, which are computed and exact -- pure arithmetic
//!   over the plan, no device, no timing, no noise;
//! - **whether every policy computes the same bits**, which is executed.
//!
//! The second is the acceptance bar, not the first. The issue is explicit: *a
//! policy that changes numerics is a bug, not a tradeoff.* A footprint table is
//! only interesting once the arms are known to be comparable.
//!
//! # What it does not report, and why that is deliberate
//!
//! **No timing.** Nothing on the LFM2 path dispatches these kernels -- the real
//! FlashDecoding kernel is Phase 4/5 (§17 items 14, 16) -- so a wall-clock
//! comparison here would be measuring a stub. §6.6a's rule is that a measurement
//! which cannot resolve its effect is not evidence in either direction, and
//! §11.3k applied it by running no A/B at all rather than a meaningless one.
//!
//! **Every row above `kv_len` 2720 is marked `unmeasured`**, per #60's
//! convention: the sizes are computed from the geometry and the *behaviour* at
//! those lengths has never been observed. Computed is not measured, and the
//! table says which is which in its own column rather than in a footnote.
//!
//! ```bash
//! cargo run --release --example scratch_ab
//! cargo run --release --example scratch_ab -- --kv-len 32768 --layout interleaved
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use lloom_tensor::kernels::metal::{Buffer, Commands, Device, ResidencySet};
use lloom_tensor::kernels::{
    call_scratch_combine, call_scratch_partials, plan_scratch, CombineOrder, Kernels,
    PartialsGeometry, ScratchLayout, ScratchParams, Sizing,
};
use objc2_metal::MTLResourceOptions;
use std::collections::BTreeSet;
use std::sync::Arc;

/// The largest `kv_len` ever recorded on this project, on a determinism run
/// rather than a timed one (`measurements/issue-5-determinism.txt:53`).
///
/// Every row above it is `unmeasured`, and the table says so per row.
const LARGEST_MEASURED_KV: usize = 2_720;

const SHARED: MTLResourceOptions = MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

#[derive(Parser, Debug)]
#[command(about = "Scratch-class sizing policy A/B (lloom issue #71)")]
struct Args {
    /// `kv_len` values to report, comma separated.
    ///
    /// The default is §9.1's table plus two short lengths, so the short-context
    /// regime this project has actually entered sits beside the long-context one
    /// it has not.
    #[arg(long, default_value = "256,2720,8192,32768,131072")]
    kv_lens: String,

    /// A single `kv_len` to run the execution comparison at.
    ///
    /// Short by default: the execution arms allocate their regions, and
    /// `Reserve` at 128k reserves for 128k whatever this is set to.
    #[arg(long, default_value_t = 2_720)]
    kv_len: usize,

    /// `planes` or `interleaved`. See `ScratchLayout` -- only the second can
    /// expose an alignment defect on LFM2's shapes.
    #[arg(long, default_value = "planes")]
    layout: String,

    /// Attention layers holding partials. 8 for LFM2 (§5.3), not 30.
    #[arg(long, default_value_t = 8)]
    layers: usize,

    /// Configured maximum context, bounding `Reserve`. §5.2's 128k.
    #[arg(long, default_value_t = 131_072)]
    max_context: usize,

    /// Skip the execution comparison and report the footprint table only.
    #[arg(long)]
    no_device: bool,
}

fn parse_layout(s: &str) -> Result<ScratchLayout> {
    match s {
        "planes" => Ok(ScratchLayout::Planes),
        "interleaved" => Ok(ScratchLayout::Interleaved),
        other => anyhow::bail!("unknown layout {other}; expected planes or interleaved"),
    }
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let layout = parse_layout(&args.layout)?;
    let geometry = PartialsGeometry::default();

    println!("== scratch class, sizing policy A/B (lloom #71) ==");
    println!(
        "geometry: n_heads {} (query heads, post-GQA), head_dim {}, page {}, B {}",
        geometry.n_heads, geometry.head_dim, geometry.page_size, geometry.batch
    );
    println!(
        "layers {}  layout {:?}  max_context {}",
        args.layers, layout, args.max_context
    );
    println!(
        "interleaved record: {} B unpadded (128-aligned: {})",
        geometry.interleaved_record_bytes(),
        geometry.interleaved_record_bytes() % 128 == 0
    );

    // ---- footprint, computed ----
    //
    // Exact arithmetic. The `status` column is the honest part: computed is not
    // measured, and #60's convention is that the unmeasured cells are the file's
    // most useful output rather than something to fill in for tidiness.
    println!("\n== footprint per policy (COMPUTED, not measured) ==");
    println!(
        "{:>8}  {:>6}  {:>10}  {:>12}  {:>12}  {:>12}  status",
        "kv_len", "chunks", "policy", "arena B", "bump cap B", "waste MiB"
    );

    let kv_lens: Vec<usize> = args
        .kv_lens
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<usize>())
        .collect::<std::result::Result<_, _>>()
        .context("parsing --kv-lens")?;

    for kv in kv_lens {
        let chunks = geometry.chunks(kv);
        let status = if kv <= LARGEST_MEASURED_KV {
            "reachable today"
        } else {
            "UNMEASURED - no run at this kv_len has ever been taken"
        };
        for sizing in Sizing::ALL {
            match plan_scratch(&geometry, kv, args.layers, sizing, layout, args.max_context) {
                Ok(plan) => println!(
                    "{:>8}  {:>6}  {:>10}  {:>12}  {:>12}  {:>12.3}  {}",
                    kv,
                    chunks,
                    format!("{sizing:?}"),
                    plan.arena_bytes(),
                    plan.bump_capacity(),
                    mib(plan.reserved_waste(&geometry)),
                    status
                ),
                Err(e) => println!(
                    "{:>8}  {:>6}  {:>10}  {:>12}  {:>12}  {:>12}  refused: {e}",
                    kv,
                    chunks,
                    format!("{sizing:?}"),
                    "-",
                    "-",
                    "-"
                ),
            }
        }
    }

    // ---- where align_up is load-bearing, and where it is not ----
    //
    // #70's warning, sharpened by running it. The blindness has **two levels**
    // and they are easy to conflate:
    //
    //   region level -- `bump_capacity` vs `arena_bytes`. Blind under BOTH
    //     layouts on LFM2's shapes, because the interleaved layout's padding is
    //     already *inside* the region size: 32 heads x 11 chunks x 384 B is a
    //     128-multiple even though the record it is built from is not.
    //   record level -- the (head, chunk) stride *within* a region. 264 B
    //     unpadded against 384 padded. This is where `align_up` decides
    //     anything on our own model's shapes.
    //
    // So "the interleaved layout can expose an alignment defect" is true of the
    // record and false of the capacity, and only the first is a fixture that can
    // fail. Deleting `align_up` is killed by the record-stride assertions and by
    // no capacity comparison -- which is #70's lesson one level finer than #70
    // stated it, and worth reporting rather than smoothing over.
    let probe = plan_scratch(
        &geometry,
        args.kv_len,
        args.layers,
        Sizing::Grow,
        layout,
        args.max_context,
    )
    .map_err(anyhow::Error::msg)?;
    let record = geometry.interleaved_record_bytes();
    let padded = record.div_ceil(128) * 128;
    println!(
        "\n== alignment: which level can see a bad align_up ==\n\
         region level ({:?}, kv_len {}): bump_capacity {} vs arena_bytes {} -> {}",
        layout,
        args.kv_len,
        probe.bump_capacity(),
        probe.arena_bytes(),
        if probe.bump_capacity() == probe.arena_bytes() {
            "EQUAL, so BLIND -- and it is blind under both layouts here, because the \
             interleaved padding is already inside the region size"
        } else {
            "DIFFER -- align_up is load-bearing at this level"
        }
    );
    println!(
        "record level: {record} B unpadded -> {padded} B padded, 128-multiple: {} \
         -> {}",
        record % 128 == 0,
        if record % 128 == 0 {
            "BLIND"
        } else {
            "THIS is the level where align_up decides something on LFM2's own shapes \
             (DESIGN.md 9.2c, #70)"
        }
    );

    // ---- the property that is not a tradeoff ----
    println!("\n== stability and rebinding ==");
    for sizing in Sizing::ALL {
        println!(
            "{:>10}: rebinds on growth = {} {}",
            format!("{sizing:?}"),
            sizing.rebinds_on_growth(),
            if sizing.rebinds_on_growth() {
                "-- a realloc changes buffer identity, which is what #69 exists to \
                 prevent (9.2c); needs a rebind path"
            } else {
                ""
            }
        );
    }

    if args.no_device {
        println!("\n(--no-device: execution comparison skipped)");
        return Ok(());
    }

    // ---- the acceptance bar, executed ----
    //
    // A policy that changes numerics is a bug, not a tradeoff. Two arms are
    // comparable only once this holds, so it is reported before anything that
    // could be read as a performance claim -- and there is no such claim here.
    println!("\n== every policy computes the same bits (EXECUTED) ==");
    let device = Device::system_default().context("no Metal device")?;
    let kernels = Kernels::new();

    let mut digests: Vec<(Sizing, u64, usize, Vec<u32>)> = Vec::new();
    for sizing in Sizing::ALL {
        let (bits, order) = run_one(
            &device,
            &kernels,
            sizing,
            &geometry,
            args.kv_len,
            layout,
            args.max_context,
        )?;
        let distinct: BTreeSet<u32> = bits.iter().copied().collect();
        let digest = fnv1a(&bits);
        println!(
            "{:>10}: digest {digest:016x}  {} distinct values over {} elements",
            format!("{sizing:?}"),
            distinct.len(),
            bits.len()
        );
        digests.push((sizing, digest, distinct.len(), order));
    }

    let first = digests[0].1;
    let agree = digests.iter().all(|(_, d, _, _)| *d == first);
    // A comparison over constant output is vacuous, and §3.7a names all-zero
    // output as the ICB path's characteristic failure. So the non-vacuity is
    // reported beside the agreement rather than assumed (#53, §11.3j).
    let non_vacuous = digests.iter().all(|(_, _, n, _)| *n > 4);
    println!(
        "\npolicies agree: {}   comparison non-vacuous: {}",
        if agree { "YES" } else { "NO -- THIS IS A BUG" },
        if non_vacuous { "YES" } else { "NO" }
    );

    // ---- the merge order, as walked ----
    println!("\n== combine merge order (EXECUTED) ==");
    let live = geometry.chunks(args.kv_len);
    for (sizing, _, _, order) in &digests {
        let head0: Vec<u32> = order.iter().take(live).copied().collect();
        let ascending = head0.windows(2).all(|w| w[0] < w[1]) && head0.first() == Some(&0);
        println!(
            "{:>10}: {} chunks, ascending = {}  first 8: {:?}",
            format!("{sizing:?}"),
            live,
            ascending,
            &head0[..head0.len().min(8)]
        );
        anyhow::ensure!(
            ascending,
            "{sizing:?} merged out of index order -- DESIGN.md 2.3.3 #1, 10.4"
        );
    }
    println!(
        "\ncombine order is {:?} for every policy, and it is asserted rather than \
         assumed: 10.4 calls a completion-ordered merge here the single most likely \
         place for nondeterminism to enter the design.",
        CombineOrder::Index
    );

    anyhow::ensure!(agree, "the policies do not compute the same bits");
    anyhow::ensure!(non_vacuous, "the comparison is vacuous");
    println!("\nOK");
    Ok(())
}

/// FNV-1a over the raw bits, so two runs are compared by *what they computed*.
///
/// Bits and not values: two f32 that print the same can differ, which is the
/// whole reason §2.3 digests logits rather than eyeballing text.
fn fnv1a(bits: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for w in bits {
        for b in w.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    h
}

/// Run one policy's partials and combine, returning `(output bits, order walked)`.
fn run_one(
    device: &Device,
    kernels: &Kernels,
    sizing: Sizing,
    geometry: &PartialsGeometry,
    kv_len: usize,
    layout: ScratchLayout,
    max_context: usize,
) -> Result<(Vec<u32>, Vec<u32>)> {
    let plan = plan_scratch(geometry, kv_len, 1, sizing, layout, max_context)
        .map_err(anyhow::Error::msg)?;
    // Asserted rather than assumed, at the point of use as well as in the plan.
    plan.check_index_ordered().map_err(anyhow::Error::msg)?;
    plan.check_disjoint().map_err(anyhow::Error::msg)?;

    let live = plan.live_chunks();
    let queue = device.new_command_queue().map_err(anyhow::Error::msg)?;
    let residency = Arc::new(ResidencySet::new(device));
    let cmds = Commands::new(queue, residency.clone()).map_err(anyhow::Error::msg)?;

    let partials = new_buffer(device, plan.bump_capacity().max(1))?;
    let out = new_buffer(device, geometry.n_heads * geometry.head_dim * 4)?;
    let order = new_buffer(device, geometry.n_heads * live.max(1) * 4)?;
    let params = new_buffer(device, std::mem::size_of::<ScratchParams>())?;

    let p = ScratchParams {
        n_heads: geometry.n_heads as u32,
        head_dim: geometry.head_dim as u32,
        live_chunks: live as u32,
        sized_chunks: plan.regions()[0].chunks as u32,
        interleaved: u32::from(layout == ScratchLayout::Interleaved),
        // Fixed across the arms: the arms must differ in the policy and in
        // nothing else, or the comparison is not one.
        seed: 0x5EED,
    };
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    let dst = unsafe { params.contents() } as *mut ScratchParams;
    anyhow::ensure!(!dst.is_null(), "params buffer has no CPU mapping");
    // SAFETY: shared storage sized for one `ScratchParams`.
    unsafe { dst.write(p) };

    {
        let guard = cmds.command_encoder().map_err(anyhow::Error::msg)?;
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_scratch_partials(
                device,
                &guard,
                kernels,
                sizing,
                &partials,
                &params,
                geometry.n_heads as u32,
                live as u32,
            )
        }
        .map_err(anyhow::Error::msg)?;
        // One barrier before the combine, and none between the partials -- §9.4.
        // It falls out of `auto_barrier` because the combine binds as an input
        // what the stub bound as an output.
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_scratch_combine(
                device,
                &guard,
                kernels,
                sizing,
                &partials,
                &out,
                &params,
                &order,
                geometry.n_heads as u32,
            )
        }
        .map_err(anyhow::Error::msg)?;
    }
    cmds.wait_until_completed().map_err(anyhow::Error::msg)?;

    let n_out = geometry.n_heads * geometry.head_dim;
    let n_order = geometry.n_heads * live.max(1);
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    let op = unsafe { out.contents() } as *const u32;
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    let rp = unsafe { order.contents() } as *const u32;
    anyhow::ensure!(!op.is_null() && !rp.is_null(), "no CPU mapping");
    // SAFETY: shared storage of the sizes allocated above, and the command
    // buffer has completed.
    let bits = unsafe { std::slice::from_raw_parts(op, n_out) }.to_vec();
    let walked = unsafe { std::slice::from_raw_parts(rp, n_order) }.to_vec();
    Ok((bits, walked))
}

fn new_buffer(device: &Device, bytes: usize) -> Result<Buffer> {
    device
        .new_buffer(bytes.max(4), SHARED)
        .map_err(|e| anyhow::anyhow!("allocating {bytes} B: {e:?}"))
}
