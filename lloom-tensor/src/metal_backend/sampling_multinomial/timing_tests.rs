//! Per-kernel GPU time of the device sampler at the served geometries: the
//! selector, the multinomial draw and the filtered Gumbel draw, each alone in
//! its own command buffer, read from the profiler's `GPUStartTime` /
//! `GPUEndTime` record of that buffer. A measurement, ignored by default; run
//! by hand under `lloom-arb measure` from a build with `--features
//! metal,profile`, the record in `design/measurements/` (lloom #1036).
//!
//! ```text
//! cargo nextest run --locked -p lloom-tensor --features metal,profile \
//!   --cargo-profile ci --run-ignored all --no-capture \
//!   -E 'test(=metal_backend::sampling_multinomial::timing_tests::sampler_kernel_gpu_time)'
//! ```
//!
//! `SAMPLER_TIMING_CELLS=lfm2-ckpt-b1,…` selects cells by name,
//! `SAMPLER_TIMING_REPS=n` the repetitions (default 25). `SAMPLER_TIMING_STAGES=1`
//! adds each stage's own time from stage-boundary counter samples with one
//! encoder per dispatch (`LLOOM_METAL_API=metal3`; Metal 4 refuses the
//! counters, #1128) — a different submission, reported beside the whole.
use crate::kernels::metal::profile;
use crate::metal_backend::MetalDevice;
use crate::{DType, Device, Result, Storage, Tensor};

struct Cell {
    name: &'static str,
    vocab: usize,
    rows: usize,
    temperature: f64,
    k: usize,
    p: f64,
}

const LFM2: usize = 128_000;
const QWEN: usize = 248_320;

const CELLS: &[Cell] = &[
    // LFM2.5's checkpoint sampling: T 0.1, top-k 50.
    Cell {
        name: "lfm2-ckpt-b1",
        vocab: LFM2,
        rows: 1,
        temperature: 0.1,
        k: 50,
        p: 1.0,
    },
    Cell {
        name: "lfm2-ckpt-b4",
        vocab: LFM2,
        rows: 4,
        temperature: 0.1,
        k: 50,
        p: 1.0,
    },
    // Qwen3.6's checkpoint sampling: T 1.0, top-k 20, top-p 0.95.
    Cell {
        name: "lfm2-qwenp-b1",
        vocab: LFM2,
        rows: 1,
        temperature: 1.0,
        k: 20,
        p: 0.95,
    },
    Cell {
        name: "lfm2-qwenp-b4",
        vocab: LFM2,
        rows: 4,
        temperature: 1.0,
        k: 20,
        p: 0.95,
    },
    // Temperature only.
    Cell {
        name: "lfm2-temp-b1",
        vocab: LFM2,
        rows: 1,
        temperature: 0.8,
        k: 0,
        p: 1.0,
    },
    Cell {
        name: "lfm2-temp-b4",
        vocab: LFM2,
        rows: 4,
        temperature: 0.8,
        k: 0,
        p: 1.0,
    },
    // Top-p alone (PR1156's end-to-end arm: p 0.9, T 0.8).
    Cell {
        name: "lfm2-topp-b1",
        vocab: LFM2,
        rows: 1,
        temperature: 0.8,
        k: 0,
        p: 0.9,
    },
    Cell {
        name: "qwen-ckpt-b1",
        vocab: QWEN,
        rows: 1,
        temperature: 0.1,
        k: 50,
        p: 1.0,
    },
    Cell {
        name: "qwen-ckpt-b4",
        vocab: QWEN,
        rows: 4,
        temperature: 0.1,
        k: 50,
        p: 1.0,
    },
    Cell {
        name: "qwen-qwenp-b1",
        vocab: QWEN,
        rows: 1,
        temperature: 1.0,
        k: 20,
        p: 0.95,
    },
    Cell {
        name: "qwen-qwenp-b4",
        vocab: QWEN,
        rows: 4,
        temperature: 1.0,
        k: 20,
        p: 0.95,
    },
    Cell {
        name: "qwen-temp-b1",
        vocab: QWEN,
        rows: 1,
        temperature: 0.8,
        k: 0,
        p: 1.0,
    },
    Cell {
        name: "qwen-temp-b4",
        vocab: QWEN,
        rows: 4,
        temperature: 0.8,
        k: 0,
        p: 1.0,
    },
];

fn device() -> Result<Device> {
    crate::isolation::claim("sampler kernel gpu time");
    let mut tuning = crate::tuning::Tuning::default();
    tuning.from_env().expect("valid test environment");
    tuning.contract.profile = "fast".into();
    tuning.instrument.profile = true;
    if stages() {
        tuning.instrument.profile_counters = true;
        tuning.instrument.profile_encoder_per_dispatch = true;
    }
    crate::tuning::install(tuning).expect("install before tuning is read");
    assert!(
        profile::recorder_enabled(),
        "the detailed recorder needs --features profile (or debug assertions)"
    );
    Device::new_metal(0)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
    /// Roughly normal: twelve uniforms, centred.
    fn normal(&mut self) -> f32 {
        (0..12).map(|_| self.next()).sum::<f32>() - 6.0
    }
}

/// A logit row shaped like a language model's: a Gaussian bulk (sigma 3) and
/// a sparse upper tail, so the k-th largest sits well above the bulk.
pub(super) fn logits(vocab: usize, rows: usize) -> Vec<f32> {
    let mut rng = Lcg(0x1036);
    let mut z = Vec::with_capacity(vocab * rows);
    for row in 0..rows {
        for i in 0..vocab {
            let mut v = 3.0 * rng.normal();
            if (i + 7 * row) % 997 == 0 {
                v += 6.0 + 4.0 * rng.next();
            }
            z.push(v);
        }
    }
    z
}

pub(super) fn variates(vocab: usize, rows: usize) -> Vec<f32> {
    let mut rng = Lcg(0x6b5);
    (0..vocab * rows)
        .map(|_| {
            let u = (rng.next() as f64).clamp(1e-9, 0.999_999_9);
            (-(-u.ln()).ln()) as f32
        })
        .collect()
}

fn metal(t: &Tensor) -> crate::MetalStorage {
    let (s, _) = t.storage_and_layout();
    let Storage::Metal(s) = &*s else {
        unreachable!("Metal tensor")
    };
    s.clone()
}

fn settle(m: &MetalDevice) -> Result<()> {
    m.wait_until_completed()?;
    // The completion handler that files a command buffer's record runs
    // asynchronously; let the previous one land before the window resets.
    std::thread::sleep(std::time::Duration::from_millis(3));
    profile::reset();
    Ok(())
}

/// GPU microseconds of the one command buffer `encode` fills, read from the
/// record whose last dispatch carries `label`.
fn timed(m: &MetalDevice, label: &str, encode: impl FnOnce() -> Result<()>) -> Result<f64> {
    settle(m)?;
    encode()?;
    m.wait_until_completed()?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let records = profile::command_buffer_records();
        let mine: Vec<_> = records
            .iter()
            .filter(|r| r.label.as_deref() == Some(label))
            .collect();
        if let [r] = mine[..] {
            return Ok((r.interval.end - r.interval.start) * 1e6);
        }
        assert!(mine.len() <= 1, "{label}: {} records", mine.len());
        assert!(
            std::time::Instant::now() < deadline,
            "{label}: no timed command buffer; records {records:?}"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

fn stages() -> bool {
    std::env::var_os("SAMPLER_TIMING_STAGES").is_some()
}

/// Each sampled encoder's GPU microseconds over `reps` runs of `encode`, from
/// the counter samples: `(label:dispatches, min, p50, max)` in order of first
/// appearance, the label its first dispatch's. The per-dispatch cut acts when
/// an encoder is acquired, and a pipeline's stages share the one its entry
/// acquired, so an encoder here holds a whole pipeline's dispatches.
fn stage_breakdown(
    m: &MetalDevice,
    encode: impl Fn() -> Result<()>,
    reps: usize,
) -> Result<Vec<(String, f64, f64, f64)>> {
    let mut by_label: Vec<(String, Vec<f64>)> = Vec::new();
    for _ in 0..reps {
        settle(m)?;
        encode()?;
        m.wait_until_completed()?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let records = loop {
            let snap = profile::snapshot();
            if let Some(why) = &snap.counters_unavailable {
                panic!("counters unavailable: {why}");
            }
            if !snap.encoder_records.is_empty() && snap.encoder_buffers_unresolved == 0 {
                break snap.encoder_records;
            }
            assert!(std::time::Instant::now() < deadline, "no encoder records");
            std::thread::sleep(std::time::Duration::from_millis(2));
        };
        for r in records {
            let label = r.label.as_deref().unwrap_or("<unlabelled>").to_string();
            let us = (r.end - r.start) * 1e6;
            let label = format!("{label}:{}", r.dispatches);
            match by_label.iter_mut().find(|(l, _)| *l == label) {
                Some((_, v)) => v.push(us),
                None => by_label.push((label, vec![us])),
            }
        }
    }
    Ok(by_label
        .into_iter()
        .map(|(l, v)| {
            let (min, p50, max) = stats(v);
            (l, min, p50, max)
        })
        .collect())
}

fn stats(mut us: Vec<f64>) -> (f64, f64, f64) {
    us.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    (us[0], us[us.len() / 2], us[us.len() - 1])
}

#[test]
#[ignore]
fn sampler_kernel_gpu_time() -> Result<()> {
    let d = device()?;
    let m = d.as_metal_device()?;
    let reps: usize = std::env::var("SAMPLER_TIMING_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(25);
    let only: Vec<String> = std::env::var("SAMPLER_TIMING_CELLS")
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();
    println!(
        "SAMPLER_TIMING api={} contract=fast reps={reps} warmup=3 unit=us (GPUStartTime..GPUEndTime of one command buffer holding the one dispatch)",
        m.commands().actual_command_api()
    );
    let mut table = Vec::new();
    for cell in CELLS {
        if !only.is_empty() && !only.iter().any(|n| n == cell.name) {
            continue;
        }
        let (vocab, rows) = (cell.vocab, cell.rows);
        let z = Tensor::from_vec(logits(vocab, rows), (rows, vocab), &d)?;
        let uniforms: Vec<f32> = (0..rows).map(|r| 0.37 + 0.13 * r as f32).collect();
        let u = Tensor::from_vec(uniforms, rows, &d)?;
        let g = Tensor::from_vec(variates(vocab, rows), (rows, vocab), &d)?;
        let (z, u, g) = (metal(&z), metal(&u), metal(&g));
        let alloc = |count: usize, dtype, label| {
            m.new_buffer_builder()
                .with_size_for(count, dtype)
                .with_label(label)
                .build()
        };
        let elements = vocab * rows;
        let words =
            crate::kernels::kernels::sampling_topk::sampling_work_words(rows, vocab).unwrap();
        let work = alloc(words, DType::U32, "timing_work")?;
        let scratch = alloc(elements, DType::U32, "timing_scratch_ids")?;
        let selection = alloc(rows * 4, DType::U32, "timing_selection")?;
        let weights = alloc(elements, DType::F32, "timing_weights")?;
        let support = alloc(elements, DType::U32, "timing_support")?;
        let ids = alloc(rows, DType::U32, "timing_ids")?;
        let status = alloc(rows * 4, DType::U32, "timing_status")?;
        let inv_t = (1.0 / cell.temperature) as f32;
        let filter_p = u32::from(cell.p != 1.0);
        let k = cell.k.min(vocab) as u32;
        let select = || -> Result<()> {
            let encoder = m.command_encoder()?;
            // SAFETY: sized, disjoint, live buffers owned by this test.
            unsafe {
                crate::kernels::kernels::sampling_topk::call_sampling_select(
                    m.metal_device(),
                    &encoder,
                    m.kernels(),
                    (z.buffer(), 0),
                    rows as u32,
                    vocab as u32,
                    k,
                    filter_p,
                    &work,
                    &scratch,
                    &selection,
                )
            }
            .map_err(crate::Error::wrap)
        };
        let draw = |gumbel: bool| -> Result<()> {
            let encoder = m.command_encoder()?;
            let call = if gumbel {
                crate::kernels::kernels::sampling_topk::call_sampling_filtered_gumbel_f32
            } else {
                crate::kernels::kernels::sampling_topk::call_sampling_multinomial
            };
            let noise = if gumbel { &g } else { &u };
            // SAFETY: sized, disjoint, live buffers owned by this test; the
            // selector outputs come from the same rows.
            unsafe {
                call(
                    m.metal_device(),
                    &encoder,
                    m.kernels(),
                    (z.buffer(), 0),
                    rows as u32,
                    vocab as u32,
                    inv_t,
                    cell.p as f32,
                    filter_p,
                    (noise.buffer(), 0),
                    &work,
                    &scratch,
                    &weights,
                    &support,
                    &ids,
                    &status,
                )
            }
            .map_err(crate::Error::wrap)
        };
        // Each pipeline is one command buffer; the record carries its last
        // dispatch's label.
        let kernels: [(&str, &str, Box<dyn Fn() -> Result<()> + '_>); 5] = [
            ("select", "sampling_select_rank_f32", Box::new(&select)),
            (
                "multinomial",
                "sampling_support_f32",
                Box::new(|| draw(false)),
            ),
            (
                "gumbel",
                "sampling_gumbel_pick_f32",
                Box::new(|| draw(true)),
            ),
            // The production shape: the selection and the draw in one buffer.
            (
                "select+multinomial",
                "sampling_support_f32",
                Box::new(|| select().and_then(|()| draw(false))),
            ),
            (
                "select+gumbel",
                "sampling_gumbel_pick_f32",
                Box::new(|| select().and_then(|()| draw(true))),
            ),
        ];
        // The selection runs first so the draws read a real kept set.
        for (_, label, encode) in &kernels {
            for _ in 0..3 {
                timed(m, label, encode)?;
            }
        }
        for (stage, label, encode) in &kernels {
            let us: Vec<f64> = (0..reps)
                .map(|_| timed(m, label, encode))
                .collect::<Result<_>>()?;
            let (min, p50, max) = stats(us);
            println!(
                "SAMPLER_GPU_US cell={} vocab={vocab} rows={rows} T={} k={} p={} stage={stage} last={label} n={reps} min={min:.1} p50={p50:.1} max={max:.1}",
                cell.name, cell.temperature, cell.k, cell.p
            );
            table.push((cell.name, stage.to_string(), rows, p50));
            if stages() {
                for (label, min, p50, max) in stage_breakdown(m, encode, reps.min(7))? {
                    println!(
                        "SAMPLER_STAGE_US cell={} rows={rows} stage={stage} kernel={label} min={min:.1} p50={p50:.1} max={max:.1}",
                        cell.name
                    );
                }
            }
        }
        drop(kernels);
        let ids = crate::MetalStorage::new_allocated(ids, m.clone(), rows, DType::U32)?;
        let status = crate::MetalStorage::new_allocated(status, m.clone(), rows * 4, DType::U32)?;
        println!(
            "SAMPLER_RESULT cell={} ids={:?} status={:?}",
            cell.name,
            ids.to_cpu::<u32>()?,
            status.to_cpu::<u32>()?
        );
    }
    println!(
        "| cell | rows | select us | multinomial us | gumbel us | select+multinomial us | select+gumbel us |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|");
    for chunk in table.chunks(5) {
        if let [a, b, c, d, e] = chunk {
            println!(
                "| {} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} |",
                a.0, a.2, a.3, b.3, c.3, d.3, e.3
            );
        }
    }
    Ok(())
}
