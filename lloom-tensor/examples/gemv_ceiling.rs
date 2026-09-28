//! Is the copy sweep's 349.7 GB/s the ceiling of the decode GEMV's access pattern?
//!
//! `DESIGN.md` §3.4's achievable bandwidth is a copy kernel's figure:
//! `stream_h4x2` reading ONE 5 GiB buffer in ONE dispatch. Decode reads the
//! same bytes as 167 matrices, one dispatch each, through the production GEMV
//! tile, behind a barrier. This times the same weight set every way in one
//! process and one lease, so the distance between the two figures is measured
//! rather than argued (issue #716):
//!
//! * `copy-single-tg<T>` -- the §3.4 kernel over one 5 GiB buffer: the
//!   reproduction of the denominator.
//! * `copy-seq[-nb]` -- the same kernel over the 167 decode weight buffers, one
//!   dispatch each: what splitting the sweep costs a kernel that only streams.
//! * `gemv:<tile>[-nb]` -- the production `gemv` template over those buffers at
//!   the shipping tile (`prod`: `bm8` at N >= 4096 else `bm4`, `tm4 tn4`, the
//!   `derive_gemv_tile` rule) and at wider tiles.
//!
//! `-nb` drops the buffer barrier between dispatches and `-dec` keeps it
//! exactly where decode has one (every GEMV but `k`, `v` and `w3`, which read
//! their predecessor's input and are free to overlap it — the trace's ordering
//! census). The barrier-everywhere arm is the fully serialized bound, `-dec` is
//! the decode step's GEMVs alone, and `-nb` bounds what overlap could buy.
//!
//! One command buffer per sweep, timed by `GPUEndTime - GPUStartTime` (§3.4b,
//! domain A). `--per-dispatch` also runs chosen arms with ONE ENCODER PER
//! DISPATCH carrying stage-boundary counter samples (§13.1d: the only sampling
//! point this GPU supports), which prices every matrix and is the stand-alone
//! validation of the engine's per-encoder instrument: every sample must fall
//! inside its command buffer's interval, and the encoders must sum to it.
//!
//! Correctness gates the timing: every tile is checked against a CPU reference
//! first, beside an inverted arm (the shipping tile told K is half its size)
//! that must fail the same check.
//!
//! ```bash
//! lloom-arb measure -- ./target/release/examples/gemv_ceiling --rotations 5
//! ```

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLAllocation, MTLBarrierScope, MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer,
    MTLCommandEncoder, MTLCommandQueue, MTLCommonCounterSetTimestamp, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePassDescriptor, MTLComputePipelineState,
    MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor, MTLCounterSamplingPoint,
    MTLCounterSet, MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLDispatchType,
    MTLFunctionConstantValues, MTLLibrary, MTLMathFloatingPointFunctions, MTLMathMode,
    MTLResidencySet, MTLResidencySetDescriptor, MTLResourceOptions, MTLSize, MTLStorageMode,
};
use std::ptr::NonNull;

type Dev = ProtocolObject<dyn MTLDevice>;
type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipe = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type CmdBuf = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type Enc = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;
type SampleBuf = Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>;

#[derive(Parser, Debug)]
#[command(about = "copy sweep vs the decode GEMV over LFM2's weight set (#716)")]
struct Args {
    /// Arms, in rotation order. See the module doc for the spellings; a
    /// `#null` suffix repeats an arm in a second slot as the null control.
    #[arg(
        long,
        value_delimiter = ',',
        default_values_t = [
            "copy-single-tg256", "copy-single-tg512", "copy-single-tg1024",
            "copy-seq", "copy-seq-nb",
            "gemv:prod", "gemv:prod-dec", "gemv:prod-nb", "gemv:prod#null",
            "gemv:tm4tn8", "gemv:tm8tn4", "gemv:tm8tn8", "gemv:tm2tn8",
            "gemv:bm4all", "gemv:bm8all",
        ].map(String::from)
    )]
    arms: Vec<String>,

    /// Interleaved rotations over every arm (§3.2: A, B, A, B).
    #[arg(long, default_value_t = 5)]
    rotations: usize,

    /// Timed sweeps per arm per rotation.
    #[arg(long, default_value_t = 5)]
    iters: usize,

    /// Untimed sweeps before each arm's timed ones.
    #[arg(long, default_value_t = 2)]
    warmup: usize,

    /// Storage of the weight buffers: `shared` is what the engine allocates
    /// (`RESOURCE_OPTIONS`, `metal_backend/device.rs`); `private` is what the
    /// §3.4 sweep allocated.
    #[arg(long, default_value = "shared")]
    storage: String,

    /// Put every buffer in a residency set attached to the queue, as the
    /// engine does. `--no-residency` measures without it.
    #[arg(long = "no-residency", action = clap::ArgAction::SetFalse, default_value_t = true)]
    residency: bool,

    /// Arms to also run with one counter-sampled encoder per dispatch, once
    /// per rotation after the timed arms.
    #[arg(long, value_delimiter = ',')]
    per_dispatch: Vec<String>,

    /// Print one `SWEEP` line per timed sweep (the raw series).
    #[arg(long)]
    raw: bool,

    /// Replace the decode sequence with a chain of distinct `NxK` matrices
    /// (e.g. `2048x2048`), `--total-mb` of them: one shape at a time, so a
    /// per-dispatch fixed cost and the streaming rate can be separated by size.
    #[arg(long)]
    shape: Option<String>,

    /// The chain's total size under `--shape`, MB. Past the 48 MB SLC by a wide
    /// margin, so no matrix is still cached when the chain comes back to it.
    #[arg(long, default_value_t = 1100)]
    total_mb: usize,
}

/// One weight matrix of a decode step: `[n, k]` row-major f16, `y = W x`.
#[derive(Clone, Copy, Debug)]
struct Mat {
    name: &'static str,
    n: usize,
    k: usize,
}

impl Mat {
    fn bytes(&self) -> usize {
        self.n * self.k * 2
    }

    /// Whether decode runs this GEMV with no barrier before it: `k`/`v` read
    /// `q`'s input and `w3` reads `w1`'s, so nothing orders them after their
    /// predecessor. Observed, not modelled: the trace's ordering census (the
    /// `lfm2-chat` `trace ordering` line) finds 16 free 2 MB and 30 free
    /// 44 MB GEMVs a step and every other GEMV ordered.
    fn decode_free(&self) -> bool {
        matches!(self.name, "k_proj" | "v_proj" | "w3")
    }
}

/// LFM2's decode-step weight sweep in layer order (`DESIGN.md` §5.3, §5.4):
/// 30 FFNs of three, 22 conv layers of two, 8 attention layers of four, and the
/// tied lm_head. 167 matrices, the census's GEMV count (§13.1a).
fn decode_sequence() -> Vec<Mat> {
    const ATTN: [usize; 8] = [2, 5, 9, 13, 17, 21, 24, 27];
    let mut v = Vec::new();
    for layer in 0..30 {
        if ATTN.contains(&layer) {
            v.push(Mat {
                name: "q_proj",
                n: 2048,
                k: 2048,
            });
            v.push(Mat {
                name: "k_proj",
                n: 512,
                k: 2048,
            });
            v.push(Mat {
                name: "v_proj",
                n: 512,
                k: 2048,
            });
            v.push(Mat {
                name: "attn_out",
                n: 2048,
                k: 2048,
            });
        } else {
            v.push(Mat {
                name: "conv_in",
                n: 6144,
                k: 2048,
            });
            v.push(Mat {
                name: "conv_out",
                n: 2048,
                k: 2048,
            });
        }
        v.push(Mat {
            name: "w1",
            n: 10752,
            k: 2048,
        });
        v.push(Mat {
            name: "w3",
            n: 10752,
            k: 2048,
        });
        v.push(Mat {
            name: "w2",
            n: 2048,
            k: 10752,
        });
    }
    v.push(Mat {
        name: "lm_head",
        n: 128000,
        k: 2048,
    });
    v
}

/// A GEMV tile: `bm` simdgroups of `tm` rows, each lane `tn` columns wide.
/// `bm: None` is the production rule (`derive_gemv_tile`: 8 at N >= 4096).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Tile {
    bm: Option<usize>,
    tm: usize,
    tn: usize,
}

impl Tile {
    fn bm_for(&self, n: usize) -> usize {
        self.bm.unwrap_or(if n >= 4096 { 8 } else { 4 })
    }
    fn fn_name(&self, n: usize) -> String {
        tile_name(self.bm_for(n), self.tm, self.tn)
    }
}

/// The `(bm, tm, tn)` tiles `gemv.metal` already instantiates at `bn1 sm1
/// sn32` (`instantiate_gemv_blocks`). MSL rejects a second explicit
/// instantiation of the same template arguments, so these are reached by
/// their production `[[host_name]]` -- the name the census reports -- and only
/// the other tiles are instantiated here.
const LIBRARY_TILES: [(usize, usize, usize); 3] = [(4, 1, 4), (4, 4, 4), (8, 4, 4)];

fn tile_name(bm: usize, tm: usize, tn: usize) -> String {
    if LIBRARY_TILES.contains(&(bm, tm, tn)) {
        format!("gemv_float16_bm{bm}_bn1_sm1_sn32_tm{tm}_tn{tn}_nc0_axpby0")
    } else {
        format!("probe_gemv_bm{bm}_tm{tm}_tn{tn}")
    }
}

const PROD: Tile = Tile {
    bm: None,
    tm: 4,
    tn: 4,
};

fn parse_tile(s: &str) -> Result<Tile> {
    Ok(match s {
        "prod" => PROD,
        "bm4all" => Tile {
            bm: Some(4),
            ..PROD
        },
        "bm8all" => Tile {
            bm: Some(8),
            ..PROD
        },
        other => {
            let (tm, tn) = other
                .strip_prefix("tm")
                .and_then(|r| r.split_once("tn"))
                .context(format!(
                    "tile {other:?}: want prod|bm4all|bm8all|tm<a>tn<b>"
                ))?;
            Tile {
                bm: None,
                tm: tm.parse()?,
                tn: tn.parse()?,
            }
        }
    })
}

#[derive(Clone, Debug, PartialEq)]
enum Kind {
    CopySingle { tg: usize },
    CopySeq,
    Gemv(Tile),
}

#[derive(Clone, Debug)]
struct Arm {
    label: String,
    kind: Kind,
    barrier: bool,
    /// `-dec`: a barrier only where decode has one (`Mat::decode_free`).
    decode_order: bool,
}

fn parse_arm(s: &str) -> Result<Arm> {
    let base = s.split('#').next().unwrap_or(s);
    let (base, barrier, decode_order) = if let Some(b) = base.strip_suffix("-nb") {
        (b, false, false)
    } else if let Some(b) = base.strip_suffix("-dec") {
        (b, true, true)
    } else {
        (base, true, false)
    };
    let kind = if let Some(tg) = base.strip_prefix("copy-single-tg") {
        Kind::CopySingle { tg: tg.parse()? }
    } else if base == "copy-seq" {
        Kind::CopySeq
    } else if let Some(t) = base.strip_prefix("gemv:") {
        Kind::Gemv(parse_tile(t)?)
    } else {
        bail!("unknown arm {s:?}")
    };
    Ok(Arm {
        label: s.to_string(),
        kind,
        barrier,
        decode_order,
    })
}

/// The §3.4 streaming kernel, verbatim from `examples/bandwidth.rs`.
const STREAM_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void stream_h4x2(device const half4* src [[buffer(0)]],
                        device float*       dst [[buffer(1)]],
                        constant uint&      n   [[buffer(2)]],
                        uint tid  [[thread_position_in_grid]],
                        uint nthr [[threads_per_grid]]) {
    float4 a = float4(0.0f);
    float4 b = float4(0.0f);
    uint stride = nthr * 2;
    for (uint i = tid * 2; i + 1 < n; i += stride) {
        a += float4(src[i]);
        b += float4(src[i + 1]);
    }
    float4 acc = a + b;
    dst[tid] = acc.x + acc.y + acc.z + acc.w;
}
"#;

/// The production GEMV source plus one explicit instantiation per tile, named
/// by `Tile::fn_name`. Same template, same compile options as the engine.
fn gemv_source(tiles: &[Tile]) -> String {
    let mut s = String::from(lloom_tensor::kernels::source::GEMV);
    let mut seen = std::collections::HashSet::new();
    for t in tiles {
        for bm in [4usize, 8] {
            if t.bm.is_some_and(|b| b != bm) {
                continue;
            }
            if LIBRARY_TILES.contains(&(bm, t.tm, t.tn)) {
                continue;
            }
            let name = tile_name(bm, t.tm, t.tn);
            if seen.insert(name.clone()) {
                s.push_str(&format!(
                    "\ntemplate [[host_name(\"{name}\")]] [[kernel]] \
                     decltype(gemv<half, {bm}, 1, 1, 32, {tm}, {tn}, false, false>) \
                     gemv<half, {bm}, 1, 1, 32, {tm}, {tn}, false, false>;\n",
                    tm = t.tm,
                    tn = t.tn
                ));
            }
        }
    }
    s
}

struct Probe {
    device: Retained<Dev>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    stream: Pipe,
    gemv: std::collections::HashMap<String, Pipe>,
    mats: Vec<Mat>,
    weights: Vec<Buf>,
    outs: Vec<Buf>,
    x_small: Buf,
    x_large: Buf,
    copy_dst: Buf,
    single: Option<Buf>,
    weight_bytes: usize,
    counter_set: Option<Retained<ProtocolObject<dyn MTLCounterSet>>>,
}

fn options(storage: &str) -> Result<MTLResourceOptions> {
    let base = MTLResourceOptions::HazardTrackingModeUntracked.0;
    Ok(MTLResourceOptions(match storage {
        "shared" => MTLResourceOptions::StorageModeShared.0 | base,
        "private" => MTLResourceOptions::StorageModePrivate.0 | base,
        other => bail!("--storage must be shared|private, got {other:?}"),
    }))
}

fn new_buffer(device: &Dev, bytes: usize, opts: MTLResourceOptions) -> Result<Buf> {
    device
        .newBufferWithLength_options(bytes, opts)
        .with_context(|| format!("allocating {bytes} bytes"))
}

/// Deterministic f16 fill with magnitudes in [2^-5, 2^0): nonzero everywhere,
/// finite, and different per buffer, so nothing downstream can shortcut it.
fn fill_f16(dst: &mut [u16], seed: u64) {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    for chunk in dst.chunks_mut(4) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        for (i, h) in chunk.iter_mut().enumerate() {
            let r = (x >> (16 * i)) as u16;
            let exp = 10 + (r >> 10) % 5;
            *h = (r & 0x83FF) | (exp << 10);
        }
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1F) as i32;
    let man = (h & 0x3FF) as f32;
    if exp == 0 {
        sign * man * 2f32.powi(-24)
    } else {
        sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15)
    }
}

/// Copy host bytes into `dst` (shared: directly; private: through a staging
/// blit), `len` u16 elements generated by `fill_f16(seed)`.
fn upload(
    device: &Dev,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    dst: &Buf,
    len: usize,
    seed: u64,
    private: bool,
) -> Result<()> {
    if !private {
        // SAFETY: a shared buffer's contents are CPU-visible for `length()`
        // bytes, and `len * 2 <= length()` by construction at every call site.
        let s =
            unsafe { std::slice::from_raw_parts_mut(dst.contents().as_ptr().cast::<u16>(), len) };
        fill_f16(s, seed);
        return Ok(());
    }
    const CHUNK: usize = 64 << 20;
    let staging = new_buffer(device, CHUNK, MTLResourceOptions::StorageModeShared)?;
    let mut off = 0usize;
    let mut part = 0u64;
    while off < len * 2 {
        let n = CHUNK.min(len * 2 - off);
        // SAFETY: `staging` is shared and CHUNK bytes long; `n <= CHUNK`.
        let s = unsafe {
            std::slice::from_raw_parts_mut(staging.contents().as_ptr().cast::<u16>(), n / 2)
        };
        fill_f16(s, seed ^ (part << 32));
        let cb = queue.commandBuffer().context("command buffer")?;
        let blit = cb.blitCommandEncoder().context("blit encoder")?;
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &staging, 0, dst, off, n,
            )
        };
        blit.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        off += n;
        part += 1;
    }
    Ok(())
}

fn compile(device: &Dev, src: &str) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let opts = MTLCompileOptions::new();
    // The engine's default (`kernel.rs` `get_compile_options`, fast math on).
    opts.setMathMode(MTLMathMode::Fast);
    opts.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Fast);
    device
        .newLibraryWithSource_options_error(&NSString::from_str(src), Some(&opts))
        .map_err(|e| anyhow::anyhow!("compiling: {e}"))
}

fn pipeline(
    device: &Dev,
    lib: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
    eos_constant: bool,
) -> Result<Pipe> {
    let f = if eos_constant {
        // `gemv.metal` references the EOS gate's function constant (777), so
        // the function exists only specialised; `false` is the ungated form.
        let values = MTLFunctionConstantValues::new();
        let off = false;
        unsafe {
            values.setConstantValue_type_atIndex(NonNull::from(&off).cast(), MTLDataType::Bool, 777)
        };
        lib.newFunctionWithName_constantValues_error(&NSString::from_str(name), &values)
            .map_err(|e| anyhow::anyhow!("function {name}: {e}"))?
    } else {
        lib.newFunctionWithName(&NSString::from_str(name))
            .with_context(|| format!("function {name}"))?
    };
    device
        .newComputePipelineStateWithFunction_error(&f)
        .map_err(|e| anyhow::anyhow!("pipeline {name}: {e}"))
}

fn set_bytes<T>(enc: &Enc, v: &T, index: usize) {
    // SAFETY: `v` is a live value of `size_of::<T>()` bytes; Metal copies it.
    unsafe { enc.setBytes_length_atIndex(NonNull::from(v).cast(), std::mem::size_of::<T>(), index) }
}

impl Probe {
    /// One GEMV dispatch of `mat` against `x` into `out`, `in_size` being what
    /// the kernel is TOLD K is (the inverted parity arm lies here).
    #[allow(clippy::too_many_arguments)]
    fn encode_gemv(
        &self,
        enc: &Enc,
        tile: Tile,
        mat: &Mat,
        w: &Buf,
        x: &Buf,
        out: &Buf,
        in_size: usize,
    ) -> Result<()> {
        let bm = tile.bm_for(mat.n);
        let name = tile.fn_name(mat.n);
        let p = self
            .gemv
            .get(&name)
            .with_context(|| format!("no pipeline {name}"))?;
        enc.setComputePipelineState(p);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(w), 0, 0);
            enc.setBuffer_offset_atIndex(Some(x), 0, 1);
            enc.setBuffer_offset_atIndex(Some(x), 0, 2);
            enc.setBuffer_offset_atIndex(Some(out), 0, 3);
        }
        set_bytes(enc, &(in_size as i32), 4);
        set_bytes(enc, &(mat.n as i32), 5);
        set_bytes(enc, &(mat.k as i32), 6);
        set_bytes(enc, &1.0f32, 7);
        set_bytes(enc, &0.0f32, 8);
        set_bytes(enc, &1i32, 9);
        set_bytes(enc, &[1i32; 1], 10);
        set_bytes(enc, &[0i64; 1], 11);
        set_bytes(enc, &[0i64; 1], 12);
        set_bytes(enc, &[0i64; 1], 13);
        set_bytes(enc, &1i32, 14);
        let rows = bm * tile.tm;
        ensure!(
            mat.n.is_multiple_of(rows),
            "{}: N={} not a multiple of {rows}",
            mat.name,
            mat.n
        );
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: mat.n / rows,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: bm,
            },
        );
        Ok(())
    }

    fn encode_stream(&self, enc: &Enc, src: &Buf, bytes: usize, tg: usize) {
        let n_h4 = (bytes / 8) as u32;
        // Each thread streams at least 16 half4, and the grid is capped where
        // §3.4's sweep capped it (4096 groups).
        let groups = (bytes / 8).div_ceil(tg * 16).clamp(1, 4096);
        enc.setComputePipelineState(&self.stream);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&self.copy_dst), 0, 1);
        }
        set_bytes(enc, &n_h4, 2);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Encode one full sweep of `arm` into `cb`. With `sample`, one encoder
    /// per dispatch, each sampled at its stage boundaries into indices 2i/2i+1.
    fn encode_sweep(&self, cb: &CmdBuf, arm: &Arm, sample: Option<&SampleBuf>) -> Result<usize> {
        let open = |i: usize| -> Result<Enc> {
            match sample {
                None => cb
                    .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
                    .context("encoder"),
                Some(sb) => {
                    let d = MTLComputePassDescriptor::computePassDescriptor();
                    d.setDispatchType(MTLDispatchType::Concurrent);
                    let a = unsafe { d.sampleBufferAttachments().objectAtIndexedSubscript(0) };
                    a.setSampleBuffer(Some(sb));
                    unsafe {
                        a.setStartOfEncoderSampleIndex(2 * i);
                        a.setEndOfEncoderSampleIndex(2 * i + 1);
                    }
                    cb.computeCommandEncoderWithDescriptor(&d)
                        .context("sampled encoder")
                }
            }
        };
        let mut dispatches = 0usize;
        match &arm.kind {
            Kind::CopySingle { tg } => {
                let enc = open(0)?;
                let src = self
                    .single
                    .as_ref()
                    .context("copy-single needs its buffer")?;
                self.encode_stream(&enc, src, src.length(), *tg);
                enc.endEncoding();
                dispatches = 1;
            }
            Kind::CopySeq | Kind::Gemv(_) => {
                // Per-dispatch encoders are ORDERED by a fence, as the engine
                // orders its encoders: without one, independent encoders run
                // concurrently and each interval holds its neighbours' time
                // (measured: 23 ms of intervals in an 18 ms buffer).
                let fence = if sample.is_some() {
                    Some(self.device.newFence().context("fence")?)
                } else {
                    None
                };
                let mut enc = open(0)?;
                for (i, (m, w)) in self.mats.iter().zip(&self.weights).enumerate() {
                    if i > 0 {
                        if let Some(f) = &fence {
                            enc.updateFence(f);
                            enc.endEncoding();
                            enc = open(i)?;
                            enc.waitForFence(f);
                        } else if arm.barrier && !(arm.decode_order && m.decode_free()) {
                            enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                        }
                    }
                    match &arm.kind {
                        Kind::CopySeq => self.encode_stream(&enc, w, m.bytes(), 1024),
                        Kind::Gemv(t) => {
                            let x = if m.k == 2048 {
                                &self.x_small
                            } else {
                                &self.x_large
                            };
                            self.encode_gemv(&enc, *t, m, w, x, &self.outs[i], m.k)?
                        }
                        Kind::CopySingle { .. } => unreachable!(),
                    }
                    dispatches += 1;
                }
                enc.endEncoding();
            }
        }
        Ok(dispatches)
    }

    fn sweep_bytes(&self, arm: &Arm) -> usize {
        match arm.kind {
            Kind::CopySingle { .. } => self.single.as_ref().map_or(0, |b| b.length()),
            _ => self.weight_bytes,
        }
    }
}

/// `f64` quantile of a sorted slice, nearest-rank.
fn q(xs: &[f64], f: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs[(((xs.len() - 1) as f64) * f).round() as usize]
}

/// The CPU reference and the GPU's answer for one matrix, both f32.
fn parity(probe: &Probe, tile: Tile, mat: Mat, told_k: usize) -> Result<f32> {
    let opts = MTLResourceOptions::StorageModeShared;
    let w = new_buffer(&probe.device, mat.bytes(), opts)?;
    let x = new_buffer(&probe.device, mat.k * 2, opts)?;
    let y = new_buffer(&probe.device, mat.n * 2, opts)?;
    upload(&probe.device, &probe.queue, &w, mat.n * mat.k, 11, false)?;
    upload(&probe.device, &probe.queue, &x, mat.k, 12, false)?;
    let cb = probe.queue.commandBuffer().context("cb")?;
    let enc = cb
        .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
        .context("enc")?;
    probe.encode_gemv(&enc, tile, &mat, &w, &x, &y, told_k)?;
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    // SAFETY: the three shared buffers were sized above and the GPU is done.
    let (wh, xh, yh) = unsafe {
        (
            std::slice::from_raw_parts(w.contents().as_ptr().cast::<u16>(), mat.n * mat.k),
            std::slice::from_raw_parts(x.contents().as_ptr().cast::<u16>(), mat.k),
            std::slice::from_raw_parts(y.contents().as_ptr().cast::<u16>(), mat.n),
        )
    };
    let xf: Vec<f32> = xh.iter().map(|&h| f16_to_f32(h)).collect();
    let mut worst = 0f32;
    let mut scale = 0f32;
    for r in 0..mat.n {
        let row = &wh[r * mat.k..(r + 1) * mat.k];
        let want: f32 = row.iter().zip(&xf).map(|(&a, &b)| f16_to_f32(a) * b).sum();
        let got = f16_to_f32(yh[r]);
        worst = worst.max((got - want).abs());
        scale = scale.max(want.abs());
    }
    Ok(worst / scale.max(1e-30))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let arms: Vec<Arm> = args
        .arms
        .iter()
        .map(|s| parse_arm(s))
        .collect::<Result<_>>()?;
    let pd_arms: Vec<Arm> = args
        .per_dispatch
        .iter()
        .map(|s| parse_arm(s))
        .collect::<Result<_>>()?;
    let private = args.storage == "private";
    let opts = options(&args.storage)?;

    let device = MTLCreateSystemDefaultDevice().context("no Metal device")?;
    let queue = device.newCommandQueue().context("no command queue")?;
    println!(
        "device {} storage={} residency={}",
        device.name(),
        args.storage,
        args.residency
    );

    let mut tiles: Vec<Tile> = vec![PROD];
    for a in arms.iter().chain(&pd_arms) {
        if let Kind::Gemv(t) = a.kind {
            if !tiles.contains(&t) {
                tiles.push(t);
            }
        }
    }
    let stream_lib = compile(&device, STREAM_SRC)?;
    let gemv_lib = compile(&device, &gemv_source(&tiles))?;
    let stream = pipeline(&device, &stream_lib, "stream_h4x2", false)?;
    let mut gemv = std::collections::HashMap::new();
    for t in &tiles {
        for n in [512usize, 10752] {
            if let std::collections::hash_map::Entry::Vacant(slot) = gemv.entry(t.fn_name(n)) {
                let p = pipeline(&device, &gemv_lib, slot.key(), true)?;
                println!(
                    "pipeline {}: maxTotalThreadsPerThreadgroup={} threadExecutionWidth={}",
                    slot.key(),
                    p.maxTotalThreadsPerThreadgroup(),
                    p.threadExecutionWidth()
                );
                slot.insert(p);
            }
        }
    }

    let mats = match &args.shape {
        None => decode_sequence(),
        Some(s) => {
            let (n, k) = s
                .split_once('x')
                .context("--shape wants NxK")
                .and_then(|(n, k)| Ok((n.parse::<usize>()?, k.parse::<usize>()?)))?;
            let m = Mat {
                name: "uniform",
                n,
                k,
            };
            vec![m; (args.total_mb * 1_000_000).div_ceil(m.bytes()).max(2)]
        }
    };
    let weight_bytes: usize = mats.iter().map(Mat::bytes).sum();
    println!(
        "weights {} matrices, {:.3} MB",
        mats.len(),
        weight_bytes as f64 / 1e6
    );

    let counter_set = device.counterSets().and_then(|sets| {
        // SAFETY: a framework-provided static NSString.
        let ts = unsafe { MTLCommonCounterSetTimestamp };
        sets.iter().find(|s| s.name().isEqualToString(ts))
    });
    println!(
        "counters: stage-boundary sampling {} / dispatch-boundary {}; timestamp set {}",
        device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary),
        device.supportsCounterSampling(MTLCounterSamplingPoint::AtDispatchBoundary),
        if counter_set.is_some() {
            "present"
        } else {
            "ABSENT"
        }
    );

    let mut probe = Probe {
        stream,
        gemv,
        weights: Vec::new(),
        outs: Vec::new(),
        x_small: new_buffer(&device, 2048 * 2, MTLResourceOptions::StorageModeShared)?,
        x_large: new_buffer(&device, 10752 * 2, MTLResourceOptions::StorageModeShared)?,
        copy_dst: new_buffer(
            &device,
            1024 * 4096 * 4,
            MTLResourceOptions::StorageModePrivate,
        )?,
        single: None,
        weight_bytes,
        counter_set,
        mats: mats.clone(),
        device,
        queue,
    };
    upload(&probe.device, &probe.queue, &probe.x_small, 2048, 1, false)?;
    upload(&probe.device, &probe.queue, &probe.x_large, 10752, 2, false)?;

    // ---- correctness first: nothing below is readable if this fails -------
    let checks = [
        Mat {
            name: "k_proj",
            n: 512,
            k: 2048,
        },
        Mat {
            name: "w2",
            n: 2048,
            k: 10752,
        },
    ];
    for m in checks {
        for t in &tiles {
            let e = parity(&probe, *t, m, m.k)?;
            println!(
                "PARITY tile={t:?} {} [{}x{}] rel_err={e:.2e} {}",
                m.name,
                m.n,
                m.k,
                if e < 2e-2 { "ok" } else { "FAIL" }
            );
            ensure!(
                e < 2e-2,
                "tile {t:?} disagrees with the CPU reference on {}: {e}",
                m.name
            );
        }
        let inv = parity(&probe, PROD, m, m.k / 2)?;
        println!(
            "PARITY inverted-arm (prod told K/2) {} rel_err={inv:.2e} {}",
            m.name,
            if inv >= 2e-2 {
                "FAILS, as it must"
            } else {
                "PASSES: the check is blind"
            }
        );
        ensure!(
            inv >= 2e-2,
            "the parity check cannot see a kernel that reads half the matrix"
        );
    }

    // ---- the weight set, then the single buffer -----------------------------
    for (i, m) in mats.iter().enumerate() {
        let b = new_buffer(&probe.device, m.bytes(), opts)?;
        upload(
            &probe.device,
            &probe.queue,
            &b,
            m.n * m.k,
            1000 + i as u64,
            private,
        )?;
        probe.weights.push(b);
        probe.outs.push(new_buffer(
            &probe.device,
            m.n * 2,
            MTLResourceOptions::StorageModeShared,
        )?);
    }
    if arms
        .iter()
        .chain(&pd_arms)
        .any(|a| matches!(a.kind, Kind::CopySingle { .. }))
    {
        let bytes = 5120usize << 20;
        let b = new_buffer(&probe.device, bytes, opts)?;
        upload(&probe.device, &probe.queue, &b, bytes / 2, 7, private)?;
        probe.single = Some(b);
    }
    let residency = if args.residency {
        let set = probe
            .device
            .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
            .map_err(|e| anyhow::anyhow!("residency set: {e}"))?;
        let all = probe
            .weights
            .iter()
            .chain(&probe.outs)
            .chain(probe.single.iter())
            .chain([&probe.x_small, &probe.x_large, &probe.copy_dst]);
        for b in all {
            // SAFETY: an MTLBuffer is an MTLAllocation; the cast is the one
            // `residency_set.rs` `as_allocation` makes.
            let a = unsafe {
                &*((&**b) as *const ProtocolObject<dyn MTLBuffer>
                    as *const ProtocolObject<dyn MTLAllocation>)
            };
            set.addAllocation(a);
        }
        set.commit();
        set.requestResidency();
        probe.queue.addResidencySet(&set);
        println!("residency set: {} allocations", set.allocationCount());
        Some(set)
    } else {
        None
    };

    // ---- timed sweeps, interleaved ------------------------------------------
    let mut series: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
    let order: Vec<usize> = (0..arms.len()).collect();
    for rot in 0..args.rotations {
        // Rotate the starting arm so no arm always runs first (§3.4a finding 3's
        // positional bias).
        let start = rot % arms.len().max(1);
        for &ai in order[start..].iter().chain(&order[..start]) {
            let arm = &arms[ai];
            for it in 0..(args.warmup + args.iters) {
                let cb = probe.queue.commandBuffer().context("cb")?;
                let d = probe.encode_sweep(&cb, arm, None)?;
                cb.commit();
                cb.waitUntilCompleted();
                if it < args.warmup {
                    continue;
                }
                let s = cb.GPUEndTime() - cb.GPUStartTime();
                ensure!(s > 0.0, "{}: no GPU interval", arm.label);
                let gbs = probe.sweep_bytes(arm) as f64 / s / 1e9;
                if args.raw {
                    println!(
                        "SWEEP arm={} rot={rot} it={} dispatches={d} gpu_ms={:.4} gbs={gbs:.2}",
                        arm.label,
                        it - args.warmup,
                        s * 1e3
                    );
                }
                series.entry(arm.label.clone()).or_default().push(gbs);
            }
        }
        for arm in &pd_arms {
            per_dispatch(&probe, arm, rot)?;
        }
    }

    println!(
        "\n{:<22} {:>4} {:>9} {:>9} {:>9} {:>9} {:>10}",
        "arm", "n", "p10", "median", "p90", "best", "median ms"
    );
    for arm in &arms {
        let mut xs = series.remove(&arm.label).unwrap_or_default();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let bytes = probe.sweep_bytes(arm) as f64;
        println!(
            "{:<22} {:>4} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>10.3}",
            arm.label,
            xs.len(),
            q(&xs, 0.1),
            q(&xs, 0.5),
            q(&xs, 0.9),
            xs.last().copied().unwrap_or(f64::NAN),
            bytes / q(&xs, 0.5) / 1e6
        );
    }
    if let Some(set) = residency {
        set.removeAllAllocations();
    }
    Ok(())
}

/// One sweep with one counter-sampled encoder per dispatch: every matrix's
/// interval, and the checks that make the numbers readable -- each sample is a
/// value (not the error marker, not the idle zero), falls inside its command
/// buffer's interval, and the encoders sum to that interval within the seams.
fn per_dispatch(probe: &Probe, arm: &Arm, rot: usize) -> Result<()> {
    let set = probe
        .counter_set
        .as_ref()
        .context("no timestamp counter set")?;
    let n = match arm.kind {
        Kind::CopySingle { .. } => 1,
        _ => probe.mats.len(),
    };
    let desc = MTLCounterSampleBufferDescriptor::new();
    desc.setCounterSet(Some(set));
    desc.setStorageMode(MTLStorageMode::Shared);
    unsafe { desc.setSampleCount(2 * n) };
    let sb = probe
        .device
        .newCounterSampleBufferWithDescriptor_error(&desc)
        .map_err(|e| anyhow::anyhow!("counter sample buffer of {}: {e}", 2 * n))?;
    let cb = probe.queue.commandBuffer().context("cb")?;
    probe.encode_sweep(&cb, arm, Some(&sb))?;
    cb.commit();
    cb.waitUntilCompleted();
    let data = unsafe { sb.resolveCounterRange(NSRange::new(0, 2 * n)) }.context("resolve")?;
    let raw: Vec<u64> = data
        .to_vec()
        .chunks_exact(8)
        .map(|c| u64::from_ne_bytes(c.try_into().unwrap()))
        .collect();
    ensure!(
        raw.len() == 2 * n,
        "resolved {} samples, want {}",
        raw.len(),
        2 * n
    );
    let bad = raw.iter().filter(|&&t| t == 0 || t == u64::MAX).count();
    #[repr(C)]
    struct Timebase {
        numer: u32,
        denom: u32,
    }
    extern "C" {
        fn mach_timebase_info(info: *mut Timebase) -> i32;
    }
    let mut tb = Timebase { numer: 0, denom: 0 };
    // SAFETY: writes one `mach_timebase_info_data_t` (two u32s) through a
    // valid out-pointer.
    unsafe { mach_timebase_info(&mut tb) };
    let tick = tb.numer as f64 / tb.denom as f64 * 1e-9;
    let (cb0, cb1) = (cb.GPUStartTime(), cb.GPUEndTime());
    // The sample's unit, read from the data as `profile::sample_unit_s` does:
    // the candidate that puts the first sample on the buffer's start.
    let ratio = cb0 / raw[0] as f64;
    let unit = [1e-9, tick]
        .into_iter()
        .find(|u| (ratio / u - 1.0).abs() < 1e-3)
        .with_context(|| {
            format!("sample unit: cb_start/sample = {ratio:e} is neither 1 ns nor a tick")
        })?;
    let to_s = |t: u64| t as f64 * unit;
    let mut sum = 0.0;
    let mut outside = 0usize;
    let mut first = f64::MAX;
    let mut last = 0f64;
    for i in 0..n {
        let (s, e) = (to_s(raw[2 * i]), to_s(raw[2 * i + 1]));
        if s < cb0 - 1e-6 || e > cb1 + 1e-6 || e < s {
            outside += 1;
        }
        first = first.min(s);
        last = last.max(e);
        sum += e - s;
        let (name, nn, kk, bytes) = match arm.kind {
            Kind::CopySingle { .. } => ("single", 0, 0, probe.sweep_bytes(arm)),
            _ => {
                let m = probe.mats[i];
                (m.name, m.n, m.k, m.bytes())
            }
        };
        println!(
            "DISPATCH arm={} rot={rot} idx={i} name={name} n={nn} k={kk} bytes={bytes} us={:.3} gbs={:.1} start_off_us={:.3}",
            arm.label,
            (e - s) * 1e6,
            bytes as f64 / (e - s) / 1e9,
            (s - cb0) * 1e6
        );
    }
    println!(
        "PERDISPATCH arm={} rot={rot} encoders={n} cb_ms={:.4} sum_encoders_ms={:.4} first_to_last_ms={:.4} \
         bad_samples={bad} outside_cb={outside} lead_us={:.2} tail_us={:.2} sample_unit_ns={:.4}",
        arm.label,
        (cb1 - cb0) * 1e3,
        sum * 1e3,
        (last - first) * 1e3,
        (first - cb0) * 1e6,
        (cb1 - last) * 1e6,
        unit * 1e9
    );
    Ok(())
}
