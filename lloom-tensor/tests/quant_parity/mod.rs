//! The quantized-kernel parity matrix: every (format × entry point ×
//! activation dtype) the Metal backend instantiates, against the CPU dequant
//! reference, at geometries that cross each kernel's tiles and blocks.
//!
//! The cells are read from the COMPILED libraries (`Library::function_names`),
//! never from a list kept here: a format or an entry gaining an instantiation
//! gains a cell, and an instantiation no production dispatch reaches must be
//! named in [`UNREACHED`] with its reason or the matrix fails (§3.1c).
//!
//! Each cell runs through the dispatch production runs (`QTensor`'s matmul,
//! `indexed_moe_forward*`, `embedding`, the KV kernels' wrappers), and the
//! dispatch census must show the cell's own kernel ran: a parity check another
//! kernel passes is not a check of this one.
//!
//! The reference is the CPU `to_float` of the same bytes. A matmul cell's
//! error is `|got - Σ x·w| / Σ |x|·M`, `M` the largest `|w|` of the quant block
//! the product's weight sits in: the scale every block-factored core's
//! rounding is proportional to. The operands are the ones the entry declares
//! (`w` rounded to half where the kernel stages it as half), the sum is f64,
//! and an f16 store is credited the half ulp its rounding costs. The
//! elementwise entries compare in ulps. [`tolerance`] states each bound and
//! why.
//!
//! Shared by `quant_parity_matrix` (default tuning) and
//! `quant_parity_matrix_rows` (the rows arm, a process-wide switch).
#![allow(dead_code)]

use half::f16;
use lloom_tensor::kernels::source::Source;
use lloom_tensor::kernels::{KvQuantBits, SdpaDType, SdpaQuantParams, SdpaQuantWriteParams};
use lloom_tensor::quantized::{GgmlDType, MoeArm, QStorage, QTensor};
use lloom_tensor::{DType, Device, Result, Tensor};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

/// Every `GgmlDType`. [`stem`] matches exhaustively, so a new variant does
/// not compile until it is named there and here.
pub const ALL: [GgmlDType; 25] = [
    GgmlDType::F32,
    GgmlDType::F16,
    GgmlDType::BF16,
    GgmlDType::Q4_0,
    GgmlDType::Q4_1,
    GgmlDType::Q5_0,
    GgmlDType::Q5_1,
    GgmlDType::Q8_0,
    GgmlDType::Q8_1,
    GgmlDType::Q2K,
    GgmlDType::Q3K,
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
    GgmlDType::Q8K,
    GgmlDType::IQ2XXS,
    GgmlDType::IQ2XS,
    GgmlDType::IQ2S,
    GgmlDType::IQ3XXS,
    GgmlDType::IQ3S,
    GgmlDType::IQ4NL,
    GgmlDType::IQ4XS,
    GgmlDType::Q2_0,
    GgmlDType::MXFP4,
    GgmlDType::FP8E4M3,
];

/// The kernel-name stem of a format.
pub fn stem(d: GgmlDType) -> &'static str {
    match d {
        GgmlDType::F32 => "f32",
        GgmlDType::F16 => "f16",
        GgmlDType::BF16 => "bf16",
        GgmlDType::Q4_0 => "q4_0",
        GgmlDType::Q4_1 => "q4_1",
        GgmlDType::Q5_0 => "q5_0",
        GgmlDType::Q5_1 => "q5_1",
        GgmlDType::Q8_0 => "q8_0",
        GgmlDType::Q8_1 => "q8_1",
        GgmlDType::Q2K => "q2_K",
        GgmlDType::Q3K => "q3_K",
        GgmlDType::Q4K => "q4_K",
        GgmlDType::Q5K => "q5_K",
        GgmlDType::Q6K => "q6_K",
        GgmlDType::Q8K => "q8_K",
        GgmlDType::IQ2XXS => "iq2_xxs",
        GgmlDType::IQ2XS => "iq2_xs",
        GgmlDType::IQ2S => "iq2_s",
        GgmlDType::IQ3XXS => "iq3_xxs",
        GgmlDType::IQ3S => "iq3_s",
        GgmlDType::IQ4NL => "iq4_nl",
        GgmlDType::IQ4XS => "iq4_xs",
        GgmlDType::Q2_0 => "q2_0",
        GgmlDType::MXFP4 => "mxfp4",
        GgmlDType::FP8E4M3 => "fp8_e4m3",
    }
}

fn from_stem(s: &str) -> Option<GgmlDType> {
    ALL.iter().copied().find(|d| stem(*d) == s)
}

/// Instantiated kernels no production dispatch selects, each with the reason.
/// The matrix fails if a name here is no longer in the library, or is
/// launched by a cell: the list cannot go stale silently.
pub const UNREACHED: &[(&str, &str)] = &[(
    "kernel_get_rows_i32",
    "an integer gather, not a quant decode",
)];

/// Name prefixes of instantiated families no dispatch selects. Empty since
/// #892 B2 dropped the unmapped routed GEMM (`kernel_mul_mm_id_<fmt>`: only
/// `kernel_mul_mm_id_mapped{,16}` was ever dispatched), the `iq1_s` /
/// `iq1_m` instantiations (no `GgmlDType`, no CPU reference) and the three
/// dense f16 variants no dispatch named — 50 kernels of library bytes and
/// compile time with no caller (#907's finding).
pub const UNREACHED_PREFIX: &[(&str, &str)] = &[];

/// Format stems instantiated with no `GgmlDType`: no tensor can carry them,
/// so no dispatch reaches them and there is no CPU reference. Empty since
/// #892 B2 (see [`UNREACHED_PREFIX`]).
pub const NO_GGML_DTYPE: &[&str] = &[];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Entry {
    MulMv,
    Rows,
    MulMvId,
    MulMvId2,
    MulMm,
    DequantMm,
    MmId32,
    MmId16,
    GetRows,
    DequantRowsF16,
    KvSdpa,
    KvSdpaSel,
    KvDequant,
}

impl Entry {
    pub fn label(self) -> &'static str {
        match self {
            Entry::MulMv => "mul_mv",
            Entry::Rows => "rows",
            Entry::MulMvId => "mul_mv_id",
            Entry::MulMvId2 => "mul_mv_id2",
            Entry::MulMm => "mul_mm",
            Entry::DequantMm => "dequant+mm",
            Entry::MmId32 => "mul_mm_id",
            Entry::MmId16 => "mul_mm_id16",
            Entry::GetRows => "get_rows",
            Entry::DequantRowsF16 => "dequant_rows_f16",
            Entry::KvSdpa => "kv_sdpa",
            Entry::KvSdpaSel => "kv_sdpa_sel",
            Entry::KvDequant => "kv_dequant",
        }
    }

    fn is_kv(self) -> bool {
        matches!(self, Entry::KvSdpa | Entry::KvSdpaSel | Entry::KvDequant)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Act {
    F32,
    F16,
    /// An f32 activation the routed f16 kernels read as the half it rounds
    /// to (`kernel_mul_mv_id{,2}_<q>_f32h_f16`): the f16 arm without the cast.
    F32H,
}

impl Act {
    fn label(self) -> &'static str {
        match self {
            Act::F32 => "f32",
            Act::F16 => "f16",
            Act::F32H => "f32h",
        }
    }

    /// The matvec kernels' triple suffix.
    fn mv_suffix(self) -> &'static str {
        match self {
            Act::F32 => "f32",
            Act::F16 => "f16_f16",
            Act::F32H => "f32h_f16",
        }
    }

    /// The activation tensor's dtype.
    fn dtype(self) -> DType {
        match self {
            Act::F32 | Act::F32H => DType::F32,
            Act::F16 => DType::F16,
        }
    }

    /// The dtype the kernel computes in (the routed calls' `act`).
    fn compute(self) -> DType {
        match self {
            Act::F32 => DType::F32,
            Act::F16 | Act::F32H => DType::F16,
        }
    }

    fn stores_f16(self) -> bool {
        self.compute() == DType::F16
    }
}

pub type Key = (Entry, String, Act);

/// How a cell's error is measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Unit {
    /// `|got - ref| / Σ|x|·M` (matmul entries), or `/ max|v|` (attention).
    Norm,
    /// Distance in units in the last place of the output dtype.
    Ulp,
}

#[derive(Debug)]
pub struct Cell {
    pub kernels: BTreeSet<String>,
    pub cases: usize,
    pub worst: f64,
    pub worst_at: String,
    pub unit: Unit,
    pub tol: f64,
    pub fails: Vec<String>,
    pub notes: Vec<String>,
    /// FNV-1a over the bits of every output the cell's cases produced, in
    /// case order: equal digests are equal outputs, so two builds compare
    /// cell for cell without either being the reference.
    pub digest: u64,
}

impl Cell {
    fn new(unit: Unit, tol: f64) -> Self {
        Self {
            kernels: BTreeSet::new(),
            cases: 0,
            worst: 0.0,
            worst_at: String::new(),
            unit,
            tol,
            fails: Vec::new(),
            notes: Vec::new(),
            digest: 0xcbf2_9ce4_8422_2325,
        }
    }

    fn absorb(&mut self, bits: impl IntoIterator<Item = u32>) {
        for b in bits {
            for byte in b.to_le_bytes() {
                self.digest = (self.digest ^ byte as u64).wrapping_mul(0x0100_0000_01b3);
            }
        }
    }

    fn record(&mut self, err: f64, at: &str) {
        self.cases += 1;
        let worse = err.is_nan() || err > self.worst || self.worst_at.is_empty();
        if worse && !self.worst.is_nan() {
            self.worst = err;
            self.worst_at = at.to_string();
        }
        if err.is_nan() || err > self.tol {
            self.fails
                .push(format!("{at}: error {err:e} > {:e}", self.tol));
        }
    }

    pub fn passed(&self) -> bool {
        self.fails.is_empty() && self.cases > 0
    }
}

/// The bound a cell's worst error must stay under, and why.
///
/// **Elementwise entries are bit-identical** to the CPU arithmetic they
/// mirror: `get_rows` and `dequant_rows_f16` against `to_float` (rounded to
/// half for the latter), `kv_dequant` against `fma(code, scale, bias)`, the
/// form the kernel's `code * scale + bias` contracts to.
///
/// **The reductions get one measured bound, [`TOL_NORM`].** They accumulate
/// in f32 over `K` products, so the CPU reference (f64) cannot be bitwise;
/// Higham's bound for any summation order is `(K-1)·u` in this normalisation
/// (`u = 2⁻²⁴`, `7.6e-5` at the largest `K` here) and the measured worst over
/// every reduction cell is `2.3e-7` (3.9 `u`): a dense GEMM cell, the f32 and
/// f16 weights staged as half. `TOL_NORM` is four times that, rounded up to a
/// power of two. The inputs are fixed, so a cell's error is the same number
/// on every run. One wrong weight in a row moves a cell by about `1/K`
/// (`7.8e-4` at the largest `K`): 800 times the bound.
///
/// **And bit-identity where two entries run the same core in the same
/// order**, checked beside the reference: a routed plane against the plain
/// matmul-vec of its expert, and the gate|up pair against its two single
/// launches.
pub fn tolerance(entry: Entry, _act: Act) -> (Unit, f64, &'static str) {
    match entry {
        Entry::MulMv | Entry::Rows | Entry::MulMvId | Entry::MulMvId2 => (
            Unit::Norm,
            TOL_NORM,
            "f32 accumulate of the dequantized products",
        ),
        Entry::MulMm | Entry::DequantMm | Entry::MmId32 | Entry::MmId16 => (
            Unit::Norm,
            TOL_NORM,
            "f32 accumulate, the weight staged as half (bf16: as bfloat)",
        ),
        Entry::KvSdpa | Entry::KvSdpaSel => (
            Unit::Norm,
            TOL_NORM,
            "online softmax with fast::exp, against an f64 softmax",
        ),
        Entry::GetRows | Entry::DequantRowsF16 | Entry::KvDequant => (
            Unit::Ulp,
            0.0,
            "elementwise: the reference's own arithmetic",
        ),
    }
}

/// `2⁻²⁰`: see [`tolerance`].
pub const TOL_NORM: f64 = 9.5367431640625e-7;

/// The decode-family kernels the libraries instantiate, parsed into cells.
pub struct Instantiated {
    /// kernel name → its cell.
    pub cells: BTreeMap<String, Key>,
    /// Decode-family names that parse to no cell: they must be in
    /// [`UNREACHED`] / [`UNREACHED_PREFIX`].
    pub unparsed: BTreeSet<String>,
    /// Every function in the quantized library (the non-vacuity count).
    pub quantized_total: usize,
    pub sdpa_total: usize,
}

fn act_suffix(rest: &str) -> Option<(&str, Act)> {
    if let Some(s) = rest.strip_suffix("_f32h_f16") {
        return Some((s, Act::F32H));
    }
    if let Some(s) = rest.strip_suffix("_f16_f16") {
        return Some((s, Act::F16));
    }
    rest.strip_suffix("_f32").map(|s| (s, Act::F32))
}

fn is_quant_decode_family(name: &str) -> bool {
    [
        "kernel_mul_mv_",
        "kernel_mul_mm_",
        "kernel_get_rows_",
        "kernel_dequant_rows_",
    ]
    .iter()
    .any(|p| name.starts_with(p))
}

fn is_kv_read_family(name: &str) -> bool {
    name.starts_with("sdpa_vector_quant") || name.starts_with("dequantize_kv_")
}

/// One quantized-library name as its cell.
pub fn parse_quantized(name: &str) -> Option<Key> {
    let cell = |e: Entry, s: &str, a: Act| Some((e, s.to_string(), a));
    if let Some(r) = name.strip_prefix("kernel_mul_mv_id2_") {
        let (s, a) = act_suffix(r)?;
        return cell(Entry::MulMvId2, s, a);
    }
    if let Some(r) = name.strip_prefix("kernel_mul_mv_id_") {
        let (s, a) = act_suffix(r)?;
        return cell(Entry::MulMvId, s, a);
    }
    if let Some(r) = name.strip_prefix("kernel_mul_mv_") {
        for nr in ["_f32_r2", "_f32_r4"] {
            if let Some(s) = r.strip_suffix(nr) {
                return cell(Entry::Rows, s, Act::F32);
            }
        }
        let (s, a) = act_suffix(r)?;
        return cell(Entry::MulMv, s, a);
    }
    if let Some(r) = name.strip_prefix("kernel_mul_mm_id_mapped16_") {
        return cell(Entry::MmId16, r.strip_suffix("_f32")?, Act::F32);
    }
    if let Some(r) = name.strip_prefix("kernel_mul_mm_id_mapped_") {
        return cell(Entry::MmId32, r.strip_suffix("_f32")?, Act::F32);
    }
    if name.starts_with("kernel_mul_mm_id_") {
        return None;
    }
    if let Some(r) = name.strip_prefix("kernel_mul_mm_") {
        return cell(Entry::MulMm, r.strip_suffix("_f32")?, Act::F32);
    }
    if let Some(r) = name.strip_prefix("kernel_get_rows_") {
        return cell(Entry::GetRows, r, Act::F32);
    }
    if let Some(r) = name.strip_prefix("kernel_dequant_rows_f16_") {
        return cell(Entry::DequantRowsF16, r, Act::F16);
    }
    None
}

/// One SDPA-library name as its KV-read cell: the format is `int{bits}_d{D}`,
/// the dtype the planes' and the queries'.
pub fn parse_kv(name: &str) -> Option<Key> {
    let (entry, rest) = if let Some(r) = name.strip_prefix("sdpa_vector_quant_sel_") {
        (Entry::KvSdpaSel, r)
    } else if let Some(r) = name.strip_prefix("sdpa_vector_quant_") {
        (Entry::KvSdpa, r)
    } else if let Some(r) = name.strip_prefix("dequantize_kv_") {
        (Entry::KvDequant, r)
    } else {
        return None;
    };
    let mut parts = rest.rsplitn(3, '_');
    let bits = parts.next()?;
    let d = parts.next()?;
    let act = match parts.next()? {
        "float16_t" => Act::F16,
        "float" => Act::F32,
        _ => return None,
    };
    Some((entry, format!("int{bits}_d{d}"), act))
}

pub fn instantiated(dev: &Device) -> Result<Instantiated> {
    let md = dev.as_metal_device()?;
    let names = |src: Source| -> Result<Vec<String>> {
        Ok(md
            .kernels()
            .load_library(md.device(), src)
            .map_err(|e| lloom_tensor::Error::Msg(format!("{src:?}: {e}")))?
            .function_names())
    };
    let quantized = names(Source::Quantized)?;
    let sdpa = names(Source::Sdpa)?;
    let mut cells = BTreeMap::new();
    let mut unparsed = BTreeSet::new();
    for n in quantized.iter().filter(|n| is_quant_decode_family(n)) {
        match parse_quantized(n) {
            Some(k) => {
                cells.insert(n.clone(), k);
            }
            None => {
                unparsed.insert(n.clone());
            }
        }
    }
    for n in sdpa.iter().filter(|n| is_kv_read_family(n)) {
        match parse_kv(n) {
            Some(k) => {
                cells.insert(n.clone(), k);
            }
            None => {
                unparsed.insert(n.clone());
            }
        }
    }
    Ok(Instantiated {
        cells,
        unparsed,
        quantized_total: quantized.len(),
        sdpa_total: sdpa.len(),
    })
}

// ---------------------------------------------------------------- fixtures

struct Lcg(u32);

impl Lcg {
    fn new(seed: u32) -> Self {
        Self(seed.wrapping_mul(2654435761).wrapping_add(12345))
    }
    fn byte(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 24) as u8
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 8) as f32 / (1u32 << 24) as f32
    }
    fn sym(&mut self) -> f32 {
        2.0 * self.unit() - 1.0
    }
}

/// A weight of `shape` (`[n, k]` or `[e, n, k]`) on the device, with the CPU
/// dequant of the same bytes as the reference.
pub struct Weight {
    pub q: QTensor,
    pub raw: Vec<u8>,
    pub dtype: GgmlDType,
    pub k: usize,
    /// The CPU `to_float`, row-major.
    pub w: Vec<f32>,
    /// `w` rounded to half (the GEMM kernels stage weights as half).
    pub w16: Vec<f32>,
    /// Per element, the largest `|w|` of its quant block (and of `w16`'s).
    pub m: Vec<f32>,
    pub m16: Vec<f32>,
}

fn block_max(w: &[f32], block: usize) -> Vec<f32> {
    let mut m = vec![0f32; w.len()];
    for (wc, mc) in w.chunks(block).zip(m.chunks_mut(block)) {
        let mx = wc.iter().fold(0f32, |a, v| a.max(v.abs()));
        mc.fill(mx);
    }
    m
}

/// The packed bytes of `elems` weights of `d`. Formats with a quantizer get
/// quantized from values whose magnitude varies block to block (every code
/// level used); the read-only ones (IQ, Q2_0) are LCG bytes, every pattern a
/// valid block, with a sane f16 scale at offset 0 of each block.
fn raw_bytes(d: GgmlDType, elems: usize, seed: u32, mag: f32) -> Result<Vec<u8>> {
    let mut rng = Lcg::new(seed);
    if d.is_read_only() {
        let (bs, ts) = (d.block_size(), d.type_size());
        let n_blocks = elems / bs;
        let mut raw: Vec<u8> = (0..n_blocks * ts).map(|_| rng.byte()).collect();
        for i in 0..n_blocks {
            let s = f16::from_f32(mag * (0.004 + 0.03 * rng.unit()));
            raw[i * ts..i * ts + 2].copy_from_slice(&s.to_le_bytes());
        }
        return Ok(raw);
    }
    let block = d.block_size().max(32);
    let mut v = Vec::with_capacity(elems);
    while v.len() < elems {
        let scale = mag * (0.1 + 0.9 * rng.unit());
        for _ in 0..block.min(elems - v.len()) {
            v.push(scale * rng.sym());
        }
    }
    let t = Tensor::from_vec(v, elems, &Device::Cpu)?;
    Ok(QTensor::quantize(&t, d)?.data()?.into_owned())
}

pub fn weight(dev: &Device, d: GgmlDType, shape: &[usize], seed: u32, mag: f32) -> Result<Weight> {
    let elems: usize = shape.iter().product();
    let k = *shape.last().expect("a weight has a last dim");
    let raw = raw_bytes(d, elems, seed, mag)?;
    let q = QTensor::new(QStorage::from_data(Cow::Borrowed(&raw), dev, d)?, shape)?;
    let cpu = QTensor::new(
        QStorage::from_data(Cow::Borrowed(&raw), &Device::Cpu, d)?,
        shape,
    )?;
    let w = cpu
        .dequantize(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let w16: Vec<f32> = w.iter().map(|v| f16::from_f32(*v).to_f32()).collect();
    let block = d.block_size();
    Ok(Weight {
        q,
        dtype: d,
        k,
        m: block_max(&w, block),
        m16: block_max(&w16, block),
        w,
        w16,
        raw,
    })
}

/// Activations of `shape` at `act`, and the values the kernel reads.
pub fn activations(
    dev: &Device,
    shape: &[usize],
    act: Act,
    seed: u32,
) -> Result<(Tensor, Vec<f32>)> {
    let n: usize = shape.iter().product();
    let mut rng = Lcg::new(seed ^ 0x5bd1_e995);
    let v: Vec<f32> = (0..n).map(|_| rng.sym()).collect();
    match act {
        Act::F32 => Ok((Tensor::from_vec(v.clone(), shape, dev)?, v)),
        // The kernel rounds each element to half as it loads it.
        Act::F32H => {
            let r = v.iter().map(|x| f16::from_f32(*x).to_f32()).collect();
            Ok((Tensor::from_vec(v, shape, dev)?, r))
        }
        Act::F16 => {
            let h: Vec<f16> = v.iter().map(|x| f16::from_f32(*x)).collect();
            let r = h.iter().map(|x| x.to_f32()).collect();
            Ok((Tensor::from_vec(h, shape, dev)?, r))
        }
    }
}

fn to_f32(t: &Tensor) -> Result<Vec<f32>> {
    t.to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
}

fn bits_of(t: &Tensor) -> Result<Vec<u32>> {
    Ok(to_f32(t)?.iter().map(|v| v.to_bits()).collect())
}

// ---------------------------------------------------------------- metrics

fn ulp16(v: f64) -> f64 {
    let v = v.abs();
    if v < 2f64.powi(-14) {
        2f64.powi(-24)
    } else {
        2f64.powi(v.log2().floor() as i32 - 10)
    }
}

/// `|got - r| / a`, less the half ulp an f16 store's rounding costs.
fn norm_err(got: f32, r: f64, a: f64, f16_store: bool) -> f64 {
    if !got.is_finite() {
        return f64::INFINITY;
    }
    let mut d = (got as f64 - r).abs();
    if f16_store {
        d = (d - 0.5 * ulp16(r.abs().max(got.abs() as f64))).max(0.0);
    }
    if d == 0.0 {
        0.0
    } else if a == 0.0 {
        f64::INFINITY
    } else {
        d / a
    }
}

/// Whether the GEMM kernels stage this format's weight tile as half: every
/// `kernel_mul_mm*` instantiation does (`T = half`), except bf16's, which
/// stages `bfloat` and so holds the weight exactly.
fn stages_half(d: GgmlDType) -> bool {
    d != GgmlDType::BF16
}

/// `Σ x·w` and `Σ |x|·m`, in f64.
fn dot(x: &[f32], w: &[f32], m: &[f32]) -> (f64, f64) {
    let (mut s, mut a) = (0f64, 0f64);
    for i in 0..x.len() {
        s += x[i] as f64 * w[i] as f64;
        a += (x[i] as f64).abs() * m[i] as f64;
    }
    (s, a)
}

fn ordered32(v: f32) -> i64 {
    let b = v.to_bits() as i32;
    (if b < 0 { i32::MIN - b } else { b }) as i64
}

fn ordered16(v: f16) -> i64 {
    let b = v.to_bits() as i16;
    (if b < 0 { i16::MIN - b } else { b }) as i64
}

fn ulps32(a: f32, b: f32) -> f64 {
    if !a.is_finite() || !b.is_finite() {
        return if a.to_bits() == b.to_bits() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    (ordered32(a) - ordered32(b)).abs() as f64
}

fn ulps16(a: f16, b: f16) -> f64 {
    if !a.is_finite() || !b.is_finite() {
        return if a.to_bits() == b.to_bits() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    (ordered16(a) - ordered16(b)).abs() as f64
}

// ---------------------------------------------------------------- the matrix

pub struct Matrix {
    pub dev: Device,
    pub lib: Instantiated,
    pub cells: BTreeMap<Key, Cell>,
    pub launched: BTreeSet<String>,
    /// Dispatches of a (format, entry, dtype) with no instantiation: what the
    /// production entry did with it.
    pub refusals: Vec<String>,
    pub problems: Vec<String>,
}

type Census = Vec<(String, u64)>;

/// One K or V plane set: codes, scales and biases on the device, then the
/// same three as the host reads them.
type KvPlanes = (Tensor, Tensor, Tensor, Vec<u8>, Vec<f32>, Vec<f32>);

impl Matrix {
    pub fn new(dev: Device) -> Result<Self> {
        assert!(
            lloom_tensor::kernels::metal::profile::enabled(),
            "the dispatch census decides which kernel each cell ran"
        );
        let lib = instantiated(&dev)?;
        Ok(Self {
            dev,
            lib,
            cells: BTreeMap::new(),
            launched: BTreeSet::new(),
            refusals: Vec::new(),
            problems: Vec::new(),
        })
    }

    fn cell(&mut self, key: &Key) -> &mut Cell {
        let (unit, tol, _) = tolerance(key.0, key.2);
        self.cells
            .entry(key.clone())
            .or_insert_with(|| Cell::new(unit, tol))
    }

    /// Run `f` with the device's census window open; the labels it counted.
    fn census<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<(T, Census)> {
        self.dev.synchronize()?;
        let window = self
            .dev
            .as_metal_device()?
            .dispatch_census()
            .open()
            .map_err(|e| lloom_tensor::Error::Msg(format!("census: {e:?}")))?;
        let r = f();
        self.dev.synchronize()?;
        let c = window.close();
        Ok((r?, c))
    }

    /// `expect` all ran, and no other decode-family kernel did.
    fn check_census(&mut self, key: &Key, at: &str, census: &Census, expect: &[String]) {
        let ran: BTreeSet<&str> = census.iter().map(|(l, _)| l.as_str()).collect();
        for l in &ran {
            if is_quant_decode_family(l)
                || is_kv_read_family(l)
                || l.starts_with("kernel_moe_rowmap")
            {
                self.launched.insert(l.to_string());
            }
        }
        let missing: Vec<&String> = expect
            .iter()
            .filter(|e| !ran.contains(e.as_str()))
            .collect();
        let stray: Vec<&&str> = ran
            .iter()
            .filter(|l| {
                (is_quant_decode_family(l) || is_kv_read_family(l))
                    && !expect.iter().any(|e| e == **l)
            })
            .collect();
        let cell = self.cell(key);
        for e in expect {
            cell.kernels.insert(e.clone());
        }
        if !missing.is_empty() || !stray.is_empty() {
            cell.fails.push(format!(
                "{at}: census missing {missing:?}, stray {stray:?} (ran {census:?})"
            ));
        }
    }

    fn note_repeat(&mut self, key: &Key, at: &str, a: &Tensor, b: &Tensor) -> Result<()> {
        if bits_of(a)? != bits_of(b)? {
            self.cell(key).fails.push(format!("{at}: two runs differ"));
        }
        Ok(())
    }

    // ------------------------------------------------------------ matmul family

    /// Score every output of `[rows, n]` against `x [rows, k] · w[sel(r)]ᵀ`.
    #[allow(clippy::too_many_arguments)]
    fn score_rows(
        &mut self,
        key: &Key,
        at: &str,
        got: &[f32],
        x: &[f32],
        x_row: impl Fn(usize) -> usize,
        w: &Weight,
        w_plane: impl Fn(usize) -> usize,
        rows: usize,
        n: usize,
        half_weight: bool,
        f16_store: bool,
    ) {
        let k = w.k;
        let (ws, ms) = if half_weight {
            (&w.w16, &w.m16)
        } else {
            (&w.w, &w.m)
        };
        let mut worst = 0f64;
        let mut where_ = String::new();
        for r in 0..rows {
            let xr = &x[x_row(r) * k..(x_row(r) + 1) * k];
            let base = w_plane(r) * n * k;
            for j in 0..n {
                let wr = &ws[base + j * k..base + (j + 1) * k];
                let mr = &ms[base + j * k..base + (j + 1) * k];
                let (s, a) = dot(xr, wr, mr);
                let e = norm_err(got[r * n + j], s, a, f16_store);
                if e.is_nan() || e > worst || (e.is_infinite() && worst.is_finite()) {
                    worst = e;
                    where_ = format!("{at} row={r} col={j} got={} ref={s:e}", got[r * n + j]);
                }
            }
        }
        self.cell(key).record(worst, &where_);
    }

    /// The weight magnitudes: unit-scale blocks, and blocks 2⁻⁸ smaller, where
    /// a K-quant's super-block scale `d` sits at or below half's normal range
    /// (6.1e-5) as real checkpoints' do: an arithmetic step a decoder does in
    /// half loses bits there and nowhere else.
    pub const MAGS: [f32; 2] = [1.0, 1.0 / 256.0];

    /// Every `(K, magnitude)` a weight is built at.
    fn kmags(d: GgmlDType) -> Vec<(usize, f32)> {
        Self::ks(d)
            .into_iter()
            .flat_map(|k| Self::MAGS.into_iter().map(move |m| (k, m)))
            .collect()
    }

    fn ks(d: GgmlDType) -> [usize; 2] {
        match d.block_size() {
            1 | 32 => [32, 32 * 37],
            64 => [64, 64 * 13],
            256 => [256, 256 * 5],
            b => panic!("block size {b}"),
        }
    }

    pub fn run_mul_mv(&mut self, d: GgmlDType, act: Act, rows_arm: bool) -> Result<()> {
        let entry = if rows_arm { Entry::Rows } else { Entry::MulMv };
        let key: Key = (entry, stem(d).into(), act);
        let ms: &[usize] = if rows_arm {
            &[2, 3, 4, 5, 8]
        } else {
            &[1, 2, 3, 8]
        };
        let mut first = true;
        for (k, mag) in Self::kmags(d) {
            for n in [1usize, 2, 37, 69] {
                let w = weight(&self.dev, d, &[n, k], 7 + n as u32 + k as u32, mag)?;
                for &m in ms {
                    let at = format!("m={m} n={n} k={k} mag={mag}");
                    let (x, xv) = activations(&self.dev, &[m, k], act, m as u32 * 31 + k as u32)?;
                    let (out, census) = self.census(|| x.apply_op1_no_bwd(&w.q))?;
                    let expect = if rows_arm {
                        let nr = if m <= 2 { 2 } else { 4 };
                        format!("kernel_mul_mv_{}_f32_r{nr}", stem(d))
                    } else {
                        format!("kernel_mul_mv_{}_{}", stem(d), act.mv_suffix())
                    };
                    self.check_census(&key, &at, &census, &[expect]);
                    let got = to_f32(&out)?;
                    self.cell(&key).absorb(got.iter().map(|v| v.to_bits()));
                    self.score_rows(
                        &key,
                        &at,
                        &got,
                        &xv,
                        |r| r,
                        &w,
                        |_| 0,
                        m,
                        n,
                        false,
                        act == Act::F16,
                    );
                    if first {
                        let again = x.apply_op1_no_bwd(&w.q)?;
                        self.note_repeat(&key, &at, &out, &again)?;
                        first = false;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn run_mul_mm(&mut self, d: GgmlDType, dequant_once: bool) -> Result<()> {
        let entry = if dequant_once {
            Entry::DequantMm
        } else {
            Entry::MulMm
        };
        let key: Key = (entry, stem(d).into(), Act::F32);
        let ms: &[usize] = if dequant_once { &[64, 70] } else { &[9, 37] };
        let mut first = true;
        for (k, mag) in Self::kmags(d) {
            for n in [1usize, 37, 69] {
                let w = weight(&self.dev, d, &[n, k], 11 + n as u32 + k as u32, mag)?;
                for &m in ms {
                    let at = format!("m={m} n={n} k={k} mag={mag}");
                    let (x, xv) =
                        activations(&self.dev, &[m, k], Act::F32, m as u32 * 7 + k as u32)?;
                    let (out, census) = self.census(|| x.apply_op1_no_bwd(&w.q))?;
                    let expect = if dequant_once {
                        vec![
                            format!("kernel_dequant_rows_f16_{}", stem(d)),
                            "kernel_mul_mm_f16_f32".to_string(),
                        ]
                    } else {
                        vec![format!("kernel_mul_mm_{}_f32", stem(d))]
                    };
                    self.check_census(&key, &at, &census, &expect);
                    let got = to_f32(&out)?;
                    self.cell(&key).absorb(got.iter().map(|v| v.to_bits()));
                    self.score_rows(
                        &key,
                        &at,
                        &got,
                        &xv,
                        |r| r,
                        &w,
                        |_| 0,
                        m,
                        n,
                        stages_half(d),
                        false,
                    );
                    if first {
                        let again = x.apply_op1_no_bwd(&w.q)?;
                        self.note_repeat(&key, &at, &out, &again)?;
                        first = false;
                    }
                }
            }
        }
        Ok(())
    }

    /// The routed launches: `ids` repeat an expert across slots and across
    /// the batch, and name the last expert, so a plane read past its expert
    /// or a store past its plane lands on a checked output.
    fn routed_ids(batch: usize, topk: usize, e: usize) -> Vec<u32> {
        (0..batch * topk)
            .map(|i| match i % 5 {
                0 => (e - 1) as u32,
                1 => 0,
                2 => (e - 1) as u32,
                3 => 2,
                _ => ((i * 3) % e) as u32,
            })
            .collect()
    }

    pub fn run_mv_id(&mut self, d: GgmlDType, act: Act, pair: bool) -> Result<()> {
        let entry = if pair {
            Entry::MulMvId2
        } else {
            Entry::MulMvId
        };
        let key: Key = (entry, stem(d).into(), act);
        let (e, topk) = (5usize, 2usize);
        let mut first = true;
        for (k, mag) in Self::kmags(d) {
            for n in [1usize, 2, 37, 69] {
                let g = weight(&self.dev, d, &[e, n, k], 3 + n as u32 + k as u32, mag)?;
                let u = if pair {
                    Some(weight(
                        &self.dev,
                        d,
                        &[e, n, k],
                        1000 + n as u32 + k as u32,
                        mag,
                    )?)
                } else {
                    None
                };
                for batch in [1usize, 3] {
                    for in1 in [1usize, topk] {
                        let at = format!("batch={batch} in_dim1={in1} n={n} k={k} mag={mag}");
                        let ids = Self::routed_ids(batch, topk, e);
                        let ids_t = Tensor::from_vec(ids.clone(), (batch, topk), &self.dev)?;
                        let (x, xv) = activations(
                            &self.dev,
                            &[batch, in1, k],
                            act,
                            batch as u32 * 13 + in1 as u32 + k as u32,
                        )?;
                        let x_row = |r: usize| if in1 == 1 { r / topk } else { r };
                        let plane = |r: usize| ids[r] as usize;
                        if let Some(u) = &u {
                            let ((go, uo), census) = self.census(|| {
                                g.q.indexed_moe_forward_pair_as(
                                    &u.q,
                                    &x,
                                    &ids_t,
                                    MoeArm::MatVec,
                                    act.compute(),
                                )
                            })?;
                            let expect =
                                format!("kernel_mul_mv_id2_{}_{}", stem(d), act.mv_suffix());
                            self.check_census(&key, &at, &census, &[expect]);
                            let gv = to_f32(&go)?;
                            let uv = to_f32(&uo)?;
                            self.cell(&key).absorb(gv.iter().chain(&uv).map(|v| v.to_bits()));
                            self.score_rows(
                                &key,
                                &format!("{at} gate"),
                                &gv,
                                &xv,
                                x_row,
                                &g,
                                plane,
                                batch * topk,
                                n,
                                false,
                                act.stores_f16(),
                            );
                            self.score_rows(
                                &key,
                                &format!("{at} up"),
                                &uv,
                                &xv,
                                x_row,
                                u,
                                plane,
                                batch * topk,
                                n,
                                false,
                                act.stores_f16(),
                            );
                            // The same core in the same order: the pair is the two
                            // single launches bit for bit.
                            let g1 = g.q.indexed_moe_forward_as(
                                &x,
                                &ids_t,
                                MoeArm::MatVec,
                                act.compute(),
                            )?;
                            let u1 = u.q.indexed_moe_forward_as(
                                &x,
                                &ids_t,
                                MoeArm::MatVec,
                                act.compute(),
                            )?;
                            if bits_of(&go)? != bits_of(&g1)? || bits_of(&uo)? != bits_of(&u1)? {
                                self.cell(&key).fails.push(format!(
                                    "{at}: the pair is not the two single launches' bits"
                                ));
                            }
                            if first {
                                let (g2, _) = g.q.indexed_moe_forward_pair_as(
                                    &u.q,
                                    &x,
                                    &ids_t,
                                    MoeArm::MatVec,
                                    act.compute(),
                                )?;
                                self.note_repeat(&key, &at, &go, &g2)?;
                                first = false;
                            }
                        } else {
                            let (out, census) = self.census(|| {
                                g.q.indexed_moe_forward_as(
                                    &x,
                                    &ids_t,
                                    MoeArm::MatVec,
                                    act.compute(),
                                )
                            })?;
                            let expect =
                                format!("kernel_mul_mv_id_{}_{}", stem(d), act.mv_suffix());
                            self.check_census(&key, &at, &census, &[expect]);
                            let got = to_f32(&out)?;
                            self.cell(&key).absorb(got.iter().map(|v| v.to_bits()));
                            self.score_rows(
                                &key,
                                &at,
                                &got,
                                &xv,
                                x_row,
                                &g,
                                plane,
                                batch * topk,
                                n,
                                false,
                                act.stores_f16(),
                            );
                            self.mv_id_is_mul_mv(
                                &key, &at, &g, &x, &out, &ids, batch, topk, in1, n, k,
                            )?;
                            if first {
                                let again = g.q.indexed_moe_forward_as(
                                    &x,
                                    &ids_t,
                                    MoeArm::MatVec,
                                    act.compute(),
                                )?;
                                self.note_repeat(&key, &at, &out, &again)?;
                                first = false;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// A routed plane is the plain matmul-vec of its expert on its row: the
    /// same per-format core at one row, so the same bits.
    #[allow(clippy::too_many_arguments)]
    fn mv_id_is_mul_mv(
        &mut self,
        key: &Key,
        at: &str,
        w: &Weight,
        x: &Tensor,
        out: &Tensor,
        ids: &[u32],
        batch: usize,
        topk: usize,
        in1: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let d = w.dtype;
        if matches!(d, GgmlDType::F32 | GgmlDType::F16) {
            return Ok(());
        }
        let plane_bytes = n * k / d.block_size() * d.type_size();
        let mut differ = 0usize;
        for b in 0..batch {
            for t in 0..topk {
                let ex = ids[b * topk + t] as usize;
                let raw = &w.raw[ex * plane_bytes..(ex + 1) * plane_bytes];
                let one = QTensor::new(
                    QStorage::from_data(Cow::Borrowed(raw), &self.dev, d)?,
                    (n, k),
                )?;
                let xr = x.get(b)?.get(if in1 == 1 { 0 } else { t })?.unsqueeze(0)?;
                // The plain matmul-vec has no f32-read-as-half arm: its f16
                // arm over the activation cast, which the fold claims to be.
                let xr = xr.to_dtype(key.2.compute())?;
                let plain = xr.apply_op1_no_bwd(&one)?;
                if bits_of(&plain)? != bits_of(&out.get(b)?.get(t)?.unsqueeze(0)?)? {
                    differ += 1;
                }
            }
        }
        if differ > 0 {
            self.cell(key).fails.push(format!(
                "{at}: {differ} routed planes are not the plain matmul-vec's bits"
            ));
        }
        Ok(())
    }

    pub fn run_mm_id(&mut self, d: GgmlDType, bn: usize) -> Result<()> {
        let entry = if bn == 16 {
            Entry::MmId16
        } else {
            Entry::MmId32
        };
        let key: Key = (entry, stem(d).into(), Act::F32);
        lloom_tensor::quantized::metal::set_moe_mm_bn(bn);
        let (e, topk) = (5usize, 2usize);
        let mut first = true;
        for (k, mag) in Self::kmags(d) {
            for n in [1usize, 37, 69] {
                let w = weight(&self.dev, d, &[e, n, k], 5 + n as u32 + k as u32, mag)?;
                for batch in [3usize, 40] {
                    for in1 in [1usize, topk] {
                        let at = format!("batch={batch} in_dim1={in1} n={n} k={k} mag={mag}");
                        let ids = Self::routed_ids(batch, topk, e);
                        let ids_t = Tensor::from_vec(ids.clone(), (batch, topk), &self.dev)?;
                        let (x, xv) = activations(
                            &self.dev,
                            &[batch, in1, k],
                            Act::F32,
                            batch as u32 * 17 + in1 as u32 + k as u32,
                        )?;
                        let (out, census) =
                            self.census(|| w.q.indexed_moe_forward_with(&x, &ids_t, MoeArm::Gemm))?;
                        let name = if bn == 16 {
                            "kernel_mul_mm_id_mapped16"
                        } else {
                            "kernel_mul_mm_id_mapped"
                        };
                        self.check_census(&key, &at, &census, &[format!("{name}_{}_f32", stem(d))]);
                        let got = to_f32(&out)?;
                        self.cell(&key).absorb(got.iter().map(|v| v.to_bits()));
                        let x_row = |r: usize| if in1 == 1 { r / topk } else { r };
                        self.score_rows(
                            &key,
                            &at,
                            &got,
                            &xv,
                            x_row,
                            &w,
                            |r| ids[r] as usize,
                            batch * topk,
                            n,
                            stages_half(d),
                            false,
                        );
                        if first {
                            let again = w.q.indexed_moe_forward_with(&x, &ids_t, MoeArm::Gemm)?;
                            self.note_repeat(&key, &at, &out, &again)?;
                            first = false;
                        }
                    }
                }
            }
        }
        lloom_tensor::quantized::metal::set_moe_mm_bn(0);
        Ok(())
    }

    // ------------------------------------------------------------ elementwise

    pub fn run_get_rows(&mut self, d: GgmlDType) -> Result<()> {
        let key: Key = (Entry::GetRows, stem(d).into(), Act::F32);
        let rows = 37usize;
        for (k, mag) in Self::kmags(d) {
            let w = weight(&self.dev, d, &[rows, k], 23 + k as u32, mag)?;
            for ids in [vec![36u32], vec![36, 0, 5, 36, 17]] {
                let at = format!("ids={ids:?} k={k} mag={mag}");
                let ids_t = Tensor::from_vec(ids.clone(), ids.len(), &self.dev)?;
                let (out, census) = self.census(|| w.q.embedding(&ids_t))?;
                self.check_census(
                    &key,
                    &at,
                    &census,
                    &[format!("kernel_get_rows_{}", stem(d))],
                );
                let got = to_f32(&out)?;
                self.cell(&key).absorb(got.iter().map(|v| v.to_bits()));
                let mut worst = 0f64;
                let mut where_ = String::new();
                for (i, &id) in ids.iter().enumerate() {
                    for j in 0..k {
                        let (g, r) = (got[i * k + j], w.w[id as usize * k + j]);
                        let u = ulps32(g, r);
                        if u > worst || where_.is_empty() {
                            worst = u;
                            where_ = format!("{at} id={id} col={j} got={g:e} ref={r:e}");
                        }
                    }
                }
                self.cell(&key).record(worst, &where_);
            }
        }
        Ok(())
    }

    pub fn run_dequant_rows_f16(&mut self, d: GgmlDType) -> Result<()> {
        let key: Key = (Entry::DequantRowsF16, stem(d).into(), Act::F16);
        for (k, mag) in Self::kmags(d) {
            for n in [1usize, 37] {
                let at = format!("n={n} k={k} mag={mag}");
                let w = weight(&self.dev, d, &[n, k], 29 + n as u32 + k as u32, mag)?;
                let out = Tensor::zeros(n * k, DType::F16, &self.dev)?;
                let (_, census) = self.census(|| {
                    let md = self.dev.as_metal_device()?;
                    let (s, _) = out.storage_and_layout();
                    let dst = match &*s {
                        lloom_tensor::Storage::Metal(m) => m.buffer(),
                        _ => unreachable!("a Metal tensor"),
                    };
                    let enc = md.command_encoder()?;
                    lloom_tensor::kernels::call_quantized_dequant_rows_f16(
                        md.device(),
                        &enc,
                        md.kernels(),
                        d.into(),
                        (n, k),
                        w.q.metal_buffer()?,
                        dst,
                    )
                    .map_err(lloom_tensor::Error::wrap)?;
                    Ok(())
                })?;
                self.check_census(
                    &key,
                    &at,
                    &census,
                    &[format!("kernel_dequant_rows_f16_{}", stem(d))],
                );
                let got = out.to_device(&Device::Cpu)?.to_vec1::<f16>()?;
                self.cell(&key).absorb(got.iter().map(|v| v.to_bits() as u32));
                let mut worst = 0f64;
                let mut where_ = String::new();
                for (i, g) in got.iter().enumerate() {
                    let r = f16::from_f32(w.w[i]);
                    let u = ulps16(*g, r);
                    if u > worst || where_.is_empty() {
                        worst = u;
                        where_ = format!("{at} at={i} got={g} ref={r} (f32 {:e})", w.w[i]);
                    }
                }
                self.cell(&key).record(worst, &where_);
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------ the KV read

    fn kv_planes(
        &self,
        bits: KvQuantBits,
        act: Act,
        heads: usize,
        cap: usize,
        dim: usize,
        seed: u32,
    ) -> Result<KvPlanes> {
        let mut rng = Lcg::new(seed);
        let codes: Vec<u8> = (0..bits.code_bytes(heads * cap * dim))
            .map(|_| rng.byte())
            .collect();
        let sc: Vec<f32> = (0..heads * cap)
            .map(|_| 0.004 + 0.03 * rng.unit())
            .collect();
        let bi: Vec<f32> = (0..heads * cap).map(|_| -0.5 * rng.unit()).collect();
        let (sc, bi) = match act {
            Act::F16 => (
                sc.iter()
                    .map(|v| f16::from_f32(*v).to_f32())
                    .collect::<Vec<_>>(),
                bi.iter()
                    .map(|v| f16::from_f32(*v).to_f32())
                    .collect::<Vec<_>>(),
            ),
            Act::F32 => (sc, bi),
            Act::F32H => unreachable!("a KV cell's dtype is f32 or f16 (`parse_kv`)"),
        };
        let plane = |v: &[f32]| -> Result<Tensor> {
            Tensor::from_vec(v.to_vec(), v.len(), &self.dev)?.to_dtype(act.dtype())
        };
        let codes_t = Tensor::from_vec(codes.clone(), codes.len(), &self.dev)?;
        Ok((codes_t, plane(&sc)?, plane(&bi)?, codes, sc, bi))
    }

    fn code(codes: &[u8], bits: KvQuantBits, idx: usize) -> f64 {
        match bits {
            KvQuantBits::Eight => codes[idx] as f64,
            KvQuantBits::Four => {
                let b = codes[idx >> 1];
                (if idx & 1 == 1 { b >> 4 } else { b & 0x0F }) as f64
            }
        }
    }

    fn kv_geometry(fmt: &str) -> (KvQuantBits, usize) {
        let rest = fmt.strip_prefix("int").expect("int{bits}_d{dim}");
        let (b, d) = rest.split_once("_d").expect("int{bits}_d{dim}");
        let bits = if b == "8" {
            KvQuantBits::Eight
        } else {
            KvQuantBits::Four
        };
        (bits, d.parse().expect("a head dim"))
    }

    fn guard(t: &Tensor) -> parking_lot::RwLockReadGuard<'_, lloom_tensor::Storage> {
        t.storage_and_layout().0
    }

    pub fn run_kv_dequant(&mut self, fmt: &str, act: Act) -> Result<()> {
        let key: Key = (Entry::KvDequant, fmt.into(), act);
        let (bits, dim) = Self::kv_geometry(fmt);
        let heads = 3usize;
        for n_tok in [1usize, 5, 37] {
            let cap = n_tok + 3;
            let at = format!("tokens={n_tok} cap={cap}");
            let (codes_t, sc_t, bi_t, codes, sc, bi) =
                self.kv_planes(bits, act, heads, cap, dim, n_tok as u32 * 7 + dim as u32)?;
            let out = Tensor::zeros(heads * n_tok * dim, act.dtype(), &self.dev)?;
            let (_, census) = self.census(|| {
                let md = self.dev.as_metal_device()?;
                let wp = SdpaQuantWriteParams::for_append(n_tok * dim, cap, dim, 0, bits);
                let params = md.new_buffer_with_data(&[wp])?;
                let c = Self::guard(&codes_t);
                let s = Self::guard(&sc_t);
                let b = Self::guard(&bi_t);
                let o = Self::guard(&out);
                let m = |g: &lloom_tensor::Storage| match g {
                    lloom_tensor::Storage::Metal(m) => m.buffer().clone(),
                    _ => unreachable!("a Metal tensor"),
                };
                let enc = md.command_encoder()?;
                lloom_tensor::kernels::call_dequantize_kv(
                    md.device(),
                    &enc,
                    md.kernels(),
                    &m(&c),
                    &m(&s),
                    &m(&b),
                    &m(&o),
                    &params,
                    heads,
                    n_tok,
                    dim,
                    bits,
                    if act == Act::F16 {
                        SdpaDType::F16
                    } else {
                        SdpaDType::F32
                    },
                )
                .map_err(lloom_tensor::Error::wrap)?;
                Ok(())
            })?;
            let t = if act == Act::F16 {
                "float16_t"
            } else {
                "float"
            };
            self.check_census(
                &key,
                &at,
                &census,
                &[format!("dequantize_kv_{t}_{dim}_{}", bits.name())],
            );
            let got = to_f32(&out)?;
            self.cell(&key).absorb(got.iter().map(|v| v.to_bits()));
            let mut worst = 0f64;
            let mut where_ = String::new();
            for h in 0..heads {
                for tk in 0..n_tok {
                    for j in 0..dim {
                        let c = Self::code(&codes, bits, h * cap * dim + tk * dim + j) as f32;
                        let r = c.mul_add(sc[h * cap + tk], bi[h * cap + tk]);
                        let g = got[h * n_tok * dim + tk * dim + j];
                        let u = match act {
                            Act::F32 => ulps32(g, r),
                            Act::F16 => ulps16(f16::from_f32(g), f16::from_f32(r)),
                            Act::F32H => {
                                unreachable!("a KV cell's dtype is f32 or f16 (`parse_kv`)")
                            }
                        };
                        if u > worst || where_.is_empty() {
                            worst = u;
                            where_ = format!("{at} head={h} tok={tk} j={j} got={g:e} ref={r:e}");
                        }
                    }
                }
            }
            self.cell(&key).record(worst, &where_);
        }
        Ok(())
    }

    pub fn run_kv_sdpa(&mut self, fmt: &str, act: Act, sel: bool) -> Result<()> {
        let entry = if sel { Entry::KvSdpaSel } else { Entry::KvSdpa };
        let key: Key = (entry, fmt.into(), act);
        let (bits, dim) = Self::kv_geometry(fmt);
        let (hq, hkv) = (4usize, 2usize);
        for n in [1usize, 37, 100] {
            let mut cap = n + 3;
            while cap % 3 == 0 {
                cap += 1;
            }
            let cells: Vec<u32> = if sel {
                (0..n).map(|i| ((i * 3 + 1) % cap) as u32).collect()
            } else {
                (0..n as u32).collect()
            };
            let at = format!("n={n} cap={cap}");
            let (kc_t, ks_t, kb_t, kc, ks, kb) =
                self.kv_planes(bits, act, hkv, cap, dim, n as u32 * 3 + dim as u32)?;
            let (vc_t, vs_t, vb_t, vc, vs, vb) =
                self.kv_planes(bits, act, hkv, cap, dim, n as u32 * 5 + dim as u32 + 1)?;
            let (q, qv) = activations(&self.dev, &[1, hq, 1, dim], act, n as u32 + dim as u32)?;
            let out = Tensor::zeros((1, hq, 1, dim), act.dtype(), &self.dev)?;
            let sel_t = Tensor::from_vec(cells.clone(), n, &self.dev)?;
            let scale = 1.0 / (dim as f32).sqrt();
            let (_, census) = self.census(|| {
                let md = self.dev.as_metal_device()?;
                let sp = SdpaQuantParams::for_step(hq / hkv, n, cap, dim, scale, 1.0, bits);
                let params = md.new_buffer_with_data(&[sp])?;
                let m = |t: &Tensor| match &*t.storage_and_layout().0 {
                    lloom_tensor::Storage::Metal(m) => m.buffer().clone(),
                    _ => unreachable!("a Metal tensor"),
                };
                let itype = if act == Act::F16 {
                    SdpaDType::F16
                } else {
                    SdpaDType::F32
                };
                let enc = md.command_encoder()?;
                let q_shape = [1usize, hq, 1, dim];
                if sel {
                    lloom_tensor::kernels::call_sdpa_vector_quant_sel(
                        md.device(),
                        &enc,
                        md.kernels(),
                        0,
                        &q_shape,
                        &m(&q),
                        &m(&kc_t),
                        &m(&ks_t),
                        &m(&kb_t),
                        &m(&vc_t),
                        &m(&vs_t),
                        &m(&vb_t),
                        &m(&out),
                        &params,
                        (&m(&sel_t), 0),
                        bits,
                        itype,
                    )
                } else {
                    lloom_tensor::kernels::call_sdpa_vector_quant(
                        md.device(),
                        &enc,
                        md.kernels(),
                        0,
                        &q_shape,
                        &m(&q),
                        &m(&kc_t),
                        &m(&ks_t),
                        &m(&kb_t),
                        &m(&vc_t),
                        &m(&vs_t),
                        &m(&vb_t),
                        &m(&out),
                        &params,
                        bits,
                        itype,
                    )
                }
                .map_err(lloom_tensor::Error::wrap)?;
                Ok(())
            })?;
            let t = if act == Act::F16 {
                "float16_t"
            } else {
                "float"
            };
            let name = if sel {
                "sdpa_vector_quant_sel"
            } else {
                "sdpa_vector_quant"
            };
            self.check_census(
                &key,
                &at,
                &census,
                &[format!("{name}_{t}_{dim}_{}", bits.name())],
            );
            let got = to_f32(&out)?;
            self.cell(&key).absorb(got.iter().map(|v| v.to_bits()));
            let mut worst = 0f64;
            let mut where_ = String::new();
            for h in 0..hq {
                let kvh = h / (hq / hkv);
                let deq = |codes: &[u8], s: &[f32], b: &[f32], c: usize, j: usize| {
                    Self::code(codes, bits, kvh * cap * dim + c * dim + j) * s[kvh * cap + c] as f64
                        + b[kvh * cap + c] as f64
                };
                let scores: Vec<f64> = cells
                    .iter()
                    .map(|&c| {
                        (0..dim)
                            .map(|j| {
                                scale as f64
                                    * qv[h * dim + j] as f64
                                    * deq(&kc, &ks, &kb, c as usize, j)
                            })
                            .sum::<f64>()
                    })
                    .collect();
                let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let p: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
                let z: f64 = p.iter().sum();
                for j in 0..dim {
                    let mut r = 0f64;
                    let mut vmax = 0f64;
                    for (i, &c) in cells.iter().enumerate() {
                        let v = deq(&vc, &vs, &vb, c as usize, j);
                        r += p[i] / z * v;
                        vmax = vmax.max(v.abs());
                    }
                    let e = norm_err(got[h * dim + j], r, vmax, act == Act::F16);
                    if e.is_nan() || e > worst || where_.is_empty() {
                        worst = e;
                        where_ = format!("{at} head={h} j={j} got={} ref={r:e}", got[h * dim + j]);
                    }
                }
            }
            self.cell(&key).record(worst, &where_);
        }
        Ok(())
    }

    // ------------------------------------------------------------ driving

    /// Run every instantiated cell of `entries`; probe every (format, entry,
    /// dtype) with no instantiation through the production entry.
    pub fn run(&mut self, entries: &[Entry]) -> Result<()> {
        let keys: BTreeSet<Key> = self.lib.cells.values().cloned().collect();
        for key in keys.iter().filter(|k| entries.contains(&k.0)) {
            let (entry, fmt, act) = key.clone();
            if entry.is_kv() {
                match entry {
                    Entry::KvDequant => self.run_kv_dequant(&fmt, act)?,
                    Entry::KvSdpa => self.run_kv_sdpa(&fmt, act, false)?,
                    Entry::KvSdpaSel => self.run_kv_sdpa(&fmt, act, true)?,
                    _ => unreachable!(),
                }
                continue;
            }
            let Some(d) = from_stem(&fmt) else { continue };
            match entry {
                Entry::MulMv => self.run_mul_mv(d, act, false)?,
                Entry::Rows => self.run_mul_mv(d, act, true)?,
                Entry::MulMvId => self.run_mv_id(d, act, false)?,
                Entry::MulMvId2 => self.run_mv_id(d, act, true)?,
                Entry::MulMm => self.run_mul_mm(d, false)?,
                Entry::MmId32 => self.run_mm_id(d, 32)?,
                Entry::MmId16 => self.run_mm_id(d, 16)?,
                Entry::GetRows => self.run_get_rows(d)?,
                Entry::DequantRowsF16 => {
                    self.run_dequant_rows_f16(d)?;
                    if entries.contains(&Entry::DequantMm) {
                        self.run_mul_mm(d, true)?;
                    }
                }
                Entry::DequantMm | Entry::KvSdpa | Entry::KvSdpaSel | Entry::KvDequant => {}
            }
        }
        for d in ALL {
            for entry in entries
                .iter()
                .copied()
                .filter(|e| !e.is_kv() && *e != Entry::DequantMm)
            {
                for act in [Act::F32, Act::F16] {
                    let key: Key = (entry, stem(d).into(), act);
                    if !keys.contains(&key) {
                        self.probe(entry, d, act)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// What the production entry does with a (format, entry, dtype) the
    /// library has no kernel for: refuse (recorded) or launch something (a
    /// problem: a kernel outside the matrix).
    fn probe(&mut self, entry: Entry, d: GgmlDType, act: Act) -> Result<()> {
        let k = Self::ks(d)[0];
        let (e, topk) = (2usize, 1usize);
        let r: Result<(Tensor, Census)> = (|| -> Result<(Tensor, Census)> {
            match entry {
                Entry::MulMv | Entry::Rows => {
                    let w = weight(&self.dev, d, &[4, k], 1, 1.0)?;
                    let m = if entry == Entry::Rows { 2 } else { 1 };
                    let (x, _) = activations(&self.dev, &[m, k], act, 1)?;
                    self.census(|| x.apply_op1_no_bwd(&w.q))
                }
                Entry::MulMm => {
                    let w = weight(&self.dev, d, &[4, k], 1, 1.0)?;
                    let (x, _) = activations(&self.dev, &[9, k], act, 1)?;
                    if act == Act::F16 {
                        return Err(lloom_tensor::Error::Msg(
                            "not probed: the mm path asserts f32 activations".into(),
                        ));
                    }
                    self.census(|| x.apply_op1_no_bwd(&w.q))
                }
                Entry::MulMvId | Entry::MulMvId2 | Entry::MmId32 | Entry::MmId16 => {
                    let w = weight(&self.dev, d, &[e, 4, k], 1, 1.0)?;
                    let (x, _) = activations(&self.dev, &[1, 1, k], act, 1)?;
                    let ids = Tensor::from_vec(vec![1u32], (1, topk), &self.dev)?;
                    let arm = if matches!(entry, Entry::MmId32 | Entry::MmId16) {
                        MoeArm::Gemm
                    } else {
                        MoeArm::MatVec
                    };
                    if entry == Entry::MulMvId2 {
                        self.census(|| {
                            Ok(w.q.indexed_moe_forward_pair_with(&w.q, &x, &ids, arm)?.0)
                        })
                    } else {
                        self.census(|| w.q.indexed_moe_forward_with(&x, &ids, arm))
                    }
                }
                Entry::GetRows => {
                    if act == Act::F16 {
                        return Err(lloom_tensor::Error::Msg("no activation".into()));
                    }
                    let w = weight(&self.dev, d, &[4, k], 1, 1.0)?;
                    let ids = Tensor::from_vec(vec![1u32], 1, &self.dev)?;
                    self.census(|| w.q.embedding(&ids))
                }
                Entry::DequantRowsF16 => Err(lloom_tensor::Error::Msg(
                    "not probed: no tensor-level entry".into(),
                )),
                _ => unreachable!(),
            }
        })();
        let what = format!("{} {} {}", entry.label(), stem(d), act.label());
        match r {
            Err(err) => {
                let msg = err.to_string();
                let msg = msg.lines().next().unwrap_or("").to_string();
                if !msg.starts_with("not probed") && !msg.starts_with("no activation") {
                    self.refusals.push(format!("{what}: {msg}"));
                }
            }
            Ok((_, census)) => {
                let ran: Vec<&String> = census
                    .iter()
                    .map(|(l, _)| l)
                    .filter(|l| is_quant_decode_family(l))
                    .collect();
                self.refusals.push(format!("{what}: served by {ran:?}"));
                for l in ran {
                    if !self.lib.cells.contains_key(l.as_str()) {
                        self.problems
                            .push(format!("{what} launched {l}, which is not a matrix cell"));
                    }
                }
            }
        }
        Ok(())
    }

    /// Every instantiated decode kernel of `families` is a cell that ran, or is
    /// named unreached with a reason.
    pub fn check_coverage(&mut self, entries: &[Entry]) {
        if self.lib.quantized_total == 0 || self.lib.sdpa_total == 0 {
            self.problems
                .push("a library reported no functions: the reader is blind".into());
        }
        let in_scope = |key: &Key| entries.contains(&key.0);
        for (name, key) in &self.lib.cells {
            if !in_scope(key) {
                continue;
            }
            if self.launched.contains(name) {
                continue;
            }
            if NO_GGML_DTYPE.contains(&key.1.as_str()) || UNREACHED.iter().any(|(n, _)| n == name) {
                continue;
            }
            self.problems.push(format!(
                "{name} ({}, {}, {}) is instantiated and no cell ran it",
                key.0.label(),
                key.1,
                key.2.label()
            ));
        }
        for name in &self.lib.unparsed {
            let named = UNREACHED.iter().any(|(n, _)| n == name)
                || UNREACHED_PREFIX.iter().any(|(p, _)| name.starts_with(p));
            if !named {
                self.problems.push(format!("{name}: a decode-family name the matrix cannot parse and UNREACHED does not name"));
            }
        }
        for (n, _) in UNREACHED {
            let present = self.lib.unparsed.contains(*n) || self.lib.cells.contains_key(*n);
            if !present {
                self.problems.push(format!(
                    "UNREACHED names {n}, which the library no longer has"
                ));
            }
            if self.launched.contains(*n) {
                self.problems
                    .push(format!("UNREACHED names {n}, and a cell launched it"));
            }
        }
        for (p, _) in UNREACHED_PREFIX {
            if self
                .launched
                .iter()
                .any(|l| l.starts_with(p) && parse_quantized(l).is_none())
            {
                self.problems
                    .push(format!("UNREACHED_PREFIX {p}: a cell launched one"));
            }
        }
    }

    /// The per-cell table, the unreached instantiations and the refusals; the
    /// failures as one list.
    pub fn report(&self, entries: &[Entry]) -> Vec<String> {
        let mut fails = Vec::new();
        println!("\n=== quant parity matrix: {} cells ===", self.cells.len());
        let mut stated = BTreeSet::new();
        for (e, _, a) in self.cells.keys() {
            if stated.insert(*e) {
                let (unit, tol, why) = tolerance(*e, *a);
                println!(
                    "TOL entry={} unit={unit:?} tol={tol:e} why=[{why}]",
                    e.label()
                );
            }
        }
        println!("| entry | format | act | kernels | cases | worst | tol | verdict |");
        println!("|---|---|---|---|---|---|---|---|");
        for ((e, f, a), c) in &self.cells {
            let unit = if c.unit == Unit::Ulp { " ulp" } else { "" };
            let verdict = if c.passed() { "pass" } else { "FAIL" };
            println!(
                "| {} | {f} | {} | {} | {} | {:.3e}{unit} | {:.1e}{unit} | {verdict} |",
                e.label(),
                a.label(),
                c.kernels.iter().cloned().collect::<Vec<_>>().join(" "),
                c.cases,
                c.worst,
                c.tol,
            );
            println!(
                "CELL entry={} fmt={f} act={} worst={:e} at=[{}] digest={:016x} verdict={verdict}",
                e.label(),
                a.label(),
                c.worst,
                c.worst_at,
                c.digest
            );
            if !c.passed() {
                if c.cases == 0 {
                    fails.push(format!("{} {f} {}: no case ran", e.label(), a.label()));
                }
                for m in c.fails.iter().take(4) {
                    fails.push(format!("{} {f} {}: {m}", e.label(), a.label()));
                }
                if c.fails.len() > 4 {
                    fails.push(format!(
                        "{} {f} {}: … {} more",
                        e.label(),
                        a.label(),
                        c.fails.len() - 4
                    ));
                }
                println!("FAIL cell={}/{f}/{}", e.label(), a.label());
            }
        }
        println!("\n=== instantiated, unreached by any dispatch ===");
        for (name, key) in &self.lib.cells {
            if entries.contains(&key.0) && !self.launched.contains(name) {
                println!("UNREACHED {name}");
            }
        }
        for name in &self.lib.unparsed {
            println!("UNREACHED {name}");
        }
        println!("\n=== (format, entry, dtype) with no instantiation: the dispatch's answer ===");
        for r in &self.refusals {
            println!("NO-INSTANTIATION {r}");
        }
        println!(
            "\nlibraries: quantized {} functions, sdpa {}; decode-family cells instantiated {}; launched {}",
            self.lib.quantized_total,
            self.lib.sdpa_total,
            self.lib.cells.len(),
            self.launched.len()
        );
        for p in &self.problems {
            println!("PROBLEM {p}");
            fails.push(p.clone());
        }
        fails
    }
}
