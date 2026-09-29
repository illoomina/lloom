//! Zero-initialised memory, named by site, with a poison mode that measures
//! which sites actually read their zeros (`docs/pass-plan-design.md` §18).
//!
//! A zero site is one of four things, and only the first two may skip the
//! fill: every element is written before it is read (a); the unwritten part
//! is never read (b); the unwritten part is read as a real zero (c); the
//! reader masks it and holds the zero in registers (d). [`uninit`] and
//! [`poison_region`] are for (a)/(b); [`zeros`] and [`zero_region`] for (c).
//!
//! `LLOOM_POISON_ZEROS=all|<site>[,<site>…]` fills the named sites with a
//! sentinel instead: `LLOOM_POISON_VALUE=nan` shows any read that reaches
//! arithmetic (`0 × NaN` is NaN), `finite` shows a read whose value matters.
//! A run whose outputs are byte-identical under both has no output that
//! depends on the site's unwritten part. [`poisoned`] counts the fills, so an
//! unarmed run cannot pass for a clean one. Off, each entry point is one
//! `OnceLock` load and one relaxed atomic load.

use lloom_tensor::scalar::Scalar;
use lloom_tensor::{DType, Device, Result, Shape, Tensor};
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

/// The sliding ring's columns past the live window after a prefill: its
/// history and slack, which no fold reads before decode writes them.
pub const LFM2_CONV_RING_SLACK: &str = "lfm2.conv.ring_slack";
/// The rotating ring's columns past the live window after a prefill: folded
/// against a zero weight until decode writes them.
pub const LFM2_CONV_RING_HISTORY: &str = "lfm2.conv.ring_history";
/// LFM2's short-conv state after a prefill shorter than the window: the
/// causal zero history in front of the first input.
pub const LFM2_CONV_LEFT_PAD: &str = "lfm2.conv.left_pad";
/// LFM2's short-conv state storage, allocated on a sequence's first prefill.
pub const LFM2_CONV_STATE: &str = "lfm2.conv.state";
/// The sliding ring's compaction: the slack after the moved live window.
pub const LFM2_CONV_COMPACT_PAD: &str = "lfm2.conv.compact_pad";
/// A decode step on a sequence with no conv state yet: the causal zero
/// history every decode arm starts from.
pub const LFM2_CONV_FIRST_STATE: &str = "lfm2.conv.first_state";
/// The contiguous KV slab (`KvSlot`): positions past `len`.
pub const LFM2_KV_SLAB: &str = "lfm2.kv.slab";
/// The quantised KV planes (`QuantKvSlot`): positions past `len`.
pub const LFM2_KV_QUANT: &str = "lfm2.kv.quant";
/// The paged KV pools (`PagedKv`): slots no page has written.
pub const LFM2_KV_PAGES: &str = "lfm2.kv.pages";

/// Every site a poison list may name.
pub const SITES: &[&str] = &[
    LFM2_CONV_RING_SLACK,
    LFM2_CONV_RING_HISTORY,
    LFM2_CONV_LEFT_PAD,
    LFM2_CONV_STATE,
    LFM2_CONV_COMPACT_PAD,
    LFM2_CONV_FIRST_STATE,
    LFM2_KV_SLAB,
    LFM2_KV_QUANT,
    LFM2_KV_PAGES,
];

struct Poison {
    all: bool,
    sites: Vec<&'static str>,
    nan: bool,
}

static FILLS: AtomicU64 = AtomicU64::new(0);
static ARMED: AtomicBool = AtomicBool::new(false);
static ARM: RwLock<Option<Poison>> = RwLock::new(None);
static SEEN: parking_lot::Mutex<Vec<&str>> = parking_lot::Mutex::new(Vec::new());

fn init() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        let t = &crate::tuning::tuning().lfm2;
        let p = parse(&t.poison_zeros, &t.poison_value)
            .unwrap_or_else(|e| panic!("LLOOM_POISON_ZEROS: {e}"));
        ARMED.store(p.is_some(), Ordering::Relaxed);
        *ARM.write() = p;
    });
}

fn armed(site: &str) -> Option<bool> {
    init();
    if !ARMED.load(Ordering::Relaxed) {
        return None;
    }
    let arm = ARM.read();
    let p = arm.as_ref()?;
    (p.all || p.sites.contains(&site)).then_some(p.nan)
}

/// Arm the poison from a harness or a test, replacing what the environment
/// armed: `list` and `value` as `LLOOM_POISON_ZEROS` / `LLOOM_POISON_VALUE`
/// spell them; an empty `list` disarms it.
pub fn set_poison(list: &str, value: &str) -> Result<()> {
    init();
    let p = parse(list, value).map_err(lloom_tensor::Error::msg)?;
    ARMED.store(p.is_some(), Ordering::Relaxed);
    *ARM.write() = p;
    Ok(())
}

fn parse(list: &str, value: &str) -> std::result::Result<Option<Poison>, String> {
    let list = list.trim();
    if list.is_empty() {
        return Ok(None);
    }
    if value != "nan" && value != "finite" {
        return Err(format!("the sentinel `{value}` is not `nan` or `finite`"));
    }
    let mut p = Poison {
        all: false,
        sites: Vec::new(),
        nan: value != "finite",
    };
    for name in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if name == "all" {
            p.all = true;
        } else if let Some(s) = SITES.iter().find(|s| **s == name) {
            p.sites.push(s);
        } else {
            return Err(format!(
                "`{name}` is not a zero site; the sites are {}",
                SITES.join(", ")
            ));
        }
    }
    eprintln!(
        "[poison zeros] armed: {} with {}",
        if p.all {
            "all".to_string()
        } else {
            p.sites.join(",")
        },
        if p.nan { "NaN" } else { "a finite sentinel" }
    );
    Ok(Some(p))
}

/// Buffers and regions filled with the sentinel so far in this process.
pub fn poisoned() -> u64 {
    FILLS.load(Ordering::Relaxed)
}

fn sentinel(dtype: DType, nan: bool) -> Result<Scalar> {
    use half::{bf16, f16};
    Ok(match (dtype, nan) {
        (DType::F16, true) => Scalar::F16(f16::NAN),
        (DType::F16, false) => Scalar::F16(f16::from_f32(30000.0)),
        (DType::BF16, true) => Scalar::BF16(bf16::NAN),
        (DType::BF16, false) => Scalar::BF16(bf16::from_f32(30000.0)),
        (DType::F32, true) => Scalar::F32(f32::NAN),
        (DType::F32, false) => Scalar::F32(30000.0),
        (DType::U8, true) => Scalar::U8(0xff),
        (DType::U8, false) => Scalar::U8(0xa5),
        (DType::U32, _) => Scalar::U32(0xa5a5_a5a5),
        (other, _) => lloom_tensor::bail!("zero-site poison: no sentinel for {other:?}"),
    })
}

fn fill(site: &'static str, t: &Tensor, nan: bool) -> Result<()> {
    t.const_set(sentinel(t.dtype(), nan)?)?;
    FILLS.fetch_add(1, Ordering::Relaxed);
    let first = {
        let mut seen = SEEN.lock();
        let first = !seen.contains(&site);
        if first {
            seen.push(site);
        }
        first
    };
    if first {
        eprintln!(
            "[poison zeros] first fill: {site} {:?} {:?}",
            t.dims(),
            t.dtype()
        );
    }
    Ok(())
}

fn sentinel_tensor(
    site: &'static str,
    shape: Shape,
    dtype: DType,
    device: &Device,
    nan: bool,
) -> Result<Tensor> {
    // SAFETY: every element is written by the fill before the tensor is returned.
    let t = unsafe { Tensor::empty(shape, dtype, device)? };
    fill(site, &t, nan)?;
    Ok(t)
}

/// A tensor of zeros that its readers read as zeros (case c).
pub fn zeros<S: Into<Shape>>(
    site: &'static str,
    shape: S,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    match armed(site) {
        Some(nan) => sentinel_tensor(site, shape.into(), dtype, device, nan),
        None => Tensor::zeros(shape, dtype, device),
    }
}

/// A tensor whose every element is written before any is read, or whose
/// unwritten elements are never read (cases a and b): no fill at all.
///
/// Callers name a site whose poison runs are byte-identical; the poison is
/// what keeps that claim re-checkable.
pub fn uninit<S: Into<Shape>>(
    site: &'static str,
    shape: S,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    match armed(site) {
        Some(nan) => sentinel_tensor(site, shape.into(), dtype, device, nan),
        // SAFETY: the site's contract (above): no element is read before it
        // is written.
        None => unsafe { Tensor::empty(shape, dtype, device) },
    }
}

/// Zero a region of existing storage in place, for a reader that reads it
/// as zeros (case c).
pub fn zero_region(site: &'static str, region: &Tensor) -> Result<()> {
    match armed(site) {
        Some(nan) => fill(site, region, nan),
        None => region.zero_set(),
    }
}

/// A region of existing storage whose stale contents are never read (case
/// b): nothing to do, except under the poison.
pub fn poison_region(site: &'static str, region: &Tensor) -> Result<()> {
    match armed(site) {
        Some(nan) => fill(site, region, nan),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_list_names_known_sites_or_is_refused() {
        assert!(parse("", "nan").unwrap().is_none());
        let p = parse("lfm2.kv.slab, lfm2.conv.left_pad", "finite")
            .unwrap()
            .unwrap();
        assert_eq!(p.sites, vec![LFM2_KV_SLAB, LFM2_CONV_LEFT_PAD]);
        assert!(!p.nan && !p.all);
        assert!(parse("all", "nan").unwrap().unwrap().all);
        assert!(parse("lfm2.kv.slb", "nan").is_err());
        assert!(parse("all", "inf").is_err());
    }

    #[test]
    fn the_sentinels_are_the_values_named() -> Result<()> {
        let dev = Device::Cpu;
        let nan = sentinel_tensor("test", Shape::from(4), DType::F16, &dev, true)?;
        assert!(nan
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?
            .iter()
            .all(|v| v.is_nan()));
        let fin = sentinel_tensor("test", Shape::from(4), DType::F32, &dev, false)?;
        assert_eq!(fin.to_vec1::<f32>()?, vec![30000.0; 4]);
        let codes = sentinel_tensor("test", Shape::from(3), DType::U8, &dev, true)?;
        assert_eq!(codes.to_vec1::<u8>()?, vec![0xff; 3]);
        Ok(())
    }
}
