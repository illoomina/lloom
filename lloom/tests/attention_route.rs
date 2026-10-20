//! The prepared attention route on Metal: the kernel it names is the kernel
//! that runs, a refused request asks the Metal pool for nothing, the workspace it reports
//! is what the call asks the pool for, and the shapes it newly routes are the
//! attention (`docs/dependency-ordered-reuse-design.md` §8).
//!
//! The reference is a scalar f64 softmax over inputs exact in both tested
//! dtypes (|q| <= 5/16, |k| <= 6/16, |v| <= 15/8), so every difference is the
//! kernel's rounding, bounded before running it. The tiled kernel rounds
//! `scale * log2(e)` to the input dtype and then each scaled query element:
//! two relative roundings of u (2^-11 at f16, 2^-24 at f32) on every score,
//! whose magnitude is at most `scale * hd * 30/256`, about 1.05 at head dim
//! 80. A score moved by d moves an output by at most `2 * d * max|v|`.
//! Scores, the softmax and the weighted sum accumulate in f32 over at most
//! 80 products and 33 keys. At f16 that is `2 * 2u * 1.05 * 15/8` = 0.0039,
//! plus the stored output's half-ulp below 2 (2^-11): 0.0044, bounded by
//! 0.005. At f32 the same terms come to about 2.5e-5, bounded by 5e-5.
#![cfg(feature = "metal")]

use lloom::nn::attention::route::{
    AttentionBounds, AttentionKernel, AttentionSpec, KvLayout, MaskKind, MetalKernelSet,
    PreparedAttention,
};
use lloom::nn::ops;
use lloom_tensor::{DType, Device, Result, Tensor};

const HEADS: usize = 4;
const KV_HEADS: usize = 2;

struct Inputs {
    hd: usize,
    queries: usize,
    keys: usize,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl Inputs {
    /// Dyadic values: exact at f16 and f32.
    fn new(hd: usize, queries: usize, keys: usize) -> Self {
        let q = (0..HEADS * queries * hd)
            .map(|i| ((i % 11) as f32 - 5.) / 16.)
            .collect();
        let k = (0..KV_HEADS * keys * hd)
            .map(|i| ((i % 13) as f32 - 6.) / 16.)
            .collect();
        let v = (0..KV_HEADS * keys * hd)
            .map(|i| ((i % 31) as f32 - 15.) / 8.)
            .collect();
        Inputs {
            hd,
            queries,
            keys,
            q,
            k,
            v,
        }
    }

    /// Unmasked attention, every query over every key.
    fn reference(&self, scale: f64) -> Vec<f32> {
        let (hd, nq, nk) = (self.hd, self.queries, self.keys);
        let mut out = vec![0f32; HEADS * nq * hd];
        for h in 0..HEADS {
            let kv = h / (HEADS / KV_HEADS);
            for qi in 0..nq {
                let row = h * nq + qi;
                let scores: Vec<f64> = (0..nk)
                    .map(|key| {
                        (0..hd)
                            .map(|d| {
                                f64::from(self.q[row * hd + d])
                                    * f64::from(self.k[(kv * nk + key) * hd + d])
                            })
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = w.iter().sum();
                for d in 0..hd {
                    out[row * hd + d] = (w
                        .iter()
                        .enumerate()
                        .map(|(key, w)| w * f64::from(self.v[(kv * nk + key) * hd + d]))
                        .sum::<f64>()
                        / sum) as f32;
                }
            }
        }
        out
    }

    fn tensors(&self, dtype: DType, dev: &Device) -> Result<(Tensor, Tensor, Tensor)> {
        let (hd, nq, nk) = (self.hd, self.queries, self.keys);
        Ok((
            Tensor::from_vec(self.q.clone(), (1, HEADS, nq, hd), dev)?.to_dtype(dtype)?,
            Tensor::from_vec(self.k.clone(), (1, KV_HEADS, nk, hd), dev)?.to_dtype(dtype)?,
            Tensor::from_vec(self.v.clone(), (1, KV_HEADS, nk, hd), dev)?.to_dtype(dtype)?,
        ))
    }
}

fn spec(hd: usize, dtype: DType, mask: MaskKind) -> AttentionSpec {
    AttentionSpec {
        q_heads: HEADS,
        kv_heads: KV_HEADS,
        head_dim: hd,
        v_head_dim: hd,
        dtype,
        kv_dtype: dtype,
        mask,
        softcap: 1.0,
        layout: KvLayout::Strided,
        bounds: AttentionBounds {
            batch: 1,
            min_queries: 1,
            queries: 64,
            keys: 4096,
        },
    }
}

fn prepare(dev: &Device, s: AttentionSpec) -> Result<PreparedAttention> {
    let metal = dev.as_metal_device()?;
    PreparedAttention::prepare(s, &MetalKernelSet { device: metal })
        .map_err(|e| lloom_tensor::Error::Msg(e.to_string()))
}

fn bits(t: &Tensor) -> Result<Vec<u32>> {
    Ok(t.to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .map(|v| v.to_bits())
        .collect())
}

fn requested(dev: &Device) -> Result<u64> {
    let (shared, private) = dev.as_metal_device()?.pool_counters();
    Ok(shared.requested_bytes + private.requested_bytes)
}

fn tolerance(dtype: DType) -> f32 {
    if dtype == DType::F16 {
        0.005
    } else {
        0.00005
    }
}

fn metal() -> Option<Device> {
    Device::new_metal(0).ok()
}

/// 72 and 80 have tiled kernels and no one-query entry point. The route
/// sends one query there to the tiled kernel; the result is the attention,
/// and repeats bit for bit.
#[test]
fn one_query_at_a_head_dim_without_a_vector_entry_takes_the_tiled_kernel() -> Result<()> {
    let Some(dev) = metal() else { return Ok(()) };
    for hd in [72usize, 80] {
        for dtype in [DType::F16, DType::F32] {
            let route = prepare(&dev, spec(hd, dtype, MaskKind::None))?;
            assert_eq!(route.kernel_for(1, 1, 33)?, AttentionKernel::Full);
            let f = Inputs::new(hd, 1, 33);
            let scale = 1.0 / (hd as f64).sqrt();
            let want = f.reference(scale);
            let (q, k, v) = f.tensors(dtype, &dev)?;
            let mut first = None;
            for _ in 0..3 {
                let got = ops::attention(&q, &k, &v, None, scale as f32, &route)?;
                let vals = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                for (i, (g, w)) in vals.iter().zip(&want).enumerate() {
                    assert!(
                        (g - w).abs() <= tolerance(dtype),
                        "hd {hd} {dtype:?} element {i}: {g} against {w}"
                    );
                }
                let b = bits(&got)?;
                match &first {
                    None => first = Some(b),
                    Some(f) => assert_eq!(&b, f, "hd {hd} {dtype:?} not bit-stable"),
                }
            }
            // The unrouted op now serves the same shape with the same kernel.
            let unrouted = ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.0)?;
            assert_eq!(bits(&unrouted)?, first.unwrap(), "hd {hd} {dtype:?}");
        }
    }
    Ok(())
}

/// Routing changes no arithmetic: wherever the unrouted op already chose a
/// kernel, the route chooses the same one and the bits are identical.
#[test]
fn the_route_takes_the_kernel_the_unrouted_op_took() -> Result<()> {
    let Some(dev) = metal() else { return Ok(()) };
    for (queries, keys, kernel) in [
        (1, 100, AttentionKernel::Vector),
        (1, 2048, AttentionKernel::VectorTwoPass),
        (8, 100, AttentionKernel::Full),
    ] {
        let route = prepare(&dev, spec(64, DType::F16, MaskKind::Causal))?;
        assert_eq!(route.kernel_for(1, queries, keys)?, kernel);
        let f = Inputs::new(64, queries, keys);
        let (q, k, v) = f.tensors(DType::F16, &dev)?;
        let routed = ops::attention(&q, &k, &v, None, 0.125, &route)?;
        let unrouted = ops::sdpa(&q, &k, &v, None, queries > 1, 0.125, 1.0)?;
        assert_eq!(bits(&routed)?, bits(&unrouted)?, "{kernel:?}");
    }
    Ok(())
}

/// A request the route refuses, and a shape the op refuses, allocate
/// nothing: every check precedes the first allocation.
#[test]
fn a_refused_request_allocates_nothing() -> Result<()> {
    let Some(dev) = metal() else { return Ok(()) };
    let route = prepare(&dev, spec(64, DType::F16, MaskKind::Causal))?;
    let f = Inputs::new(64, 1, 33);
    let (q, k, v) = f.tensors(DType::F16, &dev)?;
    dev.synchronize()?;

    let mut narrow = spec(64, DType::F16, MaskKind::Causal);
    narrow.bounds.keys = 32;
    let narrow = prepare(&dev, narrow)?;
    let before = requested(&dev)?;
    assert!(ops::attention(&q, &k, &v, None, 0.125, &narrow).is_err());
    assert_eq!(requested(&dev)?, before, "past the key bound");

    let mask = Tensor::zeros((1, HEADS, 1, 33), DType::F16, &dev)?;
    let before = requested(&dev)?;
    assert!(ops::attention(&q, &k, &v, Some(&mask), 0.125, &route).is_err());
    assert_eq!(
        requested(&dev)?,
        before,
        "a mask the route was not prepared for"
    );

    // A route prepared for an additive mask, called without one, is refused:
    // the tiled kernel would otherwise serve it unmasked.
    let masked = prepare(&dev, spec(64, DType::F16, MaskKind::Additive))?;
    let before = requested(&dev)?;
    assert!(ops::attention(&q, &k, &v, None, 0.125, &masked).is_err());
    assert_eq!(
        requested(&dev)?,
        before,
        "the mask the route was prepared for, missing"
    );

    // The unrouted op at a head dim no kernel instantiates.
    let g = Inputs::new(48, 1, 33);
    let (q48, k48, v48) = g.tensors(DType::F16, &dev)?;
    let before = requested(&dev)?;
    assert!(ops::sdpa(&q48, &k48, &v48, None, false, 0.125, 1.0).is_err());
    assert_eq!(
        requested(&dev)?,
        before,
        "the op allocated its output before refusing"
    );

    // A mask of the wrong shape is refused before the output is allocated.
    let wrong = Tensor::zeros((1, HEADS, 1, 32), DType::F16, &dev)?;
    let before = requested(&dev)?;
    assert!(ops::sdpa(&q, &k, &v, Some(&wrong), false, 0.125, 1.0).is_err());
    assert_eq!(
        requested(&dev)?,
        before,
        "the op allocated before checking the mask"
    );
    Ok(())
}

/// Mask refusals precede mixed-KV conversion, not just attention output allocation.
/// The valid arm also proves the dispatch census observes conversion and attention.
#[test]
fn an_invalid_mask_precedes_mixed_kv_conversion() -> Result<()> {
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    let dev = Device::new_metal(0)?;
    let mut s = spec(64, DType::F16, MaskKind::Additive);
    s.kv_dtype = DType::F32;
    let route = prepare(&dev, s)?;
    let f = Inputs::new(64, 1, 33);
    let (q, _, _) = f.tensors(DType::F16, &dev)?;
    let (_, k, v) = f.tensors(DType::F32, &dev)?;
    let valid = Tensor::zeros((1, HEADS, 1, 33), DType::F16, &dev)?;
    let wrong_shape = Tensor::zeros((1, HEADS, 1, 32), DType::F16, &dev)?;
    let wrong_dtype = Tensor::zeros((1, HEADS, 1, 33), DType::F32, &dev)?;
    let wrong_device = Tensor::zeros((1, HEADS, 1, 33), DType::F16, &Device::Cpu)?;
    dev.synchronize()?;
    for (name, mask, reason) in [
        ("shape", &wrong_shape, "Mask shape"),
        ("dtype", &wrong_dtype, "Mask type"),
        ("device", &wrong_device, "query device"),
    ] {
        let before = requested(&dev)?;
        let window = dev
            .as_metal_device()?
            .dispatch_census()
            .open()
            .expect("census enabled");
        let error = ops::attention(&q, &k, &v, Some(mask), 0.125, &route)
            .expect_err("invalid mask must be refused");
        let dispatches = window.close();
        assert!(error.to_string().contains(reason), "{name}: {error}");
        assert_eq!(
            requested(&dev)?,
            before,
            "{name}: conversion allocated before refusal"
        );
        assert!(
            dispatches.is_empty(),
            "{name}: dispatched before refusal: {dispatches:?}"
        );
    }

    let expected = ops::sdpa(
        &q,
        &k.to_dtype(DType::F16)?,
        &v.to_dtype(DType::F16)?,
        Some(&valid),
        false,
        0.125,
        1.0,
    )?;
    dev.synchronize()?;
    let before = requested(&dev)?;
    let window = dev
        .as_metal_device()?
        .dispatch_census()
        .open()
        .expect("census enabled");
    let got = ops::attention(&q, &k, &v, Some(&valid), 0.125, &route)?;
    let dispatches: u64 = window.close().iter().map(|(_, count)| count).sum();
    assert!(
        dispatches >= 3,
        "two conversions and attention must be observed: {dispatches}"
    );
    let ws = route.workspace(1, 1, 33)?;
    assert_eq!(
        requested(&dev)? - before,
        (ws.output + ws.conversion) as u64
    );
    assert_eq!(
        bits(&got)?,
        bits(&expected)?,
        "valid mixed-KV route changed math"
    );
    eprintln!("mixed-KV mask preflight: invalid=3 allocated=0 dispatched=0; valid dispatches={dispatches} bytes={} exact=true", ws.output + ws.conversion);
    Ok(())
}

fn without_backtrace(error: &lloom_tensor::Error) -> &lloom_tensor::Error {
    match error {
        lloom_tensor::Error::WithBacktrace { inner, .. } => without_backtrace(inner),
        _ => error,
    }
}

fn assert_attention_device_error(
    error: &lloom_tensor::Error,
    q: &Device,
    other: &Device,
    op: &str,
) {
    match without_backtrace(error) {
        lloom_tensor::Error::DeviceMismatchBinaryOp(mismatch) => {
            assert_eq!(mismatch.op, op);
            assert_eq!(mismatch.lhs, q.location());
            assert_eq!(mismatch.rhs, other.location());
        }
        other => panic!("expected attention device mismatch, got {other:?}"),
    }
}

fn assert_conversion_is_observable(device: &Device) -> Result<()> {
    let (_, k, _) = Inputs::new(64, 1, 33).tensors(DType::F32, device)?;
    device.synchronize()?;
    let before = requested(device)?;
    let window = device
        .as_metal_device()?
        .dispatch_census()
        .open()
        .expect("census enabled");
    let converted = k.to_dtype(DType::F16)?;
    let dispatches: u64 = window.close().iter().map(|(_, n)| n).sum();
    assert_eq!(requested(device)? - before, (k.elem_count() * 2) as u64);
    assert_eq!(dispatches, 1, "conversion positive control");
    assert_eq!(converted.dtype(), DType::F16);
    device.synchronize()
}

/// Offset views with a padded row stride exercise the conversion's strided arm.
fn offset_strided_kv(f: &Inputs, dtype: DType, dev: &Device) -> Result<(Tensor, Tensor)> {
    let lay = |values: &[f32]| -> Result<Tensor> {
        let mut padded = Vec::with_capacity(4 * values.len());
        for _ in 0..2 {
            for row in values.chunks_exact(f.hd) {
                padded.extend_from_slice(row);
                padded.extend(std::iter::repeat_n(0., f.hd));
            }
        }
        Tensor::from_vec(padded, (2, KV_HEADS, f.keys, 2 * f.hd), dev)?
            .to_dtype(dtype)?
            .narrow(0, 1, 1)?
            .narrow(3, 0, f.hd)
    };
    Ok((lay(&f.k)?, lay(&f.v)?))
}

/// Drive the production wrapper, not its private validator. Both device windows
/// observe foreign Metal conversions; CPU conversion allocations are not counted.
#[test]
fn foreign_kv_devices_are_refused_before_routed_conversion() -> Result<()> {
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    let dev = Device::new_metal(0)?;
    let foreign = Device::new_metal(0)?;
    assert!(!dev.same_device(&foreign));
    assert_eq!(
        dev.location(),
        foreign.location(),
        "same GPU, distinct devices"
    );

    // Pin each device's instrument level with a real conversion before refusals.
    for device in [&dev, &foreign] {
        assert_conversion_is_observable(device)?;
    }

    for (queries, keys, mask_kind, kernel) in [
        (1, 33, MaskKind::None, AttentionKernel::Vector),
        (1, 2048, MaskKind::Causal, AttentionKernel::VectorTwoPass),
        (3, 33, MaskKind::Causal, AttentionKernel::Full),
        (1, 33, MaskKind::Additive, AttentionKernel::Full),
    ] {
        let f = Inputs::new(64, queries, keys);
        for kv_dtype in [DType::F16, DType::F32] {
            let mut s = spec(64, DType::F16, mask_kind);
            s.kv_dtype = kv_dtype;
            let route = prepare(&dev, s)?;
            assert_eq!(route.kernel_for(1, queries, keys)?, kernel);
            let (q, _, _) = f.tensors(DType::F16, &dev)?;
            let mask = if mask_kind == MaskKind::Additive {
                Some(Tensor::zeros((2, HEADS, queries, keys), DType::F16, &dev)?.narrow(0, 1, 1)?)
            } else {
                None
            };
            for strided in [false, true] {
                let kv = |device: &Device| -> Result<(Tensor, Tensor)> {
                    if strided {
                        offset_strided_kv(&f, kv_dtype, device)
                    } else {
                        let (_, k, v) = f.tensors(kv_dtype, device)?;
                        Ok((k, v))
                    }
                };
                let (k, v) = kv(&dev)?;
                if strided {
                    assert!(!k.is_contiguous());
                    assert_ne!(k.layout().start_offset(), 0);
                }
                for other in [&Device::Cpu, &foreign] {
                    let (foreign_k, foreign_v) = kv(other)?;
                    for (bad_k, bad_v, op) in [
                        (&foreign_k, &v, "attention: query/key"),
                        (&k, &foreign_v, "attention: query/value"),
                        (&foreign_k, &foreign_v, "attention: query/key"),
                    ] {
                        dev.synchronize()?;
                        foreign.synchronize()?;
                        let before = (requested(&dev)?, requested(&foreign)?);
                        let local_window = dev
                            .as_metal_device()?
                            .dispatch_census()
                            .open()
                            .expect("local census");
                        let foreign_window = foreign
                            .as_metal_device()?
                            .dispatch_census()
                            .open()
                            .expect("foreign census");
                        let error = ops::attention(&q, bad_k, bad_v, mask.as_ref(), 0.125, &route)
                            .expect_err("foreign K/V must be refused");
                        let local_dispatches = local_window.close();
                        let foreign_dispatches = foreign_window.close();
                        assert_attention_device_error(&error, &dev, other, op);
                        assert_eq!(
                            (requested(&dev)?, requested(&foreign)?),
                            before,
                            "Metal conversion requested storage before device refusal"
                        );
                        assert!(
                            local_dispatches.is_empty(),
                            "local dispatch before refusal: {local_dispatches:?}"
                        );
                        assert!(
                            foreign_dispatches.is_empty(),
                            "foreign dispatch before refusal: {foreign_dispatches:?}"
                        );
                    }
                }

                // Same route/math with explicit conversion; no peer-algorithm oracle.
                let expected = ops::attention(
                    &q,
                    &k.to_dtype(DType::F16)?,
                    &v.to_dtype(DType::F16)?,
                    mask.as_ref(),
                    0.125,
                    &prepare(&dev, spec(64, DType::F16, mask_kind))?,
                )?;
                dev.synchronize()?;
                let before = requested(&dev)?;
                let window = dev
                    .as_metal_device()?
                    .dispatch_census()
                    .open()
                    .expect("census enabled");
                let got = ops::attention(&q, &k, &v, mask.as_ref(), 0.125, &route)?;
                let dispatches: u64 = window.close().iter().map(|(_, n)| n).sum();
                let kernel_dispatches = if kernel == AttentionKernel::VectorTwoPass {
                    2
                } else {
                    1
                };
                assert_eq!(
                    dispatches,
                    kernel_dispatches + if kv_dtype == DType::F32 { 2 } else { 0 }
                );
                let ws = route.workspace(1, queries, keys)?;
                assert_eq!(
                    requested(&dev)? - before,
                    (ws.output + ws.scratch + ws.conversion) as u64
                );
                assert_eq!(
                    bits(&got)?,
                    bits(&expected)?,
                    "same-route conversion changed math"
                );
            }
        }
    }
    Ok(())
}

/// Paged requests reach the same validator; mixed-dtype page pools remain
/// unsupported rather than silently converting or changing the page semantics.
#[test]
fn foreign_paged_kv_devices_are_refused_at_route_preflight() -> Result<()> {
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    let dev = Device::new_metal(0)?;
    let foreign = Device::new_metal(0)?;
    assert!(!dev.same_device(&foreign));
    for device in [&dev, &foreign] {
        assert_conversion_is_observable(device)?;
    }
    let (page, queries, keys, slots) = (32usize, 9usize, 70usize, 128usize);
    let table = [2u32, 1, 0];
    let live = [keys as u32];
    let (q, _, _) = Inputs::new(64, queries, keys).tensors(DType::F16, &dev)?;
    for kv_dtype in [DType::F16, DType::F32] {
        let mut s = spec(64, DType::F16, MaskKind::Causal);
        s.layout = KvLayout::Paged { page_size: page };
        s.bounds.min_queries = 2;
        s.kv_dtype = kv_dtype;
        let route = prepare(&dev, s)?;
        let pool = Inputs::new(64, queries, slots);
        let (_, k, v) = pool.tensors(kv_dtype, &dev)?;
        for other in [&Device::Cpu, &foreign] {
            let (_, foreign_k, foreign_v) = pool.tensors(kv_dtype, other)?;
            for (bad_k, bad_v, op) in [
                (&foreign_k, &v, "attention: query/key"),
                (&k, &foreign_v, "attention: query/value"),
                (&foreign_k, &foreign_v, "attention: query/key"),
            ] {
                dev.synchronize()?;
                foreign.synchronize()?;
                let before = (requested(&dev)?, requested(&foreign)?);
                let local_window = dev
                    .as_metal_device()?
                    .dispatch_census()
                    .open()
                    .expect("local census");
                let foreign_window = foreign
                    .as_metal_device()?
                    .dispatch_census()
                    .open()
                    .expect("foreign census");
                let error = ops::attention_paged(&q, bad_k, bad_v, 0.125, &route, &table, 3, &live)
                    .expect_err("foreign page pools must be refused");
                let local_dispatches = local_window.close();
                let foreign_dispatches = foreign_window.close();
                assert_attention_device_error(&error, &dev, other, op);
                assert_eq!((requested(&dev)?, requested(&foreign)?), before);
                assert!(local_dispatches.is_empty());
                assert!(foreign_dispatches.is_empty());
            }
        }
        if kv_dtype == DType::F16 {
            let expected = ops::sdpa_paged(&q, &k, &v, true, 0.125, page, &table, 3, keys)?;
            let got = ops::attention_paged(&q, &k, &v, 0.125, &route, &table, 3, &live)?;
            assert_eq!(bits(&got)?, bits(&expected)?, "same paged kernel/table");
        } else {
            dev.synchronize()?;
            let before = requested(&dev)?;
            let window = dev
                .as_metal_device()?
                .dispatch_census()
                .open()
                .expect("census enabled");
            let error = ops::attention_paged(&q, &k, &v, 0.125, &route, &table, 3, &live)
                .expect_err("mixed page pools remain unsupported");
            assert_eq!(
                without_backtrace(&error).to_string(),
                "attention_paged: the pool is read at the compute dtype"
            );
            assert_eq!(requested(&dev)?, before);
            assert!(window.close().is_empty());
        }
    }
    Ok(())
}

/// The workspace the route reports is exactly what one call asks the pool
/// for: the output, and the two-pass scratch when that kernel runs.
#[test]
fn the_workspace_is_what_the_call_requests() -> Result<()> {
    let Some(dev) = metal() else { return Ok(()) };
    for dtype in [DType::F16, DType::F32] {
        let route = prepare(&dev, spec(64, dtype, MaskKind::Causal))?;
        for (queries, keys) in [(1, 100), (1, 2048), (8, 100)] {
            let f = Inputs::new(64, queries, keys);
            let (q, k, v) = f.tensors(dtype, &dev)?;
            let ws = route.workspace(1, queries, keys)?;
            dev.synchronize()?;
            let before = requested(&dev)?;
            let out = ops::attention(&q, &k, &v, None, 0.125, &route)?;
            let asked = requested(&dev)? - before;
            assert_eq!(
                asked,
                (ws.output + ws.scratch) as u64,
                "{dtype:?} {queries}x{keys} via {:?}: {ws:?}",
                route.kernel_for(1, queries, keys)?
            );
            assert_eq!(out.elem_count() * dtype.size_in_bytes(), ws.output);
        }
    }
    Ok(())
}

/// A masked single query goes to the tiled kernel through the route, and a
/// mask read through a view at a non-zero offset gives the bits of the same
/// mask materialized.
#[test]
fn a_masked_single_query_and_an_offset_mask_view_through_the_route() -> Result<()> {
    let Some(dev) = metal() else { return Ok(()) };
    let route = prepare(&dev, spec(64, DType::F16, MaskKind::Additive))?;
    assert_eq!(route.kernel_for(1, 1, 33)?, AttentionKernel::Full);
    let f = Inputs::new(64, 1, 33);
    let (q, k, v) = f.tensors(DType::F16, &dev)?;
    let rows: Vec<f32> = (0..2 * HEADS * 33)
        .map(|i| if i % 33 == (i / 33) % 33 { 0.5 } else { -0.75 })
        .collect();
    let wide = Tensor::from_vec(rows, (2, HEADS, 1, 33), &dev)?.to_dtype(DType::F16)?;
    let view = wide.narrow(0, 1, 1)?;
    assert_ne!(view.layout().start_offset(), 0);
    let materialized = view.contiguous()?;
    let a = ops::attention(&q, &k, &v, Some(&view), 0.125, &route)?;
    let b = ops::attention(&q, &k, &v, Some(&materialized), 0.125, &route)?;
    assert_eq!(bits(&a)?, bits(&b)?);
    let c = ops::sdpa(&q, &k, &v, Some(&materialized), false, 0.125, 1.0)?;
    assert_eq!(
        bits(&a)?,
        bits(&c)?,
        "the route took another kernel than the op"
    );
    Ok(())
}

/// The Metal kernel set agrees with the compiled entry points: the vector
/// family at its six head dims, the tiled family at eight.
#[test]
fn the_metal_kernel_set_names_what_is_compiled() -> Result<()> {
    let Some(dev) = metal() else { return Ok(()) };
    for hd in [32usize, 64, 72, 80, 96, 128, 256] {
        let route = prepare(&dev, spec(hd, DType::F16, MaskKind::Causal))?;
        let one = route.kernel_for(1, 1, 10)?;
        let vector = !matches!(hd, 72 | 80);
        assert_eq!(
            one == AttentionKernel::Vector,
            vector,
            "hd {hd}: {}",
            route.describe()
        );
    }
    assert!(prepare(&dev, spec(48, DType::F16, MaskKind::Causal)).is_err());
    let mut f32_512 = spec(512, DType::F32, MaskKind::Causal);
    f32_512.kv_dtype = DType::F32;
    assert!(prepare(&dev, f32_512).is_err());
    Ok(())
}

/// The routed page-pool entry reads the pool through its table and gives the
/// bits of the unrouted paged op, and of the same keys read from a strided
/// cache; one query over a pool is refused before anything is allocated
/// (one query over a pool is flash decoding's).
#[test]
fn a_page_pool_through_the_route_is_the_unrouted_paged_op() -> Result<()> {
    let Some(dev) = metal() else { return Ok(()) };
    let (hd, page, keys, queries) = (64usize, 32usize, 70usize, 9usize);
    let mut s = spec(hd, DType::F16, MaskKind::Causal);
    s.layout = KvLayout::Paged { page_size: page };
    s.bounds.min_queries = 2;
    let route = prepare(&dev, s)?;
    let f = Inputs::new(hd, queries, keys);
    let (q, k, v) = f.tensors(DType::F16, &dev)?;
    // The live keys laid into a pool one page larger, through a reversed
    // table, so a read that ignored the table would read another page.
    let pages = keys.div_ceil(page);
    let table: Vec<u32> = (0..pages as u32).rev().collect();
    let slots = (pages + 1) * page;
    let lay = |live: &[f32]| -> Result<Tensor> {
        let mut pool = vec![0f32; KV_HEADS * slots * hd];
        for h in 0..KV_HEADS {
            for t in 0..keys {
                let slot = table[t / page] as usize * page + t % page;
                let (d, s) = ((h * slots + slot) * hd, (h * keys + t) * hd);
                pool[d..d + hd].copy_from_slice(&live[s..s + hd]);
            }
        }
        Tensor::from_vec(pool, (1, KV_HEADS, slots, hd), &dev)?.to_dtype(DType::F16)
    };
    let (pk, pv) = (lay(&f.k)?, lay(&f.v)?);
    let live = [keys as u32];
    let routed = ops::attention_paged(&q, &pk, &pv, 0.125, &route, &table, pages, &live)?;
    let unrouted = ops::sdpa_paged(&q, &pk, &pv, true, 0.125, page, &table, pages, keys)?;
    assert_eq!(bits(&routed)?, bits(&unrouted)?, "routed against unrouted");
    let strided = ops::sdpa(&q, &k, &v, None, true, 0.125, 1.0)?;
    assert_eq!(
        bits(&routed)?,
        bits(&strided)?,
        "the pool read against the cache read"
    );

    let one = Inputs::new(hd, 1, keys);
    let (q1, _, _) = one.tensors(DType::F16, &dev)?;
    dev.synchronize()?;
    let before = requested(&dev)?;
    assert!(ops::attention_paged(&q1, &pk, &pv, 0.125, &route, &table, pages, &live).is_err());
    assert_eq!(
        requested(&dev)?,
        before,
        "one query over a pool allocated before refusing"
    );
    Ok(())
}
