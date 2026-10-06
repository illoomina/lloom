//! The short-conv block and the GDN step block against the kernels they
//! replaced, bit for bit: every kernel the blocks re-express, driven through
//! the public entry points at fixed inputs, its output and the state it leaves
//! hashed and held to the hash the pre-block tree computed.
//!
//! The hashes are an absolute reference (CONTRIBUTING §3.1b rule 1): taken by
//! running this file, unchanged, on the tree before the blocks existed
//! (`design/measurements/issue-840-raw/golden-base.txt`), so a moved bit is a
//! failure naming the fixture it moved in. `LLOOM_GOLDEN_PRINT=1` prints the
//! table instead of asserting it — how the base tree's column was taken.
//!
//! Covered: LFM2's `shortconv_step` (the crown's `Shuffle` arm) at k ∈ {2, 3,
//! 4} and its `shortconv_ring` / `shortconv_ring_rows` (the rows and
//! speculative arm), each at f16 / f32 / bf16 and at a channel count (37) and
//! ring widths no model uses; Gated DeltaNet's decode prologue and step (one
//! row, pooled rows, a multi-position window) and its chunked prefill, at
//! Qwen4-exp's head geometry (k = 4) and at one no model has (k = 3, d = 24,
//! 2 → 6 heads).
//!
//! The GDN fixtures name the step's form rather than take the configured one:
//! the loop form's are the pre-block kernels' hashes above; the register
//! form's (`gdn … register …`) were taken on the tree that gave the register
//! form its rows and window kernels, whose bits
//! `gdn_step_forms.rs` holds to the one-row step's
//! (`design/measurements/issue-839-raw/golden.txt`).
#![cfg(feature = "metal")]

use lloom::models::qwen4exp::{Config, GatedDeltaNet};
use lloom::nn::gdn::GdnStepForm;
use lloom::nn::{RecurrentState, StatePool};
use lloom::profile::CrownExact;
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{DType, Device, Result, Tensor};

fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 2654435761 + seed * 40503) % 1000) as f32) / 500.0 - 1.0)
        .collect()
}

fn tensor(dims: &[usize], seed: usize, scale: f32, dtype: DType, dev: &Device) -> Result<Tensor> {
    let n = dims.iter().product();
    let v: Vec<f32> = fill(n, seed).into_iter().map(|x| x * scale).collect();
    Tensor::from_vec(v, dims, dev)?.to_dtype(dtype)
}

/// FNV-1a over the f32 bit patterns (f16 / bf16 widen exactly).
fn fnv(parts: &[&Tensor]) -> Result<String> {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for t in parts {
        for x in t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()? {
            for b in x.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
    Ok(format!("{h:016x}"))
}

/// The per-phase permuted taps `[width, hidden, width]`: slot `s` at `phase`
/// weighted by the tap for the input `(phase − s) mod width` back, zero past
/// the window.
fn ring_table(w: &Tensor, l_cache: usize, width: usize) -> Result<Tensor> {
    let (hidden, _) = w.dims2()?;
    let mut per_phase = Vec::with_capacity(width);
    for phase in 0..width {
        let mut cols = Vec::with_capacity(width);
        for s in 0..width {
            let back = (phase + width - s) % width;
            cols.push(if back < l_cache {
                w.narrow(1, l_cache - 1 - back, 1)?
            } else {
                Tensor::zeros((hidden, 1), w.dtype(), w.device())?
            });
        }
        per_phase.push(Tensor::cat(&cols, 1)?.contiguous()?);
    }
    Tensor::stack(&per_phase, 0)?.contiguous()
}

/// `b`, `c`, `x` as LFM2 narrows them from one `[b, 3·hidden, seq]` projection.
fn bcx(
    b_sz: usize,
    hidden: usize,
    seq: usize,
    seed: usize,
    dtype: DType,
    dev: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let all = tensor(&[b_sz, 3 * hidden, seq], seed, 1.0, dtype, dev)?;
    Ok((
        all.narrow(1, 0, hidden)?,
        all.narrow(1, hidden, hidden)?,
        all.narrow(1, 2 * hidden, hidden)?,
    ))
}

fn dtype_name(d: DType) -> &'static str {
    match d {
        DType::F16 => "f16",
        DType::F32 => "f32",
        DType::BF16 => "bf16",
        _ => "other",
    }
}

fn lfm2_fixtures(dev: &Device, out: &mut Vec<(String, String)>) -> Result<()> {
    for dtype in [DType::F16, DType::F32, DType::BF16] {
        let dn = dtype_name(dtype);
        for (hidden, k, b_sz) in [
            (37usize, 2usize, 2usize),
            (37, 3, 2),
            (37, 4, 1),
            (2048, 3, 1),
        ] {
            let (b, c, x) = bcx(b_sz, hidden, 1, 11 + k, dtype, dev)?;
            let state = tensor(&[b_sz, hidden, k], 23 + k, 1.0, dtype, dev)?;
            let w = tensor(&[hidden, k], 31 + k, 0.5, dtype, dev)?;
            let (o, next) = lloom::nn::ops::shortconv_step(&b, &x, &c, &state, &w, k)?;
            let tag = format!("step {dn} hidden={hidden} k={k} b={b_sz}");
            out.push((format!("{tag} out"), fnv(&[&o])?));
            out.push((format!("{tag} state"), fnv(&[&next, &state])?));
        }
        for (hidden, l_cache, width, seq, phase_last, b_sz) in [
            (37usize, 3usize, 4usize, 1usize, 2usize, 2usize),
            (37, 3, 9, 9, 4, 1),
            (37, 4, 7, 3, 1, 2),
            (2048, 3, 4, 1, 3, 1),
            (2048, 3, 10, 10, 6, 1),
        ] {
            let (b, c, x) = bcx(b_sz, hidden, seq, 41 + width, dtype, dev)?;
            let ring = tensor(&[b_sz, hidden, width], 53 + width, 1.0, dtype, dev)?;
            let w = tensor(&[hidden, l_cache], 61 + l_cache, 0.5, dtype, dev)?;
            let table = ring_table(&w, l_cache, width)?;
            let o = lloom::nn::ops::shortconv_ring(&b, &x, &c, &ring, &table, phase_last)?;
            let tag = format!("ring {dn} hidden={hidden} k={l_cache} width={width} seq={seq} phase={phase_last} b={b_sz}");
            out.push((format!("{tag} out"), fnv(&[&o])?));
            out.push((format!("{tag} state"), fnv(&[&ring])?));
        }
        // Two rows of a pool of three rings, at their own slots and phases.
        let (hidden, l_cache, width) = (37usize, 3usize, 5usize);
        let (b, c, x) = bcx(2, hidden, 1, 71, dtype, dev)?;
        let pool = tensor(&[3, hidden, width], 73, 1.0, dtype, dev)?;
        let w = tensor(&[hidden, l_cache], 79, 0.5, dtype, dev)?;
        let table = ring_table(&w, l_cache, width)?;
        let slots = Tensor::new(&[2u32, 0], dev)?;
        let phases = Tensor::new(&[1u32, 3], dev)?;
        let o = lloom::nn::ops::shortconv_ring_rows(&b, &x, &c, &pool, &table, &slots, &phases)?;
        let tag = format!("ring-rows {dn} hidden={hidden} k={l_cache} width={width}");
        out.push((format!("{tag} out"), fnv(&[&o])?));
        out.push((format!("{tag} state"), fnv(&[&pool])?));
    }
    Ok(())
}

fn lin(out: usize, inp: usize, seed: usize, dev: &Device) -> Result<QMatMul> {
    Ok(QMatMul::Tensor(tensor(
        &[out, inp],
        seed,
        0.05,
        DType::F32,
        dev,
    )?))
}

fn gdn_block(cfg: &Config, form: GdnStepForm, dev: &Device) -> Result<GatedDeltaNet<CrownExact>> {
    let d = cfg.hidden_size;
    let conv_ch = cfg.conv_channels();
    let value_dim = cfg.ssm_state_size * cfg.ssm_dt_rank;
    let h_v = cfg.ssm_dt_rank;
    let ramp = |n: usize, f: &dyn Fn(usize) -> f32| {
        Tensor::from_vec((0..n).map(f).collect::<Vec<f32>>(), n, dev)
    };
    Ok(GatedDeltaNet::<CrownExact>::new(
        lin(conv_ch, d, 1, dev)?,
        lin(value_dim, d, 2, dev)?,
        tensor(&[conv_ch, cfg.ssm_conv_kernel], 3, 0.3, DType::F32, dev)?,
        ramp(h_v, &|i| -0.5 - 0.01 * i as f32)?,
        ramp(h_v, &|i| 0.1 * (i % 7) as f32)?,
        lin(h_v, d, 4, dev)?,
        lin(h_v, d, 5, dev)?,
        ramp(cfg.ssm_state_size, &|i| 1.0 + 0.01 * (i % 5) as f32)?,
        lin(d, value_dim, 6, dev)?,
        cfg,
    )
    .with_step_form(form))
}

fn x_at(d: usize, t: usize, dev: &Device) -> Result<Tensor> {
    tensor(&[1, 1, d], 100 + t, 0.6, DType::F32, dev)
}

fn gdn_fixtures(
    name: &str,
    cfg: &Config,
    form: GdnStepForm,
    dev: &Device,
    out: &mut Vec<(String, String)>,
) -> Result<()> {
    let blk = gdn_block(cfg, form, dev)?;
    assert_eq!(blk.step_form(), form);
    let name = match form {
        GdnStepForm::Loop => name.to_string(),
        GdnStepForm::Register => format!("{name} register"),
    };
    let d = cfg.hidden_size;
    let dims = cfg.deltanet_dims();
    let (dk, h_v, conv_ch, k) = (dims.d_k, dims.h_v, cfg.conv_channels(), cfg.ssm_conv_kernel);

    // One row: the decode prologue, the step, the gated norm, the slabs swapped.
    let (mut s, mut c) = (RecurrentState::new(1), RecurrentState::new(1));
    let mut outs = Vec::new();
    for t in 0..6 {
        outs.push(blk.decode(&x_at(d, t, dev)?, 0, &mut s, &mut c)?);
    }
    let refs: Vec<&Tensor> = outs.iter().collect();
    out.push((format!("gdn {name} decode"), fnv(&refs)?));
    out.push((
        format!("gdn {name} decode state"),
        fnv(&[s.get(0).unwrap(), c.get(0).unwrap()])?,
    ));

    // The chunked prefill (two chunks, the second partial), then decode after it.
    let (mut s, mut c) = (RecurrentState::new(1), RecurrentState::new(1));
    let seq = 70;
    let xs = tensor(&[1, seq, d], 7, 0.6, DType::F32, dev)?;
    let p = blk.prefill(&xs, 0, &mut s, &mut c)?;
    out.push((format!("gdn {name} prefill"), fnv(&[&p])?));
    out.push((
        format!("gdn {name} prefill state"),
        fnv(&[s.get(0).unwrap(), c.get(0).unwrap()])?,
    ));
    let o = blk.decode(&x_at(d, 50, dev)?, 0, &mut s, &mut c)?;
    out.push((
        format!("gdn {name} decode after prefill"),
        fnv(&[&o, s.get(0).unwrap(), c.get(0).unwrap()])?,
    ));

    // Two rows of one pool: each row's first step alone (it zeroes its slots),
    // then the rows kernels over both.
    let s_pool = StatePool::new(3, 1, &[dk, dk, h_v], DType::F32, dev);
    let c_pool = StatePool::new(3, 1, &[conv_ch, k], DType::F32, dev);
    let mut rows: Vec<(RecurrentState, RecurrentState)> = (0..2)
        .map(|_| {
            Ok((
                RecurrentState::in_pool(&s_pool)?,
                RecurrentState::in_pool(&c_pool)?,
            ))
        })
        .collect::<Result<_>>()?;
    for (r, (s, c)) in rows.iter_mut().enumerate() {
        blk.decode(&x_at(d, 10 + r, dev)?, 0, s, c)?;
    }
    let mut outs = Vec::new();
    for t in 0..4 {
        let x = Tensor::cat(&[&x_at(d, 20 + t, dev)?, &x_at(d, 30 + t, dev)?], 0)?;
        let mut st: Vec<(&mut RecurrentState, &mut RecurrentState)> =
            rows.iter_mut().map(|(a, b)| (a, b)).collect();
        outs.push(blk.decode_rows(&x, 0, &mut st)?);
    }
    let refs: Vec<&Tensor> = outs.iter().collect();
    out.push((format!("gdn {name} rows"), fnv(&refs)?));
    let states: Vec<&Tensor> = rows
        .iter()
        .flat_map(|(s, c)| [s.get(0).unwrap(), c.get(0).unwrap()])
        .collect();
    out.push((format!("gdn {name} rows state"), fnv(&states)?));

    // A three-position window over a four-phase pool: every intermediate state
    // kept by phase.
    let s_pool = StatePool::with_phases(1, 4, 1, &[dk, dk, h_v], DType::F32, dev);
    let c_pool = StatePool::with_phases(1, 4, 1, &[conv_ch, k], DType::F32, dev);
    let (mut s, mut c) = (
        RecurrentState::in_pool(&s_pool)?,
        RecurrentState::in_pool(&c_pool)?,
    );
    for t in 0..2 {
        blk.decode(&x_at(d, 40 + t, dev)?, 0, &mut s, &mut c)?;
    }
    let xs = Tensor::cat(
        &[&x_at(d, 60, dev)?, &x_at(d, 61, dev)?, &x_at(d, 62, dev)?],
        0,
    )?;
    let o = blk.decode_seq(&xs, 0, &mut s, &mut c)?;
    out.push((format!("gdn {name} seq"), fnv(&[&o])?));
    out.push((
        format!("gdn {name} seq pools"),
        fnv(&[&s_pool.layer(0)?, &c_pool.layer(0)?])?,
    ));
    Ok(())
}

fn gdn_configs() -> Vec<(&'static str, Config)> {
    let mut real = Config::flash_next();
    real.hidden_size = 128;
    let mut odd = Config::flash_next();
    odd.hidden_size = 40;
    odd.ssm_state_size = 24;
    odd.ssm_n_group = 2;
    odd.ssm_dt_rank = 6;
    odd.ssm_conv_kernel = 3;
    vec![("heads16x48-d128-k4", real), ("heads2x6-d24-k3", odd)]
}

/// The pre-block tree's hashes (`design/measurements/issue-840-raw/golden-base.txt`),
/// then the register form's (`design/measurements/issue-839-raw/golden.txt`).
const GOLDEN: &[(&str, &str)] = &[
    ("step f16 hidden=37 k=2 b=2 out", "8cdf5b20c0383f3f"),
    ("step f16 hidden=37 k=2 b=2 state", "cf4cc264cab0cfc0"),
    ("step f16 hidden=37 k=3 b=2 out", "2b5d3c1cb78218d0"),
    ("step f16 hidden=37 k=3 b=2 state", "56127b3c288a174e"),
    ("step f16 hidden=37 k=4 b=1 out", "e7c04c62f8cbc40f"),
    ("step f16 hidden=37 k=4 b=1 state", "749120c0dc9a3f7c"),
    ("step f16 hidden=2048 k=3 b=1 out", "07dfa7b5b69ef642"),
    ("step f16 hidden=2048 k=3 b=1 state", "aee3c2c638e749db"),
    (
        "ring f16 hidden=37 k=3 width=4 seq=1 phase=2 b=2 out",
        "0cb42f33398a2dc3",
    ),
    (
        "ring f16 hidden=37 k=3 width=4 seq=1 phase=2 b=2 state",
        "1568286769635c17",
    ),
    (
        "ring f16 hidden=37 k=3 width=9 seq=9 phase=4 b=1 out",
        "8c6ff58bbecbcb04",
    ),
    (
        "ring f16 hidden=37 k=3 width=9 seq=9 phase=4 b=1 state",
        "06d8cdbbb43d1f33",
    ),
    (
        "ring f16 hidden=37 k=4 width=7 seq=3 phase=1 b=2 out",
        "7c23610c2fac454e",
    ),
    (
        "ring f16 hidden=37 k=4 width=7 seq=3 phase=1 b=2 state",
        "c68dfdfb31b3cd7c",
    ),
    (
        "ring f16 hidden=2048 k=3 width=4 seq=1 phase=3 b=1 out",
        "7ee71f4ff95ac1b3",
    ),
    (
        "ring f16 hidden=2048 k=3 width=4 seq=1 phase=3 b=1 state",
        "0d527fa2102e7f9e",
    ),
    (
        "ring f16 hidden=2048 k=3 width=10 seq=10 phase=6 b=1 out",
        "d92ad3413bf83cdf",
    ),
    (
        "ring f16 hidden=2048 k=3 width=10 seq=10 phase=6 b=1 state",
        "725a595d748ad127",
    ),
    (
        "ring-rows f16 hidden=37 k=3 width=5 out",
        "c97115f9d9b4b809",
    ),
    (
        "ring-rows f16 hidden=37 k=3 width=5 state",
        "1e82da3147b8ba2a",
    ),
    ("step f32 hidden=37 k=2 b=2 out", "7d7edf8b273214ad"),
    ("step f32 hidden=37 k=2 b=2 state", "53a5af3584027951"),
    ("step f32 hidden=37 k=3 b=2 out", "0a135c48a2a5a60f"),
    ("step f32 hidden=37 k=3 b=2 state", "68a020aa2da09580"),
    ("step f32 hidden=37 k=4 b=1 out", "e18fa89238264b73"),
    ("step f32 hidden=37 k=4 b=1 state", "03be0c32d829bb9c"),
    ("step f32 hidden=2048 k=3 b=1 out", "467dde09896248d1"),
    ("step f32 hidden=2048 k=3 b=1 state", "05964863860930b2"),
    (
        "ring f32 hidden=37 k=3 width=4 seq=1 phase=2 b=2 out",
        "461bfbc83cc09715",
    ),
    (
        "ring f32 hidden=37 k=3 width=4 seq=1 phase=2 b=2 state",
        "83a29d07e1a4b2a0",
    ),
    (
        "ring f32 hidden=37 k=3 width=9 seq=9 phase=4 b=1 out",
        "aa6f61b2281ee768",
    ),
    (
        "ring f32 hidden=37 k=3 width=9 seq=9 phase=4 b=1 state",
        "08a0c226025a60b9",
    ),
    (
        "ring f32 hidden=37 k=4 width=7 seq=3 phase=1 b=2 out",
        "b5b3f62fc085cced",
    ),
    (
        "ring f32 hidden=37 k=4 width=7 seq=3 phase=1 b=2 state",
        "f82078a836bf83ed",
    ),
    (
        "ring f32 hidden=2048 k=3 width=4 seq=1 phase=3 b=1 out",
        "e3533172e8a2c97e",
    ),
    (
        "ring f32 hidden=2048 k=3 width=4 seq=1 phase=3 b=1 state",
        "af3d8d5c545190b9",
    ),
    (
        "ring f32 hidden=2048 k=3 width=10 seq=10 phase=6 b=1 out",
        "091b4b8c8e62b7dd",
    ),
    (
        "ring f32 hidden=2048 k=3 width=10 seq=10 phase=6 b=1 state",
        "d96d00818fc4062f",
    ),
    (
        "ring-rows f32 hidden=37 k=3 width=5 out",
        "edfb9523b8a8679c",
    ),
    (
        "ring-rows f32 hidden=37 k=3 width=5 state",
        "53cf3c6795c6c06a",
    ),
    ("step bf16 hidden=37 k=2 b=2 out", "67e3b303d6ebfe53"),
    ("step bf16 hidden=37 k=2 b=2 state", "3dc66bd1d9e570e3"),
    ("step bf16 hidden=37 k=3 b=2 out", "b7a36e1e6a23d623"),
    ("step bf16 hidden=37 k=3 b=2 state", "2dd366d69893ea17"),
    ("step bf16 hidden=37 k=4 b=1 out", "0e8858ccc31ab2fb"),
    ("step bf16 hidden=37 k=4 b=1 state", "c6d4e91040de09b4"),
    ("step bf16 hidden=2048 k=3 b=1 out", "e5c76d5155cbf170"),
    ("step bf16 hidden=2048 k=3 b=1 state", "d83eb680b14967cd"),
    (
        "ring bf16 hidden=37 k=3 width=4 seq=1 phase=2 b=2 out",
        "4e97489676f1422f",
    ),
    (
        "ring bf16 hidden=37 k=3 width=4 seq=1 phase=2 b=2 state",
        "6be2ecf11eb4e384",
    ),
    (
        "ring bf16 hidden=37 k=3 width=9 seq=9 phase=4 b=1 out",
        "16266e0761a9d340",
    ),
    (
        "ring bf16 hidden=37 k=3 width=9 seq=9 phase=4 b=1 state",
        "6ec1b58bbb65d2c9",
    ),
    (
        "ring bf16 hidden=37 k=4 width=7 seq=3 phase=1 b=2 out",
        "0cb13eefd7ba104c",
    ),
    (
        "ring bf16 hidden=37 k=4 width=7 seq=3 phase=1 b=2 state",
        "9893ba05e95b0c7f",
    ),
    (
        "ring bf16 hidden=2048 k=3 width=4 seq=1 phase=3 b=1 out",
        "463c1a75cedec8ba",
    ),
    (
        "ring bf16 hidden=2048 k=3 width=4 seq=1 phase=3 b=1 state",
        "4c04d6c96e9d1bb1",
    ),
    (
        "ring bf16 hidden=2048 k=3 width=10 seq=10 phase=6 b=1 out",
        "70ced577f75561dd",
    ),
    (
        "ring bf16 hidden=2048 k=3 width=10 seq=10 phase=6 b=1 state",
        "500616209e4584dc",
    ),
    (
        "ring-rows bf16 hidden=37 k=3 width=5 out",
        "e8e42f1ae8f16c93",
    ),
    (
        "ring-rows bf16 hidden=37 k=3 width=5 state",
        "cfd687ad91e90eb1",
    ),
    ("gdn heads16x48-d128-k4 decode", "56ef16db7616f444"),
    ("gdn heads16x48-d128-k4 decode state", "96292100888e7ca8"),
    ("gdn heads16x48-d128-k4 prefill", "6b6b6f6c30bccddd"),
    ("gdn heads16x48-d128-k4 prefill state", "566bea49e41be0f2"),
    (
        "gdn heads16x48-d128-k4 decode after prefill",
        "b3b9b332529b4e47",
    ),
    ("gdn heads16x48-d128-k4 rows", "f6e12482cfd51213"),
    ("gdn heads16x48-d128-k4 rows state", "8f991a37ea052e4d"),
    ("gdn heads16x48-d128-k4 seq", "39b8d0a59639e6e6"),
    ("gdn heads16x48-d128-k4 seq pools", "6209b2fad1c5ace7"),
    ("gdn heads2x6-d24-k3 decode", "4bfa92498c634325"),
    ("gdn heads2x6-d24-k3 decode state", "cb9876996383ec03"),
    ("gdn heads2x6-d24-k3 prefill", "e11009398d1e3b1f"),
    ("gdn heads2x6-d24-k3 prefill state", "992e5e06a8963597"),
    (
        "gdn heads2x6-d24-k3 decode after prefill",
        "5a050afbd71acf41",
    ),
    ("gdn heads2x6-d24-k3 rows", "eb364625f3eb22ab"),
    ("gdn heads2x6-d24-k3 rows state", "2048d3f4868d6dd0"),
    ("gdn heads2x6-d24-k3 seq", "8e8f82ae7c31a4c4"),
    ("gdn heads2x6-d24-k3 seq pools", "68a5540f0ff196cf"),
    ("gdn heads16x48-d128-k4 register decode", "0045079f925f38de"),
    (
        "gdn heads16x48-d128-k4 register decode state",
        "43eb30165271b56f",
    ),
    (
        "gdn heads16x48-d128-k4 register prefill",
        "6b6b6f6c30bccddd",
    ),
    (
        "gdn heads16x48-d128-k4 register prefill state",
        "566bea49e41be0f2",
    ),
    (
        "gdn heads16x48-d128-k4 register decode after prefill",
        "4b2d3e9399798ad3",
    ),
    ("gdn heads16x48-d128-k4 register rows", "f328eca06a4f53c0"),
    (
        "gdn heads16x48-d128-k4 register rows state",
        "7a0d842ad0fe6066",
    ),
    ("gdn heads16x48-d128-k4 register seq", "ecfcacda3a6fa4c2"),
    (
        "gdn heads16x48-d128-k4 register seq pools",
        "a903bfed10e23106",
    ),
    ("gdn heads2x6-d24-k3 register decode", "9ac02ea26a7ee376"),
    (
        "gdn heads2x6-d24-k3 register decode state",
        "91c8617d3fc43bfc",
    ),
    ("gdn heads2x6-d24-k3 register prefill", "e11009398d1e3b1f"),
    (
        "gdn heads2x6-d24-k3 register prefill state",
        "992e5e06a8963597",
    ),
    (
        "gdn heads2x6-d24-k3 register decode after prefill",
        "556e6009be1e2a0a",
    ),
    ("gdn heads2x6-d24-k3 register rows", "974aca3c8fe03982"),
    (
        "gdn heads2x6-d24-k3 register rows state",
        "47ae84353b3f156c",
    ),
    ("gdn heads2x6-d24-k3 register seq", "82c110ce53c6ec0a"),
    ("gdn heads2x6-d24-k3 register seq pools", "a98d06de3aefbc64"),
];

#[test]
fn the_blocks_compute_what_the_kernels_they_replaced_did() -> Result<()> {
    let dev = Device::new_metal(0)?;
    // The fixtures run bf16 too, which the GPU may only emulate (§15.2a).
    lloom_tensor::test_utils::allow_emulated(&dev, DType::BF16);
    let mut got = Vec::new();
    lfm2_fixtures(&dev, &mut got)?;
    for form in [GdnStepForm::Loop, GdnStepForm::Register] {
        for (name, cfg) in gdn_configs() {
            gdn_fixtures(name, &cfg, form, &dev, &mut got)?;
        }
    }
    for (name, h) in &got {
        println!("GOLDEN {h} {name}");
    }
    if std::env::var("LLOOM_GOLDEN_PRINT").is_ok() {
        return Ok(());
    }
    assert_eq!(
        got.len(),
        GOLDEN.len(),
        "the fixture list and the pinned table differ in length"
    );
    let moved: Vec<String> = got
        .iter()
        .zip(GOLDEN)
        .filter_map(|((name, h), (pname, ph))| {
            assert_eq!(name, pname, "fixture order");
            (h != ph).then(|| format!("{name}: {h} (pinned {ph})"))
        })
        .collect();
    assert!(
        moved.is_empty(),
        "{} of {} fixtures moved:\n{}",
        moved.len(),
        got.len(),
        moved.join("\n")
    );
    Ok(())
}

/// The fixtures are deterministic run to run (CONTRIBUTING §3.1 #3): the
/// same hashes twice in one process.
#[test]
fn the_fixtures_are_deterministic() -> Result<()> {
    let dev = Device::new_metal(0)?;
    // The fixtures run bf16 too, which the GPU may only emulate (§15.2a).
    lloom_tensor::test_utils::allow_emulated(&dev, DType::BF16);
    let run = || -> Result<Vec<(String, String)>> {
        let mut v = Vec::new();
        lfm2_fixtures(&dev, &mut v)?;
        for form in [GdnStepForm::Loop, GdnStepForm::Register] {
            for (name, cfg) in gdn_configs() {
                gdn_fixtures(name, &cfg, form, &dev, &mut v)?;
            }
        }
        Ok(v)
    };
    assert_eq!(run()?, run()?);
    Ok(())
}
