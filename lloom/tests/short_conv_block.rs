//! The short-conv block against the CPU, at geometries neither model uses —
//! the reuse claim tested rather than asserted.
//!
//! The block ([`ShortConv`]) is one type over two layouts and serves two
//! models: LFM2's gated conv (the shift layout's `gated_step`, the ring's
//! `gated_pass`) and Gated DeltaNet's prologue (the shift layout inside
//! `GatedDeltaNet`'s launches). Each is held here to a reference computed on
//! the CPU backend in f32 from the same inputs, at k ∈ {3, 4}, f16 and f32
//! where the kernel has both, a channel count (37) no model has, and ring
//! widths no model allocates. The window a step leaves is data movement and
//! is held bit for bit; the fold within a relative tolerance of the dtype.
//!
//! The bit-level contract against the pre-block kernels is
//! `short_conv_golden.rs`'s; this file is what says the block computes the
//! convolution at all at a new geometry.
#![cfg(feature = "metal")]

use lloom::models::qwen4exp::{Config, GatedDeltaNet};
use lloom::nn::short_conv::{Ring, Shift, ShortConv};
use lloom::nn::{RecurrentState, StatePool};
use lloom::profile::CrownExact;
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{DType, Device, Result, Tensor};

fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 2654435761 + seed * 40503) % 1000) as f32) / 500.0 - 1.0)
        .collect()
}

fn host(t: &Tensor) -> Result<Vec<f32>> {
    t.to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()
}

/// Round an f32 through `dtype` and back (what storing it as `dtype` does).
fn through(v: f32, dtype: DType) -> Result<f32> {
    Tensor::new(&[v], &Device::Cpu)?
        .to_dtype(dtype)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()
        .map(|x| x[0])
}

fn tol(dtype: DType) -> f32 {
    match dtype {
        DType::F32 => 2e-6,
        _ => 4e-3,
    }
}

fn close(name: &str, got: &[f32], want: &[f32], rel: f32) {
    assert_eq!(got.len(), want.len(), "{name}: lengths");
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
    let (i, diff) = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .enumerate()
        .fold((0, 0f32), |m, (i, d)| if d > m.1 { (i, d) } else { m });
    assert!(
        diff <= rel * scale,
        "{name}: max diff {diff} at {i} (got {} want {}) over scale {scale}",
        got[i],
        want[i]
    );
    assert!(scale > 1e-3, "{name}: a degenerate fixture");
}

struct Proj {
    b: Tensor,
    c: Tensor,
    x: Tensor,
}

/// `b`, `c`, `x` narrowed from one `[bsz, 3·hidden, seq]` projection, as LFM2 does.
fn proj(
    bsz: usize,
    hidden: usize,
    seq: usize,
    seed: usize,
    dtype: DType,
    dev: &Device,
) -> Result<Proj> {
    let all = Tensor::from_vec(
        fill(bsz * 3 * hidden * seq, seed),
        (bsz, 3 * hidden, seq),
        dev,
    )?
    .to_dtype(dtype)?;
    Ok(Proj {
        b: all.narrow(1, 0, hidden)?,
        c: all.narrow(1, hidden, hidden)?,
        x: all.narrow(1, 2 * hidden, hidden)?,
    })
}

/// The shift layout's gated step: the window it leaves bit for bit, the gated
/// fold against `Σ_j win'[j]·w[j] · c` in f32 on the CPU.
#[test]
fn the_shift_layout_is_the_causal_conv() -> Result<()> {
    let dev = Device::new_metal(0)?;
    for dtype in [DType::F16, DType::F32] {
        for k in [3usize, 4] {
            let (bsz, hidden) = (2usize, 37usize);
            let p = proj(bsz, hidden, 1, 3 + k, dtype, &dev)?;
            let w =
                Tensor::from_vec(fill(hidden * k, 7 + k), (hidden, k), &dev)?.to_dtype(dtype)?;
            let state = Tensor::from_vec(fill(bsz * hidden * k, 11 + k), (bsz, hidden, k), &dev)?
                .to_dtype(dtype)?;
            let conv = ShortConv::<Shift>::new(w.clone(), k);
            let (out, next) = conv.gated_step(&p.b, &p.x, &p.c, &state)?;
            let (bv, cv, xv) = (host(&p.b)?, host(&p.c)?, host(&p.x)?);
            let (wv, sv) = (host(&w)?, host(&state)?);
            let (mut want_next, mut want_out) = (Vec::new(), Vec::new());
            for r in 0..bsz {
                for ch in 0..hidden {
                    let i = r * hidden + ch;
                    let bx = through(bv[i] * xv[i], dtype)?;
                    let win: Vec<f32> = (0..k)
                        .map(|j| if j + 1 < k { sv[i * k + j + 1] } else { bx })
                        .collect();
                    let fold: f32 = (0..k).map(|j| win[j] * wv[ch * k + j]).sum();
                    want_next.extend(win);
                    want_out.push(fold * cv[i]);
                }
            }
            assert_eq!(
                host(&next)?,
                want_next,
                "{dtype:?} k={k}: the window after the step"
            );
            close(
                &format!("{dtype:?} k={k} out"),
                &host(&out)?,
                &want_out,
                tol(dtype),
            );
        }
    }
    Ok(())
}

/// The ring layout's gated pass: each position written to its phase slot in
/// place, the gated fold over the live `k` inputs.
#[test]
fn the_ring_layout_is_the_causal_conv() -> Result<()> {
    let dev = Device::new_metal(0)?;
    for dtype in [DType::F16, DType::F32] {
        for (k, width, seq, phase_last) in [
            (3usize, 3usize, 3usize, 1usize),
            (3, 7, 2, 0),
            (4, 9, 9, 5),
            (4, 5, 1, 3),
        ] {
            let (bsz, hidden) = (2usize, 37usize);
            let p = proj(bsz, hidden, seq, 13 + width, dtype, &dev)?;
            let w =
                Tensor::from_vec(fill(hidden * k, 17 + k), (hidden, k), &dev)?.to_dtype(dtype)?;
            let ring = Tensor::from_vec(
                fill(bsz * hidden * width, 19 + width),
                (bsz, hidden, width),
                &dev,
            )?
            .to_dtype(dtype)?;
            let before = host(&ring)?;
            let conv = ShortConv::<Ring>::new(w.clone(), width)?;
            let out = conv.gated_pass(&p.b, &p.x, &p.c, &ring, phase_last)?;
            let (bv, cv, xv, wv) = (host(&p.b)?, host(&p.c)?, host(&p.x)?, host(&w)?);
            let mut want_ring = before.clone();
            let mut want_out = vec![0f32; bsz * seq * hidden];
            for r in 0..bsz {
                for ch in 0..hidden {
                    let row = (r * hidden + ch) * width;
                    for i in 0..seq {
                        let phase = (phase_last + width - (seq - 1 - i)) % width;
                        let pi = (r * hidden + ch) * seq + i;
                        want_ring[row + phase] = through(bv[pi] * xv[pi], dtype)?;
                        // Tap j reads the input k−1−j before the newest.
                        let fold: f32 = (0..k)
                            .map(|j| {
                                want_ring[row + (phase + width - (k - 1 - j)) % width]
                                    * wv[ch * k + j]
                            })
                            .sum();
                        want_out[(r * seq + i) * hidden + ch] = through(fold, dtype)? * cv[pi];
                    }
                }
            }
            let tag = format!("{dtype:?} k={k} width={width} seq={seq}");
            assert_eq!(host(&ring)?, want_ring, "{tag}: the ring after the pass");
            close(&format!("{tag} out"), &host(&out)?, &want_out, tol(dtype));
        }
    }
    Ok(())
}

fn lin(out: usize, inp: usize, seed: usize, dev: &Device) -> Result<QMatMul> {
    let v: Vec<f32> = fill(out * inp, seed)
        .into_iter()
        .map(|x| x * 0.05)
        .collect();
    Ok(QMatMul::Tensor(Tensor::from_vec(v, (out, inp), dev)?))
}

fn gdn(cfg: &Config, dev: &Device) -> Result<GatedDeltaNet<CrownExact>> {
    let d = cfg.hidden_size;
    let conv_ch = cfg.conv_channels();
    let value_dim = cfg.ssm_state_size * cfg.ssm_dt_rank;
    let h_v = cfg.ssm_dt_rank;
    let ramp = |n: usize, f: &dyn Fn(usize) -> f32| {
        Tensor::from_vec((0..n).map(f).collect::<Vec<f32>>(), n, dev)
    };
    let conv_w: Vec<f32> = fill(conv_ch * cfg.ssm_conv_kernel, 3)
        .into_iter()
        .map(|x| x * 0.3)
        .collect();
    Ok(GatedDeltaNet::<CrownExact>::new(
        lin(conv_ch, d, 1, dev)?,
        lin(value_dim, d, 2, dev)?,
        Tensor::from_vec(conv_w, (conv_ch, cfg.ssm_conv_kernel), dev)?,
        ramp(h_v, &|i| -0.5 - 0.01 * i as f32)?,
        ramp(h_v, &|i| 0.1 * (i % 7) as f32)?,
        lin(h_v, d, 4, dev)?,
        lin(h_v, d, 5, dev)?,
        ramp(cfg.ssm_state_size, &|i| 1.0 + 0.01 * (i % 5) as f32)?,
        lin(d, value_dim, 6, dev)?,
        cfg,
    ))
}

fn x_at(d: usize, t: usize, dev: &Device) -> Result<Tensor> {
    let v: Vec<f32> = fill(d, 100 + t).into_iter().map(|x| x * 0.6).collect();
    Tensor::from_vec(v, (1, 1, d), dev)
}

/// Gated DeltaNet on Metal (the prologue on the block, the step block) against
/// its composed CPU path, every form: one row, pooled rows, a window of
/// positions, the chunked prefill — outputs and both state channels.
#[test]
fn gdn_on_the_block_matches_the_cpu_at_new_geometries() -> Result<()> {
    let (cpu, gpu) = (Device::Cpu, Device::new_metal(0)?);
    // (name, hidden, d, h_k, h_v, k): no model has either.
    for (name, hidden, d, h_k, h_v, k) in [
        ("d24 2->6 k3", 40usize, 24usize, 2usize, 6usize, 3usize),
        ("d16 3->6 k4", 32, 16, 3, 6, 4),
    ] {
        let mut cfg = Config::flash_next();
        cfg.hidden_size = hidden;
        cfg.ssm_state_size = d;
        cfg.ssm_n_group = h_k;
        cfg.ssm_dt_rank = h_v;
        cfg.ssm_conv_kernel = k;
        let conv_ch = cfg.conv_channels();
        let (bc, bg) = (gdn(&cfg, &cpu)?, gdn(&cfg, &gpu)?);
        // The chunked prefill reorders its sums against the CPU's chunk path
        // more than a step does (`deltanet_block`'s own Metal-vs-CPU test
        // holds it at the same 2e-3).
        let same = |what: &str, a: &Tensor, b: &Tensor| -> Result<()> {
            let rel = if what.starts_with("prefill") {
                2e-3
            } else {
                2e-4
            };
            close(&format!("{name} {what}"), &host(b)?, &host(a)?, rel);
            Ok(())
        };

        // One row, then the chunked prefill on the same channels.
        let (mut sc, mut cc) = (RecurrentState::new(1), RecurrentState::new(1));
        let (mut sg, mut cg) = (RecurrentState::new(1), RecurrentState::new(1));
        for t in 0..12 {
            let a = bc.decode(&x_at(hidden, t, &cpu)?, 0, &mut sc, &mut cc)?;
            let b = bg.decode(&x_at(hidden, t, &gpu)?, 0, &mut sg, &mut cg)?;
            same(&format!("decode {t}"), &a, &b)?;
        }
        same("decode window", cc.get(0).unwrap(), cg.get(0).unwrap())?;
        same("decode S", sc.get(0).unwrap(), sg.get(0).unwrap())?;
        let seq = 70;
        let xs: Vec<f32> = fill(seq * hidden, 9).into_iter().map(|x| x * 0.6).collect();
        let a = bc.prefill(
            &Tensor::from_vec(xs.clone(), (1, seq, hidden), &cpu)?,
            0,
            &mut sc,
            &mut cc,
        )?;
        let b = bg.prefill(
            &Tensor::from_vec(xs, (1, seq, hidden), &gpu)?,
            0,
            &mut sg,
            &mut cg,
        )?;
        same("prefill", &a, &b)?;
        same("prefill window", cc.get(0).unwrap(), cg.get(0).unwrap())?;
        same("prefill S", sc.get(0).unwrap(), sg.get(0).unwrap())?;

        // Two pooled rows (the rows kernels) against two CPU rows.
        let s_pool = StatePool::new(2, 1, &[d, d, h_v], DType::F32, &gpu);
        let c_pool = StatePool::new(2, 1, &[conv_ch, k], DType::F32, &gpu);
        let mut rg: Vec<(RecurrentState, RecurrentState)> = (0..2)
            .map(|_| {
                Ok((
                    RecurrentState::in_pool(&s_pool)?,
                    RecurrentState::in_pool(&c_pool)?,
                ))
            })
            .collect::<Result<_>>()?;
        let mut rc: Vec<(RecurrentState, RecurrentState)> = (0..2)
            .map(|_| (RecurrentState::new(1), RecurrentState::new(1)))
            .collect();
        for r in 0..2 {
            let (s, c) = &mut rg[r];
            bg.decode(&x_at(hidden, 20 + r, &gpu)?, 0, s, c)?;
            let (s, c) = &mut rc[r];
            bc.decode(&x_at(hidden, 20 + r, &cpu)?, 0, s, c)?;
        }
        for t in 0..5 {
            let xg = Tensor::cat(
                &[&x_at(hidden, 30 + t, &gpu)?, &x_at(hidden, 40 + t, &gpu)?],
                0,
            )?;
            let mut st: Vec<(&mut RecurrentState, &mut RecurrentState)> =
                rg.iter_mut().map(|(a, b)| (a, b)).collect();
            let og = bg.decode_rows(&xg, 0, &mut st)?;
            for r in 0..2 {
                let (s, c) = &mut rc[r];
                let oc = bc.decode(&x_at(hidden, [30, 40][r] + t, &cpu)?, 0, s, c)?;
                same(&format!("rows step {t} row {r}"), &oc, &og.narrow(0, r, 1)?)?;
            }
        }
        for r in 0..2 {
            same(
                &format!("rows window {r}"),
                rc[r].1.get(0).unwrap(),
                rg[r].1.get(0).unwrap(),
            )?;
            same(
                &format!("rows S {r}"),
                rc[r].0.get(0).unwrap(),
                rg[r].0.get(0).unwrap(),
            )?;
        }

        // A window of three positions over a four-phase pool, against three
        // CPU steps; then an accept of two by phase is the state after two.
        let s_pool = StatePool::with_phases(1, 4, 1, &[d, d, h_v], DType::F32, &gpu);
        let c_pool = StatePool::with_phases(1, 4, 1, &[conv_ch, k], DType::F32, &gpu);
        let (mut sg, mut cg) = (
            RecurrentState::in_pool(&s_pool)?,
            RecurrentState::in_pool(&c_pool)?,
        );
        let (mut sc, mut cc) = (RecurrentState::new(1), RecurrentState::new(1));
        for t in 0..2 {
            bg.decode(&x_at(hidden, 50 + t, &gpu)?, 0, &mut sg, &mut cg)?;
            bc.decode(&x_at(hidden, 50 + t, &cpu)?, 0, &mut sc, &mut cc)?;
        }
        sg.open_window()?;
        cg.open_window()?;
        let xs = Tensor::cat(
            &[
                &x_at(hidden, 60, &gpu)?,
                &x_at(hidden, 61, &gpu)?,
                &x_at(hidden, 62, &gpu)?,
            ],
            0,
        )?;
        let og = bg.decode_seq(&xs, 0, &mut sg, &mut cg)?;
        let mut after_two = None;
        for t in 0..3 {
            let oc = bc.decode(&x_at(hidden, 60 + t, &cpu)?, 0, &mut sc, &mut cc)?;
            same(&format!("window position {t}"), &oc, &og.narrow(0, t, 1)?)?;
            if t == 1 {
                after_two = Some((sc.get(0).unwrap().copy()?, cc.get(0).unwrap().copy()?));
            }
        }
        sg.accept_window(2)?;
        cg.accept_window(2)?;
        let (s2, c2) = after_two.expect("two positions ran");
        same("accept 2 window", &c2, cg.get(0).unwrap())?;
        same("accept 2 S", &s2, sg.get(0).unwrap())?;
    }
    Ok(())
}
