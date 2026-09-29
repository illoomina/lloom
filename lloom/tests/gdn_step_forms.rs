//! One arithmetic per form of the Gated DeltaNet step. In each form
//! ([`LoopForm`], [`RegisterForm`]) a row stepped alone and the same row among
//! the rows of a pool compute the same bits, and so do the positions of a
//! window stepped one launch at a time and in one multi-position launch: a
//! sequence's output does not depend on whether it was batched or verified in
//! a window (`docs/architecture.md` §2.3.3 #7). The two forms differ from each
//! other by rounding (§2.3.5a); that is checked too, or the comparisons above
//! could not tell a form from the other.
//!
//! And both against an f64 CPU reference of the recurrence (§2.3.5a's
//! parity): neither form is the exact answer, and the reference says how far
//! each is from it.
//!
//! At Qwen4-exp's head geometry (`d` 128, 48 value heads) and at two no model
//! has (`d` 24 and `d` 136, 6 heads): at the default split of 8 slices a head,
//! `d` 128 and 24 take the 16-value register kernels and `d` 136 the 32-value
//! ones.
#![cfg(feature = "metal")]

use lloom::nn::gdn::{GdnStep, LoopForm, RegisterForm, StepForm};
use lloom::nn::StatePool;
use lloom_tensor::{DType, Device, Result, Tensor};

fn fill(n: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((((i * 2654435761 + seed * 40503) % 1000) as f32) / 500.0 - 1.0) * scale)
        .collect()
}

fn host(t: &Tensor) -> Result<Vec<f32>> {
    t.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()
}

fn bits(t: &Tensor) -> Result<Vec<u32>> {
    Ok(host(t)?.into_iter().map(f32::to_bits).collect())
}

/// The first element two bit patterns disagree at, if any.
fn first_diff(a: &[u32], b: &[u32]) -> Option<usize> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).position(|(x, y)| x != y)
}

struct Inputs {
    d: usize,
    h_v: usize,
    /// Per row / position: `heads [3, h_v, d]`, `alpha`, `beta` `[h_v]`.
    heads: Vec<Vec<f32>>,
    alpha: Vec<Vec<f32>>,
    beta: Vec<Vec<f32>>,
    /// Per row, its `S [d, d, h_v]` before the step.
    s0: Vec<Vec<f32>>,
    dt_bias: Tensor,
    ssm_a: Tensor,
}

impl Inputs {
    fn new(d: usize, h_v: usize, n: usize, dev: &Device) -> Result<Self> {
        let ssm_a: Vec<f32> = (0..h_v)
            .map(|h| -0.05 - 0.4 * (h as f32) / h_v as f32)
            .collect();
        Ok(Self {
            d,
            h_v,
            heads: (0..n).map(|r| fill(3 * h_v * d, 10 + r, 0.3)).collect(),
            alpha: (0..n).map(|r| fill(h_v, 40 + r, 2.0)).collect(),
            beta: (0..n).map(|r| fill(h_v, 70 + r, 3.0)).collect(),
            s0: (0..n).map(|r| fill(d * d * h_v, 100 + r, 0.5)).collect(),
            dt_bias: Tensor::from_vec(fill(h_v, 7, 0.5), h_v, dev)?,
            ssm_a: Tensor::from_vec(ssm_a, h_v, dev)?,
        })
    }

    fn slab(&self) -> usize {
        self.d * self.d * self.h_v
    }

    /// Rows (or positions) `range` stacked: `heads [n, 3, h_v, d]`, `alpha` /
    /// `beta` `[n, h_v]`.
    fn stacked(&self, n: usize, dev: &Device) -> Result<(Tensor, Tensor, Tensor)> {
        let (d, h_v) = (self.d, self.h_v);
        Ok((
            Tensor::from_vec(self.heads[..n].concat(), (n, 3, h_v, d), dev)?,
            Tensor::from_vec(self.alpha[..n].concat(), (n, h_v), dev)?,
            Tensor::from_vec(self.beta[..n].concat(), (n, h_v), dev)?,
        ))
    }
}

/// Row `r` alone from `s`: `(o, S after)`.
fn step_alone<F: StepForm>(
    g: &GdnStep<F>,
    x: &Inputs,
    r: usize,
    s: &Tensor,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let (d, h_v) = (x.d, x.h_v);
    let s_out = Tensor::zeros((d, d, h_v), DType::F32, dev)?;
    let o = g.step(
        &Tensor::from_vec(x.heads[r].clone(), (3, h_v, d), dev)?,
        &Tensor::from_vec(x.alpha[r].clone(), h_v, dev)?,
        &Tensor::from_vec(x.beta[r].clone(), h_v, dev)?,
        s,
        &s_out,
    )?;
    Ok((o, s_out))
}

/// A pool of `slots` rows and `phases` phases whose phase 0 holds `s0[..slots]`.
fn pool_with(x: &Inputs, slots: usize, phases: usize, dev: &Device) -> Result<StatePool> {
    let (d, h_v) = (x.d, x.h_v);
    let pool = StatePool::with_phases(slots, phases, 1, &[d, d, h_v], DType::F32, dev);
    let mut v = x.s0[..slots].concat();
    v.resize(phases * slots * x.slab(), 0.0);
    pool.layer(0)?
        .slice_set(&Tensor::from_vec(v, (phases, slots, d, d, h_v), dev)?, 0, 0)?;
    Ok(pool)
}

/// In form `F`: every row alone against the rows launch, and a window's
/// positions one launch each against the multi-position launch — bit for bit.
/// Returns the rows launch's output and states, for the cross-form check.
fn one_arithmetic<F: StepForm>(x: &Inputs, dev: &Device, tag: &str) -> Result<Vec<u32>> {
    let (d, h_v) = (x.d, x.h_v);
    let rows = 3usize;
    let g = GdnStep::<F>::new(d, h_v, &x.dt_bias, &x.ssm_a);

    // The rows launch: rows at slots 0.., every one at phase 0.
    let pool = pool_with(x, rows, 2, dev)?;
    let (heads, alpha, beta) = x.stacked(rows, dev)?;
    let slots = Tensor::from_vec((0..rows as u32).collect::<Vec<_>>(), rows, dev)?;
    let phases = Tensor::from_vec(vec![0u32; rows], rows, dev)?;
    let o_rows = g.rows(&heads, &alpha, &beta, pool.pool_layer(0)?, slots, phases)?;
    let after = pool.layer(0)?.get(1)?;
    for r in 0..rows {
        let s = Tensor::from_vec(x.s0[r].clone(), (d, d, h_v), dev)?;
        let (o, s_out) = step_alone(&g, x, r, &s, dev)?;
        let (a, b) = (bits(&o)?, bits(&o_rows.get(r)?)?);
        assert_eq!(
            first_diff(&a, &b),
            None,
            "{tag} {:?}: row {r}'s output alone and among the rows",
            F::FORM
        );
        let (a, b) = (bits(&s_out)?, bits(&after.get(r)?)?);
        assert_eq!(
            first_diff(&a, &b),
            None,
            "{tag} {:?}: row {r}'s state alone and among the rows",
            F::FORM
        );
    }
    let mut out = bits(&o_rows)?;
    out.extend(bits(&after)?);

    // The window: `steps` positions of slot 0 from phase 0.
    let steps = 3usize;
    let pool = pool_with(x, 1, steps + 1, dev)?;
    let (heads, alpha, beta) = x.stacked(steps, dev)?;
    let o_seq = g.seq(&heads, &alpha, &beta, pool.pool_layer(0)?, 0, 0)?;
    let mut s = Tensor::from_vec(x.s0[0].clone(), (d, d, h_v), dev)?;
    for t in 0..steps {
        let (o, s_out) = step_alone(&g, x, t, &s, dev)?;
        let (a, b) = (bits(&o)?, bits(&o_seq.get(t)?)?);
        assert_eq!(
            first_diff(&a, &b),
            None,
            "{tag} {:?}: position {t}'s output stepped and in the window",
            F::FORM
        );
        let phase = pool.layer(0)?.get(t + 1)?.get(0)?;
        let (a, b) = (bits(&s_out)?, bits(&phase)?);
        assert_eq!(
            first_diff(&a, &b),
            None,
            "{tag} {:?}: position {t}'s state stepped and in the window",
            F::FORM
        );
        s = s_out;
    }
    out.extend(bits(&o_seq)?);
    Ok(out)
}

/// One step of the recurrence in f64 for row `r` of `x` from `s` (`[d, d, h_v]`,
/// head-last): `(o [h_v, d], S after)`.
fn step_f64(x: &Inputs, r: usize, s: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
    let (d, h_v) = (x.d, x.h_v);
    let dt_bias = host(&x.dt_bias)?;
    let ssm_a = host(&x.ssm_a)?;
    let heads = &x.heads[r];
    let (mut o, mut sn) = (vec![0f64; h_v * d], vec![0f64; d * d * h_v]);
    for h in 0..h_v {
        let a = x.alpha[r][h] as f64 + dt_bias[h] as f64;
        let sp = a.max(0.0) + (1.0 + (-a.abs()).exp()).ln();
        let eg = (ssm_a[h] as f64 * sp).exp();
        let b = 1.0 / (1.0 + (-(x.beta[r][h] as f64)).exp());
        let q = |i: usize| heads[h * d + i] as f64;
        let k = |i: usize| heads[(h_v + h) * d + i] as f64;
        let v = |i: usize| heads[(2 * h_v + h) * d + i] as f64;
        let at = |dk: usize, dv: usize| (dk * d + dv) * h_v + h;
        for dv in 0..d {
            let r_: f64 = (0..d).map(|dk| k(dk) * eg * s[at(dk, dv)]).sum();
            let delta = b * (v(dv) - r_);
            let mut acc = 0f64;
            for dk in 0..d {
                let n = eg * s[at(dk, dv)] + k(dk) * delta;
                sn[at(dk, dv)] = n;
                acc += q(dk) * n;
            }
            o[h * d + dv] = acc;
        }
    }
    Ok((o, sn))
}

/// What [`one_arithmetic`] returns, in f64: the rows' outputs, their states,
/// the window's outputs.
fn reference(x: &Inputs) -> Result<Vec<f64>> {
    let (mut o_rows, mut states) = (Vec::new(), Vec::new());
    for r in 0..3 {
        let s: Vec<f64> = x.s0[r].iter().map(|&v| v as f64).collect();
        let (o, sn) = step_f64(x, r, &s)?;
        o_rows.extend(o);
        states.extend(sn);
    }
    let mut s: Vec<f64> = x.s0[0].iter().map(|&v| v as f64).collect();
    let mut o_seq = Vec::new();
    for t in 0..3 {
        let (o, sn) = step_f64(x, t, &s)?;
        o_seq.extend(o);
        s = sn;
    }
    Ok([o_rows, states, o_seq].concat())
}

fn check(d: usize, h_v: usize, tag: &str) -> Result<()> {
    let dev = Device::new_metal(0)?;
    let x = Inputs::new(d, h_v, 3, &dev)?;
    let lp = one_arithmetic::<LoopForm>(&x, &dev, tag)?;
    let rg = one_arithmetic::<RegisterForm>(&x, &dev, tag)?;
    // The forms are two roundings of one recurrence: close, and not the same
    // bits, or the checks above ran one kernel under two names.
    // Relative to the values' scale, not per element: an element near zero
    // makes a per-element ratio say nothing about the arithmetic.
    let differ = lp.iter().zip(&rg).filter(|(a, b)| a != b).count();
    let f = |v: &u32| f32::from_bits(*v);
    let scale = lp.iter().map(|v| f(v).abs()).fold(0f32, f32::max);
    let max_d = lp
        .iter()
        .zip(&rg)
        .map(|(a, b)| (f(a) - f(b)).abs())
        .fold(0f32, f32::max);
    println!(
        "GDNFORMS {tag} elems={} loop_vs_register_differ={differ} max_abs_diff={max_d:e} scale={scale:e} rel_to_scale={:e}",
        lp.len(),
        max_d / scale
    );
    assert!(
        differ > 0,
        "{tag}: the register form computed the loop form's bits"
    );
    let exact = reference(&x)?;
    let err = |got: &[u32]| {
        got.iter()
            .zip(&exact)
            .map(|(g, e)| (f(g) as f64 - e).abs())
            .fold(0f64, f64::max)
    };
    let (e_loop, e_reg) = (err(&lp), err(&rg));
    println!(
        "GDNFORMS {tag} against f64: loop max_abs_err={e_loop:e} register max_abs_err={e_reg:e} (scale {scale:e})"
    );
    for (form, e) in [("loop", e_loop), ("register", e_reg)] {
        assert!(
            e <= 1e-5 * scale as f64,
            "{tag}: the {form} form is {e} from the f64 recurrence at scale {scale}"
        );
    }
    assert!(
        max_d <= 1e-5 * scale,
        "{tag}: the forms differ by {max_d} at scale {scale}"
    );
    Ok(())
}

#[test]
fn each_form_is_one_arithmetic_alone_batched_and_in_a_window_real_geometry() -> Result<()> {
    check(128, 48, "d128 h48")
}

#[test]
fn each_form_is_one_arithmetic_alone_batched_and_in_a_window_odd_geometries() -> Result<()> {
    check(24, 6, "d24 h6")?;
    check(136, 6, "d136 h6")
}
