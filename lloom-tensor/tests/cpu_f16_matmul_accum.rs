//! The CPU f16 matmul's accumulator axis (`cpu::f16_matmul`): each declared
//! arm against a scalar model of the rounding it declares, bit for bit, and
//! against the exact product-sum within the bound that rounding implies; the
//! `Tensor::matmul` entry routed to the arm the tuning names; a row's bits
//! independent of the rows beside it.
//!
//! The model is the oracle: eight f32 lanes along `k`, a fused multiply-add
//! per lane step (`f32/1`), or f16 chains of sixteen products per lane
//! promoted into those lanes (`f16/16`), one fixed tree, the scalar tail in
//! f32. Dropping the kernel's promotion (the mutation) makes it one f16 chain
//! per lane and breaks the identity at k ≥ 136.
use half::f16;
use lloom_tensor::cpu::f16_matmul::{declared, instruction, matmul_with, Kind, Operands};
use lloom_tensor::{DType, Device, Result, Tensor};

/// A fixed-seed generator, so every platform sees the same operands.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (u, v) = (self.unit().max(1e-300), self.unit());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// `x ~ N(0, 1)`, `w ~ N(0, 1/k)`: a projection, outputs O(1).
fn operands(m: usize, n: usize, k: usize, seed: u64) -> (Vec<f16>, Vec<f16>) {
    let mut r = Rng(seed ^ ((k as u64) << 20) ^ ((n as u64) << 40) ^ m as u64);
    let x: Vec<f16> = (0..m * k).map(|_| f16::from_f64(r.gauss())).collect();
    let s = 1.0 / (k as f64).sqrt();
    let w: Vec<f16> = (0..n * k).map(|_| f16::from_f64(r.gauss() * s)).collect();
    (x, w)
}

/// One f16 rounding of the exact `a·b + c` (the fused `fmla .8h`).
fn fma16(a: f16, b: f16, c: f16) -> f16 {
    f16::from_f64(a.to_f64() * b.to_f64() + c.to_f64())
}

/// The declared arithmetic of one output, in scalar Rust.
fn model_dot(kind: Kind, x: &[f16], w: &[f16]) -> f16 {
    let k = x.len();
    let k8 = k / 8 * 8;
    let mut lanes = [0f32; 8];
    match kind {
        Kind::F32 => {
            for kk in 0..k8 {
                let l = kk % 8;
                lanes[l] = x[kk].to_f32().mul_add(w[kk].to_f32(), lanes[l]);
            }
        }
        Kind::F16Chain16 => {
            let mut chain = [f16::ZERO; 8];
            let mut steps = 0;
            for kk in (0..k8).step_by(8) {
                for l in 0..8 {
                    chain[l] = fma16(x[kk + l], w[kk + l], chain[l]);
                }
                steps += 1;
                if steps == 16 {
                    for l in 0..8 {
                        lanes[l] += chain[l].to_f32();
                    }
                    chain = [f16::ZERO; 8];
                    steps = 0;
                }
            }
            for l in 0..8 {
                lanes[l] += chain[l].to_f32();
            }
        }
    }
    let s = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    let mut tail = 0f32;
    for kk in k8..k {
        tail = x[kk].to_f32().mul_add(w[kk].to_f32(), tail);
    }
    f16::from_f32(s + tail)
}

fn model(kind: Kind, m: usize, n: usize, k: usize, x: &[f16], w: &[f16]) -> Vec<f16> {
    let mut y = vec![f16::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            y[i * n + j] = model_dot(kind, &x[i * k..(i + 1) * k], &w[j * k..(j + 1) * k]);
        }
    }
    y
}

/// The exact product-sum over the same f16 operands, in f64.
fn exact(m: usize, n: usize, k: usize, x: &[f16], w: &[f16]) -> Vec<f64> {
    let mut y = vec![0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            y[i * n + j] = (0..k)
                .map(|kk| x[i * k + kk].to_f64() * w[j * k + kk].to_f64())
                .sum();
        }
    }
    y
}

fn kernel(
    kind: Kind,
    m: usize,
    n: usize,
    k: usize,
    x: &[f16],
    w: &[f16],
    threads: usize,
) -> Result<Vec<f16>> {
    let mut y = vec![f16::ZERO; m * n];
    let o = Operands {
        m,
        n,
        k,
        x: x.as_ptr(),
        x_rs: k,
        w: w.as_ptr(),
        w_rs: k,
        dst: y.as_mut_ptr(),
        dst_rs: n,
        dst_cs: 1,
    };
    unsafe { matmul_with(&o, kind, threads)? };
    Ok(y)
}

fn bits(v: &[f16]) -> Vec<u16> {
    v.iter().map(|h| h.to_bits()).collect()
}

/// max |y − exact| / max |exact|.
fn err_scale(y: &[f16], exact: &[f64]) -> f64 {
    let scale = exact.iter().fold(0f64, |m, v| m.max(v.abs()));
    y.iter()
        .zip(exact)
        .fold(0f64, |m, (a, b)| m.max((a.to_f64() - b).abs()))
        / scale
}

/// Half an f16 ulp at `v`'s binade: the rounding the f16 store declares.
fn half_ulp16(v: f64) -> f64 {
    let a = v.abs().max(f16::MIN_POSITIVE.to_f64());
    2f64.powi(a.log2().floor() as i32 - 11)
}

/// The arms this CPU runs: `f32/1` everywhere, `f16/16` where its
/// instruction exists.
fn arms() -> Vec<Kind> {
    [Kind::F32, Kind::F16Chain16]
        .into_iter()
        .filter(|k| instruction(*k).is_ok())
        .collect()
}

const SHAPES: [(usize, usize, usize); 9] = [
    (1, 64, 64),
    (1, 512, 2048),
    (8, 64, 8),
    (8, 64, 23),
    (8, 64, 512),
    (4, 96, 2048),
    (3, 37, 136),
    (2, 1000, 15),
    (70, 128, 64),
];

/// Every arm is its model, bit for bit, on one thread and on many.
#[test]
fn each_arm_computes_its_declared_rounding_bit_for_bit() -> Result<()> {
    let arms = arms();
    assert!(arms.contains(&Kind::F32), "the f32 arm exists everywhere");
    let mut checked = 0usize;
    for kind in arms {
        for (m, n, k) in SHAPES {
            let (x, w) = operands(m, n, k, 939);
            let want = model(kind, m, n, k, &x, &w);
            for threads in [1usize, 8] {
                let got = kernel(kind, m, n, k, &x, &w, threads)?;
                let same = bits(&got)
                    .iter()
                    .zip(bits(&want))
                    .filter(|(a, b)| **a == *b)
                    .count();
                println!(
                    "ACCUM arm={} m={m} n={n} k={k} threads={threads} kernel_vs_model={same}/{}",
                    kind.name(),
                    want.len()
                );
                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "{} m={m} n={n} k={k} threads={threads}",
                    kind.name()
                );
                checked += want.len();
            }
        }
    }
    assert!(checked >= 20_000, "non-vacuous: {checked} outputs compared");
    Ok(())
}

/// `f32/1` sits at the f16 store's own rounding: every output within half an
/// f16 ulp of the exact sum plus the f32 chain's error bound, so the kernel
/// rounds to f16 once. `f16/16`'s declared rounding is the model's (above);
/// against the exact sum it stays within four times the f32 arm's worst
/// error at every shape here, where one f16 chain of `k` reads 10–50× it.
#[test]
fn the_f32_arm_rounds_once_and_the_chains_stay_near_the_floor() -> Result<()> {
    let arms = arms();
    for (m, n, k) in SHAPES {
        let (x, w) = operands(m, n, k, 7);
        let ex = exact(m, n, k, &x, &w);
        let y32 = kernel(Kind::F32, m, n, k, &x, &w, 1)?;
        // The f32 chain over 8 lanes: at most k/8 roundings of 2^-24 each,
        // of a running sum bounded by the sum of |products|.
        let u32 = 2f64.powi(-24);
        for (i, (y, e)) in y32.iter().zip(&ex).enumerate() {
            let (r, c) = (i / n, i % n);
            let abs_sum: f64 = (0..k)
                .map(|kk| (x[r * k + kk].to_f64() * w[c * k + kk].to_f64()).abs())
                .sum();
            let bound = half_ulp16(*e) + (k as f64) * u32 * abs_sum;
            assert!(
                (y.to_f64() - e).abs() <= bound,
                "f32/1 m={m} n={n} k={k} out {i}: {} against {e}, bound {bound:e}",
                y.to_f64()
            );
        }
        let e32 = err_scale(&y32, &ex);
        println!("ACCUM arm=f32/1 m={m} n={n} k={k} err/scale={e32:.3e}");
        if arms.contains(&Kind::F16Chain16) {
            let y16 = kernel(Kind::F16Chain16, m, n, k, &x, &w, 1)?;
            let e16 = err_scale(&y16, &ex);
            // One f16 chain of k per lane: the mutation's arithmetic.
            let one_chain: Vec<f16> = (0..m)
                .flat_map(|i| {
                    let x = &x;
                    let w = &w;
                    (0..n).map(move |j| {
                        let mut lanes = [f16::ZERO; 8];
                        for kk in 0..k / 8 * 8 {
                            lanes[kk % 8] = fma16(x[i * k + kk], w[j * k + kk], lanes[kk % 8]);
                        }
                        let l: Vec<f32> = lanes.iter().map(|h| h.to_f32()).collect();
                        let s = ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]));
                        let mut t = 0f32;
                        for kk in k / 8 * 8..k {
                            t = x[i * k + kk].to_f32().mul_add(w[j * k + kk].to_f32(), t);
                        }
                        f16::from_f32(s + t)
                    })
                })
                .collect();
            let e1 = err_scale(&one_chain, &ex);
            println!(
                "ACCUM arm=f16/16 m={m} n={n} k={k} err/scale={e16:.3e} f32_floor={e32:.3e} one_f16_chain={e1:.3e}"
            );
            assert!(
                e16 <= 4.0 * e32.max(2f64.powi(-12)),
                "f16/16 m={m} n={n} k={k}: {e16:e} against the f32 arm's {e32:e}"
            );
        }
    }
    Ok(())
}

/// A row's bits do not depend on the rows beside it: row `i` of an `m`-row
/// pass equals the same row computed alone, every `m`, both arms.
#[test]
fn a_row_keeps_its_bits_whatever_the_pass_carries() -> Result<()> {
    let (m, n, k) = (23usize, 96usize, 200usize);
    let (x, w) = operands(m, n, k, 11);
    for kind in arms() {
        let full = kernel(kind, m, n, k, &x, &w, 8)?;
        for rows in [1usize, 2, 3, 8, 15, 22] {
            let part = kernel(kind, rows, n, k, &x[..rows * k], &w, 1)?;
            assert_eq!(
                bits(&part),
                bits(&full[..rows * n]),
                "{} rows={rows}",
                kind.name()
            );
        }
        for i in 0..m {
            let alone = kernel(kind, 1, n, k, &x[i * k..(i + 1) * k], &w, 1)?;
            assert_eq!(
                bits(&alone),
                bits(&full[i * n..(i + 1) * n]),
                "{} row {i} alone",
                kind.name()
            );
        }
    }
    Ok(())
}

/// `Tensor::matmul` on the CPU takes the arm the tuning names: the default
/// (`auto`) is `f32/1`, bit for bit the kernel's, in every operand layout —
/// `x · wᵀ` from a row-major `w`, a row-major `[k, n]` rhs (packed), a
/// batched lhs over a shared rhs, and a transposed lhs view.
#[test]
fn the_tensor_matmul_takes_the_arm_the_tuning_names() -> Result<()> {
    let dev = Device::Cpu;
    let kind = declared("auto", "auto").expect("auto is declared");
    assert_eq!(kind, Kind::F32);
    let t = lloom_tensor::tuning::Tuning::default();
    assert_eq!(t.cpu.f16_matmul_accumulate, "auto");
    let _ = lloom_tensor::tuning::install(t);
    let r = lloom_tensor::cpu::f16_matmul::resolved()?;
    assert_eq!(r.kind, Kind::F32);
    println!("ACCUM resolved={}", r.report());
    assert!(
        lloom_tensor::tuning::contract()
            .report()
            .contains(&format!("cpu_f16_accum={}", r.report())),
        "the CONTRACT line names the accumulator: {}",
        lloom_tensor::tuning::contract().report()
    );

    let (m, n, k) = (5usize, 70usize, 100usize);
    let (x, w) = operands(m, n, k, 3);
    let want = kernel(Kind::F32, m, n, k, &x, &w, 1)?;
    let xt = Tensor::from_vec(x.clone(), (m, k), &dev)?;
    let wt = Tensor::from_vec(w.clone(), (n, k), &dev)?;
    assert_eq!(xt.dtype(), DType::F16);

    // x · wᵀ: both operands k-contiguous, the layout every linear takes.
    let y = xt.matmul(&wt.t()?)?;
    assert_eq!(
        bits(&y.flatten_all()?.to_vec1::<f16>()?),
        bits(&want),
        "x · wᵀ"
    );
    // A row-major [k, n] rhs: packed into k-contiguous rows first.
    let w_kn = wt.t()?.contiguous()?;
    let y = xt.matmul(&w_kn)?;
    assert_eq!(
        bits(&y.flatten_all()?.to_vec1::<f16>()?),
        bits(&want),
        "x · w[k, n]"
    );
    // A transposed lhs view: packed likewise.
    let x_t = xt.t()?.contiguous()?.t()?;
    let y = x_t.matmul(&wt.t()?)?;
    assert_eq!(
        bits(&y.flatten_all()?.to_vec1::<f16>()?),
        bits(&want),
        "xᵀᵀ · wᵀ"
    );
    // A batch of lhs rows over one rhs: the rows stack.
    let xb = xt.reshape((m, 1, k))?;
    let y = xb.broadcast_matmul(&wt.t()?)?;
    assert_eq!(
        bits(&y.flatten_all()?.to_vec1::<f16>()?),
        bits(&want),
        "[m, 1, k] · wᵀ"
    );
    // The f16 chains, where the CPU has them, are a different arm: the
    // tensor path must not be taking them under `auto`.
    if instruction(Kind::F16Chain16).is_ok() {
        let other = kernel(Kind::F16Chain16, m, n, k, &x, &w, 1)?;
        assert_ne!(
            bits(&other),
            bits(&want),
            "the two arms differ on this fixture"
        );
    }
    Ok(())
}

/// The undeclared pairings and the missing instruction are refused by name.
#[test]
fn an_undeclared_combination_is_refused_by_name() {
    for (t, p) in [("f32", "16"), ("f16", "1")] {
        let err = declared(t, p).unwrap_err();
        assert!(err.contains("kernels.matmul_f16.accumulate"), "{err}");
        assert!(
            err.contains(&format!("type={t}, promote_every={p}")),
            "{err}"
        );
    }
    assert_eq!(declared("f16", "auto"), Ok(Kind::F16Chain16));
    assert_eq!(declared("auto", "16"), Ok(Kind::F16Chain16));
    assert_eq!(declared("f32", "1"), Ok(Kind::F32));
    let fp16_arm = cfg!(all(target_arch = "aarch64", target_feature = "fp16"));
    match instruction(Kind::F16Chain16) {
        Ok(i) => assert!(fp16_arm, "{i:?} on a CPU without f16 arithmetic"),
        Err(e) => {
            assert!(!fp16_arm, "{e}");
            assert!(e.contains("f16/16") && e.contains("fp16"), "{e}");
        }
    }
    // The tuning's parsers refuse what the kernel would.
    let mut t = lloom_tensor::tuning::Tuning::default();
    assert!(t.set("cpu.f16_matmul_accumulate", "bf16").is_err());
    assert!(t.set("cpu.f16_matmul_promote_every", "64").is_err());
    t.set("cpu.f16_matmul_accumulate", "f16").unwrap();
    t.set("cpu.f16_matmul_promote_every", "16").unwrap();
    assert_eq!(
        declared(
            &t.cpu.f16_matmul_accumulate,
            &t.cpu.f16_matmul_promote_every
        ),
        Ok(Kind::F16Chain16)
    );
}
