//! Single-thread speed of the CPU f16 matmul's accumulator arms (lloom #939,
//! `cpu::f16_matmul`), interleaved with the `gemm` crate's f16 matmul the
//! engine took before, and a null control (the f32 arm timed a second time
//! under another name). Three shapes: decode (m = 1, memory-bound), prefill
//! (m = 64) and an L1-resident product (the arithmetic's own cost). Best of
//! N timed reps per arm per round, GFLOP/s; rounds interleave the arms
//! (CONTRIBUTING §3.2).
//!
//!   lloom-arb measure -- cargo run --release -p lloom-tensor --example cpu_f16_matmul_speed -- [rounds] [threads]
use half::f16;
use lloom_tensor::cpu::f16_matmul::{instruction, matmul_with, Kind, Operands};
use std::time::Instant;

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

fn operands(m: usize, n: usize, k: usize) -> (Vec<f16>, Vec<f16>) {
    let mut r = Rng(939);
    let x: Vec<f16> = (0..m * k).map(|_| f16::from_f64(r.gauss())).collect();
    let s = 1.0 / (k as f64).sqrt();
    let w: Vec<f16> = (0..n * k).map(|_| f16::from_f64(r.gauss() * s)).collect();
    (x, w)
}

/// One product's operands, `x · wᵀ` into `y`.
struct Shape<'a> {
    m: usize,
    n: usize,
    k: usize,
    x: &'a [f16],
    w: &'a [f16],
}

fn kernel(kind: Kind, s: &Shape, y: &mut [f16], threads: usize) {
    let o = Operands {
        m: s.m,
        n: s.n,
        k: s.k,
        x: s.x.as_ptr(),
        x_rs: s.k,
        w: s.w.as_ptr(),
        w_rs: s.k,
        dst: y.as_mut_ptr(),
        dst_rs: s.n,
        dst_cs: 1,
    };
    unsafe { matmul_with(&o, kind, threads).expect("an implemented arm") };
}

type Arm<'a> = (&'static str, Box<dyn FnMut() + 'a>);

/// The `gemm` crate's f16 matmul, as `cpu_backend` called it before this
/// kernel: `x · wᵀ`, f16 accumulators on fp16 Arm.
fn gemm_crate(s: &Shape, y: &mut [f16], threads: usize) {
    let par = if threads > 1 {
        gemm::Parallelism::Rayon(threads)
    } else {
        gemm::Parallelism::None
    };
    let (m, n, k) = (s.m, s.n, s.k);
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            y.as_mut_ptr(),
            1,
            n as isize,
            false,
            s.x.as_ptr(),
            1,
            k as isize,
            s.w.as_ptr(),
            k as isize,
            1,
            f16::ZERO,
            f16::ONE,
            false,
            false,
            false,
            par,
        )
    }
}

/// Best-of-N wall time of `f`, as GFLOP/s: at least 5 reps and 0.5 s.
fn time<F: FnMut()>(mut f: F, flops: f64) -> f64 {
    f();
    let mut best = f64::MAX;
    let t_all = Instant::now();
    let mut reps = 0;
    while reps < 5 || (t_all.elapsed().as_secs_f64() < 0.5 && reps < 500) {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
        reps += 1;
    }
    flops / best / 1e9
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rounds: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2);
    let threads: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let f32_instr = instruction(Kind::F32).expect("f32/1 exists everywhere");
    let chains = instruction(Kind::F16Chain16).ok();
    println!(
        "SPEED platform arch={} os={} f32_instr={} f16_chains={} threads={threads} rounds={rounds}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        f32_instr.name(),
        chains.map(|i| i.name()).unwrap_or("none")
    );
    let cases = [
        ("decode", 1usize, 2048usize, 2048usize),
        ("prefill", 64, 2048, 2048),
        ("l1", 64, 128, 128),
    ];
    for (name, m, n, k) in cases {
        let (x, w) = operands(m, n, k);
        let s = Shape {
            m,
            n,
            k,
            x: &x,
            w: &w,
        };
        let mut y = vec![f16::ZERO; m * n];
        let flops = 2.0 * (m * n * k) as f64;
        let mut y2 = vec![f16::ZERO; m * n];
        let mut y3 = vec![f16::ZERO; m * n];
        let mut y4 = vec![f16::ZERO; m * n];
        for round in 1..=rounds {
            let mut arms: Vec<Arm> =
                vec![("f32/1", Box::new(|| kernel(Kind::F32, &s, &mut y, threads)))];
            if chains.is_some() {
                arms.push((
                    "f16/16",
                    Box::new(|| kernel(Kind::F16Chain16, &s, &mut y2, threads)),
                ));
            }
            arms.push(("gemm-crate", Box::new(|| gemm_crate(&s, &mut y3, threads))));
            arms.push((
                "f32/1-null",
                Box::new(|| kernel(Kind::F32, &s, &mut y4, threads)),
            ));
            for (arm, f) in arms.iter_mut() {
                let gf = time(f, flops);
                println!(
                    "SPEED case={name} m={m} n={n} k={k} round={round} arm={arm} gflops={gf:.2}"
                );
            }
        }
    }
}
