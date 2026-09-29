//! Spikes P-a and P-b (`docs/porting-models.md` §21.8, INV-3b):
//! which value law the tiny models' weights follow.
//!
//! One line per case — law × seed × geometry — on stdout, tab-separated:
//!
//! - `fixed`: the logits of a fixed-token workload (a 5-token prompt, then 3
//!   decode steps), every step, as f32 bit patterns in hex, with each step's
//!   argmax and top-1/top-2 margin relative to the largest logit;
//! - `greedy`: 12 tokens greedily decoded from the same prompt.
//!
//! `--device cpu` is P-a's arm (run it on a clean build and on the rope
//! pair-swap mutant, then diff the two files). `--device metal` runs the
//! same cases in f32 and f16 on Metal beside the CPU's f32, for P-b: how
//! often two arms that legitimately reorder floating point pick different
//! tokens under each law.

use lloom::models::lfm2::Config;
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::nn::model::CausalLM;
use lloom::testing::tiny::{self, TinyCheckpoint, TinyFamily, ValueLaw, Weights, Workload};
use lloom_tensor::{DType, Device, Result, Tensor};

const LAWS: &[&str] = &[
    "ramp-mod17",
    "ramp-sin",
    "hash:0.5",
    "hash:1",
    "hash:2",
    "hash:4",
    "uniform:0.001",
    "uniform:0.01",
];
const SEEDS: u64 = 16;
const FIXED: Workload = Workload {
    prompt: 5,
    decode: 3,
};
const GREEDY: usize = 12;

/// `(name, hidden, heads, kv heads)`: head_dim 64, 32, 16.
const GEOMS: &[(&str, usize, usize, usize)] =
    &[("hd64", 128, 2, 1), ("hd32", 128, 4, 2), ("hd16", 64, 4, 2)];

fn cfg(hidden: usize, heads: usize, kv: usize) -> Config {
    let mut c = Lfm2Tiny::template();
    c.hidden_size = hidden;
    c.num_attention_heads = heads;
    c.num_key_value_heads = kv;
    c
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m })
        .0
}

/// Top-1 minus top-2, over the largest magnitude.
fn margin(v: &[f32]) -> f32 {
    let mut s: Vec<f32> = v.to_vec();
    s.sort_by(|a, b| b.total_cmp(a));
    let scale = v.iter().fold(0f32, |a, x| a.max(x.abs())).max(1e-30);
    (s[0] - s[1]) / scale
}

fn greedy(model: &lloom::models::lfm2::Model, c: &Config, dev: &Device) -> Result<Vec<u32>> {
    let mut cache = model.new_cache(c, dev)?;
    let (prompt, _) = FIXED.tokens(c.vocab_size);
    let input = Tensor::from_vec(prompt.clone(), (1, prompt.len()), dev)?;
    let mut logits = model.forward(&input, 0, &mut cache)?;
    let mut out = Vec::with_capacity(GREEDY);
    for pos in (prompt.len()..).take(GREEDY) {
        let v = logits
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let t = argmax(&v) as u32;
        out.push(t);
        let input = Tensor::from_vec(vec![t], (1, 1), dev)?;
        logits = model.forward(&input, pos, &mut cache)?;
    }
    Ok(out)
}

fn hex(v: &[f32]) -> String {
    v.iter().map(|x| format!("{:08x}", x.to_bits())).collect()
}

fn main() -> Result<()> {
    let device = std::env::args()
        .skip_while(|a| a != "--device")
        .nth(1)
        .unwrap_or_else(|| "cpu".into());
    let metal = match device.as_str() {
        "cpu" => None,
        "metal" => Some(Device::new_metal(0)?),
        other => lloom_tensor::bail!("--device {other}: cpu | metal"),
    };
    println!("#law\tseed\tgeom\tarm\tkind\tstep\targmax\tmargin\tpayload");
    for law in LAWS {
        let law_v = ValueLaw::parse(law).map_err(lloom_tensor::Error::msg)?;
        for &(g, h, nh, nkv) in GEOMS {
            for seed in 0..SEEDS {
                let ckpt = TinyCheckpoint::<Lfm2Tiny>::new(
                    cfg(h, nh, nkv),
                    Weights {
                        law: law_v,
                        seed,
                        ..Weights::default()
                    },
                );
                let mut arms: Vec<(&str, Device, DType)> = vec![("cpu32", Device::Cpu, DType::F32)];
                if let Some(m) = &metal {
                    arms.push(("metal32", m.clone(), DType::F32));
                    arms.push(("metal16", m.clone(), DType::F16));
                }
                for (arm, dev, dt) in arms {
                    let c = ckpt.clone().with_dtype(dt);
                    let model = c.model(&dev)?;
                    let steps = tiny::run(&model, &c.cfg, c.cfg.vocab_size, &dev, &FIXED)?;
                    for (i, s) in steps.iter().enumerate() {
                        println!(
                            "{law}\t{seed}\t{g}\t{arm}\tfixed\t{i}\t{}\t{:e}\t{}",
                            argmax(s),
                            margin(s),
                            hex(s)
                        );
                    }
                    let toks = greedy(&model, &c.cfg, &dev)?;
                    let toks: Vec<String> = toks.iter().map(u32::to_string).collect();
                    println!(
                        "{law}\t{seed}\t{g}\t{arm}\tgreedy\t-\t-\t-\t{}",
                        toks.join(",")
                    );
                }
            }
        }
    }
    Ok(())
}
