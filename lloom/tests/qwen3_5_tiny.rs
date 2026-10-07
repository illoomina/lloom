//! The Qwen3.5 family at tiny geometry (`docs/porting-models.md` §21.1
//! Step 4a, §21.8): the dense and the MoE tiny models through the real
//! loader, on the CPU against the f32 reference and on Metal against the
//! CPU, deterministic run to run; the gated norm's SiLU gate against its
//! composed form and against the sigmoid gate (the one block Qwen3.5
//! generalized); the generic decoder's layer against the blocks composed
//! by hand.
use lloom::models::qwen3_5::testing::{Qwen3_5MoeTiny, Qwen3_5Tiny};
use lloom::models::qwen3_5::Config;
use lloom::nn::blocks::{Ffn, Mixer, Norm};
use lloom::nn::decoder::{Hybrid, HybridState, Layer, PreNorm, StateCapacity};
use lloom::nn::model::{CacheReset, CausalLM};
use lloom::profile::CrownExact;
use lloom::testing::tiny::{run, TinyCheckpoint, TinyFamily, Weights, Workload};
use lloom_tensor::quantized::GgmlDType;
use lloom_tensor::{Device, Result, Tensor};

fn pin() {
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
}

/// The largest relative difference between two logit runs, over the CPU's
/// largest magnitude.
fn rel(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut worst = 0f32;
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x.len(), y.len());
        let scale = x.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        for (p, q) in x.iter().zip(y) {
            worst = worst.max((p - q).abs() / scale);
        }
    }
    worst
}

fn workloads() -> [Workload; 3] {
    [
        Workload {
            prompt: 1,
            decode: 2,
        },
        Workload {
            prompt: 7,
            decode: 2,
        },
        Workload {
            prompt: 70,
            decode: 1,
        },
    ]
}

/// Both variants, built through the loader at f32 weights, agree with the
/// f32 reference on the CPU. The served load's f16 stems cast the
/// activations to f16 on the CPU (the Qwen4-exp campaign's precision class,
/// `porting-models.md` §21.8), which at this geometry flips a near-tied
/// routed top-2 — so the MoE is held with every expert routed (no selection
/// to flip) at the precision bound, and at its top-2 against the reference's
/// own routing on Metal. The packed kinds are held on Metal (below),
/// where the quantized kernels read f32 activations; the CPU quantized
/// matmul quantizes them (Q8_0 at hidden 32 sits at 6.6e-2 against f32).
#[test]
fn the_served_load_agrees_with_the_f32_reference_on_the_cpu() -> Result<()> {
    pin();
    let dev = Device::Cpu;
    for kind in [GgmlDType::F32] {
        let w = Weights {
            dense: kind,
            experts: kind,
            ..Weights::default()
        };
        let dense = TinyCheckpoint::<Qwen3_5Tiny>::new(Config::tiny(), w);
        let served = dense.model::<CrownExact>(&dev)?;
        let reference = Qwen3_5Tiny::reference::<CrownExact>(&dense, &dev)?;
        for wl in workloads() {
            let a = run(&served, &dense.cfg, dense.cfg.vocab_size, &dev, &wl)?;
            let b = run(&reference, &dense.cfg, dense.cfg.vocab_size, &dev, &wl)?;
            let r = rel(&a, &b);
            eprintln!("QWEN35 dense {kind:?} prompt={} rel={r:e}", wl.prompt);
            assert!(r < 5e-2, "dense {kind:?} prompt {}: rel {r}", wl.prompt);
        }
        let mut all = Config::tiny_moe();
        all.n_experts_per_tok = all.n_experts;
        let moe = TinyCheckpoint::<Qwen3_5MoeTiny>::new(all, w);
        let served = moe.model::<CrownExact>(&dev)?;
        let reference = Qwen3_5MoeTiny::reference::<CrownExact>(&moe, &dev)?;
        for wl in workloads() {
            let a = run(&served, &moe.cfg, moe.cfg.vocab_size, &dev, &wl)?;
            let b = run(&reference, &moe.cfg, moe.cfg.vocab_size, &dev, &wl)?;
            let r = rel(&a, &b);
            eprintln!(
                "QWEN35 moe(all routed) {kind:?} prompt={} rel={r:e}",
                wl.prompt
            );
            assert!(r < 5e-2, "moe {kind:?} prompt {}: rel {r}", wl.prompt);
        }
    }
    // The routed top-2 is held on Metal (`metal_vs_cpu`), where the
    // activations stay f32: on the CPU served path one flipped expert at a
    // prefill position carries into every position after it (measured:
    // 3 of 3 positions off at prompt 7, none at prompt 1).
    Ok(())
}

/// A prompt fed whole and the same prompt fed one position at a time leave
/// the state at the same place: the next token's logits agree (the chunked
/// DeltaNet and the chunk attention against their decode forms), on the f32
/// reference where nothing but the chunking differs. (The served f16 stems
/// sit at ~2e-2 between the two on the CPU: the GEMM and the GEMV cast the
/// activations differently there — the campaign's precision class.)
#[test]
fn a_chunked_prefill_agrees_with_the_replayed_one() -> Result<()> {
    pin();
    let dev = Device::Cpu;
    let ckpt = TinyCheckpoint::<Qwen3_5Tiny>::new(Config::tiny(), Weights::default());
    let m = Qwen3_5Tiny::reference::<CrownExact>(&ckpt, &dev)?;
    let vocab = ckpt.cfg.vocab_size as u32;
    let prompt: Vec<u32> = (0..70u32).map(|i| (i * 7 + 3) % vocab).collect();
    let flat = |t: Tensor| -> Result<Vec<f32>> { t.flatten_all()?.to_vec1::<f32>() };
    let mut whole = m.new_cache(&ckpt.cfg, &dev)?;
    let a = flat(m.forward(&input_of(&prompt, &dev)?, 0, &mut whole)?)?;
    let mut replay = m.new_cache(&ckpt.cfg, &dev)?;
    let mut b = Vec::new();
    for (pos, t) in prompt.iter().enumerate() {
        let input = Tensor::from_vec(vec![*t], (1, 1), &dev)?;
        b = flat(m.forward(&input, pos, &mut replay)?)?;
    }
    let r = rel(std::slice::from_ref(&a), &[b]);
    eprintln!("QWEN35 chunk-vs-replay rel={r:e}");
    assert!(r < 1e-4, "rel {r}");
    // A reset state decodes the prompt again to the same logits, bit for bit.
    whole.reset();
    let c = flat(m.forward(&input_of(&prompt, &dev)?, 0, &mut whole)?)?;
    assert_eq!(a, c);
    Ok(())
}

fn input_of(ids: &[u32], dev: &Device) -> Result<Tensor> {
    Tensor::from_vec(ids.to_vec(), (1, ids.len()), dev)
}

/// On Metal: the served load (f16 stems or the packed kinds the real
/// checkpoints store, the GDN and attention kernels) against the CPU f32
/// reference, and bit for bit identical over three fresh runs — the digest
/// gate at tiny geometry.
#[cfg(feature = "metal")]
#[test]
fn the_tiny_models_are_deterministic_on_metal_and_agree_with_the_cpu() -> Result<()> {
    pin();
    // The routed top-2 at f32 weights (the served and the reference route
    // the same experts: Metal reads the activations at f32), then every
    // kind with every expert routed (a quantized weight legitimately moves
    // a near-tied selection).
    metal_vs_cpu(GgmlDType::F32, false)?;
    for kind in [
        GgmlDType::F32,
        GgmlDType::Q8_0,
        GgmlDType::Q4K,
        GgmlDType::Q6K,
    ] {
        metal_vs_cpu(kind, true)?;
    }
    Ok(())
}

#[cfg(feature = "metal")]
fn metal_vs_cpu(kind: GgmlDType, route_all: bool) -> Result<()> {
    let dev = Device::new_metal(0)?;
    let cpu = Device::Cpu;
    let w = Weights {
        dense: kind,
        experts: kind,
        ..Weights::default()
    };
    let dense = TinyCheckpoint::<Qwen3_5Tiny>::new(Config::tiny(), w);
    let mut all = Config::tiny_moe();
    if route_all {
        all.n_experts_per_tok = all.n_experts;
    }
    let moe = TinyCheckpoint::<Qwen3_5MoeTiny>::new(all, w);
    let dm = dense.model::<CrownExact>(&dev)?;
    let dr = Qwen3_5Tiny::reference::<CrownExact>(&dense, &cpu)?;
    let mm = moe.model::<CrownExact>(&dev)?;
    let mr = Qwen3_5MoeTiny::reference::<CrownExact>(&moe, &cpu)?;
    for wl in workloads() {
        let runs: Vec<Vec<Vec<f32>>> = (0..3)
            .map(|_| run(&dm, &dense.cfg, dense.cfg.vocab_size, &dev, &wl))
            .collect::<Result<_>>()?;
        assert!(
            runs[0] == runs[1] && runs[1] == runs[2],
            "dense {kind:?} prompt {}: not deterministic",
            wl.prompt
        );
        let r = rel(
            &run(&dr, &dense.cfg, dense.cfg.vocab_size, &cpu, &wl)?,
            &runs[0],
        );
        eprintln!(
            "QWEN35 metal dense {kind:?} prompt={} rel_to_cpu={r:e}",
            wl.prompt
        );
        assert!(r < 5e-2, "dense {kind:?} prompt {}: rel {r}", wl.prompt);
        let runs: Vec<Vec<Vec<f32>>> = (0..3)
            .map(|_| run(&mm, &moe.cfg, moe.cfg.vocab_size, &dev, &wl))
            .collect::<Result<_>>()?;
        assert!(
            runs[0] == runs[1] && runs[1] == runs[2],
            "moe {kind:?} prompt {}: not deterministic",
            wl.prompt
        );
        let r = rel(
            &run(&mr, &moe.cfg, moe.cfg.vocab_size, &cpu, &wl)?,
            &runs[0],
        );
        eprintln!(
            "QWEN35 metal moe({}) {kind:?} prompt={} rel_to_cpu={r:e}",
            if route_all { "all routed" } else { "top-2" },
            wl.prompt
        );
        assert!(r < 5e-2, "moe {kind:?} prompt {}: rel {r}", wl.prompt);
    }
    Ok(())
}

/// The gated norm's SiLU gate on Metal (the kernel instantiated for Qwen3.5)
/// equals its composed form and differs from the sigmoid gate — the
/// mutation: a checkpoint read with the wrong gate moves the block's output.
#[cfg(feature = "metal")]
#[test]
fn the_gdn_out_gate_is_a_variant_the_block_carries() -> Result<()> {
    use lloom::models::qwen4exp::testing::{lin, ones, wt};
    use lloom::models::qwen4exp::{GatedDeltaNet, GdnGeometry};
    use lloom::nn::gdn::OutGate;
    use lloom::nn::RecurrentState;
    pin();
    let dev = Device::new_metal(0)?;
    let cpu = Device::Cpu;
    let cfg = Config::tiny();
    let g = cfg.gdn();
    let (d, cc, vd, hv) = (
        cfg.hidden_size,
        g.conv_channels(),
        g.value_dim(),
        g.dims.h_v,
    );
    let build = |dev: &Device, gate: OutGate| -> GatedDeltaNet<CrownExact> {
        let a = Tensor::from_vec(
            (0..hv).map(|i| -0.5 - 0.1 * i as f32).collect::<Vec<_>>(),
            hv,
            dev,
        )
        .unwrap();
        GatedDeltaNet::with_geometry(
            lin(cc, d, 1.0, dev),
            lin(vd, d, 2.0, dev),
            wt((cc, cfg.ssm_conv_kernel), 3.0, dev),
            a,
            Tensor::from_vec(
                (0..hv).map(|i| 0.01 * i as f32).collect::<Vec<_>>(),
                hv,
                dev,
            )
            .unwrap(),
            lin(hv, d, 4.0, dev),
            lin(hv, d, 5.0, dev),
            ones(g.dims.d_v, dev),
            lin(d, vd, 6.0, dev),
            &GdnGeometry { gate, ..g },
        )
    };
    let x1 = Tensor::arange(0f32, d as f32, &cpu)?
        .affine(0.02, -0.4)?
        .reshape((1, 1, d))?;
    let x3 = Tensor::arange(0f32, (3 * d) as f32, &cpu)?
        .affine(0.01, -0.3)?
        .reshape((1, 3, d))?;
    let flat = |t: Tensor| -> Result<Vec<f32>> {
        t.flatten_all()?
            .to_dtype(lloom_tensor::DType::F32)?
            .to_vec1::<f32>()
    };
    let mut out = std::collections::BTreeMap::new();
    for (name, dev) in [("cpu", &cpu), ("metal", &dev)] {
        for gate in [OutGate::Silu, OutGate::Sigmoid] {
            let block = build(dev, gate);
            assert_eq!(block.out_gate(), gate);
            let (mut s, mut c) = (RecurrentState::new(1), RecurrentState::new(1));
            let pre = flat(block.prefill(&x3.to_device(dev)?, 0, &mut s, &mut c)?)?;
            let dec = flat(block.decode(&x1.to_device(dev)?, 0, &mut s, &mut c)?)?;
            out.insert((name, gate.name()), (pre, dec));
        }
    }
    let close = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= 2e-3 * (1.0 + y.abs()))
    };
    for gate in ["silu", "sigmoid"] {
        let (cp, cd) = &out[&("cpu", gate)];
        let (mp, md) = &out[&("metal", gate)];
        assert!(
            close(cp, mp),
            "{gate}: the Metal prefill differs from the composed CPU one"
        );
        assert!(
            close(cd, md),
            "{gate}: the Metal decode differs from the composed CPU one"
        );
    }
    // The mutation: the two gates do not compute the same block.
    let (sp, sd) = &out[&("metal", "silu")];
    let (gp, gd) = &out[&("metal", "sigmoid")];
    assert!(
        !close(sp, gp) && !close(sd, gd),
        "silu and sigmoid gates computed the same output"
    );
    Ok(())
}

/// The generic layer (`PreNorm<RmsNorm, Hybrid<..>, SwiGlu>`) computes what
/// the blocks composed by hand compute: the topology adds no arithmetic.
#[test]
fn the_pre_norm_layer_is_the_blocks_composed() -> Result<()> {
    pin();
    let dev = Device::Cpu;
    let ckpt = TinyCheckpoint::<Qwen3_5Tiny>::new(Config::tiny(), Weights::default());
    let m = ckpt.model::<CrownExact>(&dev)?;
    let cap = StateCapacity::new(64);
    let d = ckpt.cfg.hidden_size;
    let x = Tensor::arange(0f32, (5 * d) as f32, &dev)?
        .affine(0.01, -0.3)?
        .reshape((1, 5, d))?;
    for (l, layer) in m.layers().iter().enumerate() {
        let mut st = layer.new_state(&dev, &cap)?;
        let via_layer = layer.execute(&x, &mut st)?;
        // By hand, through the same blocks' trait methods.
        let h = layer.norm_in.execute(&x)?;
        let mut ms = match &layer.mixer {
            Hybrid::A(_) => HybridState::A(lloom::models::qwen4exp::GdnState {
                layer: 0,
                s: lloom::nn::RecurrentState::new(1),
                conv: lloom::nn::RecurrentState::new(1),
            }),
            Hybrid::B(_) => HybridState::B(lloom::models::qwen4exp::KvCache::with_capacity(64)),
        };
        let mixed = layer.mixer.execute(&h, &mut ms)?;
        let x1 = (&x + mixed)?;
        let h2 = layer.norm_mid.execute(&x1)?;
        let f = layer.ffn.execute(&h2, &mut ())?;
        let by_hand = (x1 + f)?;
        let a = via_layer.flatten_all()?.to_vec1::<f32>()?;
        let b = by_hand.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(a, b, "layer {l}: the topology changed the arithmetic");
    }
    let _ = PreNorm::<
        lloom::nn::RmsNorm,
        lloom::models::qwen3_5::Qwen3_5Mixer<CrownExact>,
        lloom::nn::blocks::SwiGlu,
    >::reset_state;
    Ok(())
}
