//! The routed SwiGLU on the gate|up launch's store (`kernels.fusion.routed_glu`,
//! `Qwen4Exp::set_routed_glu`; `docs/expert-streaming.md` §20.15f) on the tiny
//! model, experts stored Q8_0 at a 32-wide intermediate, resident and streamed (6 slots for 8 experts:
//! misses, settles, replays): a short prefill (12 rows, the matmul-vec) and
//! decode steps compute the same logits with the arm on and off, at f32 and
//! f16 routed activations. On the resident model the census shows where it
//! ran — on, one `kernel_mul_mv_id2_q8_0_<triple>_glu` a layer and a pass, no
//! plain pair, one `silu_mul` a layer fewer (the shared expert keeps its
//! own); off, none — and on the streamed one the arm's launches are counted.
#![cfg(feature = "metal")]
use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::models::qwen4exp::{
    Config, ExpertResidency, LoadOptions, Qwen4Exp, SlotBudget, State, StreamLoader, StreamOptions,
};
use lloom::nn::model::CausalLM;
use lloom::profile::CrownExact;
use lloom::testing::fixture::Fixture;
use lloom::testing::tiny::Weights;
use lloom_tensor::quantized::GgmlDType;
use lloom_tensor::{DType, Device, Result, Tensor};

fn install() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        // The installed tuning replaces the environment's, so the API an arm
        // asks for (`LLOOM_METAL_API`) is carried over, or every arm runs
        // the default queue.
        if let Ok(api) = std::env::var("LLOOM_METAL_API") {
            t.metal.command_api = api;
        }
        t.contract.profile = "deterministic".into();
        t.instrument.profile = true;
        let _ = lloom_tensor::tuning::install(t);
    });
    assert!(lloom_tensor::kernels::metal::profile::enabled());
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

type Census = Vec<(String, u64)>;

fn count(c: &[(String, u64)], pred: impl Fn(&str) -> bool) -> u64 {
    c.iter().filter(|(l, _)| pred(l)).map(|(_, n)| *n).sum()
}

/// The prompt then each step, one pass each: every pass's logits and census.
fn passes<R: ExpertResidency>(
    model: &Qwen4Exp<CrownExact, R>,
    cfg: &Config,
    dev: &Device,
    inputs: &[Vec<u32>],
) -> Result<(Vec<Vec<f32>>, Vec<Census>)> {
    let mut cache: State<CrownExact> = model.new_cache(cfg, dev)?;
    let (mut logits, mut census) = (Vec::new(), Vec::new());
    for ids in inputs {
        dev.synchronize()?;
        let w = dev
            .as_metal_device()?
            .dispatch_census()
            .open()
            .expect("a census window on this test's device");
        let input = Tensor::from_vec(ids.clone(), (1, ids.len()), dev)?;
        let pos = cache.pos;
        let l: Vec<f32> = model
            .forward(&input, pos, &mut cache)?
            .flatten_all()?
            .to_vec1()?;
        dev.synchronize()?;
        logits.push(l);
        census.push(
            w.close()
                .into_iter()
                .map(|(l, n)| (l.to_string(), n))
                .collect(),
        );
    }
    Ok((logits, census))
}

#[test]
fn the_routed_glu_arm_computes_the_same_logits_and_runs_where_the_pair_did() -> Result<()> {
    install();
    let Ok(dev) = Device::new_metal(0) else {
        return Ok(());
    };
    let asked = std::env::var("LLOOM_METAL_API").unwrap_or_else(|_| "default".into());
    let actual = dev.as_metal_device()?.commands().actual_command_api();
    eprintln!("ROUTED_GLU api requested={asked} actual={actual}");
    if asked == "metal3" || asked == "metal4" {
        assert_eq!(
            actual, asked,
            "the device did not take the requested Metal API"
        );
    }
    let Some(fx) = Fixture::<Qwen4ExpTiny>::new("qwen4exp_routed_glu", Config::tiny())? else {
        return Ok(());
    };
    let fx = fx.with_weights(Weights {
        experts: GgmlDType::Q8_0,
        ..Weights::default()
    });
    if fx.at_scale() {
        return Ok(());
    }
    // `down`'s K is the intermediate width: at the template's it is under
    // Q8_0's 32-wide block, so the fixture stores `down` F16, which has no
    // f16-activation arm (as `qwen4exp_moe_act_tile` widens it).
    let cfg = fx.config(|c| c.moe_intermediate_size = 32);
    let layers = cfg.n_layers as u64;
    let opts = LoadOptions::default();
    let inputs: Vec<Vec<u32>> = std::iter::once((0..12u32).map(|i| (i * 37 + 5) % 128).collect())
        .chain([7u32, 19, 3, 64, 101, 2, 55, 88].iter().map(|&t| vec![t]))
        .collect();
    for act in [DType::F32, DType::F16] {
        let suffix = if act == DType::F16 {
            "_f16_f16"
        } else {
            "_f32"
        };
        let glu_name = format!("kernel_mul_mv_id2_q8_0{suffix}_glu");
        let pair_name = format!("kernel_mul_mv_id2_q8_0{suffix}");
        let silu = if act == DType::F16 {
            "silu_mul_f16"
        } else {
            "silu_mul_f32"
        };

        let mut resident = Vec::new();
        for on in [false, true] {
            let mut model = fx.resident(&cfg, &dev, &opts)?;
            model.set_routed_act(act);
            model.set_routed_glu(on);
            resident.push(passes(&model, &cfg, &dev, &inputs)?);
        }
        let ((off, off_c), (on, on_c)) = (&resident[0], &resident[1]);
        for i in 0..inputs.len() {
            let (glu_on, pair_on) = (
                count(&on_c[i], |l| l == glu_name),
                count(&on_c[i], |l| l == pair_name),
            );
            let (glu_off, pair_off) = (
                count(&off_c[i], |l| l == glu_name),
                count(&off_c[i], |l| l == pair_name),
            );
            let (s_off, s_on) = (
                count(&off_c[i], |l| l == silu),
                count(&on_c[i], |l| l == silu),
            );
            println!(
                "ROUTEDGLU resident act={act:?} pass={i} on: {glu_name}={glu_on} {pair_name}={pair_on} {silu}={s_on} | off: {glu_name}={glu_off} {pair_name}={pair_off} {silu}={s_off}"
            );
            assert_eq!(
                bits(&on[i]),
                bits(&off[i]),
                "{act:?} resident: pass {i}'s logits moved with the arm"
            );
            assert_eq!(
                (glu_on, pair_on),
                (layers, 0),
                "{act:?} resident on, pass {i}: {:?}",
                on_c[i]
            );
            assert_eq!(
                (glu_off, pair_off),
                (0, layers),
                "{act:?} resident off, pass {i}: {:?}",
                off_c[i]
            );
            assert_eq!(
                s_off - s_on,
                layers,
                "{act:?} resident pass {i}: not one {silu} a layer fewer"
            );
        }

        let stream = StreamOptions {
            slots: SlotBudget::Fixed(6),
            loader: StreamLoader::Pread,
            prefetch: false,
            io_hints: false,
            drafter_bytes: 0,
        };
        let mut streamed = Vec::new();
        for on in [false, true] {
            let mut model = fx.streamed(&cfg, &dev, &opts, &stream)?;
            model.set_routed_act(act);
            model.set_routed_glu(on);
            streamed.push(passes(&model, &cfg, &dev, &inputs)?);
        }
        let ((soff, soff_c), (son, son_c)) = (&streamed[0], &streamed[1]);
        let glu_total: u64 = son_c.iter().map(|c| count(c, |l| l == glu_name)).sum();
        let glu_off_total: u64 = soff_c.iter().map(|c| count(c, |l| l == glu_name)).sum();
        println!("ROUTEDGLU streamed act={act:?} on: {glu_name}={glu_total} off: {glu_off_total}");
        // Every one-row step takes it (a replay adds more); a prefill group
        // of 2 to 8 rows may take the pair's union launch, which it declines.
        let steps = inputs.len() as u64 - 1;
        assert!(
            glu_total >= layers * steps,
            "{act:?} streamed: the arm ran {glu_total} GLU launches"
        );
        assert_eq!(glu_off_total, 0, "{act:?} streamed off ran the GLU");
        for i in 0..inputs.len() {
            assert_eq!(
                bits(&son[i]),
                bits(&soff[i]),
                "{act:?} streamed: pass {i}'s logits moved with the arm"
            );
            assert_eq!(
                bits(&son[i]),
                bits(&off[i]),
                "{act:?}: streamed pass {i} is not the resident model's"
            );
        }
    }
    Ok(())
}
