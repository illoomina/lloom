//! Which block of the tiny Qwen4-exp model varies run to run on Metal
//! (P30 (1): `qwen4exp_verify_window`'s Metal arm found the whole prefill
//! non-deterministic at the tiny geometry — every channel from layer 0).
//! Each block, the same input, three fresh runs, bit for bit.
#![cfg(feature = "metal")]
use lloom::models::qwen4exp::{testing, Config};
use lloom::nn::RecurrentState;
use lloom_tensor::{Device, Result, Tensor};

fn v(t: &Tensor) -> Result<Vec<f32>> {
    t.flatten_all()?.to_vec1::<f32>()
}

fn stable(name: &str, mut f: impl FnMut() -> Result<Vec<f32>>) -> Result<bool> {
    let a = f()?;
    let b = f()?;
    let c = f()?;
    let ok = a == b && b == c;
    let maxdiff = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    eprintln!("TINYDET {name}: deterministic={ok} maxdiff_run0_run1={maxdiff:e} n={}", a.len());
    Ok(ok)
}

#[test]
fn which_block_varies() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let cfg = Config::tiny();
    let (n, d) = (cfg.hc_count, cfg.hidden_size);
    let seq = 7usize;
    let h = Tensor::from_vec(
        (0..seq * n * d).map(|i| ((i as f32) * 0.021).cos() * 0.8).collect::<Vec<f32>>(),
        (1, seq, n * d),
        &dev,
    )?;
    let x = Tensor::from_vec(
        (0..seq * d).map(|i| ((i as f32) * 0.017).sin() * 0.7).collect::<Vec<f32>>(),
        (1, seq, d),
        &dev,
    )?;
    let hc = testing::hc(&cfg, 1.0, &dev);
    let mut all = true;
    all &= stable("hc.mix (mixed)", || v(&hc.mix(&h)?.0))?;
    all &= stable("hc.mix (inject)", || v(&hc.mix(&h)?.1))?;
    let bo = x.clone();
    let inj = hc.mix(&h)?.1;
    all &= stable("hc.combine", || v(&hc.combine(&h, &bo, &inj)?))?;
    let dn = testing::deltanet(&cfg, &dev);
    all &= stable("deltanet.prefill (out)", || {
        let mut s = RecurrentState::new(cfg.n_layers);
        let mut c = RecurrentState::new(cfg.n_layers);
        v(&dn.prefill(&x, 0, &mut s, &mut c)?)
    })?;
    all &= stable("deltanet.prefill (S)", || {
        let mut s = RecurrentState::new(cfg.n_layers);
        let mut c = RecurrentState::new(cfg.n_layers);
        dn.prefill(&x, 0, &mut s, &mut c)?;
        v(s.get(0).expect("written"))
    })?;
    all &= stable("deltanet.prefill (conv)", || {
        let mut s = RecurrentState::new(cfg.n_layers);
        let mut c = RecurrentState::new(cfg.n_layers);
        dn.prefill(&x, 0, &mut s, &mut c)?;
        v(c.get(0).expect("written"))
    })?;
    let qsa = testing::qsa(&cfg, &dev);
    all &= stable("qsa.prefill", || {
        let mut kv = lloom::models::qwen4exp::KvCache::with_capacity(64);
        v(&qsa.prefill(&x, &mut kv)?)
    })?;
    assert!(all, "a block varies (see TINYDET)");
    Ok(())
}

/// The whole tiny model's prefill: the logits and the state checksums over
/// three fresh runs, with and without a device synchronize before the
/// state is read — separates a computation that varies from a state read
/// that races the GPU.
#[test]
fn whole_model_prefill() -> Result<()> {
    use lloom::nn::model::CausalLM;
    let dev = Device::new_metal(0)?;
    let cfg = Config::tiny();
    let model = testing::resident_model(&cfg, &dev);
    let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9];
    let input = Tensor::from_vec(prompt.clone(), (1, prompt.len()), &dev)?;
    for sync in [false, true] {
        let mut logits_runs = Vec::new();
        let mut sums_runs = Vec::new();
        for _ in 0..3 {
            let mut c = model.new_cache(&cfg, &dev)?;
            let logits = model.forward(&input, 0, &mut c)?;
            let lv: Vec<f32> = logits.flatten_all()?.to_vec1()?;
            if sync {
                dev.synchronize()?;
            }
            sums_runs.push(c.checksums()?);
            logits_runs.push(lv);
        }
        let logits_det = logits_runs[0] == logits_runs[1] && logits_runs[1] == logits_runs[2];
        let sums_det = sums_runs[0] == sums_runs[1] && sums_runs[1] == sums_runs[2];
        eprintln!("TINYDET model.forward sync_before_checksums={sync}: logits deterministic={logits_det} state checksums deterministic={sums_det}");
        if !sums_det {
            for i in 0..sums_runs[0].len() {
                let (n, a, _) = &sums_runs[0][i];
                let (_, b, _) = &sums_runs[1][i];
                if a != b { eprintln!("TINYDET   {n} varies"); }
            }
        }
    }
    Ok(())
}

/// The pool's freed buffers poisoned (`set_free_poison`, the `fuzzing`
/// feature): a kernel that reads memory it never wrote reads NaN, and the
/// state checksums' non-finite counts name the channel.
#[test]
fn whole_model_prefill_poisoned() -> Result<()> {
    use lloom::nn::model::CausalLM;
    let dev = Device::new_metal(0)?;
    if let Device::Metal(m) = &dev {
        m.set_free_poison(Some(0xff));
    }
    let cfg = Config::tiny();
    let model = testing::resident_model(&cfg, &dev);
    let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9];
    let input = Tensor::from_vec(prompt.clone(), (1, prompt.len()), &dev)?;
    // Warm the pool with a run so freed (poisoned) buffers get recycled.
    for run in 0..3 {
        let mut c = model.new_cache(&cfg, &dev)?;
        let logits = model.forward(&input, 0, &mut c)?;
        let lv: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        let nan_logits = lv.iter().filter(|x| !x.is_finite()).count();
        let sums = c.checksums()?;
        let bad: Vec<String> = sums.iter().filter(|(_, _, n)| *n > 0).map(|(n, _, k)| format!("{n}:{k}")).collect();
        eprintln!("TINYDET poisoned run {run}: nonfinite logits={nan_logits} channels with nonfinite={bad:?}");
    }
    Ok(())
}

/// The verify-window test's sequence: a pooled cache (spec pools) prefilled
/// first, then three fresh slab prefills — does the pooled run leave
/// something behind that makes the slab runs vary?
#[test]
fn slab_prefill_after_a_pooled_one() -> Result<()> {
    use lloom::models::qwen4exp::Cache;
    use lloom::nn::model::CausalLM;
    let dev = Device::new_metal(0)?;
    let cfg = Config::tiny();
    let model = testing::resident_model(&cfg, &dev);
    let prompt: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9];
    let input = Tensor::from_vec(prompt.clone(), (1, prompt.len()), &dev)?;
    let check = |label: &str| -> Result<bool> {
        let mut sums = Vec::new();
        for _ in 0..3 {
            let mut c = model.new_cache(&cfg, &dev)?;
            model.forward(&input, 0, &mut c)?;
            sums.push(c.checksums()?);
        }
        let det = sums[0] == sums[1] && sums[1] == sums[2];
        eprintln!("TINYDET slab ×3 {label}: deterministic={det}");
        Ok(det)
    };
    check("before any pooled cache")?;
    let pools = model.spec_pools(1, 3, &dev);
    let mut pc = Cache::in_pools(&cfg, &pools)?;
    check("pools created, pooled cache built")?;
    model.forward(&input, 0, &mut pc)?;
    check("after a pooled prefill")?;
    let _ = pc.checksums()?;
    check("after reading the pooled checksums")?;
    let mut pc2 = Cache::in_pools(&cfg, &model.spec_pools(1, 3, &dev))?;
    pc2.want_nextn = true;
    model.forward(&input, 0, &mut pc2)?;
    let t = Tensor::from_vec(vec![4u32], (1, 1), &dev)?;
    let _ = model.forward(&t, prompt.len(), &mut pc2)?;
    check("after a pooled prefill + decode step (want_nextn)")?;
    Ok(())
}
