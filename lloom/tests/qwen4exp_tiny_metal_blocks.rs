//! Each block at the TINY geometry on Metal against the CPU: finite, and
//! within fp32 reorder of the CPU — the tiny model is what every model-level
//! test decodes, so a kernel that misreads an edge shape here (a head under
//! a simdgroup, a 2-wide top-k over 8 experts) poisons them silently.
#![cfg(feature = "metal")]
use lloom::models::qwen4exp::testing;
use lloom::models::qwen4exp::Config;
use lloom_tensor::{Device, Result, Tensor};

fn row(cfg: &Config, seed: f32, seq: usize, dev: &Device) -> Tensor {
    let d = cfg.hidden_size;
    Tensor::from_vec(
        (0..seq * d)
            .map(|i| ((i as f32 * 0.05 + seed).sin()) * 0.3)
            .collect::<Vec<f32>>(),
        (1, seq, d),
        dev,
    )
    .unwrap()
}

fn compare(name: &str, a: &Tensor, b: &Tensor) -> Result<()> {
    let av: Vec<f32> = a.flatten_all()?.to_vec1()?;
    let bv: Vec<f32> = b.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?;
    let nonfinite = bv.iter().filter(|v| !v.is_finite()).count();
    let scale = av.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
    let diff = av
        .iter()
        .zip(&bv)
        .map(|(p, q)| (p - q).abs())
        .fold(0f32, f32::max);
    println!("BLOCK {name} nonfinite={nonfinite} maxdiff={diff:e} scale={scale:e}");
    assert_eq!(nonfinite, 0, "{name}: non-finite on Metal");
    assert!(
        diff <= 1e-4 * scale,
        "{name}: metal differs by {diff} at scale {scale}"
    );
    Ok(())
}

#[test]
fn tiny_blocks_metal_vs_cpu() -> Result<()> {
    let Ok(gpu) = Device::new_metal(0) else {
        return Ok(());
    };
    let cpu = Device::Cpu;
    let cfg = Config::tiny();
    for seq in [1usize, 7] {
        let xc = row(&cfg, 0.0, seq, &cpu);
        let xg = xc.to_device(&gpu)?;
        // MoE (resident, f32 stacks as the tiny model's)
        let mc = testing::moe(&cfg, &cpu);
        let mg = testing::moe(&cfg, &gpu);
        compare(
            &format!("moe seq={seq}"),
            &mc.forward(&xc)?,
            &mg.forward(&xg)?,
        )?;
        // HC bracket
        let hc_c = testing::hc(&cfg, 1.0, &cpu);
        let hc_g = testing::hc(&cfg, 1.0, &gpu);
        let nd = cfg.hc_dim();
        let hcx = Tensor::from_vec(
            (0..seq * nd)
                .map(|i| ((i as f32 * 0.07).cos()) * 0.3)
                .collect::<Vec<f32>>(),
            (1, seq, nd),
            &cpu,
        )?;
        let hgx = hcx.to_device(&gpu)?;
        let (mixed_c, inj_c) = hc_c.mix(&hcx)?;
        let (mixed_g, inj_g) = hc_g.mix(&hgx)?;
        compare(&format!("hc.mix seq={seq}"), &mixed_c, &mixed_g)?;
        compare(&format!("hc.inject seq={seq}"), &inj_c, &inj_g)?;
        compare(
            &format!("hc.combine seq={seq}"),
            &hc_c.combine(&hcx, &mixed_c, &inj_c)?,
            &hc_g.combine(&hgx, &mixed_g, &inj_g)?,
        )?;
    }
    Ok(())
}
