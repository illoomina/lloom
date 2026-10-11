use super::*;
use serde_json::Value;

fn data() -> Value {
    serde_json::from_str(include_str!("scalar_fixture.json")).unwrap()
}
fn expected(data: &Value, key: &str) -> Vec<f64> {
    data[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}
fn close(got: &[f32], want: &[f64], tolerance: f64, label: &str) {
    assert_eq!(got.len(), want.len());
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        assert!(
            (a as f64 - b).abs() < tolerance,
            "frontend {label} index{i}: {a} != {b}"
        );
    }
}

#[test]
fn host_frontend_matches_sparse_direct_dft_and_cast_contract() -> Result<()> {
    let data = data();
    let frontend = MelFrontend::new(4)?;
    let bank = expected(&data, "filterbank");
    let window = expected(&data, "window");
    assert_eq!(bank.len(), 4 * 257);
    assert_eq!(window.len(), 512);
    close(&frontend.filterbank, &bank, 1e-9, "Slaney cast order");
    close(&frontend.window, &window, 2e-6, "nonperiodic centered Hann");
    let input = data["samples_i16"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i16)
        .collect::<Vec<_>>();
    assert_eq!(input.len(), 37);
    let want = expected(&data, "output");
    assert_eq!(want.len(), 4 * 51);
    for key in ["wrong_center", "wrong_padding_before_preemphasis"] {
        let wrong = expected(&data, key);
        assert_eq!(wrong.len(), want.len());
        assert!(
            want.iter().zip(wrong).any(|(a, b)| (a - b).abs() > 1e-2),
            "frontend oracle {key} discriminator vacuous"
        );
    }
    let mut prior = None;
    for _ in 0..3 {
        let got = frontend.prepare_i16(&input)?;
        assert_eq!(
            (got.features, got.computed_frames, got.valid_frames),
            (4, 51, 50)
        );
        close(&got.values, &want, 5e-5, "sparse DFT normalization");
        for row in got.values.chunks_exact(51) {
            assert_eq!(row[50].to_bits(), 0f32.to_bits());
        }
        let bits = got.values.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        if let Some(ref before) = prior {
            assert_eq!(&bits, before);
        }
        prior = Some(bits);
    }
    let floats = input.iter().map(|&x| x as f32 / 32768.).collect::<Vec<_>>();
    let converted = frontend
        .prepare_f32(&floats)?
        .values
        .iter()
        .map(|x| x.to_bits())
        .collect::<Vec<_>>();
    assert_eq!(Some(converted), prior);
    Ok(())
}

#[test]
fn frontend_demand_is_checked_without_large_allocation() -> Result<()> {
    assert!(MelFrontend::table_bytes(0).is_err());
    assert!(MelFrontend::table_bytes(usize::MAX).is_err());
    assert!(MelFrontend::table_bytes(usize::MAX / 2).is_err());
    assert_eq!(MelFrontend::table_bytes(4)?, (6160, 48));
    let frontend = MelFrontend::new(4)?;
    let min = frontend.scratch(0)?;
    assert_eq!(
        (min.samples, min.computed_frames, min.valid_frames),
        (8000, 51, 50)
    );
    assert_eq!(min.output_bytes, 816);
    assert_eq!(min.allocation_volume_bytes, 3064884);
    assert_eq!(min.peak_numeric_bytes_bound, 52276);
    let max = frontend.scratch(usize::MAX)?; // clamps length metadata only
    assert_eq!(
        (max.samples, max.computed_frames, max.valid_frames),
        (480000, 3001, 3000)
    );
    assert_eq!(max.output_bytes, 4 * 3001 * 4);
    assert!(max.allocation_volume_bytes > max.peak_numeric_bytes_bound);
    assert!(frontend.scratch(8001)?.output_bytes == min.output_bytes);
    Ok(())
}

#[test]
fn charged_tables_retain_one_domain_after_scratch_and_pass_release() -> Result<()> {
    use lloom_tensor::memory_admission::{process_account, AccountSnapshot};
    let (retained, scratch) = MelFrontend::table_bytes(4)?;
    assert_eq!((retained, scratch), (6160, 48));
    // Numerical ownership control only: deliberately no machine admission.
    let domain = process_account()
        .unadmitted_domain(Some(retained + scratch))
        .map_err(lloom_tensor::Error::wrap)?;
    let pass = domain
        .reserve(retained + scratch)
        .map_err(lloom_tensor::Error::wrap)?;
    let context = pass.context();
    let frontend = MelFrontend::new_in(4, &context)?;
    assert_eq!(
        domain.snapshot(),
        AccountSnapshot {
            pending: 0,
            committed: retained
        }
    );
    let legacy = MelFrontend::new(4)?;
    assert_eq!(frontend.filterbank.len(), legacy.filterbank.len());
    assert_eq!(frontend.window.len(), legacy.window.len());
    for (a, b) in frontend.filterbank.iter().zip(legacy.filterbank.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (a, b) in frontend.window.iter().zip(legacy.window.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    drop(pass);
    assert!(MelFrontend::new_in(4, &context).is_err());
    assert_eq!(
        domain.snapshot(),
        AccountSnapshot {
            pending: 0,
            committed: retained
        }
    );
    drop(frontend);
    assert_eq!(domain.snapshot(), AccountSnapshot::default());
    Ok(())
}

#[test]
fn table_refusal_releases_partial_scratch_without_reopening_issuance() -> Result<()> {
    use lloom_tensor::memory_admission::{process_account, AccountSnapshot};
    let domain = process_account()
        .unadmitted_domain(Some(48))
        .map_err(lloom_tensor::Error::wrap)?;
    let pass = domain.reserve(48).map_err(lloom_tensor::Error::wrap)?;
    let context = pass.context();
    // Only the six F64 edges fit; filterbank ticket refuses before backing.
    assert!(MelFrontend::new_in(4, &context).is_err());
    assert_eq!(domain.snapshot(), AccountSnapshot::default());
    drop(pass);
    assert!(MelFrontend::new_in(4, &context).is_err());
    assert_eq!(domain.snapshot(), AccountSnapshot::default());
    Ok(())
}
