//! Private primitive controls only; supplied variates are not a new RNG engine.
//! Reference: the independent contract oracle prepares support/weights; then
//! ascending original ids compare separately rounded f32 ((z-max)*inv_t)+g,
//! ignoring zero weights, with lower id winning numeric ties (including ±0).
//! The f64 contrast is intentional; centered-f32 is the shared scoring contract.
use super::PrivateMultinomial;
use crate::backend::BackendStorage;
use crate::metal_backend::{readback_counts, CutPolicy, HostSlot};
use crate::{DType, Device, Layout, MetalStorage, Result, Storage, Tensor};

use crate::oracle;
use oracle::Params;

fn device() -> Result<Device> {
    crate::isolation::claim("private filtered Gumbel primitive");
    let mut tuning = crate::tuning::Tuning::default();
    tuning.from_env().expect("valid environment");
    tuning.contract.profile = "deterministic".into();
    tuning.instrument.profile = true;
    crate::tuning::install(tuning).expect("install before device");
    assert!(crate::kernels::metal::profile::enabled());
    let device = Device::new_metal(0)?;
    let actual_api = {
        let encoder = device.as_metal_device()?.command_encoder()?;
        if encoder.as_ref().command_buffer.is_metal4() {
            "metal4"
        } else {
            "metal3"
        }
    };
    println!(
        "PRIVATE_GUMBEL_TEST_SETUP actual_api={actual_api} profile=deterministic merged_unit={}",
        crate::kernels::kernel::merged_translation_unit()
    );
    Ok(device)
}

fn checked_encode(
    z: &MetalStorage,
    l: &Layout,
    p: Params,
    g: &MetalStorage,
    gl: &Layout,
) -> Result<PrivateMultinomial> {
    let before = readback_counts();
    let window = z.device().dispatch_census().open().expect("census");
    let result = z.sampling_filtered_gumbel_f32(l, p.temperature, p.k, p.p, g, gl);
    let census = window.close();
    assert_eq!(readback_counts(), before, "primitive readback");
    let n = u64::from(result.is_ok() && l.dims().iter().all(|&d| d != 0));
    for name in ["sampling_topk_mask_f32", "sampling_filtered_gumbel_f32"] {
        assert_eq!(
            census
                .iter()
                .filter(|(label, _)| label == name)
                .map(|(_, n)| *n)
                .sum::<u64>(),
            n,
            "{name}: {census:?}"
        );
    }
    assert_eq!(
        census.iter().map(|(_, n)| *n).sum::<u64>(),
        2 * n,
        "{census:?}"
    );
    result
}
fn encode(z: &Tensor, p: Params, g: &Tensor) -> Result<PrivateMultinomial> {
    let (z, l) = z.storage_and_layout();
    let (g, gl) = g.storage_and_layout();
    let (Storage::Metal(z), Storage::Metal(g)) = (&*z, &*g) else {
        unreachable!()
    };
    checked_encode(z, l, p, g, gl)
}
#[derive(Debug, PartialEq)]
struct Observed {
    ids: Vec<u32>,
    support: Vec<u32>,
    status: Vec<u32>,
}
fn read(result: PrivateMultinomial) -> Result<Observed> {
    Ok(Observed {
        ids: result.ids.to_cpu::<u32>()?,
        support: result.support.to_cpu::<u32>()?,
        status: result.status.to_cpu::<u32>()?,
    })
}
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|x| x.to_bits()).collect()
}

fn reference(z: &[f32], p: Params, noise: &[f32]) -> u32 {
    assert_eq!(z.len(), noise.len());
    assert!(noise.iter().all(|x| x.is_finite()));
    let support = oracle::CONTRACT.nucleus(z, p).unwrap();
    let inv = oracle::inv_t(p.temperature).unwrap();
    let maximum = z[support[0]];
    let weights = oracle::CONTRACT.weights(z, &support, maximum, inv);
    let mut best = None;
    for i in 0..z.len() {
        if weights[i] > 0.0 {
            let delta = z[i] - maximum;
            let scaled = delta * inv;
            let score = scaled + noise[i];
            if best.is_none_or(|(_, previous)| score > previous) {
                best = Some((i as u32, score));
            }
        }
    }
    best.expect("maximum has weight one").0
}
fn check(z: &[f32], p: Params, noise: &[f32], d: &Device) -> Result<Observed> {
    let input = Tensor::from_vec(z.to_vec(), z.len(), d)?;
    let g = Tensor::from_vec(noise.to_vec(), noise.len(), d)?;
    let got = read(encode(&input, p, &g)?)?;
    assert_eq!(bits(&input.to_vec1::<f32>()?), bits(z));
    assert_eq!(bits(&g.to_vec1::<f32>()?), bits(noise));
    let kept = oracle::kept_ids(z, p).unwrap();
    let mut mask = vec![0u32; z.len()];
    for &id in &kept {
        mask[id as usize] = 1;
    }
    assert_eq!(got.support, mask);
    assert_eq!(got.ids, [reference(z, p, noise)]);
    assert_eq!(
        got.status,
        [
            0,
            u32::MAX,
            z.iter().filter(|x| x.is_finite()).count() as u32,
            kept.len() as u32
        ]
    );
    Ok(got)
}

#[test]
fn literal_support_method_original_id_and_zero_weight_contrasts() -> Result<()> {
    let d = device()?;
    for (z, k, p, noise, mask, id) in [
        (
            vec![-0.0, 0.0, 0.0, 0.0],
            0,
            0.5,
            vec![0.0, 1.0, 100.0, 100.0],
            vec![1, 1, 0, 0],
            1,
        ),
        (
            vec![0.0; 4],
            2,
            0.5,
            vec![0.0, 100.0, 100.0, 100.0],
            vec![1, 0, 0, 0],
            0,
        ),
        (
            vec![f32::NEG_INFINITY, -0.0, 0.0],
            0,
            0.0,
            vec![100.0, 0.0, 100.0],
            vec![0, 1, 0],
            1,
        ),
        (
            vec![0.0, 2.0, 1.0],
            2,
            1.0,
            vec![100.0, 0.0, 2.0],
            vec![0, 1, 1],
            2,
        ),
        (vec![0.0, -88.0], 0, 1.0, vec![0.0, 1e6], vec![1, 1], 0),
        (vec![0.0, 0.0], 0, 1.0, vec![-0.0, 0.0], vec![1, 1], 0),
        (
            vec![0.0, 0.0],
            0,
            1.0,
            vec![0.0, f32::from_bits(1)],
            vec![1, 1],
            1,
        ),
    ] {
        let got = check(&z, Params::new(1.0, k, p), &noise, &d)?;
        assert_eq!(got.support, mask);
        assert_eq!(got.ids, [id]);
    }
    // For [0,2,1], K=[id1,id2]. Rank-indexed noise would give id1 the
    // discarded id0's 100 and choose id1; the original-id answer above is id2.
    // Multinomial u=0 on the flat half support chooses id0, Gumbel chooses id1.
    let input = Tensor::from_vec(vec![0.0f32; 4], 4, &d)?;
    let g = Tensor::from_vec(vec![0f32, 1.0, 100.0, 100.0], 4, &d)?;
    let u = Tensor::from_vec(vec![0f32], 1, &d)?;
    let p = Params::new(1.0, 0, 0.5);
    let gumbel = read(encode(&input, p, &g)?)?;
    let (zs, l) = input.storage_and_layout();
    let (us, ul) = u.storage_and_layout();
    let (Storage::Metal(zs), Storage::Metal(us)) = (&*zs, &*us) else {
        unreachable!()
    };
    let multinomial = read(zs.sampling_multinomial(l, p.temperature, p.k, p.p, us, ul)?)?;
    assert_eq!(gumbel.support, multinomial.support);
    assert_eq!(gumbel.status, multinomial.status);
    assert_eq!(gumbel.ids, [1]);
    assert_eq!(multinomial.ids, [0]);
    Ok(())
}

#[test]
fn general_k_endpoints_masks_temperature_and_large_common_offset() -> Result<()> {
    let d = device()?;
    for v in [1usize, 31, 257, 1025, 4099] {
        let z: Vec<f32> = (0..v)
            .map(|i| match i % 6 {
                0 => -0.0,
                1 => 0.0,
                2 => f32::from_bits(1),
                3 => -f32::from_bits(1),
                4 => f32::NEG_INFINITY,
                _ => 0.25,
            })
            .collect();
        let noise: Vec<f32> = (0..v).map(|i| (i % 13) as f32 / 4.0 - 1.0).collect();
        for (k, p, t) in [
            (0, 1.0, 0.7),
            (1, 1.0, 1.0),
            (v / 2 + 1, 0.51, 1.3),
            (v, 0.0, 1.0),
            (usize::MAX, 0.97, 0.125),
        ] {
            check(&z, Params::new(t, k, p), &noise, &d)?;
        }
    }
    let z = [0.0, -1.0];
    let noise = [0.0, 1.0];
    assert_eq!(check(&z, Params::new(0.5, 0, 1.0), &noise, &d)?.ids, [0]);
    assert_eq!(check(&z, Params::new(2.0, 0, 1.0), &noise, &d)?.ids, [1]);
    // Centering before adding noise is required even with all filters off.
    for k in [0, 3, 4, usize::MAX] {
        assert_eq!(
            check(&[1e30; 3], Params::new(1.0, k, 1.0), &[0.0, 1.0, 2.0], &d)?.ids,
            [2]
        );
    }
    for v in [128_000usize, 248_320] {
        let z = vec![0.0; v];
        let noise: Vec<f32> = (0..v).map(|i| i as f32).collect();
        for (k, p) in [(0, 1.0), (1, 1.0), (v, 1.0), (v / 2 + 1, 0.5)] {
            let count = if k == 0 { v } else { k };
            let count = if p == 0.5 { count.div_ceil(2) } else { count };
            let got = check(&z, Params::new(1.0, k, p), &noise, &d)?;
            assert_eq!(got.ids, [(count - 1) as u32]);
            assert_eq!(got.status[3], count as u32);
        }
    }
    // f64 p!=1 remains active even if narrowing gives f32 1.
    let p = f64::from_bits(1.0f64.to_bits() - 1);
    assert_eq!(
        check(
            &[-50.0, 0.0, 0.0],
            Params::new(1.0, 0, p),
            &[100.0, 0.0, 0.0],
            &d
        )?
        .support,
        [0, 1, 1]
    );
    Ok(())
}

#[test]
fn centered_f32_discriminator_rejects_f64_and_keeps_low_id_tie() -> Result<()> {
    let d = device()?;
    let noise = [0.0, f32::from_bits(0x3eaa_aaab)];
    // Host centered f64 score for id1 is positive; f32 -inv+g is zero.
    assert!(-1.0f64 / 3.0 + noise[1] as f64 > 0.0);
    let got = check(&[0.0, -1.0], Params::new(3.0, 0, 1.0), &noise, &d)?;
    assert_eq!(got.ids, [0]);
    println!("GUMBEL_ARITHMETIC_CONTRAST f64_id=1 shared_centered_f32_id=0");
    Ok(())
}

#[test]
fn invalid_rows_and_variates_never_return_successful_ids() -> Result<()> {
    let d = device()?;
    for (z, noise, code, bad, finite) in [
        (
            vec![f32::NEG_INFINITY; 3],
            vec![f32::NAN; 3],
            3,
            u32::MAX,
            0,
        ),
        (
            vec![5.0, f32::from_bits(0x7f800001), 0.0],
            vec![f32::NAN; 3],
            1,
            1,
            2,
        ),
        (vec![5.0, f32::INFINITY, 0.0], vec![f32::NAN; 3], 2, 1, 2),
        (
            vec![5.0, 0.0, f32::NEG_INFINITY],
            vec![0.0, 0.0, f32::NAN],
            4,
            2,
            2,
        ),
    ] {
        let input = Tensor::from_vec(z, 3, &d)?;
        let g = Tensor::from_vec(noise, 3, &d)?;
        let got = read(encode(&input, Params::new(1.0, 1, 0.0), &g)?)?;
        assert_eq!(got.ids, [u32::MAX]);
        assert_eq!(got.support, [0, 0, 0]);
        assert_eq!(got.status, [code, bad, finite, 0]);
    }
    for bad in [
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7f800001),
        f32::from_bits(0xffc01234),
    ] {
        let z = Tensor::from_vec(vec![0f32; 3], 3, &d)?;
        let g = Tensor::from_vec(vec![0.0, bad, bad], 3, &d)?;
        let got = read(encode(&z, Params::new(1.0, 1, 0.0), &g)?)?;
        assert_eq!(got.ids, [u32::MAX]);
        assert_eq!(got.support, [0, 0, 0]);
        assert_eq!(got.status, [4, 1, 3, 0]);
    }
    Ok(())
}

#[test]
fn row_offsets_strides_permutations_and_repeated_supplied_noise() -> Result<()> {
    let d = device()?;
    let (rows, v) = (3usize, 4usize);
    let data = vec![
        0f32,
        2.0,
        1.0,
        -1.0,
        1e30,
        1e30,
        1e30,
        1e30,
        0.0,
        f32::NEG_INFINITY,
        -0.0,
        0.0,
    ];
    let noise = vec![
        100f32, 0.0, 2.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 100.0, 1.0, 2.0,
    ];
    let p = Params::new(1.0, 3, 1.0);
    let input = Tensor::from_vec(data.clone(), (rows, v), &d)?;
    let g = Tensor::from_vec(noise.clone(), (rows, v), &d)?;
    let baseline = read(encode(&input, p, &g)?)?;
    assert_eq!(baseline.ids, [0, 2, 3]);
    let pad = |values: &[f32]| -> Result<Tensor> {
        let mut data = vec![f32::NAN; v];
        data.extend_from_slice(values);
        data.extend(vec![f32::INFINITY; v]);
        Tensor::from_vec(data, (rows + 2, v), &d)?.narrow(0, 1, rows)
    };
    let zpad = pad(&data)?;
    let gpad = pad(&noise)?;
    for _ in 0..3 {
        assert_eq!(read(encode(&zpad, p, &gpad)?)?, baseline);
    }
    assert_eq!(bits(&zpad.flatten_all()?.to_vec1::<f32>()?), bits(&data));
    assert_eq!(bits(&gpad.flatten_all()?.to_vec1::<f32>()?), bits(&noise));
    for &r in &[2usize, 0, 1] {
        let one = input.narrow(0, r, 1)?.squeeze(0)?;
        let one_g = g.narrow(0, r, 1)?.squeeze(0)?;
        let got = read(encode(&one, p, &one_g)?)?;
        assert_eq!(got.ids, [baseline.ids[r]]);
        assert_eq!(
            got.ids,
            [reference(
                &data[r * v..(r + 1) * v],
                p,
                &noise[r * v..(r + 1) * v]
            )]
        );
    }
    let order = [2usize, 0, 1];
    let permute = |data: &[f32]| {
        order
            .iter()
            .flat_map(|&r| data[r * v..(r + 1) * v].iter().copied())
            .collect::<Vec<_>>()
    };
    let zp = Tensor::from_vec(permute(&data), (1, rows, v), &d)?;
    let gp = Tensor::from_vec(permute(&noise), (1, rows, v), &d)?;
    let got = read(encode(&zp, p, &gp)?)?;
    for (i, &r) in order.iter().enumerate() {
        assert_eq!(got.ids[i], baseline.ids[r]);
        assert_eq!(
            &got.support[i * v..(i + 1) * v],
            &baseline.support[r * v..(r + 1) * v]
        );
        assert_eq!(
            &got.status[4 * i..4 * i + 4],
            &baseline.status[4 * r..4 * r + 4]
        );
    }
    Ok(())
}

#[test]
fn complete_preflight_and_zero_rows_encode_nothing_invalid() -> Result<()> {
    let d = device()?;
    let input = Tensor::from_vec(vec![0f32; 8], (2, 4), &d)?;
    let noise = Tensor::from_vec(vec![0f32; 8], (2, 4), &d)?;
    let (zs, l) = input.storage_and_layout();
    let (gs, gl) = noise.storage_and_layout();
    let (Storage::Metal(z), Storage::Metal(g)) = (&*zs, &*gs) else {
        unreachable!()
    };
    let p = Params::new(1.0, 0, 1.0);
    let invalid_layouts = [
        Layout::contiguous(()),
        Layout::contiguous((1, 0)),
        Layout::contiguous_with_offset((2, 4), 1),
        Layout::new((2, 4).into(), vec![1, 2], 0),
        Layout::new((2, 4).into(), vec![4], 0),
        Layout::new((usize::MAX, 2).into(), vec![2, 1], 0),
        Layout::contiguous_with_offset((2, 4), usize::MAX),
        Layout::new(
            (1, u32::MAX as usize + 1).into(),
            vec![u32::MAX as usize + 1, 1],
            0,
        ),
        Layout::new((u32::MAX as usize + 1, 1).into(), vec![1, 1], 0),
    ];
    for bad in &invalid_layouts {
        assert!(checked_encode(z, bad, p, g, gl).is_err());
        assert!(checked_encode(z, l, p, g, bad).is_err());
    }
    assert!(checked_encode(z, l, p, g, &Layout::contiguous((1, 8))).is_err());
    let overstated = MetalStorage::new(
        g.buffer_handle().clone(),
        g.device().clone(),
        g.buffer().length() / 4 + 1,
        DType::F32,
    );
    let words = overstated.buffer().length() / 4 + 1;
    let over_layout = Layout::contiguous((1, words));
    assert!(checked_encode(&overstated, &over_layout, p, g, &over_layout).is_err());
    assert!(checked_encode(
        z,
        l,
        p,
        &overstated,
        &Layout::contiguous_with_offset((2, 4), words)
    )
    .is_err());
    for (t, p) in [
        (0.0, 1.0),
        (-1.0, 1.0),
        (f64::NAN, 1.0),
        (f64::INFINITY, 1.0),
        (1e-45, 1.0),
        (1.0, -0.1),
        (1.0, 1.01),
        (1.0, f64::NAN),
    ] {
        assert!(checked_encode(z, l, Params::new(t, 0, p), g, gl).is_err());
    }
    let zero = Layout::contiguous((0, 4));
    assert_eq!(
        read(checked_encode(z, &zero, p, g, &zero)?)?,
        Observed {
            ids: vec![],
            support: vec![],
            status: vec![]
        }
    );
    assert!(checked_encode(z, &zero, Params::new(0.0, 0, 1.0), g, &zero).is_err());
    drop(zs);
    drop(gs);
    assert!(encode(&input.to_dtype(DType::F16)?, p, &noise).is_err());
    assert!(encode(&input, p, &noise.to_dtype(DType::F16)?).is_err());
    let other = Device::new_metal(0)?;
    let other_noise = Tensor::from_vec(vec![0f32; 8], (2, 4), &other)?;
    assert!(encode(&input, p, &other_noise).is_err());
    Ok(())
}

#[test]
fn raw_abi_offsets_scratch_initialization_and_canaries() -> Result<()> {
    let d = device()?;
    let (rows, v, guard) = (4usize, 3usize, 11usize);
    let data = vec![
        f32::NAN,
        0.0,
        2.0,
        1.0,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        1.0,
        f32::NAN,
        0.0,
        0.0,
        0.0,
        0.0,
        f32::INFINITY,
    ];
    let noise = vec![
        f32::NAN,
        100.0,
        0.0,
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        f32::INFINITY,
        0.0,
        f32::NAN,
    ];
    let input = Tensor::from_vec(data.clone(), data.len(), &d)?;
    let g = Tensor::from_vec(noise.clone(), noise.len(), &d)?;
    let make = |n| Tensor::from_vec(vec![0xdecafbadu32; n + guard], n + guard, &d);
    let w = make(rows * v)?;
    let support = make(rows * v)?;
    let ids = make(rows)?;
    let status = make(4 * rows)?;
    {
        let (zs, _) = input.storage_and_layout();
        let (gs, _) = g.storage_and_layout();
        let (ws, _) = w.storage_and_layout();
        let (ms, _) = support.storage_and_layout();
        let (is, _) = ids.storage_and_layout();
        let (ss, _) = status.storage_and_layout();
        let (
            Storage::Metal(z),
            Storage::Metal(g),
            Storage::Metal(w),
            Storage::Metal(mask),
            Storage::Metal(id),
            Storage::Metal(s),
        ) = (&*zs, &*gs, &*ws, &*ms, &*is, &*ss)
        else {
            unreachable!()
        };
        let (k, ks) = z.sampling_topk_mask(&Layout::contiguous_with_offset((rows, v), 1), 2)?;
        let m = z.device();
        let encoder = m.command_encoder()?;
        // SAFETY: all inputs/outputs disjoint and sized with guard tails,
        // selector belongs to these exact rows, live storage retains custody.
        unsafe {
            crate::kernels::kernels::sampling_topk::call_sampling_filtered_gumbel_f32(
                m.metal_device(),
                &encoder,
                m.kernels(),
                (z.buffer(), 4),
                rows as u32,
                v as u32,
                1.0,
                1.0,
                0,
                (g.buffer(), 4),
                k.buffer(),
                ks.buffer(),
                w.buffer(),
                mask.buffer(),
                id.buffer(),
                s.buffer(),
            )
        }
        .map_err(crate::Error::wrap)?;
    }
    let got_ids = ids.to_vec1::<u32>()?;
    let mask = support.to_vec1::<u32>()?;
    let stat = status.to_vec1::<u32>()?;
    let weights = w.to_vec1::<u32>()?;
    assert_eq!(&got_ids[..rows], &[2, u32::MAX, u32::MAX, u32::MAX]);
    assert_eq!(&mask[..rows * v], &[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(
        &stat[..rows * 4],
        &[0, u32::MAX, 3, 2, 3, u32::MAX, 0, 0, 1, 1, 2, 0, 4, 1, 3, 0]
    );
    assert_eq!(weights[0], 0);
    assert_eq!(weights[1], 1f32.to_bits());
    assert_eq!(weights[2], oracle::exp32(-1.0).to_bits());
    assert!(weights[3..rows * v].iter().all(|&x| x == 0));
    for (buffer, n) in [
        (&got_ids, rows),
        (&mask, rows * v),
        (&stat, 4 * rows),
        (&weights, rows * v),
    ] {
        assert!(buffer[n..].iter().all(|&x| x == 0xdecafbad));
    }
    assert_eq!(bits(&input.to_vec1::<f32>()?), bits(&data));
    assert_eq!(bits(&g.to_vec1::<f32>()?), bits(&noise));
    Ok(())
}

fn dependency(force_cut: bool) -> Result<()> {
    use crate::kernels::metal::commands::CommandBufferKind;
    use std::sync::Arc;
    let d = device()?;
    let m = d.as_metal_device()?;
    assert_eq!(crate::tuning::tuning().metal.command_api, "metal3");
    assert_eq!(m.commands().kind(), CommandBufferKind::Retaining);
    assert!(m.commands().executor().is_classical());
    assert!(!m.base_mode());
    assert_eq!(m.cut_policy(), CutPolicy::DispatchCounter);
    m.set_compute_per_buffer(usize::MAX);
    let z = Tensor::from_vec(vec![3f32, 1.0, -2.0, -2.0], (1, 4), &d)?;
    let noise = Tensor::from_vec(vec![0f32, 0.0, 0.0, 1.0], (1, 4), &d)?;
    let id_slot = HostSlot::new(m, 1, DType::U32)?;
    let support_slot = HostSlot::new(m, 4, DType::U32)?;
    let status_slot = HostSlot::new(m, 4, DType::U32)?;
    for (slot, n) in [(&id_slot, 1), (&support_slot, 4), (&status_slot, 4)] {
        slot.fill_u32(0xdecafbad, n)?;
    }
    let snapshot = || -> Result<_> {
        let enc = m.command_encoder()?;
        let enc = enc.as_ref();
        Ok((enc.command_buffer.clone(), Arc::clone(&enc.state)))
    };
    let (zs, l) = z.storage_and_layout();
    let (gs, gl) = noise.storage_and_layout();
    let (Storage::Metal(zs), Storage::Metal(gs)) = (&*zs, &*gs) else {
        unreachable!()
    };
    let produced = zs.affine(l, -1.0, 1.0)?;
    let produced_noise = gs.affine(gl, 2.0, 0.0)?;
    let producer = snapshot()?;
    if force_cut {
        m.commands().flush().map_err(crate::Error::wrap)?;
        m.set_compute_per_buffer(1);
    }
    let before = readback_counts();
    let result = checked_encode(&produced, l, Params::new(1.0, 2, 1.0), &produced_noise, gl)?;
    if force_cut {
        m.set_compute_per_buffer(usize::MAX);
    }
    let sampler = snapshot()?;
    assert_eq!(Arc::ptr_eq(&producer.1, &sampler.1), !force_cut);
    assert_eq!(producer.0.identity() == sampler.0.identity(), !force_cut);
    if force_cut {
        m.commands().flush().map_err(crate::Error::wrap)?;
    }
    let wrap = |s, shape| {
        Tensor::from_storage(
            Storage::Metal(s),
            shape,
            crate::op::BackpropOp::none(),
            false,
        )
    };
    // A test-only diagnostic sink; never a public sampled Tensor/embedding.
    let ids = wrap(result.ids, crate::Shape::from((1,)));
    let support = wrap(result.support, crate::Shape::from((4,)));
    let status = wrap(result.status, crate::Shape::from((4,)));
    id_slot.write_from(&ids)?;
    let consumer = snapshot()?;
    support_slot.write_from(&support)?;
    status_slot.write_from(&status)?;
    assert_eq!(Arc::ptr_eq(&sampler.1, &consumer.1), !force_cut);
    assert_eq!(sampler.0.identity() == consumer.0.identity(), !force_cut);
    assert_eq!(readback_counts(), before);
    m.wait_until_completed()?;
    assert_eq!(id_slot.read_unchecked::<u32>(1)?, [3]);
    assert_eq!(support_slot.read_unchecked::<u32>(4)?, [0, 0, 1, 1]);
    assert_eq!(status_slot.read_unchecked::<u32>(4)?, [0, u32::MAX, 4, 2]);
    assert_eq!(readback_counts().since(&before).slot, 3);
    println!("PRIVATE_GUMBEL_DEPENDENCY force_cut={force_cut} data_wait_completed=true id=3");
    Ok(())
}
#[test]
fn queued_score_and_noise_producers_draw_diagnostic_consumers_same_encoder() -> Result<()> {
    dependency(false)
}
#[test]
fn queued_score_and_noise_producers_draw_diagnostic_consumers_forced_cuts() -> Result<()> {
    dependency(true)
}
