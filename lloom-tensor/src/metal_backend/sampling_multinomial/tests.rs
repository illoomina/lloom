use super::PrivateMultinomial;
use crate::backend::BackendStorage;
use crate::metal_backend::{readback_counts, CutPolicy, HostSlot};
use crate::{DType, Device, Layout, MetalStorage, Result, Storage, Tensor};

use crate::{fixtures, oracle};
use oracle::Params;

fn device() -> Result<Device> {
    crate::isolation::claim("private contract sampler census");
    let mut tuning = crate::tuning::Tuning::default();
    tuning.from_env().expect("valid test environment");
    tuning.contract.profile = "deterministic".into();
    tuning.instrument.profile = true;
    crate::tuning::install(tuning).expect("install before tuning is read");
    assert!(crate::kernels::metal::profile::enabled());
    let device = Device::new_metal(0)?;
    eprintln!(
        "SAMPLER_NATIVE_SETUP actual_api={}",
        device.as_metal_device()?.commands().actual_command_api()
    );
    Ok(device)
}

fn encode(z: &Tensor, params: Params, u: &Tensor) -> Result<PrivateMultinomial> {
    let (z, layout) = z.storage_and_layout();
    let (u, ul) = u.storage_and_layout();
    let (Storage::Metal(z), Storage::Metal(u)) = (&*z, &*u) else {
        unreachable!()
    };
    checked_encode(z, layout, params, u, ul)
}

fn checked_encode(
    z: &MetalStorage,
    layout: &Layout,
    params: Params,
    u: &MetalStorage,
    ul: &Layout,
) -> Result<PrivateMultinomial> {
    let before = readback_counts();
    let window = z.device().dispatch_census().open().expect("census window");
    let result = z.sampling_multinomial(layout, params.temperature, params.k, params.p, u, ul);
    let census = window.close();
    assert_eq!(readback_counts(), before, "sampler read back");
    let expected = if result.is_ok() && layout.shape().elem_count() != 0 {
        1
    } else {
        0
    };
    use crate::kernels::kernels::sampling_topk::{MULTINOMIAL_KERNELS, SELECT_KERNELS};
    for name in SELECT_KERNELS.iter().chain(MULTINOMIAL_KERNELS) {
        let count: u64 = census
            .iter()
            .filter(|(label, _)| label == name)
            .map(|(_, n)| *n)
            .sum();
        assert_eq!(count, expected, "{name}: {census:?}");
    }
    assert_eq!(
        census.iter().map(|(_, n)| *n).sum::<u64>(),
        expected * (SELECT_KERNELS.len() + MULTINOMIAL_KERNELS.len()) as u64,
        "unexpected dispatch: {census:?}"
    );
    result
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
fn bits(z: &[f32]) -> Vec<u32> {
    z.iter().map(|x| x.to_bits()).collect()
}

fn batch(z: &[f32], params: Params, uniforms: &[f32], d: &Device) -> Result<Observed> {
    let input = Tensor::from_vec(z.repeat(uniforms.len()), (uniforms.len(), z.len()), d)?;
    let u = Tensor::from_vec(uniforms.to_vec(), uniforms.len(), d)?;
    let got = read(encode(&input, params, &u)?)?;
    assert_eq!(
        bits(&input.flatten_all()?.to_vec1::<f32>()?),
        bits(&z.repeat(uniforms.len()))
    );
    assert_eq!(bits(&u.to_vec1::<f32>()?), bits(uniforms));
    for (row, &u) in uniforms.iter().enumerate() {
        assert_row(
            z,
            params,
            u,
            got.ids[row],
            &got.support[row * z.len()..(row + 1) * z.len()],
            &got.status[row * 4..row * 4 + 4],
        );
    }
    Ok(got)
}

fn assert_row(z: &[f32], params: Params, u: f32, id: u32, mask: &[u32], status: &[u32]) {
    assert!(mask.iter().all(|&x| x <= 1));
    let kept: Vec<u32> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &x)| (x == 1).then_some(i as u32))
        .collect();
    let finite = z.iter().filter(|x| x.is_finite()).count() as u32;
    match oracle::draw(z, params, u) {
        Ok(want) => {
            assert_eq!(id, want, "{params:?}, u={u:?}");
            assert_eq!(kept, oracle::kept_ids(z, params).unwrap());
            assert_eq!(status, [0, u32::MAX, finite, kept.len() as u32]);
            assert!(mask[id as usize] != 0 && z[id as usize].is_finite());
        }
        Err(why) => {
            // Score validation precedes per-row uniform validation in the ABI;
            // oracle::draw checks uniforms first. Cases with both invalid are
            // tested separately against the explicit private precedence below.
            let (code, bad) = match why {
                oracle::Refusal::NotANumber(i) => (1, i as u32),
                oracle::Refusal::PositiveInfinity(i) => (2, i as u32),
                oracle::Refusal::NoCandidate => (3, u32::MAX),
                oracle::Refusal::Uniform => (4, u32::MAX),
                other => panic!("host-invalid parameter reached GPU: {other:?}"),
            };
            assert_eq!(id, u32::MAX, "invalid row returned a candidate");
            assert!(kept.is_empty());
            assert_eq!(status, [code, bad, finite, 0]);
        }
    }
}

#[test]
fn literal_support_ties_conditional_mass_and_cdf_endpoints() -> Result<()> {
    let d = device()?;
    // Exact equal weights fix these without evaluating the candidate or oracle.
    for (z, k, p, u, support, id) in [
        (vec![-0.0, 0.0, 0.0, 0.0], 0, 0.5, 0.0, vec![1, 1, 0, 0], 0),
        (vec![-0.0, 0.0, 0.0, 0.0], 0, 0.5, 0.5, vec![1, 1, 0, 0], 1),
        (
            vec![0.0; 4],
            0,
            f32::from_bits(0x3f000001) as f64,
            0.0,
            vec![1, 1, 1, 0],
            0,
        ),
        (
            vec![0.0; 4],
            0,
            f32::from_bits(0x3effffff) as f64,
            0.0,
            vec![1, 1, 0, 0],
            0,
        ),
        (vec![0.0; 4], 2, 0.5, fixtures::U_MAX, vec![1, 0, 0, 0], 0),
        (
            vec![f32::NEG_INFINITY, 0.0, -0.0],
            0,
            0.0,
            0.5,
            vec![0, 1, 0],
            1,
        ),
        (vec![-50.0, 0.0, 0.0], 0, 1.0, 0.0, vec![1, 1, 1], 0),
        // This p is NOT 1 in f64, but rounds to f32 1. Do not disable P.
        (
            vec![-50.0, 0.0, 0.0],
            0,
            f64::from_bits(1.0f64.to_bits() - 1),
            0.0,
            vec![0, 1, 1],
            1,
        ),
        (vec![0.0, -88.0], 0, 1.0, fixtures::U_MAX, vec![1, 1], 0),
    ] {
        let got = batch(&z, Params::new(1.0, k, p), &[u], &d)?;
        assert_eq!(got.support, support);
        assert_eq!(got.ids, [id]);
    }
    // Same unfiltered math arm, including k>=V and the unsigned upper bound.
    let z = [-0.0, -1.0, f32::NEG_INFINITY, 0.5, -50.0];
    let reference = batch(
        &z,
        Params::new(0.7, 0, 1.0),
        &[0.0, 0.25, 0.5, fixtures::U_MAX],
        &d,
    )?;
    for k in [z.len(), z.len() + 1, usize::MAX] {
        assert_eq!(
            batch(
                &z,
                Params::new(0.7, k, 1.0),
                &[0.0, 0.25, 0.5, fixtures::U_MAX],
                &d
            )?,
            reference
        );
    }
    // Finite-score subtraction can overflow; inv_t can round to zero.
    batch(
        &[-f32::MAX, f32::MAX],
        Params::new(f64::MAX, 0, 1.0),
        &[0.0, fixtures::U_MAX],
        &d,
    )?;
    batch(
        &[0.0, -f32::from_bits(1)],
        Params::new(1e-7, 0, 0.5),
        &[0.0, fixtures::U_MAX],
        &d,
    )?;
    println!("MULTINOMIAL_LITERALS independent support/id expectations reached");
    Ok(())
}

#[test]
fn all_existing_fixtures_and_widened_rows_match_independent_references() -> Result<()> {
    let d = device()?;
    let (mut kept, mut row_refused, mut parameter_refused, mut widened, mut literal_draws) =
        (0, 0, 0, 0, 0);
    for case in fixtures::all() {
        let invalid = oracle::inv_t(case.params.temperature).is_err()
            || !(0.0..=1.0).contains(&case.params.p);
        if invalid {
            let z = Tensor::from_vec(case.z.clone(), case.z.len(), &d)?;
            let u = Tensor::from_vec(vec![0.5f32], 1, &d)?;
            assert!(encode(&z, case.params, &u).is_err(), "{}", case.name);
            if case.f16_exact {
                let widened_z = z.to_dtype(DType::F16)?.to_dtype(DType::F32)?;
                assert!(
                    encode(&widened_z, case.params, &u).is_err(),
                    "{} widened",
                    case.name
                );
            }
            parameter_refused += 1;
        } else {
            let mut uniforms = vec![0.0, 0.5, fixtures::U_MAX];
            uniforms.extend(case.draws.iter().map(|&(u, _)| u));
            let got = batch(&case.z, case.params, &uniforms, &d)?;
            match &case.expect {
                fixtures::Expect::Kept(ids) => {
                    let expected: Vec<u32> = (0..case.z.len())
                        .map(|i| u32::from(ids.contains(&(i as u32))))
                        .collect();
                    assert_eq!(&got.support[..case.z.len()], &expected, "{}", case.name);
                    kept += 1;
                }
                fixtures::Expect::Refused(_) => row_refused += 1,
            }
            for (i, &(_, id)) in case.draws.iter().enumerate() {
                assert_eq!(got.ids[3 + i], id, "{} literal draw", case.name);
                literal_draws += 1;
            }
            if case.f16_exact {
                let z = Tensor::from_vec(case.z.clone(), case.z.len(), &d)?
                    .to_dtype(DType::F16)?
                    .to_dtype(DType::F32)?;
                let widened_z = z.to_vec1::<f32>()?;
                let again = batch(&widened_z, case.params, &uniforms, &d)?;
                assert_eq!(again, got, "{} widened", case.name);
            }
        }
        if case.f16_exact {
            widened += 1;
        }
    }
    assert_eq!(
        (kept, row_refused, parameter_refused, widened, literal_draws),
        (33, 4, 7, 41, 48)
    );
    println!("MULTINOMIAL_FIXTURES kept={kept} row_refused={row_refused} parameter_refused={parameter_refused} f16_flags={widened} literal_draws={literal_draws}");
    Ok(())
}

#[test]
fn general_k_ieee_ranks_and_every_flat_prefix_boundary() -> Result<()> {
    let d = device()?;
    for v in [1usize, 31, 257, 1025, 4099] {
        let z: Vec<f32> = (0..v)
            .map(|i| match i % 8 {
                0 => -0.0,
                1 => 0.0,
                2 => f32::from_bits(1),
                3 => -f32::from_bits(1),
                4 => -87.0,
                5 => -88.0,
                6 => f32::NEG_INFINITY,
                _ => 0.25,
            })
            .collect();
        for (k, p, t) in [
            (0, 1.0, 0.7),
            (v / 2 + 1, 0.51, 1.3),
            (v, 0.0, 1.0),
            (usize::MAX, 0.97, 0.125),
        ] {
            batch(&z, Params::new(t, k, p), &[0.0, 0.37, fixtures::U_MAX], &d)?;
        }
    }
    let z = vec![-0.0; 8];
    for j in 1..8 {
        let p = j as f32 / 8.0;
        for p in [
            f32::from_bits(p.to_bits() - 1),
            p,
            f32::from_bits(p.to_bits() + 1),
        ] {
            batch(
                &z,
                Params::new(1.0, 0, p as f64),
                &[0.0, 0.5, fixtures::U_MAX],
                &d,
            )?;
        }
        let u = j as f32 / 8.0;
        batch(
            &z,
            Params::new(1.0, 0, 1.0),
            &[
                f32::from_bits(u.to_bits() - 1),
                u,
                f32::from_bits(u.to_bits() + 1),
            ],
            &d,
        )?;
    }
    for v in [128_000usize, 248_320] {
        let z = vec![0.0; v];
        for k in [0, 20, 50, v / 2 + 1, v] {
            let got = batch(&z, Params::new(1.0, k, 0.5), &[0.0, fixtures::U_MAX], &d)?;
            let n = if k == 0 { v } else { k };
            let count = n.div_ceil(2);
            assert_eq!(got.status[3], count as u32);
            assert_eq!(got.ids, [0, (count - 1) as u32]);
        }
    }
    Ok(())
}

#[test]
fn invalid_rows_uniforms_and_masked_rounding_edge_have_no_false_success() -> Result<()> {
    let d = device()?;
    for u in [
        -0.25,
        1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7f800001),
        f32::from_bits(0xffc01234),
    ] {
        batch(&[0.0, 1.0], Params::new(1.0, 1, 0.0), &[u], &d)?;
    }
    batch(
        &[f32::NEG_INFINITY, 0.0],
        Params::new(1.0, 0, 1.0),
        &[-0.0, f32::from_bits(1)],
        &d,
    )?;
    // Refuse a bad score even when k=1/p=0 would otherwise discard it.
    for z in [
        vec![5.0, f32::from_bits(0x7f800001)],
        vec![5.0, f32::INFINITY],
        vec![f32::NEG_INFINITY; 2],
    ] {
        batch(&z, Params::new(1.0, 1, 0.0), &[0.5], &d)?;
    }
    let mut z = vec![f32::NEG_INFINITY; 1024];
    z[31] = -16.985;
    z[511] = 0.0;
    z[991] = -16.985;
    let got = batch(&z, Params::new(1.0, 0, 1.0), &[fixtures::U_MAX; 16], &d)?;
    assert!(got.ids.iter().all(|&i| [31, 511, 991].contains(&i)));
    assert!(got.ids.windows(2).all(|w| w[0] == w[1]));
    Ok(())
}

#[test]
fn ranks_offsets_row_permutations_and_replay_keep_same_row_identity() -> Result<()> {
    let d = device()?;
    let (v, rows) = (1025usize, 5usize);
    let params = Params::new(0.7, 513, 0.73);
    let uniforms = [0.0, 0.25, 0.5, 0.75, fixtures::U_MAX];
    let data: Vec<f32> = (0..rows)
        .flat_map(|r| {
            (0..v).map(move |i| {
                if (i + r) % 7 == 0 {
                    f32::NEG_INFINITY
                } else {
                    ((i * 17 + r * 3) % 29) as f32 / 8.0 - 2.0
                }
            })
        })
        .collect();
    let base = Tensor::from_vec(data.clone(), (rows, v), &d)?;
    let u = Tensor::from_vec(uniforms.to_vec(), rows, &d)?;
    let reference = read(encode(&base, params, &u)?)?;
    let shapes: [crate::Shape; 2] = [(rows, v).into(), (1, rows, v).into()];
    for shape in shapes {
        let view = base.reshape(shape)?;
        assert_eq!(read(encode(&view, params, &u)?)?, reference);
    }
    let mut padded = vec![f32::from_bits(0x7fc01234); v];
    padded.extend_from_slice(&data);
    padded.extend(vec![f32::INFINITY; v]);
    let padded = Tensor::from_vec(padded, (rows + 2, v), &d)?.narrow(0, 1, rows)?;
    let up = Tensor::from_vec(
        vec![
            f32::NAN,
            0.0,
            0.25,
            0.5,
            0.75,
            fixtures::U_MAX,
            f32::INFINITY,
        ],
        7,
        &d,
    )?
    .narrow(0, 1, rows)?;
    for _ in 0..3 {
        assert_eq!(read(encode(&padded, params, &up)?)?, reference);
    }
    let order = [4, 1, 3, 0, 2];
    let permuted: Vec<f32> = order
        .iter()
        .flat_map(|&r| data[r * v..(r + 1) * v].iter().copied())
        .collect();
    let permuted = Tensor::from_vec(permuted, (rows, v), &d)?;
    let permuted_u = Tensor::from_vec(
        order.iter().map(|&r| uniforms[r]).collect::<Vec<_>>(),
        rows,
        &d,
    )?;
    let got = read(encode(&permuted, params, &permuted_u)?)?;
    for (i, &r) in order.iter().enumerate() {
        assert_eq!(got.ids[i], reference.ids[r]);
        assert_eq!(
            &got.support[i * v..(i + 1) * v],
            &reference.support[r * v..(r + 1) * v]
        );
        assert_eq!(
            &got.status[i * 4..i * 4 + 4],
            &reference.status[r * 4..r * 4 + 4]
        );
        let one = base.narrow(0, r, 1)?.squeeze(0)?;
        let one_u = u.narrow(0, r, 1)?;
        let single = read(encode(&one, params, &one_u)?)?;
        assert_eq!(single.ids, [reference.ids[r]]);
        assert_row(
            &data[r * v..(r + 1) * v],
            params,
            uniforms[r],
            single.ids[0],
            &single.support,
            &single.status,
        );
    }
    Ok(())
}

#[test]
fn geometry_parameters_and_zero_rows_are_checked_before_dispatch() -> Result<()> {
    let d = device()?;
    let z = Tensor::from_vec(vec![0f32; 8], 8, &d)?;
    let u = Tensor::from_vec(vec![0.5f32; 2], 2, &d)?;
    let (zs, _) = z.storage_and_layout();
    let (us, _) = u.storage_and_layout();
    let (Storage::Metal(zs), Storage::Metal(us)) = (&*zs, &*us) else {
        unreachable!()
    };
    let p = Params::new(1.0, 0, 1.0);
    for l in [
        Layout::contiguous(()),
        Layout::contiguous((1, 0)),
        Layout::contiguous_with_offset((2, 4), 1),
        Layout::new((2, 4).into(), vec![1, 2], 0),
        Layout::new((usize::MAX, 2).into(), vec![2, 1], 0),
        Layout::contiguous_with_offset((2, 4), usize::MAX),
        Layout::new(
            (1, u32::MAX as usize + 1).into(),
            vec![u32::MAX as usize + 1, 1],
            0,
        ),
    ] {
        assert!(checked_encode(zs, &l, p, us, &Layout::contiguous(2)).is_err());
    }
    for ul in [
        Layout::contiguous(1),
        Layout::contiguous_with_offset(2, 1),
        Layout::new((2, 1).into(), vec![2, 2], 0),
        Layout::contiguous_with_offset(2, usize::MAX),
        Layout::new((usize::MAX, 2).into(), vec![2, 1], 0),
    ] {
        assert!(checked_encode(zs, &Layout::contiguous((2, 4)), p, us, &ul).is_err());
    }
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
        assert!(checked_encode(
            zs,
            &Layout::contiguous((2, 4)),
            Params::new(t, 0, p),
            us,
            &Layout::contiguous(2)
        )
        .is_err());
    }
    let zero = read(checked_encode(
        zs,
        &Layout::contiguous((0, 4)),
        p,
        us,
        &Layout::contiguous(0),
    )?)?;
    assert_eq!(
        zero,
        Observed {
            ids: vec![],
            support: vec![],
            status: vec![]
        }
    );
    // The logical count alone does not certify the physical view's extent.
    let words = zs.buffer().length() / 4 + 1;
    let overstated = MetalStorage::new(
        zs.buffer_handle().clone(),
        zs.device().clone(),
        words,
        DType::F32,
    );
    assert!(checked_encode(
        &overstated,
        &Layout::contiguous((1, words)),
        p,
        us,
        &Layout::contiguous(1)
    )
    .is_err());
    let words = us.buffer().length() / 4 + 1;
    let overstated_u = MetalStorage::new(
        us.buffer_handle().clone(),
        us.device().clone(),
        words,
        DType::F32,
    );
    assert!(checked_encode(
        zs,
        &Layout::contiguous((2, 4)),
        p,
        &overstated_u,
        &Layout::contiguous(words)
    )
    .is_err());
    let other_device = Device::new_metal(0)?;
    let other_u = Tensor::from_vec(vec![0.5f32; 2], 2, &other_device)?;
    assert!(encode(&z.reshape((2, 4))?, p, &other_u).is_err());
    assert!(checked_encode(
        zs,
        &Layout::contiguous((0, 4)),
        Params::new(0.0, 0, 1.0),
        us,
        &Layout::contiguous(0)
    )
    .is_err());
    let half = z.to_dtype(DType::F16)?;
    let half_u = u.to_dtype(DType::F16)?;
    assert!(encode(&half.reshape((2, 4))?, p, &u).is_err());
    assert!(encode(&z.reshape((2, 4))?, p, &half_u).is_err());
    Ok(())
}

// Frozen independent pins from tests/sampling_contract.rs::PINS. They predate
// this candidate; both the oracle and production must match their literal values.
#[test]
fn arithmetic_pins_distinguish_platform_exp_and_sequential_mass() -> Result<()> {
    let d = device()?;
    let pins: [(u64, usize, Params, Option<f32>, usize); 9] = [
        (
            7166411029125059793,
            3000,
            Params::new(2.0, 1181, 0.9696083068847656),
            None,
            1056,
        ),
        (
            14993398499489279759,
            3000,
            Params::new(1.0, 455, 0.5804826617240906),
            None,
            168,
        ),
        (
            10752749459413695495,
            3000,
            Params::new(0.5, 0, 1.0),
            Some(0.16891079),
            490,
        ),
        (
            8518066908502861862,
            3000,
            Params::new(0.5, 480, 1.0),
            Some(0.46938762),
            1505,
        ),
        (
            5445465867046481102,
            5000,
            Params::new(0.5, 0, 0.9999961853027344),
            None,
            2548,
        ),
        (
            9388319547254842595,
            5000,
            Params::new(2.0, 1896, 0.298232764005661),
            None,
            265,
        ),
        (
            11469413305603673856,
            5000,
            Params::new(1.0, 245, 1.0),
            Some(0.44293648),
            2153,
        ),
        (
            5445465867046481102,
            5000,
            Params::new(0.5, 0, 1.0),
            Some(0.2516549),
            1339,
        ),
        (
            3232577526175246933,
            5000,
            Params::new(1.0, 1306, 1.0),
            Some(0.009029534),
            32,
        ),
    ];
    for (mut state, v, params, u, want) in pins {
        let z: Vec<f32> = (0..v)
            .map(|_| {
                state = state.wrapping_add(0x9e3779b97f4a7c15);
                let mut x = state;
                x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
                x ^= x >> 31;
                (x >> 40) as f32 / (1u64 << 24) as f32 * 12.0 - 6.0
            })
            .collect();
        let got = batch(&z, params, &[u.unwrap_or(0.5)], &d)?;
        let value = if u.is_some() {
            got.ids[0]
        } else {
            got.status[3]
        };
        assert_eq!(value as usize, want, "independently pinned arithmetic");
    }
    println!("MULTINOMIAL_ARITHMETIC_PINS reached=9");
    Ok(())
}

#[test]
fn raw_abi_offsets_initialization_and_tail_canaries() -> Result<()> {
    let d = device()?;
    let (rows, v, guard) = (4usize, 37usize, 19usize);
    let mut data = vec![f32::NAN; 5];
    data.extend(vec![0.0; v]);
    data.extend(vec![f32::NEG_INFINITY; v]);
    data.extend(vec![0.0; v]);
    data[5 + 2 * v + v - 1] = f32::from_bits(0x7f800001);
    data.extend(vec![0.0; v]);
    data.extend(vec![f32::INFINITY; guard]);
    let input = Tensor::from_vec(data.clone(), data.len(), &d)?;
    let uniforms = Tensor::from_vec(
        vec![f32::NAN, 0.5, f32::NAN, 0.5, 1.0, f32::INFINITY],
        6,
        &d,
    )?;
    let make = |count| Tensor::from_vec(vec![0xdecafbad_u32; count + guard], count + guard, &d);
    let weights = make(rows * v)?;
    let support = make(rows * v)?;
    let ids = make(rows)?;
    let status = make(rows * 4)?;
    let words = crate::kernels::kernels::sampling_topk::sampling_work_words(rows, v).unwrap();
    let work = make(words)?;
    let scratch = make(rows * v)?;
    let selection = make(rows * 4)?;
    {
        let (input, _) = input.storage_and_layout();
        let (u, _) = uniforms.storage_and_layout();
        let (w, _) = weights.storage_and_layout();
        let (mask, _) = support.storage_and_layout();
        let (id, _) = ids.storage_and_layout();
        let (s, _) = status.storage_and_layout();
        let (work, _) = work.storage_and_layout();
        let (scratch, _) = scratch.storage_and_layout();
        let (ks, _) = selection.storage_and_layout();
        let (
            Storage::Metal(z),
            Storage::Metal(u),
            Storage::Metal(w),
            Storage::Metal(mask),
            Storage::Metal(id),
            Storage::Metal(s),
            Storage::Metal(work),
            Storage::Metal(scratch),
            Storage::Metal(ks),
        ) = (
            &*input, &*u, &*w, &*mask, &*id, &*s, &*work, &*scratch, &*ks,
        )
        else {
            unreachable!()
        };
        let m = z.device();
        let encoder = m.command_encoder()?;
        // SAFETY: sized disjoint live storage; the selection and the draw
        // read these exact rows; scratch is four-byte storage regardless of
        // its test-only U32 label.
        unsafe {
            crate::kernels::kernels::sampling_topk::call_sampling_select(
                m.metal_device(),
                &encoder,
                m.kernels(),
                (z.buffer(), 20),
                rows as u32,
                v as u32,
                2,
                1,
                work.buffer(),
                scratch.buffer(),
                ks.buffer(),
            )
            .and_then(|()| {
                crate::kernels::kernels::sampling_topk::call_sampling_multinomial(
                    m.metal_device(),
                    &encoder,
                    m.kernels(),
                    (z.buffer(), 20),
                    rows as u32,
                    v as u32,
                    1.0,
                    0.5,
                    1,
                    (u.buffer(), 4),
                    work.buffer(),
                    scratch.buffer(),
                    w.buffer(),
                    mask.buffer(),
                    id.buffer(),
                    s.buffer(),
                )
            })
        }
        .map_err(crate::Error::wrap)?;
    }
    let mask = support.to_vec1::<u32>()?;
    let ids = ids.to_vec1::<u32>()?;
    let status = status.to_vec1::<u32>()?;
    let weights = weights.to_vec1::<u32>()?;
    assert_eq!(&ids[..rows], &[0, u32::MAX, u32::MAX, u32::MAX]);
    assert_eq!(
        &status[..rows * 4],
        &[
            0,
            u32::MAX,
            37,
            1,
            3,
            u32::MAX,
            0,
            0,
            1,
            36,
            36,
            0,
            4,
            u32::MAX,
            37,
            0
        ]
    );
    assert_eq!(mask[0], 1);
    assert!(mask[1..rows * v].iter().all(|&x| x == 0));
    assert_eq!(&weights[..2], &[0x3f800000, 0x3f800000]);
    assert!(weights[2..rows * v].iter().all(|&x| x == 0));
    for (buf, n) in [
        (&mask, rows * v),
        (&ids, rows),
        (&status, rows * 4),
        (&weights, rows * v),
    ] {
        assert!(
            buf[n..].iter().all(|&x| x == 0xdecafbad),
            "tail canary overwritten"
        );
    }
    assert_eq!(bits(&input.to_vec1::<f32>()?), bits(&data));
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
    let u = Tensor::from_vec(vec![0.75f32], 1, &d)?;
    let id_slot = HostSlot::new(m, 1, DType::U32)?;
    let support_slot = HostSlot::new(m, 4, DType::U32)?;
    let status_slot = HostSlot::new(m, 4, DType::U32)?;
    id_slot.fill_u32(0xdecafbad, 1)?;
    support_slot.fill_u32(0xdecafbad, 4)?;
    status_slot.fill_u32(0xdecafbad, 4)?;
    let snapshot = || -> Result<_> {
        let enc = m.command_encoder()?;
        let enc = enc.as_ref();
        Ok((enc.command_buffer.clone(), Arc::clone(&enc.state)))
    };
    let before = readback_counts();
    let (zs, layout) = z.storage_and_layout();
    let (us, ul) = u.storage_and_layout();
    let (Storage::Metal(zs), Storage::Metal(us)) = (&*zs, &*us) else {
        unreachable!()
    };
    let produced = zs.affine(layout, -1.0, 1.0)?;
    let producer = snapshot()?;
    if force_cut {
        m.commands().flush().map_err(crate::Error::wrap)?;
        m.set_compute_per_buffer(1);
    }
    let result = checked_encode(&produced, layout, Params::new(1.0, 2, 0.5), us, ul)?;
    // With counter=1 the K→draw edge also crosses a command-buffer cut.
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
    assert_eq!(id_slot.read_unchecked::<u32>(1)?, [2]);
    assert_eq!(support_slot.read_unchecked::<u32>(4)?, [0, 0, 1, 0]);
    assert_eq!(status_slot.read_unchecked::<u32>(4)?, [0, u32::MAX, 4, 1]);
    assert_eq!(readback_counts().since(&before).slot, 3);
    println!("MULTINOMIAL_DEPENDENCY force_cut={force_cut} data_wait_completed=true id=2");
    Ok(())
}
#[test]
fn queued_producer_sampler_consumers_same_encoder() -> Result<()> {
    dependency(false)
}
#[test]
fn queued_producer_sampler_consumers_forced_cuts() -> Result<()> {
    dependency(true)
}

/// A legal subnormal uniform must not become zero before the rounded product.
#[test]
fn subnormal_uniform_preserves_rounded_product_contract() -> Result<()> {
    let d = device()?;
    let z: Vec<f32> = [-87.0].into_iter().chain([0.0; 36]).collect();
    let got = batch(
        &z,
        Params::new(1.0, 0, 1.0),
        &[f32::from_bits(0x007fffff)],
        &d,
    )?;
    assert_eq!(got.ids, [1]);
    Ok(())
}

/// The contract permits a nonzero subnormal reciprocal temperature.
#[test]
fn subnormal_inverse_temperature_preserves_rounded_product_contract() -> Result<()> {
    let d = device()?;
    let got = batch(&[-f32::MAX, 0.0], Params::new(1e39, 0, 1.0), &[0.45], &d)?;
    assert_eq!(got.ids, [1]);
    Ok(())
}

/// Literal binary32 products from exact rational distances to adjacent floats;
/// no sampler/reference implementation computes these expected values.
#[test]
fn bit_product_literal_rounding_boundaries() -> Result<()> {
    use crate::kernels::metal::ComputeCommandEncoder;
    let d = device()?;
    let m = d.as_metal_device()?;
    let cases: &[(u32, u32, u32)] = &[
        (0x007fffff, 0x42100000, 0x030fffff),
        (0x000ae398, 0xff7fffff, 0xbeae397f),
        (0x00000001, 0x3f000000, 0x00000000),
        (0x00000003, 0x3f000000, 0x00000002),
        (0x00000005, 0x3f000000, 0x00000002),
        (0x007fffff, 0x3f800001, 0x00800000),
        (0x007fffff, 0x3f800000, 0x007fffff),
        (0x00800000, 0x3f000000, 0x00400000),
        (0x00800001, 0x3f000000, 0x00400000),
        (0x00800003, 0x3f000000, 0x00400002),
        (0x00800000, 0x3f7fffff, 0x00800000),
        (0x00000001, 0x7f7fffff, 0x34ffffff),
        (0x007fffff, 0x007fffff, 0x00000000),
        (0x80000001, 0x3f000000, 0x80000000),
        (0x80000000, 0x3f800000, 0x80000000),
        (0x3fc00000, 0x40000000, 0x40400000),
        (0x007fffff, 0x40000001, 0x01000000),
        (0x00800000, 0x3f800000, 0x00800000),
        (0x80800000, 0x3f000000, 0x80400000),
        (0x3f800001, 0x3f800001, 0x3f800002),
        (0x7f800000, 0x00000001, 0x7f800000),
        (0xff800000, 0x00000001, 0xff800000),
        (0x00000000, 0x7f800000, 0x7fc00000),
        (0x7f800001, 0x3f800000, 0x7fc00000),
    ];
    let pairs: Vec<u32> = cases.iter().flat_map(|&(a, b, _)| [a, b]).collect();
    let input = Tensor::from_vec(pairs.clone(), pairs.len(), &d)?;
    let out = Tensor::from_vec(vec![0xdecafbadu32; cases.len()], cases.len(), &d)?;
    let source = format!(
        "{}\n{}",
        crate::kernels::source::RANDOM,
        r#"kernel void sampling_mul_bits_probe(
            device const uint *pairs [[buffer(0)]],
            device uint *out [[buffer(1)]],
            uint i [[thread_position_in_grid]]) {
            out[i] = sampling_mul_bits(pairs[2u * i], pairs[2u * i + 1u]);
        }"#
    );
    let options = crate::kernels::kernel::get_compile_options();
    let library = m
        .metal_device()
        .new_library_with_source(&source, Some(&options))
        .unwrap();
    let function = library
        .get_function("sampling_mul_bits_probe", None)
        .unwrap();
    let pipeline = m
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .unwrap()
        .with_name("sampling_mul_bits_probe");
    {
        let (input, _) = input.storage_and_layout();
        let (out, _) = out.storage_and_layout();
        let (Storage::Metal(input), Storage::Metal(out)) = (&*input, &*out) else {
            unreachable!()
        };
        let encoder = m.command_encoder()?;
        let enc: &ComputeCommandEncoder = encoder.as_ref();
        enc.set_compute_pipeline_state(&pipeline);
        // SAFETY: two input words and one output per thread, disjoint storage;
        // Tensor owners stay live until completion and the read below.
        unsafe {
            enc.set_input_buffer(0, Some(input.buffer()), 0);
            enc.set_output_buffer(1, Some(out.buffer()), 0);
        }
        enc.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: cases.len(),
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
    }
    m.wait_until_completed()?;
    let got = out.to_vec1::<u32>()?;
    for (i, &(a, b, want)) in cases.iter().enumerate() {
        assert_eq!(got[i], want, "{a:08x} * {b:08x}");
    }
    assert_eq!(input.to_vec1::<u32>()?, pairs);
    println!(
        "MULTINOMIAL_PRODUCT_LITERAL count={} native helper reached",
        cases.len()
    );
    Ok(())
}

/// These controls distinguish the bit-product candidate from clipping u or T,
/// and retain both the normal-input boundary and zero-reciprocal behavior.
#[test]
fn bit_product_sampler_boundary_controls() -> Result<()> {
    let d = device()?;
    let z: Vec<f32> = [-87.0].into_iter().chain([0.0; 36]).collect();
    let got = batch(
        &z,
        Params::new(1.0, 0, 1.0),
        &[
            0.0,
            -0.0,
            f32::from_bits(1),
            f32::from_bits(0x007fffff),
            f32::from_bits(0x00800000),
            fixtures::U_MAX,
        ],
        &d,
    )?;
    assert_eq!(got.ids, [0, 0, 0, 1, 1, 36]);
    for (t, inv) in [
        (8.507059173023462e37, 0x00800000u32),
        (1.7014118346046923e38, 0x00400000u32),
        (1e39, 0x000ae398u32),
        (f64::MAX, 0x00000000u32),
    ] {
        assert_eq!(oracle::inv_t(t).unwrap().to_bits(), inv);
        let got = batch(&[-f32::MAX, 0.0], Params::new(t, 0, 1.0), &[0.45, 0.75], &d)?;
        assert_eq!(got.ids, if inv == 0 { [0, 1] } else { [1, 1] });
    }
    // f64 p is active even if its conversion becomes zero or one in f32.
    for p in [f64::from_bits(1), f32::from_bits(1) as f64] {
        let got = batch(
            &[0.0; 4],
            Params::new(1.0, 0, p),
            &[0.0, fixtures::U_MAX],
            &d,
        )?;
        assert_eq!(got.support, [1, 0, 0, 0, 1, 0, 0, 0]);
        assert_eq!(got.ids, [0, 0]);
    }
    Ok(())
}

/// The private positive-T domain is wider than the public greedy resolver.
/// Product repair cannot recover score bits already lost by subtraction.
#[test]
fn subnormal_score_subtraction_remains_an_independent_obligation() -> Result<()> {
    let d = device()?;
    let t = 1.1754943508222875e-38f64;
    assert_eq!(oracle::inv_t(t).unwrap().to_bits(), 0x7e800000);
    for magnitude in [0x007fffffu32, 0x00800000] {
        let got = batch(
            &[-f32::from_bits(magnitude), 0.0],
            Params::new(t, 0, 1.0),
            &[0.4],
            &d,
        )?;
        assert_eq!(got.ids, [1]);
    }
    Ok(())
}

// Independent literal controls call the production helpers in the same Random
// text/options as the sampler. This observer reads integer result words.
fn probe_arithmetic_words(
    words: &[u32],
    threads: usize,
    output_words: usize,
    body: &str,
    d: &Device,
) -> Result<Vec<u32>> {
    use crate::kernels::metal::ComputeCommandEncoder;
    let m = d.as_metal_device()?;
    let input = Tensor::from_vec(words.to_vec(), words.len(), d)?;
    let out = Tensor::from_vec(vec![0xdecafbadu32; output_words], output_words, d)?;
    let source = format!(
        "{}\nkernel void sampling_arithmetic_probe(device const uint *input [[buffer(0)]], device uint *out [[buffer(1)]], uint i [[thread_position_in_grid]]) {{ {} }}",
        crate::kernels::source::RANDOM, body
    );
    let options = crate::kernels::kernel::get_compile_options();
    let library = m
        .metal_device()
        .new_library_with_source(&source, Some(&options))
        .unwrap();
    let function = library
        .get_function("sampling_arithmetic_probe", None)
        .unwrap();
    let pipeline = m
        .metal_device()
        .new_compute_pipeline_state_with_function(&function)
        .unwrap()
        .with_name("sampling_arithmetic_probe");
    {
        let (input, _) = input.storage_and_layout();
        let (out, _) = out.storage_and_layout();
        let (Storage::Metal(input), Storage::Metal(out)) = (&*input, &*out) else {
            unreachable!()
        };
        let encoder = m.command_encoder()?;
        let enc: &ComputeCommandEncoder = encoder.as_ref();
        enc.set_compute_pipeline_state(&pipeline);
        // SAFETY: the caller's fixed probe indexes one disjoint output span
        // per thread within these sized inputs/outputs, held through completion.
        unsafe {
            enc.set_input_buffer(0, Some(input.buffer()), 0);
            enc.set_output_buffer(1, Some(out.buffer()), 0);
        }
        enc.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
    }
    m.wait_until_completed()?;
    assert_eq!(input.to_vec1::<u32>()?, words);
    out.to_vec1::<u32>()
}

/// Literal RNE subtraction bits, independent of the helper and CPU oracle.
#[test]
fn bit_subtraction_literal_rounding_boundaries() -> Result<()> {
    let d = device()?;
    let cases: &[(u32, u32, u32)] = &[
        (0x00000000, 0x00000000, 0x00000000),
        (0x80000000, 0x00000000, 0x80000000),
        (0x00000000, 0x80000000, 0x00000000),
        (0x80000000, 0x80000000, 0x00000000),
        (0x00000001, 0x00000000, 0x00000001),
        (0x00000000, 0x00000001, 0x80000001),
        (0x80000001, 0x00000000, 0x80000001),
        (0x00000000, 0x80000001, 0x00000001),
        (0x007fffff, 0x00800000, 0x80000001),
        (0x00800000, 0x00800001, 0x80000001),
        (0x00800000, 0x00ffffff, 0x807fffff),
        (0x0bffffff, 0x0c000000, 0x80400000),
        (0x807fffff, 0x00000000, 0x807fffff),
        (0x80800001, 0x80800000, 0x80000001),
        (0x00800000, 0x807fffff, 0x00ffffff),
        (0x807fffff, 0x00800000, 0x80ffffff),
        (0x00ffffff, 0x80000001, 0x01000000),
        (0x01000000, 0x80000001, 0x01000000),
        (0x01000001, 0x80000001, 0x01000002),
        (0x01800000, 0x007fffff, 0x01400000),
        (0x01800001, 0x007fffff, 0x01400002),
        (0x02000000, 0x007fffff, 0x01e00000),
        (0x02000000, 0x807fffff, 0x02100000),
        (0x00000001, 0x3f800000, 0xbf800000),
        (0x3f800000, 0x3f800000, 0x00000000),
        (0xff7fffff, 0x7f7fffff, 0xff800000),
        (0xff7fffff, 0x73000000, 0xff800000),
        (0xff7fffff, 0x72ffffff, 0xff7fffff),
        (0xff7fffff, 0x72800000, 0xff7fffff),
        (0x7f7fffff, 0xf3000000, 0x7f800000),
        (0x7f7fffff, 0xf2ffffff, 0x7f7fffff),
        (0xff000000, 0x7f000000, 0xff800000),
        (0xff000000, 0x7effffff, 0xff800000),
        (0x0c000000, 0x0bffffff, 0x00400000),
        (0x0c800000, 0x007fffff, 0x0c7fffff),
        (0x0c800001, 0x007fffff, 0x0c800001),
        (0x0c800000, 0x807fffff, 0x0c800000),
        (0x0d000000, 0x007fffff, 0x0d000000),
        (0x007fffff, 0x0d000000, 0x8d000000),
        (0x807fffff, 0x0d000000, 0x8d000000),
    ];
    let words: Vec<u32> = cases.iter().flat_map(|&(a, b, _)| [a, b]).collect();
    let got = probe_arithmetic_words(
        &words,
        cases.len(),
        cases.len(),
        "out[i] = sampling_sub_bits(input[2u*i], input[2u*i+1u]);",
        &d,
    )?;
    for (i, &(a, b, want)) in cases.iter().enumerate() {
        assert_eq!(got[i], want, "{a:08x} - {b:08x}");
    }
    println!(
        "MULTINOMIAL_SUB_LITERAL count={} actual helper reached",
        cases.len()
    );
    Ok(())
}

/// Every edge is observed through delta -> product -> weight, including
/// rounded subtraction infinity times zero (NaN), not an ideal real softmax.
#[test]
fn bit_weight_path_literal_exceptional_boundaries() -> Result<()> {
    let d = device()?;
    let cases: &[(u32, u32, u32, u32, u32, u32)] = &[
        (
            0xff7fffff, 0x7f7fffff, 0x00000000, 0xff800000, 0x7fc00000, 0x00000000,
        ),
        (
            0xff7fffff, 0x00000000, 0x00000000, 0xff7fffff, 0x80000000, 0x3f800000,
        ),
        (
            0xff7fffff, 0x73000000, 0x00000000, 0xff800000, 0x7fc00000, 0x00000000,
        ),
        (
            0xff7fffff, 0x72ffffff, 0x00000000, 0xff7fffff, 0x80000000, 0x3f800000,
        ),
        (
            0xff7fffff, 0x7f7fffff, 0x00000001, 0xff800000, 0xff800000, 0x00000000,
        ),
        (
            0xff7fffff, 0x7f7fffff, 0x3f800000, 0xff800000, 0xff800000, 0x00000000,
        ),
        (
            0xff7fffff, 0x00000000, 0x40000000, 0xff7fffff, 0xff800000, 0x00000000,
        ),
        (
            0x807fffff, 0x00000000, 0x7e800000, 0x807fffff, 0xbf7ffffe, 0x3ebc5ab3,
        ),
        (
            0x00800000, 0x00ffffff, 0x7e800000, 0x807fffff, 0xbf7ffffe, 0x3ebc5ab3,
        ),
        (
            0x0bffffff, 0x0c000000, 0x7e800000, 0x80400000, 0xbf000000, 0x3f1b4598,
        ),
        (
            0x80800001, 0x80800000, 0x7e800000, 0x80000001, 0xb4000000, 0x3f7ffffe,
        ),
        (
            0x80000001, 0x00000000, 0x3f000000, 0x80000001, 0x80000000, 0x3f800000,
        ),
        (
            0x80000003, 0x00000000, 0x3f000000, 0x80000003, 0x80000002, 0x3f800000,
        ),
        (
            0x80000000, 0x00000000, 0x7f7fffff, 0x80000000, 0x80000000, 0x3f800000,
        ),
        (
            0x00000000, 0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000,
        ),
        (
            0xc2ae0000, 0x00000000, 0x3f800000, 0xc2ae0000, 0xc2ae0000, 0x00b33687,
        ),
        (
            0xc2ae0001, 0x00000000, 0x3f800000, 0xc2ae0001, 0xc2ae0001, 0x00000000,
        ),
    ];
    let words: Vec<u32> = cases
        .iter()
        .flat_map(|&(a, b, t, _, _, _)| [a, b, t])
        .collect();
    let got = probe_arithmetic_words(
        &words,
        cases.len(),
        3 * cases.len(),
        r#"
        uint delta = sampling_sub_bits(input[3u*i], input[3u*i+1u]);
        uint scaled = sampling_mul_bits(delta, input[3u*i+2u]);
        out[3u*i] = delta;
        out[3u*i+1u] = scaled;
        out[3u*i+2u] = as_type<uint>(sampling_weight_from_bits(scaled));
    "#,
        &d,
    )?;
    for (i, &(a, b, t, delta, scaled, weight)) in cases.iter().enumerate() {
        assert_eq!(
            &got[3 * i..3 * i + 3],
            &[delta, scaled, weight],
            "score={a:08x} max={b:08x} inv={t:08x}"
        );
    }
    let products: &[(u32, u32, u32)] = &[
        (0xff7fffff, 0x40000000, 0xff800000),
        (0xff7fffff, 0x3f800001, 0xff800000),
        (0xff7fffff, 0x3f800000, 0xff7fffff),
        (0x7f000000, 0x40000000, 0x7f800000),
        (0x7f000000, 0x3fffffff, 0x7f7fffff),
        (0x7f7fffff, 0x3f800001, 0x7f800000),
        (0x5f118e00, 0x5fe11fff, 0x7f7ffffe),
        (0x5f118e00, 0x5fe12000, 0x7f800000),
        (0x5f118e00, 0x5fe12001, 0x7f800000),
    ];
    let words: Vec<u32> = products.iter().flat_map(|&(a, b, _)| [a, b]).collect();
    let got = probe_arithmetic_words(
        &words,
        products.len(),
        products.len(),
        "out[i] = sampling_mul_bits(input[2u*i], input[2u*i+1u]);",
        &d,
    )?;
    for (i, &(a, b, want)) in products.iter().enumerate() {
        assert_eq!(got[i], want, "{a:08x} * {b:08x}");
    }
    println!(
        "MULTINOMIAL_WEIGHT_PATH_LITERAL count={} overflow_products={} actual helpers reached",
        cases.len(),
        products.len()
    );
    Ok(())
}

/// Full private pipeline, independent literal ids/support as well as oracle;
/// positive private T below the public greedy boundary is intentionally legal.
#[test]
fn bit_subtraction_sampler_boundary_controls() -> Result<()> {
    let d = device()?;
    let tiny_t = 1.1754943508222875e-38f64;
    for (a, b, u) in [
        (0x807fffffu32, 0x00000000, 0x3ecccccdu32),
        (0x00800000, 0x00ffffff, 0x3ecccccd),
        (0x0bffffff, 0x0c000000, 0x3ee66666),
        (0x80800001, 0x80800000, 0x3effffff),
    ] {
        let got = batch(
            &[f32::from_bits(a), f32::from_bits(b)],
            Params::new(tiny_t, 0, 1.0),
            &[f32::from_bits(u)],
            &d,
        )?;
        assert_eq!(got.ids, [1]);
        assert_eq!(got.support, [1, 1]);
    }
    assert_eq!(oracle::inv_t(7.1362384635298e44).unwrap().to_bits(), 1);
    for (maximum, t, id) in [
        (0x7f7fffffu32, f64::MAX, 1u32),
        (0x73000000, f64::MAX, 1), // exact RNE overflow midpoint
        (0x72ffffff, f64::MAX, 0), // below midpoint: finite delta * 0
        (0x00000000, f64::MAX, 0),
        (0x7f7fffff, 7.1362384635298e44, 1), // inv is min subnormal
        (0x00000000, 0.5, 1),                // finite delta, rounded product overflow
    ] {
        let z = [-f32::MAX, f32::from_bits(maximum)];
        let got = batch(&z, Params::new(t, 0, 1.0), &[0.0, fixtures::U_MAX], &d)?;
        assert_eq!(got.ids, [id, 1]);
        assert_eq!(got.support, [1, 1, 1, 1]);
        let got = batch(&z, Params::new(t, 0, 0.5), &[0.0, fixtures::U_MAX], &d)?;
        assert_eq!(got.ids, [1, 1]);
        assert_eq!(got.support, [0, 1, 0, 1]);
    }
    for z in [[-0.0, 0.0], [0.0, -0.0], [-0.0, -0.0], [0.0, 0.0]] {
        let got = batch(
            &z,
            Params::new(tiny_t, 0, 1.0),
            &[0.0, 0.5, fixtures::U_MAX],
            &d,
        )?;
        assert_eq!(got.ids, [0, 1, 1]);
        assert_eq!(got.support, [1, 1, 1, 1, 1, 1]);
        let got = batch(&z, Params::new(tiny_t, 1, 1.0), &[fixtures::U_MAX], &d)?;
        assert_eq!(got.ids, [0]);
        assert_eq!(got.support, [1, 0]);
    }
    println!("MULTINOMIAL_SUB_SAMPLER literal tiny/cancellation/zero/overflow paths reached");
    Ok(())
}

#[test]
fn checked_bridge_waits_for_its_copy_epoch_and_preserves_refusal_before_ids() -> Result<()> {
    use super::readback::SampleReadback;
    let d = device()?;
    let m = d.as_metal_device()?;
    let mut bridge = SampleReadback::new(m, 2)?;
    let z = Tensor::new(&[[0.0f32, 0.0], [f32::NEG_INFINITY, f32::NEG_INFINITY]], &d)?;
    let u = Tensor::new(&[0.75f32, 0.75], &d)?;
    bridge.enqueue(encode(&z, Params::new(1.0, 0, 1.0), &u)?)?;
    assert!(bridge.poll()?.is_none(), "copy epoch was not committed");
    let error = match bridge.read_committed() {
        Err(e) => e,
        _ => panic!("invalid later row exposed ids"),
    };
    assert!(error.to_string().contains("row 1: Empty"), "{error}");
    let good = Tensor::new(&[[0.0f32, 0.0], [0.0, 0.0]], &d)?;
    bridge.enqueue(encode(&good, Params::new(1.0, 0, 1.0), &u)?)?;
    assert!(bridge.poll()?.is_none());
    assert_eq!(
        bridge.read_committed()?.expect("completed snapshot").ids(),
        &[1, 1]
    );
    assert!(bridge.poll()?.is_none(), "result delivered twice");
    Ok(())
}

#[test]
fn checked_bridge_refuses_overwrite_and_capacity_without_adding_waits() -> Result<()> {
    use super::readback::SampleReadback;
    let d = device()?;
    let m = d.as_metal_device()?;
    let z = Tensor::new(&[[0.0f32, 0.0]], &d)?;
    let u = Tensor::new(&[0.75f32], &d)?;
    let mut bridge = SampleReadback::new(m, 1)?;
    let sample = || encode(&z, Params::new(1.0, 0, 1.0), &u);
    bridge.enqueue(sample()?)?;
    assert!(bridge.enqueue(sample()?).is_err());
    assert!(bridge.poll()?.is_none());
    assert_eq!(bridge.read_committed()?.expect("completed").ids(), &[1]);
    assert!(
        bridge.read_committed()?.is_none(),
        "blocking result delivered twice"
    );
    let mut too_small = SampleReadback::new(m, 0)?;
    assert!(too_small.enqueue(sample()?).is_err());
    assert!(too_small.poll()?.is_none());
    m.wait_marker(&m.flush_marker()?)?;
    Ok(())
}

/// Test-only hold. Drop releases the callback before a failed assertion can
/// unwind the device; the callback also has a deadline if the test aborts.
struct SnapshotCallbackHold(std::sync::Arc<(parking_lot::Mutex<bool>, parking_lot::Condvar)>);

impl Drop for SnapshotCallbackHold {
    fn drop(&mut self) {
        let (lock, wake) = &*self.0;
        *lock.lock() = true;
        wake.notify_all();
    }
}

#[test]
fn checked_bridge_own_data_ready_does_not_require_callback_clock() -> Result<()> {
    use super::readback::SampleReadback;
    use parking_lot::{Condvar, Mutex};
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    };
    use std::time::Duration;

    let d = device()?;
    let m = d.as_metal_device()?;
    eprintln!(
        "SNAPSHOT_SETUP actual_api={}",
        m.commands().actual_command_api()
    );
    m.set_compute_per_buffer(usize::MAX);
    let z = Tensor::new(&[[0.0f32, 0.0]], &d)?;
    let u = Tensor::new(&[0.75f32], &d)?;
    let mut bridge = SampleReadback::new(m, 1)?;
    let sample = encode(&z, Params::new(1.0, 0, 1.0), &u)?;
    // A producer cut must not make the snapshot use the producer's identity:
    // the copy itself is encoded into the next recording.
    m.flush_marker()?;
    bridge.enqueue(sample)?;
    assert!(bridge.poll()?.is_none(), "copy not submitted yet");
    let epoch = m.gpu_clock().current_epoch();
    let released = Arc::new((Mutex::new(false), Condvar::new()));
    let hold = SnapshotCallbackHold(released.clone());
    let expired = Arc::new(AtomicBool::new(false));
    let callback_expired = expired.clone();
    let (entered_tx, entered_rx) = mpsc::channel();
    {
        let encoder = m.command_encoder()?;
        encoder
            .as_ref()
            .command_buffer
            .on_completion(move || {
                let _ = entered_tx.send(());
                let (lock, wake) = &*released;
                let mut guard = lock.lock();
                let timeout =
                    wake.wait_while_for(&mut guard, |released| !*released, Duration::from_secs(10));
                callback_expired.store(timeout.timed_out() && !*guard, Ordering::Release);
            })
            .map_err(|e| crate::Error::Msg(format!("snapshot callback registration: {e:?}")))?;
    }
    let marker = m.flush_marker()?;
    entered_rx
        .recv_timeout(Duration::from_secs(5))
        .map_err(|e| crate::Error::Msg(format!("snapshot callback entry: {e}")))?;
    // Entry proves native feedback is available. The callback is still held,
    // so the ordered hook has not advanced this recording's clock.
    m.wait_marker(&marker)?;
    assert!(
        !m.gpu_clock().is_complete(epoch),
        "callback clock escaped hold"
    );
    assert!(
        bridge.poll()?.is_none(),
        "async poll must remain conservative"
    );
    assert_eq!(
        bridge.read_committed()?.expect("own-data snapshot").ids(),
        &[1]
    );
    assert!(
        !m.gpu_clock().is_complete(epoch),
        "read joined callback completion"
    );
    assert!(
        !expired.load(Ordering::Acquire),
        "callback deadline escaped hold"
    );
    assert!(bridge.poll()?.is_none(), "snapshot delivered twice");
    assert!(
        bridge.read_committed()?.is_none(),
        "blocking snapshot delivered twice"
    );
    drop(hold);
    // Explicit cleanup joins the callback after releasing it, outside the
    // property being tested. No worker or held callback survives the fixture.
    m.wait_until_completed()?;
    assert!(!expired.load(Ordering::Acquire));
    eprintln!("SNAPSHOT_DATA_READY callback_clock_pending=true own_data_read=true ids=[1] exactly_once=true");
    Ok(())
}

struct SnapshotRelease<F: Fn()>(F);

impl<F: Fn()> Drop for SnapshotRelease<F> {
    fn drop(&mut self) {
        (self.0)();
    }
}

#[test]
fn checked_bridge_exact_recording_ignores_foreign_completion_and_later_tail() -> Result<()> {
    use super::readback::SampleReadback;
    use objc2_metal::{MTLDevice, MTLSharedEvent};
    use std::sync::mpsc;
    use std::time::Duration;

    let d = device()?;
    let m = d.as_metal_device()?;
    eprintln!(
        "SNAPSHOT_SETUP actual_api={}",
        m.commands().actual_command_api()
    );
    m.set_compute_per_buffer(usize::MAX);
    let own_gate = m
        .metal_device()
        .as_ref()
        .newSharedEvent()
        .expect("own gate");
    let tail_gate = m
        .metal_device()
        .as_ref()
        .newSharedEvent()
        .expect("tail gate");
    // Release on every refusal/unwind, before the worker is joined and before
    // any buffer owner is dropped. No fixture can strand a native event wait.
    let release = SnapshotRelease(|| {
        own_gate.setSignaledValue(1);
        tail_gate.setSignaledValue(1);
    });
    let z = Tensor::new(&[[0.0f32, 0.0]], &d)?;
    let u = Tensor::new(&[0.75f32], &d)?;
    let mut bridge = SampleReadback::new(m, 1)?;
    let sample = encode(&z, Params::new(1.0, 0, 1.0), &u)?;
    m.wait_marker(&m.flush_marker()?)?;
    m.commands()
        .encode_wait_for_event(&own_gate, 1)
        .map_err(crate::MetalError::from)?;
    bridge.enqueue(sample)?;
    let own = m.flush_marker()?;
    m.commands()
        .encode_wait_for_event(&tail_gate, 1)
        .map_err(crate::MetalError::from)?;
    let tail_value = z.affine(1.0, 0.0)?;
    let tail = m.flush_marker()?;

    // A genuinely separate queue completes a later recording. Neither its
    // completion nor the shared physical GPU authenticates our pending copy.
    let other = Device::new_metal(0)?;
    let other_m = other.as_metal_device()?;
    let foreign_value = Tensor::new(&[19.0f32], &other)?.affine(2.0, 0.0)?;
    let foreign = other_m.flush_marker()?;
    other_m.wait_marker(&foreign)?;
    assert!(other_m.marker_complete(&foreign));
    assert!(!m.marker_complete(&own), "snapshot gate did not hold");
    assert!(!m.marker_complete(&tail), "tail gate did not hold");
    assert!(
        bridge.poll()?.is_none(),
        "foreign completion exposed snapshot bytes"
    );

    std::thread::scope(|scope| -> Result<()> {
        // This guard is inside the scope so it releases before scope's implicit
        // join on error/panic. The outer guard also covers setup refusals.
        let inner_release = SnapshotRelease(|| {
            own_gate.setSignaledValue(1);
            tail_gate.setSignaledValue(1);
        });
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let worker = scope.spawn(move || {
            let _ = started_tx.send(());
            let result = bridge.read_committed().and_then(|sample| {
                sample
                    .map(|s| s.ids().to_vec())
                    .ok_or_else(|| crate::Error::Msg("missing own snapshot".into()))
            });
            let _ = done_tx.send(result);
        });
        started_rx
            .recv_timeout(Duration::from_secs(2))
            .map_err(|e| crate::Error::Msg(format!("snapshot worker start: {e}")))?;
        assert!(
            matches!(
                done_rx.recv_timeout(Duration::from_millis(200)),
                Err(mpsc::RecvTimeoutError::Timeout)
            ),
            "read escaped the held copy recording"
        );
        assert!(!m.marker_complete(&own));
        own_gate.setSignaledValue(1);
        let ids = done_rx
            .recv_timeout(Duration::from_secs(5))
            .map_err(|e| crate::Error::Msg(format!("own read joined unrelated tail: {e}")))??;
        assert_eq!(ids, [1]);
        assert!(
            !m.marker_complete(&tail),
            "tail gate did not remain pending"
        );
        drop(inner_release);
        worker.join().expect("snapshot worker panicked");
        Ok(())
    })?;
    drop(release);
    m.wait_until_completed()?;
    other_m.wait_until_completed()?;
    drop((tail_value, foreign_value));
    eprintln!("SNAPSHOT_EXACT_RECORDING foreign_complete=true own_pending_refused=true own_ids=[1] later_tail_pending_at_return=true");
    Ok(())
}

const CHI2_999: [f64; 21] = [
    0.0, 10.828, 13.816, 16.266, 18.467, 20.515, 22.458, 24.322, 26.124, 27.877, 29.588, 31.264,
    32.909, 34.528, 36.123, 37.697, 39.252, 40.790, 42.312, 43.820, 45.315,
];

/// The reference's distribution: its f32 weights over its kept set,
/// normalised in f64. Zero off the kept set.
pub(super) fn reference_distribution(z: &[f32], params: Params) -> Vec<f64> {
    let kept = oracle::CONTRACT.nucleus(z, params).unwrap();
    let inv = oracle::inv_t(params.temperature).unwrap();
    let w = oracle::CONTRACT.weights(z, &kept, z[kept[0]], inv);
    let total: f64 = w.iter().map(|&x| x as f64).sum();
    w.iter().map(|&x| x as f64 / total).collect()
}

/// (χ², degrees of freedom, draws where the distribution is zero).
pub(super) fn chi_square(draws: &[u32], want: &[f64]) -> (f64, usize, usize) {
    let mut counts = vec![0usize; want.len()];
    for &id in draws {
        counts[id as usize] += 1;
    }
    let n = draws.len() as f64;
    let support: Vec<usize> = (0..want.len()).filter(|&i| want[i] > 0.0).collect();
    let outside = draws.len() - support.iter().map(|&i| counts[i]).sum::<usize>();
    let chi2 = support
        .iter()
        .map(|&i| (counts[i] as f64 - n * want[i]).powi(2) / (n * want[i]))
        .sum();
    (chi2, support.len() - 1, outside)
}

pub(super) fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The host χ² test's two rows and five cells (`sampling_host_contract.rs`).
pub(super) fn chi_square_cells() -> Vec<(&'static str, Vec<f32>, Params)> {
    let peaked: Vec<f32> = (0..64).map(|i| 2.0 - 0.25 * i as f32).collect();
    let mut s = 0xc415u64;
    let spread: Vec<f32> = (0..48)
        .map(|_| (splitmix(&mut s) >> 40) as f32 / (1u64 << 24) as f32 * 12.0 - 6.0)
        .collect();
    vec![
        ("top-k", peaked.clone(), Params::new(0.7, 6, 1.0)),
        ("top-k", peaked, Params::new(1.3, 6, 1.0)),
        ("top-p", spread.clone(), Params::new(0.7, 0, 0.8)),
        ("top-p", spread.clone(), Params::new(1.3, 0, 0.8)),
        ("top-k-top-p", spread, Params::new(1.3, 6, 0.9)),
    ]
}

/// The device draws over 16 000 uniforms have the reference's distribution
/// (dispatch §11.2j's χ²): top-k, top-p and both, at two temperatures. Each
/// draw also equals the reference's for its uniform (`batch` asserts every
/// row). Control: the same draws against the distribution with the
/// temperature dropped fail.
#[test]
fn device_draws_have_the_reference_distribution() -> Result<()> {
    let d = device()?;
    let (n, rows) = (16_000usize, 2_000usize);
    for (what, z, params) in chi_square_cells() {
        let want = reference_distribution(&z, params);
        let mut s = 77u64;
        let uniforms: Vec<f32> = (0..n)
            .map(|_| (splitmix(&mut s) >> 40) as f32 / (1u64 << 24) as f32)
            .collect();
        let mut draws = Vec::with_capacity(n);
        for chunk in uniforms.chunks(rows) {
            draws.extend(batch(&z, params, chunk, &d)?.ids);
        }
        let (chi2, df, outside) = chi_square(&draws, &want);
        assert!(
            (2..CHI2_999.len()).contains(&df),
            "{what}: support {}",
            df + 1
        );
        println!(
            "CHI2 device multinomial {what} T={} k={} p={} chi2={chi2:.2} df={df} critical={} outside={outside}",
            params.temperature, params.k, params.p, CHI2_999[df]
        );
        assert_eq!(outside, 0, "{what} {params:?}: drew outside the kept set");
        assert!(chi2 < CHI2_999[df], "{what} {params:?}: chi2 {chi2}");
        let cold = reference_distribution(&z, Params::new(1.0, params.k, params.p));
        let (chi2, _, _) = chi_square(&draws, &cold);
        println!("CHI2-CONTROL device multinomial {what} T dropped: chi2={chi2:.2}");
        assert!(chi2 > CHI2_999[df], "{what}: T = 1 passed at {chi2}");
    }
    Ok(())
}

/// χ² of `draws` against `want` with the cells expected fewer than five
/// times pooled into one, so a real row's long tail of rare ids does not void
/// the statistic: (χ², degrees of freedom, draws where `want` is zero).
pub(super) fn chi_square_pooled(draws: &[u32], want: &[f64]) -> (f64, usize, usize) {
    let mut counts = vec![0usize; want.len()];
    for &id in draws {
        counts[id as usize] += 1;
    }
    let n = draws.len() as f64;
    let outside = (0..want.len())
        .filter(|&i| want[i] == 0.0)
        .map(|i| counts[i])
        .sum();
    let (mut cells, mut pool) = (Vec::new(), (0.0f64, 0usize));
    for i in (0..want.len()).filter(|&i| want[i] > 0.0) {
        if n * want[i] >= 5.0 {
            cells.push((n * want[i], counts[i]));
        } else {
            pool = (pool.0 + n * want[i], pool.1 + counts[i]);
        }
    }
    if pool.0 > 0.0 {
        match cells.iter_mut().min_by(|a, b| a.0.total_cmp(&b.0)) {
            Some(least) if pool.0 < 5.0 => *least = (least.0 + pool.0, least.1 + pool.1),
            _ => cells.push(pool),
        }
    }
    let chi2 = cells.iter().map(|&(e, o)| (o as f64 - e).powi(2) / e).sum();
    (chi2, cells.len().saturating_sub(1), outside)
}

/// The χ² distribution's 0.999 quantile at `df`: [`CHI2_999`] to 20 degrees
/// of freedom, the Wilson–Hilferty approximation beyond (within 0.3 % of
/// the table at 20).
pub(super) fn chi2_999(df: usize) -> f64 {
    if df < CHI2_999.len() {
        return CHI2_999[df];
    }
    let (k, z) = (df as f64, 3.090_232_306_167_813f64);
    let a = 2.0 / (9.0 * k);
    k * (1.0 - a + z * a.sqrt()).powi(3)
}

/// The last `max` rows of an `LLOOM_LFM2_LOGITS_DUMP` file (`models::lfm2`'s
/// record: `pos, rows, vocab` as u32, then `rows × vocab` f32,
/// little-endian), each with its position: a run's decode steps, after its
/// warm-up and prefill forwards.
pub(super) fn dumped_rows(path: &str, max: usize) -> Vec<(u32, Vec<f32>)> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let (mut at, mut out) = (0usize, Vec::new());
    while at + 12 <= bytes.len() {
        let (pos, rows, vocab) = (word(at), word(at + 4) as usize, word(at + 8) as usize);
        at += 12;
        for r in 0..rows {
            let row = (0..vocab)
                .map(|i| {
                    f32::from_le_bytes(bytes[at + 4 * (r * vocab + i)..][..4].try_into().unwrap())
                })
                .collect();
            out.push((pos, row));
        }
        at += 4 * rows * vocab;
    }
    out.split_off(out.len().saturating_sub(max))
}

/// The sampling the decision-109 gates run, as (name, params): the
/// checkpoint's own, top-p, top-k at a temperature, the temperature alone.
pub(super) fn served_cells() -> [(&'static str, Params); 4] {
    [
        ("checkpoint", Params::new(0.1, 50, 1.0)),
        ("top-p", Params::new(0.8, 0, 0.9)),
        ("top-k", Params::new(0.8, 40, 1.0)),
        ("temperature", Params::new(0.8, 0, 1.0)),
    ]
}

/// The χ² above at logits rows a real checkpoint produced, read from
/// `LLOOM_CHI2_LOGITS` (an `LLOOM_LFM2_LOGITS_DUMP` file): the family's own
/// vocabulary and logit shape through the staged kernels, at the sampling
/// its gates run. 16 000 draws a cell, 100 rows a dispatch; one draw in 32
/// is also checked equal to the reference's for its uniform. A cell whose
/// pooled support is one cell is reported and not tested. Control: the same
/// draws against the distribution with the temperature dropped fail.
/// Ignored: it needs the dump, which needs a real checkpoint.
#[test]
#[ignore]
fn device_draws_have_the_reference_distribution_at_dumped_rows() -> Result<()> {
    let Ok(path) = std::env::var("LLOOM_CHI2_LOGITS") else {
        panic!("LLOOM_CHI2_LOGITS names no logits dump");
    };
    let d = device()?;
    let (n, rows) = (16_000usize, 100usize);
    let (mut tested, mut controls) = (0usize, 0usize);
    for (pos, z) in dumped_rows(&path, 3) {
        for (what, params) in served_cells() {
            let want = reference_distribution(&z, params);
            let mut s = 77u64 ^ pos as u64;
            let uniforms: Vec<f32> = (0..n)
                .map(|_| (splitmix(&mut s) >> 40) as f32 / (1u64 << 24) as f32)
                .collect();
            let mut draws = Vec::with_capacity(n);
            for chunk in uniforms.chunks(rows) {
                let input = Tensor::from_vec(z.repeat(chunk.len()), (chunk.len(), z.len()), &d)?;
                let u = Tensor::from_vec(chunk.to_vec(), chunk.len(), &d)?;
                let got = read(encode(&input, params, &u)?)?;
                for (row, &id) in got.ids.iter().enumerate().step_by(32) {
                    assert_eq!(
                        Ok(id),
                        oracle::draw(&z, params, chunk[row]),
                        "{what} pos {pos} u {}",
                        chunk[row]
                    );
                }
                draws.extend(got.ids);
            }
            let (chi2, df, outside) = chi_square_pooled(&draws, &want);
            let support = want.iter().filter(|&&w| w > 0.0).count();
            println!(
                "CHI2 dumped pos={pos} vocab={} multinomial {what} T={} k={} p={} support={support} chi2={chi2:.2} df={df} critical={:.3} outside={outside}",
                z.len(), params.temperature, params.k, params.p, chi2_999(df)
            );
            assert_eq!(outside, 0, "{what} pos {pos}: drew outside the kept set");
            if df == 0 {
                continue;
            }
            tested += 1;
            assert!(chi2 < chi2_999(df), "{what} pos {pos}: chi2 {chi2}");
            let cold = reference_distribution(&z, Params::new(1.0, params.k, params.p));
            let (chi2, cold_df, _) = chi_square_pooled(&draws, &cold);
            let fires = chi2 > chi2_999(cold_df);
            println!(
                "CHI2-CONTROL dumped pos={pos} multinomial {what} T dropped: chi2={chi2:.2} df={cold_df} critical={:.3} fires={fires}",
                chi2_999(cold_df)
            );
            controls += fires as usize;
        }
    }
    println!("CHI2-DUMPED tested={tested} controls_fired={controls}");
    assert!(tested > 0, "no cell carried more than one pooled cell");
    assert_eq!(controls, tested, "a control did not fire");
    Ok(())
}

/// The microbench's rows at the served width with top-k AND top-p: the
/// shape the fixtures' wide rows do not carry (they are k-only or p-only).
/// Every row's kept set, draw and status against the reference, five times
/// in one process (a race shows as a run that differs).
#[test]
fn wide_rows_with_top_k_and_top_p_match_the_reference_repeatedly() -> Result<()> {
    let d = device()?;
    for (vocab, k, p, t) in [
        (128_000usize, 20usize, 0.95f64, 1.0f64),
        (248_320, 20, 0.95, 1.0),
        (128_000, 50, 1.0, 0.1),
    ] {
        let z = super::timing_tests::logits(vocab, 4);
        let uniforms = [0.37f32, 0.5, 0.63, 0.76];
        for run in 0..5 {
            let input = Tensor::from_vec(z.clone(), (4, vocab), &d)?;
            let u = Tensor::from_vec(uniforms.to_vec(), 4, &d)?;
            let got = read(encode(&input, Params::new(t, k, p), &u)?)?;
            for row in 0..4 {
                let zr = &z[row * vocab..(row + 1) * vocab];
                let want = oracle::draw(zr, Params::new(t, k, p), uniforms[row]).unwrap();
                let kept = oracle::kept_ids(zr, Params::new(t, k, p)).unwrap();
                assert_eq!(
                    (got.ids[row], &got.status[row * 4..row * 4 + 4]),
                    (want, &[0, u32::MAX, vocab as u32, kept.len() as u32][..]),
                    "vocab {vocab} k {k} p {p} run {run} row {row}"
                );
            }
            println!(
                "WIDE_KP vocab={vocab} k={k} p={p} run={run} ids={:?} counts={:?}",
                got.ids,
                (0..4).map(|r| got.status[r * 4 + 3]).collect::<Vec<_>>()
            );
        }
    }
    Ok(())
}

/// Diagnostic for the shape above: every stage's scratch words after a
/// full draw, per row, five times. Ignored; `--no-capture` prints it.
#[test]
#[ignore]
fn diag_wide_rows_with_top_k_and_top_p() -> Result<()> {
    use crate::kernels::kernels::sampling_topk::{sampling_tiles, sampling_work_words};
    let d = device()?;
    let m = d.as_metal_device()?;
    let (vocab, rows, k, p, t) = (128_000usize, 4usize, 20usize, 0.95f64, 1.0f64);
    let z = super::timing_tests::logits(vocab, rows);
    let input = Tensor::from_vec(z.clone(), (rows, vocab), &d)?;
    let u = Tensor::from_vec(vec![0.37f32, 0.5, 0.63, 0.76], rows, &d)?;
    let (zs, layout) = input.storage_and_layout();
    let (us, ul) = u.storage_and_layout();
    let (Storage::Metal(zs), Storage::Metal(us)) = (&*zs, &*us) else {
        unreachable!()
    };
    let words = sampling_work_words(rows, vocab).unwrap();
    let per_row = words / rows;
    let tiles = sampling_tiles(vocab);
    let alloc = |count: usize, dtype, label| -> Result<MetalStorage> {
        let b = m
            .new_buffer_builder()
            .with_size_for(count, dtype)
            .with_label(label)
            .build()?;
        MetalStorage::new_allocated(b, m.clone(), count, dtype)
    };
    for run in 0..5 {
        let work = alloc(words, DType::U32, "diag_work")?;
        let scratch = alloc(rows * vocab, DType::U32, "diag_scratch")?;
        let selection = alloc(rows * 4, DType::U32, "diag_selection")?;
        let weights = alloc(rows * vocab, DType::F32, "diag_weights")?;
        let support = alloc(rows * vocab, DType::U32, "diag_support")?;
        let ids = alloc(rows, DType::U32, "diag_ids")?;
        let status = alloc(rows * 4, DType::U32, "diag_status")?;
        {
            let encoder = m.command_encoder()?;
            // SAFETY: sized, disjoint, live storage owned by this test.
            unsafe {
                crate::kernels::kernels::sampling_topk::call_sampling_select(
                    m.metal_device(),
                    &encoder,
                    m.kernels(),
                    (zs.buffer(), 0),
                    rows as u32,
                    vocab as u32,
                    k as u32,
                    u32::from(p != 1.0),
                    work.buffer(),
                    scratch.buffer(),
                    selection.buffer(),
                )
            }
            .map_err(crate::Error::wrap)?;
        }
        m.wait_until_completed()?;
        let w_after_select = work.to_cpu::<u32>()?;
        {
            let encoder = m.command_encoder()?;
            // SAFETY: as above; the selection is these rows'.
            unsafe {
                crate::kernels::kernels::sampling_topk::call_sampling_multinomial(
                    m.metal_device(),
                    &encoder,
                    m.kernels(),
                    (zs.buffer(), 0),
                    rows as u32,
                    vocab as u32,
                    (1.0 / t) as f32,
                    p as f32,
                    u32::from(p != 1.0),
                    (us.buffer(), 0),
                    work.buffer(),
                    scratch.buffer(),
                    weights.buffer(),
                    support.buffer(),
                    ids.buffer(),
                    status.buffer(),
                )
            }
            .map_err(crate::Error::wrap)?;
        }
        m.wait_until_completed()?;
        // Draws repeated from the one selection: the appended compact set
        // must not accumulate across them.
        for redraw in 0..4 {
            let encoder = m.command_encoder()?;
            // SAFETY: as above.
            unsafe {
                crate::kernels::kernels::sampling_topk::call_sampling_multinomial(
                    m.metal_device(),
                    &encoder,
                    m.kernels(),
                    (zs.buffer(), 0),
                    rows as u32,
                    vocab as u32,
                    (1.0 / t) as f32,
                    p as f32,
                    u32::from(p != 1.0),
                    (us.buffer(), 0),
                    work.buffer(),
                    scratch.buffer(),
                    weights.buffer(),
                    support.buffer(),
                    ids.buffer(),
                    status.buffer(),
                )
            }
            .map_err(crate::Error::wrap)?;
            drop(encoder);
            m.wait_until_completed()?;
            let st = status.to_cpu::<u32>()?;
            let wa = work.to_cpu::<u32>()?;
            println!(
                "DIAG-REDRAW run={run} redraw={redraw} counts={:?} compact_after={:?} fallback={:?}",
                (0..rows).map(|r| st[r * 4 + 3]).collect::<Vec<_>>(),
                (0..rows).map(|r| wa[r * per_row + 16]).collect::<Vec<_>>(),
                (0..rows).map(|r| wa[r * per_row + 22]).collect::<Vec<_>>()
            );
        }
        let w_after = work.to_cpu::<u32>()?;
        let wts = weights.to_cpu::<f32>()?;
        let mem = scratch.to_cpu::<u32>()?;
        let st = status.to_cpu::<u32>()?;
        let idv = ids.to_cpu::<u32>()?;
        let _ = (layout, ul);
        for row in 0..rows {
            let ws = &w_after_select[row * per_row..row * per_row + 64];
            let wa = &w_after[row * per_row..row * per_row + 64];
            let ghist = &w_after[row * per_row + 64 + 4 * tiles + 2048 * tiles
                ..row * per_row + 64 + 4 * tiles + 2048 * tiles + 2048];
            let lanes: Vec<f32> = w_after[row * per_row + 64 + 4 * tiles + 2048 * tiles + 2048..]
                [..1024]
                .iter()
                .map(|&b| f32::from_bits(b))
                .collect();
            let top_bins: Vec<(usize, u32)> = ghist
                .iter()
                .enumerate()
                .filter(|(b, &n)| n != 0 && *b >= 1540)
                .map(|(b, &n)| (b, n))
                .collect();
            let hist_top: Vec<u32> = (0..tiles)
                .map(|tl| w_after[row * per_row + 64 + 4 * tiles + tl * 2048 + 1546])
                .collect();
            let wrow = &wts[row * vocab..(row + 1) * vocab];
            let nonzero = wrow.iter().filter(|&&x| x != 0.0).count();
            let wsum: f64 = wrow.iter().map(|&x| x as f64).sum();
            let lane_sum: f64 = lanes.iter().map(|&x| x as f64).sum();
            let n_compact = wa[16] as usize;
            let members: Vec<u32> = mem[row * vocab..row * vocab + n_compact.min(24)].to_vec();
            println!(
                "DIAG run={run} row={row} select: code={} count={} keep={} rsel={} ksel={} bin={} nabove={} cand={} cutk={:#x}:{:#x} cuts={:#x}:{:#x} compact_on={} | after: compact={} over={} fallback={} cutp={:#x}:{:#x} ncount={} total={} nmass={} code={} | weights nonzero={} sum={wsum:.4} lanes sum={lane_sum:.4} | ghist top bins={top_bins:?} hist[1546] per tile sum={} | members={members:?} | id={} status={:?}",
                ws[0], ws[2], ws[3], ws[7], ws[8], ws[9], ws[10], ws[11], ws[13], ws[12], ws[15], ws[14], ws[17],
                wa[16], wa[25], wa[22], wa[20], wa[19], wa[21], f32::from_bits(wa[23]), f32::from_bits(wa[24]), wa[0],
                nonzero, hist_top.iter().sum::<u32>(), idv[row], &st[row * 4..row * 4 + 4]
            );
        }
    }
    Ok(())
}

/// The draw stage repeated from one selection, as the microbench times it:
/// every repeat must read the selection's compact set, not the previous
/// draw's appends on top of it. Production never repeats a draw (each
/// `sampling_multinomial` / `PreparedSampler::draw` encodes the selection
/// immediately before its draw, and the selection's scan stage resets the
/// counters), so this pins the draw's own reset for the bench and for any
/// caller that reuses a selection.
#[test]
fn a_draw_repeated_from_one_selection_starts_clean() -> Result<()> {
    use crate::kernels::kernels::sampling_topk::{
        call_sampling_multinomial, call_sampling_select, sampling_work_words,
    };
    let d = device()?;
    let m = d.as_metal_device()?;
    let rows = 2usize;
    for (vocab, k, p, t) in [
        (128_000usize, 20usize, 0.95f64, 1.0f64),
        (128_000, 50, 1.0, 0.1),
        (4_096, 6, 0.9, 0.7),
    ] {
        let z = super::timing_tests::logits(vocab, rows);
        let uniforms = [0.37f32, 0.76];
        let params = Params::new(t, k, p);
        let input = Tensor::from_vec(z.clone(), (rows, vocab), &d)?;
        let u = Tensor::from_vec(uniforms.to_vec(), rows, &d)?;
        let (zs, _) = input.storage_and_layout();
        let (us, _) = u.storage_and_layout();
        let (Storage::Metal(zs), Storage::Metal(us)) = (&*zs, &*us) else {
            unreachable!()
        };
        let alloc = |count: usize, dtype, label| -> Result<MetalStorage> {
            let b = m
                .new_buffer_builder()
                .with_size_for(count, dtype)
                .with_label(label)
                .build()?;
            MetalStorage::new_allocated(b, m.clone(), count, dtype)
        };
        let work = alloc(
            sampling_work_words(rows, vocab).unwrap(),
            DType::U32,
            "redraw_work",
        )?;
        let scratch = alloc(rows * vocab, DType::U32, "redraw_scratch")?;
        let selection = alloc(rows * 4, DType::U32, "redraw_selection")?;
        let weights = alloc(rows * vocab, DType::F32, "redraw_weights")?;
        let support = alloc(rows * vocab, DType::U32, "redraw_support")?;
        let ids = alloc(rows, DType::U32, "redraw_ids")?;
        let status = alloc(rows * 4, DType::U32, "redraw_status")?;
        {
            let encoder = m.command_encoder()?;
            // SAFETY: sized, disjoint, live storage owned by this test.
            unsafe {
                call_sampling_select(
                    m.metal_device(),
                    &encoder,
                    m.kernels(),
                    (zs.buffer(), 0),
                    rows as u32,
                    vocab as u32,
                    k as u32,
                    u32::from(p != 1.0),
                    work.buffer(),
                    scratch.buffer(),
                    selection.buffer(),
                )
            }
            .map_err(crate::Error::wrap)?;
        }
        m.wait_until_completed()?;
        let want: Vec<(u32, Vec<u32>, Vec<u32>)> = (0..rows)
            .map(|row| {
                let zr = &z[row * vocab..(row + 1) * vocab];
                let kept = oracle::kept_ids(zr, params).unwrap();
                let mut mask = vec![0u32; vocab];
                for &id in &kept {
                    mask[id as usize] = 1;
                }
                (
                    oracle::draw(zr, params, uniforms[row]).unwrap(),
                    mask,
                    vec![0, u32::MAX, vocab as u32, kept.len() as u32],
                )
            })
            .collect();
        // Past `SAMPLING_CAP / |K|` repeats an accumulating count would
        // overflow the compact set and route the draw to its exact slow
        // paths, so the repeats stay inside that window, where the defect
        // is a wrong answer rather than a slow one.
        for redraw in 0..8 {
            {
                let encoder = m.command_encoder()?;
                // SAFETY: as above; the selection is these rows'.
                unsafe {
                    call_sampling_multinomial(
                        m.metal_device(),
                        &encoder,
                        m.kernels(),
                        (zs.buffer(), 0),
                        rows as u32,
                        vocab as u32,
                        (1.0 / t) as f32,
                        p as f32,
                        u32::from(p != 1.0),
                        (us.buffer(), 0),
                        work.buffer(),
                        scratch.buffer(),
                        weights.buffer(),
                        support.buffer(),
                        ids.buffer(),
                        status.buffer(),
                    )
                }
                .map_err(crate::Error::wrap)?;
            }
            m.wait_until_completed()?;
            let (got_ids, got_support, got_status) = (
                ids.to_cpu::<u32>()?,
                support.to_cpu::<u32>()?,
                status.to_cpu::<u32>()?,
            );
            for row in 0..rows {
                let (id, mask, st) = &want[row];
                assert_eq!(
                    (got_ids[row], &got_status[row * 4..row * 4 + 4]),
                    (*id, &st[..]),
                    "vocab {vocab} k {k} p {p} redraw {redraw} row {row}"
                );
                assert_eq!(
                    &got_support[row * vocab..(row + 1) * vocab],
                    &mask[..],
                    "vocab {vocab} k {k} p {p} redraw {redraw} row {row}: mask"
                );
            }
            println!(
                "REDRAW vocab={vocab} k={k} p={p} redraw={redraw} ids={got_ids:?} counts={:?}",
                (0..rows).map(|r| got_status[r * 4 + 3]).collect::<Vec<_>>()
            );
        }
    }
    Ok(())
}
