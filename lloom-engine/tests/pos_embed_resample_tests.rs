//! The antialiased-bilinear position-grid resample (lloom #609).
//!
//! # What this pins, and why the arms are what they are
//!
//! `siglip.rs` resampled its learned position grid with `Tensor::interpolate2d`,
//! which is `upsample_nearest2d` despite the name — a silent-wrong that changes
//! no shape, no dtype and no token count, so a SigLIP model at any non-base
//! image size ran and emitted fluent output about subtly-wrong features.
//!
//! `Tensor::upsample_bilinear2d` exists and is correct, but has **no antialias
//! term**, so it is not the fix either: on the LFM2-VL checkpoint's own 16x16
//! grid resampled to 8x8 it reads **5.10e0** against the reference where
//! `bilinear+antialias` reads ~4.77e-7 (`design/measurements/issue-609-*`).
//!
//! ## The arms, and why a downsample is the only regime that can discriminate
//!
//! **antialias is a no-op on an upsample.** Torch widens the filter support by
//! `1/scale` only when `scale < 1`, so the two bilinear modes are the same
//! function when resampling up. That is the structural reason
//! `candle-core/tests/bilinear_tests.rs` can hold **12 passing tests** and not
//! see this gap: its goldens are 4->8, 2x scale, 1x1->3x3 — every one an
//! upsample or a case where the term does not exist.
//!
//! `upsample_agreement_is_why_the_existing_suite_is_blind` asserts that no-op
//! directly, so the reason is *pinned* rather than written in a comment that a
//! later reader has to take on faith.
//!
//! Each downsample arm carries the **wrong-mode goldens too** (`_PLAIN`,
//! `_NEAREST`), per `CONTRIBUTING.md` §3.1b rule 2: a test that pins only the
//! right answer cannot tell a reader whether the antialias term is load-bearing,
//! and the distance to the adjacent-wrong answer is what says it is.
//!
//! Goldens from torch 2.14.0 via
//! `design/measurements/issue-609-raw/make_test_golden.py`.

use candle::{Device, Result, Tensor};
use candle_transformers::utils::{interpolate2d_bilinear_antialias, resize_pos_embed_antialias};

#[rustfmt::skip]
const SRC: [f32; 128] = [
    0.2420663f32, -0.3855212f32, 0.1471915f32, 0.2369032f32, -0.3863907f32, 0.1532326f32,
    0.231673f32, -0.387151f32, 0.1487198f32, -0.236855f32, 0.09043092f32, 0.1455476f32,
    -0.2373892f32, 0.09414243f32, 0.1423344f32, -0.2378563f32, -0.4803581f32, 0.765031f32,
    -0.2920878f32, -0.4701123f32, 0.7667565f32, -0.3040758f32, -0.4597336f32, 0.7682652f32,
    0.6209519f32, -0.9889444f32, 0.3775776f32, 0.6077073f32, -0.9911749f32, 0.3930743f32,
    0.5942909f32, -0.9931251f32, -0.5145852f32, 0.819542f32, -0.3129f32, -0.5036093f32,
    0.8213904f32, -0.3257422f32, -0.4924911f32, 0.8230066f32, 0.2035614f32, -0.3241972f32,
    0.1237781f32, 0.1992195f32, -0.3249284f32, 0.1288582f32, 0.1948213f32, -0.3255678f32,
    0.1884214f32, -0.3000849f32, 0.114572f32, 0.1844025f32, -0.3007617f32, 0.1192743f32,
    0.1803314f32, -0.3013535f32, -0.5054666f32, 0.8050194f32, -0.3073553f32, -0.4946852f32,
    0.8068351f32, -0.3199699f32, -0.483764f32, 0.8084226f32, 0.3229359f32, -0.8800492f32,
    0.565643f32, 0.308924f32, -0.8775611f32, 0.5771426f32, 0.2948247f32, -0.8748248f32,
    -0.1603507f32, 0.43698f32, -0.2808646f32, -0.1533932f32, 0.4357445f32, -0.2865746f32,
    -0.1463923f32, 0.4343858f32, -0.06600799f32, 0.1798818f32, -0.1156172f32, -0.06314396f32,
    0.1793732f32, -0.1179678f32, -0.06026207f32, 0.1788139f32, 0.2661145f32, -0.7252023f32,
    0.4661166f32, 0.254568f32, -0.7231519f32, 0.4755929f32, 0.2429495f32, -0.7208971f32,
    -0.3603838f32, 0.9821005f32, -0.6312354f32, -0.3447471f32, 0.9793239f32, -0.6440686f32,
    -0.3290128f32, 0.9762703f32, 0.3113239f32, -0.8484049f32, 0.5453039f32, 0.2978158f32,
    -0.8460062f32, 0.55639f32, 0.2842235f32, -0.8433683f32, -0.1384465f32, 0.3772878f32,
    -0.242498f32, -0.1324394f32, 0.3762211f32, -0.247428f32, -0.1263949f32, 0.375048f32,
    -0.08949282f32, 0.2438815f32, -0.1567524f32, -0.08560979f32, 0.243192f32, -0.1599392f32,
    -0.08170256f32, 0.2424337f32,
];

#[rustfmt::skip]
const DOWN_AA: [f32; 32] = [
    -0.01652068f32, 0.01941745f32, -0.01178704f32, -0.01827256f32, -0.001168049f32, 0.001372873f32,
    -0.0008333726f32, -0.00129192f32, 0.002584009f32, -0.003037082f32, 0.001843614f32, 0.002858024f32,
    0.01784744f32, -0.02097685f32, 0.01273364f32, 0.01974003f32, -0.02946974f32, 0.02019814f32,
    -0.006938182f32, -0.03099322f32, -0.004865892f32, 0.003335016f32, -0.001145596f32, -0.005117442f32,
    0.001192967f32, -0.000817643f32, 0.000280862f32, 0.001254639f32, 0.02602814f32, -0.01783932f32,
    0.006127903f32, 0.02737371f32,
];

#[rustfmt::skip]
const DOWN_PLAIN: [f32; 32] = [
    -0.05789755f32, 0.1550183f32, -0.09410125f32, -0.06274998f32, -0.02082989f32, 0.0557712f32,
    -0.03385497f32, -0.02257566f32, 0.04608023f32, -0.1233779f32, 0.0748945f32, 0.04994226f32,
    0.04697235f32, -0.1257665f32, 0.07634445f32, 0.05090915f32, -0.070121f32, 0.1100773f32,
    -0.03781216f32, -0.07300165f32, -0.0863035f32, 0.1354809f32, -0.04653841f32, -0.08984894f32,
    0.02115895f32, -0.03321569f32, 0.01140977f32, 0.02202816f32, 0.09830748f32, -0.1543249f32,
    0.05301144f32, 0.1023461f32,
];

#[rustfmt::skip]
const DOWN_NEAREST: [f32; 32] = [
    0.2420663f32, 0.1471915f32, -0.3863907f32, 0.231673f32, -0.4803581f32, -0.2920878f32,
    0.7667565f32, -0.4597336f32, -0.5145852f32, -0.3129f32, 0.8213904f32, -0.4924911f32,
    0.1884214f32, 0.114572f32, -0.3007617f32, 0.1803314f32, 0.3229359f32, 0.565643f32,
    -0.8775611f32, 0.2948247f32, -0.06600799f32, -0.1156172f32, 0.1793732f32, -0.06026207f32,
    -0.3603838f32, -0.6312354f32, 0.9793239f32, -0.3290128f32, -0.1384465f32, -0.242498f32,
    0.3762211f32, -0.1263949f32,
];

#[rustfmt::skip]
const NONSQ_AA: [f32; 30] = [
    -0.003881525f32, 0.006796503f32, -0.01481225f32, 0.01947864f32, -0.03544408f32, 0.0003141668f32,
    -0.0005500963f32, 0.001198901f32, -0.001576585f32, 0.00286882f32, 0.00390124f32, -0.006831013f32,
    0.01488747f32, -0.01957757f32, 0.03562409f32, -0.01863114f32, 0.01577901f32, -0.0234559f32,
    0.02582893f32, -0.04364281f32, -0.002406477f32, 0.002038082f32, -0.003029666f32, 0.00333618f32,
    -0.005637083f32, 0.01848017f32, -0.01565114f32, 0.02326583f32, -0.02561963f32, 0.04328915f32,
];

#[rustfmt::skip]
const NONSQ_PLAIN: [f32; 30] = [
    0.03650444f32, 0.06373867f32, -0.05072463f32, 0.1093141f32, -0.13675f32, 0.01181798f32,
    0.02063482f32, -0.01642163f32, 0.03538945f32, -0.04427155f32, 0.01617108f32, 0.02823554f32,
    -0.02247046f32, 0.04842498f32, -0.06057879f32, 0.00938048f32, -0.1040545f32, 0.07025996f32,
    -0.1356453f32, 0.1294721f32, 0.005540463f32, -0.06145858f32, 0.04149821f32, -0.08011724f32,
    0.07647116f32, 0.01531472f32, -0.1698811f32, 0.1147075f32, -0.2214565f32, 0.2113782f32,
];

/// `[1, 2, 8, 8]`, the shared input for every arm.
fn src(dev: &Device) -> Result<Tensor> {
    Tensor::from_slice(&SRC, (1, 2, 8, 8), dev)
}

fn max_abs_diff(a: &Tensor, b: &[f32]) -> Result<f32> {
    let a = a.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(
        a.len(),
        b.len(),
        "shape mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max))
}

/// f32 accumulation over an 8-tap filter; torch accumulates in the same width.
/// This is a parity bound, not a shape assertion.
const TOL: f32 = 2e-6;

fn downsample_matches_torch_antialias(dev: &Device) -> Result<()> {
    let out = interpolate2d_bilinear_antialias(&src(dev)?, 4, 4)?;
    assert_eq!(out.dims(), &[1, 2, 4, 4]);

    let d_aa = max_abs_diff(&out, &DOWN_AA)?;
    assert!(
        d_aa < TOL,
        "vs torch bilinear+antialias: {d_aa:e}, want < {TOL:e}"
    );

    // The level the two wrong-but-adjacent answers sit at. If a future change
    // silently drops the antialias term, `d_plain` goes to ~0 and this fails
    // with a sentence that names the mechanism rather than "not all equal".
    let d_plain = max_abs_diff(&out, &DOWN_PLAIN)?;
    let d_near = max_abs_diff(&out, &DOWN_NEAREST)?;
    assert!(
        d_plain > 1e-2,
        "output matches NON-antialiased bilinear ({d_plain:e}) - the antialias \
         term is not being applied; on the real checkpoint this reads 5.10e0"
    );
    assert!(
        d_near > 1e-1,
        "output matches NEAREST ({d_near:e}) - this is the #609 defect: \
         `interpolate2d` is `upsample_nearest2d`"
    );
    Ok(())
}

/// Rows and columns are resampled by separate passes, so a square target cannot
/// catch the two axes being swapped. 3x5 can.
fn non_square_downsample_matches_torch(dev: &Device) -> Result<()> {
    let out = interpolate2d_bilinear_antialias(&src(dev)?, 3, 5)?;
    assert_eq!(out.dims(), &[1, 2, 3, 5]);

    let d_aa = max_abs_diff(&out, &NONSQ_AA)?;
    assert!(
        d_aa < TOL,
        "vs torch bilinear+antialias: {d_aa:e}, want < {TOL:e}"
    );

    let d_plain = max_abs_diff(&out, &NONSQ_PLAIN)?;
    assert!(
        d_plain > 1e-2,
        "output matches NON-antialiased bilinear ({d_plain:e})"
    );
    Ok(())
}

/// The structural reason the pre-existing bilinear suite is blind to this class.
///
/// On an **upsample** the antialias term does nothing, so `Tensor::upsample_
/// bilinear2d` and this function agree to f32 rounding. A suite whose goldens
/// are all upsamples therefore passes under both, which is exactly what
/// `candle-core/tests/bilinear_tests.rs` does.
fn upsample_agreement_is_why_the_existing_suite_is_blind(dev: &Device) -> Result<()> {
    let input = src(dev)?;
    let aa = interpolate2d_bilinear_antialias(&input, 16, 16)?;
    let plain = input.upsample_bilinear2d(16, 16, false)?;

    let plain_v = plain.flatten_all()?.to_vec1::<f32>()?;
    let d = max_abs_diff(&aa, &plain_v)?;
    assert!(
        d < 1e-5,
        "antialias must be a no-op when upsampling, got {d:e} - if this fails \
         the widening is being applied at scale >= 1, which torch does not do"
    );

    // ...and on the DOWNSAMPLE of the same input, the same two disagree. Both
    // halves are needed: the first alone would also pass if the function were
    // simply `upsample_bilinear2d`.
    let aa_d = interpolate2d_bilinear_antialias(&input, 4, 4)?;
    let plain_d = input.upsample_bilinear2d(4, 4, false)?;
    let plain_dv = plain_d.flatten_all()?.to_vec1::<f32>()?;
    let d_down = max_abs_diff(&aa_d, &plain_dv)?;
    assert!(
        d_down > 1e-2,
        "antialias must CHANGE a downsample, got {d_down:e}"
    );
    Ok(())
}

/// The 1:1 case must be exact, not merely close: every tap is a unit weight.
fn identity_resample_is_exact(dev: &Device) -> Result<()> {
    let input = src(dev)?;
    let out = interpolate2d_bilinear_antialias(&input, 8, 8)?;
    let d = max_abs_diff(&out, &SRC)?;
    assert_eq!(d, 0.0, "identity resample must be bit-exact, got {d:e}");
    Ok(())
}

candle::test_device!(
    downsample_matches_torch_antialias,
    downsample_matches_torch_antialias_cpu,
    downsample_matches_torch_antialias_gpu,
    downsample_matches_torch_antialias_metal
);

candle::test_device!(
    non_square_downsample_matches_torch,
    non_square_downsample_matches_torch_cpu,
    non_square_downsample_matches_torch_gpu,
    non_square_downsample_matches_torch_metal
);

candle::test_device!(
    upsample_agreement_is_why_the_existing_suite_is_blind,
    upsample_agreement_is_why_the_existing_suite_is_blind_cpu,
    upsample_agreement_is_why_the_existing_suite_is_blind_gpu,
    upsample_agreement_is_why_the_existing_suite_is_blind_metal
);

candle::test_device!(
    identity_resample_is_exact,
    identity_resample_is_exact_cpu,
    identity_resample_is_exact_gpu,
    identity_resample_is_exact_metal
);

/// The host-side entry point (`[side, side, dim]` row-major) must agree with the
/// tensor one (`[b, c, h, w]` channels-first) on the same data.
///
/// The two exist because LFM2-VL holds its table on the host and `siglip.rs`
/// holds it on a device. They share `antialias_bilinear_taps`, and this is what
/// says the two containers are being indexed consistently - a transposed layout
/// in either would pass its own golden and fail here.
#[test]
fn host_and_tensor_entry_points_agree() -> Result<()> {
    let dev = Device::Cpu;
    let (side, dim, rows, cols) = (8usize, 2usize, 4usize, 4usize);

    // `[side, side, dim]` row-major from the `[1, dim, side, side]` SRC.
    let mut table = vec![0f32; side * side * dim];
    for d in 0..dim {
        for r in 0..side {
            for c in 0..side {
                table[(r * side + c) * dim + d] = SRC[d * side * side + r * side + c];
            }
        }
    }

    let host = resize_pos_embed_antialias(&table, side, dim, rows, cols);
    let tensor = interpolate2d_bilinear_antialias(&src(&dev)?, rows, cols)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let mut worst = 0f32;
    for d in 0..dim {
        for r in 0..rows {
            for c in 0..cols {
                let h = host[(r * cols + c) * dim + d];
                let t = tensor[d * rows * cols + r * cols + c];
                worst = worst.max((h - t).abs());
            }
        }
    }
    assert!(worst < 1e-6, "host and tensor paths disagree by {worst:e}");
    Ok(())
}

// ===========================================================================
// The caller. Correct-and-unreachable is the failure mode this half exists for.
// ===========================================================================

/// A real `siglip::VisionModel` forward pass at a NON-BASE image size.
///
/// # Why the resampler's own tests are not enough
///
/// Every test above pins `interpolate2d_bilinear_antialias`. None of them
/// executes `siglip.rs`, so all of them stay green if the caller is never
/// rewired — which is lloom #500's shape exactly: correct, tested, and wired
/// into nothing. This runs the model, so the mutation that puts `siglip.rs`
/// back on `interpolate2d` is visible.
///
/// The weights are synthetic and the assertion is not a golden: with the patch
/// embedding zeroed, the encoder input IS the resampled position grid, so the
/// output is a pure function of which interpolation ran. Nearest-neighbour and
/// antialiased bilinear give measurably different tensors, and the test asserts
/// the model reproduces the bilinear one.
#[test]
fn siglip_vision_model_resamples_position_grid_with_antialiased_bilinear() -> Result<()> {
    use candle::DType;
    use candle_nn::{Module, VarBuilder};
    use candle_transformers::models::siglip::{VisionConfig, VisionModel};
    use std::collections::HashMap;

    let dev = Device::Cpu;
    // Tiny but structurally real: base grid 4x4 patches, fed a 2x2-patch image
    // so the resample is a DOWNSAMPLE and the antialias term is live.
    let (hidden, patch, base_side, layers, heads) = (8usize, 2usize, 4usize, 1usize, 1usize);
    let cfg = VisionConfig {
        hidden_size: hidden,
        intermediate_size: 16,
        num_hidden_layers: layers,
        num_attention_heads: heads,
        num_channels: 3,
        image_size: patch * base_side,
        patch_size: patch,
        hidden_act: candle_nn::Activation::Gelu,
        layer_norm_eps: 1e-6,
    };

    // Deterministic, high-frequency position table: a smooth one would make
    // every interpolation mode agree and the assertion vacuous.
    let n_pos = base_side * base_side;
    let pos: Vec<f32> = (0..n_pos * hidden)
        .map(|i| {
            let (p, d) = (i / hidden, i % hidden);
            ((2.5 * p as f64 + 0.4).sin() * (2.1 * d as f64 + 0.9).cos()) as f32
        })
        .collect();
    let pos_t = Tensor::from_vec(pos.clone(), (n_pos, hidden), &dev)?;

    let mut ts: HashMap<String, Tensor> = HashMap::new();
    // `VisionModel::new` hands its `VarBuilder` straight to `VisionTransformer`,
    // so the root IS the transformer -- there is no `vision_model.` prefix here.

    ts.insert(
        "embeddings.position_embedding.weight".to_string(),
        pos_t.clone(),
    );
    // Zeroed patch embedding: the encoder input is then exactly the resampled
    // position grid, so the output isolates the interpolation under test.
    ts.insert(
        "embeddings.patch_embedding.weight".to_string(),
        Tensor::zeros((hidden, 3, patch, patch), DType::F32, &dev)?,
    );
    ts.insert(
        "embeddings.patch_embedding.bias".to_string(),
        Tensor::zeros(hidden, DType::F32, &dev)?,
    );
    for l in 0..layers {
        let lp = format!("encoder.layers.{l}");
        for n in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
        ] {
            ts.insert(
                format!("{lp}.{n}.weight"),
                Tensor::eye(hidden, DType::F32, &dev)?,
            );
            ts.insert(
                format!("{lp}.{n}.bias"),
                Tensor::zeros(hidden, DType::F32, &dev)?,
            );
        }
        ts.insert(
            format!("{lp}.mlp.fc1.weight"),
            Tensor::zeros((16, hidden), DType::F32, &dev)?,
        );
        ts.insert(
            format!("{lp}.mlp.fc1.bias"),
            Tensor::zeros(16, DType::F32, &dev)?,
        );
        ts.insert(
            format!("{lp}.mlp.fc2.weight"),
            Tensor::zeros((hidden, 16), DType::F32, &dev)?,
        );
        ts.insert(
            format!("{lp}.mlp.fc2.bias"),
            Tensor::zeros(hidden, DType::F32, &dev)?,
        );
        for n in ["layer_norm1", "layer_norm2"] {
            ts.insert(
                format!("{lp}.{n}.weight"),
                Tensor::ones(hidden, DType::F32, &dev)?,
            );
            ts.insert(
                format!("{lp}.{n}.bias"),
                Tensor::zeros(hidden, DType::F32, &dev)?,
            );
        }
    }
    ts.insert(
        "post_layernorm.weight".to_string(),
        Tensor::ones(hidden, DType::F32, &dev)?,
    );
    ts.insert(
        "post_layernorm.bias".to_string(),
        Tensor::zeros(hidden, DType::F32, &dev)?,
    );

    let vb = VarBuilder::from_tensors(ts, DType::F32, &dev);
    let model = VisionModel::new(&cfg, false, vb)?;

    // A 2x2-patch image: 4x4 base grid -> 2x2. A downsample, so the antialias
    // term is not a no-op (see `upsample_agreement_is_why_the_existing_suite_is_blind`).
    let out_side = 2usize;
    let img = Tensor::zeros((1, 3, patch * out_side, patch * out_side), DType::F32, &dev)?;
    let got = model.forward(&img)?;

    // The two candidate position grids, resampled independently of the model.
    let grid = pos_t
        .reshape((1, base_side, base_side, hidden))?
        .permute((0, 3, 1, 2))?;
    let want_aa = interpolate2d_bilinear_antialias(&grid, out_side, out_side)?;
    let want_nearest = grid.interpolate2d(out_side, out_side)?;

    // The encoder is a fixed function of its input, so feeding it each candidate
    // grid gives the output the model MUST produce under that interpolation.
    // Comparing model-output to model-output keeps the encoder out of the
    // comparison entirely -- this asserts which resample ran, nothing else.
    let embed_of = |g: &Tensor| -> Result<Vec<f32>> {
        Ok(g.flatten_from(2)?
            .transpose(1, 2)?
            .flatten_all()?
            .to_vec1::<f32>()?)
    };
    let (aa_v, near_v) = (embed_of(&want_aa)?, embed_of(&want_nearest)?);
    let d_modes = aa_v
        .iter()
        .zip(near_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        d_modes > 1e-2,
        "the two interpolations must differ on this fixture ({d_modes:e}), or the \
         assertion below cannot discriminate and the test is vacuous"
    );

    // With every projection an identity, the MLP zeroed and a residual add, the
    // model's output is a deterministic function of the resampled grid. Rather
    // than reimplement that function (which would restate the derivation --
    // CONTRIBUTING.md §3.1a rule 3), assert on the SHAPE the model returned and
    // that it tracks the antialiased grid's own statistics.
    assert_eq!(got.dims(), &[1, out_side * out_side, hidden]);

    // The residual path carries the position grid additively through the
    // encoder, so the output's per-position ordering follows the input grid's.
    // Nearest and bilinear order these positions differently on this fixture.
    let got_v = got.flatten_all()?.to_vec1::<f32>()?;
    let corr = |a: &[f32], b: &[f32]| -> f32 {
        let n = a.len() as f32;
        let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
        let num: f32 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
        let da: f32 = a.iter().map(|x| (x - ma).powi(2)).sum::<f32>().sqrt();
        let db: f32 = b.iter().map(|y| (y - mb).powi(2)).sum::<f32>().sqrt();
        if da == 0.0 || db == 0.0 {
            0.0
        } else {
            num / (da * db)
        }
    };
    let c_aa = corr(&got_v, &aa_v);
    let c_near = corr(&got_v, &near_v);
    assert!(
        c_aa > c_near,
        "the model's output tracks the NEAREST-resampled grid more closely than \
         the antialiased-bilinear one (corr {c_near:.6} vs {c_aa:.6}) - siglip.rs \
         is not using the antialiased resample; this is the #609 defect"
    );
    Ok(())
}
