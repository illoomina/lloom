//! LFM2's attention route (`models::lfm2::AttnRoute`), decided once at load:
//! each phase takes the requested fused kernel where it covers the head dim
//! and dtype, else the nearest fused kernel that does, with one warning naming
//! the head dim, the kernel requested and the kernel taken — on the contiguous
//! cache, the private page pool and a shared pool's rows. A shape no fused
//! kernel covers is refused at load with the missing capability named; the
//! composed attention is the CPU's path and, on Metal, only the reference arm
//! (`--attn generic`) (`docs/dependency-ordered-reuse-design.md` §8).
//!
//! The reference is the CPU backend at f32 over the same f16-rounded tiny
//! weights (`lloom::testing::tiny`), so each arm is held to an absolute
//! reference rather than to another arm (CONTRIBUTING §3.1b). The census is
//! read through the device's own window (`invariants.md` §15.1n). An
//! instantiated head dim on each arm is the control: its fused kernel is in
//! the census, so a census that could not see the kernel would fail there.

use lloom::models::lfm2::{AttnImpl, Config, DecodeAttn, Model, PrefillAttn, Shape as Lfm2Shape};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::profile::CrownExact;
#[cfg(feature = "metal")]
use lloom::profile::PoolSpecExact;
use lloom::testing::tiny::{TinyCheckpoint, TinyFamily, Weights};
use lloom_tensor::{DType, Device, Result};

/// Six layers, attention at layers 2 and 5 (two attention layers, so a
/// warning printed per layer would print twice), `n_heads` query heads over
/// one KV head of `head_dim`.
fn config(attn_impl: AttnImpl, head_dim: usize) -> Config {
    let mut cfg = Lfm2Tiny::template();
    cfg.num_hidden_layers = 6;
    cfg.layer_types = Config::tiny_layer_types(6);
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.hidden_size = 2 * head_dim;
    cfg.intermediate_size = 128;
    cfg.max_position_embeddings = 256;
    cfg.attn_impl = attn_impl;
    cfg
}

fn build<P: Lfm2Shape>(cfg: &Config, dev: &Device, dtype: DType) -> Result<Model<P>> {
    TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default())
        .with_dtype(dtype)
        .model::<P>(dev)
}

/// On the CPU the fused arms do not exist, whatever the head dim: every phase
/// is `Generic`, nothing was refused, and nothing is printed — the device,
/// not the head dim, is why.
#[test]
fn the_cpu_takes_the_composed_attention_without_a_warning() -> Result<()> {
    for arm in [AttnImpl::Sdpa, AttnImpl::FlashDecoding] {
        let cfg = config(arm, 48);
        let model = build::<CrownExact>(&cfg, &Device::Cpu, DType::F32)?;
        let route = model.attn_route();
        assert_eq!(
            (route.decode, route.prefill),
            (DecodeAttn::Generic, PrefillAttn::Generic)
        );
        assert_eq!((route.decode_refused, route.prefill_refused), (None, None));
        assert!(
            model.load_warnings().is_empty(),
            "{:?}",
            model.load_warnings()
        );
    }
    Ok(())
}

#[cfg(feature = "metal")]
mod metal {
    use super::*;
    use lloom::models::lfm2::{AttnRoute, BatchPool, ConvState, KvAppend, State};
    use lloom::nn::Row;
    use lloom_tensor::quantized::GgmlDType;
    use lloom_tensor::Tensor;

    /// The arms the CPU can run: the composed attention over the in-place
    /// cache, the unfused conv.
    fn cpu_config(cfg: &Config) -> Config {
        let mut cpu = cfg.clone();
        cpu.attn_impl = AttnImpl::Generic;
        cpu.kv_append = KvAppend::InPlace;
        cpu.conv_state = ConvState::default();
        cpu.conv_fused = false;
        cpu
    }

    /// The CPU model at f32 over the weights a `dtype` Metal model holds.
    fn cpu_model<P: Lfm2Shape>(cfg: &Config, dtype: DType) -> Result<Model<P>> {
        let dense = match dtype {
            DType::F16 => GgmlDType::F16,
            DType::BF16 => GgmlDType::BF16,
            _ => GgmlDType::F32,
        };
        TinyCheckpoint::<Lfm2Tiny>::new(
            cpu_config(cfg),
            Weights {
                dense,
                ..Weights::default()
            },
        )
        .with_dtype(DType::F32)
        .model::<P>(&Device::Cpu)
    }

    fn logits(t: &Tensor) -> Result<Vec<f32>> {
        t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()
    }

    /// A five-token prefill, two decode steps, a three-token second prefill onto
    /// the cache (§10.9) and one more decode step: the logits of each pass.
    const PASSES: [&[u32]; 5] = [&[1, 2, 3, 4, 5], &[7], &[9], &[11, 13, 17], &[19]];

    fn passes<P: Lfm2Shape>(
        model: &Model<P>,
        cache: &mut State<P>,
        dev: &Device,
    ) -> Result<Vec<Vec<f32>>> {
        let mut pos = 0;
        let mut out = Vec::new();
        for toks in PASSES {
            let inp = Tensor::new(toks, dev)?.reshape((1, toks.len()))?;
            out.push(logits(&model.forward(&inp, pos, cache)?)?);
            pos += toks.len();
        }
        Ok(out)
    }

    /// The largest |got - want| over a pass, relative to the reference's
    /// largest magnitude; NaN if any logit is NaN (`f32::max` would drop it).
    fn rel(got: &[f32], want: &[f32]) -> f32 {
        let scale = want.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
        let d = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, |m, d| {
                if d.is_nan() || m.is_nan() {
                    f32::NAN
                } else {
                    m.max(d)
                }
            });
        d / scale
    }

    /// f16 activations on Metal against the f32 reference over the same rounded
    /// weights. Measured at most 5.1e-3, on the fused and the composed arms
    /// alike (head_dim 256); a wrong route or a wrong read is far outside it.
    const TOL: f32 = 1e-2;
    /// bf16 activations: eight mantissa bits where f16 has eleven.
    const TOL_BF16: f32 = 5e-2;

    fn assert_close(what: &str, got: &[Vec<f32>], want: &[Vec<f32>], tol: f32) {
        assert_eq!(got.len(), want.len(), "{what}: pass count");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let r = rel(g, w);
            eprintln!("{what}: pass {i}: |metal - cpu| / scale = {r:e}");
            assert!(
                r.is_finite() && r <= tol,
                "{what}: pass {i} is {r:e} from the CPU (tolerance {tol})"
            );
        }
    }

    /// Before any device: the census is off unless `CANDLE_METAL_PROFILE` is set
    /// when it is first read.
    fn profile_on() {
        static ON: std::sync::Once = std::sync::Once::new();
        ON.call_once(|| std::env::set_var("CANDLE_METAL_PROFILE", "1"));
    }

    /// The labels of a census that name an attention kernel of each kind.
    fn attention_labels(census: &[(String, u64)]) -> (u64, u64, u64, u64) {
        let count = |p: &str| -> u64 {
            census
                .iter()
                .filter(|(l, _)| l.starts_with(p))
                .map(|(_, n)| *n)
                .sum()
        };
        (
            count("sdpa_vector"),
            count("flash_decoding"),
            count("steel_attention"),
            count("softmax"),
        )
    }

    /// Every pass of [`PASSES`] on `cfg`'s arms on Metal, with the census of
    /// the decode steps and of the second prefill, each read in its own
    /// window on `dev`.
    struct Run {
        /// The route the model resolved and its load warnings: what the
        /// cases read (the model itself is one profile per case).
        route: lloom::models::lfm2::AttnRoute,
        warnings: Vec<lloom::load_warning::LoadWarning>,
        logits: Vec<Vec<f32>>,
        decode_census: Vec<(String, u64)>,
        prefill_census: Vec<(String, u64)>,
    }

    fn run_on_metal(cfg: &Config, dev: &Device, dtype: DType) -> Result<Run> {
        let key = cfg.exact_key();
        lloom::profile::dispatch!(key, lloom::profile::LFM2, P => run_on_metal_at::<P>(cfg, dev, dtype))
    }

    fn run_on_metal_at<P: Lfm2Shape>(cfg: &Config, dev: &Device, dtype: DType) -> Result<Run> {
        let model = build::<P>(cfg, dev, dtype)?;
        let mut cache = State::<P>::new(true, dtype, cfg, dev)?;
        let census = dev.as_metal_device()?.dispatch_census().clone();
        let mut out = Vec::new();
        let mut decode = Vec::new();
        let mut prefill = Vec::new();
        let mut pos = 0;
        for toks in PASSES {
            dev.synchronize()?;
            let window = census.open().expect("one census window on this device");
            let inp = Tensor::new(toks, dev)?.reshape((1, toks.len()))?;
            out.push(logits(&model.forward(&inp, pos, &mut cache)?)?);
            dev.synchronize()?;
            let seen = window.close();
            if pos > 0 {
                if toks.len() == 1 {
                    decode.extend(seen);
                } else {
                    prefill.extend(seen);
                }
            }
            pos += toks.len();
        }
        Ok(Run {
            route: model.attn_route(),
            warnings: model.load_warnings().to_vec(),
            logits: out,
            decode_census: decode,
            prefill_census: prefill,
        })
    }

    fn reference(cfg: &Config, dtype: DType) -> Result<Vec<Vec<f32>>> {
        let cpu = cpu_config(cfg);
        let model = cpu_model::<CrownExact>(cfg, dtype)?;
        let mut cache = State::<CrownExact>::new(true, DType::F32, &cpu, &Device::Cpu)?;
        passes(&model, &mut cache, &Device::Cpu)
    }

    use DecodeAttn::{
        FlashDecoding as Flash, Generic as DGeneric, SdpaVector as Vector, Steel as DSteel,
    };
    use PrefillAttn::{Generic as PGeneric, Steel};

    /// One cell of the route: the arm, head dim, dtype and KV store asked
    /// for, and the kernels each phase must take, with the kernel the arm
    /// requested when the route took another.
    struct Cell {
        arm: AttnImpl,
        head_dim: usize,
        dtype: DType,
        paged: bool,
        decode: DecodeAttn,
        decode_refused: Option<DecodeAttn>,
        prefill: PrefillAttn,
        prefill_refused: Option<PrefillAttn>,
    }

    #[allow(clippy::too_many_arguments)]
    fn cell(
        arm: AttnImpl,
        head_dim: usize,
        dtype: DType,
        paged: bool,
        decode: DecodeAttn,
        decode_refused: Option<DecodeAttn>,
        prefill: PrefillAttn,
        prefill_refused: Option<PrefillAttn>,
    ) -> Cell {
        Cell {
            arm,
            head_dim,
            dtype,
            paged,
            decode,
            decode_refused,
            prefill,
            prefill_refused,
        }
    }

    fn kernel_name(k: DecodeAttn) -> &'static str {
        match k {
            DGeneric => "generic",
            Vector => "sdpa_vector",
            DSteel => "steel_attention",
            Flash => "flash_decoding",
        }
    }

    /// `maxTotalThreadsPerThreadgroup` of a compiled `sdpa_vector` pipeline,
    /// read here rather than through the route's own check.
    fn sdpa_vector_threads(dev: &Device, name: &str) -> usize {
        use lloom_tensor::kernels::metal::library::{ConstantValues, Value};
        use lloom_tensor::kernels::source::Source;
        let m = dev.as_metal_device().unwrap();
        m.kernels()
            .load_pipeline_with_constants(
                m.device(),
                Source::Sdpa,
                name.to_string(),
                Some(ConstantValues::new(vec![(20, Value::Bool(false))])),
            )
            .unwrap()
            .max_total_threads_per_threadgroup()
    }

    /// The preference order, cell by cell: the requested fused kernel where it
    /// covers the shape; else the other fused decode kernel (72 and 80 are
    /// FlashDecoding's alone, bf16 `sdpa_vector`'s alone, over a contiguous
    /// cache and over a gathered page pool); else the tiled kernel over one
    /// query. `--attn generic` is the composed reference, without a warning.
    fn cells(dev: &Device) -> Vec<Cell> {
        use lloom::models::lfm2::AttnImpl::{FlashDecoding as F, Generic as G, Sdpa as S};
        let f16 = DType::F16;
        let bf16 = DType::BF16;
        let mut v = vec![
            cell(S, 64, f16, false, Vector, None, Steel, None),
            cell(S, 72, f16, false, Flash, Some(Vector), Steel, None),
            cell(S, 80, f16, false, Flash, Some(Vector), Steel, None),
            cell(S, 256, f16, false, Vector, None, Steel, None),
            cell(S, 512, f16, false, Vector, None, Steel, None),
            cell(F, 64, f16, false, Flash, None, Steel, None),
            cell(F, 72, f16, false, Flash, None, Steel, None),
            cell(F, 80, f16, false, Flash, None, Steel, None),
            cell(F, 256, f16, false, Flash, None, Steel, None),
            cell(F, 512, f16, false, Flash, None, Steel, None),
            cell(F, 64, f16, true, Flash, None, Steel, None),
            cell(F, 72, f16, true, Flash, None, Steel, None),
            cell(F, 256, f16, true, Flash, None, Steel, None),
            cell(G, 64, f16, false, DGeneric, None, PGeneric, None),
            cell(F, 64, bf16, false, Vector, Some(Flash), Steel, None),
            cell(F, 64, bf16, true, Vector, Some(Flash), Steel, None),
        ];
        // `sdpa_vector` at bf16 x 512 is dispatched at 1024 threads; where its
        // pipeline admits fewer it writes nothing, FlashDecoding has no bf16,
        // and the tiled kernel serves the one query.
        let fits = sdpa_vector_threads(dev, "sdpa_vector_bfloat16_t_512") >= 1024;
        v.push(if fits {
            cell(S, 512, bf16, false, Vector, None, Steel, None)
        } else {
            cell(S, 512, bf16, false, DSteel, Some(Vector), Steel, None)
        });
        v
    }

    /// Every cell: the model builds, every pass agrees with the CPU at f32
    /// over the same rounded weights, the route is the cell's, the census
    /// shows the kernel the route names and none other, and a load that took
    /// another kernel than requested printed exactly one warning naming the
    /// head dim, the dtype, the kernel requested and the kernel taken.
    #[test]
    fn each_shape_takes_the_nearest_kernel_that_covers_it_and_matches_the_cpu() -> Result<()> {
        profile_on();
        let dev = Device::new_metal(0)?;
        lloom_tensor::test_utils::allow_emulated(&dev, DType::BF16);
        for c in cells(&dev) {
            let what = format!(
                "{:?} head_dim {} {:?}{}",
                c.arm,
                c.head_dim,
                c.dtype,
                if c.paged { " paged" } else { "" }
            );
            let mut cfg = config(c.arm, c.head_dim);
            if c.paged {
                cfg.kv_append = KvAppend::Paged;
            }
            let run = run_on_metal(&cfg, &dev, c.dtype)?;
            let tol = if c.dtype == DType::BF16 {
                TOL_BF16
            } else {
                TOL
            };
            assert_close(&what, &run.logits, &reference(&cfg, c.dtype)?, tol);

            let route = run.route;
            assert_eq!(
                (
                    route.decode,
                    route.decode_refused,
                    route.prefill,
                    route.prefill_refused
                ),
                (c.decode, c.decode_refused, c.prefill, c.prefill_refused),
                "{what}: the route"
            );

            let (vec_d, flash_d, steel_d, soft_d) = attention_labels(&run.decode_census);
            let (vec_p, flash_p, steel_p, soft_p) = attention_labels(&run.prefill_census);
            eprintln!(
                "{what}: decode sdpa_vector {vec_d} flash {flash_d} steel {steel_d} softmax {soft_d}; \
                 prefill sdpa_vector {vec_p} flash {flash_p} steel {steel_p} softmax {soft_p}"
            );
            // Two attention layers, three decode steps.
            let want_decode = match c.decode {
                Vector => (6, 0, 0, 0),
                Flash => (0, 6, 0, 0),
                DSteel => (0, 0, 6, 0),
                DGeneric => (0, 0, 0, 6),
            };
            assert_eq!(
                (vec_d, flash_d, steel_d, soft_d),
                want_decode,
                "{what}: the decode census"
            );
            let (want_steel, want_soft) = match c.prefill {
                Steel => (2, 0),
                PGeneric => (0, 2),
            };
            assert_eq!(
                (vec_p, flash_p, steel_p, soft_p),
                (0, 0, want_steel, want_soft),
                "{what}: the second prefill's census"
            );

            let warnings = &run.warnings;
            if c.decode_refused.is_none() && c.prefill_refused.is_none() {
                assert!(warnings.is_empty(), "{what}: {warnings:?}");
                continue;
            }
            assert_eq!(
                warnings.len(),
                1,
                "{what}: one warning per load: {warnings:?}"
            );
            let w = &warnings[0];
            assert_eq!(w.kind, lloom::load_warning::LoadWarningKind::AttentionRoute);
            let m = &w.message;
            assert!(
                m.starts_with(&format!(
                    "lfm2: attention head_dim {} at {:?} under {:?}: ",
                    c.head_dim, c.dtype, c.arm
                )),
                "{what}: {m}"
            );
            match c.decode_refused {
                Some(k) => {
                    let phrase = format!(
                        "decode requested {} and takes {} (the nearest fused kernel)",
                        kernel_name(k),
                        kernel_name(c.decode)
                    );
                    assert!(m.contains(&phrase), "{what}: {m}\n  wants: {phrase}");
                }
                None => assert!(!m.contains("decode"), "{what}: {m}"),
            }
            assert!(
                !m.contains("composed"),
                "{what}: a fused route named the composed path: {m}"
            );
            assert_eq!(
                Some(m.clone()),
                route.warning(),
                "{what}: the printed line is the route's"
            );
        }
        Ok(())
    }

    /// A shape no fused kernel covers is refused at load, naming what is
    /// missing, on every fused arm and store: head dims no kernel family
    /// instantiates, and f32 at 512, which none admits at its dispatched width.
    /// The composed attention serves none of them; at 48 the reference arm
    /// still loads, and is the attention.
    #[test]
    fn a_shape_no_fused_kernel_covers_is_refused_at_load() -> Result<()> {
        let dev = Device::new_metal(0)?;
        for (arm, head_dim, dtype, paged) in [
            (AttnImpl::Sdpa, 16, DType::F16, false),
            (AttnImpl::Sdpa, 48, DType::F16, false),
            (AttnImpl::FlashDecoding, 16, DType::F16, false),
            (AttnImpl::FlashDecoding, 48, DType::F16, false),
            (AttnImpl::FlashDecoding, 48, DType::F16, true),
            (AttnImpl::Sdpa, 512, DType::F32, false),
            (AttnImpl::FlashDecoding, 512, DType::F32, false),
        ] {
            let mut cfg = config(arm, head_dim);
            if paged {
                cfg.kv_append = KvAppend::Paged;
            }
            let what = format!("{arm:?} head_dim {head_dim} {dtype:?} paged={paged}");
            // The profile the config's axes name, as a served load takes it.
            let key = cfg.exact_key();
            let built: Result<()> = lloom::profile::dispatch!(
                key,
                lloom::profile::LFM2,
                P => build::<P>(&cfg, &dev, dtype).map(|_| ())
            );
            let err = match built {
                Ok(()) => panic!("{what}: loaded with no fused kernel covering it"),
                Err(e) => e.to_string(),
            };
            eprintln!("{what}: {err}");
            assert!(
                err.contains(&format!("head dim {head_dim}"))
                    && err.contains("does not serve on Metal"),
                "{what}: refused without naming the capability: {err}"
            );
        }
        let cfg = config(AttnImpl::Generic, 48);
        let model = build::<CrownExact>(&cfg, &dev, DType::F16)?;
        assert!(model.attn_route().prepared.is_none());
        let mut cache = State::<CrownExact>::new(true, DType::F16, &cfg, &dev)?;
        let got = passes::<CrownExact>(&model, &mut cache, &dev)?;
        assert_close(
            "reference arm head_dim 48",
            &got,
            &reference(&cfg, DType::F16)?,
            TOL,
        );
        Ok(())
    }

    /// A shared pool's rows (`BatchPool`, the `--preset pool` arms): the
    /// batched arms run, reading every row in one paged FlashDecoding dispatch
    /// per layer, each row agreeing with its own CPU forward: 64, and 72 on
    /// FlashDecoding's masked lanes.
    #[test]
    fn a_shared_pool_decodes_an_off_set_head_dim_row_by_row() -> Result<()> {
        profile_on();
        let dev = Device::new_metal(0)?;
        let rows: [(&[u32], u32); 3] = [
            (&[3, 11, 20, 5, 9], 12),
            (&[40, 2], 33),
            (&[1, 4, 9, 16, 25, 36, 49, 8], 21),
        ];
        for head_dim in [64, 72] {
            let mut cfg = config(AttnImpl::FlashDecoding, head_dim);
            cfg.kv_append = KvAppend::Paged;
            cfg.conv_state = ConvState::RotatingRing { k: 2 };
            cfg.conv_fused = true;
            cfg.flash_page_size = 32;
            cfg.flash_pages_per_chunk = 1;
            let what = format!("pool head_dim {head_dim}");
            let model = build::<PoolSpecExact>(&cfg, &dev, DType::F16)?;
            let pool = BatchPool::new(&cfg, &dev, 4, 64)?;
            pool.set_batched_arms(true);
            let mut caches = Vec::new();
            for (prompt, _) in rows {
                let mut cache =
                    State::<PoolSpecExact>::for_batch(true, DType::F16, &cfg, &dev, &pool)?;
                let inp = Tensor::new(prompt, &dev)?.unsqueeze(0)?;
                model.forward(&inp, 0, &mut cache)?;
                caches.push(cache);
            }
            let nexts: Vec<u32> = rows.iter().map(|(_, n)| *n).collect();
            let inp = Tensor::new(nexts.as_slice(), &dev)?.reshape((rows.len(), 1))?;
            let mut refs: Vec<Row<'_, State<PoolSpecExact>>> = caches
                .iter_mut()
                .zip(rows)
                .map(|(cache, (prompt, _))| Row {
                    pos: prompt.len(),
                    cache,
                })
                .collect();
            dev.synchronize()?;
            let window = dev
                .as_metal_device()?
                .dispatch_census()
                .open()
                .expect("one census window on this device");
            let got = model.forward_rows(&inp, &mut refs)?;
            dev.synchronize()?;
            let census = window.close();
            drop(refs);
            // The batched attention arm reads every row in one paged
            // FlashDecoding dispatch per layer, and nothing is composed.
            let (_, flash, _, soft) = attention_labels(&census);
            eprintln!("{what}: forward_rows flash {flash} softmax {soft}");
            assert_eq!(
                (flash > 0, soft),
                (true, 0),
                "{what}: the batched attention arm: {census:?}"
            );

            let cpu = cpu_config(&cfg);
            let reference_model = cpu_model::<CrownExact>(&cfg, DType::F16)?;
            for (r, (prompt, next)) in rows.iter().enumerate() {
                let mut cache = State::<CrownExact>::new(true, DType::F32, &cpu, &Device::Cpu)?;
                let inp = Tensor::new(*prompt, &Device::Cpu)?.unsqueeze(0)?;
                reference_model.forward(&inp, 0, &mut cache)?;
                let inp = Tensor::new(&[*next], &Device::Cpu)?.unsqueeze(0)?;
                let want = logits(&reference_model.forward(&inp, prompt.len(), &mut cache)?)?;
                let row = logits(&got.get(r)?)?;
                let d = rel(&row, &want);
                eprintln!("{what}: row {r}: |metal - cpu| / scale = {d:e}");
                assert!(
                    d <= TOL,
                    "{what}: row {r} is {d:e} from the CPU (tolerance {TOL})"
                );
            }
        }
        Ok(())
    }

    /// The route is the model's, resolved once: every attention layer takes
    /// it, and the resolution itself is a pure function of its inputs.
    #[test]
    fn the_route_is_resolved_from_the_arm_the_device_the_dtype_and_the_head_dim() -> Result<()> {
        let dev = Device::new_metal(0)?;
        let cfg = config(AttnImpl::FlashDecoding, 72);
        let a = AttnRoute::resolve(AttnImpl::FlashDecoding, &cfg, DType::BF16, &dev)?;
        let b = AttnRoute::resolve(AttnImpl::FlashDecoding, &cfg, DType::BF16, &dev)?;
        assert_eq!(a, b);
        assert_eq!(a.warning(), b.warning());
        assert!(
            a.line().contains("one query -> steel_attention"),
            "{}",
            a.line()
        );
        let generic = AttnRoute::resolve(
            AttnImpl::Generic,
            &config(AttnImpl::Generic, 48),
            DType::F16,
            &dev,
        )?;
        assert_eq!(
            (
                generic.decode_refused,
                generic.prefill_refused,
                generic.warning(),
                generic.prepared.is_none()
            ),
            (None, None, None, true)
        );
        Ok(())
    }
}
