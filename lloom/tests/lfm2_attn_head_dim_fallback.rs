//! LFM2's attention on a head dim a fused kernel is not instantiated at
//! (`models::lfm2::AttnRoute`): the phase takes the composed attention,
//! decided once at load, with one warning naming the head dim, the kernel
//! requested and the kernel taken — on decode as on prefill, on the
//! contiguous cache, the private page pool and a shared pool's rows.
//!
//! The reference is the CPU backend at f32 over the same f16-rounded tiny
//! weights (`lloom::testing::tiny`), so each arm is held to an absolute
//! reference rather than to another arm (CONTRIBUTING §3.1b). The census is
//! read through the device's own window (`invariants.md` §15.1n). An
//! instantiated head dim on each arm is the control: its fused kernel is in
//! the census, so a census that could not see the kernel would fail there.

use lloom::models::lfm2::{AttnImpl, Config, DecodeAttn, Model, PrefillAttn};
use lloom::models::lfm2_testing::Lfm2Tiny;
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

fn build(cfg: &Config, dev: &Device, dtype: DType) -> Result<Model> {
    TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default())
        .with_dtype(dtype)
        .model(dev)
}

/// On the CPU the fused arms do not exist, whatever the head dim: every phase
/// is `Generic`, nothing was refused, and nothing is printed — the device,
/// not the head dim, is why.
#[test]
fn the_cpu_takes_the_composed_attention_without_a_warning() -> Result<()> {
    for arm in [AttnImpl::Sdpa, AttnImpl::FlashDecoding] {
        let cfg = config(arm, 48);
        let model = build(&cfg, &Device::Cpu, DType::F32)?;
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
    use lloom::models::lfm2::{AttnRoute, BatchPool, Cache, ConvState, KvAppend};
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
    fn cpu_model(cfg: &Config, dtype: DType) -> Result<Model> {
        let dense = if dtype == DType::F16 {
            GgmlDType::F16
        } else {
            GgmlDType::F32
        };
        TinyCheckpoint::<Lfm2Tiny>::new(
            cpu_config(cfg),
            Weights {
                dense,
                ..Weights::default()
            },
        )
        .with_dtype(DType::F32)
        .model(&Device::Cpu)
    }

    fn logits(t: &Tensor) -> Result<Vec<f32>> {
        t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()
    }

    /// A five-token prefill, two decode steps, a three-token second prefill onto
    /// the cache (§10.9) and one more decode step: the logits of each pass.
    const PASSES: [&[u32]; 5] = [&[1, 2, 3, 4, 5], &[7], &[9], &[11, 13, 17], &[19]];

    fn passes(model: &Model, cache: &mut Cache, dev: &Device) -> Result<Vec<Vec<f32>>> {
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
    /// largest magnitude.
    fn rel(got: &[f32], want: &[f32]) -> f32 {
        let scale = want.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
        let d = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        d / scale
    }

    /// f16 activations on Metal against the f32 reference over the same rounded
    /// weights. Measured at most 5.1e-3, on the fused and the composed arms
    /// alike (head_dim 256); a wrong route or a wrong read is far outside it.
    const TOL: f32 = 1e-2;

    fn assert_close(what: &str, got: &[Vec<f32>], want: &[Vec<f32>]) {
        assert_eq!(got.len(), want.len(), "{what}: pass count");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let r = rel(g, w);
            eprintln!("{what}: pass {i}: |metal - cpu| / scale = {r:e}");
            assert!(
                r.is_finite() && r <= TOL,
                "{what}: pass {i} is {r:e} from the CPU (tolerance {TOL})"
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
        model: Model,
        logits: Vec<Vec<f32>>,
        decode_census: Vec<(String, u64)>,
        prefill_census: Vec<(String, u64)>,
    }

    fn run_on_metal(cfg: &Config, dev: &Device) -> Result<Run> {
        let model = build(cfg, dev, DType::F16)?;
        let mut cache = Cache::new(true, DType::F16, cfg, dev)?;
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
            model,
            logits: out,
            decode_census: decode,
            prefill_census: prefill,
        })
    }

    fn reference(cfg: &Config) -> Result<Vec<Vec<f32>>> {
        let cpu = cpu_config(cfg);
        let model = cpu_model(cfg, DType::F16)?;
        let mut cache = Cache::new(true, DType::F32, &cpu, &Device::Cpu)?;
        passes(&model, &mut cache, &Device::Cpu)
    }

    /// The route an arm must take at `head_dim` on f16, by the kernels'
    /// instantiated sets, with the kernel the head dim refused.
    fn expected(
        arm: AttnImpl,
        head_dim: usize,
    ) -> (
        DecodeAttn,
        Option<DecodeAttn>,
        PrefillAttn,
        Option<PrefillAttn>,
    ) {
        let wanted = match arm {
            AttnImpl::Sdpa => DecodeAttn::SdpaVector,
            _ => DecodeAttn::FlashDecoding,
        };
        let decode_fits = match arm {
            AttnImpl::Sdpa => [32, 64, 96, 128, 256, 512].contains(&head_dim),
            _ => [32, 64, 96, 128].contains(&head_dim),
        };
        let steel_fits = [32, 64, 72, 80, 96, 128, 256, 512].contains(&head_dim);
        let (d, dr) = if decode_fits {
            (wanted, None)
        } else {
            (DecodeAttn::Generic, Some(wanted))
        };
        let (p, pr) = if steel_fits {
            (PrefillAttn::Steel, None)
        } else {
            (PrefillAttn::Generic, Some(PrefillAttn::Steel))
        };
        (d, dr, p, pr)
    }

    /// Each arm across head dims inside and outside its kernels' sets: the
    /// model builds, every pass agrees with the CPU, the route is the one
    /// the sets give, the census shows the kernel the route names and none
    /// it refused, and a load that fell back printed exactly one warning
    /// naming the head dim, the kernel requested and the kernel taken.
    #[test]
    fn off_set_head_dims_take_the_composed_attention_and_match_the_cpu() -> Result<()> {
        profile_on();
        let dev = Device::new_metal(0)?;
        for arm in [AttnImpl::Sdpa, AttnImpl::FlashDecoding] {
            for head_dim in [16, 48, 64, 72, 80, 256] {
                let what = format!("{arm:?} head_dim {head_dim}");
                let cfg = config(arm, head_dim);
                let run = run_on_metal(&cfg, &dev)?;
                assert_close(&what, &run.logits, &reference(&cfg)?);

                let route = run.model.attn_route();
                let (d, dr, p, pr) = expected(arm, head_dim);
                assert_eq!(
                    (
                        route.decode,
                        route.decode_refused,
                        route.prefill,
                        route.prefill_refused
                    ),
                    (d, dr, p, pr),
                    "{what}: the route"
                );

                let (vec_d, flash_d, steel_d, soft_d) = attention_labels(&run.decode_census);
                let (vec_p, flash_p, steel_p, soft_p) = attention_labels(&run.prefill_census);
                eprintln!(
                    "{what}: decode sdpa_vector {vec_d} flash {flash_d} steel {steel_d} softmax {soft_d}; \
                     prefill sdpa_vector {vec_p} flash {flash_p} steel {steel_p} softmax {soft_p}"
                );
                // Two attention layers, three decode steps.
                let (want_vec, want_flash, want_soft) = match d {
                    DecodeAttn::SdpaVector => (6, 0, 0),
                    DecodeAttn::FlashDecoding => (0, 6, 0),
                    DecodeAttn::Generic => (0, 0, 6),
                };
                assert_eq!(
                    (vec_d, flash_d, steel_d, soft_d),
                    (want_vec, want_flash, 0, want_soft),
                    "{what}: the decode census"
                );
                let (want_steel, want_soft) = match p {
                    PrefillAttn::Steel => (2, 0),
                    PrefillAttn::Generic => (0, 2),
                };
                assert_eq!(
                    (vec_p, flash_p, steel_p, soft_p),
                    (0, 0, want_steel, want_soft),
                    "{what}: the second prefill's census"
                );

                let warnings = run.model.load_warnings();
                if dr.is_none() && pr.is_none() {
                    assert!(warnings.is_empty(), "{what}: {warnings:?}");
                    continue;
                }
                assert_eq!(
                    warnings.len(),
                    1,
                    "{what}: one warning per load: {warnings:?}"
                );
                let w = &warnings[0];
                assert!(w.contains(&format!("head_dim {head_dim} ")), "{what}: {w}");
                assert!(w.contains(&format!("{arm:?}")), "{what}: {w}");
                if let Some(k) = dr {
                    let requested = match k {
                        DecodeAttn::SdpaVector => "decode requested sdpa_vector and takes generic",
                        _ => "decode requested flash_decoding and takes generic",
                    };
                    assert!(w.contains(requested), "{what}: {w}");
                } else {
                    assert!(!w.contains("decode"), "{what}: {w}");
                }
                assert_eq!(
                    w.contains("prefill requested steel_attention and takes generic"),
                    pr.is_some(),
                    "{what}: {w}"
                );
                assert_eq!(
                    Some(w.clone()),
                    route.warning(),
                    "{what}: the printed line is the route's"
                );
            }
        }
        Ok(())
    }

    /// f32 has no 512 in either `sdpa_vector` or `steel_attention`: the
    /// dtype's half of the check, on its own. A route resolved without the
    /// device's dtype would take the fused kernels here.
    #[test]
    fn head_dim_512_at_f32_takes_the_composed_attention() -> Result<()> {
        profile_on();
        let dev = Device::new_metal(0)?;
        let cfg = config(AttnImpl::Sdpa, 512);
        let model = build(&cfg, &dev, DType::F32)?;
        let route = model.attn_route();
        assert_eq!(
            (
                route.decode,
                route.decode_refused,
                route.prefill,
                route.prefill_refused
            ),
            (
                DecodeAttn::Generic,
                Some(DecodeAttn::SdpaVector),
                PrefillAttn::Generic,
                Some(PrefillAttn::Steel)
            )
        );
        assert_eq!(
            model.load_warnings().len(),
            1,
            "{:?}",
            model.load_warnings()
        );
        let mut cache = Cache::new(true, DType::F32, &cfg, &dev)?;
        let got = passes(&model, &mut cache, &dev)?;
        let cpu = cpu_config(&cfg);
        let reference_model = cpu_model(&cfg, DType::F32)?;
        let mut cpu_cache = Cache::new(true, DType::F32, &cpu, &Device::Cpu)?;
        let want = passes(&reference_model, &mut cpu_cache, &Device::Cpu)?;
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let r = rel(g, w);
            eprintln!("Sdpa head_dim 512 f32: pass {i}: {r:e}");
            assert!(r <= 1e-4, "pass {i} is {r:e} from the CPU");
        }
        // At f16 the same head dim is inside both sets.
        let f16 = build(&cfg, &dev, DType::F16)?.attn_route();
        assert_eq!(
            (f16.decode, f16.prefill),
            (DecodeAttn::SdpaVector, PrefillAttn::Steel)
        );
        Ok(())
    }

    /// The private page pool (`KvAppend::Paged`, which requires the flash
    /// arm): at a head dim FlashDecoding does not instantiate, a decode step
    /// attends over the keys gathered from the pool. 256 keeps the steel
    /// prefill; 48 loses it too.
    #[test]
    fn the_page_pool_decodes_an_off_set_head_dim_over_a_gathered_copy() -> Result<()> {
        profile_on();
        let dev = Device::new_metal(0)?;
        for head_dim in [48, 256, 64] {
            let mut cfg = config(AttnImpl::FlashDecoding, head_dim);
            cfg.kv_append = KvAppend::Paged;
            let what = format!("Paged head_dim {head_dim}");
            let run = run_on_metal(&cfg, &dev)?;
            assert_close(&what, &run.logits, &reference(&cfg)?);
            let (_, flash_d, _, soft_d) = attention_labels(&run.decode_census);
            let fits = head_dim == 64;
            assert_eq!(
                (flash_d > 0, soft_d > 0),
                (fits, !fits),
                "{what}: {:?}",
                run.decode_census
            );
            assert_eq!(
                run.model.load_warnings().len(),
                usize::from(!fits),
                "{what}"
            );
        }
        Ok(())
    }

    /// A shared pool's rows (`BatchPool`, the `--preset pool` arms): at an
    /// off-set head dim the batched arms decline — their paged kernels are
    /// FlashDecoding and the paged steel read — and every row decodes alone
    /// over its gathered keys, each agreeing with its own CPU forward. At an
    /// instantiated head dim the batched arms run: the control.
    #[test]
    fn a_shared_pool_decodes_an_off_set_head_dim_row_by_row() -> Result<()> {
        profile_on();
        let dev = Device::new_metal(0)?;
        let rows: [(&[u32], u32); 3] = [
            (&[3, 11, 20, 5, 9], 12),
            (&[40, 2], 33),
            (&[1, 4, 9, 16, 25, 36, 49, 8], 21),
        ];
        for head_dim in [48, 64] {
            let mut cfg = config(AttnImpl::FlashDecoding, head_dim);
            cfg.kv_append = KvAppend::Paged;
            cfg.conv_state = ConvState::RotatingRing { k: 2 };
            cfg.conv_fused = true;
            cfg.flash_page_size = 32;
            cfg.flash_pages_per_chunk = 1;
            let what = format!("pool head_dim {head_dim}");
            let model = build(&cfg, &dev, DType::F16)?;
            let pool = BatchPool::new(&cfg, &dev, 4, 64)?;
            pool.set_batched_arms(true);
            let mut caches = Vec::new();
            for (prompt, _) in rows {
                let mut cache = Cache::for_batch(true, DType::F16, &cfg, &dev, &pool)?;
                let inp = Tensor::new(prompt, &dev)?.unsqueeze(0)?;
                model.forward(&inp, 0, &mut cache)?;
                caches.push(cache);
            }
            let nexts: Vec<u32> = rows.iter().map(|(_, n)| *n).collect();
            let inp = Tensor::new(nexts.as_slice(), &dev)?.reshape((rows.len(), 1))?;
            let mut refs: Vec<Row<'_, Cache>> = caches
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
            // FlashDecoding dispatch per layer; the per-row fallback is one
            // composed attention per row per layer (three rows, two layers).
            let (_, flash, _, soft) = attention_labels(&census);
            eprintln!("{what}: forward_rows flash {flash} softmax {soft}");
            let fits = head_dim == 64;
            assert_eq!(
                (flash > 0, soft),
                (fits, if fits { 0 } else { 6 }),
                "{what}: the batched attention arm runs iff the head dim fits: {census:?}"
            );

            let cpu = cpu_config(&cfg);
            let reference_model = cpu_model(&cfg, DType::F16)?;
            for (r, (prompt, next)) in rows.iter().enumerate() {
                let mut cache = Cache::new(true, DType::F32, &cpu, &Device::Cpu)?;
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
        let a = AttnRoute::resolve(AttnImpl::FlashDecoding, 48, DType::F16, &dev);
        let b = AttnRoute::resolve(AttnImpl::FlashDecoding, 48, DType::F16, &dev);
        assert_eq!(a, b);
        assert_eq!(a.warning(), b.warning());
        let generic = AttnRoute::resolve(AttnImpl::Generic, 48, DType::F16, &dev);
        assert_eq!(
            (
                generic.decode_refused,
                generic.prefill_refused,
                generic.warning()
            ),
            (None, None, None)
        );
        Ok(())
    }
}
