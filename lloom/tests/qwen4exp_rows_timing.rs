//! Each block at the real geometry with the checkpoint's weight kinds (Q4_K
//! stems, f16 HC low-rank and gate stems, Q8_0 experts), timed at 1 and 4
//! rows: which term of the batched step does not amortize. A measurement
//! (ignored; run by hand with `--ignored --nocapture` under `lloom-arb
//! measure`).
#![cfg(feature = "metal")]
use lloom::models::qwen4exp::{
    testing, Cache, Config, GatedDeltaNet, HyperConnection, KvCache, KvPool, Layer, Mixer,
    MoeLayer, QsaAttention, Resident,
};
use lloom::nn::{RecurrentState, RmsNorm};
use lloom_tensor::quantized::{GgmlDType, QMatMul, QTensor};
use lloom_tensor::{DType, Device, Module, Tensor};
use std::sync::Arc;
use std::time::Instant;

fn q4(out: usize, inp: usize, dev: &Device) -> QMatMul {
    let w = Tensor::randn(0f32, 0.05, (out, inp), dev).unwrap();
    let dt = if inp.is_multiple_of(256) {
        GgmlDType::Q4K
    } else {
        GgmlDType::Q8_0
    };
    QMatMul::from_qtensor(QTensor::quantize(&w, dt).unwrap()).unwrap()
}
fn f16w(out: usize, inp: usize, dev: &Device) -> QMatMul {
    let w = Tensor::randn(0f32, 0.05, (out, inp), dev).unwrap();
    QMatMul::TensorF16(w.to_dtype(DType::F16).unwrap())
}
fn ones(n: usize, dev: &Device) -> Tensor {
    Tensor::ones(n, DType::F32, dev).unwrap()
}

fn time(dev: &Device, steps: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..3 {
        f();
    }
    dev.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..steps {
        f();
    }
    dev.synchronize().unwrap();
    t.elapsed().as_secs_f64() * 1e3 / steps as f64
}

#[test]
#[ignore]
fn blocks_at_rows_1_and_4() {
    let dev = Device::new_metal(0).unwrap();
    let mut cfg = Config::flash_next();
    cfg.n_experts = 64;
    let d = cfg.hidden_size;
    let nd = cfg.hc_dim();
    let cc = cfg.conv_channels();
    let vd = cfg.ssm_state_size * cfg.ssm_dt_rank;
    let hv = cfg.ssm_dt_rank;
    let (hd, nh, nkv) = (cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let hc = || {
        HyperConnection::new(
            ones(nd, &dev),
            f16w(cfg.hc_lowrank, nd, &dev),
            f16w(nd, cfg.hc_lowrank, &dev),
            Some(f16w(cfg.hc_count, nd, &dev)),
            cfg.hc_count,
            d,
            cfg.norm_eps as f32,
        )
        .unwrap()
    };
    let dn = GatedDeltaNet::new(
        q4(cc, d, &dev),
        q4(vd, d, &dev),
        Tensor::randn(0f32, 0.2, (cc, cfg.ssm_conv_kernel), &dev).unwrap(),
        Tensor::full(-0.5f32, hv, &dev).unwrap(),
        Tensor::zeros(hv, DType::F32, &dev).unwrap(),
        f16w(hv, d, &dev),
        f16w(hv, d, &dev),
        ones(cfg.ssm_state_size, &dev),
        q4(d, vd, &dev),
        &cfg,
    );
    let qsa = QsaAttention::new(
        q4(nh * hd * 2, d, &dev),
        q4(nkv * hd, d, &dev),
        q4(nkv * hd, d, &dev),
        q4(d, nh * hd, &dev),
        RmsNorm::new(ones(hd, &dev), cfg.norm_eps),
        RmsNorm::new(ones(hd, &dev), cfg.norm_eps),
        &cfg,
    );
    let moe = |seed: usize| {
        let [g, u, dn_w] = testing::expert_stacks(&cfg, seed, &dev, GgmlDType::Q8_0);
        MoeLayer::new(
            q4(cfg.n_experts, d, &dev),
            Resident::new(Arc::new(g), Arc::new(u), Arc::new(dn_w)),
            q4(cfg.shared_expert_intermediate_size, d, &dev),
            q4(cfg.shared_expert_intermediate_size, d, &dev),
            q4(d, cfg.shared_expert_intermediate_size, &dev),
            QMatMul::Tensor(Tensor::randn(0f32, 0.05, (1, d), &dev).unwrap()),
            &cfg,
        )
    };
    let layer = Layer::new(hc(), Mixer::Linear(dn), hc(), moe(0));
    let (hc_a, moe_a) = (hc(), moe(1));
    let steps = 20;
    for rows in [1usize, 4, 8] {
        let h = Tensor::randn(0f32, 0.3, (rows, 1, nd), &dev).unwrap();
        let xin = Tensor::randn(0f32, 0.3, (rows, 1, d), &dev).unwrap();
        let t_hc = time(&dev, steps, || {
            let (m, i) = hc_a.mix(&h).unwrap();
            let _ = hc_a.combine(&h, &m, &i).unwrap();
        });
        let dn = match layer.mixer() {
            Mixer::Linear(dn) => dn,
            _ => unreachable!(),
        };
        let mut s: Vec<(RecurrentState, RecurrentState)> = (0..rows)
            .map(|_| (RecurrentState::new(1), RecurrentState::new(1)))
            .collect();
        // Under `CANDLE_METAL_PROFILE=1 CANDLE_METAL_NODES_PER_CB=1` every
        // dispatch is its own command buffer, so the profiler's per-label
        // GPU time is per kernel: the step's three kernels apart from the
        // block's stems (P29 (6)).
        lloom_tensor::kernels::metal::profile::reset();
        let t_dn = time(&dev, steps, || {
            let mut st: Vec<(&mut RecurrentState, &mut RecurrentState)> =
                s.iter_mut().map(|(a, b)| (a, b)).collect();
            let _ = dn.decode_rows(&xin, 0, &mut st).unwrap();
        });
        if lloom_tensor::kernels::metal::profile::enabled() {
            for k in lloom_tensor::kernels::metal::profile::gpu_time_by_label_detailed() {
                if k.label.contains("gdn") || k.label.contains("mul_mv") {
                    println!(
                        "DNKERNEL rows={rows} {} buffers={} us_per_step={:.1}",
                        k.label,
                        k.buffers,
                        k.total_s * 1e6 / (steps + 3) as f64
                    );
                }
            }
        }
        let mut kvs: Vec<KvCache> = (0..rows).map(|_| KvCache::default()).collect();
        for _ in 0..200 {
            let mut refs: Vec<&mut KvCache> = kvs.iter_mut().collect();
            let _ = qsa.decode_rows(&xin, &mut refs).unwrap();
        }
        let t_qsa = time(&dev, steps, || {
            let mut refs: Vec<&mut KvCache> = kvs.iter_mut().collect();
            let _ = qsa.decode_rows(&xin, &mut refs).unwrap();
        });
        // The same rows in one KV page pool (the one-launch arm).
        let pool = KvPool::new(rows, 1, nkv, hd, DType::F32, &dev);
        let leases: Vec<_> = (0..rows).map(|_| pool.lease().unwrap()).collect();
        let mut pkvs: Vec<KvCache> = leases
            .iter()
            .map(|l| KvCache::in_pool(&pool, l, 0))
            .collect();
        for _ in 0..200 {
            let mut refs: Vec<&mut KvCache> = pkvs.iter_mut().collect();
            let _ = qsa.decode_rows(&xin, &mut refs).unwrap();
        }
        let t_qsa_pool = time(&dev, steps, || {
            let mut refs: Vec<&mut KvCache> = pkvs.iter_mut().collect();
            let _ = qsa.decode_rows(&xin, &mut refs).unwrap();
        });
        // The same at ~1800 keys (still within the dense range): the per-row
        // vector kernel is `nh` threadgroups per launch; the pool's one
        // launch is `rows · nh`.
        for _ in 0..1600 {
            let mut refs: Vec<&mut KvCache> = kvs.iter_mut().collect();
            let _ = qsa.decode_rows(&xin, &mut refs).unwrap();
            let mut refs: Vec<&mut KvCache> = pkvs.iter_mut().collect();
            let _ = qsa.decode_rows(&xin, &mut refs).unwrap();
        }
        let t_qsa_long = time(&dev, steps, || {
            let mut refs: Vec<&mut KvCache> = kvs.iter_mut().collect();
            let _ = qsa.decode_rows(&xin, &mut refs).unwrap();
        });
        let t_qsa_pool_long = time(&dev, steps, || {
            let mut refs: Vec<&mut KvCache> = pkvs.iter_mut().collect();
            let _ = qsa.decode_rows(&xin, &mut refs).unwrap();
        });
        let t_moe = time(&dev, steps, || {
            let _ = moe_a.decode(&xin).unwrap();
        });
        let mut caches: Vec<Cache> = (0..rows).map(|_| Cache::new(&cfg)).collect();
        let t_layer = time(&dev, steps, || {
            let mut refs: Vec<&mut Cache> = caches.iter_mut().collect();
            let _ = layer.decode_parts_rows(&h, 0, &mut refs, false).unwrap();
        });
        println!(
            "ROWTIME rows={rows} hc_bracket={t_hc:.3}ms deltanet={t_dn:.3}ms qsa={t_qsa:.3}ms qsa_pooled={t_qsa_pool:.3}ms qsa_1800={t_qsa_long:.3}ms qsa_pooled_1800={t_qsa_pool_long:.3}ms moe={t_moe:.3}ms linear_layer={t_layer:.3}ms  per_row: hc={:.3} dn={:.3} qsa={:.3} moe={:.3} layer={:.3}",
            t_hc / rows as f64,
            t_dn / rows as f64,
            t_qsa / rows as f64,
            t_moe / rows as f64,
            t_layer / rows as f64
        );
    }
}

/// A 512-position prefill tile through each block at the real geometry
/// (64 Q8_0 experts stand in for 512: the routed cost is per (row, expert)
/// pair, not per expert count): where the tile's GPU time goes, the MoE
/// split into the router (`route_ids`: gemm + top-k) and the rest. The
/// command-buffer-granular profile cannot see inside a layer (a buffer's
/// label is its last dispatch); this can.
#[test]
#[ignore]
fn blocks_at_seq_512() {
    let dev = Device::new_metal(0).unwrap();
    let mut cfg = Config::flash_next();
    cfg.n_experts = 64;
    let d = cfg.hidden_size;
    let nd = cfg.hc_dim();
    let cc = cfg.conv_channels();
    let vd = cfg.ssm_state_size * cfg.ssm_dt_rank;
    let hv = cfg.ssm_dt_rank;
    let (hd, nh, nkv) = (cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let hc = || {
        HyperConnection::new(
            ones(nd, &dev),
            f16w(cfg.hc_lowrank, nd, &dev),
            f16w(nd, cfg.hc_lowrank, &dev),
            Some(f16w(cfg.hc_count, nd, &dev)),
            cfg.hc_count,
            d,
            cfg.norm_eps as f32,
        )
        .unwrap()
    };
    let dn = GatedDeltaNet::new(
        q4(cc, d, &dev),
        q4(vd, d, &dev),
        Tensor::randn(0f32, 0.2, (cc, cfg.ssm_conv_kernel), &dev).unwrap(),
        Tensor::full(-0.5f32, hv, &dev).unwrap(),
        Tensor::zeros(hv, DType::F32, &dev).unwrap(),
        f16w(hv, d, &dev),
        f16w(hv, d, &dev),
        ones(cfg.ssm_state_size, &dev),
        q4(d, vd, &dev),
        &cfg,
    );
    let qsa = QsaAttention::new(
        q4(nh * hd * 2, d, &dev),
        q4(nkv * hd, d, &dev),
        q4(nkv * hd, d, &dev),
        q4(d, nh * hd, &dev),
        RmsNorm::new(ones(hd, &dev), cfg.norm_eps),
        RmsNorm::new(ones(hd, &dev), cfg.norm_eps),
        &cfg,
    );
    let [g, u, dn_w] = testing::expert_stacks(&cfg, 0, &dev, GgmlDType::Q8_0);
    let moe = MoeLayer::new(
        q4(cfg.n_experts, d, &dev),
        Resident::new(Arc::new(g), Arc::new(u), Arc::new(dn_w)),
        q4(cfg.shared_expert_intermediate_size, d, &dev),
        q4(cfg.shared_expert_intermediate_size, d, &dev),
        q4(d, cfg.shared_expert_intermediate_size, &dev),
        QMatMul::Tensor(Tensor::randn(0f32, 0.05, (1, d), &dev).unwrap()),
        &cfg,
    );
    let hc_a = hc();
    let steps = 10;
    for seq in [512usize, 128] {
        let h = Tensor::randn(0f32, 0.3, (1, seq, nd), &dev).unwrap();
        let xin = Tensor::randn(0f32, 0.3, (1, seq, d), &dev).unwrap();
        let t_hc = time(&dev, steps, || {
            let (m, i) = hc_a.mix(&h).unwrap();
            let _ = hc_a.combine(&h, &m, &i).unwrap();
        });
        let mut s = RecurrentState::new(1);
        let mut c = RecurrentState::new(1);
        let t_dn = time(&dev, steps, || {
            let _ = dn.prefill(&xin, 0, &mut s, &mut c).unwrap();
        });
        let t_qsa = time(&dev, steps, || {
            let mut kv = KvCache::default();
            let _ = qsa.prefill(&xin, &mut kv).unwrap();
        });
        let t_route = time(&dev, steps, || {
            let _ = moe.route_ids(&xin).unwrap();
        });
        let t_moe = time(&dev, steps, || {
            let _ = moe.forward(&xin).unwrap();
        });
        println!(
            "PREFILLTIME seq={seq} hc_bracket={t_hc:.2}ms deltanet_prefill={t_dn:.2}ms qsa_prefill={t_qsa:.2}ms moe_route={t_route:.2}ms moe_forward={t_moe:.2}ms  per_token_ms: hc={:.3} dn={:.3} qsa={:.3} route={:.3} moe={:.3}",
            t_hc / seq as f64,
            t_dn / seq as f64,
            t_qsa / seq as f64,
            t_route / seq as f64,
            t_moe / seq as f64,
        );
    }
}

/// A `[e, out, inp]` expert stack of `dtype` from deterministic bytes (the
/// IQ kinds and Q2_0 have no quantiser; any bytes decode to a finite-enough
/// block for a timing), read through `TensorInfo::read` as the loader does.
fn raw_stack(e: usize, out: usize, inp: usize, dtype: GgmlDType, dev: &Device) -> QTensor {
    use lloom_tensor::quantized::gguf_file::TensorInfo;
    let n = e * out * inp;
    let bytes = n / dtype.block_size() * dtype.type_size();
    let mut x: u32 = 0x9e37_79b9;
    let data: Vec<u8> = (0..bytes)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            // scale bytes (f16) in a sane range: keep the exponent modest
            (x & 0x3f) as u8 | 0x30
        })
        .collect();
    let info = TensorInfo {
        ggml_dtype: dtype,
        shape: lloom_tensor::Shape::from((e, out, inp)),
        offset: 0,
    };
    info.read(&mut std::io::Cursor::new(data), 0, dev).unwrap()
}

/// The MoE tile per weight kind of the checkpoint (IQ3_XXS gate/up, IQ4_XS
/// and Q2_0 down, Q8_0 for reference): the routed GEMM with the rows
/// grouped by expert (`kernel_mul_mm_id_mapped`) against the routed
/// matmul-vec at 512 and 128 rows, 64 experts. The GEMM dequantises every
/// weight of every expert once per 32-row block through the generic
/// `dequantize_*`; the matmul-vec's IQ arms are hand-tuned — which wins is
/// per kind (issue #749 P22).
#[test]
#[ignore]
fn moe_tile_by_weight_kind() {
    moe_tile_by_weight_kind_at(64);
}

/// The same at the checkpoint's 512 experts: ~10 rows an expert at a
/// 512-row tile, the regime the real model runs the GEMM in.
#[test]
#[ignore]
fn moe_tile_at_512_experts() {
    moe_tile_by_weight_kind_at(512);
}

fn moe_tile_by_weight_kind_at(n_experts: usize) {
    use lloom_tensor::quantized::metal::{set_moe_mm_bn, set_moe_mm_from};
    let dev = Device::new_metal(0).unwrap();
    let mut cfg = Config::flash_next();
    cfg.n_experts = n_experts;
    let d = cfg.hidden_size;
    let ff = cfg.moe_intermediate_size;
    let steps = 10;
    // (gate/up kind over k = 2560, down kind over k = 640 — the 256-block
    // kinds cannot pack a 640 row, which is why the checkpoint's down planes
    // are Q2_0 / IQ4_NL).
    let kinds: Vec<(GgmlDType, GgmlDType)> = if n_experts > 64 {
        vec![
            (GgmlDType::IQ3XXS, GgmlDType::IQ4NL),
            (GgmlDType::Q8_0, GgmlDType::Q8_0),
        ]
    } else {
        vec![
            (GgmlDType::IQ3XXS, GgmlDType::IQ4NL),
            (GgmlDType::IQ3S, GgmlDType::Q2_0),
            (GgmlDType::IQ2S, GgmlDType::IQ4NL),
            (GgmlDType::IQ4XS, GgmlDType::Q2_0),
            (GgmlDType::Q8_0, GgmlDType::Q8_0),
        ]
    };
    for (dtype, down) in kinds {
        let moe = MoeLayer::new(
            q4(cfg.n_experts, d, &dev),
            Resident::new(
                Arc::new(raw_stack(cfg.n_experts, ff, d, dtype, &dev)),
                Arc::new(raw_stack(cfg.n_experts, ff, d, dtype, &dev)),
                Arc::new(raw_stack(cfg.n_experts, d, ff, down, &dev)),
            ),
            q4(cfg.shared_expert_intermediate_size, d, &dev),
            q4(cfg.shared_expert_intermediate_size, d, &dev),
            q4(d, cfg.shared_expert_intermediate_size, &dev),
            QMatMul::Tensor(Tensor::randn(0f32, 0.05, (1, d), &dev).unwrap()),
            &cfg,
        );
        for seq in [512usize, 128, 32] {
            let xin = Tensor::randn(0f32, 0.3, (1, seq, d), &dev).unwrap();
            set_moe_mm_from(0);
            let t_mv = time(&dev, steps, || {
                let _ = moe.forward(&xin).unwrap();
            });
            set_moe_mm_from(1);
            set_moe_mm_bn(32);
            let t_mm = time(&dev, steps, || {
                let _ = moe.forward(&xin).unwrap();
            });
            set_moe_mm_bn(16);
            let t_mm16 = time(&dev, steps, || {
                let _ = moe.forward(&xin).unwrap();
            });
            set_moe_mm_bn(0);
            println!(
                "MOEKIND experts={n_experts} gate_up={dtype:?} down={down:?} seq={seq} moe_mv={t_mv:.2}ms moe_mm32={t_mm:.2}ms moe_mm16={t_mm16:.2}ms ratio32={:.2} ratio16={:.2}",
                t_mv / t_mm,
                t_mv / t_mm16
            );
        }
    }
    set_moe_mm_from(lloom_tensor::quantized::metal::DEFAULT_MOE_MM_FROM);
}

/// The DeltaNet prefill's stages at the real geometry (h_v 48, d 128, 8
/// chunks of 64 = a 512-position tile), each on its own tensors: the
/// tri-solve, the three chunk products, and the block's whole `prefill` —
/// where the 17 ms a layer-tile go (the chunk scan replaced the host loop
/// and moved nothing, P24).
#[test]
#[ignore]
fn deltanet_prefill_stages() {
    use lloom::models::qwen4exp::deltanet::unit_lower_inverse;
    let dev = Device::new_metal(0).unwrap();
    let (h_v, nc, cs, d) = (48usize, 8usize, 64usize, 128usize);
    let steps = 20;
    let r =
        |shape: &[usize], seed: f32| Tensor::randn(seed * 0.0 + 0f32, 0.3, shape, &dev).unwrap();
    let a = r(&[h_v, nc, cs, cs], 1.0);
    let k = r(&[h_v, nc, cs, d], 2.0);
    let kb = r(&[h_v, nc, cs, d], 3.0);
    let t = r(&[h_v, nc, cs, cs], 4.0);
    let vb = r(&[h_v, nc, cs, d], 5.0);
    let decay = r(&[h_v, nc, cs, cs], 6.0);
    let k_t = k.transpose(2, 3).unwrap();
    let t_inv = time(&dev, steps, || {
        let _ = unit_lower_inverse(&a).unwrap();
    });
    let t_kkt = time(&dev, steps, || {
        let _ = kb.matmul(&k_t).unwrap().mul(&decay).unwrap();
    });
    let t_tv = time(&dev, steps, || {
        let _ = t.matmul(&vb).unwrap();
    });
    let t_qkt = time(&dev, steps, || {
        let _ = kb.matmul(&k_t).unwrap();
    });
    println!(
        "DNSTAGES tri_solve={t_inv:.2}ms k_beta_kt_decay={t_kkt:.2}ms t_v={t_tv:.2}ms (x2) q_kt={t_qkt:.2}ms"
    );
    // The stems at 512 rows: the quantized GEMM (Q4_K, the checkpoint's
    // stem kind) against an f16 weight through the quantized GEMM's f16
    // arm (P23's route) and through the steel f16 gemm (activations cast).
    let (rows, dd, out_dim) = (512usize, 2560usize, 10240usize);
    let x = Tensor::randn(0f32, 0.3, (1, rows, dd), &dev).unwrap();
    let wq = q4(out_dim, dd, &dev);
    let w16 = f16w(out_dim, dd, &dev);
    let wf = match &w16 {
        QMatMul::TensorF16(t) => t.clone(),
        _ => unreachable!(),
    };
    let t_q4 = time(&dev, steps, || {
        let _ = wq.forward(&x).unwrap();
    });
    let t_f16mm = time(&dev, steps, || {
        let _ = w16.forward(&x).unwrap();
    });
    let x16 = x.to_dtype(DType::F16).unwrap();
    let wt = wf.broadcast_left(1).unwrap().t().unwrap();
    let t_steel = time(&dev, steps, || {
        let _ = x16.matmul(&wt).unwrap();
    });
    println!(
        "DNSTEMS rows={rows} [{out_dim}x{dd}]: q4k_mm={t_q4:.2}ms f16w_quant_mm={t_f16mm:.2}ms steel_f16_gemm={t_steel:.2}ms"
    );
}

/// The checkpoint's own stems, one row, by weight kind: GB/s of the
/// quantized matvec at N = 1 for every (stem, kind) the IQ3_XXS shard holds
/// (up to 8 layers' tensors of a kind cycled, so the weights stream from
/// memory), and the lm_head. GPU time is the profiler's busy union
/// (`CANDLE_METAL_PROFILE=1`), the wall beside it. The census that decides
/// where the decode token's GPU time goes (#749 P29 (3)).
///
/// `QWEN4EXP_GGUF=<shard 1>` names the shard (the ISTA IQ3_XXS by default).
#[test]
#[ignore]
fn stem_mv_by_weight_kind() {
    use lloom_tensor::kernels::metal::profile;
    use lloom_tensor::quantized::gguf_file::Content;
    let dev = Device::new_metal(0).unwrap();
    let path = std::env::var("QWEN4EXP_GGUF").unwrap_or_else(|_| {
        format!(
            "{}/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/2c4721899b4382bd07dfb61ae4fbad90c09caf7d/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
            std::env::var("HOME").unwrap()
        )
    });
    let mut file = std::fs::File::open(&path).unwrap();
    let content = Content::read(&mut file).unwrap();
    let stems = ["attn_qkv", "attn_gate", "ssm_out", "attn_output", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "ffn_gate_inp"];
    // (stem, kind) → tensor names, in layer order.
    let mut groups: std::collections::BTreeMap<(String, String), Vec<String>> = Default::default();
    for (name, info) in &content.tensor_infos {
        for stem in stems {
            if name.starts_with("blk.") && name.ends_with(&format!(".{stem}.weight")) {
                groups
                    .entry((stem.to_string(), format!("{:?}", info.ggml_dtype)))
                    .or_default()
                    .push(name.clone());
            }
        }
    }
    groups.insert(("output".into(), "Q5K".into()), vec!["output.weight".into()]);
    println!("STEMMV shard={path}");
    for ((stem, kind), mut names) in groups {
        names.sort();
        names.truncate(8);
        let mats: Vec<QMatMul> = names
            .iter()
            .map(|n| QMatMul::from_qtensor(content.tensor(&mut file, n, &dev).unwrap()).unwrap())
            .collect();
        let (out, inp) = match &mats[0] {
            QMatMul::QTensor(q) => (q.shape().dims()[0], q.shape().dims()[1]),
            QMatMul::TensorF16(t) | QMatMul::Tensor(t) => (t.dims()[0], t.dims()[1]),
        };
        let bytes: usize = match &mats[0] {
            QMatMul::QTensor(q) => q.storage_size_in_bytes(),
            QMatMul::TensorF16(t) => t.elem_count() * 2,
            QMatMul::Tensor(t) => t.elem_count() * 4,
        };
        let x = Tensor::randn(0f32, 1.0, (1, 1, inp), &dev).unwrap();
        let iters = 16 * mats.len();
        for m in &mats {
            let _ = m.forward(&x).unwrap();
        }
        dev.synchronize().unwrap();
        profile::reset();
        let t = Instant::now();
        for i in 0..iters {
            let _ = mats[i % mats.len()].forward(&x).unwrap();
        }
        dev.synchronize().unwrap();
        let wall_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let gpu_us = if profile::enabled() {
            profile::snapshot().gpu_busy_union_s * 1e6 / iters as f64
        } else {
            f64::NAN
        };
        println!(
            "STEMMV stem={stem} kind={kind} shape=[{out}, {inp}] bytes={bytes} n={} wall_us={wall_us:.1} gpu_us={gpu_us:.1} gb_s={:.0}",
            mats.len(),
            bytes as f64 / gpu_us / 1e3
        );
    }
}

/// The routed experts' kernels on the checkpoint's own expert tensors, one
/// token (10 experts of 512): `gate_up` and `down` through `Resident`
/// (`indexed_moe_forward`, `kernel_mul_mv_id_*`), two layers' tensors
/// cycled (~700 MB, past the system cache). GPU time from the profiler's
/// busy union; GB/s over the ten experts' bytes actually read.
#[test]
#[ignore]
fn routed_mv_id_census() {
    use lloom::nn::expert_stream::{ExpertResidency, Resident};
    use lloom_tensor::kernels::metal::profile;
    use lloom_tensor::quantized::gguf_file::Content;
    let dev = Device::new_metal(0).unwrap();
    let path = std::env::var("QWEN4EXP_GGUF").unwrap_or_else(|_| {
        format!(
            "{}/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/2c4721899b4382bd07dfb61ae4fbad90c09caf7d/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
            std::env::var("HOME").unwrap()
        )
    });
    let mut file = std::fs::File::open(&path).unwrap();
    let content = Content::read(&mut file).unwrap();
    let layers = [0usize, 1];
    let tiers: Vec<(Resident, String, usize, usize, usize)> = layers
        .iter()
        .map(|l| {
            let mut load = |n: &str| Arc::new(content.tensor(&mut file, &format!("blk.{l}.{n}.weight"), &dev).unwrap());
            let (g, u, d) = (load("ffn_gate_exps"), load("ffn_up_exps"), load("ffn_down_exps"));
            let kinds = format!("gate={:?} up={:?} down={:?}", g.dtype(), u.dtype(), d.dtype());
            let n_experts = g.shape().dims()[0];
            let per_expert = (g.storage_size_in_bytes() + u.storage_size_in_bytes() + d.storage_size_in_bytes()) / n_experts;
            let gu_per_expert = (g.storage_size_in_bytes() + u.storage_size_in_bytes()) / n_experts;
            (Resident::new(g, u, d), kinds, n_experts, per_expert, gu_per_expert)
        })
        .collect();
    let hidden = 2560usize;
    let inter = 640usize;
    let topk = 10usize;
    let x = Tensor::randn(0f32, 1.0, (1, 1, hidden), &dev).unwrap();
    let h = Tensor::randn(0f32, 1.0, (1, topk, inter), &dev).unwrap();
    let mut ids_v: Vec<u32> = (0..topk as u32).map(|i| (i * 47 + 3) % 512).collect();
    ids_v.sort_unstable();
    let ids = Tensor::from_vec(ids_v, (1, topk), &dev).unwrap();
    println!("ROUTEDMV shard={path} kinds={} experts={} bytes/expert={}", tiers[0].1, tiers[0].2, tiers[0].3);
    let time = |name: &str, bytes: usize, f: &dyn Fn(usize)| {
        for k in 0..tiers.len() {
            f(k);
        }
        dev.synchronize().unwrap();
        let iters = 32 * tiers.len();
        profile::reset();
        let t = Instant::now();
        for i in 0..iters {
            f(i % tiers.len());
        }
        dev.synchronize().unwrap();
        let wall_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let gpu_us = if profile::enabled() {
            profile::snapshot().gpu_busy_union_s * 1e6 / iters as f64
        } else {
            f64::NAN
        };
        println!(
            "ROUTEDMV {name} bytes_read={bytes} wall_us={wall_us:.1} gpu_us={gpu_us:.1} gb_s={:.0}",
            bytes as f64 / gpu_us / 1e3
        );
    };
    let gu_bytes = tiers[0].4 * topk;
    let d_bytes = (tiers[0].3 - tiers[0].4) * topk;
    time("gate_up(10 experts)", gu_bytes, &|k| {
        let _ = tiers[k].0.gate_up(&x, &ids).unwrap();
    });
    time("down(10 experts)", d_bytes, &|k| {
        let _ = tiers[k].0.down(&h, &ids).unwrap();
    });
}
