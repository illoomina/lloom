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
use lloom_tensor::{DType, Device, Tensor};
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
        let t_dn = time(&dev, steps, || {
            let mut st: Vec<(&mut RecurrentState, &mut RecurrentState)> =
                s.iter_mut().map(|(a, b)| (a, b)).collect();
            let _ = dn.decode_rows(&xin, 0, &mut st).unwrap();
        });
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
