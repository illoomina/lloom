//! The DSpark drafter's paged context KV (`CtxKv::Paged`, issue #708) against
//! its contiguous twin, on a tiny drafter with random weights, Metal only.
//!
//! The claim is that the paged backing changes WHERE the context lives and
//! nothing about what the block attends: the page walk is an address, and the
//! kernel is bit-identical to the contiguous one on the same keys
//! (`lloom/tests/sdpa.rs`). So `block_hidden` must agree BIT FOR BIT between
//! the two arms — across appends that cross page boundaries, a rollback, a
//! reset that recycles pages (so the table is NOT the identity, §10.3h), two
//! sequences sharing one pool, and a grow past the first bucket.
#![cfg(feature = "metal")]

use lloom::models::lfm2_dspark::{CtxKv, DSparkConfig, DSparkDrafter};
use lloom::nn::kv_cache::seam::KvCache;
use lloom::nn::VarBuilder;
use lloom_tensor::{DType, Device, Result, Tensor};
use rand::SeedableRng;
use rand_distr::Distribution;
use std::collections::HashMap;

/// Page size 32: one tiled-kernel key block per page, so every block of every
/// draft is seated on a different page and a context of a few dozen tokens
/// spans several.
const PAGE: usize = 32;

fn tiny_cfg() -> DSparkConfig {
    DSparkConfig {
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 32,
        intermediate_size: 96,
        vocab_size: 64,
        rms_norm_eps: 1e-5,
        rope_theta: 1e4,
        rope_is_neox_style: false,
        block_size: 9,
        mask_token_id: 63,
        target_layer_ids: vec![0, 1],
        num_target_layers: 2,
        markov_rank: 0,
        enable_confidence_head: false,
        logits_start: 0,
    }
}

fn randn(rng: &mut rand::rngs::StdRng, shape: Vec<usize>, dev: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let normal = rand_distr::Normal::new(0.0f32, 0.2).unwrap();
    let vs: Vec<f32> = (0..n).map(|_| normal.sample(rng)).collect();
    Tensor::from_vec(vs, shape.as_slice(), dev)
}

/// Every tensor `DSparkDrafter::new` loads, random, at `dtype`.
fn tiny_drafter(dtype: DType, dev: &Device, seed: u64) -> Result<DSparkDrafter> {
    let cfg = tiny_cfg();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let h = cfg.hidden_size;
    let hd = cfg.head_dim;
    let q = cfg.num_attention_heads * hd;
    let kv = cfg.num_key_value_heads * hd;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let mut fill = |name: String, shape: Vec<usize>| -> Result<()> {
        t.insert(name, randn(&mut rng, shape, dev)?.to_dtype(dtype)?);
        Ok(())
    };
    fill("fc.weight".into(), vec![h, cfg.target_layer_ids.len() * h])?;
    fill("hidden_norm.weight".into(), vec![h])?;
    fill("norm.weight".into(), vec![h])?;
    for i in 0..cfg.num_hidden_layers {
        let p = format!("layers.{i}");
        fill(format!("{p}.self_attn.q_proj.weight"), vec![q, h])?;
        fill(format!("{p}.self_attn.k_proj.weight"), vec![kv, h])?;
        fill(format!("{p}.self_attn.v_proj.weight"), vec![kv, h])?;
        fill(format!("{p}.self_attn.o_proj.weight"), vec![h, q])?;
        fill(format!("{p}.self_attn.q_norm.weight"), vec![hd])?;
        fill(format!("{p}.self_attn.k_norm.weight"), vec![hd])?;
        fill(
            format!("{p}.mlp.gate_proj.weight"),
            vec![cfg.intermediate_size, h],
        )?;
        fill(
            format!("{p}.mlp.up_proj.weight"),
            vec![cfg.intermediate_size, h],
        )?;
        fill(
            format!("{p}.mlp.down_proj.weight"),
            vec![h, cfg.intermediate_size],
        )?;
        fill(format!("{p}.input_layernorm.weight"), vec![h])?;
        fill(format!("{p}.post_attention_layernorm.weight"), vec![h])?;
    }
    let vb = VarBuilder::from_tensors(t, dtype, dev);
    DSparkDrafter::new(cfg, 8192, vb)
}

fn bits(t: &Tensor) -> Result<Vec<u32>> {
    let t = t.to_device(&Device::Cpu)?.flatten_all()?;
    Ok(match t.dtype() {
        DType::F32 => t.to_vec1::<f32>()?.into_iter().map(f32::to_bits).collect(),
        DType::F16 => t
            .to_vec1::<half::f16>()?
            .into_iter()
            .map(|x| x.to_bits() as u32)
            .collect(),
        other => panic!("bits: {other:?}"),
    })
}

/// One round on both arms: the same taps keyed at `offset`, the same block
/// drafted at `offset + n`; returns the two `block_hidden`s.
struct Twin {
    contiguous: Vec<lloom::models::lfm2_dspark::CtxCache>,
    paged: Vec<lloom::models::lfm2_dspark::CtxCache>,
    offset: usize,
}

impl Twin {
    fn round(
        &mut self,
        drafter: &DSparkDrafter,
        rng: &mut rand::rngs::StdRng,
        n: usize,
        dtype: DType,
        dev: &Device,
        what: &str,
    ) -> Result<()> {
        let cfg = drafter.config();
        let taps: Vec<Tensor> = (0..cfg.target_layer_ids.len())
            .map(|_| randn(rng, vec![1, n, cfg.hidden_size], dev))
            .collect::<Result<_>>()?;
        drafter.update_context(&taps, self.offset, &mut self.contiguous)?;
        drafter.update_context(&taps, self.offset, &mut self.paged)?;
        self.offset += n;
        assert_eq!(
            self.contiguous[0].len(),
            self.offset,
            "{what}: contiguous len"
        );
        assert_eq!(self.paged[0].len(), self.offset, "{what}: paged len");
        let block = randn(rng, vec![1, cfg.block_size, cfg.hidden_size], dev)?.to_dtype(dtype)?;
        let hc = drafter.block_hidden(&block, self.offset, &self.contiguous)?;
        let hp = drafter.block_hidden(&block, self.offset, &self.paged)?;
        let (bc, bp) = (bits(&hc)?, bits(&hp)?);
        let distinct = bc.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(distinct > 8, "{what}: vacuous ({distinct} distinct values)");
        assert!(
            bc == bp,
            "{what}: block_hidden differs between the contiguous and paged arms"
        );
        println!(
            "{what}: ctx {} -> block_hidden bit-identical ({} values)",
            self.offset,
            bc.len()
        );
        Ok(())
    }
}

fn run_arms(dtype: DType) -> Result<()> {
    let dev = Device::new_metal(0)?;
    let drafter = tiny_drafter(dtype, &dev, 708)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(708_001);
    let mut twin = Twin {
        contiguous: drafter
            .make_ctx_caches_for(CtxKv::Contiguous, 1, &dev)?
            .remove(0),
        paged: drafter
            .make_ctx_caches_for(
                CtxKv::Paged {
                    page_size: PAGE,
                    shared: false,
                },
                1,
                &dev,
            )?
            .remove(0),
        offset: 0,
    };
    assert_eq!(
        twin.paged[0].kind(),
        CtxKv::Paged {
            page_size: PAGE,
            shared: false
        }
    );

    // Appends that land mid-page, on a boundary, and across several pages.
    for (i, n) in [5usize, 27, 1, 40, 3, 70].into_iter().enumerate() {
        twin.round(
            &drafter,
            &mut rng,
            n,
            dtype,
            &dev,
            &format!("{dtype:?} round {i} (+{n})"),
        )?;
    }
    let table = twin.paged[0].page_table().expect("paged");
    assert!(
        table.len() >= 5,
        "146 positions + 9 reserved need >= 5 pages of 32, got {table:?}"
    );

    // A rollback across a page boundary (146 -> 100), then another round.
    for c in twin.contiguous.iter_mut().chain(twin.paged.iter_mut()) {
        KvCache::rollback(c, 100)?;
    }
    twin.offset = 100;
    assert_eq!(twin.paged[0].len(), 100);
    twin.round(
        &drafter,
        &mut rng,
        33,
        dtype,
        &dev,
        &format!("{dtype:?} after rollback (+33)"),
    )?;

    // Reset both and re-seed: the released pages sit in the quarantine until
    // the epoch they were read in completes, so the re-seed allocates past
    // them (or recycles them, in a different order) — either way the table is
    // not the identity, the fixture §10.3h says a fresh pool cannot produce.
    for c in twin.contiguous.iter_mut().chain(twin.paged.iter_mut()) {
        KvCache::reset(c);
    }
    assert!(twin.paged[0].is_empty());
    dev.synchronize()?;
    twin.offset = 0;
    twin.round(
        &drafter,
        &mut rng,
        50,
        dtype,
        &dev,
        &format!("{dtype:?} after reset (+50)"),
    )?;
    let table = twin.paged[0].page_table().expect("paged");
    assert!(
        table.iter().enumerate().any(|(i, &p)| p as usize != i),
        "after a reset the recycled table should not be the identity, got {table:?}"
    );
    println!("{dtype:?}: post-reset table {table:?}");
    twin.round(
        &drafter,
        &mut rng,
        90,
        dtype,
        &dev,
        &format!("{dtype:?} after reset (+90)"),
    )?;

    // Past the first bucket: the pool grows (a wider buffer, the old contents
    // in its prefix, every page index unchanged) and the answer does not move.
    twin.round(
        &drafter,
        &mut rng,
        4100,
        dtype,
        &dev,
        &format!("{dtype:?} grow (+4100)"),
    )?;
    twin.round(
        &drafter,
        &mut rng,
        7,
        dtype,
        &dev,
        &format!("{dtype:?} after grow (+7)"),
    )?;
    Ok(())
}

#[test]
fn paged_ctx_matches_contiguous_bit_for_bit_f32() -> Result<()> {
    run_arms(DType::F32)
}

#[test]
fn paged_ctx_matches_contiguous_bit_for_bit_f16() -> Result<()> {
    run_arms(DType::F16)
}

/// Two sequences over one pool set, their contexts interleaved so their pages
/// alternate in the pool: each is bit-identical to its own contiguous twin.
#[test]
fn two_sequences_share_one_pool_and_read_their_own_pages() -> Result<()> {
    let dtype = DType::F32;
    let dev = Device::new_metal(0)?;
    let drafter = tiny_drafter(dtype, &dev, 709)?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(709_001);
    let mut paged = drafter.make_ctx_caches_for(
        CtxKv::Paged {
            page_size: PAGE,
            shared: true,
        },
        2,
        &dev,
    )?;
    let mut twins: Vec<Twin> = paged
        .drain(..)
        .map(|p| Twin {
            contiguous: drafter.make_ctx_caches(),
            paged: p,
            offset: 0,
        })
        .collect();
    for (i, n) in [20usize, 45, 13, 70, 33].into_iter().enumerate() {
        for (s, twin) in twins.iter_mut().enumerate() {
            twin.round(
                &drafter,
                &mut rng,
                n + s * 7,
                dtype,
                &dev,
                &format!("seq {s} round {i}"),
            )?;
        }
    }
    let t0 = twins[0].paged[0].page_table().unwrap();
    let t1 = twins[1].paged[0].page_table().unwrap();
    assert!(
        t0.iter().all(|p| !t1.contains(p)),
        "the two sequences must not share a page: {t0:?} {t1:?}"
    );
    assert!(
        t0.windows(2).any(|w| w[1] != w[0] + 1),
        "sequence 0's pages should be interleaved with sequence 1's, got {t0:?}"
    );
    println!("tables: seq0 {t0:?} seq1 {t1:?}");
    Ok(())
}

/// The refusals: a page size the kernel cannot walk, and the paged arm on the
/// CPU.
#[test]
fn paged_ctx_refuses_a_bad_page_size_and_the_cpu() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let drafter = tiny_drafter(DType::F32, &dev, 710)?;
    let paged = |page_size, shared| CtxKv::Paged { page_size, shared };
    assert!(drafter
        .make_ctx_caches_for(paged(48, false), 1, &dev)
        .is_err());
    assert!(drafter
        .make_ctx_caches_for(paged(32, false), 1, &Device::Cpu)
        .is_err());
    assert_eq!(
        CtxKv::parse("paged")?,
        paged(CtxKv::DEFAULT_PAGE_SIZE, false)
    );
    assert_eq!(CtxKv::parse("paged:64")?, paged(64, false));
    assert_eq!(CtxKv::parse("paged:shared")?, paged(256, true));
    assert_eq!(CtxKv::parse("paged:32:shared")?, paged(32, true));
    assert_eq!(CtxKv::parse("paged:32:shared")?.name(), "paged:32:shared");
    assert_eq!(CtxKv::parse("contiguous")?, CtxKv::Contiguous);
    assert!(CtxKv::parse("pagd").is_err());
    Ok(())
}
