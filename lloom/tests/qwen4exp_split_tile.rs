//! A prefill tile the streamed pool splits into groups is computed on the
//! tile's routed arm (`design/docs/expert-streaming.md` §20.15a): the logits
//! do not depend on the pool, and on the tiny model they are the resident
//! model's, bit for bit.
//!
//! The tiny model reaches the routed GEMM only at a geometry the template
//! does not have: the experts' `down` K is 32 (the GEMM reads K in 32-wide
//! blocks; the template's 16 keeps the matmul-vec at every row count), and
//! the indexer's budget covers the prompt, so its attention is the dense
//! range's. So a 100-row tile, over `LLOOM_MOE_MM_FROM`, takes the GEMM on the resident
//! model. With 64 experts the streamed tier splits it into many groups at 40
//! slots and fewer, larger ones at 56: a group choosing its own arm by its
//! rows would compute the two pools' rows on different kernels. The same
//! resident run with the GEMM off moves the logits: the arm is on the path.
//!
//! At scale (`LLOOM_FIXTURE=real|synthetic`, Metal) the resident model is
//! 47 GB and refused at load (`memory.md` §9.5q): 32 and 128 slots against
//! 256 over a 192-row tile. At 32 slots (ten experts a row) every group of
//! the tile is a few rows, far under 64; a wider pool can split a tile only
//! into groups of 64 rows or more, which take the GEMM however their arm is
//! chosen.
#![cfg(feature = "metal")]
use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::models::qwen4exp::{
    Cache, Config, ExpertResidency, LoadOptions, Qwen4Exp, SlotBudget, StreamLoader, StreamOptions,
};
use lloom::nn::model::CausalLM;
use lloom::testing::fixture::Fixture;
use lloom_tensor::quantized::metal::{moe_mm_from, set_moe_mm_from};
use lloom_tensor::{Device, Result, Tensor};

fn stream(slots: usize) -> StreamOptions {
    StreamOptions {
        slots: SlotBudget::Fixed(slots),
        loader: StreamLoader::Pread,
        prefetch: false,
        io_hints: false,
    }
}

/// The prompt's prefill, then `steps` greedy tokens: every step's logits,
/// the first being the prefill's.
fn run<R: ExpertResidency>(
    model: &Qwen4Exp<R>,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    steps: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut cache: Cache = model.new_cache(cfg, dev)?;
    let input = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), dev)?;
    let mut logits = model.forward(&input, 0, &mut cache)?;
    let mut rows = Vec::with_capacity(steps + 1);
    for _ in 0..steps {
        let row: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        let next = row
            .iter()
            .enumerate()
            .fold(
                (0usize, f32::MIN),
                |m, (i, &v)| if v > m.1 { (i, v) } else { m },
            )
            .0 as u32;
        rows.push(row);
        let input = Tensor::from_vec(vec![next], (1, 1), dev)?;
        logits = model.forward(&input, cache.pos, &mut cache)?;
    }
    rows.push(logits.flatten_all()?.to_vec1()?);
    Ok(rows)
}

fn bits(row: &[f32]) -> Vec<u32> {
    row.iter().map(|v| v.to_bits()).collect()
}

/// Bit for bit: a NaN in both is not agreement, and none passes unseen.
fn same(tag: &str, got: &[Vec<f32>], want: &[Vec<f32>]) {
    assert_eq!(got.len(), want.len(), "{tag}: step count");
    for (t, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(bits(a) == bits(b), "{tag}: step {t} logits differ");
    }
}

#[test]
fn a_split_tile_computes_on_the_tiles_arm() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        return Ok(());
    };
    let Some(fx) = Fixture::<Qwen4ExpTiny>::new("qwen4exp_split_tile", Config::tiny())? else {
        return Ok(());
    };
    let tiny = !fx.at_scale();
    let cfg = fx.config(|c| {
        if tiny {
            c.moe_intermediate_size = 32;
            c.indexer_budget = 128;
            c.n_experts = 64;
        }
    });
    let opts = LoadOptions::default();
    let rows = fx.pick(100usize, 192);
    let prompt: Vec<u32> = if tiny {
        (0..rows as u32).map(|i| (i * 37 + 5) % 128).collect()
    } else {
        let base = [3u32, 5, 2, 8, 8, 1, 9, 3, 5];
        (0..rows as u32)
            .map(|i| base[(i % 9) as usize] + 1000 * (i / 9))
            .collect()
    };
    assert!(
        rows >= moe_mm_from() && moe_mm_from() > 0,
        "the tile ({rows} rows) is under LLOOM_MOE_MM_FROM ({}): it takes the matmul-vec whole",
        moe_mm_from()
    );
    let steps = 8;

    // Tiny: the resident model, then the GEMM's presence on its path.
    // At scale: the widest pool of the sweep.
    let (want, pools) = if tiny {
        let resident = fx.resident(&cfg, &dev, &opts)?;
        let want = run(&resident, &cfg, &dev, &prompt, steps)?;
        let from = moe_mm_from();
        set_moe_mm_from(0);
        let matvec = run(&resident, &cfg, &dev, &prompt, steps);
        set_moe_mm_from(from);
        assert!(
            bits(&matvec?[0]) != bits(&want[0]),
            "the resident tile's logits are the same with the GEMM off: the GEMM is not on the path"
        );
        (want, vec![40usize, 56])
    } else {
        let wide = fx.streamed(&cfg, &dev, &opts, &stream(256))?;
        (run(&wide, &cfg, &dev, &prompt, steps)?, vec![32usize, 128])
    };
    assert!(
        want.iter().flatten().all(|v| v.is_finite()),
        "the reference's logits are not all finite"
    );

    let mut splits = Vec::new();
    let mut gots = Vec::new();
    for &slots in &pools {
        let streamed = fx.streamed(&cfg, &dev, &opts, &stream(slots))?;
        let got = run(&streamed, &cfg, &dev, &prompt, steps)?;
        let st = streamed.stream_stats();
        println!(
            "SPLITTILE backend={} rows={rows} slots={slots} plans={} groups={} misses={} evictions={}",
            fx.backend().name(),
            st.plans,
            st.groups,
            st.misses,
            st.evictions
        );
        splits.push(st.groups - st.plans);
        gots.push(got);
    }
    for (slots, got) in pools.iter().zip(&gots).skip(1) {
        same(
            &format!("{slots} slots against {}", pools[0]),
            got,
            &gots[0],
        );
    }
    for (slots, got) in pools.iter().zip(&gots) {
        same(&format!("{slots} slots against the reference"), got, &want);
    }
    assert!(
        splits.iter().all(|&s| s > 0),
        "a pool did not split the tile: extra groups {splits:?}"
    );
    assert!(
        splits.windows(2).all(|w| w[0] != w[1]),
        "the pools split the tile into the same number of groups: {splits:?}"
    );
    Ok(())
}
