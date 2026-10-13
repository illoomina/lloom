//! The generic decoder's state on the ledger (`memory.md` §9.10): a
//! state's KV and recurrent rows are registered with the state and follow
//! what it holds at the passes that allocate, and a pass whose state growth
//! the quota cannot take is refused before anything is written — the
//! growth admission the topology had none of (its KV grew inside the
//! append).
#![cfg(feature = "metal")]

use lloom::models::qwen3_5::testing::Qwen3_5Tiny;
use lloom::nn::model::CausalLM;
use lloom::profile::CrownExact;
use lloom::testing::tiny::{TinyCheckpoint, TinyFamily, Weights};
use lloom_tensor::{Device, Result, Tensor};

fn rows(m: &lloom_tensor::MetalDevice, family: &str) -> (usize, usize) {
    let snap = m.ledger().snapshot();
    let held = |name: String| {
        snap.tenants
            .iter()
            .filter(|t| t.name == name)
            .map(|t| t.held)
            .sum::<usize>()
    };
    (
        held(format!("{family}-kv")),
        held(format!("{family}-state")),
    )
}

#[test]
fn a_decoder_state_is_on_the_ledger_and_its_growth_is_admitted() -> Result<()> {
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
    let Ok(dev @ Device::Metal(_)) = Device::new_metal(0) else {
        return Ok(());
    };
    let Device::Metal(m) = &dev else {
        unreachable!()
    };
    let ckpt = TinyCheckpoint::<Qwen3_5Tiny>::new(Qwen3_5Tiny::template(), Weights::default());
    let model = Qwen3_5Tiny::build::<CrownExact>(&ckpt, &dev)?;
    let family = model.family();
    let ids: Vec<u32> = (0..7).collect();
    let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;

    // Registered with the state, holding nothing before its first pass.
    // The first pass admits the KV slab and every recurrent layer's slab
    // and the spare its decode steps write into; the rows say what was
    // admitted, the state holds the KV and one or both recurrent buffers.
    let mut cache = model.new_cache(model.config(), &dev)?;
    assert_eq!(rows(m, family), (0, 0), "a fresh state holds nothing");
    let admitted = model.grow_bytes(&cache, ids.len());
    assert!(admitted.kv > 0 && admitted.recurrent > 0, "{admitted:?}");
    model.prepare_pass(&mut cache, ids.len())?;
    model.forward(&input, 0, &mut cache)?;
    let step = Tensor::from_vec(vec![3u32], (1, 1), &dev)?;
    model.prepare_pass(&mut cache, 1)?;
    model.forward(&step, ids.len(), &mut cache)?;
    let held = model.held_bytes(&cache);
    let (kv, state) = rows(m, family);
    assert_eq!(kv, admitted.kv, "the KV row is the slab admitted");
    assert_eq!(held.kv, admitted.kv, "and the slab is what the state holds");
    assert_eq!(
        state, admitted.recurrent,
        "the recurrent row is what was admitted"
    );
    assert!(
        held.recurrent == admitted.recurrent || 2 * held.recurrent == admitted.recurrent,
        "the state holds the slabs and their spares, or the slabs alone: {held:?} of {admitted:?}"
    );

    // A quota the weights alone exceed — the free lists the admission gives
    // back first cannot make room — so the next state's first pass, its KV
    // slab and recurrent buffers, is refused before it runs.
    let quota = m.ledger().quota();
    m.ledger().set_quota(1);
    let mut fresh = model.new_cache(model.config(), &dev)?;
    let grow = model.grow_bytes(&fresh, ids.len());
    assert!(grow.total() > 1, "{grow:?}");
    let e = model
        .prepare_pass(&mut fresh, ids.len())
        .expect_err("a growth past the quota is refused");
    eprintln!("REFUSED {:?} grow={}", e.allocation(), grow.total());
    assert_eq!(
        e.allocation(),
        Some(lloom_tensor::memory_admission::AllocationError::Refused {
            requested: grow.total(),
            available: 0,
            domain: lloom_tensor::memory_admission::RefusalDomain::Model,
        }),
        "the growth is refused typed, for its own bytes, against a one-byte quota: {e}"
    );
    assert_eq!(
        model.held_bytes(&fresh).total(),
        0,
        "nothing was written by the refused pass"
    );
    m.ledger().set_quota(quota);
    model.prepare_pass(&mut fresh, ids.len())?;
    Ok(())
}
