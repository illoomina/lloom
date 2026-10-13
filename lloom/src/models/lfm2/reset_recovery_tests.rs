//! CPU cache metadata/reset controls only: one layer, hidden8, max8 RoPE/KV
//! positions, no weights, model forward, native device or speculative verifier.
use super::*;
use crate::nn::branch::RetainState;
use crate::profile::CrownExact;

fn cache() -> Result<State<CrownExact>> {
    let config = Config {
        vocab_size: 8,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        max_position_embeddings: 8,
        layer_types: vec![LayerType::Conv],
        kv_append: KvAppend::InPlace,
        conv_state: ConvState::RotatingRing { k: 4 },
        conv_l_cache: 3,
        tap_layers: vec![0],
        ..Default::default()
    };
    State::new(true, DType::F32, &config, &Device::Cpu)
}
#[test]
fn reset_closes_abandoned_window_and_discards_undrained_taps() -> Result<()> {
    let mut cache = cache()?;
    let tap = Tensor::zeros((1, 1, 8), DType::F32, &Device::Cpu)?;
    cache.taps.push(Some(tap));
    cache.taps_dropped = 7;
    let capacity = cache.taps.capacity();
    let slots = cache.taps.as_ptr();
    let abandoned = cache.advance(1)?;
    drop(abandoned);
    assert!(cache.speculative);
    assert!(
        cache.retain_state().is_err(),
        "fixture did not open a real window"
    );
    cache.reset();
    assert!(
        !cache.speculative,
        "CacheReset left an abandoned window open"
    );
    assert!(
        cache.taps.iter().all(Option::is_none),
        "reset exposed old pass taps"
    );
    assert_eq!(
        cache.taps.capacity(),
        capacity,
        "reset discarded tap capacity"
    );
    assert_eq!(cache.taps.as_ptr(), slots, "reset replaced tap storage");
    assert_eq!(
        cache.taps_dropped(),
        7,
        "reset changed cumulative probe diagnostics"
    );
    cache.prepare_branching()?;
    cache.retain_state()?;
    Model::<CrownExact>::begin_taps(&mut cache);
    assert_eq!(cache.taps_dropped(), 7, "fresh pass counted abandoned taps");
    assert!(
        cache.take_taps().is_empty(),
        "fresh pass read abandoned probes"
    );
    Ok(())
}
#[test]
fn clear_invalidates_pass_metadata_but_keeps_scratch_policy_and_counters() -> Result<()> {
    let mut cache = cache()?;
    let scratch = Tensor::zeros((1, 8, 3), DType::F32, &Device::Cpu)?;
    let anchor = Tensor::new(&[3u32], &Device::Cpu)?;
    cache.conv_spare[0] = Some(scratch.clone());
    cache.conv_states[0] = Some(scratch.clone());
    cache.anchor_slot = Some(anchor.clone());
    cache.taps.push(Some(scratch.clone()));
    cache.branch_continuation = true;
    cache.paged_pass_pending = true;
    cache.conv_phase = 2;
    cache.conv_phase_prev = 1;
    cache.conv_history = 1;
    cache.conv_compact = true;
    cache.taps_dropped = 7;
    cache.kv_grows = 11;
    let capacity = cache.taps.capacity();
    let abandoned = cache.advance(1)?;
    drop(abandoned);
    cache.clear();
    assert!(!cache.speculative, "clear left an abandoned window open");
    assert!(
        cache.taps.iter().all(Option::is_none),
        "clear retained stale pass probes"
    );
    assert_eq!(cache.taps.capacity(), capacity);
    assert!(cache.conv_states.iter().all(Option::is_none));
    assert_eq!(
        (cache.conv_phase, cache.conv_phase_prev, cache.conv_history),
        (0, 0, 0)
    );
    assert!(!cache.conv_compact);
    assert!(!cache.paged_pass_pending);
    assert_eq!(cache.conv_spare[0].as_ref().unwrap().id(), scratch.id());
    assert_eq!(cache.anchor_slot.as_ref().unwrap().id(), anchor.id());
    assert!(
        cache.branch_continuation,
        "clear changed private continuation policy"
    );
    assert_eq!(cache.taps_dropped(), 7);
    assert_eq!(cache.kv_grows, 11);
    cache.retain_state()?;
    Ok(())
}
