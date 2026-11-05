//! #944 item 13's slot-grant control, outside `expert_stream.rs` because it
//! waits for the device between layers (`tests/qwen4exp_eviction_no_drain.rs`
//! holds that file to no device wait).

use super::tests::{open, resize_control_device, stacked, write_gguf};
use super::*;

/// A grant shrinking several tiers holds one layer's old and new banks at
/// a time, not the whole tier twice (#944 item 13). The device's
/// allocated bytes are read inside `retire`, right after each layer's
/// resize allocated its replacement and before its old banks retire: the
/// peak the serve path reached. With the retirement waits the excess over
/// the start stays within two layers' replacements (the one just made,
/// and Metal's slack); without them it grows by one layer per layer.
#[cfg(feature = "metal")]
#[test]
fn a_slot_grant_holds_one_layers_replacement_banks_at_a_time() {
    const LAYERS: usize = 6;
    const SLOTS: usize = 12;
    const WANTED: usize = 10;
    let dev = resize_control_device();
    let m = dev.as_metal_device().unwrap();
    // 16 experts of 256 x 256 f32 (256 KiB each): a pool of 12 slots is
    // 3 MiB per projection, 9 MiB per layer.
    let gate = stacked(16, 256, 256, 1.0, GgmlDType::F32);
    let up = stacked(16, 256, 256, 2.0, GgmlDType::F32);
    let down = stacked(16, 256, 256, 3.0, GgmlDType::F32);
    let path = write_gguf(&[("g", &gate), ("u", &up), ("d", &down)]);
    let tiers: Vec<Streamed> = (0..LAYERS)
        .map(|_| {
            Streamed::new(
                open(&path, "g"),
                open(&path, "u"),
                open(&path, "d"),
                SLOTS,
                LoaderArm::Pread,
                &dev,
            )
            .unwrap()
        })
        .collect();
    m.wait_until_completed().unwrap();
    let replacement = 3 * WANTED * 256 * 256 * 4; // one layer's new banks
    let start = m.device().current_allocated_size();
    let mut excess = Vec::new();
    let parked = resize_layers_one_at_a_time(tiers.iter(), WANTED, 1, || {
        excess.push(m.device().current_allocated_size().saturating_sub(start));
        m.wait_until_completed()?;
        m.commands().release_idle_slot_residency();
        Ok(())
    })
    .unwrap();
    m.wait_until_completed().unwrap();
    let after = m.device().current_allocated_size();
    let peak = *excess.iter().max().unwrap();
    eprintln!(
        "SLOT_GRANT layers={LAYERS} slots={SLOTS}->{WANTED} replacement_per_layer={replacement} \
         start={start} excess_per_layer={excess:?} peak_excess={peak} after={after} parked={parked} api={}",
        m.commands().actual_command_api()
    );
    assert_eq!(excess.len(), LAYERS, "every layer resized");
    assert!(
        peak <= 2 * replacement,
        "a slot grant held {peak} B over its start, more than two layers' \
         replacements ({replacement} B each): {excess:?}"
    );
    assert!(
        after < start,
        "the grant gave nothing back: start {start} B, after {after} B"
    );
}
