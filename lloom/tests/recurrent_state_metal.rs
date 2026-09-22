//! `RecurrentState`'s speculative window on Metal: the snapshot is storage of
//! its own, so a rewind after a write restores the pre-window state (issue
//! #749 — `Tensor::copy` is a shallow clone on Metal, and a snapshot taken
//! with it aliased the slab; the CPU tests could not see it).
#![cfg(feature = "metal")]

use lloom::nn::RecurrentState;
use lloom_tensor::{Device, Result, Tensor};

fn state(v: f32, dev: &Device) -> Tensor {
    Tensor::full(v, (4, 8, 8), dev).unwrap()
}

fn first(t: &Tensor) -> Result<f32> {
    Ok(t.flatten_all()?.to_vec1::<f32>()?[0])
}

#[test]
fn a_rewind_after_a_write_restores_the_snapshot_on_metal() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let mut rs = RecurrentState::new(2);
    rs.write(0, state(1.0, &dev))?;
    rs.write(1, state(10.0, &dev))?;
    rs.open_window()?;
    // the window's steps advance both layers
    rs.write(0, state(2.0, &dev))?;
    rs.write(1, state(20.0, &dev))?;
    rs.write(0, state(3.0, &dev))?;
    assert_eq!(first(rs.get(0).unwrap())?, 3.0);
    rs.rewind_to_window()?;
    assert_eq!(
        first(rs.get(0).unwrap())?,
        1.0,
        "layer 0 must return to the snapshot"
    );
    assert_eq!(
        first(rs.get(1).unwrap())?,
        10.0,
        "layer 1 must return to the snapshot"
    );
    // a second window reuses the snapshot buffer in place
    rs.write(0, state(4.0, &dev))?;
    rs.open_window()?;
    rs.write(0, state(5.0, &dev))?;
    rs.rewind_to_window()?;
    assert_eq!(first(rs.get(0).unwrap())?, 4.0);
    // and an accepted window keeps the advanced state
    rs.open_window()?;
    rs.write(0, state(6.0, &dev))?;
    rs.resolve_window()?;
    assert_eq!(first(rs.get(0).unwrap())?, 6.0);
    // the slab is never the snapshot's buffer
    let slab_id = rs.get(0).unwrap().id();
    rs.open_window()?;
    rs.write(0, state(7.0, &dev))?;
    rs.rewind_to_window()?;
    assert_eq!(
        rs.get(0).unwrap().id(),
        slab_id,
        "the slab stays the same buffer"
    );
    assert_eq!(first(rs.get(0).unwrap())?, 6.0);
    Ok(())
}
