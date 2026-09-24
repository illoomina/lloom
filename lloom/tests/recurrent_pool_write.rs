//! A pooled `RecurrentState::write` lands in the phase and slot it names on
//! Metal as on the CPU: written, flipped, read back through `get` (P30 (1):
//! the pooled prefill's state differed from the slab's on Metal only).
#![cfg(feature = "metal")]
use lloom::nn::{RecurrentState, StatePool};
use lloom_tensor::{DType, Device, Result, Tensor};

#[test]
fn a_pooled_write_lands_in_its_phase_and_slot_on_metal() -> Result<()> {
    let cpu = Device::Cpu;
    let gpu = Device::new_metal(0)?;
    for dev in [&cpu, &gpu] {
        let pool = StatePool::with_phases(2, 3, 1, &[4, 4], DType::F32, dev);
        let _first = RecurrentState::in_pool(&pool)?; // slot 0 taken
        let mut st = RecurrentState::in_pool(&pool)?; // slot 1
        let a = Tensor::arange(0f32, 16.0, dev)?.reshape((4, 4))?;
        st.write(0, a.clone())?;
        let back: Vec<f32> = st.get(0).expect("written").flatten_all()?.to_vec1()?;
        let want: Vec<f32> = a.flatten_all()?.to_vec1()?;
        assert_eq!(back, want, "{dev:?}: first write");
        let b = (a.clone() * 2.0)?;
        st.write(0, b.clone())?;
        let back: Vec<f32> = st.get(0).expect("written").flatten_all()?.to_vec1()?;
        let want: Vec<f32> = b.flatten_all()?.to_vec1()?;
        assert_eq!(back, want, "{dev:?}: second write (the other phase)");
        // The whole pool: only slot 1's phases hold data.
        let all: Vec<f32> = pool.layer(0)?.flatten_all()?.to_vec1()?;
        eprintln!("{dev:?} pool layer 0: {all:?}");
    }
    Ok(())
}
