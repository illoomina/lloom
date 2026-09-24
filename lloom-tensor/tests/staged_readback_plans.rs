//! The staged readback under pass plans: a kernel's output copied to a
//! CPU-visible buffer inside a recorded, an observed and a served pass
//! (issue #749 P28: the paired prefill read garbage ids under plans).
#![cfg(feature = "metal")]
use lloom_tensor::metal_backend::{
    stage_readback, stage_readback_deferred, ArenaLayout, PassKey, PassKind,
};
use lloom_tensor::{DType, Device, Tensor};

#[test]
fn staged_readback_is_right_in_every_pass_state() {
    let dev = Device::new_metal(0).unwrap();
    let Device::Metal(m) = &dev else {
        unreachable!()
    };
    m.enable_pass_plans(ArenaLayout::Packed);
    let n = 4096usize;
    let base = Tensor::arange(0f32, n as f32, &dev).unwrap();
    for pass in 0..5 {
        m.begin_pass(PassKey::new(PassKind::Prefill { w: 8, fresh: false }))
            .unwrap();
        // Two kernel outputs, the second read back staged: its slot is the
        // plan's on a served pass.
        let a = base.affine(2.0, pass as f64).unwrap();
        let y = a.affine(1.0, 1.0).unwrap();
        let st = stage_readback(&y).unwrap();
        let got: Vec<f32> = st.finish().unwrap();
        let want: Vec<f32> = (0..n).map(|i| 2.0 * i as f32 + pass as f32 + 1.0).collect();
        let bad = got.iter().zip(&want).filter(|(g, w)| g != w).count();
        // The deferred form: finished after a plain readback drained the queue.
        let z = y.affine(1.0, 2.0).unwrap();
        let st2 = stage_readback_deferred(&z).unwrap();
        let _drain = z.narrow(0, 0, 1).unwrap().to_vec1::<f32>().unwrap();
        let got2: Vec<f32> = st2.finish().unwrap();
        let bad2 = got2
            .iter()
            .zip(&want)
            .filter(|(g, w)| **g != *w + 2.0)
            .count();
        println!(
            "pass {pass} base_mode={} staged bad={bad} deferred bad={bad2} first={:?} {:?}",
            m.base_mode(),
            &got[..3],
            &got2[..3]
        );
        m.finish_pass().unwrap();
        assert_eq!(bad, 0, "pass {pass}: the staged readback read wrong values");
        assert_eq!(
            bad2, 0,
            "pass {pass}: the deferred staged readback read wrong values"
        );
    }
    let _ = DType::F32;
}
