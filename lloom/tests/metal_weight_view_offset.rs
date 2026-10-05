//! A weight that is a view at a non-zero offset into a larger buffer (a
//! weight-group slot, `nn::weight_groups`, #892 A6a) must multiply like a
//! copy of it at offset 0 — on the decode shape (one row), where the Metal
//! matmul takes its GEMV route, and on a prefill shape.
#![cfg(feature = "metal")]
use lloom_tensor::{DType, Device, Result, Tensor};

#[test]
fn a_weight_view_at_an_offset_multiplies_like_its_copy() -> Result<()> {
    let Ok(dev) = Device::new_metal(0) else {
        return Ok(());
    };
    let (o, i) = (16usize, 8usize);
    for dtype in [DType::F32, DType::F16] {
        // Three "slots" of o*i elements; the weight is slot 1 (offset o*i).
        let n = 3 * o * i;
        let backing = Tensor::arange(0f32, n as f32, &dev)?
            .affine(1.0 / 64.0, -0.5)?
            .to_dtype(dtype)?;
        let view = backing.narrow(0, o * i, o * i)?.reshape((o, i))?;
        let copy = view
            .contiguous()?
            .to_device(&Device::Cpu)?
            .to_device(&dev)?;
        for rows in [1usize, 5] {
            let x = Tensor::arange(0f32, (rows * i) as f32, &dev)?
                .affine(1.0 / 16.0, 0.25)?
                .to_dtype(dtype)?
                .reshape((rows, i))?;
            let want = x
                .matmul(&copy.t()?)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            let got = x
                .matmul(&view.t()?)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            assert!(
                got == want,
                "{dtype:?} rows={rows}: a weight view at offset {} multiplies differently:\n got  {:?}\n want {:?}",
                o * i,
                &got[0][..4],
                &want[0][..4]
            );
        }
    }
    Ok(())
}
