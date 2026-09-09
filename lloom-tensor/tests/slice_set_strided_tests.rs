//! `Tensor::slice_set` with a STRIDED source (#727): a `transpose` view is
//! written into the destination window from where it lies -- one
//! strided-to-strided copy on Metal, a strided walk on the CPU -- and the
//! result is what `slice_set` over the view's contiguous copy writes.

use lloom_tensor::{test_device, test_utils, DType, Device, Result, Tensor};

fn slab_and_view(dev: &Device) -> Result<(Tensor, Tensor)> {
    // `[b, t, h, d]` -> `transpose(1, 2)` -> a strided `[b, h, t, d]`, the
    // layout `v_proj`'s output takes on the way to the KV slab.
    let (b, t, h, d) = (2usize, 3usize, 4usize, 8usize);
    let src = Tensor::arange(0f32, (b * t * h * d) as f32, dev)?
        .to_dtype(DType::F16)?
        .reshape((b, t, h, d))?
        .transpose(1, 2)?;
    assert!(
        !src.is_contiguous(),
        "the view must be strided for this test"
    );
    let slab = Tensor::zeros((b, h, 16, d), DType::F16, dev)?;
    Ok((slab, src))
}

fn strided_source_writes_what_its_copy_writes(dev: &Device) -> Result<()> {
    let (slab_a, src) = slab_and_view(dev)?;
    let (slab_b, _) = slab_and_view(dev)?;
    slab_a.slice_set(&src, 2, 5)?;
    slab_b.slice_set(&src.contiguous()?, 2, 5)?;
    let a = slab_a
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let b = slab_b
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    assert_eq!(
        a, b,
        "a strided source must write exactly what its contiguous copy writes"
    );
    // The window is the only thing that moved: `numel` elements left zero.
    let written = a.iter().filter(|v| **v != 0.0).count();
    // Element 0 of the source is 0.0, so it is one short of the source count.
    assert_eq!(written, src.elem_count() - 1);
    // And it landed at rows 5..8 of every `(b, h)` plane, not elsewhere.
    let rows = slab_a.narrow(2, 5, 3)?.contiguous()?;
    test_utils::assert_tensor_eq(&rows, &src.contiguous()?)?;
    Ok(())
}

fn a_contiguous_destination_is_still_required(dev: &Device) -> Result<()> {
    let (slab, src) = slab_and_view(dev)?;
    let strided_dst = slab.transpose(1, 2)?;
    assert!(strided_dst.slice_set(&src, 1, 5).is_err());
    Ok(())
}

test_device!(
    strided_source_writes_what_its_copy_writes,
    strided_source_writes_what_its_copy_writes_cpu,
    strided_source_writes_what_its_copy_writes_gpu,
    strided_source_writes_what_its_copy_writes_metal
);
test_device!(
    a_contiguous_destination_is_still_required,
    a_contiguous_destination_is_still_required_cpu,
    a_contiguous_destination_is_still_required_gpu,
    a_contiguous_destination_is_still_required_metal
);
