//! `size_of::<Error>()` is paid on every return: each `Result<T>` carries the
//! enum by value. The refusals a warm path meets stay inline so they widen
//! without allocating (`allocation_refusal_alloc_free.rs`), so the enum can be
//! no smaller than they are; every other payload is boxed so it is no larger
//! (`docs/invariants.md` §15.2 #15a).

use lloom_tensor::memory_admission::AllocationError;
use lloom_tensor::{Error, Result, Tensor};
use std::mem::size_of;

#[test]
fn error_is_the_size_of_its_inline_refusals_plus_a_tag() {
    eprintln!("size_of AllocationError = {}", size_of::<AllocationError>());
    eprintln!(
        "size_of MetalError = {}",
        size_of::<lloom_tensor::MetalError>()
    );
    eprintln!("size_of Error = {}", size_of::<Error>());
    eprintln!("size_of Result<()> = {}", size_of::<Result<()>>());
    eprintln!("size_of Result<Tensor> = {}", size_of::<Result<Tensor>>());
    #[cfg(feature = "metal")]
    eprintln!(
        "size_of MetalKernelError = {}",
        size_of::<lloom_tensor::kernels::MetalKernelError>()
    );

    // The floor: the largest hot refusal, `Copy` and inline by design.
    // `Refused` and `PredictionExceeded` hold 17 bytes of data each, so 24 is
    // as small as it can be without losing a field.
    assert_eq!(size_of::<AllocationError>(), 24);
    #[cfg(feature = "metal")]
    {
        use lloom_tensor::kernels::MetalKernelError;
        // The kernel layer carries the same refusal inline beside its own
        // tag, and nothing larger than the refusal.
        assert_eq!(
            size_of::<MetalKernelError>(),
            size_of::<AllocationError>() + 8,
            "a MetalKernelError payload is larger than AllocationError; box it"
        );
        assert_eq!(
            size_of::<lloom_tensor::MetalError>(),
            size_of::<MetalKernelError>(),
            "a MetalError payload is larger than MetalKernelError; box it"
        );
    }
    // One word of tag beside the floor: a payload larger than the floor,
    // other than the refusal carriers, should be behind a `Box`.
    assert_eq!(
        size_of::<Error>(),
        size_of::<AllocationError>() + 8,
        "an Error payload is larger than the inline refusals; box it"
    );
    assert_eq!(size_of::<Error>(), 32);
    assert_eq!(size_of::<Result<()>>(), 32);
    assert_eq!(size_of::<Result<Tensor>>(), 32);
}
