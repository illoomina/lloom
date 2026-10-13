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

    // The floor: the largest hot refusal, `Copy` and inline by design.
    assert_eq!(size_of::<AllocationError>(), 40);
    #[cfg(feature = "metal")]
    {
        use lloom_tensor::kernels::MetalKernelError;
        eprintln!(
            "size_of MetalKernelError = {}",
            size_of::<MetalKernelError>()
        );
        // The kernel layer carries the same refusal inline and nothing larger.
        assert_eq!(
            size_of::<MetalKernelError>(),
            size_of::<AllocationError>(),
            "a MetalKernelError payload is larger than AllocationError; box it"
        );
        assert_eq!(
            size_of::<lloom_tensor::MetalError>(),
            size_of::<AllocationError>(),
            "a MetalError payload is larger than AllocationError; box it"
        );
    }
    // One word of tag beside the floor: a variant larger than that is a cold
    // payload that should be behind a `Box`.
    assert_eq!(
        size_of::<Error>(),
        size_of::<AllocationError>() + 8,
        "an Error payload is larger than the inline refusals; box it"
    );
    assert_eq!(size_of::<Error>(), 48);
    assert_eq!(size_of::<Result<()>>(), size_of::<Error>());
    assert_eq!(size_of::<Result<Tensor>>(), size_of::<Error>());
}
