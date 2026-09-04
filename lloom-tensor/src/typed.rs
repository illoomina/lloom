//! Backend-as-a-type: a `Typed<B>` view over the runtime-tagged [`Tensor`].
//!
//! The type axis is the **backend** (`Cpu`/`Metal`/`Cuda`), never the device: a
//! device is a runtime value with identity (which GPU, its teardown), so it
//! stays inside [`Tensor`]. `Typed<B>` is a newtype *view* that carries the
//! backend at the type level while deref-ing to the untyped `Tensor`, so the
//! whole existing API works unchanged and the view is opt-in ergonomics.
//!
//! Two properties it buys engine code that opts in:
//! - a mismatch is a **compile error**: [`Typed::add`] and its siblings take a
//!   `Typed<B>` on both sides, so an op over `Typed<Cpu>` and `Typed<Metal>`
//!   does not typecheck (proven by a `compile_fail` doctest on [`Typed`]).
//! - transfers are **visible**: [`Typed::cpu`], [`Typed::gpu`] and the general
//!   [`Typed::to`] are explicit, named methods — no silent device movement.
//!
//! `B` is a zero-sized marker, so `Typed<B>` is one pointer wide and every
//! method monomorphizes: no `dyn`, nothing on the per-token path pays for the
//! view (§3.6). It is designed to share its type axis with #620's
//! `Param`/`Frozen` — a second type-level axis on the same struct — so those
//! land as `Param<Typed<B>>`-shaped wrappers without a change here.

use crate::{Device, Error, Result, Tensor};
use std::marker::PhantomData;

mod sealed {
    pub trait Sealed {}
}

/// A compile-time backend marker. Sealed: the backend axis is exactly the set
/// of real [`Device`] backends, so a downstream crate cannot invent a fourth.
///
/// `Ane` is deliberately **absent**: the ANE is an executor over shared memory
/// (a storage mode), not a backend a tensor allocates into, so there is no
/// `Typed<Ane>` (RESTRUCTURING §5, §6).
pub trait Backend: sealed::Sealed + std::fmt::Debug + Copy + 'static {
    /// Human-readable name, for error messages.
    const NAME: &'static str;

    /// Whether `device` is on this backend. The one runtime discriminant the
    /// view reads — it never calls into a backend beyond this check.
    fn matches(device: &Device) -> bool;

    /// A device on this backend, or an error naming the backend that was asked
    /// for. Used by [`Typed::to`] to build the transfer target.
    fn device(ordinal: usize) -> Result<Device>;
}

macro_rules! backend {
    ($marker:ident, $name:literal, $matches:expr, $device:expr) => {
        #[doc = concat!("The ", $name, " backend, as a type.")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
        pub struct $marker;

        impl sealed::Sealed for $marker {}

        impl Backend for $marker {
            const NAME: &'static str = $name;

            fn matches(device: &Device) -> bool {
                let f: fn(&Device) -> bool = $matches;
                f(device)
            }

            fn device(ordinal: usize) -> Result<Device> {
                let f: fn(usize) -> Result<Device> = $device;
                f(ordinal)
            }
        }
    };
}

backend!(Cpu, "cpu", |d| d.is_cpu(), |_| Ok(Device::Cpu));
backend!(Metal, "metal", |d| d.is_metal(), |ordinal| {
    Device::new_metal(ordinal)
});
backend!(Cuda, "cuda", |d| d.is_cuda(), |ordinal| Device::new_cuda(
    ordinal
));

/// A [`Tensor`] whose backend is known at compile time.
///
/// Construct one with [`Typed::try_from`] (checked at the boundary) and unwrap
/// with [`Typed::into_inner`]; in between, it derefs to `Tensor`, so every
/// inherent `Tensor` method is available directly:
///
/// ```
/// use lloom_tensor::{Device, Tensor};
/// use lloom_tensor::typed::{Cpu, Typed};
/// let t = Tensor::zeros((2, 3), lloom_tensor::DType::F32, &Device::Cpu)?;
/// let typed: Typed<Cpu> = Typed::try_from(t)?;
/// // Deref-through: an untyped `Tensor` method works on the view unchanged.
/// assert_eq!(typed.dims(), &[2, 3]);
/// # Ok::<(), lloom_tensor::Error>(())
/// ```
///
/// A cross-backend op does not typecheck — this is the mismatch-is-a-compile-
/// error proof (asserted on the signature, not an orphan `impl`, so it is not
/// the vacuous E0117 pass §3.1a warns of):
///
/// ```compile_fail
/// use lloom_tensor::typed::{Cpu, Metal, Typed};
/// fn add_them(a: &Typed<Cpu>, b: &Typed<Metal>) {
///     // `add` takes `&Typed<Cpu>` on both sides for a `Typed<Cpu>` receiver,
///     // so passing a `Typed<Metal>` is a type error.
///     let _ = a.add(b);
/// }
/// ```
///
/// The same-backend call satisfies the signature (positive control — the bound
/// is real, the compile-fail above is not vacuous):
///
/// ```
/// use lloom_tensor::{Device, Tensor};
/// use lloom_tensor::typed::{Cpu, Typed};
/// let a: Typed<Cpu> = Tensor::zeros(4, lloom_tensor::DType::F32, &Device::Cpu)?.try_into()?;
/// let b: Typed<Cpu> = Tensor::ones(4, lloom_tensor::DType::F32, &Device::Cpu)?.try_into()?;
/// let c = a.add(&b)?;
/// assert_eq!(c.dims(), &[4]);
/// # Ok::<(), lloom_tensor::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Typed<B: Backend>(Tensor, PhantomData<B>);

impl<B: Backend> Typed<B> {
    /// The untyped tensor, by reference. Prefer deref for method calls; this is
    /// for the rare case that needs the `&Tensor` itself.
    pub fn as_tensor(&self) -> &Tensor {
        &self.0
    }

    /// Drop the backend tag, returning the untyped [`Tensor`].
    pub fn into_inner(self) -> Tensor {
        self.0
    }

    /// Re-tag a tensor already known to be on `B` — the unchecked companion to
    /// [`Typed::try_from`]. Private: the only way in from outside is the checked
    /// boundary, so the tag cannot lie.
    fn wrap(tensor: Tensor) -> Self {
        debug_assert!(B::matches(tensor.device()), "typed backend tag would lie");
        Self(tensor, PhantomData)
    }

    /// Move to backend `B2`, returning a `Typed<B2>`. The one general transfer;
    /// [`Typed::cpu`]/[`Typed::gpu`] are named conveniences over it. Visible by
    /// construction — a transfer only happens where this is written.
    pub fn to<B2: Backend>(&self) -> Result<Typed<B2>> {
        if B2::matches(self.0.device()) {
            return Ok(Typed::<B2>::wrap(self.0.clone()));
        }
        let device = B2::device(0)?;
        Ok(Typed::<B2>::wrap(self.0.to_device(&device)?))
    }

    /// Move to the CPU backend.
    pub fn cpu(&self) -> Result<Typed<Cpu>> {
        self.to::<Cpu>()
    }

    /// Move to the Metal backend.
    pub fn metal(&self) -> Result<Typed<Metal>> {
        self.to::<Metal>()
    }

    /// Move to the CUDA backend.
    pub fn cuda(&self) -> Result<Typed<Cuda>> {
        self.to::<Cuda>()
    }

    /// Move to this platform's GPU backend — Metal here, CUDA on a CUDA build.
    /// A named alias for the common case; use [`Typed::to`] when the backend
    /// must be explicit.
    #[cfg(feature = "metal")]
    pub fn gpu(&self) -> Result<Typed<Metal>> {
        self.to::<Metal>()
    }

    /// Move to this platform's GPU backend — CUDA on a CUDA build.
    #[cfg(all(feature = "cuda", not(feature = "metal")))]
    pub fn gpu(&self) -> Result<Typed<Cuda>> {
        self.to::<Cuda>()
    }
}

/// Same-backend elementwise/matmul ops. Each takes a `Typed<B>` on both sides
/// and returns a `Typed<B>`, so the backend cannot silently change and a
/// cross-backend op does not typecheck. The compute is the untyped `Tensor`
/// op verbatim — the view adds a type check, never an arithmetic change.
impl<B: Backend> Typed<B> {
    pub fn add(&self, rhs: &Typed<B>) -> Result<Typed<B>> {
        Ok(Typed::wrap((self.0.add(&rhs.0))?))
    }

    pub fn mul(&self, rhs: &Typed<B>) -> Result<Typed<B>> {
        Ok(Typed::wrap((self.0.mul(&rhs.0))?))
    }

    pub fn matmul(&self, rhs: &Typed<B>) -> Result<Typed<B>> {
        Ok(Typed::wrap(self.0.matmul(&rhs.0)?))
    }

    pub fn broadcast_add(&self, rhs: &Typed<B>) -> Result<Typed<B>> {
        Ok(Typed::wrap(self.0.broadcast_add(&rhs.0)?))
    }
}

impl<B: Backend> std::ops::Deref for Typed<B> {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<B: Backend> TryFrom<Tensor> for Typed<B> {
    type Error = Error;

    fn try_from(tensor: Tensor) -> Result<Self> {
        if B::matches(tensor.device()) {
            Ok(Self(tensor, PhantomData))
        } else {
            Err(Error::Msg(format!(
                "tensor is on {:?}, not the {} backend",
                tensor.device().location(),
                B::NAME
            )))
        }
    }
}

impl<B: Backend> From<Typed<B>> for Tensor {
    fn from(typed: Typed<B>) -> Tensor {
        typed.0
    }
}
