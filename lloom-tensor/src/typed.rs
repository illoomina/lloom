//! Backend and dtype as types: a `Typed<B, E>` view over the runtime-tagged
//! [`Tensor`].
//!
//! Two type-level axes on one struct (RESTRUCTURING §6):
//! - the **backend** `B` (`Cpu`/`Metal`/`Cuda`), never the device: a device is
//!   a runtime value with identity (which GPU, its teardown), so it stays
//!   inside [`Tensor`];
//! - the **dtype** `E`, an element type ([`half::f16`], `f32`, …) or
//!   [`AnyDtype`], the default, which leaves the dtype to run time. `Typed<B>`
//!   is `Typed<B, AnyDtype>` and means exactly what it meant before the dtype
//!   axis existed.
//!
//! The view derefs to the untyped `Tensor`, so the whole existing API works
//! unchanged and opting in is local. What it buys the code that opts in:
//! - a mismatch on either axis is a **compile error**: [`Typed::add`] and its
//!   siblings take the same `Typed<B, E>` on both sides, so an op over
//!   `Typed<Cpu>` and `Typed<Metal>`, or over `Typed<Cpu, f16>` and
//!   `Typed<Cpu, f32>`, does not typecheck (the `compile_fail` doctests on
//!   [`Typed`]);
//! - a block declares its `(activation, accumulate, store)` [`Triple`] in its
//!   signature ([`ActOf`] in, [`StoreOf`] out), so feeding it a view of another
//!   dtype does not typecheck either;
//! - transfers and casts are **visible**: [`Typed::to`], [`Typed::cpu`],
//!   [`Typed::cast`] are explicit, named methods.
//!
//! Both markers are zero-sized, so `Typed<B, E>` is one pointer wide and every
//! method monomorphizes: nothing on the per-token path pays for the view.
//! The train/frozen axis composes on the same struct: a third defaulted type
//! parameter, or a `Param<Typed<B, E>>` wrapper; either is additive here.

use crate::{DType, Device, Error, Result, Tensor, WithDType};
use std::marker::PhantomData;

mod sealed {
    pub trait Backend {}
    pub trait Element {}
    pub trait Triple {}
}

/// A compile-time backend marker. Sealed: the backend axis is exactly the set
/// of real [`Device`] backends, so a downstream crate cannot invent a fourth.
///
/// `Ane` is deliberately **absent**: the ANE is an executor over shared memory
/// (a storage mode), not a backend a tensor allocates into, so there is no
/// `Typed<Ane>` (RESTRUCTURING §5, §6).
pub trait Backend: sealed::Backend + std::fmt::Debug + Copy + 'static {
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

        impl sealed::Backend for $marker {}

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

/// A compile-time dtype marker: the element type a [`Typed`] view holds.
/// Sealed: exactly the tensor element types ([`WithDType`]'s scalars, whose
/// mapping to [`DType`] this reuses) plus [`AnyDtype`].
pub trait Element: sealed::Element + std::fmt::Debug + Copy + 'static {
    /// Human-readable name, for error messages.
    const NAME: &'static str;

    /// The runtime dtype a view with this marker holds; `None` for
    /// [`AnyDtype`], which holds whatever the tensor holds.
    const RUNTIME_DTYPE: Option<DType>;

    /// Whether a tensor of `dtype` may carry this marker.
    fn matches(dtype: DType) -> bool {
        match Self::RUNTIME_DTYPE {
            Some(d) => d == dtype,
            None => true,
        }
    }
}

/// The dtype axis left to run time: any dtype matches, and ops between two
/// `AnyDtype` views check dtypes the way the untyped [`Tensor`] does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AnyDtype;

impl sealed::Element for AnyDtype {}

impl Element for AnyDtype {
    const NAME: &'static str = "any";
    const RUNTIME_DTYPE: Option<DType> = None;
}

macro_rules! element {
    ($($ty:ty => $name:literal),* $(,)?) => {
        $(
            impl sealed::Element for $ty {}

            impl Element for $ty {
                const NAME: &'static str = $name;
                const RUNTIME_DTYPE: Option<DType> = Some(<$ty as WithDType>::DTYPE);
            }
        )*
    };
}

element!(
    u8 => "u8",
    u32 => "u32",
    i16 => "i16",
    i32 => "i32",
    i64 => "i64",
    half::f16 => "f16",
    half::bf16 => "bf16",
    f32 => "f32",
    f64 => "f64",
    float8::F8E4M3 => "f8e4m3",
);

/// A block's `(activation, accumulate, store)` dtype triple, as a type.
///
/// A block declaring `T` takes an [`ActOf<B, T>`] and returns a
/// [`StoreOf<B, T>`], so a view of any other dtype does not typecheck at its
/// boundary. The accumulate leg is carried for the kernel selector and the
/// numeric model: a tensor never holds an accumulator, so no signature can
/// check it, and the kernel the block dispatches is what honours it — F32 for
/// every float triple (`docs/invariants.md` §15.2a).
///
/// The two float triples the kernels instantiate are named. The native-int
/// `(i8, i32, i8)` arm is not: no tensor dtype is `i8`, so no view could carry
/// its activation leg.
pub trait Triple: sealed::Triple + std::fmt::Debug + Copy + 'static {
    type Act: Element + WithDType;
    type Acc: Element + WithDType;
    type Store: Element + WithDType;

    /// The runtime selector this triple lowers to on the quantized matmul-vec
    /// dispatch — the one place the type and the runtime enum are paired.
    #[cfg(feature = "metal")]
    const GEMV: crate::kernels::GemvTriple;
}

/// `(f32, f32, f32)`: the full-precision triple, and the CPU parity reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct F32F32F32;

/// `(f16, f32, f16)`: f16 activations and store, f32 accumulate — LFM2's
/// dense blocks and the quantized f16 arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct F16F32F16;

impl sealed::Triple for F32F32F32 {}
impl Triple for F32F32F32 {
    type Act = f32;
    type Acc = f32;
    type Store = f32;
    #[cfg(feature = "metal")]
    const GEMV: crate::kernels::GemvTriple = crate::kernels::GemvTriple::F32F32F32;
}

impl sealed::Triple for F16F32F16 {}
impl Triple for F16F32F16 {
    type Act = half::f16;
    type Acc = f32;
    type Store = half::f16;
    #[cfg(feature = "metal")]
    const GEMV: crate::kernels::GemvTriple = crate::kernels::GemvTriple::F16F32F16;
}

/// The view a block declaring triple `T` on backend `B` takes.
pub type ActOf<B, T> = Typed<B, <T as Triple>::Act>;

/// The view a block declaring triple `T` on backend `B` returns.
pub type StoreOf<B, T> = Typed<B, <T as Triple>::Store>;

/// A [`Tensor`] whose backend, and optionally dtype, is known at compile time.
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
/// A cross-backend op does not typecheck — the mismatch-is-a-compile-error
/// proof, asserted on the signature rather than on an orphan `impl`, which
/// would fail on coherence (E0117) whatever the bound says:
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
/// Nor does a cross-dtype op, once both views name their dtype:
///
/// ```compile_fail
/// use lloom_tensor::typed::{Cpu, Typed};
/// fn add_them(a: &Typed<Cpu, half::f16>, b: &Typed<Cpu, f32>) {
///     // `add` takes `&Typed<Cpu, f16>` on both sides for a `Typed<Cpu, f16>`
///     // receiver, so passing a `Typed<Cpu, f32>` is a type error.
///     let _ = a.add(b);
/// }
/// ```
///
/// Nor does feeding a block a view of the wrong dtype for the triple it
/// declares:
///
/// ```compile_fail
/// use lloom_tensor::typed::{ActOf, Cpu, F16F32F16, Typed};
/// fn block(_x: &ActOf<Cpu, F16F32F16>) {}
/// fn feed(x: &Typed<Cpu, f32>) {
///     // The block reads f16 activations; an f32 view is a type error.
///     block(x)
/// }
/// ```
///
/// The matching calls satisfy the same signatures (positive controls). They
/// are what makes the compile-fails above evidence: stable rustdoc does not
/// check an error code on a `compile_fail` block, so a doctest that failed to
/// compile for any other reason would pass as well.
///
/// ```
/// use lloom_tensor::{DType, Device, Tensor};
/// use lloom_tensor::typed::{ActOf, Cpu, F16F32F16, StoreOf, Typed};
/// let a: Typed<Cpu> = Tensor::zeros(4, DType::F32, &Device::Cpu)?.try_into()?;
/// let b: Typed<Cpu> = Tensor::ones(4, DType::F32, &Device::Cpu)?.try_into()?;
/// assert_eq!(a.add(&b)?.dims(), &[4]);
///
/// let x: Typed<Cpu, half::f16> = Tensor::ones(4, DType::F16, &Device::Cpu)?.try_into()?;
/// let y: Typed<Cpu, half::f16> = Tensor::ones(4, DType::F16, &Device::Cpu)?.try_into()?;
/// assert_eq!(x.add(&y)?.dtype(), DType::F16);
///
/// fn block(x: &ActOf<Cpu, F16F32F16>) -> lloom_tensor::Result<StoreOf<Cpu, F16F32F16>> {
///     x.add(x)
/// }
/// assert_eq!(block(&x)?.dtype(), DType::F16);
/// # Ok::<(), lloom_tensor::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Typed<B: Backend, E: Element = AnyDtype>(Tensor, PhantomData<(B, E)>);

impl<B: Backend, E: Element> Typed<B, E> {
    /// The untyped tensor, by reference. Prefer deref for method calls; this is
    /// for the rare case that needs the `&Tensor` itself.
    pub fn as_tensor(&self) -> &Tensor {
        &self.0
    }

    /// Drop both tags, returning the untyped [`Tensor`].
    pub fn into_inner(self) -> Tensor {
        self.0
    }

    /// Drop the dtype tag and keep the backend one.
    pub fn erase_dtype(self) -> Typed<B> {
        Typed(self.0, PhantomData)
    }

    /// Re-tag a tensor already known to be on `B` with a dtype matching `E` —
    /// the unchecked companion to [`Typed::try_from`]. Private: the only way in
    /// from outside is the checked boundary, so the tags cannot lie.
    fn wrap(tensor: Tensor) -> Self {
        debug_assert!(B::matches(tensor.device()), "typed backend tag would lie");
        debug_assert!(E::matches(tensor.dtype()), "typed dtype tag would lie");
        Self(tensor, PhantomData)
    }

    /// Convert to dtype `E2`, returning a `Typed<B, E2>`. Visible by
    /// construction, like a transfer: a conversion only happens where this is
    /// written.
    pub fn cast<E2: Element + WithDType>(&self) -> Result<Typed<B, E2>> {
        Ok(Typed::<B, E2>::wrap(
            self.0.to_dtype(<E2 as WithDType>::DTYPE)?,
        ))
    }

    /// Move to backend `B2`, returning a `Typed<B2, E>`; the dtype is kept.
    /// The one general transfer; [`Typed::cpu`]/[`Typed::gpu`] are named
    /// conveniences over it. Visible by construction — a transfer only
    /// happens where this is written.
    pub fn to<B2: Backend>(&self) -> Result<Typed<B2, E>> {
        if B2::matches(self.0.device()) {
            return Ok(Typed::<B2, E>::wrap(self.0.clone()));
        }
        let device = B2::device(0)?;
        Ok(Typed::<B2, E>::wrap(self.0.to_device(&device)?))
    }

    /// Move to the CPU backend.
    pub fn cpu(&self) -> Result<Typed<Cpu, E>> {
        self.to::<Cpu>()
    }

    /// Move to the Metal backend.
    pub fn metal(&self) -> Result<Typed<Metal, E>> {
        self.to::<Metal>()
    }

    /// Move to the CUDA backend.
    pub fn cuda(&self) -> Result<Typed<Cuda, E>> {
        self.to::<Cuda>()
    }

    /// Move to this platform's GPU backend — Metal here, CUDA on a CUDA build.
    /// A named alias for the common case; use [`Typed::to`] when the backend
    /// must be explicit.
    #[cfg(feature = "metal")]
    pub fn gpu(&self) -> Result<Typed<Metal, E>> {
        self.to::<Metal>()
    }

    /// Move to this platform's GPU backend — CUDA on a CUDA build.
    #[cfg(all(feature = "cuda", not(feature = "metal")))]
    pub fn gpu(&self) -> Result<Typed<Cuda, E>> {
        self.to::<Cuda>()
    }
}

/// Same-backend, same-dtype elementwise/matmul ops. Each takes a
/// `Typed<B, E>` on both sides and returns a `Typed<B, E>`, so neither tag can
/// silently change and a mismatched op does not typecheck. The compute is the
/// untyped `Tensor` op verbatim — the view adds a type check, never an
/// arithmetic change.
impl<B: Backend, E: Element> Typed<B, E> {
    pub fn add(&self, rhs: &Typed<B, E>) -> Result<Typed<B, E>> {
        Ok(Typed::wrap(self.0.add(&rhs.0)?))
    }

    pub fn mul(&self, rhs: &Typed<B, E>) -> Result<Typed<B, E>> {
        Ok(Typed::wrap(self.0.mul(&rhs.0)?))
    }

    pub fn matmul(&self, rhs: &Typed<B, E>) -> Result<Typed<B, E>> {
        Ok(Typed::wrap(self.0.matmul(&rhs.0)?))
    }

    pub fn broadcast_add(&self, rhs: &Typed<B, E>) -> Result<Typed<B, E>> {
        Ok(Typed::wrap(self.0.broadcast_add(&rhs.0)?))
    }
}

impl<B: Backend, E: Element> std::ops::Deref for Typed<B, E> {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<B: Backend, E: Element> TryFrom<Tensor> for Typed<B, E> {
    type Error = Error;

    fn try_from(tensor: Tensor) -> Result<Self> {
        if !B::matches(tensor.device()) {
            return Err(Error::Msg(format!(
                "tensor is on {:?}, not the {} backend",
                tensor.device().location(),
                B::NAME
            )));
        }
        if !E::matches(tensor.dtype()) {
            return Err(Error::Msg(format!(
                "tensor is {}, not {}",
                tensor.dtype().as_str(),
                E::NAME
            )));
        }
        Ok(Self(tensor, PhantomData))
    }
}

impl<B: Backend, E: Element> From<Typed<B, E>> for Tensor {
    fn from(typed: Typed<B, E>) -> Tensor {
        typed.0
    }
}
