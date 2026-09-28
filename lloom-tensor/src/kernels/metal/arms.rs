//! Native Metal arms: what this GPU executes, read from the families it reports.
//!
//! The device reports its `MTLGPUFamily` membership; [`NATIVE`] maps a family
//! to the element types its ALUs execute. An arm names what it needs as a
//! [`Features`] set and a dispatch under it is refused by [`require_all`] when
//! the device lacks any of it, naming what is missing — never emulated in
//! software. The set type and the refusal text are the backend-generic ones in
//! [`crate::arms`], shared with the CPU's [`crate::cpu::arms`]. (§15.2a)
//!
//! A type is registered on a family only where a probe has shown the family
//! executes it, by execution rate (§15.2a). A family nobody has probed
//! registers only what every Apple GPU executes, so an arm needing more is
//! refused there rather than trusted.
//!
//! `LLOOM_METAL_DISABLE` (a comma list of [`Feature::name`]s) withdraws
//! features, so a refusal can be exercised on a device that has the feature.

use crate::arms::{Capability, Set};
use crate::kernels::{Device, MetalKernelError};
use objc2_metal::{MTLDevice, MTLGPUFamily};
use std::sync::OnceLock;

/// A capability a Metal arm can depend on: a GPU family the device reports,
/// or an element type its ALUs execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Feature {
    /// `MTLGPUFamilyApple7`: M1, A14.
    Apple7,
    /// `MTLGPUFamilyApple8`.
    Apple8,
    /// `MTLGPUFamilyApple9`.
    Apple9,
    /// `MTLGPUFamilyApple10`.
    Apple10,
    /// `MTLGPUFamilyMac2`: every macOS GPU, Apple silicon included.
    Mac2,
    /// `MTLGPUFamilyMetal3`.
    Metal3,
    /// `MTLGPUFamilyMetal4`.
    Metal4,
    /// `float` arithmetic. Every Metal GPU.
    F32,
    /// `half` arithmetic.
    F16,
    /// `bfloat` arithmetic.
    Bf16,
    /// An int8 matrix multiply into int32: the M5 native-int arm.
    Int8Mma,
}

impl Feature {
    pub const ALL: [Feature; 11] = [
        Feature::Apple7,
        Feature::Apple8,
        Feature::Apple9,
        Feature::Apple10,
        Feature::Mac2,
        Feature::Metal3,
        Feature::Metal4,
        Feature::F32,
        Feature::F16,
        Feature::Bf16,
        Feature::Int8Mma,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Feature::Apple7 => "apple7",
            Feature::Apple8 => "apple8",
            Feature::Apple9 => "apple9",
            Feature::Apple10 => "apple10",
            Feature::Mac2 => "mac2",
            Feature::Metal3 => "metal3",
            Feature::Metal4 => "metal4",
            Feature::F32 => "f32",
            Feature::F16 => "f16",
            Feature::Bf16 => "bf16",
            Feature::Int8Mma => "int8-mma",
        }
    }

    /// The family this feature is, when it is one.
    pub const fn family(self) -> Option<MTLGPUFamily> {
        Some(match self {
            Feature::Apple7 => MTLGPUFamily::Apple7,
            Feature::Apple8 => MTLGPUFamily::Apple8,
            Feature::Apple9 => MTLGPUFamily::Apple9,
            Feature::Apple10 => MTLGPUFamily::Apple10,
            Feature::Mac2 => MTLGPUFamily::Mac2,
            Feature::Metal3 => MTLGPUFamily::Metal3,
            Feature::Metal4 => MTLGPUFamily::Metal4,
            Feature::F32 | Feature::F16 | Feature::Bf16 | Feature::Int8Mma => return None,
        })
    }
}

impl Capability for Feature {
    const ALL: &'static [Self] = &Feature::ALL;
    const BACKEND: &'static str = "Metal";
    const EXECUTOR: &'static str = "GPU";
    const NOUN: (&'static str, &'static str) = ("feature", "features");

    fn name(self) -> &'static str {
        Feature::name(self)
    }

    #[inline(always)]
    fn index(self) -> u32 {
        self as u32
    }
}

impl std::fmt::Display for Feature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A set of Metal [`Feature`]s.
pub type Features = Set<Feature>;

/// The element types each family executes, beyond `f32`.
///
/// `apple7` is probed on an M1 Max: `half` runs at or above the `float` rate,
/// and `bfloat` at the rate of `float` plus a software round-to-nearest-even, so
/// `bf16` is not registered (§15.2a). The later Apple families are unprobed: they
/// register `half`, which every Apple GPU executes, and not `bfloat` or an int8
/// matrix unit until a probe on that family shows one.
pub const NATIVE: &[(Feature, &[Feature])] = &[
    (Feature::Apple7, &[Feature::F16]),
    (Feature::Apple8, &[Feature::F16]),
    (Feature::Apple9, &[Feature::F16]),
    (Feature::Apple10, &[Feature::F16]),
];

impl Set<Feature> {
    /// What `device` reports: its families, `f32`, and the types [`NATIVE`]
    /// registers for them. Asks the device per family; a repeated caller wants
    /// [`for_device`].
    pub fn detect(device: &Device) -> Self {
        let raw = device.as_ref();
        let families = Feature::ALL
            .into_iter()
            .filter(|f| f.family().is_some_and(|family| raw.supportsFamily(family)))
            .fold(Features::empty().with(Feature::F32), Features::with);
        NATIVE
            .iter()
            .filter(|(family, _)| families.has(*family))
            .flat_map(|(_, types)| types.iter().copied())
            .fold(families, Features::with)
    }
}

/// The features `device` may execute: what it reports, less
/// `LLOOM_METAL_DISABLE`. Cached for the first device asked about.
///
/// # Panics
///
/// When `LLOOM_METAL_DISABLE` names a feature this registry does not know
/// (the rule [`crate::cpu::arms::host`] follows).
pub fn for_device(device: &Device) -> Features {
    static CACHE: OnceLock<(u64, Features)> = OnceLock::new();
    let id = device.registry_id();
    if let Some((cached, features)) = CACHE.get() {
        if *cached == id {
            return *features;
        }
    }
    let features = Features::detect(device).minus(Features::from_env("LLOOM_METAL_DISABLE"));
    let _ = CACHE.set((id, features));
    features
}

/// Refuse the arm `arm()` names unless `device` executes every member of
/// `needed`. `f32` is every Metal GPU's, so an arm needing nothing more costs no
/// device query; the name is built only for a refusal.
#[inline]
pub fn require_all(
    device: &Device,
    needed: Features,
    arm: impl FnOnce() -> String,
) -> Result<(), MetalKernelError> {
    let needed = needed.without(Feature::F32);
    if needed.is_empty() {
        return Ok(());
    }
    let have = for_device(device);
    if have.contains_all(needed) {
        return Ok(());
    }
    match have.refusal(needed, &arm()) {
        None => Ok(()),
        Some(msg) => Err(MetalKernelError::ArmRefused(msg)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Device {
        Device::system_default().expect("a Metal device")
    }

    /// The family set against a reference that does not come from
    /// `supportsFamily`: the architecture generation in `architecture_name()`.
    /// `applegpu_g13*` is M1, Apple7 and not Apple8 — present and absent rows
    /// together, which a detector returning a constant cannot match.
    #[test]
    fn detection_agrees_with_the_architecture() {
        let dev = device();
        let arch = dev.architecture_name();
        let detected = Features::detect(&dev);
        eprintln!("{arch}: {detected}");
        assert!(detected.has(Feature::F32), "{detected}");
        if arch.starts_with("applegpu_g13") {
            assert!(detected.has(Feature::Apple7), "{arch}: {detected}");
            assert!(!detected.has(Feature::Apple8), "{arch}: {detected}");
            assert!(detected.has(Feature::F16), "{arch}: {detected}");
            assert!(!detected.has(Feature::Bf16), "{arch}: {detected}");
            assert!(!detected.has(Feature::Int8Mma), "{arch}: {detected}");
        } else {
            eprintln!("{arch}: no reference row, the family set is printed only");
        }
    }

    #[test]
    fn a_missing_feature_refuses_the_arm_by_name() {
        let have = Features::empty().with(Feature::Apple7).with(Feature::F32);
        let need = Features::empty().with(Feature::F32).with(Feature::F16);
        assert_eq!(
            have.refusal(need, "q4_K (f16, f32, f16)").as_deref(),
            Some(
                "Metal arm `q4_K (f16, f32, f16)` refused: this GPU does not execute `f16`, \
                 and the arm is not emulated in software (Metal features: apple7 f32)"
            )
        );
        assert_eq!(have.with(Feature::F16).refusal(need, "x"), None);
    }

    /// An arm needing only `f32` passes on any device; one needing a feature no
    /// family registers is refused through the device's real set.
    #[test]
    fn require_all_consults_the_device() {
        let dev = device();
        assert!(require_all(&dev, Features::empty().with(Feature::F32), || "f32".into()).is_ok());
        let err = require_all(&dev, Features::empty().with(Feature::Int8Mma), || {
            "int8".into()
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("`int8-mma`"), "{err}");
    }

    /// The price of the check on an arm that needs more than `f32`, per call:
    /// a timing, so it runs only when asked (`--ignored`), under a lease.
    #[test]
    #[ignore]
    fn require_all_cost() {
        let dev = device();
        let need = Features::empty().with(Feature::F32).with(Feature::F16);
        let n = 10_000_000u32;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            for _ in 0..n {
                std::hint::black_box(require_all(&dev, std::hint::black_box(need), || {
                    String::new()
                }))
                .ok();
            }
            let ns = t.elapsed().as_nanos() as f64 / n as f64;
            eprintln!("require_all(f32+f16): {ns:.2} ns/call over {n} calls");
        }
    }

    /// Live only when the process runs with `LLOOM_METAL_DISABLE` set: every
    /// feature it names is withdrawn from the device's set and refused.
    #[test]
    fn the_device_set_honours_the_disable_list() {
        let Ok(list) = std::env::var("LLOOM_METAL_DISABLE") else {
            eprintln!("LLOOM_METAL_DISABLE unset: nothing to check");
            return;
        };
        let disabled = Features::parse_list(&list).expect("a valid list");
        assert_ne!(
            disabled,
            Features::empty(),
            "set but empty: nothing to check"
        );
        let dev = device();
        for feature in disabled.iter().filter(|f| *f != Feature::F32) {
            assert!(!for_device(&dev).has(feature), "{feature}");
            let err = require_all(&dev, Features::empty().with(feature), || {
                "disabled arm".into()
            })
            .unwrap_err()
            .to_string();
            assert!(err.contains(&format!("`{feature}`")), "{err}");
        }
        eprintln!("device: {}", for_device(&dev));
    }
}
