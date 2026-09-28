//! Native CPU arms: the ISA features this CPU executes, read at run time.
//!
//! A kernel that needs a feature asks [`native`] and takes its native arm only
//! when the answer is yes; an arm a caller *selects* on a CPU that lacks its
//! feature is refused by [`require`], with an error naming the feature.
//! Neither path emulates a missing instruction in software. The set type and
//! the refusal are the backend-generic ones in [`crate::arms`]. (§15.2a)
//!
//! The set is detected once per process. `LLOOM_CPU_DISABLE` (a comma list of
//! [`Feature::name`]s) removes features from it, so an arm's absent path can be
//! exercised on a CPU that has the feature. A feature the build compiles in for
//! every function (the target's baseline, or `-C target-feature`) is native
//! regardless: the compiler may already have emitted it anywhere.
//!
//! SME/SME2 and AMX are not listed: the standard library has no stable detector
//! for them, and a feature that cannot be detected cannot be registered.

use crate::arms::{Capability, Set};
use crate::Result;
use std::fmt;
use std::sync::OnceLock;

/// A CPU ISA feature an arm can depend on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Feature {
    /// aarch64 `FEAT_DotProd`: `sdot`/`udot`, an int8 dot product into int32.
    DotProd,
    /// aarch64 `FEAT_FP16`: half-precision vector arithmetic.
    Fp16,
    /// aarch64 `FEAT_FHM`: `fmlal`, f16 products accumulated into f32.
    Fhm,
    /// aarch64 `FEAT_I8MM`: the int8 matrix multiply `smmla`.
    I8mm,
    /// aarch64 `FEAT_BF16`: `bfdot`/`bfmmla`.
    Bf16,
    /// x86 AVX2.
    Avx2,
    /// x86 F16C: f16 <-> f32 conversion.
    F16c,
    /// x86 AVX-VNNI: 256-bit int8 dot product.
    AvxVnni,
    /// x86 AVX512-VNNI.
    Avx512Vnni,
    /// x86 AVX512-BF16.
    Avx512Bf16,
}

impl Feature {
    /// Every feature the registry knows, in declaration order.
    pub const ALL: [Feature; 10] = [
        Feature::DotProd,
        Feature::Fp16,
        Feature::Fhm,
        Feature::I8mm,
        Feature::Bf16,
        Feature::Avx2,
        Feature::F16c,
        Feature::AvxVnni,
        Feature::Avx512Vnni,
        Feature::Avx512Bf16,
    ];

    /// The name `target_feature` and the standard library's detectors use.
    pub const fn name(self) -> &'static str {
        match self {
            Feature::DotProd => "dotprod",
            Feature::Fp16 => "fp16",
            Feature::Fhm => "fhm",
            Feature::I8mm => "i8mm",
            Feature::Bf16 => "bf16",
            Feature::Avx2 => "avx2",
            Feature::F16c => "f16c",
            Feature::AvxVnni => "avxvnni",
            Feature::Avx512Vnni => "avx512vnni",
            Feature::Avx512Bf16 => "avx512bf16",
        }
    }

    /// The feature whose [`name`](Self::name) is `name`.
    pub fn from_name(name: &str) -> Option<Feature> {
        Feature::ALL.into_iter().find(|f| f.name() == name)
    }

    /// Whether the build enables the feature for every function. Such a
    /// feature needs no run-time check, and cannot be withdrawn by one.
    pub const fn compiled_in(self) -> bool {
        match self {
            Feature::DotProd => cfg!(target_feature = "dotprod"),
            Feature::Fp16 => cfg!(target_feature = "fp16"),
            Feature::Fhm => cfg!(target_feature = "fhm"),
            Feature::I8mm => cfg!(target_feature = "i8mm"),
            Feature::Bf16 => cfg!(target_feature = "bf16"),
            Feature::Avx2 => cfg!(target_feature = "avx2"),
            Feature::F16c => cfg!(target_feature = "f16c"),
            Feature::AvxVnni => cfg!(target_feature = "avxvnni"),
            Feature::Avx512Vnni => cfg!(target_feature = "avx512vnni"),
            Feature::Avx512Bf16 => cfg!(target_feature = "avx512bf16"),
        }
    }

    /// What the operating system reports for this CPU. A feature of another
    /// architecture is absent.
    fn detect(self) -> bool {
        match self {
            #[cfg(target_arch = "aarch64")]
            Feature::DotProd => std::arch::is_aarch64_feature_detected!("dotprod"),
            #[cfg(target_arch = "aarch64")]
            Feature::Fp16 => std::arch::is_aarch64_feature_detected!("fp16"),
            #[cfg(target_arch = "aarch64")]
            Feature::Fhm => std::arch::is_aarch64_feature_detected!("fhm"),
            #[cfg(target_arch = "aarch64")]
            Feature::I8mm => std::arch::is_aarch64_feature_detected!("i8mm"),
            #[cfg(target_arch = "aarch64")]
            Feature::Bf16 => std::arch::is_aarch64_feature_detected!("bf16"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Feature::Avx2 => std::arch::is_x86_feature_detected!("avx2"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Feature::F16c => std::arch::is_x86_feature_detected!("f16c"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Feature::AvxVnni => std::arch::is_x86_feature_detected!("avxvnni"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Feature::Avx512Vnni => std::arch::is_x86_feature_detected!("avx512vnni"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Feature::Avx512Bf16 => std::arch::is_x86_feature_detected!("avx512bf16"),
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl Capability for Feature {
    const ALL: &'static [Self] = &Feature::ALL;
    const BACKEND: &'static str = "CPU";
    const EXECUTOR: &'static str = "CPU";
    const NOUN: (&'static str, &'static str) = ("feature", "features");

    fn name(self) -> &'static str {
        Feature::name(self)
    }

    #[inline(always)]
    fn index(self) -> u32 {
        self as u32
    }
}

/// A set of [`Feature`]s.
pub type Features = Set<Feature>;

impl Set<Feature> {
    /// Every feature the operating system reports for this CPU. Asks the OS
    /// per feature; a repeated caller wants [`host`].
    pub fn detect() -> Self {
        Feature::ALL
            .into_iter()
            .filter(|f| f.detect())
            .fold(Features::empty(), Features::with)
    }
}

/// The features this process may execute: what the OS reports, less
/// `LLOOM_CPU_DISABLE`, plus what the build compiles in. Detected once.
///
/// # Panics
///
/// When `LLOOM_CPU_DISABLE` names a feature this registry does not know: it
/// is an operator's explicit request, and ignoring it would run the arm it
/// asked to withdraw.
pub fn host() -> Features {
    static HOST: OnceLock<Features> = OnceLock::new();
    *HOST.get_or_init(|| {
        let detected = Features::detect().minus(Features::from_env("LLOOM_CPU_DISABLE"));
        Feature::ALL
            .into_iter()
            .filter(|f| f.compiled_in())
            .fold(detected, Features::with)
    })
}

/// Whether this process may execute `feature`. Constant-folds to `true` in a
/// build that compiles the feature in, so a kernel's check costs nothing there.
#[inline(always)]
pub fn native(feature: Feature) -> bool {
    feature.compiled_in() || host().has(feature)
}

/// Refuse `arm` unless this process may execute `feature`.
pub fn require(feature: Feature, arm: &str) -> Result<()> {
    if native(feature) {
        Ok(())
    } else {
        host().require(feature, arm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_feature_refuses_the_arm_by_name() {
        let err = Features::empty()
            .require(Feature::DotProd, "q8_0 sdot")
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "CPU arm `q8_0 sdot` refused: this CPU does not execute `dotprod`, \
             and the arm is not emulated in software (CPU features: none)"
        );
        let present = Features::empty().with(Feature::DotProd);
        assert!(present.require(Feature::DotProd, "q8_0 sdot").is_ok());
        assert!(present.require(Feature::I8mm, "smmla").is_err());
    }

    #[test]
    fn names_round_trip_and_bits_are_distinct() {
        let mut all = Features::empty();
        for f in Feature::ALL {
            assert_eq!(Feature::from_name(f.name()), Some(f));
            assert!(!all.has(f), "{f} shares a bit with an earlier feature");
            all = all.with(f);
        }
        assert_eq!(all.iter().count(), Feature::ALL.len());
        assert_eq!(
            all.without(Feature::Fhm).iter().count(),
            Feature::ALL.len() - 1
        );
    }

    #[test]
    fn the_disable_list_parses_and_refuses_unknown_names() -> Result<()> {
        let set = Features::parse_list(" dotprod, fp16 ,")?;
        assert_eq!(
            set,
            Features::empty().with(Feature::DotProd).with(Feature::Fp16)
        );
        assert_eq!(set.to_string(), "dotprod fp16");
        assert_eq!(Features::parse_list("")?, Features::empty());
        let err = Features::parse_list("dotpod").unwrap_err().to_string();
        assert!(err.contains("`dotpod`"), "{err}");
        Ok(())
    }

    /// Live only when the process runs with `LLOOM_CPU_DISABLE` set: every
    /// feature it names is withdrawn unless the build compiles it in, and the
    /// real [`require`] refuses it.
    #[test]
    fn the_host_set_honours_the_disable_list() {
        let Ok(list) = std::env::var("LLOOM_CPU_DISABLE") else {
            eprintln!("LLOOM_CPU_DISABLE unset: nothing to check");
            return;
        };
        let disabled = Features::parse_list(&list).expect("a valid list");
        assert_ne!(
            disabled,
            Features::empty(),
            "set but empty: nothing to check"
        );
        for feature in disabled.iter() {
            assert_eq!(native(feature), feature.compiled_in(), "{feature}");
            if !feature.compiled_in() {
                let err = require(feature, "disabled arm").unwrap_err().to_string();
                assert!(err.contains(&format!("`{feature}`")), "{err}");
            }
        }
        eprintln!("host: {}", host());
    }

    /// The detector against the OS's own report, feature by feature. On an M1
    /// the rows mix present (`dotprod`, `fp16`, `fhm`) and absent (`bf16`,
    /// `i8mm`), which a detector returning a constant cannot match.
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    #[test]
    fn detection_agrees_with_sysctl() {
        let rows = [
            (Feature::DotProd, "hw.optional.arm.FEAT_DotProd"),
            (Feature::Fp16, "hw.optional.arm.FEAT_FP16"),
            (Feature::Fhm, "hw.optional.arm.FEAT_FHM"),
            (Feature::I8mm, "hw.optional.arm.FEAT_I8MM"),
            (Feature::Bf16, "hw.optional.arm.FEAT_BF16"),
        ];
        let detected = Features::detect();
        let mut seen = [false; 2];
        for (feature, key) in rows {
            let out = std::process::Command::new("sysctl")
                .args(["-n", key])
                .output()
                .expect("sysctl runs");
            let reported = String::from_utf8_lossy(&out.stdout).trim() == "1";
            assert_eq!(detected.has(feature), reported, "{feature} vs {key}");
            seen[reported as usize] = true;
        }
        eprintln!("detected: {detected}");
        // Every Apple silicon Mac executes `sdot`.
        assert!(detected.has(Feature::DotProd));
        assert!(
            seen[1],
            "no feature reported present: the comparison is blind"
        );
    }
}
