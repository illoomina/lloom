//! Load-time warnings: the one channel a load reports a degradation through.
//!
//! A load that takes a slower path than the one asked for — a kernel the head
//! dim or dtype does not instantiate, a type the GPU only emulates, a
//! readback arm clamped off, a configuration measured slower than its
//! alternative — prints its line on stderr as it always has, and keeps a
//! [`LoadWarning`] so a caller can render the list without reading stderr
//! (`lloom chat`'s loaded line, `serve`'s `/health` and model card).

use std::fmt;

/// What degraded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoadWarningKind {
    /// An attention phase takes another kernel than the arm requested: the
    /// nearest fused one, or the composed attention when none covers the
    /// shape (`lfm2::AttnRoute`).
    AttentionRoute,
    /// A dtype this GPU executes only through software emulation, admitted
    /// by the operator's opt-in (`kernels::metal::arms::EmulatedArms`).
    EmulatedType,
    /// A readback or commit arm the run asked for does not run as asked.
    Readback,
    /// A speculative configuration measured slower than decoding without it.
    Speculation,
}

impl LoadWarningKind {
    pub const ALL: [LoadWarningKind; 4] = [
        LoadWarningKind::AttentionRoute,
        LoadWarningKind::EmulatedType,
        LoadWarningKind::Readback,
        LoadWarningKind::Speculation,
    ];

    /// The `kind` a structured reader keys on.
    pub fn name(self) -> &'static str {
        match self {
            LoadWarningKind::AttentionRoute => "attention_route",
            LoadWarningKind::EmulatedType => "emulated_type",
            LoadWarningKind::Readback => "readback",
            LoadWarningKind::Speculation => "speculation",
        }
    }
}

/// One load-time degradation: its kind and the sentence stderr carried.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadWarning {
    pub kind: LoadWarningKind,
    pub message: String,
}

impl LoadWarning {
    pub fn new(kind: LoadWarningKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// The warnings as one `[loaded …]`-line field: `none`, or each
    /// `kind: message` joined by ` | `.
    pub fn line(warnings: &[LoadWarning]) -> String {
        if warnings.is_empty() {
            return "none".to_string();
        }
        warnings
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" | ")
    }
}

impl fmt::Display for LoadWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind.name(), self.message)
    }
}

/// The emulated types `device` has admitted, one warning each, in the words
/// its gate printed. Empty off Metal and on a device that emulates nothing.
pub fn emulated(device: &lloom_tensor::Device) -> Vec<LoadWarning> {
    #[cfg(feature = "metal")]
    if let lloom_tensor::Device::Metal(m) = device {
        return m
            .emulated_arms()
            .admitted()
            .iter()
            .map(|f| {
                LoadWarning::new(
                    LoadWarningKind::EmulatedType,
                    lloom_tensor::kernels::metal::arms::emulated_warning(f),
                )
            })
            .collect();
    }
    let _ = device;
    Vec::new()
}
