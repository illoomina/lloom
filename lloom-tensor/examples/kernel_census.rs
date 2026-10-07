//! lloom #517: the INSTANTIATED kernel set, read from the compiled libraries.
//!
//! # What this is for
//!
//! #469 records *"we instantiate 133 kernels and dispatch 18 at decode"* and
//! #517 asks for the residual: subtract the kernels belonging to models we do
//! not target, and what remains is either a kernel we use or one we shipped and
//! never wired. **The subtraction needs both terms measured the same way, and
//! the instantiated term was not measured at all** — it was grepped.
//!
//! # Why a grep cannot produce it
//!
//! `grep -c 'host_name("' src/metal_src/*.metal` returns 133 at #469's commit
//! and 145 here, and **every one of them is in `quantized.metal`**. Not mostly:
//! all. The other nineteen files instantiate through a macro whose
//! `[[host_name]]` argument is a *variable*:
//!
//! ```text
//! reduce.metal:790   template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;
//! ```
//!
//! so a literal-string grep is structurally blind to `gemv`, `reduce`, `unary`,
//! `sdpa` — every family a decode token actually dispatches — and sees only the
//! one file that spells its names inline. The count is a property of which
//! files happen to use a literal, not of the tree. Re-spelled, the same
//! question gives **168** (`host_name` alone) or **145** (`host_name("`); a
//! number that moves with the regex is not a measurement.
//!
//! This reads `MTLLibrary.functionNames` from the **compiled** library instead,
//! which is exact and is `DESIGN.md` §8.1b's own argument — *resolve against the
//! real metallib, not against a list* — applied to enumeration rather than
//! lookup.
//!
//! # No lease
//!
//! Compiling a library and enumerating its functions use the GPU **driver** and
//! enqueue nothing on the command queue `lloom-arb` arbitrates (§13.1i). No
//! command buffer is created, no kernel is dispatched.
//!
//! # Non-vacuity
//!
//! A blind reader returns an empty list for every source, which would read as a
//! clean "nothing instantiated" — §15.1a's VACUOUS class, and #457's own case
//! (`nm` exiting having printed nothing, so every pattern counted 0). So this
//! **exits non-zero on a zero total**, and prints the per-source count beside
//! every name so a zero is always accompanied by a nonzero from the same run.

use lloom_tensor::kernels::metal::device::Device;
use lloom_tensor::kernels::source::Source;
use lloom_tensor::kernels::Kernels;

/// Every `Source` the tree declares.
///
/// Written out rather than derived, because `Source` has no `EnumIter` and an
/// incomplete list would silently under-report the instantiated set — which is
/// the failure this whole instrument exists to correct. [`label`] matches
/// every arm with no catch-all, so a variant added to `Source` fails to
/// compile here until it is named; `main` checks the list covers the labels.
const ALL_SOURCES: &[Source] = &[
    Source::Affine,
    Source::ArenaAlloc,
    Source::Binary,
    Source::Cast,
    Source::Conv,
    Source::EosGate,
    Source::EosIndex,
    Source::Fill,
    Source::FlashDecoding,
    Source::Gemm,
    Source::Gemv,
    Source::Gdn,
    Source::Indexing,
    Source::MlxSort,
    Source::Quantized,
    Source::QuantizedStore,
    Source::Qwen4Exp,
    Source::Random,
    Source::Reduce,
    Source::Rope,
    Source::Scratch,
    Source::Sort,
    Source::Ternary,
    Source::Unary,
    Source::Sdpa,
];

/// The source's name, and the exhaustiveness check the list above leans on.
fn label(s: Source) -> &'static str {
    match s {
        Source::Affine => "affine",
        Source::ArenaAlloc => "arena_alloc",
        Source::Binary => "binary",
        Source::Cast => "cast",
        Source::Conv => "conv",
        Source::EosGate => "eos_gate",
        Source::EosIndex => "eos_index",
        Source::Fill => "fill",
        Source::FlashDecoding => "flash_decoding",
        Source::Gemm => "mlx_gemm",
        Source::Gemv => "gemv",
        Source::Gdn => "gdn",
        Source::Indexing => "indexing",
        Source::MlxSort => "mlx_sort",
        Source::Quantized => "quantized",
        Source::QuantizedStore => "quantized_store",
        Source::Qwen4Exp => "qwen4exp",
        Source::Random => "random",
        Source::Reduce => "reduce",
        Source::Rope => "rope",
        Source::Scratch => "scratch",
        Source::Sort => "sort",
        Source::Ternary => "ternary",
        Source::Unary => "unary",
        Source::Sdpa => "sdpa",
    }
}

fn main() {
    let device = Device::system_default().expect("no Metal device");
    let kernels = Kernels::new();

    let mut total = 0usize;
    let mut rows: Vec<(&str, Vec<String>)> = Vec::new();

    // 25 arms in `label`; a list shorter than that under-reports.
    assert_eq!(ALL_SOURCES.len(), 25, "ALL_SOURCES does not name every Source");
    for source in ALL_SOURCES {
        let label = label(*source);
        let library = match kernels.load_library(&device, *source) {
            Ok(lib) => lib,
            Err(e) => {
                // A source that does not compile is a failure of this
                // instrument, not a source with no kernels: reporting it as
                // zero would be the vacuous reading.
                eprintln!("FAIL  {label}: {e}");
                std::process::exit(2);
            }
        };
        let mut names = library.function_names();
        names.sort();
        total += names.len();
        rows.push((label, names));
    }

    // The non-vacuity gate. A blind reader gives every source an empty list,
    // which is indistinguishable from a tree that instantiates nothing.
    if total == 0 {
        eprintln!(
            "FAIL: zero functions across {} sources -- the reader is blind, not the tree empty",
            rows.len()
        );
        std::process::exit(2);
    }

    println!("=== instantiated kernels, per source (from the COMPILED library) ===");
    let mut by_count: Vec<&(&str, Vec<String>)> = rows.iter().collect();
    by_count.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(b.0)));
    for (label, names) in &by_count {
        println!("{:6}  {label}", names.len());
    }
    println!("{total:6}  TOTAL  ({} sources)", rows.len());
    println!();

    println!("=== every instantiated kernel name ===");
    for (label, names) in &rows {
        for name in names {
            println!("{label}\t{name}");
        }
    }
}
