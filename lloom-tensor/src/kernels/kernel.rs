use crate::kernels::source::{
    AFFINE, ARENA_ALLOC, BINARY, CAST, CONV, EOS_INDEX, FILL, FLASH_DECODING, GEMV, INDEXING,
    MERGED_MEMBERS, MERGED_NON_MLX, MLX_GEMM, MLX_SORT, QUANTIZED, RANDOM, REDUCE, ROPE, SCRATCH,
    SDPA, SORT, TERNARY, UNARY,
};
use crate::kernels::utils::get_env_bool;
use crate::kernels::{
    ComputePipeline, ConstantValues, Device, Function, Library, MTLCompileOptions,
    MTLMathFloatingPointFunctions, MTLMathMode, MetalKernelError, Source,
};
use objc2::available;
use objc2::rc::Retained;
use objc2_foundation::NSString;
use parking_lot::RwLock;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum KernelName {
    Ref(&'static str),
    Value(String),
}

impl AsRef<str> for KernelName {
    fn as_ref(&self) -> &str {
        match self {
            Self::Ref(r) => r,
            Self::Value(v) => v.as_str(),
        }
    }
}

impl std::hash::Hash for KernelName {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Ref(r) => r.hash(state),
            Self::Value(v) => v.hash(state),
        }
    }
}

impl PartialEq for KernelName {
    fn eq(&self, other: &Self) -> bool {
        let v1: &str = self.as_ref();
        let v2: &str = other.as_ref();
        v1 == v2
    }
}

impl Eq for KernelName {}

impl From<&'static str> for KernelName {
    fn from(value: &'static str) -> Self {
        Self::Ref(value)
    }
}

impl From<String> for KernelName {
    fn from(value: String) -> Self {
        Self::Value(value)
    }
}

/// What a [`Library`] is cached under.
///
/// **Not `Source`**, because under the merged arm (#477) sixteen `Source`s
/// share ONE compiled library and caching per-`Source` would compile the same
/// 16 800-line text sixteen times — which would be the arm's cost without its
/// point, since the optimizer would still see each unit separately in the
/// sense that matters and the process would pay §11.3g's +730 ms sixteen times
/// over.
///
/// The pipeline cache stays keyed on `Source` deliberately: a pipeline is a
/// `(library, function name, constants)` and the *function name* is what
/// differs between two `Source`s resolving to the same library. Keying
/// pipelines on the library key instead would collide two kernels of the same
/// name from different sources — which cannot happen today, but the narrower
/// key costs nothing and does not have to be re-argued if it ever can.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum LibraryKey {
    /// One library per source: the shipping default, and what every
    /// `__compute` figure in the corpus was measured under.
    PerSource(Source),
    /// The non-MLX sources as one translation unit (#477).
    MergedNonMlx,
}

type Libraries = HashMap<LibraryKey, Library>;
type Pipelines = HashMap<(Source, KernelName, Option<ConstantValues>), ComputePipeline>;

#[derive(Debug)]
pub struct Kernels {
    libraries: RwLock<Libraries>,
    pipelines: RwLock<Pipelines>,
}

impl Default for Kernels {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernels {
    pub fn new() -> Self {
        let libraries = RwLock::new(Libraries::new());
        let pipelines = RwLock::new(Pipelines::new());
        Self {
            libraries,
            pipelines,
        }
    }

    /// Which library a [`Source`] resolves to, and the text to compile for it.
    ///
    /// Under the default arm this is the identity: one library per source, the
    /// same nineteen `newLibraryWithSource` calls that have always shipped.
    /// Under [`merged_translation_unit`] the non-MLX sources collapse
    /// onto one key and one text (#477).
    fn library_for(&self, source: Source) -> (LibraryKey, &'static str) {
        Self::library_for_arm(merged_translation_unit(), source)
    }

    /// [`Self::library_for`]'s decision, with the arm passed in.
    ///
    /// **Split out so a test can exercise BOTH arms in one process.**
    /// `merged_translation_unit()` is a `OnceLock` over an environment variable
    /// — read once per process and shared by every test in this binary — so a
    /// test that set the variable would decide the arm for whichever test ran
    /// next. `additional_compiler_arguments_reach_a_validating_driver` records
    /// the same constraint for `LLOOM_434_METAL_CFLAGS` and solves it the same
    /// way.
    ///
    /// **Calling this from the test is the point.** A test asserting that
    /// `MERGED_NON_MLX` compiles and resolves its names **passes when the flag
    /// is honoured nowhere** — measured, as this change's mutation arm 3, and
    /// that is #500/#536's exact shape: a mechanism built, correct, tested and
    /// reachable by nothing. `CONTRIBUTING.md` §3.1a rule 3 — a test of the
    /// text is not a test of the routing.
    fn library_for_arm(merged: bool, source: Source) -> (LibraryKey, &'static str) {
        if merged && MERGED_MEMBERS.contains(&source) {
            (LibraryKey::MergedNonMlx, MERGED_NON_MLX)
        } else {
            (LibraryKey::PerSource(source), Self::library_source(source))
        }
    }

    fn library_source(source: Source) -> &'static str {
        match source {
            Source::Affine => AFFINE,
            Source::ArenaAlloc => ARENA_ALLOC,
            Source::Binary => BINARY,
            Source::Cast => CAST,
            Source::Conv => CONV,
            Source::EosIndex => EOS_INDEX,
            Source::Fill => FILL,
            Source::FlashDecoding => FLASH_DECODING,
            Source::Gemm => MLX_GEMM,
            Source::Gemv => GEMV,
            Source::Indexing => INDEXING,
            Source::MlxSort => MLX_SORT,
            Source::Quantized => QUANTIZED,
            Source::Random => RANDOM,
            Source::Reduce => REDUCE,
            Source::Rope => ROPE,
            Source::Scratch => SCRATCH,
            Source::Sort => SORT,
            Source::Ternary => TERNARY,
            Source::Unary => UNARY,
            Source::Sdpa => SDPA,
        }
    }

    /// Load the give library from its [`source`].
    /// If this has been previously loaded it will just fetch it from cache.
    pub fn load_library(
        &self,
        device: &Device,
        source: Source,
    ) -> Result<Library, MetalKernelError> {
        let (key, source_content) = self.library_for(source);

        if let Some(lib) = self.libraries.read().get(&key) {
            return Ok(lib.clone());
        }

        let mut libraries = self.libraries.write();
        // Recheck after acquiring the write lock in case another thread inserted it.
        if let Some(lib) = libraries.get(&key) {
            Ok(lib.clone())
        } else {
            let lib = {
                let compile_options = get_compile_options();
                device
                    .new_library_with_source(source_content, Some(&compile_options))
                    .map_err(|e| MetalKernelError::LoadLibraryError(e.to_string()))?
            };
            libraries.insert(key, lib.clone());
            Ok(lib)
        }
    }

    fn load_function(
        &self,
        device: &Device,
        source: Source,
        name: &str,
        constants: Option<&ConstantValues>,
    ) -> Result<Function, MetalKernelError> {
        let func = self
            .load_library(device, source)?
            .get_function(name, constants)?;
        Ok(func)
    }

    /// Load the give pipeline
    /// loads the library from source, then gets the function [`name`] from
    /// that source
    pub fn load_pipeline_with_constants(
        &self,
        device: &Device,
        source: Source,
        name: impl Into<KernelName>,
        constants: Option<ConstantValues>,
    ) -> Result<ComputePipeline, MetalKernelError> {
        let key = (source, name.into(), constants);
        if let Some(pipeline) = self.pipelines.read().get(&key) {
            return Ok(pipeline.clone());
        }

        let mut pipelines = self.pipelines.write();
        // Recheck after acquiring the write lock in case another thread inserted it.
        if let Some(pipeline) = pipelines.get(&key) {
            Ok(pipeline.clone())
        } else {
            let (source, name, constants) = key;
            let func = self.load_function(device, source, name.as_ref(), constants.as_ref())?;
            // Record the function name here, where it is already in hand: it is
            // how an executor attributes a dispatch to a kernel, and it is
            // cached with the pipeline so the cost is once per variant, not per
            // dispatch.
            let pipeline = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| MetalKernelError::FailedToCreatePipeline(e.to_string()))?
                .with_name(name.as_ref());
            pipelines.insert((source, name, constants), pipeline.clone());

            Ok(pipeline)
        }
    }

    /// Load the give pipeline
    /// loads the library from source, then gets the function [`name`] from
    /// that source (without constants)
    pub fn load_pipeline(
        &self,
        device: &Device,
        source: Source,
        name: impl Into<KernelName>,
    ) -> Result<ComputePipeline, MetalKernelError> {
        self.load_pipeline_with_constants(device, source, name, None)
    }

    /// How many distinct Metal libraries this cache has compiled, and how many
    /// pipelines it has created.
    ///
    /// # Why this is the engagement proof for `MergedUnit`, and the census is not
    ///
    /// **The dispatch census cannot see this axis, and that is a property of
    /// the axis rather than a gap in the instrument.** `record_dispatch`
    /// (§11.2f) counts *which kernels ran*; the merged arm changes *which
    /// library each kernel was compiled from* and dispatches an identical set.
    /// Measured, the two arms' censuses are **byte-identical** — which is the
    /// correct outcome and is exactly why a census-only engagement claim would
    /// be vacuous here.
    ///
    /// §12.2i's fusions are the contrasting case: there the census IS the
    /// proof, because a fusion is visible precisely as *a name appearing and
    /// the name it replaces going to zero*. **Reading one axis's instrument
    /// onto another is §15.1a's INEXPRESSIVE class**, and #522 paid a run for
    /// it — a whole-process bracket that could not separate the pair it was
    /// pointed at.
    ///
    /// The library count is the quantity the merged arm moves **by
    /// construction**: sixteen `Source`s that compiled sixteen libraries
    /// compile one.
    ///
    /// **Both numbers are returned together deliberately** (§14.5d's licensing
    /// rule): a zero — or a suspiciously small — library count is only readable
    /// beside a nonzero pipeline count from the same call. A cache that had
    /// compiled nothing would report `(0, 0)`, which is what a broken probe
    /// reports, and the pipeline total is what distinguishes them.
    pub fn cache_sizes(&self) -> (usize, usize) {
        let libraries = self.libraries.read().len();
        let pipelines = self.pipelines.read().len();
        (libraries, pipelines)
    }
}

/// Issue #434's driver-flag arm: pass a flag string through to the Metal
/// compiler's clang driver, so `-flto` and `-frelaxed-math` can be A/B'd.
///
/// **Unset is what ships**, and unset means the property is never touched — so
/// the default is byte-for-byte what shipped before this existed (`DESIGN.md`
/// §7.1a: no default is flipped without its own argued decision).
///
/// `additionalCompilerArguments` takes an **`NSString`**, not an array: an
/// array reaches `-[NSConstantArray lengthOfBytesUsingEncoding:]` and throws
/// `unrecognized selector`. It is also **absent from `MTLLibrary.h`** and from
/// `objc2-metal`'s generated bindings, which is why this is a raw `msg_send!`
/// rather than a typed setter — §2.3.3b found the property the same way, by
/// enumerating `MTLCompileOptions` from the Objective-C runtime. Probed on this
/// machine the runtime reports `T@"NSString",C,D,N`.
///
/// The driver **validates**: `-fbogus-not-a-flag` is rejected while real flags
/// compile, which is the engagement proof at the mechanism level (§2.4, after
/// #69's vacuous determinism run). A rejected flag surfaces as a
/// `LoadLibraryError` from `new_library_with_source`, so a typo fails loudly
/// rather than silently running the baseline.
///
/// **A flag reaching the driver is not a flag changing codegen**, and this
/// function cannot tell the difference. The arm-level engagement proof is a
/// `__compute` section-size comparison between the built arms — see
/// `measurements/probes/issue-434/`.
fn additional_compiler_arguments() -> Option<&'static str> {
    use std::sync::OnceLock;
    static ARGS: OnceLock<Option<String>> = OnceLock::new();
    ARGS.get_or_init(|| match std::env::var("LLOOM_434_METAL_CFLAGS") {
        // An empty value is a mistake worth failing on rather than silently
        // treating as unset: it is what an unset shell variable expands to, so
        // accepting it would let `LLOOM_434_METAL_CFLAGS=$UNSET` run the
        // baseline while the batch script believes it ran an arm.
        Ok(v) if v.trim().is_empty() => panic!(
            "LLOOM_434_METAL_CFLAGS is set but empty. Unset it to run the \
             default arm; an empty value is almost always an unexpanded \
             shell variable, which would silently run the baseline (§2.4)."
        ),
        Ok(v) => Some(v),
        Err(std::env::VarError::NotPresent) => None,
        Err(e) => panic!("LLOOM_434_METAL_CFLAGS is not valid UTF-8: {e}"),
    })
    .as_deref()
}

/// What [`additional_compiler_arguments`] resolved to, **for the record**.
///
/// # Why this exists, and it is the gap this pass found
///
/// The passthrough has existed since #434, and this string can decide *which
/// kernels exist* — which is what makes it worth recording rather than merely
/// worth having.
///
/// **The instance that used to demonstrate that is gone, and the axis is not.**
/// Until #516 the demonstration was §12.2i's `GemvResidual=on`, which did not
/// merely benefit from `-DLLOOM_509_RESIDUAL_EPILOGUE` but *required* it —
/// without the `-D` the `_residual` name was absent from the metallib and the
/// forward pass failed at pipeline load. #516 removed that `-D`, because it
/// coupled a run-time `Config` field to a build-time flag (two sources of truth
/// for one decision), so **no shipping arm passes a `-D` today** and this axis
/// records `none` on every default run.
///
/// That is a weaker claim than the one this comment used to make, and it is
/// stated rather than quietly dropped (§2.6a: a justification that stops holding
/// is not thereby wrong to have recorded). The axis stays for three reasons that
/// do not depend on the `-D`: `-flto` is selectable through it and **is off by
/// default in the Metal compiler** (§13.1c measures it moving `__compute` on 6
/// of 18 decode kernels); the driver **validates**, so a typo is a
/// `LoadLibraryError` rather than a silent baseline; and a future `-D` would
/// otherwise re-open §2.3.9's `MathMode` hole the moment it landed.
///
/// **And the recording gap it was built for was real.** `axis_pairs()` emitted nineteen axes plus
/// `Batch` and none of them was the compile flags, so two runs that compiled
/// **different metallibs** produced **identical `config=[…]` lines** and a
/// reader pooling them would be comparing two different programs. That is
/// §2.3.9's `MathMode` failure exactly — *an axis switchable by environment
/// variable, in neither the registry nor the config line, invisible for the
/// life of the project* — at a second variable, and §14.5a names the shape
/// after #455's flags went unrecorded.
///
/// **A `-D` is compile-tier and recordable; a `const bool` in a header is
/// construction-tier and leaves no trace of what was built** (§14.5a). That
/// asymmetry is why this is worth an axis rather than a tidier in-source
/// toggle.
///
/// Read from the same `OnceLock` the compiler reads rather than echoed from a
/// flag: §7.1a-i's *present-and-wrong* case is worse than absence, because a
/// false arm compares **equal** to real runs of that arm.
pub fn metal_compile_flags() -> Option<&'static str> {
    additional_compiler_arguments()
}

/// Issue #477's merged-translation-unit arm: compile the sixteen non-MLX
/// sources as ONE unit rather than as sixteen (`DESIGN.md` §13.1e, §13.1j).
///
/// **Off is what ships**, and off is byte-for-byte the nineteen-library path
/// that produced every `__compute` figure in the corpus (§7.1a: no default is
/// flipped without its own argued decision).
///
/// # Why this is a flag and not a replacement
///
/// #452 measured the unit boundary **inert** — no kernel's `__compute` moved on
/// 17 of 17 — and its own mechanism says why: a template is instantiated per
/// use-site, so there is no cross-unit call for a merge to devirtualise. That
/// measurement was taken by compiling concatenations offline, **not** by
/// running the engine on a merged library, and it predates both #458's helper
/// hoist and #466/#477's `strided_indexer` rename. So the arm exists to be
/// *measured through the shipping path*, and the honest prior from §13.1e is
/// that it moves nothing.
///
/// **A null here is a result and not a failure.** What would make it
/// interesting is `-flto` composed with it: LTO is off by default in the Metal
/// compiler (§13.1c measures `-flto` moving `__compute` on 6 of 18 decode
/// kernels), and whole-program optimisation over one unit is a different
/// question from whole-program optimisation over sixteen. Both arms are
/// selectable together — this flag and `LLOOM_434_METAL_CFLAGS` are
/// independent.
///
/// # The failure mode this cannot have
///
/// A merged library must still resolve every kernel name the split libraries
/// did, and Metal resolves compute functions **by name at runtime**. A name
/// that failed to make it into the merged text surfaces as a
/// `FailedToCreatePipeline`/`get_function` error at the first dispatch that
/// wants it — loudly, at load, rather than as a silent fallback to the
/// baseline. That is the same two-level shape §12.2i's residual epilogue takes,
/// and it is why this returns a bool rather than silently degrading.
/// # Why this does not use `get_env_bool`
///
/// That helper maps every unrecognised value to `false`
/// (`utils.rs`: `"true"|"t"|"yes"|"y"|"1"` and `_ => false`), so
/// `LLOOM_477_MERGED_UNIT=on` — a spelling a batch script will produce —
/// **silently runs the baseline** while the script believes it ran the arm.
/// That is #69's vacuous-run shape (§2.4) and is precisely what
/// `LLOOM_434_METAL_CFLAGS` panics on an empty value to avoid. An A/B arm owes
/// a loud failure on a value it does not understand; a general-purpose feature
/// toggle does not, which is why the helper is right for its callers and wrong
/// for this one.
pub fn merged_translation_unit() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("LLOOM_477_MERGED_UNIT") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(v) => match v.trim() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            other => panic!(
                "LLOOM_477_MERGED_UNIT={other:?} is not a value this arm \
                 understands. Use 1/true/yes/on or 0/false/no/off, or unset it \
                 for the default (per-source) arm. Accepting an unknown value \
                 as `off` would let a batch script run the baseline while \
                 believing it ran the merged arm (§2.4)."
            ),
        },
        Err(e) => panic!("LLOOM_477_MERGED_UNIT is not valid UTF-8: {e}"),
    })
}

fn get_compile_options() -> Retained<MTLCompileOptions> {
    let compile_options = MTLCompileOptions::new();
    //unsafe { compile_options.setEnableLogging(true) };

    let fast_math_enabled = get_env_bool("CANDLE_METAL_ENABLE_FAST_MATH", true);
    // Ref availability:
    // https://developer.apple.com/documentation/metal/mtlcompileoptions/mathmode
    if available!(macos = 15, ios = 18) {
        if fast_math_enabled {
            compile_options.setMathMode(MTLMathMode::Fast);
            compile_options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Fast);
        } else {
            compile_options.setMathMode(MTLMathMode::Relaxed);
            compile_options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        }
    } else {
        // For older OS versions we use the old api
        #[allow(deprecated)]
        compile_options.setFastMathEnabled(fast_math_enabled);
    }

    // Deliberately last, and deliberately only when set: an untouched property
    // is not the same as a property set to the empty string, and only the
    // former is the shipped default.
    if let Some(flags) = additional_compiler_arguments() {
        let ns = NSString::from_str(flags);
        // SAFETY: `setAdditionalCompilerArguments:` is declared on
        // `MTLCompileOptions` by the Objective-C runtime (probed: the property
        // reports `T@"NSString",C,D,N`) and is absent from the generated
        // bindings, so it is unreachable through a typed setter. The selector
        // takes exactly one object argument of type `NSString *`, which is what
        // is passed; the property is `copy`, so the receiver does not borrow
        // `ns` beyond the call.
        unsafe {
            let _: () = objc2::msg_send![
                &*compile_options,
                setAdditionalCompilerArguments: &*ns
            ];
        }
    }

    compile_options
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn concurrent_cache_loads_insert_once() {
        const THREADS: usize = 8;
        const WARM_HITS: usize = 100;

        let device = Device::system_default().unwrap();
        let kernels = Kernels::new();
        let barrier = Barrier::new(THREADS);

        let pipelines = std::thread::scope(|scope| {
            let handles = (0..THREADS)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        let pipeline = kernels
                            .load_pipeline(&device, Source::Unary, "cos_f32")
                            .unwrap();
                        for _ in 0..WARM_HITS {
                            let cached = kernels
                                .load_pipeline(&device, Source::Unary, "cos_f32")
                                .unwrap();
                            assert!(std::ptr::eq(pipeline.as_ref(), cached.as_ref()));
                            kernels.load_library(&device, Source::Unary).unwrap();
                        }
                        pipeline
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });

        assert!(pipelines
            .iter()
            .all(|pipeline| std::ptr::eq(pipelines[0].as_ref(), pipeline.as_ref())));
        assert_eq!(kernels.libraries.read().len(), 1);
        assert_eq!(kernels.pipelines.read().len(), 1);
    }

    /// Issue #434. The passthrough is exercised **directly** rather than
    /// through `LLOOM_434_METAL_CFLAGS`, because that variable is read once per
    /// process through a `OnceLock` and every test in this binary shares it —
    /// a test that set it would decide the arm for whichever test ran next.
    ///
    /// Three claims, and the third is the one that makes the other two mean
    /// something (`DESIGN.md` §15.1 #1, §11.3j: a comparison whose control
    /// cannot fail is not evidence):
    ///
    /// 1. a real flag compiles,
    /// 2. the default arm compiles,
    /// 3. a **bogus** flag is REJECTED — so the string demonstrably reaches a
    ///    real clang driver rather than being dropped. Without this arm, arms 1
    ///    and 2 are equally consistent with the property being ignored
    ///    entirely, which is #69's vacuous-run shape (§2.4).
    #[test]
    fn additional_compiler_arguments_reach_a_validating_driver() {
        let device = Device::system_default().unwrap();
        // A tiny source of our own: the point is the DRIVER's verdict, and
        // compiling a whole candle source per arm would price the test at
        // several seconds for no extra evidence.
        const SRC: &str = "#include <metal_stdlib>\n\
             using namespace metal;\n\
             kernel void k434(device float* o [[buffer(0)]],\n\
                              uint t [[thread_position_in_grid]]) { o[t] = (float)t; }\n";

        let compile_with = |flags: Option<&str>| {
            let options = MTLCompileOptions::new();
            if let Some(f) = flags {
                let ns = NSString::from_str(f);
                // SAFETY: as in `get_compile_options` — the selector is
                // declared by the ObjC runtime and takes one `NSString *`.
                unsafe {
                    let _: () = objc2::msg_send![&*options, setAdditionalCompilerArguments: &*ns];
                }
            }
            device.new_library_with_source(SRC, Some(&options))
        };

        assert!(
            compile_with(None).is_ok(),
            "the default arm must compile: it is what ships"
        );
        assert!(
            compile_with(Some("-flto")).is_ok(),
            "#434 arm A: -flto is accepted by the driver"
        );
        assert!(
            compile_with(Some("-frelaxed-math")).is_ok(),
            "#434 arm B: -frelaxed-math is accepted by the driver"
        );
        assert!(
            compile_with(Some("-fbogus-not-a-flag")).is_err(),
            "a bogus flag MUST be rejected. If it compiles, the string is not \
             reaching the compiler and every arm built through this path is \
             silently the baseline (§2.4, #69's vacuous determinism run)."
        );
    }

    /// Issue #477. **Every `Source` is either a merged member or MLX**, so a
    /// new variant cannot silently opt out of the arm.
    ///
    /// This is the check the arm needs and the one a reader would assume is
    /// unnecessary. A `Source` added to the enum and to neither set keeps its
    /// own library under the merged arm — which is *correct behaviour* and
    /// therefore **silent**: the engine works, the kernels resolve, and the
    /// merged arm quietly measures a merge of fifteen sources while the PR
    /// says sixteen. §11.3h's recurring lesson is that a list written once
    /// beside something that keeps changing is a list nobody re-checks, and
    /// `MERGED_MEMBERS` is exactly such a list.
    ///
    /// The `match` is what makes it mechanical rather than a count: a new
    /// variant fails to compile here, so the author has to decide which side it
    /// is on rather than defaulting to neither.
    #[test]
    fn merged_unit_covers_every_non_mlx_source() {
        // Exhaustive by construction: adding a `Source` variant makes this
        // arm fail to compile until it is classified. A `_ =>` catch-all here
        // would defeat the whole test.
        let classify = |s: Source| -> bool {
            match s {
                Source::Affine
                | Source::ArenaAlloc
                | Source::Binary
                | Source::Cast
                | Source::Conv
                | Source::EosIndex
                | Source::Fill
                | Source::FlashDecoding
                | Source::Gemv
                | Source::Indexing
                | Source::Quantized
                | Source::Random
                | Source::Reduce
                | Source::Rope
                | Source::Scratch
                | Source::Sort
                | Source::Ternary
                | Source::Unary => true,
                // The MLX cluster: mutually incompatible `BlockMMA`/`Limits`/
                // `Transform*` bodies and colliding function-constant indices
                // 200/201/300 (§13.1e). A port, not a merge (#409).
                Source::Gemm | Source::MlxSort | Source::Sdpa => false,
            }
        };

        for &s in MERGED_MEMBERS {
            assert!(
                classify(s),
                "{s:?} is in MERGED_MEMBERS but classified as MLX-cluster"
            );
        }

        // And the converse: everything classified as a merged member is in the
        // list. Without this arm the list could be a strict subset and the
        // test above would still pass.
        let all = [
            Source::Affine,
            Source::ArenaAlloc,
            Source::Binary,
            Source::Cast,
            Source::Conv,
            Source::EosIndex,
            Source::Fill,
            Source::FlashDecoding,
            Source::Gemm,
            Source::Gemv,
            Source::Indexing,
            Source::MlxSort,
            Source::Quantized,
            Source::Random,
            Source::Reduce,
            Source::Rope,
            Source::Scratch,
            Source::Sort,
            Source::Ternary,
            Source::Unary,
            Source::Sdpa,
        ];
        for s in all {
            assert_eq!(
                classify(s),
                MERGED_MEMBERS.contains(&s),
                "{s:?} disagrees between the classifier and MERGED_MEMBERS"
            );
        }

        // Non-vacuity: the counts are stated so a list silently emptied by a
        // bad edit fails here rather than passing over an empty iteration.
        assert_eq!(
            MERGED_MEMBERS.len(),
            18,
            "18 non-MLX sources (#466 §4, +rope #659, +eos_index #702's tree)"
        );
        assert_eq!(all.len(), 21, "21 Source variants");
    }

    /// Issue #477. The merged text **compiles**, and it contains kernels from
    /// sources that are separate libraries under the default arm.
    ///
    /// The second half is what makes this more than a syntax check. A merged
    /// text that compiled but had lost a body would still pass a bare
    /// "does it compile", and the failure would surface much later as a
    /// pipeline-creation error on whichever kernel went missing.
    #[test]
    fn merged_translation_unit_compiles_and_carries_both_families() {
        let device = Device::system_default().unwrap();
        let options = MTLCompileOptions::new();

        let lib = device
            .new_library_with_source(MERGED_NON_MLX, Some(&options))
            .expect(
                "the merged unit must compile: #466 §4 verified the non-MLX \
                 sources compile as one translation unit with 0 errors",
            );

        // Two names from two DIFFERENT sources that are separate libraries
        // under the default arm. If the merge dropped a body, one of these is
        // absent and the arm would silently be a subset.
        //
        // `gemv` is the source #477 stage 3 exists for (167 of 539 decode
        // dispatches); `silu_f16` is `unary`'s, and the pair is exactly what
        // §12.2d's epilogue fusion spans.
        for name in ["silu_f16", "copy2d_f16"] {
            assert!(
                lib.get_function(name, None).is_ok(),
                "{name} must resolve from the merged library; a missing name \
                 here is a body the merge dropped"
            );
        }

        // Non-vacuity: a name that is certainly NOT in the library must fail,
        // or `get_function` returning Ok for everything would make the loop
        // above prove nothing (§14.5d's licensing rule: a zero is not evidence
        // unless something nonzero came from the same command).
        assert!(
            lib.get_function("no_such_kernel_477", None).is_err(),
            "a bogus name must fail to resolve, or the assertions above are \
             consistent with the lookup being ignored entirely"
        );
    }

    /// Issue #477. **The flag actually ROUTES**, which the two tests above do
    /// not establish.
    ///
    /// This arm exists because writing it was not optional: run against a tree
    /// where `library_for_arm`'s condition is dead (`if false && …`), the two
    /// tests above **both pass**. That mutation is #500's and #536's shape —
    /// built, correct, tested, and dispatched by nothing — and it is the one
    /// this whole change exists to avoid reproducing. A test of the merged
    /// TEXT is not a test of the merged ARM.
    ///
    /// Asserted in both directions, because only one of them is the claim:
    ///
    /// - **merged on** → every non-MLX source resolves to ONE key and the
    ///   merged text, and the three MLX sources still resolve per-source. A
    ///   change that merged everything would pass a one-directional test and
    ///   break `mlx_gemm`'s function constants (§13.1e's index collision).
    /// - **merged off** → every source resolves to its own key and its own
    ///   text, *byte-identically to what shipped before this change existed*.
    ///   That is the half §7.1a asks for: the default arm is the old path, not
    ///   a new path that happens to agree.
    #[test]
    fn the_merged_flag_routes_and_the_default_arm_is_the_old_path() {
        let all = [
            Source::Affine,
            Source::ArenaAlloc,
            Source::Binary,
            Source::Cast,
            Source::Conv,
            Source::EosIndex,
            Source::Fill,
            Source::FlashDecoding,
            Source::Gemm,
            Source::Gemv,
            Source::Indexing,
            Source::MlxSort,
            Source::Quantized,
            Source::Random,
            Source::Reduce,
            Source::Rope,
            Source::Scratch,
            Source::Sort,
            Source::Ternary,
            Source::Unary,
            Source::Sdpa,
        ];

        // --- arm: merged ON ---
        let mut merged_keys = std::collections::HashSet::new();
        let mut n_merged = 0;
        for s in all {
            let (key, text) = Kernels::library_for_arm(true, s);
            if MERGED_MEMBERS.contains(&s) {
                assert_eq!(
                    key,
                    LibraryKey::MergedNonMlx,
                    "{s:?} is a merged member and must route to the merged key \
                     when the arm is on -- a per-source key here is the silent \
                     fallback (#500/#536)"
                );
                // Compared by CONTENT, not by address. `MERGED_NON_MLX` is a
                // `const` (`source.rs`), so every mention is a fresh rvalue and
                // the compiler is free to materialise more than one copy of the
                // `concat!` -- `std::ptr::eq` then compares a codegen accident
                // rather than the routing, and it read FALSE here on a tree
                // where the routing is correct (#536: the `LibraryKey`
                // assertion above passes on the same call).
                //
                // The identity that matters is *which text this source
                // compiles*, and the discriminating half is the second
                // assertion: equality with the merged text alone would also
                // hold if `MERGED_NON_MLX` were somehow a source's own text, so
                // the arm is pinned against `library_source` as well.
                assert_eq!(
                    text, MERGED_NON_MLX,
                    "{s:?} must compile the MERGED text, not its own"
                );
                assert_ne!(
                    text,
                    Kernels::library_source(s),
                    "{s:?} routed to the merged key but compiled its OWN text -- \
                     the merge is not happening"
                );
                n_merged += 1;
            } else {
                assert_eq!(
                    key,
                    LibraryKey::PerSource(s),
                    "{s:?} is MLX-cluster and must stay per-source even with \
                     the arm on (§13.1e: constants 200/201/300 collide)"
                );
            }
            merged_keys.insert(key);
        }
        assert_eq!(n_merged, 18, "18 sources must route to the merged library");
        // 1 merged key + 3 MLX keys. A count of 20 would mean nothing merged.
        assert_eq!(
            merged_keys.len(),
            4,
            "the merged arm must collapse 20 libraries to 4, or it is not \
             merging anything"
        );

        // --- arm: merged OFF, and it must be the OLD path exactly ---
        let mut default_keys = std::collections::HashSet::new();
        for s in all {
            let (key, text) = Kernels::library_for_arm(false, s);
            assert_eq!(
                key,
                LibraryKey::PerSource(s),
                "with the arm off every source keeps its own library"
            );
            // By content, for the reason the merged arm above gives: these are
            // `const` strs, so an address comparison tests codegen rather than
            // routing. This arm happens to pass either way -- both sides call
            // `library_source` -- which is exactly why it is worth changing
            // together with the one that did not.
            //
            // **This arm's content comparison rests on a fact worth stating,
            // because it could stop holding** (#516). Unlike the merged arm --
            // where no member can equal the merged concatenation -- this
            // asserts a source gets *its own* text, so equality would be weaker
            // than identity if two sources shared content. Checked rather than
            // assumed: the `.metal` files hash to distinct contents,
            // so content uniquely identifies a source here. A future duplicate
            // silently weakens this assertion, and would want the `assert_ne!`
            // treatment the merged arm above uses.
            assert_eq!(
                text,
                Kernels::library_source(s),
                "{s:?} must compile its own source text with the arm off -- \
                 this is the byte-for-byte shipping path (§7.1a)"
            );
            default_keys.insert(key);
        }
        assert_eq!(
            default_keys.len(),
            21,
            "the default arm must be 21 distinct libraries, unchanged"
        );
    }
}
