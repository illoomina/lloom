use crate::source::{
    AFFINE, ARENA_ALLOC, BINARY, CAST, CONV, FILL, FLASH_DECODING, GEMV, INDEXING, MLX_GEMM,
    MLX_SORT, QUANTIZED, RANDOM, REDUCE, SCRATCH, SDPA, SORT, TERNARY, UNARY,
};
use crate::utils::get_env_bool;
use crate::{
    ComputePipeline, ConstantValues, Device, Function, Library, MTLCompileOptions,
    MTLMathFloatingPointFunctions, MTLMathMode, MetalKernelError, Source,
};
use objc2::available;
use objc2::rc::Retained;
use objc2_foundation::NSString;
use std::collections::HashMap;
use std::sync::RwLock;

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

type Libraries = HashMap<Source, Library>;
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

    fn get_library_source(&self, source: Source) -> &'static str {
        match source {
            Source::Affine => AFFINE,
            Source::ArenaAlloc => ARENA_ALLOC,
            Source::Binary => BINARY,
            Source::Cast => CAST,
            Source::Conv => CONV,
            Source::Fill => FILL,
            Source::FlashDecoding => FLASH_DECODING,
            Source::Gemm => MLX_GEMM,
            Source::Gemv => GEMV,
            Source::Indexing => INDEXING,
            Source::MlxSort => MLX_SORT,
            Source::Quantized => QUANTIZED,
            Source::Random => RANDOM,
            Source::Reduce => REDUCE,
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
        if let Some(lib) = self.libraries.read()?.get(&source) {
            return Ok(lib.clone());
        }

        let mut libraries = self.libraries.write()?;
        // Recheck after acquiring the write lock in case another thread inserted it.
        if let Some(lib) = libraries.get(&source) {
            Ok(lib.clone())
        } else {
            let lib = {
                let source_content = self.get_library_source(source);
                let compile_options = get_compile_options();
                device
                    .new_library_with_source(source_content, Some(&compile_options))
                    .map_err(|e| MetalKernelError::LoadLibraryError(e.to_string()))?
            };
            libraries.insert(source, lib.clone());
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
        if let Some(pipeline) = self.pipelines.read()?.get(&key) {
            return Ok(pipeline.clone());
        }

        let mut pipelines = self.pipelines.write()?;
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
        assert_eq!(kernels.libraries.read().unwrap().len(), 1);
        assert_eq!(kernels.pipelines.read().unwrap().len(), 1);
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
}
