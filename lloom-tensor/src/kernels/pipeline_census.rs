//! The pipeline census (`diagnostics.trace.pipelines`, `LLOOM_PIPELINE_CENSUS`).
//!
//! While on, every compute pipeline the process creates prints one line on
//! stderr naming what it was built from (source, function, function
//! constants) and the three fields the driver reports about its register and
//! threadgroup-memory bounds. A pipeline is created once per variant, so the
//! cost is one line per kernel the run reaches, never one per dispatch.
//!
//! The line is the input of `examples/register_census.rs`, which rebuilds
//! each pipeline from it with the same compile options and constants, so the
//! population of a census is what a real run built rather than a list kept by
//! hand (`docs/tuning.md` §13.1b reads the field; `docs/fusion.md` §12.3 is
//! what it is for).

use super::metal::{ComputePipeline, ConstantValues, Value};
use super::source::Source;
use std::sync::OnceLock;

/// Every [`Source`], in [`source_index`] order.
pub const SOURCES: [Source; 26] = [
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
    Source::QuantizedGlu,
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

/// Position of `source` in [`SOURCES`]. Exhaustive, so a new variant does not
/// compile until it has a place in the list.
pub const fn source_index(source: Source) -> usize {
    match source {
        Source::Affine => 0,
        Source::ArenaAlloc => 1,
        Source::Binary => 2,
        Source::Cast => 3,
        Source::Conv => 4,
        Source::EosGate => 5,
        Source::EosIndex => 6,
        Source::Fill => 7,
        Source::FlashDecoding => 8,
        Source::Gemm => 9,
        Source::Gemv => 10,
        Source::Gdn => 11,
        Source::Indexing => 12,
        Source::MlxSort => 13,
        Source::Quantized => 14,
        Source::QuantizedStore => 15,
        Source::QuantizedGlu => 16,
        Source::Qwen4Exp => 17,
        Source::Random => 18,
        Source::Reduce => 19,
        Source::Rope => 20,
        Source::Scratch => 21,
        Source::Sort => 22,
        Source::Ternary => 23,
        Source::Unary => 24,
        Source::Sdpa => 25,
    }
}

/// Whether the instrument is on. Read once.
pub fn armed() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().instrument.pipeline_census)
}

/// One pipeline as the census names it: enough to build it again.
#[derive(Debug, PartialEq)]
pub struct Entry {
    pub source: Source,
    pub function: String,
    /// The function constants the pipeline was specialised with, the EOS
    /// gate's included; `None` for an unspecialised function.
    pub constants: Option<Vec<(usize, Value)>>,
}

fn encode_value(value: &Value) -> String {
    match value {
        Value::Bool(v) => format!("b:{}", u8::from(*v)),
        Value::USize(v) => format!("u:{v}"),
        Value::U16(v) => format!("h:{v}"),
        Value::F32(v) => format!("f:{:#010x}", v.to_bits()),
    }
}

fn decode_value(text: &str) -> Option<Value> {
    let (kind, body) = text.split_once(':')?;
    Some(match kind {
        "b" => Value::Bool(match body {
            "0" => false,
            "1" => true,
            _ => return None,
        }),
        "u" => Value::USize(body.parse().ok()?),
        "h" => Value::U16(body.parse().ok()?),
        "f" => Value::F32(f32::from_bits(
            u32::from_str_radix(body.strip_prefix("0x")?, 16).ok()?,
        )),
        _ => return None,
    })
}

fn encode_constants(constants: Option<&[(usize, Value)]>) -> String {
    match constants {
        None => "-".to_string(),
        Some(values) => values
            .iter()
            .map(|(index, value)| format!("{index}={}", encode_value(value)))
            .collect::<Vec<_>>()
            .join(","),
    }
}

/// The census line for a pipeline just built.
pub fn line(
    source: Source,
    function: &str,
    constants: Option<&ConstantValues>,
    pipeline: &ComputePipeline,
) -> String {
    format!(
        "[pipeline: source={source:?} function={function} constants={} max_threads={} exec_width={} tg_mem={}]",
        encode_constants(constants.map(ConstantValues::values)),
        pipeline.max_total_threads_per_threadgroup(),
        pipeline.thread_execution_width(),
        pipeline.static_threadgroup_memory_length(),
    )
}

/// Parse a census line back to the pipeline it names. `None` for a line that
/// is not one.
pub fn parse(text: &str) -> Option<Entry> {
    let body = text.trim().strip_prefix("[pipeline: ")?.strip_suffix(']')?;
    let field = |key: &str| {
        body.split(' ')
            .find_map(|part| part.strip_prefix(key)?.strip_prefix('='))
    };
    let source_name = field("source")?;
    let source = *SOURCES.iter().find(|s| format!("{s:?}") == source_name)?;
    let function = field("function")?.to_string();
    let constants = match field("constants")? {
        "-" => None,
        list => Some(
            list.split(',')
                .map(|pair| {
                    let (index, value) = pair.split_once('=')?;
                    Some((index.parse().ok()?, decode_value(value)?))
                })
                .collect::<Option<Vec<_>>>()?,
        ),
    };
    Some(Entry {
        source,
        function,
        constants,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_source_has_its_own_place() {
        for (i, s) in SOURCES.iter().enumerate() {
            assert_eq!(source_index(*s), i, "{s:?}");
        }
    }

    #[test]
    fn constants_survive_the_line() {
        let values = vec![
            (30, Value::Bool(true)),
            (10, Value::Bool(false)),
            (200, Value::USize(64)),
            (917, Value::U16(3)),
            (5, Value::F32(-0.125)),
        ];
        let text = format!(
            "[pipeline: source=Quantized function=kernel_mul_mv_q4_K_f32 constants={} max_threads=1024 exec_width=32 tg_mem=0]",
            encode_constants(Some(&values))
        );
        let entry = parse(&text).expect("a census line parses");
        assert_eq!(entry.source, Source::Quantized);
        assert_eq!(entry.function, "kernel_mul_mv_q4_K_f32");
        assert_eq!(entry.constants, Some(values));
    }

    #[test]
    fn an_unspecialised_function_has_no_constants() {
        let entry = parse(
            "[pipeline: source=Sdpa function=sdpa_vector_float_256 constants=- max_threads=1024 exec_width=32 tg_mem=4352]",
        )
        .expect("parses");
        assert_eq!(entry.constants, None);
    }

    #[test]
    fn a_line_naming_no_source_is_refused() {
        assert_eq!(
            parse("[pipeline: source=Nope function=k constants=- max_threads=1 exec_width=32 tg_mem=0]"),
            None
        );
        assert_eq!(parse("[pass plans: served=1]"), None);
    }
}
