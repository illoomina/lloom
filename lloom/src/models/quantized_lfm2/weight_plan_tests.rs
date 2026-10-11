use super::weights::Lfm2Weights;
use super::*;
use crate::nn::gguf_weights::{assert_fixture_demands, WeightPlan};
use std::io::{Cursor, Read, Seek, SeekFrom};

// The existing 21-F32 fixture only. No additional weight buffers, forwards,
// checkpoint, native device, or admission claim. Every refusal is header-only.
struct PayloadReads<'a> {
    inner: Cursor<&'a [u8]>,
    calls: usize,
}
impl Read for PayloadReads<'_> {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        self.calls += 1;
        self.inner.read(bytes)
    }
}
impl Seek for PayloadReads<'_> {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(position)
    }
}
fn header(bytes: &[u8]) -> Result<gguf_file::Content> {
    gguf_file::Content::read(&mut Cursor::new(bytes))
}
fn refusal_before_payload(content: gguf_file::Content, bytes: &[u8]) -> String {
    let mut reader = PayloadReads {
        inner: Cursor::new(bytes),
        calls: 0,
    };
    let result = Model::from_gguf(content, &mut reader, &Device::Cpu);
    assert_eq!(
        reader.calls, 0,
        "invalid header reached tensor payload reader"
    );
    match result {
        Err(error) => error.to_string(),
        Ok(_) => panic!("invalid GGUF header was accepted"),
    }
}
fn copied_info(info: &gguf_file::TensorInfo) -> gguf_file::TensorInfo {
    gguf_file::TensorInfo {
        shape: info.shape.clone(),
        ggml_dtype: info.ggml_dtype,
        offset: info.offset,
    }
}

#[test]
fn complete_demands_refuse_missing_shape_and_dtype_before_payload() -> Result<()> {
    let bytes = super::tests::fixture()?;
    let mut content = header(&bytes)?;
    let plan = Lfm2Weights::resolve(&content)?;
    assert!(
        plan.output.is_none(),
        "genuinely absent output must stay tied"
    );
    let mut names = Vec::new();
    plan.visit(&mut |weight| {
        names.push(weight.name().to_owned());
        Ok(())
    })?;
    assert_eq!(names.len(), 21);
    assert_fixture_demands(&plan, header(&bytes)?, 21)?;
    for name in &names {
        let info = content.tensor_infos.remove(name).unwrap();
        let error = plan.validate_headers(&content).unwrap_err().to_string();
        assert!(error.contains(name), "missing demand {name}: {error}");
        let mut missing = header(&bytes)?;
        missing.tensor_infos.remove(name);
        refusal_before_payload(missing, &bytes);
        content.tensor_infos.insert(name.clone(), info);

        // A rank change cannot be reinterpreted as a different inferred FFN or
        // vocabulary dimension. Both the bound plan and actual loader refuse it.
        let info = content.tensor_infos.get_mut(name).unwrap();
        let old_shape = info.shape.clone();
        let mut dimensions = old_shape.dims().to_vec();
        dimensions.push(0);
        info.shape = dimensions.into();
        let error = plan.validate_headers(&content).unwrap_err().to_string();
        assert!(error.contains(name), "shape demand {name}: {error}");
        let mut misshapen = header(&bytes)?;
        misshapen.tensor_infos.get_mut(name).unwrap().shape =
            content.tensor_infos[name].shape.clone();
        refusal_before_payload(misshapen, &bytes);
        content.tensor_infos.get_mut(name).unwrap().shape = old_shape;

        // Exact is a resolved-header identity check, not a backend whitelist.
        content.tensor_infos.get_mut(name).unwrap().ggml_dtype =
            lloom_tensor::quantized::GgmlDType::F16;
        let error = plan.validate_headers(&content).unwrap_err().to_string();
        assert!(error.contains(name), "dtype demand {name}: {error}");
        content.tensor_infos.get_mut(name).unwrap().ggml_dtype =
            lloom_tensor::quantized::GgmlDType::F32;
    }
    plan.validate_headers(&content)?;
    eprintln!("LFM2_ACTUAL_DEMAND_REFUSALS=21 missing+shape; bound_dtype=21; payload_reads=0");
    Ok(())
}

#[test]
fn first_present_output_refuses_invalid_primary_not_valid_alias_or_tie() -> Result<()> {
    let bytes = super::tests::fixture()?;
    let mut content = header(&bytes)?;
    assert!(Lfm2Weights::resolve(&content)?.output.is_none());
    let valid = copied_info(&content.tensor_infos["token_embd.weight"]);
    content.tensor_infos.insert("lm_head.weight".into(), valid);
    let plan = Lfm2Weights::resolve(&content)?;
    assert_eq!(plan.output.as_ref().unwrap().name(), "lm_head.weight");
    plan.validate_headers(&content)?;
    let mut invalid = copied_info(&content.tensor_infos["token_embd.weight"]);
    invalid.shape = vec![8, 5].into();
    content.tensor_infos.insert("output.weight".into(), invalid);
    let plan = Lfm2Weights::resolve(&content)?;
    assert_eq!(plan.output.as_ref().unwrap().name(), "output.weight");
    let error = plan.validate_headers(&content).unwrap_err().to_string();
    assert!(error.contains("output.weight"));
    assert!(refusal_before_payload(content, &bytes).contains("output.weight"));
    eprintln!("LFM2_OUTPUT_ALIAS_REFUSAL=first-present; truly_absent=tied; payload_reads=0");
    Ok(())
}

#[test]
fn late_layer_and_unused_extent_refuse_before_payload_and_rope_is_not_fallback() -> Result<()> {
    let bytes = super::tests::fixture()?;
    let mut late = header(&bytes)?;
    late.tensor_infos
        .get_mut("blk.1.shortconv.out_proj.weight")
        .unwrap()
        .shape = vec![4, 5].into();
    assert!(refusal_before_payload(late, &bytes).contains("blk.1.shortconv.out_proj.weight"));
    let mut unused = header(&bytes)?;
    let mut info = copied_info(&unused.tensor_infos["token_embd.weight"]);
    info.offset = u64::MAX;
    unused.tensor_infos.insert("unused.weight".into(), info);
    assert!(refusal_before_payload(unused, &bytes).contains("tensor start offset overflow"));
    let mut malformed = header(&bytes)?;
    malformed
        .metadata
        .insert("lfm2.rope.freq_base".into(), gguf_file::Value::Bool(true));
    refusal_before_payload(malformed, &bytes);
    assert_eq!(
        Lfm2Weights::resolve(&header(&bytes)?)?.rope_base,
        1_000_000.
    );
    eprintln!(
        "LFM2_PREFLIGHT_REFUSALS=late-layer+unused-extent+present-malformed-rope; payload_reads=0"
    );
    Ok(())
}
