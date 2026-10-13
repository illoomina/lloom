//! Shared cfg(test)-only synthetic GGUF, not a production weight schema.
use lloom_tensor::quantized::{gguf_file, GgmlDType, QTensor};
use lloom_tensor::{Device, Result, Tensor};
use std::io::Cursor;

// D4/H2/KV1/FF6/V8, attention first then conv, context32, F32 GGUF.
// No checkpoint, Q8 claim, device, or image. Numeric weight payload <4KiB.
pub(crate) fn fixture(named_template: Option<&str>) -> Result<Vec<u8>> {
    build(named_template, GgmlDType::F32, 4)
}

// The same layout at width `width` (a multiple of 32), its token table stored
// as `table`, every other weight F32: a block-quantized table at the smallest
// width that holds one.
#[allow(dead_code)]
pub(crate) fn fixture_table(table: GgmlDType, width: usize) -> Result<Vec<u8>> {
    build(None, table, width)
}

fn build(named_template: Option<&str>, table: GgmlDType, width: usize) -> Result<Vec<u8>> {
    let tiny = width == 4;
    let (w, half) = (width, width / 2);
    let mut tensors = Vec::<(String, QTensor)>::new();
    let mut add = |name: String, shape: Vec<usize>, norm: bool| -> Result<()> {
        let n: usize = shape.iter().product();
        assert!(!tiny || n <= 48);
        let dtype = if name == "token_embd.weight" {
            table
        } else {
            GgmlDType::F32
        };
        let seed = tensors.len();
        let values = (0..n)
            .map(|i| {
                if norm {
                    1.
                } else {
                    ((i * 7 + seed * 3) % 19) as f32 / 24. - 0.375
                }
            })
            .collect::<Vec<_>>();
        tensors.push((
            name,
            QTensor::quantize(&Tensor::from_vec(values, shape, &Device::Cpu)?, dtype)?,
        ));
        Ok(())
    };
    add("token_embd.weight".into(), vec![8, w], false)?;
    add("output_norm.weight".into(), vec![w], true)?;
    for layer in 0..2 {
        for norm in ["attn_norm", "ffn_norm"] {
            add(format!("blk.{layer}.{norm}.weight"), vec![w], true)?;
        }
        for (name, out, input) in [("ffn_gate", 6, w), ("ffn_up", 6, w), ("ffn_down", w, 6)] {
            add(
                format!("blk.{layer}.{name}.weight"),
                vec![out, input],
                false,
            )?;
        }
    }
    for (name, out, input) in [
        ("attn_q", w, w),
        ("attn_k", half, w),
        ("attn_v", half, w),
        ("attn_output", w, w),
    ] {
        add(format!("blk.0.{name}.weight"), vec![out, input], false)?;
    }
    for name in ["attn_q_norm", "attn_k_norm"] {
        add(format!("blk.0.{name}.weight"), vec![half], true)?;
    }
    for (name, out, input) in [("in_proj", 3 * w, w), ("out_proj", w, w), ("conv", w, 3)] {
        add(
            format!("blk.1.shortconv.{name}.weight"),
            vec![out, input],
            false,
        )?;
    }
    use gguf_file::Value as V;
    let mut metadata = vec![
        ("general.architecture", V::String("lfm2".into())),
        ("lfm2.attention.head_count", V::U32(2)),
        (
            "lfm2.attention.head_count_kv",
            V::Array(vec![V::U32(1), V::U32(0)]),
        ),
        ("lfm2.embedding_length", V::U32(width as u32)),
        ("lfm2.context_length", V::U32(32)),
        ("lfm2.block_count", V::U32(2)),
        ("lfm2.attention.layer_norm_rms_epsilon", V::F32(1e-5)),
        ("lfm2.shortconv.l_cache", V::U32(3)),
    ];
    if let Some(source) = named_template {
        // llama.cpp's converter writes the decision type beside the template.
        metadata.push(("lfm2.decision.type", V::String("lfm2-d1".into())));
        metadata.push((
            "tokenizer.chat_template.systemone",
            V::String(source.into()),
        ));
    }
    let mut bytes = Cursor::new(Vec::new());
    gguf_file::write(
        &mut bytes,
        &metadata.iter().map(|(k, v)| (*k, v)).collect::<Vec<_>>(),
        &tensors
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>(),
    )?;
    assert_eq!(tensors.len(), 21);
    assert!(!tiny || bytes.get_ref().len() < 8192);
    Ok(bytes.into_inner())
}
