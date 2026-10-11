//! Header-only resolution for the pinned official d1-omni GGUF converter.
//! No weight loading/dequantization and no invented inverse Python FFN config.
use super::core::ResolvedDecisionCoreConfig;
use super::model::ResolvedOmniConfig;
use super::trunk::ResolvedTextConfig;
use super::AudioConfig;
use crate::models::lfm2_vl::VisionConfig;
use crate::nn::Activation;
use lloom_tensor::quantized::gguf_file::{Content, Value};
use lloom_tensor::{Result, Shape};
use std::collections::BTreeMap;

struct Header<'a>(&'a Content);
impl Header<'_> {
    fn value(&self, key: &str) -> Result<&Value> {
        self.0
            .metadata
            .get(key)
            .ok_or_else(|| lloom_tensor::Error::Msg(format!("missing omni GGUF metadata {key}")))
    }
    fn number(&self, key: &str) -> Result<usize> {
        integer(self.value(key)?)
    }
    fn real(&self, key: &str) -> Result<f64> {
        real(self.value(key)?)
    }
    fn string_is(&self, key: &str, expected: &str) -> Result<()> {
        if self.value(key)?.to_string()? != expected {
            lloom_tensor::bail!("omni GGUF {key} must be {expected}")
        }
        Ok(())
    }
    fn boolean_is(&self, key: &str, expected: bool) -> Result<()> {
        if self.value(key)?.to_bool()? != expected {
            lloom_tensor::bail!("omni GGUF {key} must be {expected}")
        }
        Ok(())
    }
    fn layers(&self, key: &str, count: usize) -> Result<Vec<usize>> {
        let value = self.value(key)?;
        // GGUF permits uniform scalars; the official FFN/KV converter emits
        // arrays. Validate length before allocating a derived descriptor list.
        if let Value::Array(values) = value {
            if values.len() != count {
                lloom_tensor::bail!("omni GGUF {key} layer count mismatch")
            }
        }
        let mut result = super::host::prepared_vec(count)?;
        match value {
            Value::Array(values) => {
                for value in values {
                    result.push(integer(value)?);
                }
            }
            value => result.resize(count, integer(value)?),
        }
        Ok(result)
    }
    fn shape(&self, name: &str) -> Result<&Shape> {
        self.0
            .tensor_infos
            .get(name)
            .map(|t| &t.shape)
            .ok_or_else(|| {
                lloom_tensor::Error::Msg(format!("missing omni GGUF tensor header {name}"))
            })
    }
}
fn integer(value: &Value) -> Result<usize> {
    // Shared to_u64 also accepts Bool; booleans are NOT dimension metadata.
    if !matches!(
        value,
        Value::U8(_) | Value::U16(_) | Value::U32(_) | Value::U64(_)
    ) {
        lloom_tensor::bail!("omni GGUF dimension must be an unsigned integer")
    }
    usize::try_from(value.to_u64()?).map_err(lloom_tensor::Error::wrap)
}
fn real(value: &Value) -> Result<f64> {
    match value {
        Value::F32(v) => Ok(*v as f64),
        Value::F64(v) => Ok(*v),
        _ => lloom_tensor::bail!("omni GGUF coefficient must be floating point"),
    }
}
fn equal_number(h: &Header<'_>, key: &str, expected: usize) -> Result<()> {
    if h.number(key)? != expected {
        lloom_tensor::bail!("omni GGUF {key} must be {expected}")
    }
    Ok(())
}
fn fixed_epsilon(h: &Header<'_>, key: &str) -> Result<()> {
    let epsilon = h.real(key)?;
    if epsilon != 1e-5 && epsilon != 1e-5f32 as f64 {
        lloom_tensor::bail!("omni GGUF {key} differs from fixed reference1e-5")
    }
    Ok(())
}

/// Source contract: llama.cpp23b0202a conversion/lfm2.py D1OmniModel and
/// D1OmniMmprojModel, conversion/base.py MmprojModel, conversion/gemma.py
/// ConformerAudioModel; metadata names from gguf-py/gguf/constants.py.
pub(crate) fn resolve(main: &Content, media: &Content) -> Result<ResolvedOmniConfig> {
    let m = Header(main);
    let v = Header(media);
    m.string_is("general.architecture", "lfm2")?;
    m.string_is("lfm2.decision.type", "lfm2-d1-omni")?;
    m.boolean_is("lfm2.attention.causal", false)?;
    equal_number(&m, "tokenizer.ggml.token_type_count", 3)?;
    fixed_epsilon(&m, "lfm2.attention.layer_norm_epsilon")?;
    let layers = m.number("lfm2.block_count")?;
    let head_layers = m.number("lfm2.decision.block_count")?;
    // Every supported block requires multiple named weights. Bound descriptor
    // allocation by the already-parsed tensor header count before reserving.
    if layers > main.tensor_infos.len() {
        lloom_tensor::bail!("omni GGUF block count exceeds available tensor headers")
    }
    let trunk_layers = layers
        .checked_sub(head_layers)
        .filter(|n| *n > 0)
        .ok_or_else(|| lloom_tensor::Error::Msg("invalid omni GGUF block partition".into()))?;
    let width = m.number("lfm2.embedding_length")?;
    let heads = m.number("lfm2.attention.head_count")?;
    if width == 0 || heads == 0 || width % 64 != 0 || heads != width / 64 {
        lloom_tensor::bail!("omni GGUF head width must match the existing64-wide head")
    }
    let ffn = m.layers("lfm2.feed_forward_length", layers)?;
    let kv = m.layers("lfm2.attention.head_count_kv", layers)?;
    let head_ffn = width
        .checked_mul(4)
        .ok_or_else(|| lloom_tensor::Error::Msg("omni GGUF head FFN overflow".into()))?;
    if ffn[..trunk_layers].iter().any(|n| *n != ffn[0])
        || ffn[trunk_layers..].iter().any(|n| *n != head_ffn)
        || kv[trunk_layers..].iter().any(|n| *n != heads)
    {
        lloom_tensor::bail!(
            "omni GGUF resolved trunk/head layer geometry differs from supported blocks"
        )
    }
    let trunk_kv = kv[..trunk_layers]
        .iter()
        .copied()
        .find(|n| *n != 0)
        .ok_or_else(|| {
            lloom_tensor::Error::Msg("omni GGUF trunk has no attention geometry".into())
        })?;
    if kv[..trunk_layers].iter().any(|n| *n != 0 && *n != trunk_kv) {
        lloom_tensor::bail!("omni GGUF trunk uses differing KV head counts")
    }
    let mut layer_types = super::host::prepared_vec(trunk_layers)?;
    for &n in &kv[..trunk_layers] {
        layer_types.push(if n == 0 { "conv" } else { "full_attention" }.to_owned());
    }
    let text = ResolvedTextConfig {
        vocab_size: m.number("lfm2.vocab_size")?,
        hidden_size: width,
        intermediate_size: ffn[0],
        num_hidden_layers: trunk_layers,
        num_attention_heads: heads,
        num_key_value_heads: trunk_kv,
        layer_types,
        norm_eps: m.real("lfm2.attention.layer_norm_rms_epsilon")?,
        conv_l_cache: m.number("lfm2.shortconv.l_cache")?,
        max_position_embeddings: m.number("lfm2.context_length")?,
        rope_theta: m.real("lfm2.rope.freq_base")?,
    };
    v.string_is("clip.vision.projector_type", "d1omni_v")?;
    v.string_is("clip.audio.projector_type", "d1omni_a")?;
    v.boolean_is("clip.has_vision_encoder", true)?;
    v.boolean_is("clip.has_audio_encoder", true)?;
    v.boolean_is("clip.use_gelu", true)?;
    equal_number(&v, "clip.vision.projection_dim", width)?;
    equal_number(&v, "clip.audio.projection_dim", width)?;
    equal_number(&v, "clip.vision.projector.scale_factor", 2)?;
    for key in ["clip.vision.image_mean", "clip.vision.image_std"] {
        let values = v.value(key)?.to_vec()?;
        if values.len() != 3 || values.iter().any(|x| real(x).ok() != Some(0.5)) {
            lloom_tensor::bail!("omni GGUF {key} differs from fixed byte normalization")
        }
    }
    let vh = v.number("clip.vision.embedding_length")?;
    let (positions, position_width) = v.shape("v.position_embd.weight")?.dims2()?;
    let (projector, projector_input) = v.shape("mm.1.weight")?.dims2()?;
    if position_width != vh || Some(projector_input) != vh.checked_mul(4) {
        lloom_tensor::bail!("omni GGUF vision position/projector geometry mismatch")
    }
    let vision = VisionConfig {
        hidden_size: vh,
        intermediate_size: v.number("clip.vision.feed_forward_length")?,
        num_hidden_layers: v.number("clip.vision.block_count")?,
        num_attention_heads: v.number("clip.vision.attention.head_count")?,
        num_channels: 3,
        patch_size: v.number("clip.vision.patch_size")?,
        num_patches: positions,
        layer_norm_eps: v.real("clip.vision.attention.layer_norm_epsilon")?,
        hidden_act: Activation::GeluPytorchTanh,
    };
    fixed_epsilon(&v, "clip.audio.attention.layer_norm_epsilon")?;
    let d = v.number("clip.audio.embedding_length")?;
    let aff = v.number("clip.audio.feed_forward_length")?;
    let (channels, one, kh, kw) = v.shape("a.conv1d.0.weight")?.dims4()?;
    let (depth, kernel) = v.shape("a.blk.0.conv_dw.weight")?.dims2()?;
    let (residual, residual_input) = v.shape("mm.a.mlp.5.weight")?.dims2()?;
    if d == 0 || aff % d != 0 || (one, kh, kw) != (1, 3, 3) || depth != d || residual_input != width
    {
        lloom_tensor::bail!("omni GGUF audio resolved geometry mismatch")
    }
    let audio = AudioConfig {
        feat_in: v.number("clip.audio.num_mel_bins")?,
        n_layers: v.number("clip.audio.block_count")?,
        d_model: d,
        subsampling_conv_channels: channels,
        ff_expansion_factor: aff / d,
        n_heads: v.number("clip.audio.attention.head_count")?,
        conv_kernel_size: kernel,
        residual_width: residual,
    };
    let mut temperatures = BTreeMap::new();
    // The reference's finite bucket vocabulary has an unambiguous inverse.
    // No arbitrary punctuation normalization is applied to other metadata.
    for kind in ["choice", "score", "noul"] {
        for (suffix, original) in [
            ("", ""),
            (".2", ":2"),
            (".3_5", ":3-5"),
            (".6_10", ":6-10"),
            (".11", ":11+"),
        ] {
            if let Some(value) = main
                .metadata
                .get(&format!("lfm2.decision.temperature.{kind}{suffix}"))
            {
                temperatures.insert(format!("{kind}{original}"), real(value)?);
            }
        }
    }
    // These are converter-enforced constants, not inferred text model limits.
    let cfg = ResolvedOmniConfig {
        core: ResolvedDecisionCoreConfig {
            text,
            head_layers,
            max_length: 16384,
        },
        vision_config: vision,
        audio_config: audio,
        projector_hidden_size: projector,
        image_text_length: 896,
        audio_text_length: 15360,
        temperatures,
    };
    cfg.validate()?;
    Ok(cfg)
}

#[cfg(test)]
mod tests;
