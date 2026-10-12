//! The LFM2-VL vision tower and projector from an `mmproj` GGUF, the layout
//! llama.cpp's converter writes (`general.architecture = clip`,
//! `clip.projector_type = lfm2`): the same [`VisionEncoder`] the safetensors
//! checkpoint builds, its weights dequantized and renamed to the names
//! [`VisionEncoder::new`] reads. Nothing about the tower is redefined here.
use super::lfm2_vl::{Lfm2VlConfig, VisionConfig, VisionEncoder};
use crate::nn::VarBuilder;
use lloom_tensor::quantized::gguf_file::{Content, Value};
use lloom_tensor::{DType, Device, Result, Tensor};
use std::collections::HashMap;

const TOWER: &str = "model.vision_tower.vision_model";
const PROJECTOR: &str = "model.multi_modal_projector";

/// The safetensors name an mmproj tensor carries, or `None` for a name this
/// layout does not have (refused by the loader, never ignored).
pub fn safetensors_name(gguf: &str) -> Option<String> {
    let (stem, suffix) = gguf.rsplit_once('.')?;
    if suffix != "weight" && suffix != "bias" {
        return None;
    }
    let mapped = match stem {
        "v.patch_embd" => format!("{TOWER}.embeddings.patch_embedding"),
        "v.position_embd" => format!("{TOWER}.embeddings.position_embedding"),
        "v.post_ln" => format!("{TOWER}.post_layernorm"),
        "mm.0" => format!("{PROJECTOR}.layer_norm"),
        "mm.1" => format!("{PROJECTOR}.linear_1"),
        "mm.2" => format!("{PROJECTOR}.linear_2"),
        _ => {
            let rest = stem.strip_prefix("v.blk.")?;
            let (layer, part) = rest.split_once('.')?;
            let layer: usize = layer.parse().ok()?;
            let part = match part {
                "attn_q" => "self_attn.q_proj",
                "attn_k" => "self_attn.k_proj",
                "attn_v" => "self_attn.v_proj",
                "attn_out" => "self_attn.out_proj",
                "ln1" => "layer_norm1",
                "ln2" => "layer_norm2",
                "ffn_up" => "mlp.fc1",
                "ffn_down" => "mlp.fc2",
                _ => return None,
            };
            format!("{TOWER}.encoder.layers.{layer}.{part}")
        }
    };
    Some(format!("{mapped}.{suffix}"))
}

fn number(metadata: &HashMap<String, Value>, key: &str) -> Result<usize> {
    let value = metadata
        .get(key)
        .ok_or_else(|| lloom_tensor::Error::Msg(format!("the mmproj has no {key}")))?;
    usize::try_from(value.to_u64().or_else(|_| value.to_u32().map(u64::from))?)
        .map_err(lloom_tensor::Error::wrap)
}

impl Lfm2VlConfig {
    /// The VL config an LFM2 mmproj describes, with the text model's `<image>`
    /// id (the mmproj does not carry it). The fields the file does not name
    /// (`use_image_special_tokens`, the tile size, the activations) are
    /// LFM2-VL's, the ones llama.cpp's converter assumes for this projector.
    pub fn from_mmproj(content: &Content, image_token_id: u32) -> Result<Self> {
        let m = &content.metadata;
        let get_str = |key: &str| m.get(key).and_then(|v| v.to_string().ok()).cloned();
        if get_str("general.architecture").as_deref() != Some("clip")
            || get_str("clip.projector_type").as_deref() != Some("lfm2")
        {
            lloom_tensor::bail!(
                "not an LFM2-VL mmproj: general.architecture {:?}, clip.projector_type {:?}",
                get_str("general.architecture"),
                get_str("clip.projector_type")
            )
        }
        let patch = number(m, "clip.vision.patch_size")?;
        let image = number(m, "clip.vision.image_size")?;
        if patch == 0 || image % patch != 0 {
            lloom_tensor::bail!("mmproj image_size {image} is not a multiple of patch_size {patch}")
        }
        let eps = m
            .get("clip.vision.attention.layer_norm_epsilon")
            .map(|v| v.to_f32().map(f64::from))
            .transpose()?
            .unwrap_or(1e-6);
        let projector = content
            .tensor_infos
            .get("mm.1.weight")
            .ok_or_else(|| lloom_tensor::Error::Msg("the mmproj has no mm.1.weight".into()))?;
        let defaults = Self::lfm2_vl_3b();
        Ok(Self {
            vision_config: VisionConfig {
                hidden_size: number(m, "clip.vision.embedding_length")?,
                intermediate_size: number(m, "clip.vision.feed_forward_length")?,
                num_hidden_layers: number(m, "clip.vision.block_count")?,
                num_attention_heads: number(m, "clip.vision.attention.head_count")?,
                num_channels: 3,
                patch_size: patch,
                num_patches: (image / patch) * (image / patch),
                layer_norm_eps: eps,
                hidden_act: defaults.vision_config.hidden_act,
            },
            image_token_id,
            downsample_factor: number(m, "clip.vision.projector.scale_factor")?,
            projector_hidden_size: projector.shape.dims2()?.0,
            projector_bias: content.tensor_infos.contains_key("mm.1.bias"),
            projector_use_layernorm: content.tensor_infos.contains_key("mm.0.weight"),
            projector_hidden_act: defaults.projector_hidden_act,
            encoder_patch_size: patch,
            tile_size: defaults.tile_size,
            use_image_special_tokens: defaults.use_image_special_tokens,
        })
    }
}

impl VisionEncoder {
    /// The tower and projector from an mmproj, each tensor dequantized on
    /// `device` to `dtype` and renamed. The patch projection is stored by the
    /// converter as `[D, C, P, P]`; the tower reads `[D, P*P*C]`, so it is
    /// permuted back once here. An mmproj tensor this layout does not name is
    /// refused rather than skipped.
    pub fn from_mmproj<R: std::io::Read + std::io::Seek>(
        content: &Content,
        reader: &mut R,
        cfg: &Lfm2VlConfig,
        text_hidden_size: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let mut weights = HashMap::with_capacity(content.tensor_infos.len());
        for name in content.tensor_infos.keys() {
            let target = safetensors_name(name).ok_or_else(|| {
                lloom_tensor::Error::Msg(format!("mmproj tensor {name} is not an LFM2-VL one"))
            })?;
            let mut tensor = content.tensor(reader, name, device)?.dequantize(device)?;
            if name == "v.patch_embd.weight" {
                let (d, c, p, q) = tensor.dims4()?;
                tensor = tensor
                    .permute((0, 2, 3, 1))?
                    .contiguous()?
                    .reshape((d, p * q * c))?;
            }
            weights.insert(target, tensor.to_dtype(dtype)?);
        }
        let vb = VarBuilder::from_backend(Box::new(weights), dtype, device.clone());
        Self::new(cfg, text_hidden_size, vb)
    }
}

#[cfg(test)]
mod tests {
    use super::safetensors_name;
    use crate::models::lfm2_vl::{Lfm2VlConfig, VisionConfig, VisionEncoder};
    use crate::nn::VarBuilder;
    use lloom_tensor::quantized::gguf_file::{self, Content, Value};
    use lloom_tensor::quantized::{GgmlDType, QTensor};
    use lloom_tensor::{DType, Device, Result, Tensor};
    use std::collections::HashMap;
    use std::io::Cursor;

    const P: usize = 2;
    const HIDDEN: usize = 8;
    const FF: usize = 12;
    const PROJ: usize = 6;
    const TEXT: usize = 5;

    fn cfg() -> Lfm2VlConfig {
        let mut cfg = Lfm2VlConfig::lfm2_vl_3b();
        cfg.vision_config = VisionConfig {
            hidden_size: HIDDEN,
            intermediate_size: FF,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_channels: 3,
            patch_size: P,
            num_patches: 4,
            layer_norm_eps: 1e-6,
            hidden_act: cfg_act(),
        };
        cfg.projector_hidden_size = PROJ;
        cfg.encoder_patch_size = P;
        cfg.image_token_id = 3;
        cfg
    }
    fn cfg_act() -> crate::nn::Activation {
        Lfm2VlConfig::lfm2_vl_3b().vision_config.hidden_act
    }

    /// The tensors of a one-layer tower and its projector under the GGUF
    /// names, with the safetensors shape each maps to.
    fn layout() -> Vec<(&'static str, Vec<usize>)> {
        let h = HIDDEN;
        vec![
            ("v.patch_embd.weight", vec![h, 3 * P * P]),
            ("v.patch_embd.bias", vec![h]),
            ("v.position_embd.weight", vec![4, h]),
            ("v.blk.0.attn_q.weight", vec![h, h]),
            ("v.blk.0.attn_q.bias", vec![h]),
            ("v.blk.0.attn_k.weight", vec![h, h]),
            ("v.blk.0.attn_k.bias", vec![h]),
            ("v.blk.0.attn_v.weight", vec![h, h]),
            ("v.blk.0.attn_v.bias", vec![h]),
            ("v.blk.0.attn_out.weight", vec![h, h]),
            ("v.blk.0.attn_out.bias", vec![h]),
            ("v.blk.0.ln1.weight", vec![h]),
            ("v.blk.0.ln1.bias", vec![h]),
            ("v.blk.0.ln2.weight", vec![h]),
            ("v.blk.0.ln2.bias", vec![h]),
            ("v.blk.0.ffn_up.weight", vec![FF, h]),
            ("v.blk.0.ffn_up.bias", vec![FF]),
            ("v.blk.0.ffn_down.weight", vec![h, FF]),
            ("v.blk.0.ffn_down.bias", vec![h]),
            ("v.post_ln.weight", vec![h]),
            ("v.post_ln.bias", vec![h]),
            ("mm.1.weight", vec![PROJ, 4 * h]),
            ("mm.1.bias", vec![PROJ]),
            ("mm.2.weight", vec![TEXT, PROJ]),
            ("mm.2.bias", vec![TEXT]),
        ]
    }

    /// Nonzero, position-dependent values: a permutation of any axis moves
    /// the output.
    fn values(seed: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 7 + seed * 13) % 23) as f32 / 23. - 0.45)
            .collect()
    }

    /// The safetensors map, and the same weights as an mmproj GGUF the way
    /// the converter stores them (the patch projection as `[D, C, P, P]`).
    fn fixture(permute_patch: bool) -> Result<(HashMap<String, Tensor>, Vec<u8>)> {
        let mut dense = HashMap::new();
        let mut stored = Vec::new();
        for (seed, (name, shape)) in layout().into_iter().enumerate() {
            let n = shape.iter().product();
            let tensor = Tensor::from_vec(values(seed, n), shape, &Device::Cpu)?;
            dense.insert(safetensors_name(name).unwrap(), tensor.clone());
            let gguf = if name == "v.patch_embd.weight" && permute_patch {
                tensor
                    .reshape((HIDDEN, P, P, 3))?
                    .permute((0, 3, 1, 2))?
                    .contiguous()?
            } else {
                tensor
            };
            stored.push((name, QTensor::quantize(&gguf, GgmlDType::F32)?));
        }
        let metadata = [
            ("general.architecture", Value::String("clip".into())),
            ("clip.projector_type", Value::String("lfm2".into())),
            ("clip.vision.patch_size", Value::U32(P as u32)),
            ("clip.vision.image_size", Value::U32(2 * P as u32)),
            ("clip.vision.embedding_length", Value::U32(HIDDEN as u32)),
            ("clip.vision.feed_forward_length", Value::U32(FF as u32)),
            ("clip.vision.block_count", Value::U32(1)),
            ("clip.vision.attention.head_count", Value::U32(2)),
            ("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-6)),
            ("clip.vision.projector.scale_factor", Value::U32(2)),
        ];
        let metadata: Vec<_> = metadata.iter().map(|(k, v)| (*k, v)).collect();
        let tensors: Vec<_> = stored.iter().map(|(n, t)| (*n, t)).collect();
        let mut bytes = Cursor::new(Vec::new());
        gguf_file::write(&mut bytes, &metadata, &tensors)?;
        Ok((dense, bytes.into_inner()))
    }

    fn encode(encoder: &VisionEncoder) -> Result<Vec<f32>> {
        let patches = Tensor::from_vec(values(99, 4 * 3 * P * P), (1, 4, 3 * P * P), &Device::Cpu)?;
        encoder.forward(&patches, 2, 2)?.flatten_all()?.to_vec1()
    }

    /// The mmproj loader builds the same encoder the safetensors names build:
    /// its config from the clip keys, and its output equal to the last bit
    /// (F32 storage, the patch projection permuted back exactly). A file
    /// whose patch projection was not permuted encodes differently, so the
    /// comparison sees the permutation.
    #[test]
    fn an_mmproj_loads_the_same_encoder_as_the_safetensors_names() -> Result<()> {
        let (dense, bytes) = fixture(true)?;
        let reference = VisionEncoder::new(
            &cfg(),
            TEXT,
            VarBuilder::from_backend(Box::new(dense), DType::F32, Device::Cpu),
        )?;
        let mut reader = Cursor::new(&bytes);
        let content = Content::read(&mut reader)?;
        let read = Lfm2VlConfig::from_mmproj(&content, 3)?;
        let want = cfg();
        assert_eq!(
            (
                read.vision_config.hidden_size,
                read.vision_config.intermediate_size,
                read.vision_config.num_hidden_layers,
                read.vision_config.num_attention_heads,
                read.vision_config.patch_size,
                read.vision_config.num_patches,
                read.downsample_factor,
                read.projector_hidden_size,
                read.projector_bias,
                read.projector_use_layernorm,
            ),
            (HIDDEN, FF, 1, 2, P, 4, 2, PROJ, true, false)
        );
        assert_eq!(read.image_token_id, want.image_token_id);
        let loaded = VisionEncoder::from_mmproj(
            &content,
            &mut reader,
            &read,
            TEXT,
            DType::F32,
            &Device::Cpu,
        )?;
        let (a, b) = (encode(&reference)?, encode(&loaded)?);
        assert_eq!(a, b, "mmproj encoder differs from the safetensors one");

        let (_, unpermuted) = fixture(false)?;
        let mut reader = Cursor::new(&unpermuted);
        let content = Content::read(&mut reader)?;
        let wrong = VisionEncoder::from_mmproj(
            &content,
            &mut reader,
            &read,
            TEXT,
            DType::F32,
            &Device::Cpu,
        )?;
        assert_ne!(
            encode(&wrong)?,
            a,
            "the patch layout discriminator is vacuous"
        );
        Ok(())
    }

    /// A file that is not an LFM2 mmproj, or carries a tensor this layout
    /// does not name, is refused rather than loaded partially.
    #[test]
    fn a_foreign_mmproj_is_refused() -> Result<()> {
        let mut stored = vec![(
            "v.blk.0.attn_qkv.weight",
            QTensor::quantize(
                &Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?,
                GgmlDType::F32,
            )?,
        )];
        let arch = Value::String("clip".into());
        let projector = Value::String("lfm2".into());
        let tensors: Vec<_> = stored.iter_mut().map(|(n, t)| (*n, &*t)).collect();
        let mut bytes = Cursor::new(Vec::new());
        gguf_file::write(
            &mut bytes,
            &[
                ("general.architecture", &arch),
                ("clip.projector_type", &projector),
            ],
            &tensors,
        )?;
        let bytes = bytes.into_inner();
        let mut reader = Cursor::new(&bytes);
        let content = Content::read(&mut reader)?;
        let error = VisionEncoder::from_mmproj(
            &content,
            &mut reader,
            &cfg(),
            TEXT,
            DType::F32,
            &Device::Cpu,
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("not an LFM2-VL one"), "{error}");
        let other = Value::String("gemma3".into());
        let mut bytes = Cursor::new(Vec::new());
        gguf_file::write(
            &mut bytes,
            &[
                ("general.architecture", &arch),
                ("clip.projector_type", &other),
            ],
            &[],
        )?;
        let bytes = bytes.into_inner();
        let content = Content::read(&mut Cursor::new(&bytes))?;
        assert!(Lfm2VlConfig::from_mmproj(&content, 3).is_err());
        Ok(())
    }

    /// Every name llama.cpp's LFM2 mmproj carries maps to the safetensors
    /// name `VisionEncoder::new` reads; anything else maps to nothing.
    #[test]
    fn mmproj_names_map_to_the_safetensors_layout() {
        let tower = "model.vision_tower.vision_model";
        for (gguf, want) in [
            (
                "v.patch_embd.weight",
                format!("{tower}.embeddings.patch_embedding.weight"),
            ),
            (
                "v.position_embd.weight",
                format!("{tower}.embeddings.position_embedding.weight"),
            ),
            (
                "v.blk.26.attn_out.bias",
                format!("{tower}.encoder.layers.26.self_attn.out_proj.bias"),
            ),
            (
                "v.blk.0.ffn_down.weight",
                format!("{tower}.encoder.layers.0.mlp.fc2.weight"),
            ),
            (
                "v.blk.3.ln2.weight",
                format!("{tower}.encoder.layers.3.layer_norm2.weight"),
            ),
            ("v.post_ln.bias", format!("{tower}.post_layernorm.bias")),
            (
                "mm.1.weight",
                "model.multi_modal_projector.linear_1.weight".into(),
            ),
            (
                "mm.2.bias",
                "model.multi_modal_projector.linear_2.bias".into(),
            ),
        ] {
            assert_eq!(
                safetensors_name(gguf).as_deref(),
                Some(want.as_str()),
                "{gguf}"
            );
        }
        for unknown in [
            "v.blk.x.attn_q.weight",
            "v.blk.0.attn_qkv.weight",
            "mm.a.mlp.5.weight",
            "token_embd.weight",
        ] {
            assert_eq!(safetensors_name(unknown), None, "{unknown}");
        }
    }
}
