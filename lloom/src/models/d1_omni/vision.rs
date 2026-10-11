//! d1-omni vision composition over the shared LFM2-VL/SigLIP mechanisms.
//! RGB decoding is external. Checked geometry is NOT pass admission; callers
//! must reserve image/preparation/encoder/projector work in their model Domain.
use super::weights::{LinearWeights, NormWeights, Weight, WeightDType, WeightPlan};
use crate::models::lfm2_vl::{
    ImageProcessor, Lfm2VlConfig, MultiModalProjector, ProjectorCore, RgbNormalization, TileLayout,
    VisionConfig, VisionPositionPadding, VisionTower, VisionTowerCore,
};
use crate::models::siglip::{Attention, Encoder, EncoderLayer, Mlp};
use crate::nn::linear::{ContiguousInput, Projection};
use crate::nn::{Activation, Linear, VarBuilder};
use crate::quantized_var_builder::VarBuilder as QuantizedVarBuilder;
use lloom_tensor::{DType, Device, Result, Tensor};
use std::sync::Arc;

const PATCH: usize = 16;
const TILE: usize = 512;
const MAX_PATCHES: usize = 1024;

fn product(values: &[usize]) -> Result<usize> {
    values
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| lloom_tensor::Error::Msg("omni image product/byte overflow".into()))
}
fn plus(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| lloom_tensor::Error::Msg("omni image extent overflow".into()))
}
fn processor() -> ImageProcessor {
    // Pinned vision.py constants, not caller-editable processor configuration.
    ImageProcessor {
        do_image_splitting: true,
        min_tiles: 2,
        max_tiles: 10,
        use_thumbnail: true,
        min_image_tokens: 64,
        max_image_tokens: 256,
        patch_size: PATCH,
        downsample_factor: 2,
        tile_size: TILE,
        max_pixels_tolerance: 2.,
        resample: 2,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        num_channels: 3,
    }
}

/// Pure numeric preflight plus the existing bounded grid enumeration (<=10).
/// No pixel buffer, resampling or Tensor work occurs in this constructor.
#[derive(Debug, Clone)]
pub struct VisionImagePlan {
    height: usize,
    width: usize,
    input_bytes: usize,
    layout: TileLayout,
    prefix_tokens: usize,
}
impl VisionImagePlan {
    pub fn new(height: usize, width: usize, byte_len: usize) -> Result<Self> {
        if height == 0
            || width == 0
            || height as u128 > (1u128 << 53)
            || width as u128 > (1u128 << 53)
        {
            lloom_tensor::bail!("omni image needs positive exactly representable dimensions")
        }
        let input_bytes = product(&[height, width, 3])?;
        if byte_len != input_bytes {
            lloom_tensor::bail!("omni RGB byte length differs from H*W*3")
        }
        // Every initial round-to32 side is <=input+32, including ties-to-even.
        // This proves the old layout helper's pre-branch products representable.
        product(&[plus(height, 32)?, plus(width, 32)?])?;
        // Fixed-grid products (including rejected ratio candidates) are bounded.
        product(&[10, 10, TILE, TILE, 3])?;
        // If smart_resize expands either branch, raw area<262144, so each side
        // <262144 and 1/beta or beta<=512. The post-round side fits32/64-bit.
        // If it shrinks, each output side<=max(input+32,32), already checked.
        plus(product(&[262144, 512])?, 32)?;
        let p = processor();
        let layout = p.layout(height, width);
        let (rh, rw) = layout.resized;
        validate_crop(rh, rw)?;
        // Check BOTH full image resizes, including their horizontal intermediate,
        // before tiles_rgb8 repeats the deterministic layout and starts allocating.
        validate_resize(height, width, rh, rw)?;
        if layout.is_grid() {
            if layout.rows == 0 || layout.cols == 0 || product(&[layout.rows, layout.cols])? > 10 {
                lloom_tensor::bail!("omni tile grid differs from fixed reference range")
            }
            let (gh, gw) = (
                product(&[layout.rows, TILE])?,
                product(&[layout.cols, TILE])?,
            );
            validate_resize(height, width, gh, gw)?;
            product(&[layout.rows, layout.cols, TILE, TILE, 3])?;
        }
        let prefix_tokens = p.image_tokens(&layout);
        Ok(Self {
            height,
            width,
            input_bytes,
            layout,
            prefix_tokens,
        })
    }
    pub fn layout(&self) -> TileLayout {
        self.layout
    }
    pub fn prefix_tokens(&self) -> usize {
        self.prefix_tokens
    }
}
fn validate_crop(height: usize, width: usize) -> Result<()> {
    if height == 0 || width == 0 || height % 32 != 0 || width % 32 != 0 {
        lloom_tensor::bail!("omni crop must have positive factor32 dimensions")
    }
    if product(&[height / PATCH, width / PATCH])? > MAX_PATCHES {
        // Siglip2 resize_positional_embeddings checks real grid<=max_length.
        lloom_tensor::bail!("omni crop grid exceeds1024 padded position rows")
    }
    product(&[height, width, 3])?;
    Ok(())
}
fn validate_resize(h: usize, w: usize, nh: usize, nw: usize) -> Result<()> {
    product(&[h, w, 3])?;
    product(&[h, nw, 3])?; // horizontal intermediate, not only final output
    product(&[nh, nw, 3])?;
    // Conservative tap backing and Vec metadata representability; the actual
    // bilinear support is smaller. This is not an allocation or peak estimate.
    for (source, target) in [(h, nh), (w, nw)] {
        product(&[source, target, 8])?;
        product(&[target, std::mem::size_of::<(usize, Vec<f64>)>()])?;
    }
    Ok(())
}

struct VisionTile {
    rows: usize,
    cols: usize,
    patches: Tensor,
    mask: Tensor,
}
/// Prepared numeric buffers retain the same origin device as loaded weights.
/// Owner identity prevents mixing model/config revisions; it is not a budget.
pub struct PreparedVisionImage {
    owner: Arc<()>,
    tiles: Vec<VisionTile>,
    prefix_tokens: usize,
}
impl PreparedVisionImage {
    pub fn prefix_tokens(&self) -> usize {
        self.prefix_tokens
    }
}

#[derive(Debug, Clone)]
pub struct Vision {
    inner: VisionInner<Linear>,
}

#[derive(Debug, Clone)]
struct VisionInner<P> {
    owner: Arc<()>,
    tower: VisionTowerCore<P>,
    projector: ProjectorCore<P>,
    hidden: usize,
    out: usize,
    dtype: DType,
    device: Device,
}
type QuantizedProjection = ContiguousInput<crate::quantized_nn::Linear>;
#[derive(Debug, Clone)]
pub(crate) struct QuantizedVision {
    inner: VisionInner<QuantizedProjection>,
}
impl Vision {
    pub(crate) fn validate_config(
        cfg: &VisionConfig,
        projector_hidden: usize,
        out: usize,
        dtype: DType,
    ) -> Result<()> {
        if cfg.num_channels != 3
            || cfg.patch_size != PATCH
            || cfg.hidden_size == 0
            || cfg.intermediate_size == 0
            || cfg.num_attention_heads == 0
            || cfg.hidden_size % cfg.num_attention_heads != 0
            || cfg.num_hidden_layers == 0
            || cfg.num_patches == 0
            || projector_hidden == 0
            || out == 0
            || !cfg.layer_norm_eps.is_finite()
            || cfg.layer_norm_eps <= 0.
            || !matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
        {
            lloom_tensor::bail!("omni vision needs supported floating dtype and positive compatible reference geometry")
        }
        let side = cfg.pos_embed_side();
        if product(&[side, side])? != cfg.num_patches {
            lloom_tensor::bail!("omni learned position table must be square")
        }
        for factors in [
            &[cfg.hidden_size, 3, PATCH, PATCH][..],
            &[cfg.num_patches, cfg.hidden_size][..],
            &[cfg.hidden_size, cfg.hidden_size][..],
            &[cfg.hidden_size, cfg.intermediate_size][..],
            &[projector_hidden, cfg.hidden_size, 4][..],
            &[projector_hidden, out][..],
        ] {
            product(&[product(factors)?, dtype.size_in_bytes()])?;
        }
        Ok(())
    }

    /// `vb` is rooted at the checkpoint's `vision` namespace. The caller selects
    /// safetensors/GGUF and establishes the shared load/allocation context.
    /// Device identity alone is not model-domain identity.
    /// Checkpoint tensor names remain separately verified at family loader time.
    pub fn load(
        cfg: &VisionConfig,
        projector_hidden: usize,
        out: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::validate_config(cfg, projector_hidden, out, vb.dtype())?;
        let tower = VisionTower::new(cfg, vb.pp("tower"))?.into_inner();
        let projector_cfg = Lfm2VlConfig {
            vision_config: cfg.clone(),
            image_token_id: 0,
            downsample_factor: 2,
            projector_hidden_size: projector_hidden,
            projector_bias: true,
            projector_use_layernorm: false,
            projector_hidden_act: Activation::Gelu,
            encoder_patch_size: PATCH,
            tile_size: TILE,
            use_image_special_tokens: false,
        };
        let projector =
            MultiModalProjector::new(&projector_cfg, out, vb.pp("projector"))?.into_inner();
        Ok(Self {
            inner: VisionInner {
                owner: Arc::new(()),
                tower,
                projector,
                hidden: cfg.hidden_size,
                out,
                dtype: vb.dtype(),
                device: vb.device().clone(),
            },
        })
    }

    /// Caller owns decoding and pass admission BEFORE this allocates source
    /// copies, resize intermediates, patch arrays, upload/cast and padding.
    pub fn prepare_rgb8(
        &self,
        rgb: &[u8],
        height: usize,
        width: usize,
    ) -> Result<PreparedVisionImage> {
        self.inner.prepare_rgb8(rgb, height, width)
    }
    /// Full1024 padded rows, repeat-first positions and boolean-key-mask semantics
    /// expressed as 0/-infinity additive mask; no full L² mask is constructed.
    /// Image order, tile row-major order and final thumbnail are preserved.
    pub fn forward_with_attention<F>(
        &self,
        images: &[&PreparedVisionImage],
        attend: &mut F,
    ) -> Result<Tensor>
    where
        F: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    {
        self.inner.forward_with_attention(images, attend)
    }
}

#[derive(Debug)]
struct VisionBlockWeights {
    attention: [LinearWeights; 4],
    mlp: [LinearWeights; 2],
    norms: [NormWeights; 2],
}
impl WeightPlan for VisionBlockWeights {
    fn visit(&self, visitor: &mut impl FnMut(&Weight) -> Result<()>) -> Result<()> {
        self.attention.as_slice().visit(visitor)?;
        self.mlp.as_slice().visit(visitor)?;
        self.norms.as_slice().visit(visitor)
    }
}

#[derive(Debug)]
pub(super) struct VisionWeights {
    patch: Weight,
    patch_bias: Weight,
    positions: Weight,
    blocks: Vec<VisionBlockWeights>,
    post_norm: NormWeights,
    projector: [LinearWeights; 2],
}
impl VisionWeights {
    pub(super) fn new(cfg: &VisionConfig, projector: usize, out: usize) -> Result<Self> {
        QuantizedVision::validate_config(cfg, projector, out)?;
        let h = cfg.hidden_size;
        let mut blocks = super::host::prepared_vec(cfg.num_hidden_layers)?;
        for i in 0..cfg.num_hidden_layers {
            let prefix = format!("v.blk.{i}");
            blocks.push(VisionBlockWeights {
                attention: ["attn_q", "attn_k", "attn_v", "attn_out"]
                    .map(|name| LinearWeights::new(&format!("{prefix}.{name}"), h, h, true)),
                mlp: [
                    LinearWeights::new(&format!("{prefix}.ffn_up"), h, cfg.intermediate_size, true),
                    LinearWeights::new(
                        &format!("{prefix}.ffn_down"),
                        cfg.intermediate_size,
                        h,
                        true,
                    ),
                ],
                norms: ["ln1", "ln2"].map(|name| NormWeights::new(&format!("{prefix}.{name}"), h)),
            });
        }
        Ok(Self {
            patch: Weight::new("v.patch_embd.weight", (h, 3, PATCH, PATCH))
                .dtype(WeightDType::F16OrF32),
            patch_bias: Weight::new("v.patch_embd.bias", h),
            positions: Weight::new("v.position_embd.weight", (cfg.num_patches, h)),
            blocks,
            post_norm: NormWeights::new("v.post_ln", h),
            projector: [
                LinearWeights::new("mm.1", product(&[h, 4])?, projector, true),
                LinearWeights::new("mm.2", projector, out, true),
            ],
        })
    }
}
impl WeightPlan for VisionWeights {
    fn visit(&self, visitor: &mut impl FnMut(&Weight) -> Result<()>) -> Result<()> {
        self.patch.visit(visitor)?;
        self.patch_bias.visit(visitor)?;
        self.positions.visit(visitor)?;
        self.blocks.as_slice().visit(visitor)?;
        self.post_norm.visit(visitor)?;
        self.projector.as_slice().visit(visitor)
    }
}

impl QuantizedVision {
    pub(crate) fn validate_config(
        cfg: &VisionConfig,
        projector_hidden: usize,
        out: usize,
    ) -> Result<()> {
        Vision::validate_config(cfg, projector_hidden, out, DType::F32)?;
        super::host::checked_host_capacity::<EncoderLayer<QuantizedProjection>>(
            cfg.num_hidden_layers,
        )?;
        Ok(())
    }
    /// Official mmproj names. The eager VarBuilder must already belong to the
    /// enclosing load promise; retained tables/copies require the same domain.
    pub(crate) fn load(
        cfg: &VisionConfig,
        projector_hidden: usize,
        out: usize,
        vb: QuantizedVarBuilder,
    ) -> Result<Self> {
        let weights = VisionWeights::new(cfg, projector_hidden, out)?;
        let hidden = cfg.hidden_size;
        // The converter permutes [D,P,P,C] to [D,C,P,P]. Invert once at load.
        let patch = weights
            .patch
            .dequantize(&vb)?
            .to_dtype(DType::F32)?
            .permute((0, 2, 3, 1))?
            .contiguous()?
            .reshape((hidden, 3 * PATCH * PATCH))?;
        let patch_bias = weights.patch_bias.dequantize(&vb)?;
        let positions = weights.positions.dequantize(&vb)?;
        let mut layers = super::host::prepared_vec(cfg.num_hidden_layers)?;
        for block in &weights.blocks {
            let projection = |weight: &LinearWeights| -> Result<QuantizedProjection> {
                Ok(ContiguousInput::new(weight.load(&vb)?))
            };
            let attention = Attention::from_projections(
                projection(&block.attention[0])?,
                projection(&block.attention[1])?,
                projection(&block.attention[2])?,
                projection(&block.attention[3])?,
                cfg.num_attention_heads,
            )?;
            let mlp = Mlp::from_parts(
                projection(&block.mlp[0])?,
                projection(&block.mlp[1])?,
                cfg.hidden_act,
            );
            layers.push(EncoderLayer::from_parts(
                attention,
                block.norms[0].load(&vb, cfg.layer_norm_eps)?,
                mlp,
                block.norms[1].load(&vb, cfg.layer_norm_eps)?,
            ));
        }
        let tower = VisionTowerCore::from_components(
            cfg,
            Linear::new(patch, Some(patch_bias)),
            positions,
            Encoder::from_layers(layers),
            weights.post_norm.load(&vb, cfg.layer_norm_eps)?,
        )?;
        let projector = ProjectorCore::from_components(
            ContiguousInput::new(weights.projector[0].load(&vb)?),
            ContiguousInput::new(weights.projector[1].load(&vb)?),
            Activation::Gelu,
            2,
            None,
        )?;
        Ok(Self {
            inner: VisionInner {
                owner: Arc::new(()),
                tower,
                projector,
                hidden,
                out,
                dtype: DType::F32,
                device: vb.device().clone(),
            },
        })
    }
    pub(crate) fn prepare_rgb8(
        &self,
        rgb: &[u8],
        height: usize,
        width: usize,
    ) -> Result<PreparedVisionImage> {
        self.inner.prepare_rgb8(rgb, height, width)
    }
    pub(crate) fn forward_with_attention<F>(
        &self,
        images: &[&PreparedVisionImage],
        attend: &mut F,
    ) -> Result<Tensor>
    where
        F: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    {
        self.inner.forward_with_attention(images, attend)
    }
}

impl<P: Projection> VisionInner<P> {
    /// Caller owns decoding and pass admission BEFORE this allocates source
    /// copies, resize intermediates, patch arrays, upload/cast and padding.
    pub fn prepare_rgb8(
        &self,
        rgb: &[u8],
        height: usize,
        width: usize,
    ) -> Result<PreparedVisionImage> {
        let plan = VisionImagePlan::new(height, width, rgb.len())?;
        let p = processor();
        let (layout, tiles) = p.tiles_rgb8(rgb, plan.height, plan.width)?;
        debug_assert_eq!(layout, plan.layout);
        debug_assert_eq!(rgb.len(), plan.input_bytes);
        let mut prepared = Vec::with_capacity(tiles.len());
        for tile in tiles {
            let (rows, cols) = (tile.height / PATCH, tile.width / PATCH);
            let real = product(&[rows, cols])?;
            let input = p.patchify_rgb8_with_normalization(
                &tile.rgb,
                rows,
                cols,
                &self.device,
                self.dtype,
                RgbNormalization::ByteRange {
                    mean: [127.5; 3],
                    std: [127.5; 3],
                },
            )?;
            let patches = if real == MAX_PATCHES {
                input.patches
            } else {
                Tensor::cat(
                    &[
                        input.patches,
                        Tensor::zeros(
                            (1, MAX_PATCHES - real, 3 * PATCH * PATCH),
                            self.dtype,
                            &self.device,
                        )?,
                    ],
                    1,
                )?
            };
            let mask = Tensor::from_vec(
                (0..MAX_PATCHES)
                    .map(|i| if i < real { 0f32 } else { f32::NEG_INFINITY })
                    .collect::<Vec<_>>(),
                (1, 1, 1, MAX_PATCHES),
                &self.device,
            )?
            .to_dtype(self.dtype)?;
            prepared.push(VisionTile {
                rows,
                cols,
                patches,
                mask,
            });
        }
        Ok(PreparedVisionImage {
            owner: self.owner.clone(),
            tiles: prepared,
            prefix_tokens: plan.prefix_tokens,
        })
    }

    /// Full1024 padded rows, repeat-first positions and boolean-key-mask semantics
    /// expressed as 0/-infinity additive mask; no full L² mask is constructed.
    /// Image order, tile row-major order and final thumbnail are preserved.
    pub fn forward_with_attention<F>(
        &self,
        images: &[&PreparedVisionImage],
        attend: &mut F,
    ) -> Result<Tensor>
    where
        F: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor>,
    {
        if images.is_empty() || images.iter().any(|i| !Arc::ptr_eq(&self.owner, &i.owner)) {
            lloom_tensor::bail!(
                "omni vision requires nonempty images prepared by this model revision"
            )
        }
        let tokens = images
            .iter()
            .try_fold(0usize, |n, i| plus(n, i.prefix_tokens))?;
        product(&[tokens, self.out, self.dtype.size_in_bytes()])?;
        let mut parts = Vec::new();
        for image in images {
            for tile in &image.tiles {
                let encoded = self.tower.forward_with_attention(
                    &tile.patches,
                    tile.rows,
                    tile.cols,
                    VisionPositionPadding::RepeatFirst,
                    Some(&tile.mask),
                    attend,
                )?;
                let real = tile.rows * tile.cols;
                let grid =
                    encoded
                        .narrow(1, 0, real)?
                        .reshape((1, tile.rows, tile.cols, self.hidden))?;
                parts.push(self.projector.forward(&grid)?);
            }
        }
        Tensor::cat(&parts, 0)?.reshape((1, tokens, self.out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn geometry_matches_pinned_pure_python_layout_without_resizing() -> Result<()> {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("vision/layout_fixtures.json")).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let n = |k: &str| case[k].as_u64().unwrap() as usize;
            let (h, w) = (n("height"), n("width"));
            let result = VisionImagePlan::new(h, w, h * w * 3);
            if !case["position_grid_fits"].as_bool().unwrap() {
                assert!(result.is_err(), "oversized position grid admitted {h}x{w}");
                continue;
            }
            let plan = result?;
            assert_eq!((plan.layout.rows, plan.layout.cols), (n("rows"), n("cols")));
            let shape = case["resized"].as_array().unwrap();
            assert_eq!(
                plan.layout.resized,
                (
                    shape[0].as_u64().unwrap() as usize,
                    shape[1].as_u64().unwrap() as usize
                )
            );
            assert_eq!(plan.layout.is_grid(), case["tiled"].as_bool().unwrap());
            assert_eq!(plan.prefix_tokens, n("prefix_tokens"));
        }
        Ok(())
    }
    #[test]
    fn geometry_rejects_overflows_without_allocating_rgb_or_tensors() {
        for (h, w, len) in [
            (0, 1, 0),
            (1, 0, 0),
            (1, 1, 2),
            (usize::MAX, 1, 0),
            (isize::MAX as usize, 2, 0),
            (1, 65535, 65535 * 3),
        ] {
            assert!(
                VisionImagePlan::new(h, w, len).is_err(),
                "{h}x{w} unexpectedly admitted"
            );
        }
        assert!(product(&[usize::MAX, 2]).is_err());
        assert!(plus(isize::MAX as usize, 1).is_err());
        assert!(validate_resize(usize::MAX / 4, 1, 32, 512).is_err());
        assert!(validate_crop(32, 32 * 1025).is_err());
    }
}

#[cfg(test)]
#[path = "vision/composition_tests.rs"]
mod composition_tests;
