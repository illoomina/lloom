//! LFM2-VL: the SigLIP2-NaFlex vision tower and the multi-modal projector.
//!
//! This is the image half of `Lfm2VlForConditionalGeneration`. The language
//! half is [`super::lfm2`], which stays exactly as it was: this module produces
//! `[n_image_tokens, text_hidden_size]` embeddings, and `lfm2::Model::forward_embeds`
//! substitutes them at the image-token positions. Nothing here runs when a
//! request carries no image, which is what preserves `DESIGN.md` §5.5a's
//! measurement that the tower costs nothing when unused.
//!
//! # Why this is not a `siglip.rs` parameterisation
//!
//! **The encoder body IS shared** — this module composes `siglip::Encoder` over
//! its own config (`CONTRIBUTING.md` §3.6: a block is composed, never
//! copy-pasted because it lived in another file). The checkpoint names those 27
//! layers exactly as SigLIP does, so they are one mechanism with one spelling.
//!
//! **The embedding layer is not shared, and that is a real difference rather
//! than a placement one.** `siglip.rs` builds `patch_embedding` as a
//! [`crate::nn::Conv2d`] over a dense `[B, C, H, W]` image. The checkpoint's
//! tensor is `[1152, 768]` — **rank 2** — because SigLIP2-NaFlex embeds a
//! *pre-patchified sequence*: the image is cut into `16x16x3 = 768`-element
//! patches on the host and the tower sees `[B, n_patches, 768]`. A `conv2d`
//! cannot load a rank-2 weight, so this is a different mechanism, not a
//! different set of constants.
//!
//! # References
//!
//! - `transformers` `models/lfm2_vl/modeling_lfm2_vl.py` — the projector's
//!   pixel-unshuffle and the `masked_scatter` merge this mirrors.
//! - `google/siglip2-so400m-patch16-naflex` — the tower's upstream checkpoint,
//!   named in `text_config.mm_config_image_embedder.pretrained_model_name_or_path`.

use crate::nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder};
use lloom_tensor::{DType, Device, Result, Tensor, D};

/// The vision tower's hyperparameters, as they appear under `vision_config`.
///
/// `num_patches` is the side-squared of the *learned* position-embedding grid
/// (256 = 16x16), not a bound on what an image may present: NaFlex interpolates
/// that grid to whatever patch layout an image actually has.
#[derive(serde::Deserialize, Clone, Debug)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    pub patch_size: usize,
    #[serde(default = "default_num_patches")]
    pub num_patches: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: crate::nn::Activation,
}

fn default_num_channels() -> usize {
    3
}

fn default_num_patches() -> usize {
    256
}

fn default_layer_norm_eps() -> f64 {
    1e-6
}

fn default_hidden_act() -> crate::nn::Activation {
    crate::nn::Activation::GeluPytorchTanh
}

impl VisionConfig {
    /// `LiquidAI/LFM2.5-VL-3B`'s tower, from the checkpoint's `vision_config`.
    pub fn lfm2_vl_3b() -> Self {
        Self {
            hidden_size: 1152,
            intermediate_size: 4304,
            num_hidden_layers: 27,
            num_attention_heads: 16,
            num_channels: 3,
            patch_size: 16,
            num_patches: 256,
            layer_norm_eps: 1e-6,
            hidden_act: crate::nn::Activation::GeluPytorchTanh,
        }
    }

    /// The side of the learned position-embedding grid: 16 for a 256-entry table.
    pub fn pos_embed_side(&self) -> usize {
        (self.num_patches as f64).sqrt() as usize
    }

    /// Elements per patch — `C * P * P`, the tower's embedding input width.
    pub fn patch_elems(&self) -> usize {
        self.num_channels * self.patch_size * self.patch_size
    }
}

/// The top-level `lfm2_vl` config: the parts of it this module reads.
///
/// The language half is deserialized separately by [`super::lfm2`]; the fields
/// here are the ones that size the vision path and place image tokens.
#[derive(serde::Deserialize, Clone, Debug)]
pub struct Lfm2VlConfig {
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
    #[serde(default = "default_downsample_factor")]
    pub downsample_factor: usize,
    #[serde(default = "default_projector_hidden_size")]
    pub projector_hidden_size: usize,
    #[serde(default = "default_projector_bias")]
    pub projector_bias: bool,
    #[serde(default = "default_projector_use_layernorm")]
    pub projector_use_layernorm: bool,
    #[serde(default = "default_projector_hidden_act")]
    pub projector_hidden_act: crate::nn::Activation,
    #[serde(default = "default_encoder_patch_size")]
    pub encoder_patch_size: usize,
    #[serde(default = "default_tile_size")]
    pub tile_size: usize,
}

fn default_downsample_factor() -> usize {
    2
}

fn default_projector_hidden_size() -> usize {
    2048
}

fn default_projector_bias() -> bool {
    true
}

fn default_projector_use_layernorm() -> bool {
    false
}

fn default_projector_hidden_act() -> crate::nn::Activation {
    crate::nn::Activation::Gelu
}

fn default_encoder_patch_size() -> usize {
    16
}

fn default_tile_size() -> usize {
    512
}

impl Lfm2VlConfig {
    /// `LiquidAI/LFM2.5-VL-3B`, from the checkpoint's top-level config.
    pub fn lfm2_vl_3b() -> Self {
        Self {
            vision_config: VisionConfig::lfm2_vl_3b(),
            image_token_id: 124907,
            downsample_factor: 2,
            projector_hidden_size: 2048,
            projector_bias: true,
            projector_use_layernorm: false,
            projector_hidden_act: crate::nn::Activation::Gelu,
            encoder_patch_size: 16,
            tile_size: 512,
        }
    }
}

// ---------------------------------------------------------------------------
// The encoder body is SHARED with `siglip.rs`, not copied.
//
// `CONTRIBUTING.md` §3.6 / `DESIGN.md` §15.2 #15: a block is composed, never
// copy-pasted because it happened to live in another file. The 27 encoder
// layers here are structurally the same mechanism as SigLIP's — the checkpoint
// names them identically (`self_attn.{q,k,v,out}_proj`, `mlp.fc{1,2}`,
// `layer_norm{1,2}`) — so this model COMPOSES `siglip::Encoder` over its own
// config rather than carrying a second spelling of it.
//
// The first version of this file duplicated all four structs. That is exactly
// #599's SiLU-in-three-files defect, and the two copies would have been free to
// drift with nothing to say so.
//
// What is NOT shared is the embedding layer, and that is a real difference
// rather than a placement one: SigLIP embeds a dense image through a `Conv2d`,
// NaFlex embeds a pre-patchified sequence through a rank-2 `Linear`. See the
// module header.
// ---------------------------------------------------------------------------

impl super::siglip::TransformerConfig for VisionConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }
    fn layer_norm_eps(&self) -> f64 {
        self.layer_norm_eps
    }
    fn hidden_act(&self) -> crate::nn::Activation {
        self.hidden_act
    }
}

// ---------------------------------------------------------------------------
// The NaFlex embedding: a Linear over pre-patchified pixels, plus a position
// grid interpolated to the image's own patch layout.
// ---------------------------------------------------------------------------

/// Antialiased bilinear resample of the learned position grid.
///
/// The implementation moved to `crate::utils` for lloom #609, when `siglip.rs`
/// needed the same mathematics for the same purpose. Copying it would have been
/// the SiLU-in-three-files defect (#599) in a second place: one resampler, two
/// container layouts, and the tap weights computed once
/// (`utils::antialias_bilinear_taps`) so the two cannot drift.
use crate::utils::resize_pos_embed_antialias;

#[derive(Debug, Clone)]
struct VisionEmbeddings {
    patch_embedding: Linear,
    /// The learned `[side*side, dim]` table, kept on the host in f32 so it can
    /// be resampled per image without a device round trip.
    pos_table: Vec<f32>,
    pos_side: usize,
    hidden: usize,
}

impl VisionEmbeddings {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        // Rank 2, `[hidden, C*P*P]`. This is the line that would fail against a
        // standard SigLIP checkpoint, and failing loudly there is correct: the
        // two are different mechanisms wearing the same name.
        let patch_embedding = linear(cfg.patch_elems(), cfg.hidden_size, vb.pp("patch_embedding"))?;
        let pos_side = cfg.pos_embed_side();
        let pos = crate::nn::embedding(
            cfg.num_patches,
            cfg.hidden_size,
            vb.pp("position_embedding"),
        )?;
        // Held on the host in f32. The resample is per image and the table is
        // 256 x 1152 = 1.1 MB, so this is a one-time cost at load rather than a
        // device round trip per request.
        let pos_table = pos
            .embeddings()
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(Self {
            patch_embedding,
            pos_table,
            pos_side,
            hidden: cfg.hidden_size,
        })
    }

    /// `patches` is `[b, n, C*P*P]`; `(rows, cols)` is that image's patch grid,
    /// so `rows * cols` is the number of *real* patches and anything beyond it
    /// in `n` is padding whose position embedding does not matter.
    fn forward(&self, patches: &Tensor, rows: usize, cols: usize) -> Result<Tensor> {
        let embeddings = patches.apply(&self.patch_embedding)?;
        let device = embeddings.device();

        // Resize the learned grid to this image's layout. The 1:1 case is taken
        // verbatim rather than through the resampler — at rows == cols == side
        // every tap is a unit weight and the two agree, but taking the table
        // directly makes that agreement structural instead of arithmetic.
        let pos = if rows == self.pos_side && cols == self.pos_side {
            self.pos_table.clone()
        } else {
            resize_pos_embed_antialias(&self.pos_table, self.pos_side, self.hidden, rows, cols)
        };
        let pos = Tensor::from_vec(pos, (1, rows * cols, self.hidden), device)?
            .to_dtype(embeddings.dtype())?;

        let n_real = rows * cols;
        let (_, n, _) = embeddings.dims3()?;
        if n == n_real {
            embeddings.broadcast_add(&pos)
        } else {
            // Padded batch: add positions to the real prefix, leave the tail as
            // the embedding alone. The tail is masked out of attention and
            // dropped before the projector, so its value is never read — but it
            // must stay finite, because a NaN here would propagate through the
            // softmax into every real position.
            let real = embeddings.narrow(1, 0, n_real)?.broadcast_add(&pos)?;
            let tail = embeddings.narrow(1, n_real, n - n_real)?;
            Tensor::cat(&[real, tail], 1)
        }
    }
}

/// The SigLIP2-NaFlex vision tower.
#[derive(Debug, Clone)]
pub struct VisionTower {
    embeddings: VisionEmbeddings,
    /// `siglip.rs`'s encoder, composed rather than reimplemented. See the note
    /// above the `TransformerConfig` impl.
    encoder: super::siglip::Encoder,
    post_layernorm: LayerNorm,
}

impl VisionTower {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("vision_model");
        let embeddings = VisionEmbeddings::new(cfg, vb_m.pp("embeddings"))?;
        let encoder = super::siglip::Encoder::new(cfg, vb_m.pp("encoder"))?;
        let post_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb_m.pp("post_layernorm"),
        )?;
        Ok(Self {
            embeddings,
            encoder,
            post_layernorm,
        })
    }

    /// The patch projection alone, for a staged parity gate.
    pub fn patch_embed_only(&self, patches: &Tensor) -> Result<Tensor> {
        patches.apply(&self.embeddings.patch_embedding)
    }

    /// Patch projection plus position embeddings, before the encoder.
    pub fn embeddings_only(&self, patches: &Tensor, rows: usize, cols: usize) -> Result<Tensor> {
        self.embeddings.forward(patches, rows, cols)
    }

    /// Encode one image's patches to `[1, rows*cols, hidden]`.
    ///
    /// `patches` is `[1, rows*cols, C*P*P]` in row-major patch order. There is
    /// no padding at this entry point, so no attention mask is needed — see the
    /// module docs on why the single-tile path is exact without one.
    pub fn forward(&self, patches: &Tensor, rows: usize, cols: usize) -> Result<Tensor> {
        let xs = self.embeddings.forward(patches, rows, cols)?;
        let xs = self.encoder.forward(&xs, None)?;
        xs.apply(&self.post_layernorm)
    }
}

/// The multi-modal projector: `pixel_unshuffle` then a two-layer MLP.
#[derive(Debug, Clone)]
pub struct MultiModalProjector {
    layer_norm: Option<LayerNorm>,
    linear_1: Linear,
    linear_2: Linear,
    act: crate::nn::Activation,
    factor: usize,
}

impl MultiModalProjector {
    pub fn new(cfg: &Lfm2VlConfig, text_hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let in_channels = cfg.vision_config.hidden_size * cfg.downsample_factor.pow(2);
        let layer_norm = if cfg.projector_use_layernorm {
            Some(layer_norm(
                in_channels,
                cfg.vision_config.layer_norm_eps,
                vb.pp("layer_norm"),
            )?)
        } else {
            None
        };
        let (l1, l2) = if cfg.projector_bias {
            (
                linear(in_channels, cfg.projector_hidden_size, vb.pp("linear_1"))?,
                linear(
                    cfg.projector_hidden_size,
                    text_hidden_size,
                    vb.pp("linear_2"),
                )?,
            )
        } else {
            (
                crate::nn::linear_no_bias(
                    in_channels,
                    cfg.projector_hidden_size,
                    vb.pp("linear_1"),
                )?,
                crate::nn::linear_no_bias(
                    cfg.projector_hidden_size,
                    text_hidden_size,
                    vb.pp("linear_2"),
                )?,
            )
        };
        Ok(Self {
            layer_norm,
            linear_1: l1,
            linear_2: l2,
            act: cfg.projector_hidden_act,
            factor: cfg.downsample_factor,
        })
    }

    /// `xs` is `[1, rows, cols, hidden]`; the result is
    /// `[1, ceil(cols/f), ceil(rows/f), hidden * f^2]`.
    ///
    /// # Why the axes come back transposed
    ///
    /// This mirrors the reference `pixel_unshuffle` exactly, including that its
    /// two `permute`s do not compose to the identity: it names its first axis
    /// `width` and its second `height`, so a caller passing `[1, H, W, C]` gets
    /// `[1, ceil(W/f), ceil(H/f), C*f^2]` back. Reproducing the transposition is
    /// the point — the projector's weights were trained against this ordering,
    /// and "fixing" it would silently permute every image token.
    fn pixel_unshuffle(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, width, height, channels) = xs.dims4()?;
        let f = self.factor;
        let xs = xs.reshape((b, width, height / f, channels * f))?;
        let xs = xs.permute((0, 2, 1, 3))?.contiguous()?;
        let xs = xs.reshape((b, height / f, width / f, channels * f * f))?;
        xs.permute((0, 2, 1, 3))?.contiguous()
    }

    /// `image_features` is `[1, rows, cols, vision_hidden]` — the tower's output
    /// folded back to its 2-D patch layout. Returns `[n_tokens, text_hidden]`.
    pub fn forward(&self, image_features: &Tensor) -> Result<Tensor> {
        let xs = self.pixel_unshuffle(image_features)?;
        let xs = match self.layer_norm.as_ref() {
            None => xs,
            Some(ln) => xs.apply(ln)?,
        };
        let xs = xs.apply(&self.linear_1)?.apply(&self.act)?;
        let xs = xs.apply(&self.linear_2)?;
        let hidden = xs.dim(D::Minus1)?;
        xs.reshape(((), hidden))
    }
}

/// Tower plus projector: an image's patches to language-space embeddings.
#[derive(Debug, Clone)]
pub struct VisionEncoder {
    tower: VisionTower,
    projector: MultiModalProjector,
    downsample_factor: usize,
}

impl VisionEncoder {
    /// `vb` is rooted at the checkpoint's top level, so this walks to
    /// `model.vision_tower` and `model.multi_modal_projector` itself. Those two
    /// prefixes are the *only* tensors this reads — the 441 of `DESIGN.md`
    /// §5.5a — and nothing here touches the 266 the language model names.
    pub fn new(cfg: &Lfm2VlConfig, text_hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let tower = VisionTower::new(&cfg.vision_config, vb_m.pp("vision_tower"))?;
        let projector =
            MultiModalProjector::new(cfg, text_hidden_size, vb_m.pp("multi_modal_projector"))?;
        Ok(Self {
            tower,
            projector,
            downsample_factor: cfg.downsample_factor,
        })
    }

    /// How many image tokens an image with this patch grid produces.
    ///
    /// This must agree with the processor's `_compute_tokens_for_image`, because
    /// the text side reserves exactly this many `image_token_id` positions and
    /// the merge checks the two counts against each other.
    pub fn num_tokens(&self, rows: usize, cols: usize) -> usize {
        let f = self.downsample_factor;
        rows.div_ceil(f) * cols.div_ceil(f)
    }

    /// The tower's output alone, before the projector.
    ///
    /// Exposed so a parity gate can say WHICH half diverged. A single
    /// end-to-end comparison reports a projector defect and a tower defect
    /// identically, and the two want different repairs.
    pub fn tower_only(&self, patches: &Tensor, rows: usize, cols: usize) -> Result<Tensor> {
        self.tower.forward(patches, rows, cols)
    }

    /// The patch projection alone — `patches @ W^T + b`, no positions, no
    /// encoder. The first stage a parity gate can isolate.
    pub fn patch_embed_only(&self, patches: &Tensor) -> Result<Tensor> {
        self.tower.patch_embed_only(patches)
    }

    /// Patch projection plus position embeddings, before the encoder.
    pub fn embeddings_only(&self, patches: &Tensor, rows: usize, cols: usize) -> Result<Tensor> {
        self.tower.embeddings_only(patches, rows, cols)
    }

    /// Encode one image. `patches` is `[1, rows*cols, C*P*P]`; the result is
    /// `[num_tokens(rows, cols), text_hidden_size]`.
    pub fn forward(&self, patches: &Tensor, rows: usize, cols: usize) -> Result<Tensor> {
        let feats = self.tower.forward(patches, rows, cols)?;
        let hidden = feats.dim(D::Minus1)?;
        // Back to the 2-D patch layout the projector's unshuffle expects.
        let feats = feats.reshape((1, rows, cols, hidden))?;
        self.projector.forward(&feats)
    }
}

// ---------------------------------------------------------------------------
// Host-side preprocessing.
// ---------------------------------------------------------------------------

/// One image, resized and cut into patches, ready for the tower.
#[derive(Debug, Clone)]
pub struct ProcessedImage {
    /// `[1, rows*cols, C*P*P]`
    pub patches: Tensor,
    pub rows: usize,
    pub cols: usize,
}

impl ProcessedImage {
    pub fn num_patches(&self) -> usize {
        self.rows * self.cols
    }
}

/// The image preprocessing this path implements, mirroring
/// `Lfm2VlImageProcessorFast` for the single-tile case.
///
/// # What this covers, and what it deliberately does not
///
/// The reference processor can *tile* a large image (`do_image_splitting`, up to
/// `max_tiles`) and append a thumbnail. This implements the **single-tile**
/// branch only: the image is resized so its patch grid fits `max_num_patches`,
/// then cut. That branch is exact rather than approximate — it is the same code
/// path the reference takes when an image needs no splitting — and it is chosen
/// over a partial tiling implementation because a wrong tile layout produces a
/// plausible-looking embedding at the right shape, which no shape check catches.
#[derive(Debug, Clone)]
pub struct ImageProcessor {
    pub patch_size: usize,
    pub max_num_patches: usize,
    pub num_channels: usize,
    /// Per-channel mean, matching `processor_config.json`'s `image_mean`.
    pub image_mean: f32,
    pub image_std: f32,
}

impl Default for ImageProcessor {
    fn default() -> Self {
        Self {
            patch_size: 16,
            max_num_patches: 1024,
            num_channels: 3,
            image_mean: 0.5,
            image_std: 0.5,
        }
    }
}

impl ImageProcessor {
    /// The patch grid this processor would choose for an image of `(h, w)`.
    ///
    /// Both sides are rounded to a whole number of patches and the total is held
    /// at or under `max_num_patches` while preserving aspect ratio as closely as
    /// a patch grid can.
    pub fn target_grid(&self, h: usize, w: usize) -> (usize, usize) {
        let p = self.patch_size;
        let (h, w) = (h.max(p), w.max(p));
        // Largest scale s with ceil-to-patch dimensions whose product fits.
        let mut best = (1usize, 1usize);
        let aspect = w as f64 / h as f64;
        for rows in 1..=self.max_num_patches {
            let cols = ((rows as f64 * aspect).round() as usize).max(1);
            if rows * cols > self.max_num_patches {
                break;
            }
            best = (rows, cols);
        }
        best
    }

    /// Normalise an already-resized RGB8 buffer and cut it into patches.
    ///
    /// `rgb` is `rows*patch_size` by `cols*patch_size`, interleaved RGB, one
    /// byte per channel — i.e. exactly what `image`'s `to_rgb8()` hands over,
    /// without this crate having to depend on `image`. Decoding and resizing
    /// belong to the caller (`candle-examples` owns the codec dependency); the
    /// arithmetic that has to match the reference lives here, where it can be
    /// unit-tested without a file on disk.
    ///
    /// Returns patches in **row-major patch order**, each patch flattened as
    /// `P,P,C` — row-major pixels with the three channels **interleaved**, not
    /// channels-first.
    ///
    /// # The within-patch layout is measured, not assumed
    ///
    /// `patch_embedding`'s weight is `[1152, 768]` and `768 = 3*16*16`, which
    /// is equally consistent with `C,P,P` and `P,P,C`. Both produce a correctly
    /// shaped tensor and a model that runs; only one produces the right
    /// numbers. Checked against the reference processor's own `pixel_values`
    /// for a real image:
    ///
    /// | within-patch layout | max abs difference from the reference |
    /// |---|---|
    /// | `C,P,P` channels-first | 1.80e0 |
    /// | **`P,P,C` interleaved** | **5.22e-8** |
    ///
    /// The first version of this function used `C,P,P`, and nothing but that
    /// comparison could have said so: a permutation leaves the shape, the
    /// dtype and the token count identical, so the model consumes the wrong
    /// pixels and emits fluent text about them. `DESIGN.md` §10.2n records the
    /// same class in the DSpark tap, where a permuted concatenation was
    /// invisible to every gate in the tree.
    pub fn patchify_rgb8(
        &self,
        rgb: &[u8],
        rows: usize,
        cols: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<ProcessedImage> {
        let c = self.num_channels;
        let p = self.patch_size;
        let (h, w) = (rows * p, cols * p);
        let expected = h * w * c;
        if rgb.len() != expected {
            lloom_tensor::bail!(
                "rgb buffer is {} bytes, expected {expected} for a {h}x{w} image \
                 with {c} channels",
                rgb.len()
            );
        }

        let mut data = vec![0f32; rows * cols * c * p * p];
        for pr in 0..rows {
            for pc in 0..cols {
                let base = (pr * cols + pc) * c * p * p;
                for y in 0..p {
                    let src_row = (pr * p + y) * w;
                    for x in 0..p {
                        let src = (src_row + pc * p + x) * c;
                        let dst = base + (y * p + x) * c;
                        for ch in 0..c {
                            let v = rgb[src + ch] as f32 / 255.0;
                            data[dst + ch] = (v - self.image_mean) / self.image_std;
                        }
                    }
                }
            }
        }

        let patches =
            Tensor::from_vec(data, (1, rows * cols, c * p * p), device)?.to_dtype(dtype)?;
        Ok(ProcessedImage {
            patches,
            rows,
            cols,
        })
    }
}

/// Substitute `image_features` into `embeds` at every `image_token_id` position.
///
/// This is the Rust spelling of the reference's `masked_scatter`. It is a
/// separate free function rather than a method so the language model does not
/// have to know the vision types exist.
///
/// # The count check is the load-bearing part
///
/// The text side reserved `n` positions and the vision side produced `m` rows.
/// If those disagree the model still runs — `masked_scatter` in the reference
/// would raise, but a hand-rolled loop would happily fill a prefix and leave the
/// rest as `<image>`'s own learned embedding, producing fluent output about the
/// wrong thing. Refusing here is what makes that failure loud.
pub fn merge_image_features(
    embeds: &Tensor,
    input_ids: &[u32],
    image_features: &Tensor,
    image_token_id: u32,
) -> Result<Tensor> {
    let positions: Vec<usize> = input_ids
        .iter()
        .enumerate()
        .filter_map(|(i, &t)| (t == image_token_id).then_some(i))
        .collect();
    let n_features = image_features.dim(0)?;
    if positions.len() != n_features {
        lloom_tensor::bail!(
            "image tokens and image features do not match: {} `image_token_id` \
             positions in the prompt against {} projected features",
            positions.len(),
            n_features
        );
    }
    if positions.is_empty() {
        return Ok(embeds.clone());
    }

    let (b, seq, hidden) = embeds.dims3()?;
    if b != 1 {
        lloom_tensor::bail!("merge_image_features is B=1 only, got batch {b}");
    }
    let features = image_features.to_dtype(embeds.dtype())?;

    // Splice by concatenating the runs between image positions with the feature
    // rows, rather than scattering in place: candle has no masked_scatter, and a
    // narrow/cat walk keeps every element's provenance explicit.
    let flat = embeds.reshape((seq, hidden))?;
    let mut parts: Vec<Tensor> = Vec::new();
    let mut cursor = 0usize;
    for (k, &pos) in positions.iter().enumerate() {
        if pos > cursor {
            parts.push(flat.narrow(0, cursor, pos - cursor)?);
        }
        parts.push(features.narrow(0, k, 1)?);
        cursor = pos + 1;
    }
    if cursor < seq {
        parts.push(flat.narrow(0, cursor, seq - cursor)?);
    }
    Tensor::cat(&parts, 0)?.reshape((1, seq, hidden))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Lfm2VlConfig {
        Lfm2VlConfig::lfm2_vl_3b()
    }

    #[test]
    fn the_checkpoints_patch_embedding_width_is_channels_times_patch_area() {
        // `[1152, 768]` in the checkpoint; 768 must be C*P*P or the tower is
        // being loaded against the wrong mechanism.
        assert_eq!(cfg().vision_config.patch_elems(), 768);
    }

    #[test]
    fn the_position_grid_is_square_and_matches_the_checkpoints_256_entries() {
        assert_eq!(cfg().vision_config.pos_embed_side(), 16);
        assert_eq!(cfg().vision_config.num_patches, 256);
    }

    #[test]
    fn the_projector_input_width_is_the_towers_hidden_times_the_downsample_area() {
        let c = cfg();
        // The checkpoint's `linear_1.weight` is `[2048, 4608]`.
        assert_eq!(
            c.vision_config.hidden_size * c.downsample_factor.pow(2),
            4608
        );
    }

    /// The token count this module predicts must equal the processor's, because
    /// the prompt reserves positions from one and the merge fills them from the
    /// other. This is `_compute_tokens_for_image` restated against ours.
    #[test]
    fn token_counts_agree_with_the_reference_processors_formula() {
        let f = 2usize;
        for (h, w) in [
            (512usize, 512usize),
            (384, 512),
            (256, 256),
            (512, 256),
            (511, 333),
        ] {
            let (rows, cols) = (h / 16, w / 16);
            let ours = rows.div_ceil(f) * cols.div_ceil(f);
            // reference: ceil((H // patch) / ds) * ceil((W // patch) / ds)
            let reference = ((rows as f64) / f as f64).ceil() as usize
                * ((cols as f64) / f as f64).ceil() as usize;
            assert_eq!(ours, reference, "grid {rows}x{cols}");
        }
    }

    #[test]
    fn a_512_square_image_is_exactly_1024_patches_and_256_tokens() {
        let p = ImageProcessor::default();
        let (rows, cols) = p.target_grid(512, 512);
        assert_eq!(rows * cols, 1024, "grid was {rows}x{cols}");
        assert_eq!(rows.div_ceil(2) * cols.div_ceil(2), 256);
    }

    #[test]
    fn the_chosen_grid_never_exceeds_max_num_patches() {
        let p = ImageProcessor::default();
        for (h, w) in [
            (64, 64),
            (512, 512),
            (1024, 768),
            (4000, 3000),
            (100, 4000),
            (4000, 100),
        ] {
            let (rows, cols) = p.target_grid(h, w);
            assert!(
                rows * cols <= p.max_num_patches,
                "{h}x{w} -> {rows}x{cols} = {} patches",
                rows * cols
            );
            assert!(rows >= 1 && cols >= 1);
        }
    }
}
