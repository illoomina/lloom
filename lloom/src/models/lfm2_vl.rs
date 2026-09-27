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
    /// Wrap each image in `<|image_start|>` / `<|image_end|>` and mark its
    /// tiles (`Lfm2VlProcessor`'s `use_image_special_tokens`).
    #[serde(default = "default_true")]
    pub use_image_special_tokens: bool,
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
            use_image_special_tokens: true,
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

    /// Encode one tile. `patches` is `[1, rows*cols, C*P*P]`; the result is
    /// `[num_tokens(rows, cols), text_hidden_size]`.
    pub fn forward(&self, patches: &Tensor, rows: usize, cols: usize) -> Result<Tensor> {
        let feats = self.tower.forward(patches, rows, cols)?;
        let hidden = feats.dim(D::Minus1)?;
        // Back to the 2-D patch layout the projector's unshuffle expects.
        let feats = feats.reshape((1, rows, cols, hidden))?;
        self.projector.forward(&feats)
    }

    /// Encode a prepared image: every tile, in order, concatenated to
    /// `[image_tokens, text_hidden_size]` — the rows the placeholder reserves,
    /// in the order it reserves them.
    ///
    /// Each tile is a separate unpadded pass. The reference batches the tiles
    /// through the tower padded to `max_num_patches` under an attention mask;
    /// the mask removes the padding from every real row's attention, so the
    /// two differ only in accumulation order (`DESIGN.md` §5.12b).
    pub fn encode_image(&self, image: &PreparedImage) -> Result<Tensor> {
        let parts = image
            .tiles
            .iter()
            .map(|t| self.forward(&t.patches, t.rows, t.cols))
            .collect::<Result<Vec<_>>>()?;
        match parts.len() {
            0 => lloom_tensor::bail!("an image with no tiles"),
            1 => Ok(parts.into_iter().next().expect("one part")),
            _ => Tensor::cat(&parts, 0),
        }
    }
}

/// The config `VisionEncoder` is constructed from through the `Model` trait.
///
/// It pairs the VL config with the text model's hidden size, because the tower
/// projects into the language model's embedding space and that width is a
/// property of the *text* config (`text_config.hidden_size`), not the vision
/// one. `VisionEncoder::new` has always taken it as a separate argument; the
/// trait wants a single `Config`, so this carries both.
#[derive(Debug, Clone)]
pub struct VisionEncoderConfig {
    pub vl: Lfm2VlConfig,
    pub text_hidden_size: usize,
}

impl crate::nn::Model for VisionEncoder {
    type Config = VisionEncoderConfig;

    fn from_weights(vb: VarBuilder, cfg: &Self::Config) -> Result<Self> {
        VisionEncoder::new(&cfg.vl, cfg.text_hidden_size, vb)
    }
}

impl crate::nn::VisionEncoder for VisionEncoder {
    type Input = PreparedImage;

    /// Wires to [`VisionEncoder::encode_image`]: one image, every tile. A
    /// single-tile image is one `forward`, so the §5.12 staged gate is unmoved.
    fn encode(&self, input: &Self::Input) -> Result<Tensor> {
        self.encode_image(input)
    }
}

// ---------------------------------------------------------------------------
// Host-side preprocessing.
// ---------------------------------------------------------------------------

/// One tile, resized and cut into patches, ready for the tower.
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

/// How the reference processor lays one image out: a single resized tile, or a
/// grid of `tile_size` tiles followed by a thumbnail of the whole image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileLayout {
    /// Tile rows and columns. `1 x 1` is the single-tile path.
    pub rows: usize,
    pub cols: usize,
    /// `(height, width)` of the single tile, or of the thumbnail.
    pub resized: (usize, usize),
    /// Whether a thumbnail follows the grid tiles.
    pub thumbnail: bool,
}

impl TileLayout {
    /// The reference's `is_multi_tile`: the placeholder takes row/column
    /// markers and per-tile token runs.
    pub fn is_grid(&self) -> bool {
        self.rows > 1 || self.cols > 1
    }

    /// Tiles the tower encodes, in order: the grid, then the thumbnail.
    pub fn num_tiles(&self) -> usize {
        if self.is_grid() {
            self.rows * self.cols + usize::from(self.thumbnail)
        } else {
            1
        }
    }
}

/// One tile of an image as 8-bit interleaved RGB, before normalisation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbTile {
    pub height: usize,
    pub width: usize,
    pub rgb: Vec<u8>,
}

/// An image ready for the tower: its layout and every tile's patches, in the
/// order the placeholder reserves their tokens.
#[derive(Debug, Clone)]
pub struct PreparedImage {
    pub layout: TileLayout,
    pub tiles: Vec<ProcessedImage>,
}

fn default_true() -> bool {
    true
}
fn default_min_tiles() -> usize {
    2
}
fn default_max_tiles() -> usize {
    10
}
fn default_min_image_tokens() -> usize {
    64
}
fn default_max_image_tokens() -> usize {
    256
}
fn default_patch() -> usize {
    16
}
fn default_pixels_tolerance() -> f64 {
    2.0
}
fn default_resample() -> u32 {
    // `PILImageResampling.BILINEAR`, the reference class's own default; a
    // checkpoint's `processor_config.json` overrides it (LFM2.5-VL-3B: 3).
    2
}
fn default_half() -> [f32; 3] {
    [0.5; 3]
}

/// The image preprocessing of `Lfm2VlImageProcessor` (transformers 5.16), from
/// the checkpoint's `processor_config.json`.
///
/// An image whose rounded area exceeds `max_pixels_tolerance` times the
/// single-tile budget is split into a grid of `tile_size` tiles chosen to match
/// its aspect ratio, and a thumbnail of the whole image follows the grid; any
/// other image is resized to one tile. Every size, the grid choice and the
/// resize itself reproduce the reference's, because a wrong layout produces a
/// plausible embedding at the right shape that no shape check catches. The
/// gate is `lfm2-vl --gate-tiles` (`DESIGN.md` §5.12b).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ImageProcessor {
    #[serde(default = "default_true")]
    pub do_image_splitting: bool,
    #[serde(default = "default_min_tiles")]
    pub min_tiles: usize,
    #[serde(default = "default_max_tiles")]
    pub max_tiles: usize,
    #[serde(default = "default_true")]
    pub use_thumbnail: bool,
    #[serde(default = "default_min_image_tokens")]
    pub min_image_tokens: usize,
    #[serde(default = "default_max_image_tokens")]
    pub max_image_tokens: usize,
    #[serde(rename = "encoder_patch_size", default = "default_patch")]
    pub patch_size: usize,
    #[serde(default = "default_downsample_factor")]
    pub downsample_factor: usize,
    #[serde(default = "default_tile_size")]
    pub tile_size: usize,
    #[serde(default = "default_pixels_tolerance")]
    pub max_pixels_tolerance: f64,
    /// `PILImageResampling`: 2 bilinear, 3 bicubic.
    #[serde(default = "default_resample")]
    pub resample: u32,
    #[serde(default = "default_half")]
    pub image_mean: [f32; 3],
    #[serde(default = "default_half")]
    pub image_std: [f32; 3],
    #[serde(skip, default = "default_num_channels")]
    pub num_channels: usize,
}

impl Default for ImageProcessor {
    fn default() -> Self {
        Self::lfm2_vl_3b()
    }
}

/// Python's `round(number / factor) * factor`: round half to EVEN, so 80 at a
/// factor of 32 is 64, not 96.
fn round_by_factor(number: usize, factor: usize) -> usize {
    (number as f64 / factor as f64).round_ties_even() as usize * factor
}

impl ImageProcessor {
    /// `LiquidAI/LFM2.5-VL-3B`'s `processor_config.json`.
    pub fn lfm2_vl_3b() -> Self {
        Self {
            do_image_splitting: true,
            min_tiles: 2,
            max_tiles: 10,
            use_thumbnail: true,
            min_image_tokens: 64,
            max_image_tokens: 256,
            patch_size: 16,
            downsample_factor: 2,
            tile_size: 512,
            max_pixels_tolerance: 2.0,
            resample: 3,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            num_channels: 3,
        }
    }

    /// The processor a checkpoint directory declares: `processor_config.json`'s
    /// `image_processor` block, or a flat `preprocessor_config.json`.
    ///
    /// `config.json` carries tiling keys too, and they disagree with the
    /// processor's (`min_tiles` 1 against 2 on LFM2.5-VL-3B); the reference
    /// processor reads only the processor file, so that is the one read here.
    pub fn from_checkpoint(dir: &std::path::Path) -> Result<Self> {
        let nested = dir.join("processor_config.json");
        let flat = dir.join("preprocessor_config.json");
        let value: serde_json::Value = if nested.is_file() {
            let v: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&nested).map_err(lloom_tensor::Error::wrap)?)
                    .map_err(lloom_tensor::Error::wrap)?;
            match v.get("image_processor") {
                Some(ip) => ip.clone(),
                None => v,
            }
        } else if flat.is_file() {
            serde_json::from_slice(&std::fs::read(&flat).map_err(lloom_tensor::Error::wrap)?)
                .map_err(lloom_tensor::Error::wrap)?
        } else {
            lloom_tensor::bail!(
                "{}: no processor_config.json or preprocessor_config.json; the image \
                 preprocessing is the checkpoint's, not a default",
                dir.display()
            )
        };
        let p: Self = serde_json::from_value(value).map_err(lloom_tensor::Error::wrap)?;
        p.filter()?;
        if p.splitting() && p.min_tiles > p.max_tiles {
            lloom_tensor::bail!("min_tiles {} > max_tiles {}", p.min_tiles, p.max_tiles);
        }
        Ok(p)
    }

    /// The resample filter `resample` names.
    pub fn filter(&self) -> Result<crate::utils::AaFilter> {
        Ok(match self.resample {
            2 => crate::utils::AaFilter::Bilinear,
            3 => crate::utils::AaFilter::Bicubic,
            r => lloom_tensor::bail!(
                "resample = {r}: only 2 (bilinear) and 3 (bicubic) are reproduced"
            ),
        })
    }

    fn splitting(&self) -> bool {
        self.do_image_splitting && !(self.min_tiles == 1 && self.max_tiles == 1)
    }

    fn total_factor(&self) -> usize {
        self.patch_size * self.downsample_factor
    }

    fn pixels_per_token(&self) -> usize {
        self.patch_size * self.patch_size * self.downsample_factor * self.downsample_factor
    }

    /// The patches one tower pass may take: a full tile, or the largest single
    /// image.
    pub fn max_num_patches(&self) -> usize {
        let thumbnail = self.max_image_tokens * self.downsample_factor * self.downsample_factor;
        let tile = if self.do_image_splitting {
            (self.tile_size / self.patch_size).pow(2)
        } else {
            0
        };
        thumbnail.max(tile)
    }

    /// `smart_resize`: `(height, width)` for a single tile or a thumbnail. Both
    /// sides are multiples of `patch_size * downsample_factor` and the area is
    /// brought inside `[min_image_tokens, max_image_tokens]` tokens.
    pub fn smart_resize(&self, height: usize, width: usize) -> (usize, usize) {
        let f = self.total_factor();
        let min_px = self.min_image_tokens * self.pixels_per_token();
        let max_px = self.max_image_tokens * self.pixels_per_token();
        let mut h_bar = f.max(round_by_factor(height, f));
        let mut w_bar = f.max(round_by_factor(width, f));
        let (h, w, ff) = (height as f64, width as f64, f as f64);
        if h_bar * w_bar > max_px {
            let beta = ((height * width) as f64 / max_px as f64).sqrt();
            h_bar = f.max((h / beta / ff).floor() as usize * f);
            w_bar = f.max((w / beta / ff).floor() as usize * f);
        } else if h_bar * w_bar < min_px {
            let beta = (min_px as f64 / (height * width) as f64).sqrt();
            h_bar = (h * beta / ff).ceil() as usize * f;
            w_bar = (w * beta / ff).ceil() as usize * f;
        }
        (h_bar, w_bar)
    }

    /// `_is_image_too_large`: whether the image is tiled rather than resized.
    pub fn is_too_large(&self, height: usize, width: usize) -> bool {
        let f = self.total_factor();
        let h_bar = self.patch_size.max(round_by_factor(height, f));
        let w_bar = self.patch_size.max(round_by_factor(width, f));
        (h_bar * w_bar) as f64
            > (self.max_image_tokens * self.pixels_per_token()) as f64 * self.max_pixels_tolerance
    }

    /// `_get_grid_layout`: the `(columns, rows)` grid whose aspect is closest to
    /// the image's, among grids of `min_tiles..=max_tiles` tiles. On a tie the
    /// later (larger) grid wins when the image covers more than half of it.
    ///
    /// The reference sorts a Python `set` by area, so its order WITHIN an area
    /// is hash order; this enumerates in first-occurrence order. The two choose
    /// identically on every size up to 2000 x 2000 at `min_tiles` 1 and 2
    /// (`measurements/issue-784-raw/ratio_order_check.py`), because a tie that
    /// decides the choice never falls between two grids of the same area.
    fn grid(&self, height: usize, width: usize) -> (usize, usize) {
        let mut ratios: Vec<(usize, usize)> = Vec::new();
        for n in self.min_tiles..=self.max_tiles {
            for w in 1..=n {
                for h in 1..=n {
                    let a = w * h;
                    if (self.min_tiles..=self.max_tiles).contains(&a) && !ratios.contains(&(w, h)) {
                        ratios.push((w, h));
                    }
                }
            }
        }
        ratios.sort_by_key(|&(w, h)| w * h);
        let aspect = width as f64 / height as f64;
        let area = (width * height) as f64;
        let mut best = (1usize, 1usize);
        let mut best_diff = f64::INFINITY;
        for (w, h) in ratios {
            let diff = (aspect - w as f64 / h as f64).abs();
            if diff < best_diff {
                best_diff = diff;
                best = (w, h);
            } else if diff == best_diff {
                let target_area = (self.tile_size * self.tile_size * w * h) as f64;
                if area > 0.5 * target_area {
                    best = (w, h);
                }
            }
        }
        best
    }

    /// `resize_and_split`'s layout for an image of `(height, width)`.
    ///
    /// One case is not the reference's. A 1 x 1 grid, reachable only at
    /// `min_tiles` 1, makes the reference resize to a full `tile_size` tile
    /// while its placeholder counts tokens from the `smart_resize` size, so its
    /// feature and token counts disagree and its model refuses the pair
    /// (688 x 752: 256 against 240, `measurements/issue-784-raw/min_tiles_check.txt`).
    /// That case takes the single-tile path here, whose counts agree.
    pub fn layout(&self, height: usize, width: usize) -> TileLayout {
        let resized = self.smart_resize(height, width);
        let grid = (self.splitting() && self.is_too_large(height, width))
            .then(|| self.grid(height, width))
            .filter(|&(cols, rows)| cols * rows > 1);
        if let Some((cols, rows)) = grid {
            TileLayout {
                rows,
                cols,
                resized,
                thumbnail: self.use_thumbnail && rows * cols != 1,
            }
        } else {
            TileLayout {
                rows: 1,
                cols: 1,
                resized,
                thumbnail: false,
            }
        }
    }

    /// Image tokens per grid tile.
    pub fn tokens_per_tile(&self) -> usize {
        let side = (self.tile_size / self.patch_size).div_ceil(self.downsample_factor);
        side * side
    }

    /// Image tokens for a single tile or a thumbnail of `(height, width)`.
    pub fn tokens_for(&self, (height, width): (usize, usize)) -> usize {
        (height / self.patch_size).div_ceil(self.downsample_factor)
            * (width / self.patch_size).div_ceil(self.downsample_factor)
    }

    /// Every image token `layout` reserves.
    pub fn image_tokens(&self, layout: &TileLayout) -> usize {
        if layout.is_grid() {
            layout.rows * layout.cols * self.tokens_per_tile()
                + if self.use_thumbnail {
                    self.tokens_for(layout.resized)
                } else {
                    0
                }
        } else {
            self.tokens_for(layout.resized)
        }
    }

    /// The text one `<image>` expands to (`Lfm2VlProcessor._build_image_tokens`):
    /// `<|image_start|>`, per tile `<|img_row_R_col_C|>` and its run of
    /// `<image>`, then `<|img_thumbnail|>` and the thumbnail's run, then
    /// `<|image_end|>`. The delimiters are ordinary tokens with their own
    /// embeddings; only `<image>` positions take features.
    pub fn placeholder(&self, layout: &TileLayout, special_tokens: bool) -> String {
        let mut s = String::new();
        if special_tokens {
            s.push_str("<|image_start|>");
        }
        if layout.is_grid() {
            for r in 0..layout.rows {
                for c in 0..layout.cols {
                    if special_tokens {
                        s.push_str(&format!("<|img_row_{}_col_{}|>", r + 1, c + 1));
                    }
                    s.push_str(&"<image>".repeat(self.tokens_per_tile()));
                }
            }
            if self.use_thumbnail {
                if special_tokens {
                    s.push_str("<|img_thumbnail|>");
                }
                s.push_str(&"<image>".repeat(self.tokens_for(layout.resized)));
            }
        } else {
            s.push_str(&"<image>".repeat(self.tokens_for(layout.resized)));
        }
        if special_tokens {
            s.push_str("<|image_end|>");
        }
        s
    }

    /// Resize and split an interleaved RGB8 image of `(height, width)` into the
    /// tiles the tower encodes, in the reference's order: the grid row-major
    /// (`split_to_tiles`), then the thumbnail. Every tile is byte-identical to
    /// the reference's `uint8` tile (`utils::resize_rgb8_antialias`).
    pub fn tiles_rgb8(
        &self,
        rgb: &[u8],
        height: usize,
        width: usize,
    ) -> Result<(TileLayout, Vec<RgbTile>)> {
        let c = self.num_channels;
        let filter = self.filter()?;
        let layout = self.layout(height, width);
        let resize = |nh: usize, nw: usize| -> Result<RgbTile> {
            Ok(RgbTile {
                height: nh,
                width: nw,
                rgb: crate::utils::resize_rgb8_antialias(rgb, height, width, c, nh, nw, filter)?,
            })
        };
        let mut tiles = Vec::with_capacity(layout.num_tiles());
        if layout.is_grid() {
            let t = self.tile_size;
            let grid = resize(layout.rows * t, layout.cols * t)?;
            let row_bytes = grid.width * c;
            for r in 0..layout.rows {
                for col in 0..layout.cols {
                    let mut tile = Vec::with_capacity(t * t * c);
                    for y in 0..t {
                        let start = (r * t + y) * row_bytes + col * t * c;
                        tile.extend_from_slice(&grid.rgb[start..start + t * c]);
                    }
                    tiles.push(RgbTile {
                        height: t,
                        width: t,
                        rgb: tile,
                    });
                }
            }
            if layout.thumbnail {
                tiles.push(resize(layout.resized.0, layout.resized.1)?);
            }
        } else {
            tiles.push(resize(layout.resized.0, layout.resized.1)?);
        }
        Ok((layout, tiles))
    }

    /// [`Self::tiles_rgb8`], then every tile normalised and cut into patches.
    pub fn prepare_rgb8(
        &self,
        rgb: &[u8],
        height: usize,
        width: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<PreparedImage> {
        let (layout, tiles) = self.tiles_rgb8(rgb, height, width)?;
        let p = self.patch_size;
        let tiles = tiles
            .iter()
            .map(|t| self.patchify_rgb8(&t.rgb, t.height / p, t.width / p, device, dtype))
            .collect::<Result<Vec<_>>>()?;
        Ok(PreparedImage { layout, tiles })
    }

    /// Normalise an already-resized RGB8 tile and cut it into patches.
    ///
    /// `rgb` is `rows*patch_size` by `cols*patch_size`, interleaved RGB, one
    /// byte per channel — i.e. exactly what `image`'s `to_rgb8()` hands over,
    /// without this crate having to depend on `image`. Decoding belongs to the
    /// caller; the arithmetic that has to match the reference lives here, where
    /// it can be unit-tested without a file on disk.
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
    /// A permutation leaves the shape, the dtype and the token count
    /// identical, so the model consumes the wrong pixels and emits fluent text
    /// about them. `DESIGN.md` §10.2n records the same class in the DSpark tap.
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
                            data[dst + ch] = (v - self.image_mean[ch]) / self.image_std[ch];
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

    /// `(height, width, rows, cols, resized_h, resized_w, thumbnail,
    /// image_tokens, ids_len)`, emitted by the reference processor
    /// (transformers 5.16.1, `measurements/issue-784-raw/make_golden_tiles.py`,
    /// `golden_tiles.json`'s `layout`). `ids_len` is the token count of one
    /// expanded `<image>`: image tokens plus the start/end, per-tile and
    /// thumbnail markers.
    const REFERENCE_LAYOUTS: &[(usize, usize, usize, usize, usize, usize, bool, usize, usize)] = &[
        (64, 64, 1, 1, 256, 256, false, 64, 66),
        (100, 300, 1, 1, 160, 448, false, 70, 72),
        (300, 200, 1, 1, 320, 224, false, 70, 72),
        (384, 512, 1, 1, 384, 512, false, 192, 194),
        (511, 333, 1, 1, 512, 320, false, 160, 162),
        (512, 512, 1, 1, 512, 512, false, 256, 258),
        (600, 800, 1, 1, 416, 576, false, 234, 236),
        (512, 1024, 1, 1, 352, 704, false, 242, 244),
        (544, 1024, 1, 2, 352, 672, true, 743, 748),
        (768, 768, 2, 2, 512, 512, true, 1280, 1287),
        (1024, 1024, 2, 2, 512, 512, true, 1280, 1287),
        (768, 1536, 2, 4, 352, 704, true, 2290, 2301),
        (641, 853, 2, 3, 416, 576, true, 1770, 1779),
        (900, 2000, 2, 4, 320, 736, true, 2278, 2289),
        (1366, 768, 4, 2, 672, 352, true, 2279, 2290),
        (3000, 4000, 2, 3, 416, 576, true, 1770, 1779),
        (100, 4000, 1, 1, 64, 3232, false, 202, 204),
        (4000, 100, 1, 1, 3232, 64, false, 202, 204),
        (2048, 512, 4, 1, 1024, 256, true, 1280, 1287),
        (1080, 1920, 2, 4, 384, 672, true, 2300, 2311),
        (720, 720, 1, 1, 512, 512, false, 256, 258),
        (80, 112, 1, 1, 224, 320, false, 70, 72),
        (688, 752, 2, 2, 480, 512, true, 1264, 1271),
    ];

    /// Every size in the reference table lays out as the reference laid it
    /// out: the grid, the single/thumbnail size, whether a thumbnail follows,
    /// and the image-token count. `720 x 720` is the round-half-to-even case —
    /// half-up rounding tiles it 2 x 2.
    #[test]
    fn the_layout_is_the_reference_processors_on_every_tabled_size() {
        let p = ImageProcessor::lfm2_vl_3b();
        for &(h, w, rows, cols, rh, rw, thumb, tokens, _) in REFERENCE_LAYOUTS {
            let l = p.layout(h, w);
            assert_eq!(
                (l.rows, l.cols, l.resized, l.thumbnail),
                (rows, cols, (rh, rw), thumb),
                "{h}x{w}"
            );
            assert_eq!(p.image_tokens(&l), tokens, "{h}x{w} image tokens");
        }
    }

    /// The expanded placeholder has exactly the reference's token count: its
    /// `<image>` runs plus one token per delimiter and marker. Counted from
    /// the string, so a missing marker or a wrong run length moves it.
    #[test]
    fn the_placeholder_has_the_reference_token_count() {
        let p = ImageProcessor::lfm2_vl_3b();
        for &(h, w, _, _, _, _, _, tokens, ids_len) in REFERENCE_LAYOUTS {
            let l = p.layout(h, w);
            let s = p.placeholder(&l, true);
            let images = s.matches("<image>").count();
            let markers = s.matches("<|").count();
            assert_eq!(images, tokens, "{h}x{w}");
            assert_eq!(images + markers, ids_len, "{h}x{w}: {markers} markers");
        }
    }

    /// `min_tiles` comes from the processor file (2), not from `config.json`
    /// (1). The two lay out differently only in a band just past the tiling
    /// threshold (140 of 10 000 sizes in 680..779 squared, none elsewhere,
    /// `measurements/issue-784-raw/min_tiles_check.py`); 688 x 752 is in it and
    /// is tabled above, so a processor reading 1 fails the layout test.
    #[test]
    fn the_checkpoints_processor_file_is_what_sets_min_tiles() {
        let dir = std::env::temp_dir().join(format!("lloom-784-proc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("processor_config.json"),
            r#"{"image_processor": {"min_tiles": 2, "max_tiles": 10, "resample": 3,
                "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                "encoder_patch_size": 16, "tile_size": 512}, "processor_class": "Lfm2VlProcessor"}"#,
        )
        .unwrap();
        let p = ImageProcessor::from_checkpoint(&dir).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!((p.min_tiles, p.max_tiles, p.resample), (2, 10, 3));
        assert_eq!(p.filter().unwrap(), crate::utils::AaFilter::Bicubic);
        let l = p.layout(688, 752);
        assert_eq!((l.rows, l.cols, l.thumbnail), (2, 2, true));
        let one = ImageProcessor { min_tiles: 1, ..p };
        let l = one.layout(688, 752);
        assert_eq!(
            (l.rows, l.cols, l.thumbnail),
            (1, 1, false),
            "the band min_tiles decides"
        );
    }

    /// The tiles come out row-major, then the thumbnail: each grid tile of an
    /// image whose pixels encode their own tile index carries that index.
    #[test]
    fn tiles_are_row_major_then_the_thumbnail() {
        let p = ImageProcessor::lfm2_vl_3b();
        // 1024 x 1536 is a 2 x 3 grid at exactly the tile size: no resample,
        // so every tile's bytes are its source region's.
        let (h, w) = (1024usize, 1536usize);
        let mut rgb = vec![0u8; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                let k = ((y / 512) * 3 + x / 512) as u8;
                rgb[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&[k, 10 * k, 255 - k]);
            }
        }
        let (layout, tiles) = p.tiles_rgb8(&rgb, h, w).unwrap();
        assert_eq!((layout.rows, layout.cols, layout.thumbnail), (2, 3, true));
        assert_eq!(tiles.len(), 7);
        for (k, t) in tiles.iter().take(6).enumerate() {
            assert_eq!((t.height, t.width), (512, 512));
            let k = k as u8;
            assert!(
                t.rgb.chunks(3).all(|px| px == [k, 10 * k, 255 - k]),
                "tile {k} is not region {k}"
            );
        }
        assert_eq!((tiles[6].height, tiles[6].width), layout.resized);
    }

    /// Row `r` of a `[1, seq, hidden]` embedding filled with `base + r*hidden + j`,
    /// so every element names its own row and column.
    fn numbered(seq: usize, hidden: usize, base: f32) -> Tensor {
        let v: Vec<f32> = (0..seq * hidden).map(|i| base + i as f32).collect();
        Tensor::from_vec(v, (1, seq, hidden), &Device::Cpu).unwrap()
    }

    fn rows(t: &Tensor) -> Vec<Vec<f32>> {
        let (_, seq, hidden) = t.dims3().unwrap();
        t.reshape((seq, hidden)).unwrap().to_vec2::<f32>().unwrap()
    }

    /// The splice puts feature row `k` at the `k`-th image position, in order,
    /// and leaves every text row bit-identical. The positions are deliberately
    /// non-contiguous and include the first and last slots, so an off-by-one in
    /// the run bookkeeping or a feature index that does not advance moves a
    /// value this compares against an absolute expectation.
    #[test]
    fn merge_places_feature_k_at_the_kth_image_position_and_nothing_else() {
        const IMG: u32 = 124907;
        let (seq, hidden) = (7usize, 3usize);
        let ids = [IMG, 5, IMG, IMG, 9, 11, IMG];
        let embeds = numbered(seq, hidden, 0.0);
        let features = numbered(4, hidden, 1000.0).squeeze(0).unwrap();

        let merged = merge_image_features(&embeds, &ids, &features, IMG).unwrap();
        assert_eq!(merged.dims(), &[1, seq, hidden]);

        let got = rows(&merged);
        let text = rows(&embeds);
        let feat = features.to_vec2::<f32>().unwrap();
        let mut k = 0;
        for (i, &t) in ids.iter().enumerate() {
            let want = if t == IMG {
                k += 1;
                &feat[k - 1]
            } else {
                &text[i]
            };
            assert_eq!(
                got[i].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "row {i} (token {t}): got {:?}, want {:?}",
                got[i],
                want
            );
        }
        assert_eq!(k, 4);
    }

    /// A prompt that reserved a different number of image positions than the
    /// tower produced rows must be refused, in both directions: filling a
    /// prefix would leave `<image>`'s own embedding in the tail and the model
    /// would describe the wrong thing fluently.
    #[test]
    fn merge_refuses_a_count_mismatch_in_either_direction() {
        const IMG: u32 = 7;
        let embeds = numbered(4, 2, 0.0);
        let three = numbered(3, 2, 100.0).squeeze(0).unwrap();
        let one = numbered(1, 2, 100.0).squeeze(0).unwrap();
        assert!(merge_image_features(&embeds, &[IMG, IMG, 1, 2], &three, IMG).is_err());
        assert!(merge_image_features(&embeds, &[IMG, IMG, 1, 2], &one, IMG).is_err());
        assert!(merge_image_features(
            &embeds,
            &[IMG, IMG, 1, 2],
            &numbered(2, 2, 9.0).squeeze(0).unwrap(),
            IMG
        )
        .is_ok());
    }

    /// The merge is B=1; a batch must be refused rather than flattened, which
    /// would splice every sequence's features into the first one.
    #[test]
    fn merge_refuses_a_batch() {
        const IMG: u32 = 7;
        let embeds = Tensor::zeros((2, 3, 2), DType::F32, &Device::Cpu).unwrap();
        let feats = numbered(1, 2, 1.0).squeeze(0).unwrap();
        assert!(merge_image_features(&embeds, &[IMG, 1, 2], &feats, IMG).is_err());
    }

    /// The trait impls are reachable through their bounds. Like `lfm2.rs`'s
    /// keystone test (§3.1c), this proves the callable-through-the-trait property
    /// without a forward pass: the function only compiles if `VisionEncoder`
    /// satisfies `crate::nn::Model` and `crate::nn::VisionEncoder` with the
    /// associated types lined up. The numeric behaviour is the §5.12 gate's, run
    /// against the real checkpoint by `lfm2-vl --gate`.
    #[allow(dead_code)]
    fn trait_bounds_hold<E>()
    where
        E: crate::nn::VisionEncoder<Config = VisionEncoderConfig, Input = PreparedImage>,
    {
    }

    #[test]
    fn vision_encoder_satisfies_the_domain_trait() {
        trait_bounds_hold::<VisionEncoder>();
    }
}
