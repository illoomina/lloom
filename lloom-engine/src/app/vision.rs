//! Image input for the served families: the request side (decoding an image,
//! the OpenAI content parts, the `<image>` expansion) and the composition that
//! puts encoded image rows into a model's own prefill (`RESTRUCTURING_DESIGN.md`
//! §5: `embed` -> merge -> `forward_embeds`).
//!
//! Nothing here runs on a request without an image, and nothing here runs per
//! token: [`ImageRows`] hands every forward that carries no image position to
//! the model's own `forward`, the path the text digests belong to.

use anyhow::{Context, Result};
use lloom::nn::{CausalLM, Model, VarBuilder};
use lloom_tensor::Tensor;
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// One decoded image: 8-bit interleaved RGB and a digest of it.
///
/// The digest is what a cached prefix is matched on besides its tokens: two
/// different images behind the same prompt text reserve the same `<image>`
/// ids, so a token-only match would resume from the wrong image's rows.
#[derive(Debug, Clone)]
pub struct DecodedImage {
    pub height: usize,
    pub width: usize,
    pub rgb: Vec<u8>,
    pub key: [u8; 32],
}

/// Explicit caller decoding bounds, separate from model/pass admission.
/// Decoder max_alloc is best-effort; strict dimensions/pixel checks do not
/// certify third-party decoder internal peak or host allocation enrollment.
#[derive(Clone, Copy, Debug)]
pub struct ImageDecodeLimits {
    pub max_encoded_bytes: usize,
    pub max_width: u32,
    pub max_height: u32,
    pub max_pixels: usize,
    pub max_decoder_bytes: u64,
}
impl ImageDecodeLimits {
    pub fn validate(self) -> Result<()> {
        if self.max_encoded_bytes == 0
            || self.max_width == 0
            || self.max_height == 0
            || self.max_pixels == 0
            || self.max_decoder_bytes == 0
        {
            anyhow::bail!("image decode limits must be finite and nonzero");
        }
        Ok(())
    }
}

impl DecodedImage {
    /// Decode PNG / JPEG bytes. Alpha is dropped, as `PIL.Image.convert("RGB")`
    /// and torchvision's `ImageReadMode.RGB` drop it; greyscale is replicated.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let img = image::load_from_memory(bytes).context("decoding the image")?;
        Ok(Self::from_dynamic(img))
    }

    /// PNG/JPEG content with supplied bounds before output-image decoding.
    /// Existing unbounded legacy decode/from_url APIs retain their contract.
    pub fn decode_bounded(bytes: &[u8], bounds: ImageDecodeLimits) -> Result<Self> {
        use image::ImageDecoder;
        bounds.validate()?;
        if bytes.len() > bounds.max_encoded_bytes {
            anyhow::bail!("encoded image exceeds supplied byte limit");
        }
        let format = image::guess_format(bytes).context("recognizing image content")?;
        if !matches!(format, image::ImageFormat::Png | image::ImageFormat::Jpeg) {
            anyhow::bail!("bounded image decoding accepts PNG/JPEG content");
        }
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(bounds.max_width);
        limits.max_image_height = Some(bounds.max_height);
        limits.max_alloc = Some(bounds.max_decoder_bytes);
        let mut reader = image::ImageReader::with_format(std::io::Cursor::new(bytes), format);
        reader.limits(limits.clone());
        let mut decoder = reader
            .into_decoder()
            .context("reading bounded image header")?;
        let (w, h) = decoder.dimensions();
        let pixels = (w as usize)
            .checked_mul(h as usize)
            .context("image pixel count overflow")?;
        if w == 0
            || h == 0
            || pixels > bounds.max_pixels
            || decoder.total_bytes() > bounds.max_decoder_bytes
        {
            anyhow::bail!("image dimensions/output exceed supplied decode limits");
        }
        // Same output debit as ImageReader::decode; max_alloc remains the
        // decoder library's non-strict bound, not an allocator entitlement.
        limits.reserve(decoder.total_bytes())?;
        decoder.set_limits(limits)?;
        let img = image::DynamicImage::from_decoder(decoder).context("decoding bounded image")?;
        Ok(Self::from_dynamic(img))
    }

    fn from_dynamic(img: image::DynamicImage) -> Self {
        let rgb = img.to_rgb8();
        let (w, h) = rgb.dimensions();
        Self::from_rgb8(h as usize, w as usize, rgb.into_raw())
    }

    pub fn from_rgb8(height: usize, width: usize, rgb: Vec<u8>) -> Self {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update((height as u64).to_le_bytes());
        hash.update((width as u64).to_le_bytes());
        hash.update(&rgb);
        Self {
            height,
            width,
            rgb,
            key: hash.finalize().into(),
        }
    }

    pub fn open(path: &std::path::Path) -> Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("reading the image {}", path.display()))?;
        Self::decode(&bytes).with_context(|| path.display().to_string())
    }

    /// An OpenAI `image_url.url`: a `data:image/...;base64,` URL. Fetching a
    /// remote image is not built, and a server that read `file://` paths would
    /// hand its filesystem to any client, so both are refused by name.
    pub fn from_url(url: &str) -> Result<Self> {
        let Some(rest) = url.strip_prefix("data:") else {
            anyhow::bail!(
                "image_url must be a data URL (`data:image/png;base64,...`); fetching {} is not \
                 built",
                url.split(':').next().unwrap_or("that scheme")
            );
        };
        let (meta, payload) = rest
            .split_once(',')
            .context("a data URL without a ',' before its payload")?;
        if !meta.ends_with(";base64") {
            anyhow::bail!("image data URLs must be base64 (`;base64,`), got `{meta}`");
        }
        use base64::Engine as _;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(payload.trim())
            .context("the image data URL is not valid base64")?;
        Self::decode(&bytes)
    }
}

/// The messages as the chat template reads them, and the images they carry in
/// order: every OpenAI `{"type": "image_url", ...}` part becomes the
/// template's `{"type": "image"}` (which renders `<image>`), and every `text`
/// part stays. String content is untouched, so a request with no image part
/// renders exactly as before.
pub fn split_image_parts(messages: &Value) -> Result<(Value, Vec<DecodedImage>)> {
    let Some(list) = messages.as_array() else {
        anyhow::bail!("messages must be an array");
    };
    let mut images = Vec::new();
    let mut out = Vec::with_capacity(list.len());
    for m in list {
        let Some(parts) = m.get("content").and_then(Value::as_array) else {
            out.push(m.clone());
            continue;
        };
        let mut rendered = Vec::with_capacity(parts.len());
        for p in parts {
            match p.get("type").and_then(Value::as_str) {
                Some("image_url") => {
                    let url = p
                        .get("image_url")
                        .and_then(|u| u.get("url").or(Some(u)))
                        .and_then(Value::as_str)
                        .context("an image_url part without image_url.url")?;
                    images.push(DecodedImage::from_url(url)?);
                    rendered.push(json!({ "type": "image" }));
                }
                Some("image") => anyhow::bail!(
                    "an `image` part carries no pixels; send `{{\"type\": \"image_url\", \
                     \"image_url\": {{\"url\": \"data:...\"}}}}`"
                ),
                _ => rendered.push(p.clone()),
            }
        }
        let mut m = m.clone();
        m["content"] = Value::Array(rendered);
        out.push(m);
    }
    Ok((Value::Array(out), images))
}

/// Replace the `n`-th `<image>` of a rendered prompt with `expansions[n]`, as
/// `Lfm2VlProcessor` does. The count must match: a typed `<image>` in a
/// message is indistinguishable from an attached image's once rendered, and
/// the reference refuses the same mismatch.
pub fn expand_image_tokens(rendered: &str, expansions: &[String]) -> Result<String> {
    const TOKEN: &str = "<image>";
    let found = rendered.matches(TOKEN).count();
    if found != expansions.len() {
        anyhow::bail!(
            "the prompt holds {found} `<image>` token(s) for {} attached image(s)",
            expansions.len()
        );
    }
    let mut out =
        String::with_capacity(rendered.len() + expansions.iter().map(String::len).sum::<usize>());
    let mut rest = rendered;
    for e in expansions {
        let at = rest.find(TOKEN).expect("counted above");
        out.push_str(&rest[..at]);
        out.push_str(e);
        rest = &rest[at + TOKEN.len()..];
    }
    out.push_str(rest);
    Ok(out)
}

/// A model whose prefill takes image rows at the image-token positions.
///
/// The session driving it notes each prompt chunk's ids before that chunk's
/// forward (`CausalLM::note_input_ids`) and passes the chunk's absolute
/// position to `forward`. A chunk holding image positions is embedded, the
/// rows the positions own are spliced in (`lfm2_vl::merge_image_features`, the
/// reference's `masked_scatter`), and the result runs through
/// `forward_embeds`. Every other forward — the text chunks and every decode
/// step — is the model's own `forward`, so the text path is untouched by
/// construction. A chunk is matched against the request's own prompt, so a
/// chunk the adapter did not expect is refused rather than spliced.
pub struct ImageRows<'m, M: CausalLM> {
    model: &'m M,
    prompt: Vec<u32>,
    image_token_id: u32,
    /// `[rows, hidden]`: the rows of image tokens `first_rank..`, in prompt
    /// order. Earlier ranks are in the cache already and were not encoded.
    features: Tensor,
    first_rank: usize,
    /// The ids of the chunk the next forward consumes, when it holds an image
    /// position; checked first, so a text forward takes no lock.
    pending: AtomicBool,
    chunk: Mutex<Option<Vec<u32>>>,
    spliced: AtomicUsize,
}

struct ImageChunk<'a> {
    slot: &'a Mutex<Option<Vec<u32>>>,
    ids: Vec<u32>,
}
impl Drop for ImageChunk<'_> {
    fn drop(&mut self) {
        self.ids.clear();
        *self.slot.lock() = Some(std::mem::take(&mut self.ids));
    }
}

impl<'m, M: CausalLM> ImageRows<'m, M> {
    pub fn new(
        model: &'m M,
        prompt: Vec<u32>,
        image_token_id: u32,
        features: Tensor,
        first_rank: usize,
    ) -> Self {
        Self {
            model,
            prompt,
            image_token_id,
            features,
            first_rank,
            pending: AtomicBool::new(false),
            chunk: Mutex::new(Some(Vec::new())),
            spliced: AtomicUsize::new(0),
        }
    }

    /// Reserve the existing adapter's prompt-bounded host chunk before any
    /// model work. Legacy new remains available with fallible first-note growth.
    pub fn try_new(
        model: &'m M,
        prompt: Vec<u32>,
        image_token_id: u32,
        features: Tensor,
        first_rank: usize,
    ) -> lloom_tensor::Result<Self> {
        let rows = Self::new(model, prompt, image_token_id, features, first_rank);
        {
            let mut slot = rows.chunk.lock();
            // The new, not-yet-published slot always holds its empty buffer.
            slot.as_mut()
                .expect("fresh image chunk slot")
                .try_reserve_exact(rows.prompt.len())
                .map_err(lloom_tensor::Error::wrap)?;
        }
        Ok(rows)
    }

    /// Image rows placed so far, over every forward.
    pub fn spliced(&self) -> usize {
        self.spliced.load(Ordering::Relaxed)
    }

    fn splice(&self, tokens: &Tensor, pos: usize, chunk: &[u32]) -> lloom_tensor::Result<Tensor> {
        let end = pos + chunk.len();
        if end > self.prompt.len() || self.prompt[pos..end] != *chunk {
            lloom_tensor::bail!(
                "image prefill: the chunk at position {pos} ({} ids) is not the request's prompt \
                 there",
                chunk.len()
            );
        }
        let rank = self.prompt[..pos]
            .iter()
            .filter(|&&t| t == self.image_token_id)
            .count();
        let n = chunk.iter().filter(|&&t| t == self.image_token_id).count();
        let available = self.features.dim(0)?;
        if rank < self.first_rank || rank + n > self.first_rank + available {
            lloom_tensor::bail!(
                "image prefill: the chunk at position {pos} needs image rows {rank}..{} and rows \
                 {}..{} were encoded",
                rank + n,
                self.first_rank,
                self.first_rank + available
            );
        }
        let rows = self.features.narrow(0, rank - self.first_rank, n)?;
        let embeds = self.model.embed(tokens)?;
        let merged = lloom::models::lfm2_vl::merge_image_features(
            &embeds,
            chunk,
            &rows,
            self.image_token_id,
        )?;
        self.spliced.fetch_add(n, Ordering::Relaxed);
        Ok(merged)
    }
}

impl<M: CausalLM> Model for ImageRows<'_, M> {
    type Config = M::Config;

    fn from_weights(_vb: VarBuilder, _cfg: &Self::Config) -> lloom_tensor::Result<Self> {
        lloom_tensor::bail!("ImageRows wraps a loaded model; it is not built from weights")
    }
}

impl<M: CausalLM> CausalLM for ImageRows<'_, M> {
    type Cache = M::Cache;

    fn new_cache(
        &self,
        cfg: &Self::Config,
        dev: &lloom_tensor::Device,
    ) -> lloom_tensor::Result<Self::Cache> {
        self.model.new_cache(cfg, dev)
    }

    fn embed(&self, tokens: &Tensor) -> lloom_tensor::Result<Tensor> {
        self.model.embed(tokens)
    }

    fn forward_embeds(
        &self,
        embeds: &Tensor,
        pos: usize,
        cache: &mut Self::Cache,
    ) -> lloom_tensor::Result<Tensor> {
        self.model.forward_embeds(embeds, pos, cache)
    }

    fn forward(
        &self,
        tokens: &Tensor,
        pos: usize,
        cache: &mut Self::Cache,
    ) -> lloom_tensor::Result<Tensor> {
        if self.pending.swap(false, Ordering::Relaxed) {
            let ids = self.chunk.lock().take().ok_or_else(|| {
                lloom_tensor::Error::msg("image prefill chunk is already in flight")
            })?;
            let chunk = ImageChunk {
                slot: &self.chunk,
                ids,
            };
            let merged = self.splice(tokens, pos, &chunk.ids)?;
            return self.model.forward_embeds(&merged, pos, cache);
        }
        self.model.forward(tokens, pos, cache)
    }

    fn prepare_pass(&self, cache: &mut Self::Cache, seq_len: usize) -> lloom_tensor::Result<()> {
        self.model.prepare_pass(cache, seq_len)
    }

    fn checkpoint_at(&self, cache: &mut Self::Cache, pos: usize) -> lloom_tensor::Result<()> {
        self.model.checkpoint_at(cache, pos)
    }

    fn open_checkpoint_request(
        &self,
        cache: &mut Self::Cache,
        policy: lloom::nn::CheckpointPolicy,
        end: usize,
    ) -> lloom_tensor::Result<()> {
        self.model.open_checkpoint_request(cache, policy, end)
    }

    fn close_checkpoint_request(&self, cache: &mut Self::Cache) -> lloom_tensor::Result<()> {
        self.model.close_checkpoint_request(cache)
    }

    fn note_input_ids(&self, cache: &mut Self::Cache, ids: &[u32]) -> lloom_tensor::Result<()> {
        if ids.contains(&self.image_token_id) {
            if ids.len() > self.prompt.len() {
                lloom_tensor::bail!("image input chunk exceeds its stored prompt bound");
            }
            let mut slot = self.chunk.lock();
            let chunk = slot.as_mut().ok_or_else(|| {
                lloom_tensor::Error::msg("image input note while chunk is in flight")
            })?;
            self.pending.store(false, Ordering::Relaxed);
            chunk.clear();
            if ids.len() > chunk.capacity() {
                chunk
                    .try_reserve_exact(ids.len())
                    .map_err(lloom_tensor::Error::wrap)?;
            }
            chunk.extend_from_slice(ids);
            self.pending.store(true, Ordering::Relaxed);
        } else {
            self.pending.store(false, Ordering::Relaxed);
        }
        self.model.note_input_ids(cache, ids)
    }

    fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }
}

/// Dispatches the Metal recorder saw over one window, per pipeline label
/// (`CANDLE_METAL_PROFILE=1`; `lfm2-vl --census`'s instrument).
#[derive(Debug, Clone, Default)]
pub struct DispatchWindow {
    pub dispatches: u64,
    pub labels: std::collections::BTreeMap<String, u64>,
}

/// Close the recorder's current window and open the next: synchronise (every
/// completion handler has run), snapshot, reset. `None` when the recorder is
/// off, so a request without the census pays one relaxed load.
pub fn take_window(device: &lloom_tensor::Device) -> Result<Option<DispatchWindow>> {
    #[cfg(feature = "metal")]
    {
        use lloom_tensor::kernels::metal::profile;
        if profile::recorder_enabled() {
            device.synchronize()?;
            let s = profile::snapshot();
            profile::reset();
            return Ok(Some(DispatchWindow {
                dispatches: s.dispatches,
                labels: s.by_label.into_iter().collect(),
            }));
        }
    }
    let _ = device;
    Ok(None)
}

/// The three windows of one image request: the encode, the prefill, the
/// decode steps. The tower's signature is the labels the encode dispatched and
/// the prefill did not; encode-once holds iff the decode window carries none.
#[derive(Debug, Clone)]
pub struct ImageCensus {
    pub encode: DispatchWindow,
    pub prefill: DispatchWindow,
    pub decode: DispatchWindow,
    pub decode_steps: usize,
}

impl ImageCensus {
    pub fn signature(&self) -> std::collections::BTreeMap<String, u64> {
        self.encode
            .labels
            .iter()
            .filter(|(l, _)| !self.prefill.labels.contains_key(*l))
            .map(|(l, n)| (l.clone(), *n))
            .collect()
    }

    /// The tower labels the decode window dispatched: empty is encode-once.
    pub fn leaked(&self) -> std::collections::BTreeMap<String, u64> {
        let sig = self.signature();
        self.decode
            .labels
            .iter()
            .filter(|(l, _)| sig.contains_key(*l))
            .map(|(l, n)| (l.clone(), *n))
            .collect()
    }

    /// A census that cannot tell the tower from the language model says so:
    /// an empty signature passes the leak check on anything.
    pub fn verdict(&self) -> &'static str {
        if self.encode.dispatches == 0 {
            "no-encode (every image row was already cached)"
        } else if self.signature().is_empty() {
            "BLIND (the encode dispatched no kernel the prefill did not)"
        } else if self.leaked().is_empty() {
            "encode-once holds"
        } else {
            "VIOLATED (the decode window dispatched tower kernels)"
        }
    }

    pub fn json(&self) -> Value {
        let sig = self.signature();
        json!({
            "encode_dispatches": self.encode.dispatches,
            "prefill_dispatches": self.prefill.dispatches,
            "decode_dispatches": self.decode.dispatches,
            "decode_steps": self.decode_steps,
            "signature_labels": sig.len(),
            "signature_dispatches": sig.values().sum::<u64>(),
            "signature": sig,
            "leaked": self.leaked(),
            "verdict": self.verdict(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "metal")]
    #[test]
    fn image_census_preserves_scalar_counts_without_inventing_labels() -> Result<()> {
        use lloom_tensor::kernels::metal::profile;
        let mut tuning = lloom_tensor::tuning::Tuning::default();
        tuning.instrument.profile = true;
        lloom_tensor::tuning::install(tuning).unwrap();
        profile::reset();
        let label = std::sync::Arc::<str>::from("image-census-control");
        profile::record_dispatch(Some(&label), 0);
        assert_eq!(profile::snapshot().dispatches, 1);
        let window = take_window(&lloom_tensor::Device::Cpu)?;
        if lloom_tensor::tuning::PROFILE_INSTRUMENTS {
            let window = window.expect("the active recorder observed a window");
            assert_eq!(window.dispatches, 1);
            assert_eq!(window.labels.get("image-census-control"), Some(&1));
        } else {
            assert!(window.is_none(), "scalar-only data is not a label census");
            assert_eq!(
                profile::snapshot().dispatches,
                1,
                "absence must not reset scalar counters"
            );
        }
        Ok(())
    }

    #[test]
    fn disabled_image_recording_is_unavailable_without_device_work() -> Result<()> {
        lloom_tensor::tuning::install(lloom_tensor::tuning::Tuning::default()).unwrap();
        assert!(take_window(&lloom_tensor::Device::Cpu)?.is_none());
        Ok(())
    }

    #[test]
    fn the_nth_image_token_takes_the_nth_expansion() {
        let out = expand_image_tokens("a<image>b<image>c", &["X".into(), "YY".into()]).unwrap();
        assert_eq!(out, "aXbYYc");
        let err = expand_image_tokens("a<image>b", &[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("1 `<image>` token(s) for 0"), "{err}");
        let err = expand_image_tokens("ab", &["X".into()])
            .unwrap_err()
            .to_string();
        assert!(err.contains("0 `<image>` token(s) for 1"), "{err}");
    }

    fn png_data_url(h: u32, w: u32) -> String {
        use base64::Engine as _;
        let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb([x as u8, y as u8, 7]));
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .unwrap();
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        )
    }

    /// OpenAI parts become the template's parts, the images come out in order
    /// with their pixels, and string content passes through untouched.
    #[test]
    fn image_url_parts_become_template_image_parts_in_order() {
        let messages = json!([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": png_data_url(3, 5)}},
                {"type": "text", "text": "and"},
                {"type": "image_url", "image_url": {"url": png_data_url(2, 4)}},
            ]},
        ]);
        let (rendered, images) = split_image_parts(&messages).unwrap();
        assert_eq!(rendered[0], messages[0]);
        assert_eq!(
            rendered[1]["content"],
            json!([{"type": "image"}, {"type": "text", "text": "and"}, {"type": "image"}])
        );
        assert_eq!(
            images
                .iter()
                .map(|i| (i.height, i.width))
                .collect::<Vec<_>>(),
            vec![(3, 5), (2, 4)]
        );
        assert_eq!(&images[0].rgb[..6], &[0, 0, 7, 1, 0, 7]);
        assert_ne!(images[0].key, images[1].key);
        let err = split_image_parts(&json!([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]}]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("must be a data URL"), "{err}");
    }
}

#[cfg(test)]
mod image_rows_tests;
