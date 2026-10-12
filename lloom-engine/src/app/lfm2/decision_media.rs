//! Full raw image decisions on the incumbent owner inside its admitted pass.
//! Decoder/temporary native custody is not established by request exclusion.
use super::*;
use crate::app::causal_loaded::CausalPolicy;
use crate::app::vision::{expand_image_tokens, ImageDecodeLimits};
use crate::decision::{
    parse_systemone_state, MediaSource, OrderedJson, SystemOneError, SystemOneMedia,
    SystemOneResponse,
};
use lloom::nn::CausalLM;

fn invalid(error: impl std::fmt::Display) -> SystemOneError {
    SystemOneError::InvalidRequest(error.to_string())
}
fn execution(error: impl std::fmt::Display) -> SystemOneError {
    SystemOneError::Execution(error.to_string())
}
fn decode_source(
    source: MediaSource<'_>,
    bounds: ImageDecodeLimits,
) -> std::result::Result<DecodedImage, SystemOneError> {
    use base64::Engine as _;
    let text = match source {
        MediaSource::DataUrl(text) | MediaSource::AudioBase64(text) => text,
    };
    let payload = if let Some(rest) = text.strip_prefix("data:") {
        let (meta, payload) = rest
            .split_once(',')
            .ok_or_else(|| invalid("media data URL has no payload"))?;
        if !meta.ends_with(";base64") {
            return Err(invalid("media data URL must use base64"));
        }
        payload.trim()
    } else {
        text.trim()
    };
    // STANDARD's allocation estimate includes at most two padding bytes. Bound
    // it before allocating, then enforce the actual caller byte limit below.
    let estimate = base64::decoded_len_estimate(payload.len());
    let allocation_limit = bounds
        .max_encoded_bytes
        .checked_add(2)
        .ok_or_else(|| invalid("encoded limit overflow"))?;
    if estimate > allocation_limit {
        return Err(invalid("encoded image exceeds supplied byte limit"));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(invalid)?;
    if bytes.len() > bounds.max_encoded_bytes {
        return Err(invalid("encoded image exceeds supplied byte limit"));
    }
    if !matches!(
        image::guess_format(&bytes),
        Ok(image::ImageFormat::Png | image::ImageFormat::Jpeg)
    ) {
        return Err(SystemOneError::Unsupported(
            "full LFM raw media currently accepts PNG/JPEG content; no audio encoder installed"
                .into(),
        ));
    }
    DecodedImage::decode_bounded(&bytes, bounds).map_err(invalid)
}

/// What a family's image decisions read: its processor (layout, expansion,
/// patches), its VL config (the `<image>` id, the special tokens) and the
/// dtype its tower runs in.
#[derive(Clone, Copy)]
pub(crate) struct ImageVision<'a> {
    pub processor: &'a lloom::models::lfm2_vl::ImageProcessor,
    pub config: &'a lloom::models::lfm2_vl::Lfm2VlConfig,
    pub dtype: DType,
}

/// A raw `/v1/systemone` body made ready for the model: a text request, or
/// one whose shared trunk holds every decoded image's reserved positions.
pub(crate) enum Prepared {
    Text(crate::decision::CausalRequest),
    Images {
        request: crate::decision::CausalRequest,
        images: Vec<DecodedImage>,
        positions: usize,
    },
}

/// Everything before the model: the ordered state and questions, the images
/// decoded under `bounds`, each `<image>` expanded by `vision`'s layout, the
/// certified render tokenized whole, and its image positions checked to sit
/// in the shared trunk. `vision` is `None` for a family without a tower,
/// whose image request is refused with `no_vision`.
pub(crate) fn prepare_images(
    body: &OrderedJson,
    policy: &CausalPolicy,
    capacity: usize,
    vocab: usize,
    vision: Option<ImageVision<'_>>,
    bounds: ImageDecodeLimits,
    no_vision: impl FnOnce() -> String,
    encode: impl Fn(&str) -> anyhow::Result<Vec<u32>>,
) -> std::result::Result<Prepared, SystemOneError> {
    let parsed = parse_systemone_state(body, policy.state_nullability).map_err(invalid)?;
    let questions = policy.parse_questions(body)?;
    if parsed.media.is_empty() {
        return Ok(Prepared::Text(policy.prepare_rendered(
            parsed.state.as_ref(),
            questions,
            capacity,
            SystemOneMedia::Text,
            |text| encode(text).map_err(lloom_tensor::Error::wrap),
            |text| encode(text).map_err(lloom_tensor::Error::wrap),
        )?));
    }
    let vision = vision.ok_or_else(|| SystemOneError::Unsupported(no_vision()))?;
    bounds.validate().map_err(invalid)?;
    let images = parsed
        .media
        .iter()
        .map(|source| decode_source(*source, bounds))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let layouts = images
        .iter()
        .map(|im| vision.processor.layout(im.height, im.width))
        .collect::<Vec<_>>();
    let positions = layouts
        .iter()
        .try_fold(0usize, |sum, layout| {
            sum.checked_add(vision.processor.image_tokens(layout))
        })
        .ok_or_else(|| invalid("image position count overflow"))?;
    let expansions = layouts
        .iter()
        .map(|layout| {
            vision
                .processor
                .placeholder(layout, vision.config.use_image_special_tokens)
        })
        .collect::<Vec<_>>();
    let request = policy.prepare_rendered(
        parsed.state.as_ref(),
        questions,
        capacity,
        SystemOneMedia::Images {
            count: images.len(),
            marker: "<image>",
        },
        |text| encode(text).map_err(lloom_tensor::Error::wrap),
        |text| {
            let expanded =
                expand_image_tokens(text, &expansions).map_err(lloom_tensor::Error::wrap)?;
            encode(&expanded).map_err(lloom_tensor::Error::wrap)
        },
    )?;
    super::super::common::check_context_capacity(capacity, request.max_prompt_tokens(), 0, 0)
        .map_err(invalid)?;
    request
        .validate_shared_token_positions(vision.config.image_token_id, positions)
        .map_err(invalid)?;
    request.validate_tokens(vocab).map_err(invalid)?;
    Ok(Prepared::Images {
        request,
        images,
        positions,
    })
}

/// The model half of an image request, inside the caller's request lease:
/// every image encoded by `encoder` in order, its rows spliced into the
/// trunk's reserved positions, and the questions read as branches of it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decide_images<M: CausalLM>(
    model: &M,
    request: &crate::decision::CausalRequest,
    images: &[DecodedImage],
    positions: usize,
    vision: ImageVision<'_>,
    encoder: &lloom::models::lfm2_vl::VisionEncoder,
    device: &Device,
    options: &lloom::nn::DecodeOptions,
    make_cache: impl FnOnce(&M, &Device) -> lloom_tensor::Result<M::Cache>,
) -> std::result::Result<SystemOneResponse, SystemOneError>
where
    M::Cache: lloom::nn::branch::RetainState,
{
    let mut parts = Vec::with_capacity(images.len());
    for image in images {
        let prepared = vision
            .processor
            .prepare_rgb8(&image.rgb, image.height, image.width, device, vision.dtype)
            .map_err(execution)?;
        parts.push(encoder.encode_image(&prepared).map_err(execution)?);
    }
    let features = if parts.len() == 1 {
        parts.pop().expect("one encoded image")
    } else {
        Tensor::cat(&parts, 0).map_err(execution)?
    };
    if features.dim(0).map_err(execution)? != positions {
        return Err(execution(
            "encoded image feature count differs from reserved positions",
        ));
    }
    let rows = ImageRows::try_new(
        model,
        request.trunk_tokens().to_vec(),
        vision.config.image_token_id,
        features,
        0,
    )
    .map_err(execution)?;
    let mut readback = lloom::generation::Blocking::new(&NEVER_EOS, device).map_err(execution)?;
    let response = request
        .execute_with_cache_factory(&rows, &mut readback, options, |_rows, device| {
            make_cache(model, device)
        })
        .map_err(execution)?;
    if rows.spliced() != positions {
        return Err(execution(
            "causal image prefill did not consume every encoded row",
        ));
    }
    Ok(response)
}

impl<P: Lfm2Shape> Engine<P> {
    pub(super) fn systemone_images_in_scope(
        &self,
        body: &OrderedJson,
        policy: &CausalPolicy,
        bounds: ImageDecodeLimits,
    ) -> std::result::Result<SystemOneResponse, SystemOneError> {
        let vision = self.vision.as_ref().map(|vision| ImageVision {
            processor: &vision.processor,
            config: &vision.config,
            dtype: vision.dtype,
        });
        let prepared = prepare_images(
            body,
            policy,
            self.kv_capacity,
            self.core.model.vocab_size(),
            vision,
            bounds,
            || self.no_tower().to_string(),
            |text| self.encode(text, false),
        )?;
        let (request, images, positions) = match prepared {
            Prepared::Text(request) => return self.decide_in_scope(&request).map_err(execution),
            Prepared::Images {
                request,
                images,
                positions,
            } => (request, images, positions),
        };
        let vision_state = self
            .vision
            .as_ref()
            .expect("an image request was prepared against this tower");
        let mut lease = self.checkout_single().map_err(execution)?;
        lease.start_external_work();
        let options = lease.extra().decode.options(0, ReadbackArm::Blocking, 0);
        let device = &self.core.device;
        let encoder = vision_state.encoder(device).map_err(execution)?;
        let response = decide_images(
            &self.core.model,
            &request,
            &images,
            positions,
            vision.expect("checked above"),
            &encoder,
            device,
            &options,
            |_model, device| {
                self.private_cache
                    .create::<P>(&self.config, self.dtype, device)
                    .map_err(|error| lloom_tensor::Error::Msg(format!("{error:#}")))
            },
        )?;
        lease.mark_complete();
        Ok(response)
    }
}

#[cfg(test)]
#[path = "decision_media_tests.rs"]
mod tests;
