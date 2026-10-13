//! A request's audio clip as the samples d1-omni reads: WAV or FLAC, 16 kHz
//! mono, as the reference takes it (`audio.py` `waveform` does not resample).
//! 16-bit PCM stays integer, so it is scaled by the frontend's own 1/32768.
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// The sample rate the model was trained at.
pub(crate) const RATE: u32 = 16_000;
/// The frontend reads at most 30 s; decoding stops there.
const MAX_SAMPLES: usize = 30 * RATE as usize;

pub(crate) enum Samples {
    I16(Vec<i16>),
    F32(Vec<f32>),
}

/// Whether `bytes` start like a container this decoder reads.
pub(crate) fn is_audio(bytes: &[u8]) -> bool {
    bytes.starts_with(b"fLaC") || (bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE"))
}

/// Decode `bytes` (WAV or FLAC) to at most 30 s of samples, refusing a clip
/// that is not 16 kHz mono by name.
pub(crate) fn decode(bytes: Vec<u8>) -> Result<Samples, String> {
    let source = MediaSourceStream::new(Box::new(std::io::Cursor::new(bytes)), Default::default());
    let mut format = symphonia::default::get_probe()
        .probe(
            &Hint::new(),
            source,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| format!("audio is not a WAV or FLAC file: {e}"))?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or("the audio file has no audio track")?;
    let id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or("the audio track has no codec parameters")?
        .clone();
    if params.sample_rate != Some(RATE) {
        return Err(format!(
            "audio must be {RATE} Hz mono, as the model reads it; this clip is {} Hz",
            params
                .sample_rate
                .map_or("an unknown rate".into(), |r| r.to_string())
        ));
    }
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(|e| format!("unsupported audio codec: {e}"))?;
    let mut samples: Option<Samples> = None;
    let mut frames: Vec<f32> = Vec::new();
    while let Some(packet) = format
        .next_packet()
        .map_err(|e| format!("reading the audio file: {e}"))?
    {
        if packet.track_id != id {
            continue;
        }
        let decoded = decoder
            .decode(&packet)
            .map_err(|e| format!("decoding the audio file: {e}"))?;
        if decoded.spec().channels().count() != 1 {
            return Err(format!(
                "audio must be {RATE} Hz mono, as the model reads it; this clip has {} channels",
                decoded.spec().channels().count()
            ));
        }
        use symphonia::core::audio::GenericAudioBufferRef as Buffer;
        // FLAC decodes into left-justified i32: 16-bit content is still
        // 16-bit samples, which the i16 copy recovers exactly (`>> 16`).
        let sixteen = params.bits_per_sample.is_some_and(|bits| bits <= 16);
        let integer = match decoded {
            Buffer::S16(_) => true,
            Buffer::S24(_) | Buffer::S32(_) => sixteen,
            _ => false,
        };
        let out = samples.get_or_insert_with(|| {
            if integer {
                Samples::I16(Vec::new())
            } else {
                Samples::F32(Vec::new())
            }
        });
        let full = match out {
            Samples::I16(out) => {
                let mut i16s: Vec<i16> = Vec::new();
                decoded.copy_to_vec_interleaved(&mut i16s);
                out.extend(i16s.into_iter().take(MAX_SAMPLES - out.len()));
                out.len() == MAX_SAMPLES
            }
            Samples::F32(out) => {
                frames.clear();
                decoded.copy_to_vec_interleaved(&mut frames);
                out.extend(frames.iter().take(MAX_SAMPLES - out.len()));
                out.len() == MAX_SAMPLES
            }
        };
        if full {
            break;
        }
    }
    samples.ok_or_else(|| "the audio file holds no samples".to_string())
}

#[cfg(test)]
#[path = "audio_input_tests.rs"]
mod tests;
