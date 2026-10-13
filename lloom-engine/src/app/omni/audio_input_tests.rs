//! WAV built here, FLAC from `fixtures/pattern-16k-mono.flac` (`flac 1.5.0
//! --best` of the same 1600-sample pattern): each decodes to exactly its
//! samples; a clip the model cannot read is refused by name.
use super::*;

/// 0.1 s at 16 kHz touching every byte value, so a dropped, reordered or
/// rescaled sample shows.
fn pattern() -> Vec<i16> {
    (0..1600u64)
        .map(|i| ((i * 2654435761) % 65536) as i64 - 32768)
        .map(|s| s as i16)
        .collect()
}

fn wav(samples: &[i16], rate: u32, channels: u16) -> Vec<u8> {
    let data = samples.len() * 2;
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2 * u32::from(channels)).to_le_bytes());
    out.extend_from_slice(&(2 * channels).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data as u32).to_le_bytes());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

fn i16s(samples: Samples) -> Vec<i16> {
    match samples {
        Samples::I16(s) => s,
        Samples::F32(_) => panic!("16-bit PCM decoded as float"),
    }
}

#[test]
fn wav_and_flac_decode_to_their_exact_samples() {
    let want = pattern();
    let bytes = wav(&want, RATE, 1);
    assert!(is_audio(&bytes));
    assert_eq!(i16s(decode(bytes).unwrap()), want, "WAV");
    let flac = include_bytes!("fixtures/pattern-16k-mono.flac").to_vec();
    assert!(is_audio(&flac));
    assert_eq!(i16s(decode(flac).unwrap()), want, "FLAC");
}

#[test]
fn a_clip_past_thirty_seconds_is_cut_there() {
    let long: Vec<i16> = (0..31 * RATE as usize).map(|i| (i % 1000) as i16).collect();
    let got = i16s(decode(wav(&long, RATE, 1)).unwrap());
    assert_eq!(got.len(), 30 * RATE as usize);
    assert_eq!(got[..], long[..got.len()]);
}

#[test]
fn a_clip_the_model_cannot_read_is_refused_by_name() {
    let rate = decode(wav(&pattern(), 44_100, 1)).err().unwrap();
    assert!(
        rate.contains("16000 Hz mono") && rate.contains("44100 Hz"),
        "{rate}"
    );
    let stereo = decode(wav(&pattern(), RATE, 2)).err().unwrap();
    assert!(stereo.contains("2 channels"), "{stereo}");
    assert!(!is_audio(b"\x89PNG\r\n\x1a\n"));
    assert!(decode(b"not audio at all".to_vec()).is_err());
}
