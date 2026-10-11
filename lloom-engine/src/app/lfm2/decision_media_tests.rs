//! Tiny actual decoder/source controls: 2x3 pixels, <1024 encoded bytes/image.
use super::*;
use base64::Engine as _;
fn bounds() -> ImageDecodeLimits {
    ImageDecodeLimits {
        max_encoded_bytes: 1024,
        max_width: 3,
        max_height: 2,
        max_pixels: 6,
        max_decoder_bytes: 4096,
    }
}
fn bytes(format: image::ImageFormat) -> Vec<u8> {
    let image = image::RgbImage::from_fn(3, 2, |x, y| image::Rgb([x as u8 * 40, y as u8 * 80, 7]));
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut std::io::Cursor::new(&mut bytes), format)
        .unwrap();
    assert!(bytes.len() < 1024);
    bytes
}
#[test]
fn actual_bounded_png_jpeg_match_legacy_and_refuse_each_explicit_limit() {
    for format in [image::ImageFormat::Png, image::ImageFormat::Jpeg] {
        let bytes = bytes(format);
        let legacy = DecodedImage::decode(&bytes).unwrap();
        let decoded = DecodedImage::decode_bounded(&bytes, bounds()).unwrap();
        assert_eq!((decoded.height, decoded.width), (2, 3));
        assert_eq!(decoded.rgb, legacy.rgb);
        assert_eq!(decoded.key, legacy.key);
        if format == image::ImageFormat::Png {
            assert_eq!(&decoded.rgb[..6], &[0, 0, 7, 40, 0, 7]);
        }
        for limit in [
            ImageDecodeLimits {
                max_encoded_bytes: bytes.len() - 1,
                ..bounds()
            },
            ImageDecodeLimits {
                max_width: 2,
                ..bounds()
            },
            ImageDecodeLimits {
                max_height: 1,
                ..bounds()
            },
            ImageDecodeLimits {
                max_pixels: 5,
                ..bounds()
            },
            ImageDecodeLimits {
                max_decoder_bytes: 17,
                ..bounds()
            },
        ] {
            assert!(
                DecodedImage::decode_bounded(&bytes, limit).is_err(),
                "bounded decode ignored an explicit limit"
            );
        }
    }
    assert!(bounds().validate().is_ok());
    assert!(ImageDecodeLimits {
        max_pixels: 0,
        ..bounds()
    }
    .validate()
    .is_err());
}
#[test]
fn actual_source_uses_content_not_mime_and_distinguishes_invalid_from_unsupported() {
    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes(image::ImageFormat::Png));
    let wrong_mime = format!("data:audio/wav;base64,{encoded}");
    let decoded = decode_source(MediaSource::DataUrl(&wrong_mime), bounds()).unwrap();
    let raw = decode_source(MediaSource::AudioBase64(&encoded), bounds()).unwrap();
    assert_eq!(decoded.key, raw.key);
    assert_eq!((decoded.height, decoded.width), (2, 3));
    let bad_url = "data:image/png,not-base64";
    assert!(matches!(
        decode_source(MediaSource::DataUrl(bad_url), bounds()),
        Err(SystemOneError::InvalidRequest(_))
    ));
    assert!(matches!(
        decode_source(MediaSource::AudioBase64("%%%"), bounds()),
        Err(SystemOneError::InvalidRequest(_))
    ));
    let not_image = base64::engine::general_purpose::STANDARD.encode(b"RIFF no image");
    assert!(matches!(
        decode_source(MediaSource::AudioBase64(&not_image), bounds()),
        Err(SystemOneError::Unsupported(_))
    ));
    let broken = base64::engine::general_purpose::STANDARD.encode(b"\x89PNG\r\n\x1a\n");
    assert!(matches!(
        decode_source(MediaSource::AudioBase64(&broken), bounds()),
        Err(SystemOneError::InvalidRequest(_))
    ));
    let too_long = base64::engine::general_purpose::STANDARD.encode([0u8; 1027]);
    assert!(
        matches!(decode_source(MediaSource::AudioBase64(&too_long),bounds()),Err(SystemOneError::InvalidRequest(message)) if message.contains("byte limit"))
    );
}
