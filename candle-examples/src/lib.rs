pub mod audio;
pub mod bs1770;
pub mod chat_template;
pub mod coco_classes;
pub mod imagenet;
/// Model setup shared by the LFM2 harnesses (`DESIGN.md` §14.5 step 2).
pub mod lfm2_setup;
pub mod token_output_stream;
pub mod wav;
use candle::utils::{cuda_is_available, metal_is_available};
use candle::{Device, Result, Tensor};

/// Cold requirement for examples whose explicitly requested profiling arm reports
/// GPU intervals and per-label census. Mixed-purpose engine scalar profiling does
/// not use this helper. Creates no device and opens no checkpoint.
pub fn require_profile_recorder(example: &str) -> Result<()> {
    let t = candle::tuning::try_tuning().map_err(candle::Error::wrap)?;
    if t.instrument.profile {
        candle::tuning::require_profile_recorder().map_err(|error| {
            candle::Error::msg(format!("{error}; {example}'s requested detailed profiling arm requires cargo run --manifest-path candle/Cargo.toml --profile release-with-debug -p candle-examples --features metal,profile --example {example} -- <original arguments>"))
        })?;
    }
    Ok(())
}

/// State a gate harness's contract before resolving its diagnostic request.
/// The order matters: resolving tuning first would make `pin_contract` return
/// AlreadyRead. An explicit LLOOM_PROFILE still overrides the requested pin.
/// Creates no device and opens no checkpoint.
pub fn pin_profiled_contract(
    example: &str,
    wanted: candle::tuning::Contract,
) -> Result<candle::tuning::Contract> {
    let pinned = candle::tuning::pin_contract(wanted).map_err(candle::Error::Msg)?;
    require_profile_recorder(example)?;
    Ok(pinned)
}

#[cfg(test)]
mod diagnostic_profile_setup_tests {
    use super::*;
    fn pin_with_env(profile: Option<&str>, instrument: Option<&str>) -> candle::tuning::Contract {
        candle::isolation::claim("example contract and diagnostic setup");
        for (key, value) in [
            ("LLOOM_PROFILE", profile),
            ("CANDLE_METAL_PROFILE", instrument),
        ] {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
        let pinned =
            pin_profiled_contract("qwen4exp-chat", candle::tuning::Contract::Deterministic)
                .expect("contract pin must precede diagnostic tuning resolution (not AlreadyRead)");
        assert_eq!(
            candle::tuning::gate_contract("host ordering control").unwrap(),
            pinned
        );
        pinned
    }
    #[test]
    fn default_contract_pin_precedes_diagnostic_resolution_without_device() {
        assert_eq!(
            pin_with_env(None, None),
            candle::tuning::Contract::Deterministic
        );
    }
    #[test]
    fn explicit_profile_off_keeps_deterministic_contract_without_device() {
        assert_eq!(
            pin_with_env(None, Some("0")),
            candle::tuning::Contract::Deterministic
        );
    }
    #[test]
    fn named_contract_override_survives_diagnostic_resolution_without_device() {
        assert_eq!(
            pin_with_env(Some("fast"), Some("0")),
            candle::tuning::Contract::Fast
        );
    }
    #[test]
    fn default_profile_off_preserves_example_setup() {
        candle::tuning::install(candle::tuning::Tuning::default()).unwrap();
        require_profile_recorder("qwen4exp-chat").unwrap();
    }
    #[test]
    fn explicit_profile_off_preserves_example_setup() {
        candle::isolation::claim("example detailed profile setup");
        std::env::set_var("CANDLE_METAL_PROFILE", "0");
        require_profile_recorder("lfm2-decode-profile").unwrap();
    }
    #[cfg(not(feature = "metal"))]
    #[test]
    fn missing_metal_backend_refuses_requested_heavy_without_device_work() {
        let mut t = candle::tuning::Tuning::default();
        t.instrument.profile = true;
        candle::tuning::install(t).unwrap();
        let error = require_profile_recorder("qwen4exp-chat").unwrap_err();
        let message = error.to_string();
        assert!(message.contains("diagnostics.gpu_profile.enabled"));
        assert!(message.contains("metal instrumentation is not compiled"));
        assert!(message.contains("--features metal,profile --example qwen4exp-chat"));
    }
    #[test]
    fn requested_detail_refuses_absent_or_preserves_real_recorder_data() {
        if let Ok(expected) = std::env::var("LLOOM_TEST_EXPECT_PROFILE_CAPABILITY") {
            let expected = match expected.as_str() {
                "available" => true,
                "absent" => false,
                other => panic!("unknown test capability expectation: {other}"),
            };
            assert_eq!(
                candle::tuning::PROFILE_INSTRUMENTS,
                expected,
                "unexpected transitive diagnostic feature union"
            );
            println!(
                "profile capability actual={} expected={expected}",
                candle::tuning::PROFILE_INSTRUMENTS
            );
        }
        let mut t = candle::tuning::Tuning::default();
        t.instrument.profile = true;
        candle::tuning::install(t).unwrap();
        let result = require_profile_recorder("lfm2-dspark-verify");
        if !cfg!(feature = "metal") || !candle::tuning::PROFILE_INSTRUMENTS {
            let message = result.unwrap_err().to_string();
            assert!(message.contains("diagnostics.gpu_profile.enabled"));
            let capability = if cfg!(feature = "metal") {
                "profile"
            } else {
                "metal"
            };
            assert!(message.contains(&format!("{capability} instrumentation is not compiled")));
            assert!(message.contains("--profile release-with-debug -p candle-examples --features metal,profile --example lfm2-dspark-verify"));
        } else {
            result.unwrap();
            #[cfg(feature = "metal")]
            {
                use candle::kernels::metal::profile;
                profile::reset();
                let label = std::sync::Arc::<str>::from("host-example-positive");
                profile::record_dispatch(Some(&label), 1);
                profile::record_command_buffer(1.0, 3.0, 0.5, 1);
                let s = profile::snapshot();
                assert_eq!(s.dispatches, 1);
                assert_eq!(s.by_label, vec![("host-example-positive".into(), 1)]);
                assert_eq!(s.gpu_busy_union_s, 2.0);
            }
        }
    }
}

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

pub fn load_image<P: AsRef<std::path::Path>>(
    p: P,
    resize_longest: Option<usize>,
) -> Result<(Tensor, usize, usize)> {
    let img = image::ImageReader::open(p)?
        .decode()
        .map_err(candle::Error::wrap)?;
    let (initial_h, initial_w) = (img.height() as usize, img.width() as usize);
    let img = match resize_longest {
        None => img,
        Some(resize_longest) => {
            let (height, width) = (img.height(), img.width());
            let resize_longest = resize_longest as u32;
            let (height, width) = if height < width {
                let h = (resize_longest * height) / width;
                (h, resize_longest)
            } else {
                let w = (resize_longest * width) / height;
                (resize_longest, w)
            };
            img.resize_exact(width, height, image::imageops::FilterType::CatmullRom)
        }
    };
    let (height, width) = (img.height() as usize, img.width() as usize);
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (height, width, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    Ok((data, initial_h, initial_w))
}

pub fn load_image_and_resize<P: AsRef<std::path::Path>>(
    p: P,
    width: usize,
    height: usize,
) -> Result<Tensor> {
    let img = image::ImageReader::open(p)?
        .decode()
        .map_err(candle::Error::wrap)?
        .resize_to_fill(
            width as u32,
            height as u32,
            image::imageops::FilterType::Triangle,
        );
    let img = img.to_rgb8();
    let data = img.into_raw();
    Tensor::from_vec(data, (width, height, 3), &Device::Cpu)?.permute((2, 0, 1))
}

/// Saves an image to disk using the image crate, this expects an input with shape
/// (c, height, width).
pub fn save_image<P: AsRef<std::path::Path>>(img: &Tensor, p: P) -> Result<()> {
    let p = p.as_ref();
    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
        candle::bail!("save_image expects an input of shape (3, height, width)")
    }
    let img = img.permute((1, 2, 0))?.flatten_all()?;
    let pixels = img.to_vec1::<u8>()?;
    let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
            Some(image) => image,
            None => candle::bail!("error saving image {p:?}"),
        };
    image.save(p).map_err(candle::Error::wrap)?;
    Ok(())
}

pub fn save_image_resize<P: AsRef<std::path::Path>>(
    img: &Tensor,
    p: P,
    h: usize,
    w: usize,
) -> Result<()> {
    let p = p.as_ref();
    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
        candle::bail!("save_image expects an input of shape (3, height, width)")
    }
    let img = img.permute((1, 2, 0))?.flatten_all()?;
    let pixels = img.to_vec1::<u8>()?;
    let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
            Some(image) => image,
            None => candle::bail!("error saving image {p:?}"),
        };
    let image = image::DynamicImage::from(image);
    let image = image.resize_to_fill(w as u32, h as u32, image::imageops::FilterType::CatmullRom);
    image.save(p).map_err(candle::Error::wrap)?;
    Ok(())
}

/// Loads the safetensors files for a model from the hub based on a json index file.
pub fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
    let json_file = repo.get(json_file).map_err(candle::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).map_err(candle::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => candle::bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => candle::bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| repo.get(v).map_err(candle::Error::wrap))
        .collect::<Result<Vec<_>>>()?;
    Ok(safetensors_files)
}

pub fn hub_load_local_safetensors<P: AsRef<std::path::Path>>(
    path: P,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
    let path = path.as_ref();
    let jsfile = std::fs::File::open(path.join(json_file))?;
    let json: serde_json::Value = serde_json::from_reader(&jsfile).map_err(candle::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => candle::bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => candle::bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file);
        }
    }
    let safetensors_files: Vec<_> = safetensors_files
        .into_iter()
        .map(|v| path.join(v))
        .collect();
    Ok(safetensors_files)
}
