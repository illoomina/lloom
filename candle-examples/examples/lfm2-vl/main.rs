//! LFM2-VL: an image in, a response out — and the gate that says the image mattered.
//!
//! Harness for lloom issue #597. Two modes, and they answer different questions:
//!
//! - **`--gate`** — compare the host preprocessing (from the committed PNGs),
//!   the patch embedding, the position-resampled embeddings, the tower and the
//!   projector, stage by stage, against a committed golden produced by
//!   `transformers`' own `Lfm2VlModel`. This is the **numeric** check. It loads
//!   no language model, so a failure is the vision path and nothing else.
//! - **default** — run a real image plus a prompt through the whole path and
//!   print what the model says. This is the **product** check, and on its own it
//!   proves nothing (see below).
//!
//! # Why the eyeball is not the gate, and why the control is not optional
//!
//! `DESIGN.md` §2.4e is the reason this harness has a control arm at all.
//! Measured there: **`LFM2.5-VL-3B` answers a text-only question about thread
//! scheduling with *"The provided image is a diagram of a flowchart…"*** — the
//! checkpoint hallucinates image captions **with no image attached**, on 4 of 9
//! prompts, and the incumbent prompt is its worst case.
//!
//! So *"we fed it a pelican and it described a pelican"* is **not** evidence the
//! vision path works. This checkpoint describes images that do not exist. The
//! only thing that separates *the image reached generation* from *the model is
//! doing what it always does* is **running the same prompt against two different
//! images and showing the outputs differ** — which is what `--control-image`
//! does, and why a run without one prints a warning rather than a verdict.
//!
//! # Encode once, attend every step
//!
//! The image is encoded **once**, projected once, and spliced into the prompt's
//! embeddings before the prefill. Every subsequent decode step attends to those
//! positions for free because they are in the KV cache, exactly like text.
//! There is no per-token re-encode and there is no vision dispatch in the decode
//! loop.
//!
//! `--census` checks both halves of that from the backend rather than from this
//! file's own bookkeeping:
//!
//! - **the load census** wraps the checkpoint in a recording backend and names
//!   the tensors each constructor actually loads. The language model must load
//!   exactly the header's `model.language_model.*` set and the vision encoder
//!   exactly `model.vision_tower.*` + `model.multi_modal_projector.*`, with no
//!   overlap — so a text-only process holds none of the tower (`DESIGN.md`
//!   §5.5a). The pool gauge is read after each constructor as the reading from
//!   outside the allocator.
//! - **the dispatch census** reads the Metal recorder (`CANDLE_METAL_PROFILE`)
//!   over three windows per request: the image encode, the prefill, and the
//!   decode steps. The kernels the encode dispatches and the prefill does not
//!   are the tower's signature; the decode window must contain none of them.

use anyhow::{Context, Result};
use candle::{DType, Device, Tensor};
use candle_examples::lfm2_setup::{default_model_dir, parse_config, tensor_names, weight_files};
use clap::Parser;
use lloom::generation::{LogitsProcessor, Sampling};
use lloom::models::lfm2::{Cache, Model};
use lloom::models::lfm2_vl::{merge_image_features, ImageProcessor, Lfm2VlConfig, VisionEncoder};
use lloom::nn::var_builder::SimpleBackend;
use lloom::nn::VarBuilder;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(about = "LFM2-VL image input, and the pelican test (lloom #597)")]
struct Args {
    /// Run the numeric gate against the committed golden and exit.
    #[arg(long)]
    gate: bool,

    /// Directory holding the golden produced by `make_golden.py`.
    #[arg(long)]
    golden_dir: Option<String>,

    /// The image to feed the model.
    #[arg(long)]
    image: Option<String>,

    /// A DIFFERENT image, same prompt. The image-conditioning discriminator:
    /// if the two outputs do not differ, the vision path is inert.
    #[arg(long)]
    control_image: Option<String>,

    #[arg(long, default_value = "Generate an SVG of a pelican riding a bicycle.")]
    prompt: String,

    /// Where to write the responses (and the SVGs, if any are found).
    #[arg(long, default_value = "vl-out")]
    out_dir: String,

    #[arg(long, default_value_t = 512)]
    max_tokens: usize,

    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    #[arg(long)]
    model_dir: Option<String>,

    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    dtype: Option<String>,

    /// Take the load census and the dispatch census (module docs) and fail if
    /// the language model loads any vision tensor or a decode step dispatches
    /// any of the tower's kernels. Metal only.
    #[arg(long)]
    census: bool,
}

/// The Metal dispatch recorder, with a stand-in off Metal so the call sites
/// stay unconditional.
#[cfg(feature = "metal")]
mod profile {
    pub use candle::metal_backend::profile::{enabled, reset, snapshot};
}

#[cfg(not(feature = "metal"))]
mod profile {
    #[derive(Clone, Debug, Default)]
    pub struct Snapshot {
        pub dispatches: u64,
        pub by_label: Vec<(String, u64)>,
    }
    pub fn enabled() -> bool {
        false
    }
    pub fn reset() {}
    pub fn snapshot() -> Snapshot {
        Snapshot::default()
    }
}

/// Dispatches recorded over one window, per pipeline label.
#[derive(Debug, Default, Clone)]
struct Window {
    dispatches: u64,
    labels: BTreeMap<String, u64>,
}

/// Close the current recorder window and open the next one. Synchronizes
/// first, as `profile::reset` requires.
fn take_window(device: &Device) -> Result<Window> {
    device.synchronize()?;
    let s = profile::snapshot();
    profile::reset();
    Ok(Window {
        dispatches: s.dispatches,
        labels: s.by_label.into_iter().collect(),
    })
}

/// The three windows of one image request.
#[derive(Debug, Clone)]
struct DispatchCensus {
    encode: Window,
    prefill: Window,
    decode: Window,
}

impl DispatchCensus {
    /// Labels the encode dispatched and the prefill did not: the kernels only
    /// the vision path uses.
    fn tower_signature(&self) -> BTreeMap<String, u64> {
        self.encode
            .labels
            .iter()
            .filter(|(l, _)| !self.prefill.labels.contains_key(*l))
            .map(|(l, n)| (l.clone(), *n))
            .collect()
    }
}

/// A checkpoint backend that records which tensors are loaded through it.
///
/// Loading is a lookup by name (`DESIGN.md` §5.5a), so the set of names a
/// constructor asks for is exactly the set of tensors that become device
/// buffers. One flag per header entry, set on load; no lock on the path.
struct Recording {
    inner: candle::safetensors::MmapedSafetensors,
    index: HashMap<String, usize>,
    loaded: Arc<[AtomicBool]>,
}

impl Recording {
    /// The backend, and the flags it sets — kept by the caller, since the
    /// backend itself moves into the `VarBuilder`.
    fn new(files: &[std::path::PathBuf], names: &[String]) -> Result<(Self, Arc<[AtomicBool]>)> {
        // SAFETY: the checkpoint files are not modified while mapped; this is
        // the same contract `VarBuilder::from_mmaped_safetensors` relies on.
        let inner = unsafe { candle::safetensors::MmapedSafetensors::multi(files)? };
        let index = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        let loaded: Arc<[AtomicBool]> = names.iter().map(|_| AtomicBool::new(false)).collect();
        let flags = Arc::clone(&loaded);
        Ok((
            Self {
                inner,
                index,
                loaded,
            },
            flags,
        ))
    }

    fn mark(&self, name: &str) {
        if let Some(&i) = self.index.get(name) {
            self.loaded[i].store(true, Ordering::Relaxed);
        }
    }
}

fn loaded_names(names: &[String], flags: &[AtomicBool]) -> BTreeSet<String> {
    names
        .iter()
        .zip(flags.iter())
        .filter(|(_, f)| f.load(Ordering::Relaxed))
        .map(|(n, _)| n.clone())
        .collect()
}

impl SimpleBackend for Recording {
    fn get(
        &self,
        s: candle::Shape,
        name: &str,
        h: lloom::nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle::Result<Tensor> {
        self.mark(name);
        SimpleBackend::get(&self.inner, s, name, h, dtype, dev)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle::Result<Tensor> {
        self.mark(name);
        SimpleBackend::get_unchecked(&self.inner, name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        SimpleBackend::contains_tensor(&self.inner, name)
    }
}

/// Every tensor in the header, with its size in bytes on disk.
fn header_sizes(files: &[std::path::PathBuf]) -> Result<BTreeMap<String, usize>> {
    // SAFETY: as in `Recording::new`.
    let st = unsafe { candle::safetensors::MmapedSafetensors::multi(files)? };
    Ok(st
        .tensors()
        .into_iter()
        .map(|(n, v)| (n, v.data().len()))
        .collect())
}

/// Which tensors each constructor loaded, checked against the header's own
/// prefix groups: the absolute reference, not a count carried over from a
/// document.
fn load_census(
    header: &BTreeMap<String, usize>,
    language: &BTreeSet<String>,
    vision: &BTreeSet<String>,
) -> Result<()> {
    let bytes = |set: &BTreeSet<String>| -> usize { set.iter().map(|n| header[n]).sum() };
    let want_language: BTreeSet<String> = header
        .keys()
        .filter(|n| n.starts_with("model.language_model."))
        .cloned()
        .collect();
    let want_vision: BTreeSet<String> = header
        .keys()
        .filter(|n| {
            n.starts_with("model.vision_tower.") || n.starts_with("model.multi_modal_projector.")
        })
        .cloned()
        .collect();
    let overlap = language.intersection(vision).count();
    let union = language.union(vision).count();
    println!(
        "load census: header {} tensors, {} B",
        header.len(),
        header.values().sum::<usize>()
    );
    println!(
        "  language model   loaded {:>4} tensors {:>13} B   header model.language_model.* {:>4}",
        language.len(),
        bytes(language),
        want_language.len()
    );
    println!(
        "  vision encoder   loaded {:>4} tensors {:>13} B   header vision_tower.* + multi_modal_projector.* {:>4}",
        vision.len(),
        bytes(vision),
        want_vision.len()
    );
    for prefix in ["model.vision_tower.", "model.multi_modal_projector."] {
        let part: BTreeSet<String> = vision
            .iter()
            .filter(|n| n.starts_with(prefix))
            .cloned()
            .collect();
        println!(
            "    of which {prefix:<30} {:>4} tensors {:>13} B",
            part.len(),
            bytes(&part)
        );
    }
    println!("  overlap {overlap}, union {union} of {}", header.len());
    anyhow::ensure!(
        *language == want_language,
        "the language model loaded {} tensors, the header's language group has {}: \
         missing {:?}, extra {:?}",
        language.len(),
        want_language.len(),
        want_language
            .difference(language)
            .take(5)
            .collect::<Vec<_>>(),
        language
            .difference(&want_language)
            .take(5)
            .collect::<Vec<_>>()
    );
    anyhow::ensure!(
        *vision == want_vision,
        "the vision encoder loaded {} tensors, the header's vision group has {}: \
         missing {:?}, extra {:?}",
        vision.len(),
        want_vision.len(),
        want_vision.difference(vision).take(5).collect::<Vec<_>>(),
        vision.difference(&want_vision).take(5).collect::<Vec<_>>()
    );
    anyhow::ensure!(
        overlap == 0,
        "{overlap} tensors are loaded by both constructors"
    );
    anyhow::ensure!(
        union == header.len(),
        "{} header tensors are loaded by neither constructor",
        header.len() - union
    );
    println!(
        "load census: PASS (text-only loads no vision tensor; the two sets partition the header)"
    );
    Ok(())
}

/// The pool's live bytes, synchronized so released temporaries have settled.
fn pool_live(device: &Device) -> Result<u64> {
    device.synchronize()?;
    Ok(lloom::nn::memory::MemSnapshot::read(device).pool_live)
}

/// Load one f32 tensor written by `make_golden.py`.
///
/// The corpus commits these gzipped (`DESIGN.md` §3.2 prices its growth: the
/// seven tensors are 7.4 MB raw and 1.1 MB compressed), so a fresh checkout
/// needs one command before the gate runs:
///
/// ```sh
/// gunzip -k design/measurements/issue-597-raw/*.bin.gz
/// ```
///
/// Reading the compressed form directly would mean a decompressor dependency in
/// `candle-examples` for one gate; the missing file names itself instead.
fn read_golden_bin(path: &std::path::Path, shape: &[usize], device: &Device) -> Result<Tensor> {
    let bytes = std::fs::read(path).with_context(|| {
        format!(
            "reading {} — if only {}.gz is present, `gunzip -k` it first",
            path.display(),
            path.display()
        )
    })?;
    let n: usize = shape.iter().product();
    anyhow::ensure!(
        bytes.len() == n * 4,
        "{} is {} bytes, expected {} for shape {shape:?}",
        path.display(),
        bytes.len(),
        n * 4
    );
    let vals: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(Tensor::from_vec(vals, shape, device)?)
}

#[derive(serde::Deserialize)]
struct GoldenEntry {
    shape: Vec<usize>,
}

#[derive(serde::Deserialize)]
struct GoldenIo {
    patches: Option<GoldenEntry>,
    tower_last_hidden: Option<GoldenEntry>,
    projected: Option<GoldenEntry>,
}

#[derive(serde::Deserialize)]
struct Golden {
    rows: usize,
    cols: usize,
    inputs: GoldenIo,
    outputs: GoldenIo,
    /// Optional: `{ "<image file>": { "file": "pixels_x.bin", "rows": R, "cols": C } }`
    /// — the reference processor's own `pixel_values` for a real image, so the
    /// host-side patchify is checked against it rather than assumed.
    #[serde(default)]
    pixels: std::collections::BTreeMap<String, PixelsEntry>,
}

#[derive(serde::Deserialize)]
struct PixelsEntry {
    file: String,
    rows: usize,
    cols: usize,
    image: String,
}

/// How far apart two tensors are, on the three readings that separate a
/// reordering from a defect (`DESIGN.md` §2.3.5a, §2.3.9a).
struct Diff {
    /// Largest absolute difference.
    abs: f64,
    /// Largest absolute difference as a fraction of the tensor's own scale.
    /// The denominator is the reference's absmax rather than each element's own
    /// magnitude: a per-element ratio is dominated by elements near zero, where
    /// a difference of one ulp reads as a large relative error and says nothing.
    rel_to_scale: f64,
    /// The same difference counted in f32 ulps at the reference's absmax — the
    /// unit §2.3.9a's ladder is stated in.
    ulps: f64,
    /// The reference's own absmax, so a reader can see what the ratio is of.
    scale: f64,
}

fn max_diff(got: &Tensor, want: &Tensor) -> Result<Diff> {
    let a = got.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let b = want.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    anyhow::ensure!(a.len() == b.len(), "length {} vs {}", a.len(), b.len());
    let mut abs = 0f64;
    let mut scale = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        abs = abs.max((*x as f64 - *y as f64).abs());
        scale = scale.max((*y as f64).abs());
    }
    let scale = scale.max(f64::MIN_POSITIVE);
    // One f32 ulp at this magnitude.
    let ulp = (scale as f32).next_up() as f64 - scale;
    Ok(Diff {
        abs,
        rel_to_scale: abs / scale,
        ulps: if ulp > 0.0 { abs / ulp } else { 0.0 },
        scale,
    })
}

/// The numeric gate. Returns non-zero on failure, per `CONTRIBUTING.md` §3.1a.
fn run_gate(args: &Args, device: &Device) -> Result<()> {
    let golden_dir = std::path::PathBuf::from(
        args.golden_dir
            .clone()
            .context("--gate needs --golden-dir")?,
    );
    let meta: Golden = serde_json::from_str(
        &std::fs::read_to_string(golden_dir.join("golden.json"))
            .with_context(|| format!("reading {}", golden_dir.join("golden.json").display()))?,
    )?;

    let patches_shape = meta
        .inputs
        .patches
        .as_ref()
        .context("golden has no `patches` input")?
        .shape
        .clone();
    let tower_shape = meta
        .outputs
        .tower_last_hidden
        .as_ref()
        .context("golden has no `tower_last_hidden` output")?
        .shape
        .clone();
    let proj_shape = meta
        .outputs
        .projected
        .as_ref()
        .context("golden has no `projected` output")?
        .shape
        .clone();

    let patches = read_golden_bin(&golden_dir.join("patches.bin"), &patches_shape, device)?;
    let want_tower = read_golden_bin(
        &golden_dir.join("tower_last_hidden.bin"),
        &tower_shape,
        device,
    )?;
    let want_proj = read_golden_bin(&golden_dir.join("projected.bin"), &proj_shape, device)?;

    // f32 throughout: the golden is f32 and this gate is about whether the
    // arithmetic is the same one, not about what the consumer loads.
    let model_dir = model_dir(args)?;
    let cfg = vl_config(&model_dir)?;
    let files = weight_files(&model_dir)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, device)? };
    let encoder = VisionEncoder::new(&cfg, 2048, vb).context("building the vision encoder")?;

    // The host-side preprocessing, against the reference processor's own
    // `pixel_values` for a real image file. This is the one stage the synthetic
    // input above cannot reach: it starts from patches, so it says nothing
    // about how an image BECAME patches. A wrong channel order or a
    // transposed patch walk produces a perfectly well-shaped tensor.
    let mut pixels_ok = true;
    let mut pixels_run = 0usize;
    for (name, entry) in meta.pixels.iter() {
        let want_path = golden_dir.join(&entry.file);
        if !want_path.exists() {
            // The pixel goldens are 3 MB each and regenerable from the
            // committed image plus `make_golden.py`, so they are not in the
            // corpus (`DESIGN.md` §3.2 prices its growth). Absent is not a
            // pass: it is reported, and `pixels_run` below carries the count so
            // a reader can tell "checked and matched" from "not checked".
            println!(
                "pixels[{name}]  ABSENT ({}) — regenerate with make_golden.py",
                entry.file
            );
            continue;
        }
        pixels_run += 1;
        let want = read_golden_bin(&want_path, &[entry.rows * entry.cols, 768], device)?;
        // `make_golden.py` records an ABSOLUTE image path, which is the path of
        // whichever worktree generated the golden — so the gate ran only there
        // and failed everywhere else with a missing-file error (found running it
        // from a second worktree, lloom #609). The images are committed beside
        // the golden, so fall back to resolving against `golden_dir`. The
        // recorded path is still tried first: it is what the golden was actually
        // built from, and preferring it keeps a deliberately out-of-tree image
        // working.
        let image_path = {
            let recorded = std::path::Path::new(&entry.image);
            if recorded.exists() {
                recorded.to_path_buf()
            } else {
                let rel = recorded
                    .file_name()
                    .map(|f| golden_dir.join("images").join(f))
                    .filter(|p| p.exists());
                rel.ok_or_else(|| {
                    anyhow::anyhow!(
                        "image {} not found, and no fallback under {}/images",
                        entry.image,
                        golden_dir.display()
                    )
                })?
            }
        };
        let img = image::ImageReader::open(&image_path)
            .with_context(|| format!("opening {}", image_path.display()))?
            .decode()?;
        let proc = ImageProcessor::default();
        let (rgb, r, c) = {
            use image::GenericImageView;
            let (w0, h0) = img.dimensions();
            let (r, c) = proc.target_grid(h0 as usize, w0 as usize);
            let (th, tw) = (r * proc.patch_size, c * proc.patch_size);
            let resized = if (tw as u32, th as u32) == img.dimensions() {
                img.clone()
            } else {
                img.resize_exact(
                    tw as u32,
                    th as u32,
                    image::imageops::FilterType::CatmullRom,
                )
            };
            (resized.to_rgb8().into_raw(), r, c)
        };
        anyhow::ensure!(
            (r, c) == (entry.rows, entry.cols),
            "{name}: our grid is {r}x{c}, the reference chose {}x{}",
            entry.rows,
            entry.cols
        );
        let got = proc
            .patchify_rgb8(&rgb, r, c, device, DType::F32)?
            .patches
            .squeeze(0)?;
        let d = max_diff(&got, &want)?;
        println!(
            "pixels[{name}]  grid {r}x{c}  abs {:.3e}  rel-to-scale {:.3e}  ulps {:8.1}",
            d.abs, d.rel_to_scale, d.ulps
        );
        // A tighter bound than the network stages: this is arithmetic on bytes,
        // with no accumulation, so anything above f32 rounding is a real
        // layout or normalisation difference.
        if d.rel_to_scale >= 1e-6 {
            pixels_ok = false;
        }
    }

    let (rows, cols) = (meta.rows, meta.cols);
    let patches = patches.to_dtype(DType::F32)?;

    // Stage by stage, if the golden carries the intermediate dumps. A single
    // end-to-end number says "wrong" and nothing else; the first stage that
    // diverges is the one to read.
    for (name, file, got) in [
        (
            "patch_embed",
            "stage_patchembed.bin",
            encoder.patch_embed_only(&patches)?,
        ),
        (
            "embeddings",
            "stage_embeddings.bin",
            encoder.embeddings_only(&patches, rows, cols)?,
        ),
    ] {
        let path = golden_dir.join(file);
        if !path.exists() {
            continue;
        }
        let want = read_golden_bin(&path, got.dims(), device)?;
        let d = max_diff(&got, &want)?;
        println!(
            "{name:<12} abs {:.3e}  rel-to-scale {:.3e}  ulps {:8.1}  (scale {:.3e})",
            d.abs, d.rel_to_scale, d.ulps, d.scale
        );
    }

    // The tower alone first, so a failure says which half.
    let got_tower = encoder
        .tower_only(&patches, rows, cols)
        .context("running the vision tower")?;
    let tower = max_diff(&got_tower, &want_tower)?;

    let got_proj = encoder
        .forward(&patches, rows, cols)
        .context("running tower + projector")?;
    let proj = max_diff(&got_proj, &want_proj)?;

    println!("grid         {rows}x{cols}  ({} patches)", rows * cols);
    println!(
        "tower        shape {:?} (want {:?})",
        got_tower.dims(),
        want_tower.dims()
    );
    println!(
        "tower        abs {:.3e}  rel-to-scale {:.3e}  ulps {:8.1}  (scale {:.3e})",
        tower.abs, tower.rel_to_scale, tower.ulps, tower.scale
    );
    println!(
        "projected    shape {:?} (want {:?})",
        got_proj.dims(),
        want_proj.dims()
    );
    println!(
        "projected    abs {:.3e}  rel-to-scale {:.3e}  ulps {:8.1}  (scale {:.3e})",
        proj.abs, proj.rel_to_scale, proj.ulps, proj.scale
    );

    // The bound is on the difference RELATIVE TO THE TENSOR'S OWN SCALE.
    //
    // Not an absolute bound: the tower's activations reach an absmax above 30
    // while the projected output's is 0.52, so one constant is either vacuous
    // at the top or unmeetable at the bottom. Not a per-element ratio either —
    // that is dominated by elements near zero, where one ulp of difference
    // reads as a huge relative error and means nothing.
    //
    // 1e-4 is chosen against what the two sides actually are: candle and torch
    // do not share a matmul, so their accumulation orders differ, and 27
    // encoder layers of f32 GEMMs accumulate that difference. It is ~4 orders
    // above the observed residual and ~4 orders below what the position-
    // embedding defect this gate caught was reading (rel-to-scale 0.30), so it
    // separates the two cleanly rather than splitting the difference.
    const TOL: f64 = 1e-4;
    let ok = pixels_ok && tower.rel_to_scale < TOL && proj.rel_to_scale < TOL;
    println!();
    println!(
        "stages checked: {} network + {} preprocessing of {} declared",
        4,
        pixels_run,
        meta.pixels.len()
    );
    if ok {
        println!("GATE PASS  (rel-to-scale tolerance {TOL:.0e}, preprocessing 1e-6)");
        Ok(())
    } else {
        if !pixels_ok {
            println!("  (a preprocessing stage is what failed)");
        }
        println!("GATE FAIL  (rel-to-scale tolerance {TOL:.0e}, preprocessing 1e-6)");
        std::process::exit(1);
    }
}

fn model_dir(args: &Args) -> Result<std::path::PathBuf> {
    args.model_dir
        .clone()
        .map(std::path::PathBuf::from)
        .or_else(default_model_dir)
        .context("no --model-dir given and the LFM2.5-VL-3B snapshot was not found in the HF cache")
}

fn vl_config(model_dir: &std::path::Path) -> Result<Lfm2VlConfig> {
    let raw = std::fs::read_to_string(model_dir.join("config.json"))
        .with_context(|| format!("reading {}", model_dir.join("config.json").display()))?;
    serde_json::from_str(&raw).context("parsing the VL half of config.json")
}

/// Decode an image file and resize it to the patch grid the processor chooses.
fn load_and_resize(path: &str, proc: &ImageProcessor) -> Result<(Vec<u8>, usize, usize)> {
    let img = image::ImageReader::open(path)
        .with_context(|| format!("opening {path}"))?
        .decode()
        .with_context(|| format!("decoding {path}"))?;
    let (w0, h0) = {
        use image::GenericImageView;
        img.dimensions()
    };
    let (rows, cols) = proc.target_grid(h0 as usize, w0 as usize);
    let (th, tw) = (rows * proc.patch_size, cols * proc.patch_size);
    // CatmullRom is a bicubic; `processor_config.json` asks for `resample: 3`.
    // This is an approximation to the reference's antialiased path and is
    // recorded as one — it is why the numeric gate uses synthetic patches
    // rather than a decoded file.
    let resized = img.resize_exact(
        tw as u32,
        th as u32,
        image::imageops::FilterType::CatmullRom,
    );
    Ok((resized.to_rgb8().into_raw(), rows, cols))
}

struct Run {
    label: String,
    image: String,
    text: String,
    tokens: usize,
    hit_eos: bool,
    census: Option<DispatchCensus>,
}

#[allow(clippy::too_many_arguments)]
fn generate_with_image(
    label: &str,
    image_path: &str,
    args: &Args,
    device: &Device,
    dtype: DType,
    model: &Model,
    encoder: &VisionEncoder,
    cfg: &Lfm2VlConfig,
    config: &lloom::models::lfm2::Config,
    tokenizer: &tokenizers::Tokenizer,
    eos_ids: &[u32],
    vision_runs: &mut usize,
) -> Result<Run> {
    let proc = ImageProcessor::default();
    let (rgb, rows, cols) = load_and_resize(image_path, &proc)?;
    let processed = proc.patchify_rgb8(&rgb, rows, cols, device, dtype)?;

    // The dispatch census's first window starts here: everything recorded
    // before it (weight loads, the previous request) is discarded.
    let census = profile::enabled();
    if census {
        take_window(device)?;
    }

    // ONE call, before the prefill. Everything after this attends to the result
    // through the KV cache.
    let features = encoder
        .forward(&processed.patches, processed.rows, processed.cols)
        .context("encoding the image")?;
    *vision_runs += 1;
    let n_image_tokens = features.dim(0)?;
    let encode_window = if census {
        Some(take_window(device)?)
    } else {
        None
    };

    // The prompt reserves exactly as many `image_token_id` positions as the
    // encoder produced rows. `merge_image_features` refuses if they disagree,
    // which is what keeps a miscount loud.
    //
    // The `<|image_start|>` / `<|image_end|>` delimiters are the reference
    // processor's `use_image_special_tokens`, which this checkpoint's config
    // sets to `true`. They are ORDINARY tokens with their own embeddings
    // (125009 and 125010) — they are not image positions and take no features.
    // Omitting them would put the image tokens somewhere the model was never
    // trained to find them, and nothing downstream would say so.
    let image_placeholder = format!(
        "<|image_start|>{}<|image_end|>",
        "<image>".repeat(n_image_tokens)
    );
    let templated = format!(
        "<|im_start|>user\n{image_placeholder}{}<|im_end|>\n<|im_start|>assistant\n",
        args.prompt
    );
    let ids = tokenizer
        .encode(templated.as_str(), true)
        .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?
        .get_ids()
        .to_vec();
    anyhow::ensure!(!ids.is_empty(), "prompt tokenized to nothing");

    let n_placeholders = ids.iter().filter(|&&t| t == cfg.image_token_id).count();
    anyhow::ensure!(
        n_placeholders == n_image_tokens,
        "the tokenizer produced {n_placeholders} image-token positions for {n_image_tokens} \
         image features — the `<image>` string is not mapping to id {}",
        cfg.image_token_id
    );

    let mut cache = Cache::new(true, dtype, config, device).context("allocating KV cache")?;
    let mut logits_processor = LogitsProcessor::from_sampling(args.seed, Sampling::ArgMax);

    let input = Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?;
    let text_embeds = model.embed(&input)?;
    let embeds = merge_image_features(&text_embeds, &ids, &features, cfg.image_token_id)
        .context("splicing image features into the prompt")?;

    let mut kv_len = 0usize;
    let mut logits = model
        .forward_embeds(&embeds, kv_len, &mut cache)
        .context("prefill")?
        .squeeze(0)?;
    kv_len += ids.len();
    let prefill_window = if census {
        Some(take_window(device)?)
    } else {
        None
    };

    let mut generated: Vec<u32> = Vec::with_capacity(args.max_tokens);
    let mut hit_eos = false;
    while generated.len() < args.max_tokens {
        let next = logits_processor.sample(&logits).context("sampling")?;
        if eos_ids.contains(&next) {
            hit_eos = true;
            break;
        }
        generated.push(next);
        let input = Tensor::new(&[next], device)?.unsqueeze(0)?;
        // Note: `forward`, not `forward_embeds`. The decode path is the text
        // path exactly, unchanged, and the image is reached through the cache.
        logits = model
            .forward(&input, kv_len, &mut cache)
            .context("decode")?
            .squeeze(0)?;
        kv_len += 1;
    }
    let census = match (encode_window, prefill_window) {
        (Some(encode), Some(prefill)) => Some(DispatchCensus {
            encode,
            prefill,
            decode: take_window(device)?,
        }),
        _ => None,
    };

    let text = tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("decoding: {e}"))?;

    Ok(Run {
        label: label.to_string(),
        image: image_path.to_string(),
        text,
        tokens: generated.len(),
        hit_eos,
        census,
    })
}

/// The dispatch-census verdict for one request (module docs).
fn check_dispatch_census(run: &Run) -> Result<()> {
    let c = run
        .census
        .as_ref()
        .context("no dispatch census was recorded — is CANDLE_METAL_PROFILE on a Metal build?")?;
    let signature = c.tower_signature();
    let leaked: BTreeMap<&String, &u64> = c
        .decode
        .labels
        .iter()
        .filter(|(l, _)| signature.contains_key(*l))
        .collect();
    let fmt = |m: &BTreeMap<String, u64>| -> String {
        m.iter()
            .map(|(l, n)| format!("{l}={n}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    println!(
        "dispatch census [{}]: encode {} dispatches ({} labels), prefill {} ({} labels), \
         decode {} over {} steps ({} labels)",
        run.label,
        c.encode.dispatches,
        c.encode.labels.len(),
        c.prefill.dispatches,
        c.prefill.labels.len(),
        c.decode.dispatches,
        run.tokens,
        c.decode.labels.len()
    );
    println!(
        "  tower signature (in encode, not in prefill): {} labels, {} dispatches: {}",
        signature.len(),
        signature.values().sum::<u64>(),
        fmt(&signature)
    );
    println!("  decode labels: {}", fmt(&c.decode.labels));
    // Positive control: a signature that is empty would make the check below
    // pass on anything.
    anyhow::ensure!(
        !signature.is_empty(),
        "[{}] the encode dispatched no kernel the prefill did not, so this census \
         cannot tell the tower from the language model",
        run.label
    );
    anyhow::ensure!(
        leaked.is_empty(),
        "[{}] the decode window dispatched tower kernels {:?} — the image is being \
         re-encoded per token",
        run.label,
        leaked
    );
    println!(
        "  decode window carries none of the tower signature: encode-once holds [{}]",
        run.label
    );
    Ok(())
}

/// Pull the first `<svg>…</svg>` out of a response, if there is one.
fn extract_svg(text: &str) -> Option<String> {
    let start = text.find("<svg")?;
    let end = text[start..].find("</svg>")? + start + "</svg>".len();
    Some(text[start..end].to_string())
}

fn main() -> Result<()> {
    let args = Args::parse();

    // The recorder reads its switch once, on first use, so it is set before
    // the device exists.
    if args.census {
        std::env::set_var("CANDLE_METAL_PROFILE", "1");
    }

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };

    if args.gate {
        return run_gate(&args, &device);
    }

    let image = args
        .image
        .clone()
        .context("give --image, or --gate to run the numeric check")?;

    // f16 on Metal, as every other LFM2 harness. Note `DESIGN.md` §3.9: bf16 has
    // no simdgroup primitives on this part, so the tower's 27 layer norms would
    // each take a 5-instruction shuffle tree at bf16 for no benefit.
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("f32") => DType::F32,
        Some("bf16") => DType::BF16,
        Some(other) => anyhow::bail!("unknown dtype {other}"),
        None => match &device {
            Device::Metal(_) => DType::F16,
            Device::Cuda(_) => DType::BF16,
            Device::Cpu => DType::F32,
        },
    };

    let model_dir = model_dir(&args)?;
    let config_raw = std::fs::read_to_string(model_dir.join("config.json"))
        .with_context(|| format!("reading {}", model_dir.join("config.json").display()))?;
    let config = parse_config(&config_raw)?;
    let vl_cfg = vl_config(&model_dir)?;

    let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;

    let files = weight_files(&model_dir)?;
    let names = tensor_names(&files[0])?;
    let nested = names.iter().any(|n| n.starts_with("model.language_model."))
        && !names.iter().any(|n| n == "model.embed_tokens.weight");
    anyhow::ensure!(
        nested,
        "this checkpoint is not a VL checkpoint — no `model.language_model.*` tensors"
    );

    // Under `--census` each constructor gets its own recording backend, so the
    // two load sets are observed separately rather than inferred.
    let header = header_sizes(&files)?;
    let header_names: Vec<String> = header.keys().cloned().collect();
    let (vb_raw_lang, vb_raw_vision, flags) = if args.census {
        anyhow::ensure!(
            matches!(device, Device::Metal(_)),
            "--census reads the Metal recorder and the Metal pool; run it on Metal"
        );
        let (lang, lang_flags) = Recording::new(&files, &header_names)?;
        let (vision, vision_flags) = Recording::new(&files, &header_names)?;
        (
            VarBuilder::from_backend(Box::new(lang), dtype, device.clone()),
            VarBuilder::from_backend(Box::new(vision), dtype, device.clone()),
            Some((lang_flags, vision_flags)),
        )
    } else {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
        (vb.clone(), vb, None)
    };
    // Two views of the checkpoint. The language model wants `model.*`
    // rewritten to `model.language_model.*`; the vision tower wants the
    // checkpoint's names untouched, because `model.vision_tower.*` would
    // otherwise be rewritten to `model.language_model.vision_tower.*` and fail
    // to load.
    let vb_lang = vb_raw_lang
        .clone()
        .rename_f(|name: &str| match name.strip_prefix("model.") {
            Some(rest) => format!("model.language_model.{rest}"),
            None => name.to_string(),
        });

    let live_before = pool_live(&device)?;
    let model = Model::new(&config, vb_lang).context("constructing the LFM2 language model")?;
    let live_text = pool_live(&device)?;
    let encoder = VisionEncoder::new(&vl_cfg, config.hidden_size, vb_raw_vision)
        .context("constructing the vision tower + projector")?;
    let live_vision = pool_live(&device)?;

    if let Some((lang_flags, vision_flags)) = &flags {
        let language = loaded_names(&header_names, lang_flags);
        let vision = loaded_names(&header_names, vision_flags);
        load_census(&header, &language, &vision)?;
        let gb = |b: u64| b as f64 / 1e9;
        let language_bytes: usize = language.iter().map(|n| header[n]).sum();
        let vision_bytes: usize = vision.iter().map(|n| header[n]).sum();
        println!(
            "pool_live: before load {:.4} GB, after the language model {:.4} GB \
             (+{:.4}; its tensors are {:.4} GB), after the vision encoder {:.4} GB \
             (+{:.4}; its tensors are {:.4} GB)",
            gb(live_before),
            gb(live_text),
            gb(live_text.saturating_sub(live_before)),
            language_bytes as f64 / 1e9,
            gb(live_vision),
            gb(live_vision.saturating_sub(live_text)),
            vision_bytes as f64 / 1e9,
        );
    }

    let mut eos_ids: Vec<u32> = config.eos_token_id.into_iter().collect();
    if let Some(im_end) = tokenizer.token_to_id("<|im_end|>") {
        eos_ids.push(im_end);
    }
    anyhow::ensure!(
        !eos_ids.is_empty(),
        "no EOS id: the config names none and the tokenizer has no `<|im_end|>`"
    );

    let out_dir = std::path::PathBuf::from(&args.out_dir);
    std::fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    // The exact prompt, beside the responses it produced.
    std::fs::write(out_dir.join("prompt.txt"), &args.prompt)?;

    let mut vision_runs = 0usize;
    let mut runs = Vec::new();

    runs.push(generate_with_image(
        "subject",
        &image,
        &args,
        &device,
        dtype,
        &model,
        &encoder,
        &vl_cfg,
        &config,
        &tokenizer,
        &eos_ids,
        &mut vision_runs,
    )?);

    if let Some(control) = args.control_image.as_deref() {
        runs.push(generate_with_image(
            "control",
            control,
            &args,
            &device,
            dtype,
            &model,
            &encoder,
            &vl_cfg,
            &config,
            &tokenizer,
            &eos_ids,
            &mut vision_runs,
        )?);
    }

    println!("prompt: {}\n", args.prompt);
    for run in &runs {
        println!("--- {} ({}) ---", run.label, run.image);
        println!("{}", run.text.trim());
        println!(
            "  ({} tokens, {})",
            run.tokens,
            if run.hit_eos {
                "stopped at EOS"
            } else {
                "hit --max-tokens"
            }
        );
        println!();

        let stem = out_dir.join(&run.label);
        std::fs::write(stem.with_extension("txt"), &run.text)?;
        if let Some(svg) = extract_svg(&run.text) {
            std::fs::write(stem.with_extension("svg"), &svg)?;
            println!("  wrote {}", stem.with_extension("svg").display());
        }
    }

    if args.census {
        let total_tokens: usize = runs.iter().map(|r| r.tokens).sum();
        // This count is kept beside the call it counts, so it can only report
        // what this file believes; the dispatch census below is the check.
        println!(
            "census: this harness called the encoder {vision_runs} time(s) for {} \
             request(s) and {total_tokens} decoded token(s)",
            runs.len()
        );
        for run in &runs {
            check_dispatch_census(run)?;
        }
    }

    // The verdict, and it is the whole reason the control exists. §2.4e measures
    // this checkpoint describing images that are not there, so a plausible
    // caption is not evidence of anything.
    match runs.len() {
        2 => {
            let differ = runs[0].text.trim() != runs[1].text.trim();
            println!();
            if differ {
                println!(
                    "IMAGE-CONDITIONING PASS — the same prompt over two images produced \
                     different output, so the image reaches generation"
                );
            } else {
                println!(
                    "IMAGE-CONDITIONING FAIL — two different images produced IDENTICAL \
                     output, so the vision path is plumbed but inert"
                );
                std::process::exit(1);
            }
        }
        _ => println!(
            "\nno --control-image given: this run shows the model produced text, and \
             says NOTHING about whether the image influenced it (DESIGN.md §2.4e — this \
             checkpoint captions images that do not exist)"
        ),
    }

    Ok(())
}
