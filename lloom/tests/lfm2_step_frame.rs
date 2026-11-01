//! The own paged cache's decode step takes its live count and KV write slot
//! from a step frame (`profiles-and-memory-design.md` §4.6): it uploads
//! nothing, and a rewind or a reset is the value the next frame carries.
//!
//! The reference for every logit is the contiguous `InPlace` cache running
//! the same script — the same kernels' arithmetic over keys that are not
//! addressed through a frame — so a frame that carried a stale count or a
//! wrong slot reads different bits, rather than agreeing with itself
//! (`CONTRIBUTING.md` §3.1b).
#![cfg(feature = "metal")]

use lloom::models::lfm2::{AttnImpl, Config, ConvState, KvAppend, Model, Shape, State};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::profile::{PoolSpecExact, RewindableExact};
use lloom::testing::fixture::Fixture;
use lloom::testing::tiny::TinyFamily;
use lloom_tensor::{DType, Device, Result, Tensor};

/// The paged arm's shape (a `Paged` config takes a `Paged` state) and the
/// contiguous reference's; both rewind.
type P = PoolSpecExact;
type Reference = RewindableExact;

/// Tokens a page holds: small, so the scripts cross page and chunk
/// boundaries inside rewinds.
const PAGE: usize = 8;

fn tiny() -> Config {
    let mut c = Lfm2Tiny::template();
    c.num_hidden_layers = 6;
    c.layer_types = Config::tiny_layer_types(6);
    c
}

fn arm(fixture: &Fixture<Lfm2Tiny>, kv: KvAppend) -> Config {
    fixture.config(|c| {
        c.attn_impl = AttnImpl::FlashDecoding;
        c.kv_append = kv;
        c.flash_page_size = PAGE;
        c.flash_pages_per_chunk = 1;
        c.conv_state = ConvState::RotatingRing { k: 4 };
    })
}

fn tok(i: usize) -> u32 {
    (3 + (i * 7) % 60) as u32
}

/// One step of a script.
#[derive(Clone, Copy, Debug)]
enum Op {
    /// Feed `n` tokens (a prefill piece, or one decode token).
    Feed(usize),
    /// Discard the last `n` positions.
    Rewind(usize),
    /// Open a `k`-position verify window, feed it, accept `n`.
    Verify { k: usize, n: usize },
    /// Release the sequence; the next feed starts at position 0.
    Reset,
}

/// Every forward's logits over `script`, in order, on `cfg`'s cache arm.
fn run<Q: Shape>(
    model: &Model<Q>,
    cfg: &Config,
    dev: &Device,
    script: &[Op],
) -> Result<Vec<Vec<u32>>> {
    let mut cache = State::<Q>::new(true, DType::F32, cfg, dev)?;
    let mut pos = 0usize;
    let mut fed = 0usize;
    let mut out = Vec::new();
    let mut feed = |cache: &mut State<Q>, n: usize, pos: &mut usize| -> Result<Vec<u32>> {
        let ids: Vec<u32> = (fed..fed + n).map(tok).collect();
        fed += n;
        let input = Tensor::new(ids.as_slice(), dev)?.unsqueeze(0)?;
        let y = model.forward(&input, *pos, cache)?;
        *pos += n;
        let v: Vec<f32> = y.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
        Ok(v.iter().map(|x| x.to_bits()).collect())
    };
    for op in script {
        match *op {
            Op::Feed(n) => out.push(feed(&mut cache, n, &mut pos)?),
            Op::Rewind(n) => {
                cache.rewind(n)?;
                pos -= n;
            }
            Op::Verify { k, n } => {
                let window = cache.advance(k)?;
                out.push(feed(&mut cache, k, &mut pos)?);
                cache.resolve(window, n)?;
                pos -= k - n;
            }
            Op::Reset => {
                cache.clear();
                dev.synchronize()?;
                pos = 0;
            }
        }
    }
    Ok(out)
}

fn decode(n: usize) -> impl Iterator<Item = Op> {
    std::iter::repeat_n(Op::Feed(1), n)
}

/// The paged arm against the contiguous one over `script`, bit for bit.
fn check(name: &str, script: &[Op]) -> Result<()> {
    let dev = Device::new_metal(0)?;
    let Some(fixture) = Fixture::<Lfm2Tiny>::new(name, tiny())? else {
        return Ok(());
    };
    let paged = arm(&fixture, KvAppend::Paged);
    let inplace = arm(&fixture, KvAppend::InPlace);
    let model = fixture.model_in::<P>(&paged, Some(DType::F32), &dev)?;
    let reference = fixture.model_in::<Reference>(&inplace, Some(DType::F32), &dev)?;
    let got = run(&model, &paged, &dev, script)?;
    let want = run(&reference, &inplace, &dev, script)?;
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            g, w,
            "{name}: forward {i} of {script:?} differs from InPlace"
        );
    }
    // Non-vacuity: the forwards are not all one answer.
    assert!(
        got.windows(2).any(|w| w[0] != w[1]),
        "{name}: every forward agreed"
    );
    println!(
        "STEP_FRAME {name} forwards={} identical_to_inplace=true",
        got.len()
    );
    Ok(())
}

/// A lookahead decode overruns EOS by one forward and rewinds it, at a page
/// boundary: position 16 starts page 2 and is the overrun.
#[test]
fn an_eos_overrun_rewound_reads_what_a_run_without_it_reads() -> Result<()> {
    let mut s = vec![Op::Feed(5)];
    s.extend(decode(12)); // positions 5..16
    s.push(Op::Rewind(1)); // back to 16, below the page it opened
    s.extend(decode(6));
    check("eos_overrun", &s)
}

/// A verify window across a page boundary, mostly rejected: the window
/// writes positions 6..10, two are kept, and the pages past them stay held.
#[test]
fn a_rejected_verify_window_reads_as_its_accepted_prefix() -> Result<()> {
    let mut s = vec![Op::Feed(5)];
    s.extend(decode(1)); // position 5
    s.push(Op::Verify { k: 4, n: 2 }); // 6..9 written, 6..7 kept
    s.extend(decode(10)); // 8.. through page 1 and into page 2
    s.push(Op::Verify { k: 3, n: 0 }); // fully rejected
    s.extend(decode(3));
    check("spec_reject", &s)
}

/// A turn that diverges from what the cache holds: three positions dropped
/// across a chunk boundary and replaced by different tokens.
#[test]
fn a_diverging_turn_reads_its_own_tokens() -> Result<()> {
    let mut s = vec![Op::Feed(7)];
    s.extend(decode(10)); // positions 7..16
    s.push(Op::Rewind(3)); // back to 14, before the chunk 16 opened
    s.push(Op::Feed(3)); // the new turn's tokens
    s.extend(decode(5));
    check("turn_divergence", &s)
}

/// A reset, then a different sequence on recycled pages.
#[test]
fn a_reset_sequence_reads_as_a_fresh_one() -> Result<()> {
    let mut s = vec![Op::Feed(9)];
    s.extend(decode(12));
    s.push(Op::Reset);
    s.push(Op::Feed(3));
    s.extend(decode(20));
    check("reset", &s)
}

/// A decode step uploads nothing: its live count and write slot are a step
/// frame's fields, and the chunk table is appended in place. The only steps
/// allowed to upload are the ones that open a chunk, where the attention's
/// parameter blocks change (a page boundary, §4.6's one transition).
#[cfg(any(debug_assertions, feature = "profile"))]
#[test]
fn a_decode_step_uploads_nothing() -> Result<()> {
    use lloom_tensor::metal_backend::host_upload_count;
    let dev = Device::new_metal(0)?;
    let Some(fixture) = Fixture::<Lfm2Tiny>::new("uploads", tiny())? else {
        return Ok(());
    };
    let cfg = arm(&fixture, KvAppend::Paged);
    let model = fixture.model_in::<P>(&cfg, Some(DType::F32), &dev)?;
    let mut cache = State::<P>::new(true, DType::F32, &cfg, &dev)?;

    // Positive control: the counter sees an upload.
    let before = host_upload_count();
    let _one = Tensor::new(&[1u32], &dev)?;
    assert_eq!(
        host_upload_count() - before,
        1,
        "the counter missed an upload"
    );

    let prompt: Vec<u32> = (0..5).map(tok).collect();
    let input = Tensor::new(prompt.as_slice(), &dev)?.unsqueeze(0)?;
    let _ = model.forward(&input, 0, &mut cache)?;
    let steps = 40;
    let inputs: Vec<Tensor> = (0..steps)
        .map(|i| Tensor::new(&[tok(5 + i)], &dev)?.unsqueeze(0))
        .collect::<Result<_>>()?;
    dev.synchronize()?;
    let mut uploads = Vec::with_capacity(steps);
    let mut outs = Vec::with_capacity(steps);
    for (i, input) in inputs.iter().enumerate() {
        let before = host_upload_count();
        outs.push(model.forward(input, 5 + i, &mut cache)?);
        uploads.push(host_upload_count() - before);
    }
    dev.synchronize()?;
    for o in &outs {
        assert!(o
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|x| x.is_finite()));
    }
    // Step `i` writes position `5 + i`; it opens a chunk when that position
    // is a multiple of the chunk.
    let opens = |i: usize| (5 + i) % PAGE == 0;
    let steady: Vec<(usize, u64)> = uploads
        .iter()
        .enumerate()
        .skip(1)
        .filter(|&(i, _)| !opens(i))
        .map(|(i, &n)| (i, n))
        .collect();
    println!("STEP_FRAME uploads per decode step {uploads:?}");
    assert!(
        steady.iter().all(|&(_, n)| n == 0),
        "a decode step uploaded: (step, uploads) {steady:?}"
    );
    Ok(())
}
