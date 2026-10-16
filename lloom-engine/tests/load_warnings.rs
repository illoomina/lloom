//! The load-warning channel end to end (`lloom::load_warning`): every
//! load-time degradation reaches `lloom chat`'s `[loaded …]` line, and
//! `serve`'s `/health` and model card as `[{kind, message}]`, beside the
//! stderr line it always printed.
//!
//! Sources driven here, each against a control that must stay empty:
//! - `attention_route`: a tiny LFM2 at head_dim 72, where the requested
//!   decode kernel has no entry point and the load takes the nearest fused
//!   one (the template's 64 is the control); at head_dim 48, which no fused
//!   kernel instantiates, the load is refused rather than served composed;
//! - `readback`: `--readback ring` asks for the adaptive commit its EOS read
//!   cannot carry (`auto` is the control), and Qwen4-exp's `ring-fixed`,
//!   whose forwards read their fed id back (its `auto` is the control);
//! - `emulated_type`: an emulated type admitted on the engine's own device,
//!   read through the loaded engine (a device that admitted none is the
//!   control).
#![cfg(feature = "serve")]

use lloom::models::lfm2_testing::Lfm2Tiny;
#[cfg(feature = "metal")]
use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::testing::tiny::{TinyCheckpoint, TinyConfig, TinyFamily, Weights};
#[cfg(feature = "metal")]
use serde_json::json;
use serde_json::Value;
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const PROMPT: &str = "Hi!";

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("lloom-load-warnings-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The family's template, widened to the tokenizer's ids, `hidden_size` set
/// when given (LFM2's head dim is `hidden_size / 2` on the template).
fn write<F: TinyFamily>(dir: &Path, hidden_size: Option<usize>) -> PathBuf {
    let mut cfg = F::template();
    cfg.set_axis("vocab_size", F::tokenizer().vocab_size());
    if let Some(h) = hidden_size {
        assert!(cfg.set_axis("hidden_size", h), "hidden_size is an axis");
    }
    cfg.derive();
    TinyCheckpoint::<F>::new(cfg, Weights::default())
        .write(dir)
        .expect("the tiny checkpoint writes")
}

/// The tiny Qwen4-exp at head dim 32, which its fused attention route
/// serves, so it loads as production loads it: the template's head dim (16)
/// is the composed oracle's fixture, which production refuses on Metal.
#[cfg(feature = "metal")]
fn write_production_qwen(dir: &Path) -> PathBuf {
    let mut cfg = Qwen4ExpTiny::template();
    TinyConfig::set_axis(
        &mut cfg,
        "vocab_size",
        Qwen4ExpTiny::tokenizer().vocab_size(),
    );
    assert!(
        TinyConfig::set_axis(&mut cfg, "head_dim", 32),
        "head_dim is an axis"
    );
    TinyConfig::derive(&mut cfg);
    TinyCheckpoint::<Qwen4ExpTiny>::new(cfg, Weights::default())
        .write(dir)
        .expect("the tiny checkpoint writes")
}

/// One `lloom chat --raw` turn: its stderr.
fn chat(model: &Path, extra: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_lloom"))
        .args(["chat", "--model"])
        .arg(model)
        .args(["--raw", "-p", PROMPT, "-n", "2"])
        .args(extra)
        .output()
        .expect("lloom runs");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "lloom chat {model:?} {extra:?} exited {:?}:\n{stderr}",
        out.status
    );
    stderr
}

/// The `load_warnings=` field of the `[loaded …]` line.
fn loaded_warnings(stderr: &str) -> String {
    let line = stderr
        .lines()
        .find(|l| l.starts_with("[loaded "))
        .unwrap_or_else(|| panic!("no loaded line:\n{stderr}"));
    let (_, field) = line
        .rsplit_once("; load_warnings=")
        .unwrap_or_else(|| panic!("no load_warnings field: {line}"));
    field.trim_end_matches(']').to_string()
}

struct Server {
    child: std::process::Child,
    port: u16,
}

impl Server {
    fn start(model: &Path, extra: &[&str]) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_lloom"))
            .args(["serve", "--model"])
            .arg(model)
            .args(extra)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("lloom serve runs");
        let mut line = String::new();
        std::io::BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        let ready: Value = serde_json::from_str(&line).unwrap_or_else(|e| panic!("{line:?}: {e}"));
        assert_eq!(ready["kind"], "ready");
        Self {
            child,
            port: ready["port"].as_u64().unwrap() as u16,
        }
    }

    fn get(&self, path: &str) -> Value {
        let mut s = std::net::TcpStream::connect(("127.0.0.1", self.port)).unwrap();
        write!(
            s,
            "GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        )
        .unwrap();
        let mut response = String::new();
        s.read_to_string(&mut response).unwrap();
        let (_, body) = response.split_once("\r\n\r\n").unwrap();
        serde_json::from_str(body).unwrap_or_else(|e| panic!("{body:?}: {e}"))
    }

    fn post(&self, path: &str) {
        let mut s = std::net::TcpStream::connect(("127.0.0.1", self.port)).unwrap();
        let _ = write!(
            s,
            "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
        );
        let mut response = String::new();
        let _ = s.read_to_string(&mut response);
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.post("/shutdown");
        let _ = self.child.wait();
    }
}

/// `/health`'s and the model card's lists, which must be the same list.
fn served_warnings(model: &Path, extra: &[&str]) -> Value {
    let server = Server::start(model, extra);
    let health = server.get("/health")["load_warnings"].clone();
    let card = server.get("/v1/models")["data"][0]["load_warnings"].clone();
    assert_eq!(health, card, "/health and the model card disagree");
    assert!(health.is_array(), "{health}");
    health
}

/// The attention route on Metal: at head_dim 72 the requested decode kernel
/// (`sdpa_vector`) has no entry point and the load takes the nearest fused
/// kernel, named on the loaded line, `/health` and the model card; the
/// template's head dim (64, inside every set) names nothing. At head_dim 48
/// no fused kernel covers the attention and the load is refused, naming the
/// capability: the composed attention does not serve.
#[cfg(feature = "metal")]
#[test]
fn an_uncovered_head_dim_reaches_every_surface() {
    let dir = scratch("lfm2-72");
    let off = write::<Lfm2Tiny>(&dir.join("off"), Some(144));
    let on = write::<Lfm2Tiny>(&dir.join("on"), None);
    let uncovered = write::<Lfm2Tiny>(&dir.join("uncovered"), Some(96));

    let stderr = chat(&off, &[]);
    let field = loaded_warnings(&stderr);
    eprintln!("head_dim 72: load_warnings={field}");
    assert!(
        field.starts_with("attention_route: lfm2: attention head_dim 72 at F16 under Sdpa: "),
        "{field}"
    );
    assert!(field.contains("(the nearest fused kernel)"), "{field}");
    assert!(!field.contains("composed"), "{field}");
    // The stderr line it always printed is still there, and is the message.
    let message = field.trim_start_matches("attention_route: ");
    assert!(
        stderr.lines().any(|l| l == message),
        "the stderr line is not the channel's message:\n{stderr}"
    );
    assert_eq!(loaded_warnings(&chat(&on, &[])), "none");

    let list = served_warnings(&off, &[]);
    assert_eq!(list.as_array().unwrap().len(), 1, "{list}");
    assert_eq!(list[0]["kind"], "attention_route");
    assert_eq!(list[0]["message"], message);
    assert_eq!(served_warnings(&on, &[]), json!([]));

    let out = Command::new(env!("CARGO_BIN_EXE_lloom"))
        .args(["chat", "--model"])
        .arg(&uncovered)
        .args(["--raw", "-p", PROMPT, "-n", "2"])
        .output()
        .expect("lloom runs");
    let refused = String::from_utf8_lossy(&out.stderr);
    eprintln!("head_dim 48: {refused}");
    assert!(!out.status.success(), "head_dim 48 loaded:\n{refused}");
    assert!(
        refused.contains("head dim 48") && refused.contains("does not serve on Metal"),
        "refused without naming the capability:\n{refused}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// A readback clamp: `--readback ring` asks for the adaptive commit, whose
/// EOS read reads each token back, so the ring runs on the fixed cadence.
/// On the CPU as on Metal, since the clamp is decided before the family
/// loads.
#[test]
fn a_readback_clamp_reaches_every_surface() {
    let dir = scratch("lfm2-ring");
    let model = write::<Lfm2Tiny>(&dir, None);
    let mut devices: Vec<&[&str]> = vec![&["--cpu"]];
    if cfg!(feature = "metal") {
        devices.push(&[]);
    }
    for dev in devices {
        let mut args = dev.to_vec();
        args.extend(["--readback", "ring"]);
        let field = loaded_warnings(&chat(&model, &args));
        eprintln!("{dev:?} --readback ring: load_warnings={field}");
        assert!(
            field.starts_with("readback: adaptive_commit is off on ring"),
            "{field}"
        );
        assert_eq!(loaded_warnings(&chat(&model, dev)), "none", "{dev:?} auto");
        let list = served_warnings(&model, &args);
        assert_eq!(list.as_array().unwrap().len(), 1, "{list}");
        assert_eq!(list[0]["kind"], "readback");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Qwen4-exp's own readback resolution: `ring-fixed`'s forwards read their
/// fed id back where the host fill cannot run (no PLE on the tiny), which is
/// slower than blocking; `auto` resolves to blocking, which is the family's
/// arm as asked, and warns nothing.
#[cfg(feature = "metal")]
#[test]
fn qwen4exp_readback_that_reads_its_id_back_warns() {
    let dir = scratch("qwen");
    let model = write_production_qwen(&dir);
    let tok = dir.join("tokenizer.json");
    let tok = tok.to_str().unwrap();
    let base = ["--tokenizer", tok, "--experts", "resident"];
    let mut ring = base.to_vec();
    ring.extend(["--readback", "ring-fixed"]);
    let field = loaded_warnings(&chat(&model, &ring));
    eprintln!("qwen4exp ring-fixed: load_warnings={field}");
    // The arm renders from the value that runs (`RingArm::name`).
    assert!(
        field.starts_with("readback: ring-nonblocking lookahead=1 "),
        "{field}"
    );
    assert!(
        field.contains("each decode forward reads its fed id back"),
        "{field}"
    );
    assert_eq!(loaded_warnings(&chat(&model, &base)), "none");
    std::fs::remove_dir_all(&dir).ok();
}

/// An emulated type admitted on the loaded engine's device is on its
/// channel, in the gate's words; before the admission it is not. Loaded
/// in-process so the test can reach the engine's own device.
#[cfg(feature = "metal")]
#[test]
fn an_admitted_emulated_type_reaches_the_engine() {
    use clap::Parser;
    use lloom_engine::app::{load_engine, resolve_options_uninstalled, CommonArgs, Shape};
    use lloom_tensor::{DType, Device};

    #[derive(Parser)]
    struct Args {
        #[command(flatten)]
        common: CommonArgs,
    }
    let dir = scratch("emulated");
    let lfm2 = write::<Lfm2Tiny>(&dir.join("lfm2"), None);
    let qwen = write_production_qwen(&dir.join("qwen"));
    let tok = dir.join("qwen/tokenizer.json");
    let lfm2_args = vec![
        "t".to_string(),
        "--model".into(),
        lfm2.display().to_string(),
    ];
    let qwen_args = vec![
        "t".to_string(),
        "--model".into(),
        qwen.display().to_string(),
        "--tokenizer".into(),
        tok.display().to_string(),
        "--experts".into(),
        "resident".into(),
    ];
    for argv in [lfm2_args, qwen_args] {
        let common = Args::parse_from(&argv).common;
        let options = resolve_options_uninstalled(&common, &lloom_engine::app::no_front).unwrap();
        let mut shape = Shape::of(
            &options,
            lloom_engine::frontend::Frontend::Chat,
            "The loom threads warp and weft into cloth. ".to_string(),
        );
        shape.batch = 1;
        let model = options.load.clone();
        let (engine, _) = load_engine(&model, shape).unwrap();
        let dev = engine.device().clone();
        let Device::Metal(m) = &dev else {
            panic!("the engine loaded on {dev:?}")
        };
        let before = engine.load_warnings();
        assert!(before.is_empty(), "{argv:?}: {before:?}");
        let native = m.emulated_arms().native();
        lloom_tensor::test_utils::allow_emulated(&dev, DType::BF16);
        // What a bf16 weight load asks of the device (`VarBuilder`).
        dev.admit_compute_dtype(DType::BF16).unwrap();
        let after = engine.load_warnings();
        eprintln!(
            "{}: native {native}; after a bf16 admission: {after:?}",
            argv[2]
        );
        if native.iter().any(|f| f.name() == "bf16") {
            assert!(after.is_empty(), "native bf16 is not emulated: {after:?}");
        } else {
            assert_eq!(after.len(), 1, "{after:?}");
            assert_eq!(after[0].kind.name(), "emulated_type");
            assert!(
                after[0].message.starts_with("Metal `bf16` runs emulated"),
                "{after:?}"
            );
        }
        engine.shutdown();
    }
    std::fs::remove_dir_all(&dir).ok();
}
