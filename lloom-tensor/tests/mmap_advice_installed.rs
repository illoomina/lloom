//! `mmap_advice = none` from an installed tuning group disables the advice.
//!
//! This is the path `lloom --mmap-advice none` takes: the flag is parsed into
//! the group, the group is installed before anything reads it, and the
//! loader resolves its arm from it with `CANDLE_MMAP_ADVICE` unset. Its own
//! binary, because it installs the process's tuning group (`DESIGN.md`
//! §15.1n).

use lloom_tensor::safetensors::{MmapAdvice, MmapedSafetensors, MMAP_ADVICE_CALLS};
use std::sync::atomic::Ordering;

#[test]
fn an_installed_none_makes_no_madvise_call() {
    // The flag path, not the env alias: with the variable set, `none` would
    // reach the loader through it instead.
    std::env::remove_var("CANDLE_MMAP_ADVICE");

    let mut tuning = lloom_tensor::tuning::Tuning::default();
    tuning.set("cpu.mmap_advice", "none").unwrap();
    assert!(
        lloom_tensor::tuning::install(tuning).is_ok(),
        "the tuning was read before this test installed it"
    );

    assert_eq!(
        MmapAdvice::from_env().unwrap(),
        MmapAdvice::None,
        "`mmap_advice = none` from the installed group selected an advised arm"
    );

    let bytes = b"8\0\0\0\0\0\0\0{\"x\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}   \x01\x03";
    let path = std::env::temp_dir().join(format!(
        "mmap_advice_installed_{}.safetensors",
        std::process::id()
    ));
    std::fs::write(&path, bytes).unwrap();
    let before = MMAP_ADVICE_CALLS.load(Ordering::Relaxed);
    let m = unsafe { MmapedSafetensors::new(&path) }.unwrap();
    assert_eq!(
        MMAP_ADVICE_CALLS.load(Ordering::Relaxed),
        before,
        "the loader advised a mapping under `mmap_advice = none`"
    );
    let t = m.load("x", &lloom_tensor::Device::Cpu).unwrap();
    assert_eq!(t.to_vec1::<u8>().unwrap(), vec![1, 3]);
    std::fs::remove_file(&path).unwrap();
}
