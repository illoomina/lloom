use lloom_tensor::{DType, Result, Tensor};

struct TmpFile(std::path::PathBuf);

impl TmpFile {
    fn create(base: &str) -> TmpFile {
        let filename = std::env::temp_dir().join(format!(
            "candle-{}-{}-{:?}",
            base,
            std::process::id(),
            std::thread::current().id(),
        ));
        TmpFile(filename)
    }
}

impl std::convert::AsRef<std::path::Path> for TmpFile {
    fn as_ref(&self) -> &std::path::Path {
        self.0.as_path()
    }
}

impl Drop for TmpFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.0).unwrap()
    }
}

#[test]
fn npy() -> Result<()> {
    let npy = Tensor::read_npy("tests/test.npy")?;
    assert_eq!(
        npy.to_dtype(DType::U8)?.to_vec1::<u8>()?,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
    Ok(())
}

#[test]
fn npz() -> Result<()> {
    let npz = Tensor::read_npz("tests/test.npz")?;
    assert_eq!(npz.len(), 2);
    assert_eq!(npz[0].0, "x");
    assert_eq!(npz[1].0, "x_plus_one");
    assert_eq!(
        npz[1].1.to_dtype(DType::U8)?.to_vec1::<u8>()?,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    );
    Ok(())
}

#[test]
fn safetensors() -> Result<()> {
    use lloom_tensor::safetensors::Load;

    let tmp_file = TmpFile::create("st");
    let t = Tensor::arange(0f32, 24f32, &lloom_tensor::Device::Cpu)?;
    t.save_safetensors("t", &tmp_file)?;
    // Load from file.
    let st = lloom_tensor::safetensors::load(&tmp_file, &lloom_tensor::Device::Cpu)?;
    let t2 = st.get("t").unwrap();
    let diff = (&t - t2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff, 0f32);
    // Load from bytes.
    let bytes = std::fs::read(tmp_file)?;
    let st = lloom_tensor::safetensors::SliceSafetensors::new(&bytes)?;
    let t2 = st.get("t").unwrap().load(&lloom_tensor::Device::Cpu);
    let diff = (&t - t2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff, 0f32);
    Ok(())
}

/// Exercise archive writing and every eager/lazy NPZ reader against fixed
/// independent values, not just two ZIP implementations agreeing on bytes.
#[test]
fn npz_roundtrip_all_readers() -> Result<()> {
    use lloom_tensor::{npy::NpzTensors, Device};
    let path = TmpFile::create("npz-roundtrip");
    let a = Tensor::from_vec(vec![1i64, -2, 3, 4], (2, 2), &Device::Cpu)?;
    let b = Tensor::from_vec(vec![0.25f32, -1.5, 2.0], 3, &Device::Cpu)?;
    Tensor::write_npz(&[("a", &a), ("b", &b)], &path)?;
    let eager = Tensor::read_npz(&path)?;
    assert_eq!(
        eager.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
        ["a", "b"]
    );
    assert_eq!(eager[0].1.to_vec2::<i64>()?, [[1, -2], [3, 4]]);
    assert_eq!(eager[1].1.to_vec1::<f32>()?, [0.25, -1.5, 2.0]);
    let selected = Tensor::read_npz_by_name(&path, &["b", "a"])?;
    assert_eq!(selected[0].to_vec1::<f32>()?, [0.25, -1.5, 2.0]);
    assert_eq!(selected[1].to_vec2::<i64>()?, [[1, -2], [3, 4]]);
    assert!(Tensor::read_npz_by_name(&path, &["missing"]).is_err());
    let lazy = NpzTensors::new(&path)?;
    let mut names = lazy
        .names()
        .into_iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    names.sort_unstable();
    assert_eq!(names, ["a", "b"]);
    let (shape, dtype) = lazy.get_shape_and_dtype("a")?;
    assert_eq!(shape.dims(), &[2, 2]);
    assert_eq!(dtype, DType::I64);
    assert_eq!(lazy.get("a")?.unwrap().to_vec2::<i64>()?, [[1, -2], [3, 4]]);
    assert_eq!(lazy.get("b")?.unwrap().to_vec1::<f32>()?, [0.25, -1.5, 2.0]);
    assert!(lazy.get("missing")?.is_none());
    Ok(())
}

/// Existing NumPy-produced fixture is independent of lloom's writer.
#[test]
fn npz_external_fixture_named_and_lazy() -> Result<()> {
    use lloom_tensor::npy::NpzTensors;
    let selected = Tensor::read_npz_by_name("tests/test.npz", &["x_plus_one", "x"])?;
    assert_eq!(
        selected[0].to_vec1::<i64>()?,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    );
    assert_eq!(
        selected[1].to_vec1::<i64>()?,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
    let lazy = NpzTensors::new("tests/test.npz")?;
    assert_eq!(
        lazy.get("x")?.unwrap().to_vec1::<i64>()?,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
    Ok(())
}

/// zip9's fallible name API retains its CP437 fallback for non-UTF-8 bytes.
/// Assert the exact names and tensor values rather than assuming UTF-8 flags
/// make filename decoding strict (upstream deliberately falls back).
#[test]
fn non_utf8_archive_names_preserve_cp437() -> Result<()> {
    let path = TmpFile::create("npz-cp437-name");
    let mut bytes = std::fs::read("tests/test.npz")?;
    let mut changed = 0;
    for i in 0..bytes.len().saturating_sub(46) {
        let (flags, name) = match &bytes[i..i + 4] {
            b"PK\x03\x04" => (i + 6, i + 30),
            b"PK\x01\x02" => (i + 8, i + 46),
            _ => continue,
        };
        bytes[flags + 1] |= 0x08;
        bytes[name] = 0xff; // CP437 non-breaking space, not UTF-8.
        changed += 1;
    }
    assert_eq!(changed, 4, "two local and two central headers");
    std::fs::write(&path, bytes)?;
    let eager = Tensor::read_npz(&path)?;
    assert_eq!(eager[0].0, "\u{00a0}");
    assert_eq!(eager[1].0, "\u{00a0}_plus_one");
    let lazy = lloom_tensor::npy::NpzTensors::new(&path)?;
    assert_eq!(
        lazy.get("\u{00a0}")?.unwrap().to_vec1::<i64>()?,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
    // The same filename enumeration in PTH ignores non-data.pkl entries.
    assert!(lloom_tensor::pickle::read_pth_tensor_info(&path, false, None)?.is_empty());
    Ok(())
}
