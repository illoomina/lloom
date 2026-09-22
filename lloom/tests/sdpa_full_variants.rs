//! Every steel full-attention variant a model here dispatches builds and runs
//! on this device (f16 / f32 at 64 / 128 / 256), with the causal offset
//! against a cached prefix. The f32 arm at 256 is the reduced 8x8 tile
//! (`sdpa_full_tile`): the 32x16 tile's f32 Q block alone is the whole 32KB
//! of threadgroup memory, and its pipeline does not build.
#![cfg(feature = "metal")]
use lloom::nn::ops;
use lloom_tensor::{DType, Device, Tensor};

#[test]
fn steel_full_attention_variants() {
    let dev = Device::new_metal(0).unwrap();
    for &dt in &[DType::F16, DType::F32] {
        for &hd in &[64usize, 128, 256] {
            let (nh, nkv, seq, kv_len) = (4usize, 2usize, 8usize, 40usize);
            let q = Tensor::randn(0f32, 1.0, (1, nh, seq, hd), &dev)
                .unwrap()
                .to_dtype(dt)
                .unwrap();
            let k = Tensor::randn(0f32, 1.0, (1, nkv, kv_len, hd), &dev)
                .unwrap()
                .to_dtype(dt)
                .unwrap();
            let v = Tensor::randn(0f32, 1.0, (1, nkv, kv_len, hd), &dev)
                .unwrap()
                .to_dtype(dt)
                .unwrap();
            let r = ops::sdpa(&q, &k, &v, None, true, 0.0625, 1.0)
                .and_then(|o| o.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>());
            match r {
                Ok(o) => {
                    println!("VARIANT dtype={dt:?} hd={hd} ok");
                    assert!(o.iter().all(|x| x.is_finite()), "{dt:?}/{hd}: non-finite");
                }
                Err(e) => panic!("VARIANT dtype={dt:?} hd={hd} does not run: {e}"),
            }
        }
    }
}
