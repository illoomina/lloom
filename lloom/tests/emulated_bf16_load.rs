#![cfg(feature = "metal")]
//! A model loaded at a dtype the GPU only emulates is refused at its first
//! weight, before any buffer, unless the operator opted in (`DESIGN.md`
//! §15.2a). `VarBuilder` asks the device (`Device::admit_compute_dtype`) for
//! the dtype every weight is loaded at, which is the dtype the model computes
//! in.

use lloom::nn::{linear_no_bias, Module, VarBuilder};
use lloom_tensor::kernels::metal::arms::{Feature, Features};
use lloom_tensor::{DType, Device, Result, Tensor};
use std::collections::HashMap;

fn weights() -> Result<HashMap<String, Tensor>> {
    let w = Tensor::new(&[[1f32, 2.], [3., 4.], [0.5, -1.]], &Device::Cpu)?;
    Ok(HashMap::from([("proj.weight".to_string(), w)]))
}

fn forward(vb: VarBuilder) -> Result<Vec<f32>> {
    let dev = vb.device().clone();
    let dtype = vb.dtype();
    let proj = linear_no_bias(2, 3, vb.pp("proj"))?;
    let x = Tensor::new(&[[1f32, -1.]], &dev)?.to_dtype(dtype)?;
    proj.forward(&x)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
}

#[test]
fn a_bf16_load_is_refused_at_the_first_weight_and_admitted_by_the_opt_in() -> Result<()> {
    assert!(
        lloom_tensor::tuning::tuning()
            .metal
            .allow_emulated
            .is_none(),
        "run without LLOOM_METAL_ALLOW_EMULATED: this test is the default arm"
    );
    let dev = Device::new_metal(0)?;
    let metal = dev.as_metal_device()?;
    let native = metal.emulated_arms().native().has(Feature::Bf16);

    // f16 is native: loads and runs without the opt-in and warns nothing.
    let f16 = forward(VarBuilder::from_tensors(weights()?, DType::F16, &dev))?;
    assert_eq!(f16, vec![-1.0, -1.0, 1.5]);

    let refused = forward(VarBuilder::from_tensors(weights()?, DType::BF16, &dev));
    if native {
        assert_eq!(refused?, f16);
    } else {
        let e = refused.unwrap_err().to_string();
        assert!(e.contains("Metal arm `computing at `bf16`` refused"), "{e}");
        assert!(e.contains("`--allow-emulated bf16`"), "{e}");
    }
    assert_eq!(metal.emulated_arms().warnings(), 0);

    metal.allow_emulated(Features::empty().with(Feature::Bf16));
    let bf16 = forward(VarBuilder::from_tensors(weights()?, DType::BF16, &dev))?;
    assert_eq!(bf16, f16);
    let again = forward(VarBuilder::from_tensors(weights()?, DType::BF16, &dev))?;
    assert_eq!(again, f16);
    assert_eq!(metal.emulated_arms().warnings(), if native { 0 } else { 1 });
    Ok(())
}
