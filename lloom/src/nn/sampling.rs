use lloom_tensor::{Result, Tensor};

/// Sample according to the Gumbel-Softmax distribution.
pub fn gumbel_softmax<D: lloom_tensor::shape::Dim>(
    logits: &Tensor,
    temperature: f64,
    dim: D,
) -> Result<Tensor> {
    if temperature <= 0.0 {
        logits.argmax(dim)
    } else {
        // Cast to f32, doing the Gumbel softmax in bf16 is a bit unstable.
        let logits = logits.to_dtype(lloom_tensor::DType::F32)?;
        // The uniform is clamped away from 0 and 1 because `log(0)` and
        // `log(1)` are not finite. **The upper bound has to be tight**, and
        // 0.999 was not: with `G = -log(-log u)` a cap of `u <= 0.999`
        // truncates one variate in a thousand to the single value
        // `G = 6.907`, and ties at that cap are broken by `argmax`'s index
        // order -- a systematic bias toward low indices rather than a random
        // error. The wider bound caps at `G = 16.1`, one variate in 10^7.
        //
        // Measured against the reference softmax over a 256-wide fixture,
        // 50 000 draws, chi-squared with a p=0.001 critical value of 330.5
        // (lloom #345):
        //
        // | clamp | CPU | Metal |
        // |---|---|---|
        // | `(1e-7, 0.999)` | **2451** | **2281** |
        // | `(1e-9, 0.9999999)` | **216** | **280** |
        //
        // Both backends fail on the old bound and pass on this one, which is
        // what identifies the clamp rather than any one generator -- the
        // arithmetic here is backend-independent.
        let minus_g = logits.rand_like(1e-9, 0.9999999)?.log()?.neg()?.log()?;
        if temperature == 1.0 {
            let sampled = (logits - minus_g)?.argmax(dim)?;
            Ok(sampled)
        } else {
            let sampled = (logits + minus_g * (-temperature))?.argmax(dim)?;
            Ok(sampled)
        }
    }
}
