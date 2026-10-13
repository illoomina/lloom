//! One typed demand definition for GGUF preflight and component construction.
//! Plans contain metadata only. Their storage still belongs to load admission;
//! successful header validation is not permission to allocate model weights.
use crate::quantized_var_builder::VarBuilder;
use lloom_tensor::quantized::{gguf_file::Content, GgmlDType, QTensor};
use lloom_tensor::{Device, Result, Shape, Tensor};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub(crate) enum WeightDType {
    Any,
    Exact(GgmlDType),
    F32,
    F16OrF32,
}
impl WeightDType {
    fn accepts(self, dtype: GgmlDType) -> bool {
        match self {
            Self::Any => true,
            Self::Exact(expected) => dtype == expected,
            Self::F32 => dtype == GgmlDType::F32,
            Self::F16OrF32 => matches!(dtype, GgmlDType::F16 | GgmlDType::F32),
        }
    }
}

#[derive(Debug)]
pub(crate) struct Weight {
    name: String,
    shape: Shape,
    dtype: WeightDType,
    family: &'static str,
    /// Also accept `shape` without its leading singleton axis, the form a
    /// converter that squeezes `[1, N]` to `[N]` writes.
    squeezed: bool,
}
impl Weight {
    pub fn new(name: impl Into<String>, shape: impl Into<Shape>) -> Self {
        Self {
            name: name.into(),
            shape: shape.into(),
            dtype: WeightDType::Any,
            family: "omni",
            squeezed: false,
        }
    }
    pub fn family(mut self, family: &'static str) -> Self {
        self.family = family;
        self
    }
    /// Resolve the first existing alias once. Invalid present tensors never
    /// become missing tensors and never fall through to another alias.
    pub fn first_present(content: &Content, names: &[String]) -> Result<String> {
        names
            .iter()
            .find(|name| content.tensor_infos.contains_key(*name))
            .cloned()
            .ok_or_else(|| {
                lloom_tensor::Error::msg(format!(
                    "cannot find tensor info for {}",
                    names.join(" | ")
                ))
            })
    }
    /// Reader materialization and header preflight consume this same demand.
    pub fn read<R: std::io::Read + std::io::Seek>(
        &self,
        content: &Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<QTensor> {
        self.validate(content)?;
        let tensor = content.tensor(reader, &self.name, device)?;
        self.check(tensor.shape(), tensor.dtype())?;
        Ok(tensor)
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn dtype(mut self, dtype: WeightDType) -> Self {
        self.dtype = dtype;
        self
    }
    pub fn validate(&self, content: &Content) -> Result<()> {
        let info = content.tensor_infos.get(&self.name).ok_or_else(|| {
            lloom_tensor::Error::Msg(format!(
                "missing {} GGUF tensor header {}",
                self.family, self.name
            ))
        })?;
        self.check(&info.shape, info.ggml_dtype)
    }
    /// [`Self::squeezed`]: the declared shape or its leading singleton dropped.
    pub fn squeezed(mut self) -> Self {
        self.squeezed = true;
        self
    }
    fn shape_accepted(&self, shape: &Shape) -> bool {
        shape == &self.shape
            || (self.squeezed
                && matches!(self.shape.dims(), [1, rest @ ..] if shape.dims() == rest))
    }
    fn check(&self, shape: &Shape, dtype: GgmlDType) -> Result<()> {
        if !self.shape_accepted(shape) {
            lloom_tensor::bail!(
                "{} GGUF {} shape mismatch: expected {:?}, got {:?}",
                self.family,
                self.name,
                self.shape,
                shape
            )
        }
        if !self.dtype.accepts(dtype) {
            lloom_tensor::bail!(
                "{} GGUF {} requires {:?}, got {:?}",
                self.family,
                self.name,
                self.dtype,
                dtype
            )
        }
        Ok(())
    }
    pub fn get(&self, vb: &VarBuilder) -> Result<Arc<QTensor>> {
        let tensor = if self.squeezed {
            vb.get_no_shape(&self.name)?
        } else {
            vb.get(self.shape.clone(), &self.name)?
        };
        self.check(tensor.shape(), tensor.dtype())?;
        Ok(tensor)
    }
    pub fn dequantize(&self, vb: &VarBuilder) -> Result<Tensor> {
        self.get(vb)?.dequantize(vb.device())
    }
}

pub(crate) trait WeightPlan {
    fn visit(&self, visitor: &mut impl FnMut(&Weight) -> Result<()>) -> Result<()>;
    fn validate_headers(&self, content: &Content) -> Result<()> {
        self.visit(&mut |weight| weight.validate(content))
    }
}
impl WeightPlan for Weight {
    fn visit(&self, visitor: &mut impl FnMut(&Weight) -> Result<()>) -> Result<()> {
        visitor(self)
    }
}
impl<T: WeightPlan> WeightPlan for [T] {
    fn visit(&self, visitor: &mut impl FnMut(&Weight) -> Result<()>) -> Result<()> {
        for item in self {
            item.visit(visitor)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct LinearWeights {
    weight: Weight,
    bias: Option<Weight>,
}
impl LinearWeights {
    pub fn new(prefix: &str, input: usize, output: usize, bias: bool) -> Self {
        Self {
            weight: Weight::new(format!("{prefix}.weight"), (output, input)),
            bias: bias.then(|| Weight::new(format!("{prefix}.bias"), output)),
        }
    }
    /// A one-output projection whose weight a converter may store squeezed
    /// (`[N]` for `[1, N]`).
    pub fn squeezed(mut self) -> Self {
        self.weight = self.weight.squeezed();
        self
    }
    pub fn load(&self, vb: &VarBuilder) -> Result<crate::quantized_nn::Linear> {
        // Match quantized_nn::linear_b: bias dequantization precedes QMatMul.
        let bias = self
            .bias
            .as_ref()
            .map(|weight| weight.dequantize(vb))
            .transpose()?;
        let mut weight = self.weight.get(vb)?;
        if weight.shape() != &self.weight.shape {
            // A squeezed weight takes its singleton axis back, exactly (F32).
            let restored = weight
                .dequantize(vb.device())?
                .reshape(self.weight.shape.clone())?;
            weight = Arc::new(QTensor::quantize(&restored, GgmlDType::F32)?);
        }
        crate::quantized_nn::Linear::from_arc(weight, bias)
    }
}
impl WeightPlan for LinearWeights {
    fn visit(&self, visitor: &mut impl FnMut(&Weight) -> Result<()>) -> Result<()> {
        if let Some(bias) = &self.bias {
            bias.visit(visitor)?;
        }
        self.weight.visit(visitor)
    }
}

#[derive(Debug)]
pub(crate) struct NormWeights {
    weight: Weight,
    bias: Weight,
}
impl NormWeights {
    pub fn new(prefix: &str, width: usize) -> Self {
        Self {
            weight: Weight::new(format!("{prefix}.weight"), width),
            bias: Weight::new(format!("{prefix}.bias"), width),
        }
    }
    pub fn load(&self, vb: &VarBuilder, epsilon: f64) -> Result<crate::nn::LayerNorm> {
        Ok(crate::nn::LayerNorm::new(
            self.weight.dequantize(vb)?,
            self.bias.dequantize(vb)?,
            epsilon,
        ))
    }
}
impl WeightPlan for NormWeights {
    fn visit(&self, visitor: &mut impl FnMut(&Weight) -> Result<()>) -> Result<()> {
        self.weight.visit(visitor)?;
        self.bias.visit(visitor)
    }
}

#[cfg(test)]
pub(crate) fn assert_fixture_demands(
    plan: &impl WeightPlan,
    mut content: Content,
    count: usize,
) -> Result<()> {
    use std::collections::BTreeSet;
    let mut names = BTreeSet::new();
    plan.visit(&mut |weight| {
        assert!(
            names.insert(weight.name.clone()),
            "duplicate demand {}",
            weight.name
        );
        Ok(())
    })?;
    assert_eq!(names.len(), count);
    assert_eq!(content.tensor_infos.len(), count);
    assert_eq!(names, content.tensor_infos.keys().cloned().collect());
    plan.validate_headers(&content)?;
    for name in names {
        let info = content.tensor_infos.remove(&name).unwrap();
        let error = plan.validate_headers(&content).unwrap_err().to_string();
        assert!(error.contains(&name), "missing-demand refusal: {error}");
        content.tensor_infos.insert(name, info);
    }
    eprintln!("GGUF_DEMAND_CENSUS_COMPLETE={count}");
    Ok(())
}

#[cfg(test)]
mod squeezed_tests {
    //! llama.cpp's converter writes d1-omni's `[1, D]` scorer weight as `[D]`.
    use super::*;
    use lloom_tensor::quantized::gguf_file;
    use lloom_tensor::Module;

    fn file(weight: &Tensor) -> Result<Vec<u8>> {
        let w = QTensor::quantize(weight, GgmlDType::F32)?;
        let b = QTensor::quantize(&Tensor::new(&[0.25f32], &Device::Cpu)?, GgmlDType::F32)?;
        let mut out = std::io::Cursor::new(Vec::new());
        gguf_file::write(&mut out, &[], &[("p.weight", &w), ("p.bias", &b)])?;
        Ok(out.into_inner())
    }

    #[test]
    fn a_squeezed_weight_loads_as_its_declared_shape() -> Result<()> {
        let row: Vec<f32> = (0..4).map(|i| i as f32 - 1.5).collect();
        let x = Tensor::new(&[[2f32, -1., 0.5, 3.]], &Device::Cpu)?;
        let mut outputs = Vec::new();
        for weight in [
            Tensor::new(row.as_slice(), &Device::Cpu)?,
            Tensor::new(row.as_slice(), &Device::Cpu)?.reshape((1, 4))?,
        ] {
            let bytes = file(&weight)?;
            let content = Content::read(&mut std::io::Cursor::new(&bytes))?;
            let plan = LinearWeights::new("p", 4, 1, true).squeezed();
            plan.validate_headers(&content)?;
            let vb = VarBuilder::from_gguf_buffer(&bytes, &Device::Cpu)?;
            outputs.push(plan.load(&vb)?.forward(&x)?.to_vec2::<f32>()?);
        }
        // 2·-1.5 + -1·-0.5 + 0.5·0.5 + 3·1.5 = 2.25, plus the 0.25 bias.
        assert_eq!(outputs, vec![vec![vec![2.5f32]]; 2]);
        let bytes = file(&Tensor::new(row.as_slice(), &Device::Cpu)?)?;
        let content = Content::read(&mut std::io::Cursor::new(&bytes))?;
        let strict = LinearWeights::new("p", 4, 1, true).validate_headers(&content);
        assert!(strict.is_err(), "an unsqueezable demand accepted [N]");
        Ok(())
    }
}
