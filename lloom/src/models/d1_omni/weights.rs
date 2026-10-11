//! One typed demand definition for GGUF preflight and component construction.
//! Plans contain metadata only. Their storage still belongs to load admission;
//! successful header validation is not permission to allocate model weights.
use crate::quantized_var_builder::VarBuilder;
use lloom_tensor::quantized::{gguf_file::Content, GgmlDType, QTensor};
use lloom_tensor::{Result, Shape, Tensor};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub(super) enum WeightDType {
    Any,
    F32,
    F16OrF32,
}
impl WeightDType {
    fn accepts(self, dtype: GgmlDType) -> bool {
        match self {
            Self::Any => true,
            Self::F32 => dtype == GgmlDType::F32,
            Self::F16OrF32 => matches!(dtype, GgmlDType::F16 | GgmlDType::F32),
        }
    }
}

#[derive(Debug)]
pub(super) struct Weight {
    name: String,
    shape: Shape,
    dtype: WeightDType,
}
impl Weight {
    pub fn new(name: impl Into<String>, shape: impl Into<Shape>) -> Self {
        Self {
            name: name.into(),
            shape: shape.into(),
            dtype: WeightDType::Any,
        }
    }
    pub fn dtype(mut self, dtype: WeightDType) -> Self {
        self.dtype = dtype;
        self
    }
    pub fn validate(&self, content: &Content) -> Result<()> {
        let info = content.tensor_infos.get(&self.name).ok_or_else(|| {
            lloom_tensor::Error::Msg(format!("missing omni GGUF tensor header {}", self.name))
        })?;
        self.check(&info.shape, info.ggml_dtype)
    }
    fn check(&self, shape: &Shape, dtype: GgmlDType) -> Result<()> {
        if shape != &self.shape {
            lloom_tensor::bail!(
                "omni GGUF {} shape mismatch: expected {:?}, got {:?}",
                self.name,
                self.shape,
                shape
            )
        }
        if !self.dtype.accepts(dtype) {
            lloom_tensor::bail!(
                "omni GGUF {} requires {:?}, got {:?}",
                self.name,
                self.dtype,
                dtype
            )
        }
        Ok(())
    }
    pub fn get(&self, vb: &VarBuilder) -> Result<Arc<QTensor>> {
        let tensor = vb.get(self.shape.clone(), &self.name)?;
        self.check(tensor.shape(), tensor.dtype())?;
        Ok(tensor)
    }
    pub fn dequantize(&self, vb: &VarBuilder) -> Result<Tensor> {
        self.get(vb)?.dequantize(vb.device())
    }
}

pub(super) trait WeightPlan {
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
pub(super) struct LinearWeights {
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
    pub fn load(&self, vb: &VarBuilder) -> Result<crate::quantized_nn::Linear> {
        // Match quantized_nn::linear_b: bias dequantization precedes QMatMul.
        let bias = self
            .bias
            .as_ref()
            .map(|weight| weight.dequantize(vb))
            .transpose()?;
        crate::quantized_nn::Linear::from_arc(self.weight.get(vb)?, bias)
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
pub(super) struct NormWeights {
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
pub(super) fn assert_fixture_demands(
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
