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
}
impl Weight {
    pub fn new(name: impl Into<String>, shape: impl Into<Shape>) -> Self {
        Self {
            name: name.into(),
            shape: shape.into(),
            dtype: WeightDType::Any,
            family: "omni",
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
    #[cfg(test)]
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
    fn check(&self, shape: &Shape, dtype: GgmlDType) -> Result<()> {
        if shape != &self.shape {
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
        let tensor = vb.get(self.shape.clone(), &self.name)?;
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
