//! Varbuilder for Loading gguf files
//!
//! VarBuilder is a utility to store quantized tensors from a [GGUF model file](https://huggingface.co/docs/hub/gguf).
//! These tensors can be loaded from disk using `from_gguf` or from an in-memory
//! buffer using `from_gguf_buffer`.

use lloom_tensor::quantized::QTensor;
use lloom_tensor::{Device, Result, Shape};
use std::sync::Arc;

// VarBuilder specialized for QTensors
#[derive(Clone)]
pub struct VarBuilder {
    data: Arc<std::collections::HashMap<String, Arc<QTensor>>>,
    path: Vec<String>,
    device: Device,
}

impl VarBuilder {
    pub fn from_gguf<P: AsRef<std::path::Path>>(p: P, device: &Device) -> Result<Self> {
        let mut file = std::fs::File::open(p)?;
        let content = lloom_tensor::quantized::gguf_file::Content::read(&mut file)?;
        Self::from_gguf_content(&content, &mut file, device)
    }

    pub fn from_gguf_buffer(buffer: &[u8], device: &Device) -> Result<Self> {
        let mut cursor = std::io::Cursor::new(buffer);
        let content = lloom_tensor::quantized::gguf_file::Content::read(&mut cursor)?;
        Self::from_gguf_content(&content, &mut cursor, device)
    }

    /// Reuse an inspected header while preserving the existing eager tensor
    /// materialization path. `reader` must contain the same bytes/layout from
    /// which `content` was parsed, including its tensor-data offsets.
    ///
    /// The caller enters its load allocation scope BEFORE this call. Parsing
    /// headers does not enroll HashMap/String/Arc or tensor-transfer scratch;
    /// this API is parser reuse, not a complete admission policy.
    pub fn from_gguf_content<R: std::io::Read + std::io::Seek>(
        content: &lloom_tensor::quantized::gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut data = std::collections::HashMap::new();
        for tensor_name in content.tensor_infos.keys() {
            let tensor = content.tensor(reader, tensor_name, device)?;
            data.insert(tensor_name.to_string(), Arc::new(tensor));
        }
        Ok(Self {
            data: Arc::new(data),
            path: Vec::new(),
            device: device.clone(),
        })
    }

    pub fn pp<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            data: self.data.clone(),
            path,
            device: self.device.clone(),
        }
    }

    fn path(&self, tensor_name: &str) -> String {
        if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            [&self.path.join("."), tensor_name].join(".")
        }
    }

    pub fn get<S: Into<Shape>>(&self, s: S, name: &str) -> Result<Arc<QTensor>> {
        let path = self.path(name);
        match self.data.get(&path) {
            None => {
                lloom_tensor::bail!("cannot find tensor {path}")
            }
            Some(qtensor) => {
                let shape = s.into();
                if qtensor.shape() != &shape {
                    lloom_tensor::bail!(
                        "shape mismatch for {name}, got {:?}, expected {shape:?}",
                        qtensor.shape()
                    )
                }
                Ok(qtensor.clone())
            }
        }
    }

    pub fn get_no_shape(&self, name: &str) -> Result<Arc<QTensor>> {
        let path = self.path(name);
        match self.data.get(&path) {
            None => {
                lloom_tensor::bail!("cannot find tensor {name}")
            }
            Some(qtensor) => Ok(qtensor.clone()),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom_tensor::quantized::{gguf_file, GgmlDType};
    use lloom_tensor::Tensor;
    use std::io::{Cursor, Seek, SeekFrom};

    #[test]
    fn inspected_header_reuses_reader_and_matches_buffer_constructor() -> Result<()> {
        let device = Device::Cpu;
        let tensor = Tensor::new(&[[1f32, -2.], [3., 4.5]], &device)?;
        let quantized = QTensor::quantize(&tensor, GgmlDType::F32)?;
        let mut encoded = Cursor::new(Vec::new());
        gguf_file::write(&mut encoded, &[], &[("probe.weight", &quantized)])?;
        assert!(encoded.get_ref().len() < 512);
        let bytes = encoded.into_inner();
        let mut inspected = Cursor::new(bytes.as_slice());
        let content = gguf_file::Content::read(&mut inspected)?;
        // Tensor reads honor the content's offset, not the reader's incidental
        // post-inspection position. No header reread or alternate tensor path.
        inspected.seek(SeekFrom::End(0))?;
        let reused = VarBuilder::from_gguf_content(&content, &mut inspected, &device)?;
        let incumbent = VarBuilder::from_gguf_buffer(&bytes, &device)?;
        for builder in [&reused, &incumbent] {
            assert_eq!(
                builder
                    .pp("probe")
                    .get((2, 2), "weight")?
                    .dequantize(&device)?
                    .to_vec2::<f32>()?,
                vec![vec![1., -2.], vec![3., 4.5]]
            );
            assert!(builder.get((4,), "probe.weight").is_err());
            assert!(builder.get_no_shape("absent").is_err());
        }
        // GGUF pads this16-byte payload to32; cut the payload, not only padding.
        let payload_end = content.tensor_data_offset as usize + 4 * std::mem::size_of::<f32>();
        assert!(payload_end <= bytes.len());
        let mut truncated = Cursor::new(&bytes[..payload_end - 1]);
        let error = VarBuilder::from_gguf_content(&content, &mut truncated, &device)
            .err()
            .expect("truncated payload must refuse")
            .to_string();
        assert!(error.contains("only 15 remaining"), "{error}");
        Ok(())
    }
}
