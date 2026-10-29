//! Embedding Layer.
use lloom_tensor::{Result, Tensor};

#[derive(Clone, Debug)]
pub struct Embedding {
    embeddings: Tensor,
    hidden_size: usize,
}

impl Embedding {
    pub fn new(embeddings: Tensor, hidden_size: usize) -> Self {
        Self {
            embeddings,
            hidden_size,
        }
    }

    pub fn embeddings(&self) -> &Tensor {
        &self.embeddings
    }

    #[cfg(feature = "metal")]
    pub fn guarded_layout(&self) -> Result<lloom_tensor::quantized::GuardedEmbeddingLayout> {
        lloom_tensor::quantized::QMatMul::Tensor(self.embeddings.clone()).guarded_embedding_layout()
    }

    #[cfg(feature = "metal")]
    pub fn check_guarded_output(&self, out: &Tensor) -> Result<()> {
        // Cold-only descriptor: reuse the dense/packed prepared-output law.
        lloom_tensor::quantized::QMatMul::Tensor(self.embeddings.clone())
            .check_guarded_embedding_output(out)
    }

    #[cfg(feature = "metal")]
    pub fn forward_guarded_into(
        &self,
        tokens: &lloom_tensor::metal_backend::sampling::GuardedSample<'_>,
        out: &Tensor,
    ) -> Result<()> {
        tokens.embedding_into_tensor(&self.embeddings, out)
    }

    /// Get the hidden size of the embedding matrix
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl crate::nn::Module for Embedding {
    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = indexes.flatten_all()?;
        let values = self.embeddings.index_select(&indexes, 0)?;
        let values = values.reshape(final_dims)?;
        Ok(values)
    }
}

pub fn embedding(in_size: usize, out_size: usize, vb: crate::nn::VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get_with_hints(
        (in_size, out_size),
        "weight",
        crate::nn::Init::Randn {
            mean: 0.,
            stdev: 1.,
        },
    )?;
    Ok(Embedding::new(embeddings, out_size))
}
