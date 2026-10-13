//! Same complete omni Engine+Attention custody behind the narrow HTTP seam.
use super::*;
use crate::app::decision_endpoint::{DecisionEndpoint, Slot};
use serde_json::json;

pub struct Endpoint<Q, R, T, H, V, A> {
    name: String,
    slot: Slot<(Engine<Q, R>, Attention<T, H, V, A>)>,
}
impl<Q: Predictor, R: Recovery> Engine<Q, R> {
    /// Infallible ownership transfer: no second model/tokenizer/domain. The
    /// provenance label and callback values are already caller-owned. No worker
    /// starts here; native constructor/drop refusal custody remains unproven.
    pub fn into_endpoint<T, H, V, A>(
        self,
        name: String,
        attention: Attention<T, H, V, A>,
    ) -> Endpoint<Q, R, T, H, V, A> {
        Endpoint {
            name,
            slot: Slot::new((self, attention)),
        }
    }
    fn endpoint_health(&self, name: &str) -> Result<serde_json::Value> {
        let predicted = self
            .predictor
            .predicted_extra(Request::OmniHealth { model_name: name })?;
        self.owner.with_pass(predicted, |model, _| {
            let facts = model.metadata();
            Ok(json!({"model":name,"family":"d1-omni","generation":false,
                "decisions":true,"raw_encoded_media":true,
                "max_length":facts.max_length,"vocab_size":facts.vocab_size}))
        })
    }
}
impl<Q, R, T, H, V, A> DecisionEndpoint for Endpoint<Q, R, T, H, V, A>
where
    Q: Predictor + Send,
    R: Recovery + Send,
    T: FnMut(&Tensor, &Tensor, &Tensor, &TrunkBatch, f64) -> Result<Tensor> + Send,
    H: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor> + Send,
    V: FnMut(&Tensor, &Tensor, &Tensor, Option<&Tensor>, f64) -> Result<Tensor> + Send,
    A: FnMut(RelativeInputs<'_>) -> Result<Tensor> + Send,
{
    fn model_name(&self) -> &str {
        &self.name
    }
    fn input_modalities(&self) -> &'static [&'static str] {
        &["text", "image", "audio"]
    }
    fn health_checked(&self) -> anyhow::Result<serde_json::Value> {
        let held = self.slot.checkout()?;
        held.0
            .endpoint_health(&self.name)
            .map_err(anyhow::Error::from)
    }
    fn systemone(
        &self,
        body: &OrderedJson,
    ) -> std::result::Result<SystemOneResponse, SystemOneError> {
        let mut held = self
            .slot
            .checkout()
            .map_err(|error| SystemOneError::Execution(error.to_string()))?;
        let (engine, attention) = &mut *held;
        engine.systemone_media(body, attention, crate::app::lfm2::SERVED_DECISION_IMAGES)
    }
    fn shutdown(&self) {
        // This adapter creates no workers. The sequential decision listener
        // returns only after its current request has returned. Neither fact is
        // native DATA/callback/retirement or failed-construction custody proof.
    }
}
