//! The existing LFM fixture machinery with the mixed FFN composition selected.
use crate::models::lfm2::{Config, Model};
use crate::models::lfm2_testing::Lfm2Tiny;
use crate::profile::Profile;
#[cfg(not(target_arch = "wasm32"))]
use crate::testing::fixture::{Scaled, Source};
use crate::testing::tiny::{Schema, TinyCheckpoint, TinyFamily, TinyTokenizer, Weights, Workload};
use lloom_tensor::quantized::GgmlDType;
use lloom_tensor::{DType, Device, Result};
use std::path::{Path, PathBuf};

pub struct Lfm2MoeTiny;

fn common_fixture(c: &TinyCheckpoint<Lfm2MoeTiny>) -> TinyCheckpoint<Lfm2Tiny> {
    let mut out = TinyCheckpoint::new(c.cfg.clone(), c.weights);
    out.dtype = c.dtype;
    out
}

impl TinyFamily for Lfm2MoeTiny {
    type Config = Config;
    type Model<P: Profile> = Model<P>;
    const NAME: &'static str = "lfm2_moe";
    const DENSE_KINDS: &'static [GgmlDType] = &[GgmlDType::F32];
    const EXPERT_KINDS: &'static [GgmlDType] = &[GgmlDType::F32];

    fn template() -> Config {
        let mut cfg = Lfm2Tiny::template();
        cfg.moe = Some(super::config::MoeConfig {
            num_dense_layers: 1,
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 32,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
            use_expert_bias: true,
        });
        cfg
    }
    fn covers(template: &Config, cfg: &Config) -> bool {
        Lfm2Tiny::covers(template, cfg)
            && cfg
                .moe
                .is_some_and(|m| m.num_dense_layers < cfg.num_hidden_layers)
    }
    fn tokenizer() -> TinyTokenizer {
        Lfm2Tiny::tokenizer()
    }
    fn schema(ckpt: &TinyCheckpoint<Self>) -> Result<Schema> {
        let mut schema = Lfm2Tiny::schema(&common_fixture(ckpt))?;
        for t in &mut schema.tensors {
            if t.name.ends_with(".expert_bias") {
                t.dtype = GgmlDType::F32;
            }
        }
        Ok(schema)
    }
    fn workloads(cfg: &Config) -> Vec<Workload> {
        Lfm2Tiny::workloads(cfg)
    }
    fn build<P: Profile>(ckpt: &TinyCheckpoint<Self>, dev: &Device) -> Result<Model<P>> {
        Lfm2Tiny::build::<P>(&common_fixture(ckpt), dev)
    }
    fn reference<P: Profile>(ckpt: &TinyCheckpoint<Self>, dev: &Device) -> Result<Model<P>> {
        Lfm2Tiny::reference::<P>(&common_fixture(ckpt), dev)
    }
    fn write(ckpt: &TinyCheckpoint<Self>, dir: &Path) -> Result<PathBuf> {
        Lfm2Tiny::write(&common_fixture(ckpt), dir)
    }
    fn load<P: Profile>(path: &Path, dev: &Device) -> Result<Model<P>> {
        Lfm2Tiny::load::<P>(path, dev)
    }
    const ARMS: &'static [&'static str] = Lfm2Tiny::ARMS;
    fn set_arm(cfg: &mut Config, arm: usize) {
        Lfm2Tiny::set_arm(cfg, arm)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Scaled for Lfm2MoeTiny {
    const CHECKPOINT_ENV: &'static str = "LFM2_MOE_MODEL_DIR";
    const REPO: (&'static str, &'static str) = ("LiquidAI/LFM2.5-24B-A2B", "config.json");
    fn checkpoint_of(file: PathBuf) -> PathBuf {
        file.parent().map(Path::to_path_buf).unwrap_or(file)
    }
    fn scale_config(source: &Source) -> Result<Config> {
        let bytes = match source {
            Source::Real(dir) => std::fs::read(dir.join("config.json"))?,
            Source::Synthetic => {
                include_bytes!("../../../tests/fixtures/lfm2_moe/config.json").to_vec()
            }
        };
        super::config::parse_lfm_config(
            serde_json::from_slice(&bytes).map_err(lloom_tensor::Error::wrap)?,
        )
    }
    fn scale_model<P: Profile>(
        source: &Source,
        cfg: &Config,
        dtype: Option<DType>,
        weights: &Weights,
        dev: &Device,
    ) -> Result<Model<P>> {
        // The LFM adapter already distinguishes mapped files from synthetic
        // keyed tensors. The geometry here carries the MoE composition.
        <Lfm2Tiny as Scaled>::scale_model::<P>(source, cfg, dtype, weights, dev)
    }
}
