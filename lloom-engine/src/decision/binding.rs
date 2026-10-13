//! Which decision template a loaded model answers `/v1/systemone` with.
//!
//! A native decision checkpoint carries its binding: a GGUF has the named
//! `systemone` chat template, `<arch>.decision.type` and its calibration
//! temperatures; a safetensors load gets the same template text from the
//! engine, pinned by hash per decision type. A plain causal model has no
//! binding unless one is selected (`decision.template`): `generic` reads one
//! label token per option through the model's own chat template, uncalibrated.
//! Nothing is selected implicitly.
use super::{Question, SystemOneError, SystemOneTemplate, SystemOneTemplateKind, Temperature};
use lloom_tensor::{bail, Error, Result};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// The decision architectures the engine can bind natively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionType {
    /// LiquidAI d1-3B: the LFM2-VL causal readout over option-label tokens.
    Lfm2D1,
    /// LiquidAI d1-omni: the bidirectional trunk and its decision head.
    Lfm2D1Omni,
}

impl DecisionType {
    /// The `<arch>.decision.type` value llama.cpp's converter writes.
    pub fn name(self) -> &'static str {
        match self {
            Self::Lfm2D1 => "lfm2-d1",
            Self::Lfm2D1Omni => "lfm2-d1-omni",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        [Self::Lfm2D1, Self::Lfm2D1Omni]
            .into_iter()
            .find(|kind| kind.name() == name)
    }

    pub fn template_kind(self) -> SystemOneTemplateKind {
        match self {
            Self::Lfm2D1 => SystemOneTemplateKind::D1Causal,
            Self::Lfm2D1Omni => SystemOneTemplateKind::D1Omni,
        }
    }

    /// Whether the model also generates tokens (`/v1/models` output modalities).
    pub fn generates(self) -> bool {
        matches!(self, Self::Lfm2D1)
    }
}

/// A template the engine ships as data, byte-identical to llama.cpp's
/// converter output for its decision type and pinned by SHA-256.
pub struct ShippedTemplate {
    pub decision_type: DecisionType,
    pub source: &'static str,
    pub sha256: &'static str,
}

impl ShippedTemplate {
    pub fn name(&self) -> &'static str {
        self.decision_type.name()
    }

    pub fn compile(&self) -> Result<SystemOneTemplate> {
        SystemOneTemplate::compile(self.source, self.decision_type.template_kind())
    }
}

/// llama.cpp `conversion/lfm2.py` `D1Model._systemone_template` and
/// `D1OmniModel._systemone_template` at `77757a3`; provenance in
/// `fixtures/systemone-template-provenance.json`.
pub const SHIPPED: [ShippedTemplate; 2] = [
    ShippedTemplate {
        decision_type: DecisionType::Lfm2D1,
        source: include_str!("fixtures/d1-systemone.jinja"),
        sha256: "b47201d762e02dcc447fe22f76d7375deb341d3f52714f2689b4c70e1de67a7e",
    },
    ShippedTemplate {
        decision_type: DecisionType::Lfm2D1Omni,
        source: include_str!("fixtures/d1-omni-systemone.jinja"),
        sha256: "f935b108a5412e582ce7c2f71d4caa3399a14735edcbf28eaed0735232ebfcb9",
    },
];

/// The name `decision.template` selects the generic read strategy by.
pub const GENERIC: &str = "generic";

pub fn shipped(decision_type: DecisionType) -> &'static ShippedTemplate {
    SHIPPED
        .iter()
        .find(|template| template.decision_type == decision_type)
        .expect("every decision type ships its template")
}

/// Every name `decision.template` accepts, for refusals.
pub fn template_names() -> String {
    std::iter::once(GENERIC)
        .chain(SHIPPED.iter().map(ShippedTemplate::name))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The refusal for a model with no decision binding, naming what would give
/// it one.
pub fn no_binding(model: &str) -> SystemOneError {
    SystemOneError::Unsupported(format!(
        "{model} has no decision template: it is not a native decision checkpoint and none was \
         selected; start it with decision.template set to one of: {} (or a template path)",
        template_names()
    ))
}

/// Calibration temperatures by the reference's names (`choice:3-5`, `noul:2`,
/// `score`). A question takes its type-and-cardinality entry, then its type's,
/// then 1, as llama.cpp's `get_temperature` and the omni readout do.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Calibration {
    temperatures: BTreeMap<String, Temperature>,
}

impl Calibration {
    /// Every question at one temperature (an explicit caller policy).
    pub fn uniform(temperature: Temperature) -> Self {
        Self {
            temperatures: ["choice", "score", "noul"]
                .into_iter()
                .map(|kind| (kind.to_string(), temperature))
                .collect(),
        }
    }

    pub fn from_reference(temperatures: &BTreeMap<String, f64>) -> Result<Self> {
        let mut out = BTreeMap::new();
        for (name, &value) in temperatures {
            out.insert(name.clone(), Temperature::try_from(value)?);
        }
        Ok(Self { temperatures: out })
    }

    pub fn is_empty(&self) -> bool {
        self.temperatures.is_empty()
    }

    pub fn temperature<T>(&self, question: &Question<T>) -> Temperature {
        let kind = match question {
            Question::Choice { .. } => "choice",
            Question::Score { .. } => "score",
            Question::Noul { .. } => "noul",
        };
        self.temperatures
            .get(&question.temperature_key())
            .or_else(|| self.temperatures.get(kind))
            .copied()
            .unwrap_or_else(|| Temperature::try_from(1.).expect("unit temperature"))
    }
}

/// What `decision.template` asked for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Selection {
    /// Nothing selected: a native checkpoint's own binding, else none.
    Auto,
    Generic,
    Shipped(DecisionType),
    Path(PathBuf),
}

impl Selection {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        Ok(match value.map(str::trim) {
            None | Some("") | Some("auto") => Self::Auto,
            Some(GENERIC) => Self::Generic,
            Some(name) => match DecisionType::from_name(name) {
                Some(kind) => Self::Shipped(kind),
                None if name.contains(std::path::MAIN_SEPARATOR) || name.ends_with(".jinja") => {
                    Self::Path(PathBuf::from(name))
                }
                None => bail!(
                    "decision.template {name:?} is neither a template name ({}) nor a path",
                    template_names()
                ),
            },
        })
    }
}

/// The binding facts a checkpoint carries, read before any weight.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CheckpointDecision {
    pub decision_type: Option<DecisionType>,
    /// The checkpoint's own `systemone` template source (GGUF only).
    pub systemone: Option<String>,
    pub calibration: Calibration,
}

impl CheckpointDecision {
    /// llama.cpp's metadata: `<arch>.decision.type`, the named chat template
    /// `tokenizer.chat_template.systemone`, `<arch>.decision.temperature.*`.
    /// A decision type llama.cpp knows and the engine does not is refused by
    /// name; a typed checkpoint without its template is refused as llama.cpp
    /// refuses it.
    pub fn from_gguf(
        metadata: &std::collections::HashMap<String, lloom_tensor::quantized::gguf_file::Value>,
    ) -> Result<Self> {
        let arch = metadata
            .get("general.architecture")
            .ok_or_else(|| Error::msg("GGUF is missing general.architecture"))?
            .to_string()?;
        let Some(name) = metadata.get(&format!("{arch}.decision.type")) else {
            return Ok(Self::default());
        };
        let name = name.to_string()?;
        let decision_type = DecisionType::from_name(name)
            .ok_or_else(|| Error::msg(format!("unsupported decision model type: {name}")))?;
        let systemone = metadata
            .get("tokenizer.chat_template.systemone")
            .ok_or_else(|| Error::msg("decision model has no \"systemone\" template"))?
            .to_string()?
            .clone();
        let calibration = Calibration::from_reference(
            &lloom::models::d1_omni::decision_temperatures(metadata, arch)?,
        )?;
        Ok(Self {
            decision_type: Some(decision_type),
            systemone: Some(systemone),
            calibration,
        })
    }

    /// A safetensors `config.json`: d1-omni is `model_type: d1_omni` with its
    /// `temperatures`; d1-3B is an `lfm2_vl` whose `auto_map` loads the
    /// reference's `modeling_d1.D1Model`.
    pub fn from_config(config: &serde_json::Value) -> Result<Self> {
        let model_type = config.get("model_type").and_then(serde_json::Value::as_str);
        let auto_model = config
            .pointer("/auto_map/AutoModel")
            .and_then(serde_json::Value::as_str);
        let decision_type = match (model_type, auto_model) {
            (Some("d1_omni"), _) => DecisionType::Lfm2D1Omni,
            (Some("lfm2_vl"), Some("modeling_d1.D1Model")) => DecisionType::Lfm2D1,
            _ => return Ok(Self::default()),
        };
        let mut temperatures = BTreeMap::new();
        if let Some(table) = config.get("temperatures") {
            let table = table
                .as_object()
                .ok_or_else(|| Error::msg("config.json temperatures must be an object"))?;
            for (name, value) in table {
                let value = value.as_f64().ok_or_else(|| {
                    Error::msg(format!("config.json temperature {name} is not a number"))
                })?;
                temperatures.insert(name.clone(), value);
            }
        }
        Ok(Self {
            decision_type: Some(decision_type),
            systemone: None,
            calibration: Calibration::from_reference(&temperatures)?,
        })
    }
}

/// A resolved template binding, ready to render.
pub struct TemplateBinding {
    pub template: SystemOneTemplate,
    pub calibration: Calibration,
    /// Set only for a native checkpoint's own binding.
    pub native: Option<DecisionType>,
    /// Where the template came from, for the load line.
    pub origin: String,
}

/// What a model answers decisions with.
pub enum Binding {
    Template(Box<TemplateBinding>),
    Generic,
}

impl Binding {
    /// The load line's words for this binding.
    pub fn describe(&self) -> String {
        match self {
            Self::Template(binding) => format!("decision template: {}", binding.origin),
            Self::Generic => "decision template: generic (uncalibrated)".into(),
        }
    }
}

/// Resolves `selection` against what the checkpoint carries. `kind` is the
/// template layout the family renders (causal label codes or omni markers).
/// `Ok(None)`: no binding, decisions are refused by name.
pub fn resolve(
    selection: &Selection,
    checkpoint: &CheckpointDecision,
    kind: SystemOneTemplateKind,
) -> Result<Option<Binding>> {
    let native = checkpoint.decision_type;
    if let Some(native) = native {
        if native.template_kind() != kind {
            bail!(
                "the checkpoint is a {} decision model, which this family cannot render",
                native.name()
            )
        }
    }
    let calibration_for = |kind: DecisionType| {
        if native == Some(kind) {
            checkpoint.calibration.clone()
        } else {
            Calibration::default()
        }
    };
    Ok(Some(match selection {
        Selection::Auto => {
            let Some(native) = native else {
                return Ok(None);
            };
            let (template, origin) = match &checkpoint.systemone {
                Some(source) => (
                    SystemOneTemplate::compile(source, kind)?,
                    format!("{} (the checkpoint's systemone template)", native.name()),
                ),
                None => {
                    let shipped = shipped(native);
                    (
                        shipped.compile()?,
                        format!(
                            "{} (engine-shipped, sha256 {})",
                            native.name(),
                            shipped.sha256
                        ),
                    )
                }
            };
            Binding::Template(Box::new(TemplateBinding {
                template,
                calibration: checkpoint.calibration.clone(),
                native: Some(native),
                origin,
            }))
        }
        Selection::Generic => {
            if kind != SystemOneTemplateKind::D1Causal {
                bail!("the generic decision strategy reads a causal model's reply; this family does not generate")
            }
            Binding::Generic
        }
        Selection::Shipped(selected) => {
            if selected.template_kind() != kind {
                bail!(
                    "decision.template {} renders a different model layout than this family",
                    selected.name()
                )
            }
            let shipped = shipped(*selected);
            Binding::Template(Box::new(TemplateBinding {
                template: shipped.compile()?,
                calibration: calibration_for(*selected),
                native: (native == Some(*selected)).then_some(*selected),
                origin: format!(
                    "{} (engine-shipped, sha256 {})",
                    selected.name(),
                    shipped.sha256
                ),
            }))
        }
        Selection::Path(path) => {
            let source = std::fs::read_to_string(path).map_err(|error| {
                Error::msg(format!(
                    "reading decision template {}: {error}",
                    path.display()
                ))
            })?;
            Binding::Template(Box::new(TemplateBinding {
                template: SystemOneTemplate::compile(&source, kind)?,
                calibration: Calibration::default(),
                native: None,
                origin: format!("{} (uncalibrated)", path.display()),
            }))
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom_tensor::quantized::gguf_file::Value;
    use sha2::Digest;
    use std::collections::HashMap;

    #[test]
    fn shipped_templates_are_the_pinned_bytes() {
        for template in &SHIPPED {
            let digest = sha2::Sha256::digest(template.source.as_bytes());
            let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(hex, template.sha256, "{}", template.name());
            assert_eq!(
                template.compile().unwrap().kind(),
                template.decision_type.template_kind()
            );
        }
    }

    fn gguf(entries: &[(&str, Value)]) -> HashMap<String, Value> {
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn gguf_binding_carries_type_template_and_calibration() -> Result<()> {
        let source = SHIPPED[0].source.to_string();
        let facts = CheckpointDecision::from_gguf(&gguf(&[
            ("general.architecture", Value::String("lfm2".into())),
            ("lfm2.decision.type", Value::String("lfm2-d1".into())),
            (
                "tokenizer.chat_template.systemone",
                Value::String(source.clone()),
            ),
            ("lfm2.decision.temperature.choice.3_5", Value::F32(1.5)),
            ("lfm2.decision.temperature.noul", Value::F32(2.0)),
        ]))?;
        assert_eq!(facts.decision_type, Some(DecisionType::Lfm2D1));
        assert_eq!(facts.systemone.as_deref(), Some(source.as_str()));
        let three = Question::Choice {
            instructions: (),
            criteria: vec![("a".into(), None), ("b".into(), None), ("c".into(), None)],
        };
        let two = Question::Choice {
            instructions: (),
            criteria: vec![("a".into(), None), ("b".into(), None)],
        };
        let noul: Question<()> = Question::Noul {
            instructions: (),
            criteria: None,
        };
        assert_eq!(f64::from(facts.calibration.temperature(&three)), 1.5);
        assert_eq!(f64::from(facts.calibration.temperature(&two)), 1.0);
        assert_eq!(f64::from(facts.calibration.temperature(&noul)), 2.0);
        let Some(Binding::Template(binding)) =
            resolve(&Selection::Auto, &facts, SystemOneTemplateKind::D1Causal)?
        else {
            panic!("a native checkpoint binds itself")
        };
        assert_eq!(binding.native, Some(DecisionType::Lfm2D1));
        assert_eq!(
            binding.calibration, facts.calibration,
            "native calibration applies"
        );
        assert!(
            binding.template.has_causal_certificate(),
            "{}",
            binding.origin
        );
        Ok(())
    }

    #[test]
    fn a_typed_gguf_without_its_template_or_with_an_unknown_type_is_refused() {
        let missing = CheckpointDecision::from_gguf(&gguf(&[
            ("general.architecture", Value::String("lfm2".into())),
            ("lfm2.decision.type", Value::String("lfm2-d1".into())),
        ]));
        assert!(missing
            .unwrap_err()
            .to_string()
            .contains("no \"systemone\" template"));
        let unknown = CheckpointDecision::from_gguf(&gguf(&[
            ("general.architecture", Value::String("lfm2".into())),
            ("lfm2.decision.type", Value::String("openjev".into())),
        ]));
        assert!(unknown.unwrap_err().to_string().contains("openjev"));
    }

    #[test]
    fn a_plain_checkpoint_has_no_binding_unless_one_is_selected() -> Result<()> {
        let plain = CheckpointDecision::from_gguf(&gguf(&[(
            "general.architecture",
            Value::String("lfm2".into()),
        )]))?;
        assert_eq!(plain, CheckpointDecision::default());
        assert!(resolve(&Selection::Auto, &plain, SystemOneTemplateKind::D1Causal)?.is_none());
        assert!(matches!(
            resolve(&Selection::Generic, &plain, SystemOneTemplateKind::D1Causal)?,
            Some(Binding::Generic)
        ));
        let Some(Binding::Template(binding)) = resolve(
            &Selection::Shipped(DecisionType::Lfm2D1),
            &plain,
            SystemOneTemplateKind::D1Causal,
        )?
        else {
            panic!("a shipped template binds by name")
        };
        assert_eq!(binding.native, None, "a selected template is not native");
        assert!(binding.calibration.is_empty());
        assert!(
            resolve(&Selection::Generic, &plain, SystemOneTemplateKind::D1Omni).is_err(),
            "generic needs a generating model"
        );
        Ok(())
    }

    #[test]
    fn safetensors_configs_are_recognised_by_their_reference_classes() -> Result<()> {
        let d1 = CheckpointDecision::from_config(&serde_json::json!({
            "model_type": "lfm2_vl", "auto_map": {"AutoModel": "modeling_d1.D1Model"}
        }))?;
        assert_eq!(d1.decision_type, Some(DecisionType::Lfm2D1));
        let base = CheckpointDecision::from_config(&serde_json::json!({"model_type": "lfm2_vl"}))?;
        assert_eq!(base.decision_type, None);
        let omni = CheckpointDecision::from_config(&serde_json::json!({
            "model_type": "d1_omni", "temperatures": {"choice:2": 1.75, "choice": 1.0}
        }))?;
        assert_eq!(omni.decision_type, Some(DecisionType::Lfm2D1Omni));
        let Some(Binding::Template(binding)) =
            resolve(&Selection::Auto, &d1, SystemOneTemplateKind::D1Causal)?
        else {
            panic!("d1-3B safetensors binds the shipped template")
        };
        assert!(binding.origin.contains(SHIPPED[0].sha256));
        Ok(())
    }

    #[test]
    fn selection_names_paths_and_refusals() -> Result<()> {
        assert_eq!(Selection::parse(None)?, Selection::Auto);
        assert_eq!(Selection::parse(Some("auto"))?, Selection::Auto);
        assert_eq!(Selection::parse(Some("generic"))?, Selection::Generic);
        assert_eq!(
            Selection::parse(Some("lfm2-d1-omni"))?,
            Selection::Shipped(DecisionType::Lfm2D1Omni)
        );
        assert_eq!(
            Selection::parse(Some("/t/x.jinja"))?,
            Selection::Path("/t/x.jinja".into())
        );
        let error = Selection::parse(Some("vicuna")).unwrap_err().to_string();
        assert!(error.contains("generic, lfm2-d1, lfm2-d1-omni"), "{error}");
        let SystemOneError::Unsupported(refusal) = no_binding("m") else {
            panic!("no binding is a 501")
        };
        assert!(
            refusal.contains("generic, lfm2-d1, lfm2-d1-omni"),
            "{refusal}"
        );
        Ok(())
    }
}
