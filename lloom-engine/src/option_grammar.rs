//! The value grammar's front-end half: an env variable whose value selected
//! something else before the variables joined the option tree says so once,
//! at startup ([`legacy_notes`]).

use crate::EngineOptions;
use lloom_tensor::options::{before, BeforeRow};

/// The front-end's legacy env names whose values meant something else
/// before they joined the option tree.
pub const LEGACY_BEFORE: &[BeforeRow] = &[("LFM2_MODEL_DIR", "load.model", before::path_verbatim)];

/// One line per legacy env variable `env` sets to a value whose meaning
/// differs from what it selected before the variables joined the option
/// tree, over both library crates' groups and the front-end's own.
pub fn legacy_notes(env: &dyn Fn(&str) -> Option<String>) -> Vec<String> {
    let mut out = lloom_tensor::tuning::legacy_notes(env);
    out.extend(lloom::tuning::legacy_notes(env));
    out.extend(lloom_tensor::options::legacy_notes(
        LEGACY_BEFORE,
        EngineOptions::META,
        env,
        &|alone| {
            let mut o = EngineOptions::default();
            o.from_env_with(alone)?;
            serde_json::to_value(&o).map_err(|e| e.to_string())
        },
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::option_tree::index;

    #[test]
    fn a_legacy_value_whose_meaning_moved_says_so_and_one_that_did_not_is_silent() {
        let notes = |pairs: &[(&str, &str)]| {
            let pairs: Vec<(String, String)> = pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            legacy_notes(&move |n: &str| pairs.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone()))
        };
        let moved = notes(&[
            ("LLOOM_WG_SYNC", "0"),
            ("QWEN4EXP_MASK", "true"),
            ("CANDLE_METAL_ENABLE_FAST_MATH", "on"),
            ("LLOOM_148_SHARE_TRACE", "0"),
            ("CANDLE_MMAP_ADVICE", ""),
        ]);
        assert_eq!(moved.len(), 5, "{moved:#?}");
        let line = |var: &str| {
            moved
                .iter()
                .find(|l| l.contains(&format!("{var}=")))
                .unwrap_or_else(|| panic!("no line for {var}: {moved:#?}"))
        };
        assert_eq!(
            line("LLOOM_WG_SYNC"),
            "[options] LLOOM_WG_SYNC=\"0\" selected weights.groups.sync = true before the option tree; it now selects false"
        );
        assert!(
            line("QWEN4EXP_MASK").ends_with("= false before the option tree; it now selects true")
        );
        assert!(line("CANDLE_METAL_ENABLE_FAST_MATH")
            .contains("backend.metal.compiler.fast_math = false"));
        assert!(line("LLOOM_148_SHARE_TRACE").ends_with("it now selects false"));
        assert!(line("CANDLE_MMAP_ADVICE").contains("it now selects nothing: it is refused"));
        let same = notes(&[
            ("LLOOM_WG_SYNC", "1"),
            ("QWEN4EXP_MASK", "1"),
            ("QWEN4EXP_REPLAY", "device"),
            ("CANDLE_METAL_ENABLE_FAST_MATH", "0"),
            ("CANDLE_METAL_EAGER_RESIDENCY_RELEASE", "1"),
            ("LLOOM_148_POOL_PAGES", "16"),
            ("LLOOM_383_TILE", "32x32"),
            ("CANDLE_MMAP_ADVICE", "none"),
            ("LLOOM_MOE_PAIR", "on"),
        ]);
        assert!(same.is_empty(), "{same:#?}");
    }

    /// Every row of the three tables names a legacy env name of the field it
    /// says, at that field's storage path in the tree.
    #[test]
    fn the_tables_name_real_fields() {
        let idx = index();
        let tables: [(&str, &[BeforeRow]); 3] = [
            ("tuning.", lloom_tensor::tuning::LEGACY_BEFORE),
            ("tuning.", lloom::tuning::LEGACY_BEFORE),
            ("", LEGACY_BEFORE),
        ];
        let mut n = 0;
        for (prefix, table) in tables {
            for (var, storage, _) in table {
                let storage = format!("{prefix}{storage}");
                let key = idx
                    .key_of_storage(&storage)
                    .unwrap_or_else(|| panic!("{storage} is not a field"));
                let leaf = key
                    .targets
                    .iter()
                    .chain(&key.legacy)
                    .find(|l| l.storage == storage)
                    .expect("found by storage");
                assert_eq!(leaf.meta.env, Some(*var), "{storage}");
                n += 1;
            }
        }
        assert!(n >= 25, "{n} rows");
    }
}
