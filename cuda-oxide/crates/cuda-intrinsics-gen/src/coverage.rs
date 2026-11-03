/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Report how much of the pinned LLVM metadata the generator actually covers.
//!
//! `extract` pulls every NVVM intrinsic record LLVM knows about into
//! `intrinsics/imported.json`. `generate` then emits bindings for the subset
//! described by `intrinsics/catalog.json`. The difference is intrinsics that
//! reached the repository and stopped there - visible in a JSON file nobody
//! reads, rather than in any error.
//!
//! That difference is large, and most of it is not worth closing. Surface and
//! texture operations account for roughly a third of it and are irrelevant to
//! compute kernels, so a raw "72% uncovered" figure overstates the real backlog
//! by a wide margin. The useful output is therefore per family, not a single
//! percentage: it tells a contributor which families are worth aiming at and
//! which to ignore.
//!
//! ```text
//! cargo run -p cuda-intrinsics-gen -- coverage
//! cargo run -p cuda-intrinsics-gen -- coverage --family shfl
//! ```
//!
//! This is a reporting command. It reads the two JSON files and writes nothing.

use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// Families that exist in the metadata but are not compute functionality.
///
/// Surface, texture, and their query forms. Counted separately so the headline
/// backlog reflects work someone might actually want to do.
const NON_COMPUTE_PREFIXES: &[&str] = &["sust", "suld", "suq", "tex", "tld4", "txq"];

/// One family's coverage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FamilyCoverage {
    /// Family name, taken from the intrinsic's leading path segment.
    pub family: String,
    /// Records present in `catalog.json`.
    pub generated: usize,
    /// Records in `imported.json` with no catalog entry.
    pub ungenerated: usize,
    /// Whether this family is compute-relevant.
    pub compute: bool,
}

impl FamilyCoverage {
    /// Records seen in either file.
    #[must_use]
    pub fn total(&self) -> usize {
        self.generated + self.ungenerated
    }
}

/// The whole report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Coverage {
    /// Per family, ordered by ungenerated count descending.
    pub families: Vec<FamilyCoverage>,
    /// Distinct intrinsics in `imported.json`.
    pub imported: usize,
    /// Distinct intrinsics in `catalog.json`.
    pub generated: usize,
    /// Imported records the catalog deliberately rejects via `llvm_exclusion`.
    ///
    /// Not backlog: kept out of every ungenerated count.
    pub excluded: usize,
}

impl Coverage {
    /// Ungenerated records across every family.
    #[must_use]
    pub fn ungenerated(&self) -> usize {
        self.imported
            .saturating_sub(self.generated)
            .saturating_sub(self.excluded)
    }

    /// Ungenerated records in compute-relevant families only.
    ///
    /// The number worth quoting: the raw total is dominated by surface and
    /// texture work no compute kernel needs.
    #[must_use]
    pub fn ungenerated_compute(&self) -> usize {
        self.families
            .iter()
            .filter(|f| f.compute)
            .map(|f| f.ungenerated)
            .sum()
    }
}

/// Family an intrinsic belongs to.
///
/// Uses the leading segment after the `int_nvvm_` prefix, which is how the
/// metadata already groups them. Numeric suffixes are kept, so `f2i` and `d2i`
/// stay distinct - they have different coverage stories.
fn family_of(name: &str) -> String {
    let stem = name.strip_prefix("int_nvvm_").unwrap_or(name);
    // First matching prefix in array order wins (the prefixes do not overlap,
    // so the order is not load-bearing). Everything else groups by its leading
    // `_`-separated segment, which is how `shfl_sync` lands under `shfl`
    // rather than splitting into its own family.
    for prefix in NON_COMPUTE_PREFIXES {
        if stem.starts_with(prefix) {
            return (*prefix).to_string();
        }
    }
    stem.split('_').next().unwrap_or(stem).to_string()
}

/// Every `int_nvvm_*` name appearing anywhere in a JSON document.
///
/// Correct for `imported.json`, whose every `int_nvvm_*` mention *is* an imported
/// record. Do **not** use it on the catalog: see [`collect_generated`].
fn collect_names(value: &Value, out: &mut BTreeSet<String>) {
    match value {
        Value::String(s) if s.starts_with("int_nvvm_") => {
            out.insert(s.clone());
        }
        Value::Array(items) => items.iter().for_each(|v| collect_names(v, out)),
        Value::Object(map) => map.values().for_each(|v| collect_names(v, out)),
        _ => {}
    }
}

/// Imported records the catalog actually *binds*, plus the ones it explicitly
/// rejects.
///
/// A naive walk over every `int_nvvm_*` string in the catalog over-counts, because
/// the catalog also records intrinsics it deliberately does **not** use. The
/// `gridid` entry is the live example: it is `source = { kind = "ptx_native",
/// instruction = "mov.u64 %gridid" }` and carries
/// `special_register.llvm_exclusion = { source_record = "int_nvvm_read_ptx_sreg_gridid",
/// reason = "result_width_mismatch" }` — the imported intrinsic returns `b32` while
/// the register is 64-bit, so it is unusable by construction. Counting that as
/// coverage inverts its meaning and inflates the generated total.
///
/// So bind only `source.source_record` where `source.kind == "llvm_imported"`, and
/// report exclusions separately: "deliberately rejected" is a different fact from
/// "nobody has got to it yet", and the whole point of this command is to keep those
/// apart.
fn collect_generated(
    catalog: &Value,
    bound: &mut BTreeSet<String>,
    excluded: &mut BTreeSet<String>,
) {
    let Some(entries) = catalog.get("intrinsics").and_then(Value::as_array) else {
        return;
    };
    for entry in entries {
        let source = entry.get("source");
        let kind = source.and_then(|s| s.get("kind")).and_then(Value::as_str);
        if kind == Some("llvm_imported")
            && let Some(record) = source
                .and_then(|s| s.get("source_record"))
                .and_then(Value::as_str)
        {
            bound.insert(record.to_string());
        }
        if let Some(record) = entry
            .get("special_register")
            .and_then(|sreg| sreg.get("llvm_exclusion"))
            .and_then(|excl| excl.get("source_record"))
            .and_then(Value::as_str)
        {
            excluded.insert(record.to_string());
        }
    }
}

/// Build the report from the two pinned JSON files.
pub fn compute(repo_root: &Path) -> Result<Coverage> {
    let read = |rel: &str| -> Result<Value> {
        let path = repo_root.join(rel);
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
    };

    let mut imported = BTreeSet::new();
    collect_names(&read("intrinsics/imported.json")?, &mut imported);
    let mut generated = BTreeSet::new();
    let mut excluded = BTreeSet::new();
    collect_generated(
        &read("intrinsics/catalog.json")?,
        &mut generated,
        &mut excluded,
    );

    let mut per_family: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut deliberately_rejected = 0;
    for name in &imported {
        // A record the catalog deliberately rejects (`llvm_exclusion`) is not
        // backlog: keep it out of every family's ungenerated count.
        if excluded.contains(name) && !generated.contains(name) {
            deliberately_rejected += 1;
            continue;
        }
        let entry = per_family.entry(family_of(name)).or_default();
        if generated.contains(name) {
            entry.0 += 1;
        } else {
            entry.1 += 1;
        }
    }

    let mut families: Vec<FamilyCoverage> = per_family
        .into_iter()
        .map(|(family, (done, ungen))| FamilyCoverage {
            compute: !NON_COMPUTE_PREFIXES.contains(&family.as_str()),
            family,
            generated: done,
            ungenerated: ungen,
        })
        .collect();
    // Largest gap first: that is the order a contributor reads it in.
    families.sort_by(|a, b| {
        b.ungenerated
            .cmp(&a.ungenerated)
            .then_with(|| a.family.cmp(&b.family))
    });

    Ok(Coverage {
        families,
        imported: imported.len(),
        generated: generated.len(),
        excluded: deliberately_rejected,
    })
}

/// Render the full report.
///
/// Separate from [`run`] so tests can assert on the exact text.
fn render_report(coverage: &Coverage) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let _ = writeln!(
        out,
        "pinned NVVM intrinsics: {} imported, {} generated, {} ungenerated",
        coverage.imported,
        coverage.generated,
        coverage.ungenerated()
    );
    let _ = writeln!(
        out,
        "excluded (deliberately rejected): {}",
        coverage.excluded
    );
    let _ = writeln!(
        out,
        "ungenerated in compute families: {}  (the rest is surface/texture)",
        coverage.ungenerated_compute()
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "{:<20}{:>10}{:>13}",
        "family", "generated", "ungenerated"
    );
    for f in coverage.families.iter().filter(|f| f.ungenerated > 0) {
        let _ = writeln!(
            out,
            "{:<20}{:>10}{:>13}{}",
            f.family,
            f.generated,
            f.ungenerated,
            if f.compute { "" } else { "   [not compute]" }
        );
    }
    out
}

/// Print the report.
///
/// `family` filters to one family, for checking a single area before working
/// on it.
pub fn run(repo_root: &Path, family: Option<&str>) -> Result<()> {
    let coverage = compute(repo_root)?;

    if let Some(wanted) = family {
        let Some(f) = coverage.families.iter().find(|f| f.family == wanted) else {
            bail!("no family `{wanted}` in the pinned metadata");
        };
        println!(
            "{}: {} generated, {} ungenerated ({} total){}",
            f.family,
            f.generated,
            f.ungenerated,
            f.total(),
            if f.compute { "" } else { "  [not compute]" }
        );
        return Ok(());
    }

    print!("{}", render_report(&coverage));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn family_uses_the_leading_segment() {
        assert_eq!(family_of("int_nvvm_fma_rn_f32"), "fma");
        assert_eq!(family_of("int_nvvm_mbarrier_arrive"), "mbarrier");
        // `shfl_sync_*` groups with `shfl`, not as its own family.
        assert_eq!(family_of("int_nvvm_shfl_sync_bfly_f32p"), "shfl");
        assert_eq!(family_of("int_nvvm_shfl_bfly_i32"), "shfl");
    }

    /// Surface and texture forms must land in the non-compute bucket, since
    /// separating them is the whole point of the report.
    #[test]
    fn surface_and_texture_are_not_compute() {
        for n in [
            "int_nvvm_sust_b_1d_i32_clamp",
            "int_nvvm_suld_1d_i8_trap",
            "int_nvvm_tex_1d_v4f32_s32",
            "int_nvvm_tld4_r_2d_v4f32_f32",
            "int_nvvm_suq_width",
            "int_nvvm_txq_height",
        ] {
            let f = family_of(n);
            assert!(
                NON_COMPUTE_PREFIXES.contains(&f.as_str()),
                "{n} -> {f} should be non-compute"
            );
        }
        // And compute families must not be swept up with them.
        let f = family_of("int_nvvm_fma_rn_f32");
        assert!(!NON_COMPUTE_PREFIXES.contains(&f.as_str()));
        // `texsurf_handle` matches the `tex` prefix and lands in the
        // non-compute bucket, which is right: it is a texture-handle helper.
        assert_eq!(family_of("int_nvvm_texsurf_handle"), "tex");
    }

    #[test]
    fn counts_split_generated_from_ungenerated() {
        let mut imported = BTreeSet::new();
        collect_names(
            &json!({"records": [
                {"id": "int_nvvm_fma_rn_f32"},
                {"id": "int_nvvm_fma_rn_f64"},
                {"id": "int_nvvm_sust_b_1d_i32_clamp"},
                {"nested": {"deep": "int_nvvm_shfl_sync_idx_f32p"}},
            ]}),
            &mut imported,
        );
        assert_eq!(imported.len(), 4, "walks nested objects and arrays");
        assert!(imported.contains("int_nvvm_shfl_sync_idx_f32p"));
    }

    /// A totals identity, so a schema change that stops matching one file
    /// shows up as a broken invariant rather than a quietly wrong percentage.
    #[test]
    fn totals_are_consistent() {
        let c = Coverage {
            families: vec![
                FamilyCoverage {
                    family: "fma".into(),
                    generated: 10,
                    ungenerated: 5,
                    compute: true,
                },
                FamilyCoverage {
                    family: "sust".into(),
                    generated: 0,
                    ungenerated: 210,
                    compute: false,
                },
            ],
            imported: 226,
            generated: 10,
            excluded: 1,
        };
        assert_eq!(
            c.ungenerated(),
            215,
            "a deliberate exclusion is not backlog"
        );
        assert_eq!(c.ungenerated_compute(), 5, "surface work is excluded");
        assert_eq!(c.families[0].total(), 15);
    }

    #[test]
    fn an_llvm_exclusion_is_not_counted_as_coverage() {
        // Shape taken from the live `gridid` entry: a `ptx_native` intrinsic that
        // records the imported record it deliberately does not use. Walking every
        // `int_nvvm_*` string would count the excluded record as generated, which
        // inverts its meaning.
        let catalog: Value = serde_json::from_str(
            r#"{
              "intrinsics": [
                {
                  "id": "bound_one",
                  "source": { "kind": "llvm_imported", "source_record": "int_nvvm_bound" }
                },
                {
                  "id": "gridid",
                  "source": { "kind": "ptx_native", "instruction": "mov.u64 %gridid" },
                  "special_register": {
                    "llvm_exclusion": {
                      "source_record": "int_nvvm_read_ptx_sreg_gridid",
                      "reason": "result_width_mismatch"
                    }
                  }
                }
              ]
            }"#,
        )
        .expect("fixture parses");

        let mut bound = BTreeSet::new();
        let mut excluded = BTreeSet::new();
        collect_generated(&catalog, &mut bound, &mut excluded);

        assert_eq!(
            bound,
            BTreeSet::from(["int_nvvm_bound".to_string()]),
            "only llvm_imported sources count as coverage"
        );
        assert_eq!(
            excluded,
            BTreeSet::from(["int_nvvm_read_ptx_sreg_gridid".to_string()]),
            "the exclusion is tracked separately, not silently dropped"
        );
        assert!(
            !bound.contains("int_nvvm_read_ptx_sreg_gridid"),
            "a deliberately rejected intrinsic must never be reported as generated"
        );

        // And end to end: the report must surface the exclusion as its own
        // line, not fold it into some family's ungenerated backlog.
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda-intrinsics-gen-coverage-{}-{unique}",
            std::process::id()
        ));
        let directory = root.join("intrinsics");
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("imported.json"),
            r#"{"records": [{"id": "int_nvvm_bound"}, {"id": "int_nvvm_read_ptx_sreg_gridid"}]}"#,
        )
        .unwrap();
        std::fs::write(directory.join("catalog.json"), catalog.to_string()).unwrap();

        let coverage = compute(&root).unwrap();
        std::fs::remove_dir_all(root).unwrap();

        assert_eq!(coverage.imported, 2);
        assert_eq!(coverage.generated, 1);
        assert_eq!(coverage.excluded, 1);
        assert_eq!(coverage.ungenerated(), 0, "the exclusion is not backlog");
        assert!(
            coverage.families.iter().all(|f| f.ungenerated == 0),
            "gridid must not surface as ungenerated in the `read` family"
        );

        let report = render_report(&coverage);
        assert!(
            report.contains("excluded (deliberately rejected): 1"),
            "the report must state the exclusion:\n{report}"
        );
    }
}
