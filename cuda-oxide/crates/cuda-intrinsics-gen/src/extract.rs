/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    ImportedAddressSpace, ImportedFile, ImportedImmediateBinding, ImportedIntrinsic,
    ImportedSelection, ImportedSelectionConstraints, ImportedSource, UpstreamLock,
};
use crate::util::{pretty_json, sha256_bytes, sha256_file, write_if_changed};
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(crate) const IMPORTED_SCHEMA: u32 = 2;

pub struct ExtractOptions {
    pub intrinsics_json: Option<PathBuf>,
    pub nvptx_json: Option<PathBuf>,
    pub llvm_src: Option<PathBuf>,
    pub llvm_tblgen: Option<PathBuf>,
}

pub fn read_upstream_lock(repo_root: &Path) -> Result<UpstreamLock> {
    let path = repo_root.join("intrinsics/upstream.lock");
    let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let lock: UpstreamLock = toml::from_str(&text)
        .with_context(|| format!("parse pinned metadata {}", path.display()))?;
    ensure!(
        lock.schema == 1,
        "unsupported upstream.lock schema {}",
        lock.schema
    );
    ensure!(
        lock.llvm.public_output_allowed,
        "pinned LLVM input is not approved for public generated output"
    );
    ensure!(
        !lock.llvm.provenance.trim().is_empty()
            && !lock.llvm_tblgen.name.trim().is_empty()
            && !lock.llvm_tblgen.provenance.trim().is_empty()
            && !lock.dumps.intrinsics_sha256.trim().is_empty()
            && !lock.dumps.nvptx_sha256.trim().is_empty()
            && !lock.dumps.normalized_imported_sha256.trim().is_empty(),
        "upstream.lock has incomplete source/tool provenance"
    );
    for tool in &lock.comparison_tools {
        ensure!(
            !tool.name.trim().is_empty()
                && !tool.version_line.trim().is_empty()
                && !tool.sha256.trim().is_empty()
                && !tool.provenance.trim().is_empty(),
            "upstream.lock has an incomplete comparison-tool record"
        );
    }
    Ok(lock)
}

pub fn run(repo_root: &Path, options: ExtractOptions) -> Result<()> {
    let lock = read_upstream_lock(repo_root)?;
    let (intrinsics_json, nvptx_json) = match (
        options.intrinsics_json,
        options.nvptx_json,
        options.llvm_src,
        options.llvm_tblgen,
    ) {
        (Some(intrinsics), Some(nvptx), None, None) => (intrinsics, nvptx),
        (None, None, Some(source), Some(tool)) => dump_tablegen(repo_root, &lock, &source, &tool)?,
        _ => bail!(
            "extract needs either both --intrinsics-json/--nvptx-json or both --llvm-src/--llvm-tblgen"
        ),
    };

    let intrinsics_hash = sha256_file(&intrinsics_json)?;
    let nvptx_hash = sha256_file(&nvptx_json)?;
    ensure!(
        intrinsics_hash == lock.dumps.intrinsics_sha256,
        "intrinsic TableGen JSON hash mismatch: expected {}, got {}",
        lock.dumps.intrinsics_sha256,
        intrinsics_hash
    );
    ensure!(
        nvptx_hash == lock.dumps.nvptx_sha256,
        "NVPTX TableGen JSON hash mismatch: expected {}, got {}",
        lock.dumps.nvptx_sha256,
        nvptx_hash
    );

    let intrinsic_root = read_tablegen_root(&intrinsics_json)?;
    let nvptx_root = read_tablegen_root(&nvptx_json)?;
    let imported = normalize(
        &lock,
        intrinsic_root,
        nvptx_root,
        intrinsics_hash,
        nvptx_hash,
    )?;
    let output = pretty_json(&imported)?;
    let path = repo_root.join("intrinsics/imported.json");
    let normalized_hash = sha256_bytes(output.as_bytes());
    if normalized_hash != lock.dumps.normalized_imported_sha256 {
        let candidate = repo_root.join("target/intrinsics/imported.json");
        write_if_changed(&candidate, &output)?;
        bail!(
            "normalized imported hash mismatch: upstream.lock records {}, generated {}; review {}, then explicitly refresh normalized_imported_sha256 for an intentional normalizer change",
            lock.dumps.normalized_imported_sha256,
            normalized_hash,
            candidate.display()
        );
    }
    let changed = write_if_changed(&path, &output)?;
    println!(
        "{} {} normalized NVVM declarations in {}",
        if changed { "wrote" } else { "checked" },
        imported.intrinsics.len(),
        path.display()
    );
    Ok(())
}

fn dump_tablegen(
    repo_root: &Path,
    lock: &UpstreamLock,
    llvm_src: &Path,
    llvm_tblgen: &Path,
) -> Result<(PathBuf, PathBuf)> {
    verify_tool(lock, llvm_tblgen)?;
    verify_source_revision(lock, llvm_src)?;

    let output_dir = repo_root.join("target/intrinsics/tblgen");
    fs::create_dir_all(&output_dir)?;
    let intrinsics_output = output_dir.join("intrinsics.json");
    let nvptx_output = output_dir.join("nvptx.json");
    let llvm_dir = llvm_src.join("llvm");
    let include_dir = llvm_dir.join("include");
    let nvptx_dir = llvm_dir.join("lib/Target/NVPTX");

    run_tblgen(
        llvm_tblgen,
        &[
            "--dump-json".into(),
            "-I".into(),
            include_dir.as_os_str().into(),
            llvm_dir
                .join("include/llvm/IR/Intrinsics.td")
                .as_os_str()
                .into(),
            "-o".into(),
            intrinsics_output.as_os_str().into(),
        ],
    )?;
    run_tblgen(
        llvm_tblgen,
        &[
            "--dump-json".into(),
            "-I".into(),
            include_dir.as_os_str().into(),
            "-I".into(),
            nvptx_dir.as_os_str().into(),
            nvptx_dir.join("NVPTX.td").as_os_str().into(),
            "-o".into(),
            nvptx_output.as_os_str().into(),
        ],
    )?;
    Ok((intrinsics_output, nvptx_output))
}

fn run_tblgen(tool: &Path, args: &[std::ffi::OsString]) -> Result<()> {
    let status = Command::new(tool)
        .args(args)
        .status()
        .with_context(|| format!("run {}", tool.display()))?;
    ensure!(status.success(), "{} failed with {status}", tool.display());
    Ok(())
}

fn verify_tool(lock: &UpstreamLock, tool: &Path) -> Result<()> {
    let hash = sha256_file(tool)?;
    if lock.llvm_tblgen.enforce_sha256 {
        ensure!(
            hash == lock.llvm_tblgen.sha256,
            "llvm-tblgen hash mismatch: expected {}, got {}",
            lock.llvm_tblgen.sha256,
            hash
        );
    }
    let output = Command::new(tool)
        .arg("--version")
        .output()
        .with_context(|| format!("query {} --version", tool.display()))?;
    ensure!(
        output.status.success(),
        "{} --version failed",
        tool.display()
    );
    let version = String::from_utf8_lossy(&output.stdout);
    ensure!(
        version.contains(&lock.llvm_tblgen.version_line),
        "llvm-tblgen version mismatch: expected output containing {:?}",
        lock.llvm_tblgen.version_line
    );
    ensure!(
        lock.llvm_tblgen.built_from_llvm_revision.as_deref() == Some(lock.llvm.revision.as_str()),
        "pinned extraction tool is not recorded as built from the pinned LLVM revision"
    );
    Ok(())
}

fn verify_source_revision(lock: &UpstreamLock, source: &Path) -> Result<()> {
    let output = Command::new("git")
        .arg("-C")
        .arg(source)
        .args(["rev-parse", "HEAD"])
        .output()
        .with_context(|| format!("query LLVM revision in {}", source.display()))?;
    ensure!(output.status.success(), "LLVM source is not a Git checkout");
    let revision = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    ensure!(
        revision == lock.llvm.revision,
        "LLVM source revision mismatch: expected {}, got {}",
        lock.llvm.revision,
        revision
    );
    Ok(())
}

fn read_tablegen_root(path: &Path) -> Result<Map<String, Value>> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let value: Value = serde_json::from_reader(file)
        .with_context(|| format!("parse TableGen JSON {}", path.display()))?;
    value
        .as_object()
        .cloned()
        .with_context(|| format!("{} is not a TableGen JSON object", path.display()))
}

fn normalize(
    lock: &UpstreamLock,
    intrinsic_root: Map<String, Value>,
    nvptx_root: Map<String, Value>,
    intrinsics_hash: String,
    nvptx_hash: String,
) -> Result<ImportedFile> {
    let mut names = intrinsic_names(&intrinsic_root)?;
    names.retain(|name| {
        intrinsic_root
            .get(name)
            .and_then(|record| record.get("TargetPrefix"))
            .and_then(Value::as_str)
            == Some("nvvm")
    });
    names.sort();
    names.dedup();
    let wanted: BTreeSet<&str> = names.iter().map(String::as_str).collect();
    let selections = selection_index(&nvptx_root, &wanted)?;

    let mut intrinsics = Vec::with_capacity(names.len());
    for name in names {
        let record = intrinsic_root
            .get(&name)
            .with_context(|| format!("missing intrinsic record {name}"))?;
        let arguments = type_list(record, "ParamTypes")?;
        let results = type_list(record, "RetTypes")?;
        let classes = string_list(record, "!superclasses")?;
        let properties = property_list(record, &intrinsic_root)?;
        intrinsics.push(ImportedIntrinsic {
            source_record: name.clone(),
            llvm_name: llvm_name(&name, record)?,
            arguments,
            results,
            classes,
            properties,
            selections: selections.get(&name).cloned().unwrap_or_default(),
        });
    }

    Ok(ImportedFile {
        schema: IMPORTED_SCHEMA,
        source: ImportedSource {
            llvm_repository: lock.llvm.repository.clone(),
            llvm_revision: lock.llvm.revision.clone(),
            llvm_tblgen_version: lock.llvm_tblgen.version_line.clone(),
            llvm_tblgen_source_revision: lock
                .llvm_tblgen
                .built_from_llvm_revision
                .clone()
                .context("pinned llvm-tblgen has no source revision")?,
            intrinsics_json_sha256: intrinsics_hash,
            nvptx_json_sha256: nvptx_hash,
        },
        intrinsics,
    })
}

fn string_list(record: &Value, field: &str) -> Result<Vec<String>> {
    let values = record
        .get(field)
        .and_then(Value::as_array)
        .with_context(|| format!("record has no {field} array"))?;
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .with_context(|| format!("{field} entry is not a string"))
        })
        .collect()
}

fn intrinsic_names(root: &Map<String, Value>) -> Result<Vec<String>> {
    let names = root
        .get("!instanceof")
        .and_then(|index| index.get("Intrinsic"))
        .and_then(Value::as_array)
        .context("TableGen JSON has no !instanceof.Intrinsic index")?;
    names
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .context("non-string intrinsic record name")
        })
        .collect()
}

fn selection_index(
    root: &Map<String, Value>,
    wanted: &BTreeSet<&str>,
) -> Result<BTreeMap<String, Vec<ImportedSelection>>> {
    let mut result: BTreeMap<String, Vec<ImportedSelection>> = BTreeMap::new();
    for (record_name, record) in root {
        if record_name == "!instanceof" {
            continue;
        }
        let mut references = BTreeSet::new();
        if let Some(pattern) = record.get("Pattern") {
            collect_actual_def_references(pattern, wanted, &mut references);
        }

        // Some instructions keep `Pattern = []` and identify the intrinsic
        // through `Intr` plus `IntrinsicPattern`. Accept an exact operator
        // match. Ldmatrix remains the only admitted PatFrag form because its
        // address-space constraint is modeled below.
        if references.is_empty()
            && let Some(intrinsic) = direct_wanted_intrinsic(record, wanted)
        {
            let operator = record
                .get("IntrinsicPattern")
                .and_then(|pattern| pattern.get("operator"))
                .and_then(|operator| operator.get("def"))
                .and_then(Value::as_str);
            if intrinsic.starts_with("int_nvvm_ldmatrix_") {
                let patfrag = intrinsic_pattern_patfrag(record_name, record, root)?;
                let mut patfrag_references = BTreeSet::new();
                let fragments = patfrag
                    .get("Fragments")
                    .with_context(|| format!("selection {record_name} PatFrag has no Fragments"))?;
                collect_actual_def_references(fragments, wanted, &mut patfrag_references);
                ensure!(
                    patfrag_references.contains(intrinsic),
                    "selection {record_name} names intrinsic {intrinsic}, but its IntrinsicPattern PatFrag references {:?}",
                    patfrag_references
                );
                references.insert(intrinsic.to_owned());
            } else if operator == Some(intrinsic) {
                references.insert(intrinsic.to_owned());
            }
        }
        if references.is_empty() {
            continue;
        }
        let asm = record
            .get("AsmString")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let predicates = normalized_predicates(record_name, record, root)?;
        for reference in references {
            let address_space = if reference.starts_with("int_nvvm_ldmatrix_") {
                Some(ldmatrix_address_space(record_name, record, root)?)
            } else {
                None
            };
            let constraints = ImportedSelectionConstraints {
                address_space,
                immediate_bindings: selection_immediate_bindings(record_name, record, &reference)?,
            };
            result
                .entry(reference)
                .or_default()
                .push(ImportedSelection {
                    source_record: record_name.clone(),
                    asm: asm.clone(),
                    predicates: predicates.clone(),
                    constraints,
                });
        }
    }
    for records in result.values_mut() {
        records.sort();
        records.dedup();
    }
    Ok(result)
}

fn selection_immediate_bindings(
    record_name: &str,
    record: &Value,
    intrinsic: &str,
) -> Result<Vec<ImportedImmediateBinding>> {
    let mut occurrences = Vec::new();
    if let Some(pattern) = record.get("Pattern") {
        collect_immediate_bindings(pattern, record_name, intrinsic, &mut occurrences)?;
    }
    if occurrences.is_empty()
        && let Some(pattern) = record.get("IntrinsicPattern")
    {
        collect_immediate_bindings(pattern, record_name, intrinsic, &mut occurrences)?;
    }
    occurrences.sort();
    occurrences.dedup();
    ensure!(
        occurrences.len() <= 1,
        "selection {record_name} binds intrinsic {intrinsic} through conflicting immediate patterns: {occurrences:?}"
    );
    Ok(occurrences.pop().unwrap_or_default())
}

fn collect_immediate_bindings(
    value: &Value,
    record_name: &str,
    intrinsic: &str,
    output: &mut Vec<Vec<ImportedImmediateBinding>>,
) -> Result<()> {
    match value {
        Value::Array(values) => {
            for value in values {
                collect_immediate_bindings(value, record_name, intrinsic, output)?;
            }
        }
        Value::Object(fields) => {
            let is_intrinsic_application = fields
                .get("operator")
                .and_then(|operator| operator.get("def"))
                .and_then(Value::as_str)
                == Some(intrinsic);
            if is_intrinsic_application {
                let mut bindings = Vec::new();
                if let Some(arguments) = fields.get("args") {
                    let arguments = arguments.as_array().with_context(|| {
                        format!(
                            "selection {record_name} intrinsic {intrinsic} has a non-array argument list"
                        )
                    })?;
                    for (argument_index, argument) in arguments.iter().enumerate() {
                        let pair = argument.as_array().with_context(|| {
                            format!(
                                "selection {record_name} intrinsic {intrinsic} argument {argument_index} is not a TableGen value/name pair"
                            )
                        })?;
                        let Some(value) = pair.first() else {
                            bail!(
                                "selection {record_name} intrinsic {intrinsic} argument {argument_index} has no value"
                            );
                        };
                        let value = match value {
                            Value::Number(value) => Some(value.as_i64().with_context(|| {
                                format!(
                                    "selection {record_name} intrinsic {intrinsic} argument {argument_index} has an integer outside i64"
                                )
                            })?),
                            Value::Bool(value) => Some(if *value { 1 } else { 0 }),
                            _ => None,
                        };
                        if let Some(value) = value {
                            bindings.push(ImportedImmediateBinding {
                                argument_index,
                                value,
                            });
                        }
                    }
                }
                output.push(bindings);
            }
            for value in fields.values() {
                collect_immediate_bindings(value, record_name, intrinsic, output)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn direct_wanted_intrinsic<'a>(record: &'a Value, wanted: &BTreeSet<&str>) -> Option<&'a str> {
    record
        .get("Intr")
        .and_then(|value| value.get("def"))
        .and_then(Value::as_str)
        .filter(|reference| wanted.contains(reference))
}

fn intrinsic_pattern_patfrag<'a>(
    record_name: &str,
    record: &Value,
    root: &'a Map<String, Value>,
) -> Result<&'a Value> {
    let patfrag_name = record
        .get("IntrinsicPattern")
        .and_then(|pattern| pattern.get("operator"))
        .and_then(|operator| operator.get("def"))
        .and_then(Value::as_str)
        .with_context(|| {
            format!("selection {record_name} has an Intr reference but no IntrinsicPattern PatFrag")
        })?;
    let patfrag = root.get(patfrag_name).with_context(|| {
        format!("selection {record_name} references missing PatFrag {patfrag_name}")
    })?;
    let superclasses = patfrag
        .get("!superclasses")
        .and_then(Value::as_array)
        .with_context(|| format!("selection {record_name} PatFrag has no superclass list"))?;
    ensure!(
        superclasses
            .iter()
            .any(|class| class.as_str() == Some("PatFrag")),
        "selection {record_name} IntrinsicPattern operator {patfrag_name} is not a PatFrag"
    );
    Ok(patfrag)
}

fn normalized_predicates(
    _record_name: &str,
    record: &Value,
    root: &Map<String, Value>,
) -> Result<Vec<String>> {
    let mut predicate_records = BTreeSet::new();
    if let Some(value) = record.get("Predicates") {
        collect_all_def_references(value, &mut predicate_records);
    }
    predicate_records
        .into_iter()
        .map(|predicate_name| {
            if !predicate_name.starts_with("anonymous_") {
                return Ok(predicate_name);
            }
            let Some(predicate) = root.get(&predicate_name) else {
                return Ok(predicate_name);
            };
            let Some(condition) = predicate.get("CondString").and_then(Value::as_str) else {
                return Ok(predicate_name);
            };
            let condition = normalize_whitespace(condition);
            Ok(if condition.is_empty() {
                predicate_name
            } else {
                condition
            })
        })
        .collect()
}

fn normalize_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn ldmatrix_address_space(
    record_name: &str,
    record: &Value,
    root: &Map<String, Value>,
) -> Result<ImportedAddressSpace> {
    let patfrag = intrinsic_pattern_patfrag(record_name, record, root)?;
    let predicate = patfrag
        .get("PredicateCode")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let shared = predicate.contains("llvm::ADDRESS_SPACE_SHARED");
    let generic = predicate.contains("llvm::ADDRESS_SPACE_GENERIC");
    match (shared, generic) {
        (true, false) => Ok(ImportedAddressSpace::Shared),
        (false, true) => Ok(ImportedAddressSpace::Generic),
        _ => bail!(
            "ldmatrix selection {record_name} has an unknown or ambiguous address-space PatFrag predicate: {:?}",
            normalize_whitespace(predicate)
        ),
    }
}

fn collect_actual_def_references(
    value: &Value,
    wanted: &BTreeSet<&str>,
    output: &mut BTreeSet<String>,
) {
    match value {
        Value::Array(values) => {
            for value in values {
                collect_actual_def_references(value, wanted, output);
            }
        }
        Value::Object(fields) => {
            if fields.get("kind").and_then(Value::as_str) == Some("def")
                && let Some(def) = fields.get("def").and_then(Value::as_str)
                && wanted.contains(def)
            {
                output.insert(def.to_owned());
            }
            for value in fields.values() {
                collect_actual_def_references(value, wanted, output);
            }
        }
        _ => {}
    }
}

fn collect_all_def_references(value: &Value, output: &mut BTreeSet<String>) {
    match value {
        Value::Array(values) => {
            for value in values {
                collect_all_def_references(value, output);
            }
        }
        Value::Object(fields) => {
            if fields.get("kind").and_then(Value::as_str) == Some("def")
                && let Some(def) = fields.get("def").and_then(Value::as_str)
            {
                output.insert(def.to_owned());
            }
            for value in fields.values() {
                collect_all_def_references(value, output);
            }
        }
        _ => {}
    }
}

fn type_list(record: &Value, field: &str) -> Result<Vec<String>> {
    let values = record
        .get(field)
        .and_then(Value::as_array)
        .with_context(|| format!("record has no {field} array"))?;
    values
        .iter()
        .map(|value| {
            let def = value
                .get("def")
                .and_then(Value::as_str)
                .with_context(|| format!("{field} entry is not a def reference"))?;
            Ok(normalize_type(def))
        })
        .collect()
}

fn normalize_type(def: &str) -> String {
    if let Some(width) = def
        .strip_prefix("llvm_i")
        .and_then(|rest| rest.strip_suffix("_ty"))
        && width.chars().all(|ch| ch.is_ascii_digit())
    {
        return format!("i{width}");
    }
    match def {
        "llvm_void_ty" => "void".into(),
        "llvm_half_ty" => "f16".into(),
        "llvm_bfloat_ty" => "bf16".into(),
        "llvm_float_ty" => "f32".into(),
        "llvm_double_ty" => "f64".into(),
        _ => def
            .strip_prefix("llvm_")
            .and_then(|name| name.strip_suffix("_ty"))
            .unwrap_or(def)
            .to_owned(),
    }
}

fn property_list(record: &Value, root: &Map<String, Value>) -> Result<Vec<String>> {
    let values = record
        .get("IntrProperties")
        .and_then(Value::as_array)
        .context("intrinsic record has no IntrProperties array")?;
    let mut properties = BTreeSet::new();
    for value in values {
        let def = value
            .get("def")
            .and_then(Value::as_str)
            .context("intrinsic property is not a def reference")?;
        let normalized = if def.starts_with("anonymous_") {
            root.get(def)
                .map(|property| normalize_anonymous_property(def, property))
                .unwrap_or_else(|| def.to_owned())
        } else {
            def.to_owned()
        };
        properties.insert(normalized);
    }
    Ok(properties.into_iter().collect())
}

fn normalize_anonymous_property(def: &str, property: &Value) -> String {
    let class = property
        .get("!superclasses")
        .and_then(Value::as_array)
        .and_then(|classes| classes.last())
        .and_then(Value::as_str)
        .unwrap_or(def);
    let target = match property.get("ArgNo").and_then(Value::as_u64) {
        Some(0) => "ret".to_owned(),
        Some(index) => format!("arg{}", index - 1),
        None => return class.to_owned(),
    };
    match class {
        "Range" => {
            let lower = property.get("Lower").and_then(Value::as_i64);
            let upper = property.get("Upper").and_then(Value::as_i64);
            match (lower, upper) {
                (Some(lower), Some(upper)) => format!("Range<{target},{lower},{upper}>"),
                _ => format!("Range<{target}>"),
            }
        }
        _ => format!("{class}<{target}>"),
    }
}

fn llvm_name(source_record: &str, record: &Value) -> Result<String> {
    let explicit = record
        .get("LLVMName")
        .and_then(Value::as_str)
        .context("intrinsic record has no LLVMName")?;
    if !explicit.is_empty() {
        return Ok(explicit.to_owned());
    }
    let stem = source_record
        .strip_prefix("int_")
        .with_context(|| format!("cannot derive LLVM name from {source_record}"))?;
    Ok(format!("llvm.{}", stem.replace('_', ".")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TestRepo(PathBuf);

    impl Drop for TestRepo {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn repo_with_lock(contents: &str) -> TestRepo {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "cuda-intrinsics-upstream-lock-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(root.join("intrinsics")).unwrap();
        fs::write(root.join("intrinsics/upstream.lock"), contents).unwrap();
        TestRepo(root)
    }

    fn committed_lock() -> (&'static str, UpstreamLock) {
        let text = include_str!("../../../intrinsics/upstream.lock");
        let lock = toml::from_str(text).unwrap();
        (text, lock)
    }

    #[test]
    fn upstream_lock_requires_rust_toolchain_table() {
        let (original, parsed) = committed_lock();
        let commit = format!("commit_hash = \"{}\"", parsed.rust_toolchain.commit_hash);
        let mutated = original
            .replace("[rust_toolchain]", "")
            .replace(&commit, "");
        assert_ne!(mutated, original);
        let repo = repo_with_lock(&mutated);
        let error = read_upstream_lock(&repo.0).unwrap_err();
        assert!(format!("{error:#}").contains("rust_toolchain"));
    }

    #[test]
    fn upstream_lock_rejects_malformed_rust_toolchain_commit_hash() {
        let (original, parsed) = committed_lock();
        let commit = format!("commit_hash = \"{}\"", parsed.rust_toolchain.commit_hash);
        let mutated = original.replace(&commit, "commit_hash = \"not-a-commit\"");
        assert_ne!(mutated, original);
        let repo = repo_with_lock(&mutated);
        let error = read_upstream_lock(&repo.0).unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("commit_hash"), "{message}");
        assert!(message.contains("40 lowercase hexadecimal"), "{message}");
    }

    #[test]
    fn joins_selection_by_def_reference_not_printable_text() {
        let wanted = BTreeSet::from(["int_nvvm_read_ptx_sreg_tid_x"]);
        let root = Map::from_iter([
            (
                "real".into(),
                json!({
                    "Pattern": [{"operator": {
                        "kind": "def",
                        "def": "int_nvvm_read_ptx_sreg_tid_x",
                        "printable": "unrelated spelling"
                    }}],
                    "AsmString": "mov.u32 $d, %tid.x;",
                    "Predicates": []
                }),
            ),
            (
                "text_only".into(),
                json!({
                    "Pattern": [{"printable": "int_nvvm_read_ptx_sreg_tid_x"}],
                    "AsmString": "wrong"
                }),
            ),
        ]);
        let index = selection_index(&root, &wanted).unwrap();
        assert_eq!(index["int_nvvm_read_ptx_sreg_tid_x"].len(), 1);
        assert_eq!(
            index["int_nvvm_read_ptx_sreg_tid_x"][0].source_record,
            "real"
        );
        assert!(
            index["int_nvvm_read_ptx_sreg_tid_x"][0]
                .constraints
                .is_empty()
        );
    }

    #[test]
    fn preserves_dotprod_low_and_high_immediate_bindings() {
        const INTRINSIC: &str = "int_nvvm_idp2a_s_s";
        let wanted = BTreeSet::from([INTRINSIC]);
        let selection = |value: i64, asm: &str| {
            json!({
                "Pattern": [{
                    "kind": "dag",
                    "operator": {"kind": "def", "def": "set"},
                    "args": [
                        [{"kind": "def", "def": "i32"}, "dst"],
                        [{
                            "kind": "dag",
                            "operator": {"kind": "def", "def": INTRINSIC},
                            "args": [
                                [{"kind": "def", "def": "i32"}, "a"],
                                [{"kind": "def", "def": "i32"}, "b"],
                                [value, null],
                                [{"kind": "def", "def": "i32"}, "c"]
                            ]
                        }, null]
                    ]
                }],
                "AsmString": asm,
                "Predicates": [{"kind": "def", "def": "hasDotInstructions"}]
            })
        };
        let root = Map::from_iter([
            (
                "DOT2_lo_ss".into(),
                selection(0, "dp2a.lo.s32.s32 $dst, $a, $b, $c;"),
            ),
            (
                "DOT2_hi_ss".into(),
                selection(-1, "dp2a.hi.s32.s32 $dst, $a, $b, $c;"),
            ),
        ]);

        let selections = &selection_index(&root, &wanted).unwrap()[INTRINSIC];
        let low = selections
            .iter()
            .find(|selection| selection.source_record == "DOT2_lo_ss")
            .unwrap();
        let high = selections
            .iter()
            .find(|selection| selection.source_record == "DOT2_hi_ss")
            .unwrap();
        assert_eq!(
            low.constraints.immediate_bindings,
            [ImportedImmediateBinding {
                argument_index: 2,
                value: 0,
            }]
        );
        assert_eq!(
            high.constraints.immediate_bindings,
            [ImportedImmediateBinding {
                argument_index: 2,
                value: -1,
            }]
        );
        assert_ne!(low.constraints, high.constraints);
    }

    #[test]
    fn immediate_bindings_serialize_in_argument_order() {
        const INTRINSIC: &str = "int_nvvm_test";
        let record = json!({
            "Pattern": [{
                "kind": "dag",
                "operator": {"kind": "def", "def": INTRINSIC},
                "args": [
                    [{"kind": "def", "def": "i32"}, "a"],
                    [0, null],
                    [{"kind": "def", "def": "i32"}, "b"],
                    [-1, null]
                ]
            }]
        });
        let bindings = selection_immediate_bindings("TEST", &record, INTRINSIC).unwrap();
        assert_eq!(
            serde_json::to_value(bindings).unwrap(),
            json!([
                {"argument_index": 1, "value": 0},
                {"argument_index": 3, "value": -1}
            ])
        );
    }

    #[test]
    fn rejects_conflicting_immediate_bindings_in_one_selection() {
        const INTRINSIC: &str = "int_nvvm_test";
        let application = |value: i64| {
            json!({
                "kind": "dag",
                "operator": {"kind": "def", "def": INTRINSIC},
                "args": [[value, null]]
            })
        };
        let record = json!({"Pattern": [application(0), application(-1)]});
        let error = selection_immediate_bindings("TEST", &record, INTRINSIC)
            .unwrap_err()
            .to_string();
        assert!(error.contains("conflicting immediate patterns"), "{error}");
    }

    #[test]
    fn joins_exact_intrinsic_pattern_when_pattern_is_empty() {
        const INTRINSIC: &str = "int_nvvm_mma_test";
        let wanted = BTreeSet::from([INTRINSIC]);
        let root = Map::from_iter([(
            "MMA_TEST".into(),
            json!({
                "Pattern": [],
                "Intr": {"kind": "def", "def": INTRINSIC},
                "IntrinsicPattern": {
                    "kind": "dag",
                    "operator": {"kind": "def", "def": INTRINSIC},
                    "args": [
                        [{"kind": "def", "def": "B32"}, "a"],
                        [7, null]
                    ]
                },
                "AsmString": "mma.sync.test;",
                "Predicates": [{"kind": "def", "def": "hasMmaTest"}]
            }),
        )]);

        let selection = &selection_index(&root, &wanted).unwrap()[INTRINSIC][0];
        assert_eq!(selection.source_record, "MMA_TEST");
        assert_eq!(
            selection.constraints.immediate_bindings,
            [ImportedImmediateBinding {
                argument_index: 1,
                value: 7,
            }]
        );
    }

    #[test]
    fn rejects_mismatched_exact_intrinsic_pattern() {
        const INTRINSIC: &str = "int_nvvm_mma_test";
        let wanted = BTreeSet::from([INTRINSIC]);
        let root = Map::from_iter([(
            "MMA_TEST".into(),
            json!({
                "Pattern": [],
                "Intr": {"kind": "def", "def": INTRINSIC},
                "IntrinsicPattern": {
                    "kind": "dag",
                    "operator": {"kind": "def", "def": "int_nvvm_mma_other"}
                },
                "AsmString": "mma.sync.test;",
                "Predicates": []
            }),
        )]);

        assert!(selection_index(&root, &wanted).unwrap().is_empty());
    }

    fn ldmatrix_fixture(patfrag_intrinsic: &str, shared_predicate: &str) -> Map<String, Value> {
        const INTRINSIC: &str = "int_nvvm_ldmatrix_sync_aligned_m8n8_x4_b16";
        Map::from_iter([
            (
                "shared_selection".into(),
                json!({
                    "Pattern": [],
                    "Intr": {"kind": "def", "def": INTRINSIC},
                    "IntrinsicPattern": {
                        "kind": "dag",
                        "operator": {"kind": "def", "def": "shared_patfrag"}
                    },
                    "AsmString": "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{$r0, $r1, $r2, $r3}}, [$src];",
                    "Predicates": [
                        {"kind": "def", "def": "anonymous_ptx65"},
                        {"kind": "def", "def": "anonymous_sm75"}
                    ]
                }),
            ),
            (
                "generic_selection".into(),
                json!({
                    "Pattern": [],
                    "Intr": {"kind": "def", "def": INTRINSIC},
                    "IntrinsicPattern": {
                        "kind": "dag",
                        "operator": {"kind": "def", "def": "generic_patfrag"}
                    },
                    "AsmString": "ldmatrix.sync.aligned.m8n8.x4.b16 {{$r0, $r1, $r2, $r3}}, [$src];",
                    "Predicates": [
                        {"kind": "def", "def": "anonymous_ptx65"},
                        {"kind": "def", "def": "anonymous_sm75"}
                    ]
                }),
            ),
            (
                "shared_patfrag".into(),
                json!({
                    "!superclasses": ["SDPatternOperator", "PatFrags", "PatFrag"],
                    "Fragments": [{
                        "kind": "dag",
                        "operator": {"kind": "def", "def": patfrag_intrinsic}
                    }],
                    "PredicateCode": shared_predicate
                }),
            ),
            (
                "generic_patfrag".into(),
                json!({
                    "!superclasses": ["SDPatternOperator", "PatFrags", "PatFrag"],
                    "Fragments": [{
                        "kind": "dag",
                        "operator": {"kind": "def", "def": INTRINSIC}
                    }],
                    "PredicateCode": "return cast<MemSDNode>(N)->getAddressSpace() == llvm::ADDRESS_SPACE_GENERIC;"
                }),
            ),
            (
                "anonymous_ptx65".into(),
                json!({
                    "CondString": "  Subtarget->getPTXVersion()   >= 65  "
                }),
            ),
            (
                "anonymous_sm75".into(),
                json!({
                    "CondString": "\n Subtarget->getSmVersion() >= 75\n"
                }),
            ),
        ])
    }

    #[test]
    fn joins_ldmatrix_intrinsic_through_address_space_patfrags() {
        const INTRINSIC: &str = "int_nvvm_ldmatrix_sync_aligned_m8n8_x4_b16";
        let wanted = BTreeSet::from([INTRINSIC]);
        let root = ldmatrix_fixture(
            INTRINSIC,
            "return cast<MemSDNode>(N)->getAddressSpace() == llvm::ADDRESS_SPACE_SHARED;",
        );

        let index = selection_index(&root, &wanted).unwrap();
        let selections = &index[INTRINSIC];
        assert_eq!(selections.len(), 2);

        let shared = selections
            .iter()
            .find(|selection| {
                selection.constraints.address_space == Some(ImportedAddressSpace::Shared)
            })
            .unwrap();
        assert!(shared.asm.contains(".x4.shared.b16"));
        assert_eq!(
            BTreeSet::from_iter(shared.predicates.iter().map(String::as_str)),
            BTreeSet::from([
                "Subtarget->getPTXVersion() >= 65",
                "Subtarget->getSmVersion() >= 75",
            ])
        );

        let generic = selections
            .iter()
            .find(|selection| {
                selection.constraints.address_space == Some(ImportedAddressSpace::Generic)
            })
            .unwrap();
        assert!(generic.asm.contains(".x4.b16"));
        assert!(!generic.asm.contains(".shared"));
    }

    #[test]
    fn rejects_ldmatrix_when_patfrag_names_a_different_intrinsic() {
        const INTRINSIC: &str = "int_nvvm_ldmatrix_sync_aligned_m8n8_x4_b16";
        let wanted = BTreeSet::from([INTRINSIC]);
        let root = ldmatrix_fixture(
            "int_nvvm_ldmatrix_sync_aligned_m8n8_x2_b16",
            "return cast<MemSDNode>(N)->getAddressSpace() == llvm::ADDRESS_SPACE_SHARED;",
        );

        let error = selection_index(&root, &wanted).unwrap_err().to_string();
        assert!(error.contains("names intrinsic"), "{error}");
        assert!(error.contains("PatFrag references"), "{error}");
    }

    #[test]
    fn rejects_unknown_ldmatrix_address_space_patfrag() {
        const INTRINSIC: &str = "int_nvvm_ldmatrix_sync_aligned_m8n8_x4_b16";
        let wanted = BTreeSet::from([INTRINSIC]);
        let root = ldmatrix_fixture(
            INTRINSIC,
            "return cast<MemSDNode>(N)->getAddressSpace() == llvm::ADDRESS_SPACE_GLOBAL;",
        );

        let error = selection_index(&root, &wanted).unwrap_err().to_string();
        assert!(
            error.contains("unknown or ambiguous address-space"),
            "{error}"
        );
    }

    #[test]
    fn does_not_treat_bare_intr_field_as_a_selection() {
        let wanted = BTreeSet::from(["int_nvvm_fmin_f16"]);
        let root = Map::from_iter([(
            "source_only".into(),
            json!({
                "Pattern": [],
                "Intr": {"kind": "def", "def": "int_nvvm_fmin_f16"},
                "AsmString": null
            }),
        )]);

        assert!(selection_index(&root, &wanted).unwrap().is_empty());
    }

    #[test]
    fn does_not_generalize_ldmatrix_patfrag_recovery_to_other_families() {
        let wanted = BTreeSet::from(["int_nvvm_wmma_load"]);
        let root = Map::from_iter([
            (
                "wmma_selection".into(),
                json!({
                    "Pattern": [],
                    "Intr": {"kind": "def", "def": "int_nvvm_wmma_load"},
                    "IntrinsicPattern": {
                        "kind": "dag",
                        "operator": {"kind": "def", "def": "wmma_patfrag"}
                    },
                    "AsmString": "wmma.load.d.sync...",
                    "Predicates": []
                }),
            ),
            (
                "wmma_patfrag".into(),
                json!({
                    "!superclasses": ["SDPatternOperator", "PatFrags", "PatFrag"],
                    "Fragments": [{
                        "kind": "dag",
                        "operator": {"kind": "def", "def": "int_nvvm_wmma_load"}
                    }],
                    "PredicateCode": "return true;"
                }),
            ),
        ]);

        // Other PatFrag families need their own constraint model before their
        // address-space-specific selections can be imported safely.
        assert!(selection_index(&root, &wanted).unwrap().is_empty());
    }

    #[test]
    fn empty_selection_constraints_preserve_the_existing_json_shape() {
        let selection: ImportedSelection = serde_json::from_value(json!({
            "source_record": "INT_PTX_SREG_TID_x",
            "asm": "mov.u32 $d, %tid.x;",
            "predicates": []
        }))
        .unwrap();
        assert!(selection.constraints.is_empty());
        let serialized = serde_json::to_value(selection).unwrap();
        assert!(serialized.get("constraints").is_none());
    }

    #[test]
    fn normalizes_integer_types() {
        assert_eq!(normalize_type("llvm_i32_ty"), "i32");
        assert_eq!(normalize_type("llvm_anyptr_ty"), "anyptr");
    }

    #[test]
    fn preserves_parameterized_anonymous_properties() {
        assert_eq!(
            normalize_anonymous_property(
                "anonymous_1",
                &json!({"!superclasses": ["IntrinsicProperty", "NoUndef"], "ArgNo": 0})
            ),
            "NoUndef<ret>"
        );
        assert_eq!(
            normalize_anonymous_property(
                "anonymous_2",
                &json!({
                    "!superclasses": ["IntrinsicProperty", "Range"],
                    "ArgNo": 0,
                    "Lower": 0,
                    "Upper": 1024
                })
            ),
            "Range<ret,0,1024>"
        );
    }
}
