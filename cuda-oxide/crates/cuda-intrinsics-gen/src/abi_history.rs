/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#[cfg(test)]
use crate::model::AbiRawRustSignature;
use crate::model::{AbiLedgerEntry, AbiLedgerFile};
use crate::resolve::{resolve, validate_operation_key};
use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::process::Command;

pub fn run(repo_root: &Path, base_ref: &str) -> Result<()> {
    ensure!(!base_ref.trim().is_empty(), "base ref must not be empty");
    resolve(repo_root)?;
    verify_git_ref(repo_root, base_ref)?;
    let base_paths = git_ledger_paths(repo_root, base_ref)?;
    let current_paths = current_ledger_paths(repo_root)?;

    for relative in &base_paths {
        ensure!(
            current_paths.contains(relative),
            "ABI ledger {relative} existed in {base_ref} and cannot be deleted"
        );
        let base = read_git_ledger(repo_root, base_ref, relative)?;
        let current = read_current_ledger(repo_root, relative)?;
        validate_ledger_shape(relative, &base)?;
        validate_ledger_shape(relative, &current)?;
        validate_history(&base, &current)
            .with_context(|| format!("compare {relative} against {base_ref}"))?;
        println!(
            "{relative} preserves {} base entries and has {} total entries",
            base.entries.len(),
            current.entries.len()
        );
    }

    for relative in current_paths.difference(&base_paths) {
        let current = read_current_ledger(repo_root, relative)?;
        validate_ledger_shape(relative, &current)?;
        ensure!(
            current.entries.iter().all(|entry| entry.status == "active"),
            "a first-introduction ABI ledger cannot begin with tombstones"
        );
        println!(
            "{relative} is absent from {base_ref}; accepted as the first ABI-v{} ledger introduction",
            current.intrinsic_abi
        );
    }
    Ok(())
}

fn is_ledger_path(path: &str) -> bool {
    path.strip_prefix("intrinsics/abi-v")
        .and_then(|suffix| suffix.strip_suffix(".toml"))
        .is_some_and(|version| {
            !version.is_empty() && version.bytes().all(|byte| byte.is_ascii_digit())
        })
}

fn validate_ledger_shape(relative: &str, ledger: &AbiLedgerFile) -> Result<()> {
    let path_version = relative
        .strip_prefix("intrinsics/abi-v")
        .and_then(|suffix| suffix.strip_suffix(".toml"))
        .and_then(|version| version.parse::<u32>().ok())
        .with_context(|| format!("invalid ABI ledger path {relative}"))?;
    ensure!(
        ledger.schema == 1,
        "{relative} has unsupported schema {}",
        ledger.schema
    );
    ensure!(
        ledger.intrinsic_abi == path_version,
        "{relative} declares ABI v{}",
        ledger.intrinsic_abi
    );
    ensure!(!ledger.entries.is_empty(), "{relative} contains no entries");
    let mut previous: Option<&str> = None;
    let mut catalog_ids = BTreeSet::new();
    let mut operation_keys = BTreeSet::new();
    for entry in &ledger.entries {
        ensure!(
            entry.abi_id.len() == 5
                && entry.abi_id.starts_with('i')
                && entry.abi_id[1..].bytes().all(|byte| byte.is_ascii_digit()),
            "{relative} has malformed ABI ID {}",
            entry.abi_id
        );
        if let Some(previous) = previous {
            ensure!(
                previous < entry.abi_id.as_str(),
                "{relative} IDs are not unique and ascending: {} follows {previous}",
                entry.abi_id
            );
        }
        ensure!(
            matches!(entry.status.as_str(), "active" | "tombstone"),
            "{relative} entry {} has invalid status {:?}",
            entry.abi_id,
            entry.status
        );
        ensure!(
            !entry.catalog_id.is_empty(),
            "{relative} entry {} has an empty catalog ID",
            entry.abi_id
        );
        ensure!(
            catalog_ids.insert(&entry.catalog_id),
            "{relative} has duplicate catalog ID {}",
            entry.catalog_id
        );
        validate_operation_key(&entry.operation_key)
            .with_context(|| format!("{relative} entry {}", entry.abi_id))?;
        ensure!(
            operation_keys.insert(&entry.operation_key),
            "{relative} has duplicate operation key {}",
            entry.operation_key
        );
        ensure!(
            !entry.raw_rust_signature.result.is_empty()
                && entry
                    .raw_rust_signature
                    .arguments
                    .iter()
                    .all(|argument| !argument.is_empty()),
            "{relative} entry {} has an incomplete raw Rust signature",
            entry.abi_id
        );
        previous = Some(&entry.abi_id);
    }
    Ok(())
}

fn current_ledger_paths(repo_root: &Path) -> Result<BTreeSet<String>> {
    let directory = repo_root.join("intrinsics");
    let mut paths = BTreeSet::new();
    for entry in
        fs::read_dir(&directory).with_context(|| format!("read {}", directory.display()))?
    {
        let entry = entry?;
        let relative = format!("intrinsics/{}", entry.file_name().to_string_lossy());
        if is_ledger_path(&relative) {
            paths.insert(relative);
        }
    }
    ensure!(!paths.is_empty(), "no intrinsic ABI ledger files found");
    Ok(paths)
}

fn git_ledger_paths(repo_root: &Path, base_ref: &str) -> Result<BTreeSet<String>> {
    let output = Command::new("git")
        .current_dir(repo_root)
        .args([
            "ls-tree",
            "-r",
            "--name-only",
            base_ref,
            "--",
            ":(top)intrinsics",
            ":(top)cuda-oxide/intrinsics",
        ])
        .output()
        .with_context(|| format!("list ABI ledgers in {base_ref}"))?;
    ensure!(output.status.success(), "git ls-tree failed for {base_ref}");
    Ok(String::from_utf8(output.stdout)
        .context("git ls-tree returned non-UTF-8 paths")?
        .lines()
        .map(|path| path.trim_start_matches("../"))
        .map(|path| path.strip_prefix("cuda-oxide/").unwrap_or(path))
        .filter(|path| is_ledger_path(path))
        .map(str::to_owned)
        .collect())
}

fn read_current_ledger(repo_root: &Path, relative: &str) -> Result<AbiLedgerFile> {
    let path = repo_root.join(relative);
    let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

fn read_git_ledger(repo_root: &Path, base_ref: &str, relative: &str) -> Result<AbiLedgerFile> {
    let mut object = format!("{base_ref}:cuda-oxide/{relative}");
    let mut output = Command::new("git")
        .current_dir(repo_root)
        .args(["show", &object])
        .output()
        .with_context(|| format!("read {object}"))?;
    if !output.status.success() {
        object = format!("{base_ref}:{relative}");
        output = Command::new("git")
            .current_dir(repo_root)
            .args(["show", &object])
            .output()
            .with_context(|| format!("read {object}"))?;
    }
    ensure!(output.status.success(), "git show {object} failed");
    let text = String::from_utf8(output.stdout).context("base ABI ledger is not UTF-8")?;
    toml::from_str(&text).with_context(|| format!("parse {object}"))
}

fn verify_git_ref(repo_root: &Path, base_ref: &str) -> Result<()> {
    let commit = format!("{base_ref}^{{commit}}");
    let status = Command::new("git")
        .current_dir(repo_root)
        .args(["cat-file", "-e", &commit])
        .status()
        .with_context(|| format!("resolve base ref {base_ref}"))?;
    ensure!(status.success(), "base ref {base_ref:?} is not a commit");
    Ok(())
}

fn validate_history(base: &AbiLedgerFile, current: &AbiLedgerFile) -> Result<()> {
    ensure!(
        base.schema == current.schema,
        "ABI ledger schema changed from {} to {}",
        base.schema,
        current.schema
    );
    ensure!(
        base.intrinsic_abi == current.intrinsic_abi,
        "ABI ledger version changed from v{} to v{}",
        base.intrinsic_abi,
        current.intrinsic_abi
    );
    ensure!(
        current.entries.len() >= base.entries.len(),
        "ABI ledger removed entries: base has {}, current has {}",
        base.entries.len(),
        current.entries.len()
    );

    for (index, base_entry) in base.entries.iter().enumerate() {
        let current_entry = &current.entries[index];
        ensure!(
            current_entry.abi_id == base_entry.abi_id,
            "ABI ledger entry {} changed position or was replaced by {}",
            base_entry.abi_id,
            current_entry.abi_id
        );
        ensure_identity_unchanged(base_entry, current_entry)?;
        match (base_entry.status.as_str(), current_entry.status.as_str()) {
            ("active", "active" | "tombstone") | ("tombstone", "tombstone") => {}
            ("tombstone", "active") => bail!(
                "tombstoned ABI ID {} cannot become active again",
                base_entry.abi_id
            ),
            (before, after) => bail!(
                "invalid ABI status transition for {}: {before:?} -> {after:?}",
                base_entry.abi_id
            ),
        }
    }

    if let Some(last_base) = base.entries.last() {
        let mut previous = last_base.abi_id.as_str();
        for entry in &current.entries[base.entries.len()..] {
            ensure!(
                entry.abi_id.as_str() > previous,
                "new ABI ID {} must be higher than existing maximum {}",
                entry.abi_id,
                previous
            );
            ensure!(
                entry.status == "active",
                "new ABI ID {} must be introduced as active",
                entry.abi_id
            );
            previous = &entry.abi_id;
        }
    }
    Ok(())
}

fn ensure_identity_unchanged(base: &AbiLedgerEntry, current: &AbiLedgerEntry) -> Result<()> {
    let comparisons = [
        (
            "catalog ID",
            base.catalog_id.as_str(),
            current.catalog_id.as_str(),
        ),
        (
            "operation key",
            base.operation_key.as_str(),
            current.operation_key.as_str(),
        ),
    ];
    for (field, before, after) in comparisons {
        ensure!(
            before == after,
            "ABI ID {} changed {field}: {before:?} -> {after:?}",
            base.abi_id
        );
    }
    ensure!(
        base.raw_rust_signature == current.raw_rust_signature,
        "ABI ID {} changed its raw Rust signature: {:?} -> {:?}",
        base.abi_id,
        base.raw_rust_signature,
        current.raw_rust_signature
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str) -> AbiLedgerEntry {
        AbiLedgerEntry {
            abi_id: id.into(),
            status: "active".into(),
            catalog_id: format!("catalog_{id}"),
            operation_key: format!("test.operation.{id}"),
            raw_rust_signature: AbiRawRustSignature {
                safe: true,
                arguments: vec![],
                result: "u32".into(),
            },
        }
    }

    fn ledger(entries: Vec<AbiLedgerEntry>) -> AbiLedgerFile {
        AbiLedgerFile {
            schema: 1,
            intrinsic_abi: 1,
            entries,
        }
    }

    #[test]
    fn exact_entries_may_be_preserved_and_higher_ids_appended() {
        let base = ledger(vec![entry("i0001")]);
        let current = ledger(vec![entry("i0001"), entry("i0002")]);
        validate_history(&base, &current).unwrap();
    }

    #[test]
    fn active_entry_may_be_tombstoned_without_identity_changes() {
        let base_entry = entry("i0001");
        let mut current_entry = base_entry.clone();
        current_entry.status = "tombstone".into();
        validate_history(&ledger(vec![base_entry]), &ledger(vec![current_entry])).unwrap();
    }

    #[test]
    fn identity_changes_and_tombstone_resurrection_are_rejected() {
        let base_entry = entry("i0001");
        let mut changed = base_entry.clone();
        changed.operation_key = "test.reassigned.i0001".into();
        assert!(
            validate_history(&ledger(vec![base_entry.clone()]), &ledger(vec![changed]))
                .unwrap_err()
                .to_string()
                .contains("operation key")
        );

        let mut tombstone = base_entry.clone();
        tombstone.status = "tombstone".into();
        assert!(
            validate_history(&ledger(vec![tombstone]), &ledger(vec![base_entry]))
                .unwrap_err()
                .to_string()
                .contains("cannot become active")
        );
    }

    #[test]
    fn raw_rust_safety_arguments_and_result_are_immutable() {
        let base_entry = entry("i0001");

        let mut changed_safety = base_entry.clone();
        changed_safety.raw_rust_signature.safe = false;
        assert!(
            validate_history(
                &ledger(vec![base_entry.clone()]),
                &ledger(vec![changed_safety])
            )
            .unwrap_err()
            .to_string()
            .contains("raw Rust signature")
        );

        let mut changed_arguments = base_entry.clone();
        changed_arguments.raw_rust_signature.arguments = vec!["u32".into()];
        assert!(
            validate_history(
                &ledger(vec![base_entry.clone()]),
                &ledger(vec![changed_arguments])
            )
            .unwrap_err()
            .to_string()
            .contains("raw Rust signature")
        );

        let mut changed_result = base_entry.clone();
        changed_result.raw_rust_signature.result = "u64".into();
        assert!(
            validate_history(&ledger(vec![base_entry]), &ledger(vec![changed_result]))
                .unwrap_err()
                .to_string()
                .contains("raw Rust signature")
        );
    }

    #[test]
    fn entries_cannot_be_removed_reordered_or_backfilled() {
        let base = ledger(vec![entry("i0001"), entry("i0002")]);
        assert!(validate_history(&base, &ledger(vec![entry("i0001")])).is_err());
        assert!(
            validate_history(
                &ledger(vec![entry("i0002")]),
                &ledger(vec![entry("i0001"), entry("i0002")]),
            )
            .is_err()
        );
    }
}
