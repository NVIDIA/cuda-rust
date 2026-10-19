/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    AbiLedgerEntry, AbiLedgerFile, AbiRawRustSignature, OverlayFile, OverlayIntrinsic,
};
use anyhow::{Context, Result, bail, ensure};
use std::collections::{BTreeMap, BTreeSet};

use super::guards::*;

pub(super) struct AbiLedgerIndex<'a> {
    by_catalog_id: BTreeMap<&'a str, &'a AbiLedgerEntry>,
}

impl<'a> AbiLedgerIndex<'a> {
    fn new(ledger: &'a AbiLedgerFile) -> Result<Self> {
        let mut by_catalog_id = BTreeMap::new();
        for entry in &ledger.entries {
            ensure!(
                by_catalog_id
                    .insert(entry.catalog_id.as_str(), entry)
                    .is_none(),
                "duplicate ABI ledger catalog ID: {}",
                entry.catalog_id
            );
        }
        Ok(Self { by_catalog_id })
    }

    fn active(&self, catalog_id: &str) -> Result<&'a AbiLedgerEntry> {
        let entry = self
            .by_catalog_id
            .get(catalog_id)
            .copied()
            .with_context(|| format!("generated intrinsic {catalog_id} has no ABI ledger entry"))?;
        ensure!(
            entry.status == "active",
            "generated intrinsic {catalog_id} maps to non-active ABI ledger entry {}",
            entry.abi_id
        );
        Ok(entry)
    }
}

pub(super) fn bind_generated_abi_ids(
    overlay: &mut OverlayFile,
    ledger: &AbiLedgerFile,
) -> Result<()> {
    let index = AbiLedgerIndex::new(ledger)?;
    for record in overlay
        .intrinsics
        .iter_mut()
        .filter(|record| record.abi_id.is_empty())
    {
        let entry = index.active(&record.id)?;
        ensure!(
            entry.operation_key == record.operation_key,
            "generated intrinsic {} operation key mismatch: ledger {:?}, derived {:?}",
            record.id,
            entry.operation_key,
            record.operation_key
        );
        let derived_signature = raw_rust_signature(record);
        ensure!(
            entry.raw_rust_signature == derived_signature,
            "generated intrinsic {} raw Rust signature mismatch: ledger {:?}, derived {:?}",
            record.id,
            entry.raw_rust_signature,
            derived_signature
        );
        record.abi_id.clone_from(&entry.abi_id);
    }
    Ok(())
}

pub(super) fn validate_abi_ledger(overlay: &OverlayFile, ledger: &AbiLedgerFile) -> Result<()> {
    ensure!(
        ledger.schema == 1,
        "unsupported ABI ledger schema {}",
        ledger.schema
    );
    ensure!(
        ledger.intrinsic_abi == overlay.intrinsic_abi,
        "ABI ledger v{} does not match overlay ABI v{}",
        ledger.intrinsic_abi,
        overlay.intrinsic_abi
    );
    ensure!(!ledger.entries.is_empty(), "ABI ledger contains no entries");

    let overlay_by_abi_id: BTreeMap<_, _> = overlay
        .intrinsics
        .iter()
        .map(|record| (record.abi_id.as_str(), record))
        .collect();
    let mut abi_ids = BTreeSet::new();
    let mut catalog_ids = BTreeSet::new();
    let mut operation_keys = BTreeSet::new();
    let mut previous_abi_id: Option<&str> = None;
    for entry in &ledger.entries {
        validate_abi_id(&entry.abi_id)?;
        if let Some(previous) = previous_abi_id {
            ensure!(
                previous < entry.abi_id.as_str(),
                "ABI ledger IDs must be unique and append-only in ascending order: {} follows {}",
                entry.abi_id,
                previous
            );
        }
        previous_abi_id = Some(&entry.abi_id);
        insert_unique(&mut abi_ids, &entry.abi_id, "ABI ledger ID")?;
        insert_unique(&mut catalog_ids, &entry.catalog_id, "ABI ledger catalog ID")?;
        validate_operation_key(&entry.operation_key)?;
        insert_unique(
            &mut operation_keys,
            &entry.operation_key,
            "ABI ledger operation key",
        )?;
        ensure!(
            !entry.catalog_id.is_empty()
                && !entry.raw_rust_signature.result.is_empty()
                && entry
                    .raw_rust_signature
                    .arguments
                    .iter()
                    .all(|argument| !argument.is_empty()),
            "ABI ledger entry {} has incomplete identity data",
            entry.abi_id
        );

        let overlay_record = overlay_by_abi_id.get(entry.abi_id.as_str()).copied();
        match entry.status.as_str() {
            "active" => {
                let record = overlay_record.with_context(|| {
                    format!(
                        "active ABI ledger entry {} ({}) has no overlay record",
                        entry.abi_id, entry.catalog_id
                    )
                })?;
                validate_active_ledger_entry(entry, record)?;
            }
            "tombstone" => ensure!(
                overlay_record.is_none(),
                "tombstoned ABI ID {} cannot reappear in the overlay",
                entry.abi_id
            ),
            status => bail!(
                "ABI ledger entry {} has unsupported status {status:?}; expected active or tombstone",
                entry.abi_id
            ),
        }
    }

    for record in &overlay.intrinsics {
        ensure!(
            abi_ids.contains(&record.abi_id),
            "overlay intrinsic {} uses ABI ID {} with no ledger entry",
            record.id,
            record.abi_id
        );
    }
    Ok(())
}

pub(super) fn validate_active_ledger_entry(
    entry: &AbiLedgerEntry,
    record: &OverlayIntrinsic,
) -> Result<()> {
    let comparisons = [
        ("catalog ID", entry.catalog_id.as_str(), record.id.as_str()),
        (
            "operation key",
            entry.operation_key.as_str(),
            record.operation_key.as_str(),
        ),
    ];
    for (field, ledger_value, overlay_value) in comparisons {
        ensure!(
            ledger_value == overlay_value,
            "ABI ledger {} {field} mismatch: ledger {ledger_value:?}, overlay {overlay_value:?}",
            entry.abi_id
        );
    }
    let expected_signature = raw_rust_signature(record);
    ensure!(
        entry.raw_rust_signature == expected_signature,
        "ABI ledger {} raw Rust signature mismatch: ledger {:?}, overlay {:?}",
        entry.abi_id,
        entry.raw_rust_signature,
        expected_signature
    );
    Ok(())
}

pub(super) fn raw_rust_signature(record: &OverlayIntrinsic) -> AbiRawRustSignature {
    AbiRawRustSignature {
        safe: record.safe,
        arguments: record.rust_arguments.clone(),
        result: record.rust_result.clone(),
    }
}

pub(super) fn validate_abi_id(abi_id: &str) -> Result<()> {
    ensure!(
        abi_id.len() == 5
            && abi_id.starts_with('i')
            && abi_id[1..].bytes().all(|byte| byte.is_ascii_digit()),
        "intrinsic ABI ID `{abi_id}` must use the stable `iNNNN` form"
    );
    Ok(())
}

pub(crate) fn validate_operation_key(operation_key: &str) -> Result<()> {
    ensure!(
        !operation_key.is_empty()
            && operation_key.split('.').all(|segment| {
                !segment.is_empty()
                    && segment.bytes().all(|byte| {
                        byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_'
                    })
            }),
        "intrinsic operation key `{operation_key}` must contain dot-separated lowercase semantic components"
    );
    Ok(())
}

pub(super) fn canonical_rust_path(intrinsic_abi: u32, abi_id: &str) -> String {
    format!("cuda_intrinsics::__cuda_oxide_intrinsic_abi_v{intrinsic_abi}::{abi_id}")
}
