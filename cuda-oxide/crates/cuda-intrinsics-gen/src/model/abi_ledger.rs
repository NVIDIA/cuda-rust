/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbiLedgerFile {
    pub schema: u32,
    pub intrinsic_abi: u32,
    #[serde(rename = "entry")]
    pub entries: Vec<AbiLedgerEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbiLedgerEntry {
    pub abi_id: String,
    pub status: String,
    pub catalog_id: String,
    pub operation_key: String,
    pub raw_rust_signature: AbiRawRustSignature,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbiRawRustSignature {
    pub safe: bool,
    pub arguments: Vec<String>,
    pub result: String,
}
