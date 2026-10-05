/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tile IR operations.
//!
//! Each operation corresponds to an opcode in `BytecodeOpcodes.td`.

use super::attr::Attribute;
use super::location::Location;
use super::types::Type;
use super::value::{RegionId, Value};
use crate::bytecode::Opcode;

/// A single IR operation.
///
/// Operations are the fundamental unit of computation. Each has an opcode,
/// zero or more operands (SSA values), zero or more results (typed SSA values),
/// named attributes, and optional regions (for control flow ops like `if`,
/// `for`, `loop`).
#[derive(Debug, Clone)]
pub struct Operation {
    pub opcode: Opcode,
    pub operands: Vec<Value>,
    pub result_types: Vec<Type>,
    pub attributes: Vec<(String, Attribute)>,
    pub regions: Vec<RegionId>,
    pub location: Location,
}
