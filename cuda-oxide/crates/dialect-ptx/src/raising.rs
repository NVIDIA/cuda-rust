/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Transactional raising from lossless surface PTX to native Pliron CFG.
//!
//! Planning is context-independent and performs every fallible analysis before
//! materialization mutates a caller-owned Pliron context.

mod lineage;
mod plan;

pub use lineage::{NativeCfgProjection, RaisedBlock, RaisedNode};
pub use plan::{NativeCfgPlan, RaiseError};
