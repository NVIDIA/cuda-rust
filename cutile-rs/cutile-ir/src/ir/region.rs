/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regions — containers of basic blocks attached to operations.

use super::value::BlockId;

/// A region is an ordered list of basic blocks, attached to an operation.
///
/// Control-flow operations (`if`, `for`, `loop`) contain one or more regions.
/// Functions contain a single region with one entry block.
#[derive(Debug, Clone)]
pub struct Region {
    pub blocks: Vec<BlockId>,
}
