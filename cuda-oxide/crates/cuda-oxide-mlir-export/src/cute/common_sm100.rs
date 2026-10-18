/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared input contract for independent SM100 attention operations.

use crate::OperationInput;

/// Reject results, attributes, regions, or successors on independent SM100 operations.
pub(in crate::cute) fn contract(input: &OperationInput) -> Result<(), String> {
    if !input.results.is_empty()
        || !input.attributes.is_empty()
        || !input.regions.is_empty()
        || !input.successors.is_empty()
    {
        return Err(
            "SM100 operation retained unexpected results, attributes or control flow".into(),
        );
    }
    Ok(())
}
