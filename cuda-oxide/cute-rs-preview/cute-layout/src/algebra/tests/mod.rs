/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared fixtures for the algebra family contract tests.

use super::*;
use alloc::string::ToString;
use alloc::vec;

/// Parse a test layout literal and fail immediately if the fixture is malformed.
fn l(text: &str) -> Layout {
    text.parse().unwrap()
}

/// Small layouts whose complete coordinate maps are cheap to enumerate.
fn corpus() -> Vec<Layout> {
    [
        "4:1",
        "8:1",
        "4:2",
        "6:4",
        "(2,3):(3,1)",
        "(2,3):(1,2)",
        "(4,4):(4,1)",
        "(4,4):(1,4)",
        "(2,2,2):(4,2,1)",
        "(2,2,2):(1,2,4)",
        "(8,4):(4,1)",
        "(2,4):(1,8)",
    ]
    .iter()
    .map(|t| l(t))
    .collect()
}

mod alignment;
mod mapping;
mod thread_value;
mod tiling;
