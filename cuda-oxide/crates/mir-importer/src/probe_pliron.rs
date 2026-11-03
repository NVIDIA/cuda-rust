/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Debug utility for probing pliron operations.
//!
//! Contains helpers for debugging Pliron IR operations during development.
//! Not used in production compilation.

#![allow(dead_code)]

use pliron::location::Location;
use pliron::operation::Operation;

/// Clears location info from an operation (for debugging/testing).
pub fn probe(op: &mut Operation) {
    op.loc = Location::Unknown;
}
