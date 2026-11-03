/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! NVVM dialect definition.
//!
//! This dialect maps to LLVM's NVPTX backend intrinsics.
//! It will be used to lower MIR intrinsics to GPU-specific operations.

pub mod ops;

use pliron::context::Context;
use pliron::dialect::{Dialect, DialectName};

pub const NVVM_DIALECT_NAME: &str = "nvvm";

pub fn register(ctx: &mut Context) {
    Dialect::register(
        ctx,
        &DialectName::try_new(NVVM_DIALECT_NAME).expect("valid dialect name"),
    );

    ops::register(ctx);
}
