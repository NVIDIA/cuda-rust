/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR dialect definition.

pub mod attributes;
pub mod const_fold;
pub mod ops;
pub mod rust_intrinsics;
pub mod side_effects;
pub mod types;
pub mod verification;

use pliron::context::Context;
use pliron::dialect::{Dialect, DialectName};

pub const MIR_DIALECT_NAME: &str = "mir";

pub fn register(ctx: &mut Context) {
    Dialect::register(
        ctx,
        &DialectName::try_new(MIR_DIALECT_NAME).expect("valid dialect name"),
    );
    ops::register(ctx);
    types::register(ctx);
    attributes::register(ctx);
}
