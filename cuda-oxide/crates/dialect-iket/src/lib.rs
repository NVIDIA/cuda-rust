/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Semantic operations for In-Kernel Event Tracing (IKET).
//!
//! This dialect describes what a kernel author wants to observe. It does not
//! select a physical instrumentation method and does not model runtime buffer
//! management. A later lowering pass may erase the operations or lower them to
//! an IKET runtime-compatible encoding.

pub mod attributes;
pub mod ops;
pub mod types;

use pliron::context::Context;
use pliron::dialect::{Dialect, DialectName};

pub const IKET_DIALECT_NAME: &str = "iket";

pub fn register(ctx: &mut Context) {
    Dialect::register(
        ctx,
        &DialectName::try_new(IKET_DIALECT_NAME).expect("valid dialect name"),
    );
    attributes::register(ctx);
    types::register(ctx);
    ops::register(ctx);
}
