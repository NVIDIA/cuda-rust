/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rust compiler bigint helper intrinsics.
//!
//! `core::intrinsics::carrying_mul_add` backs the stable integer methods
//! `carrying_mul_add`, `carrying_mul`, and `widening_mul`. It computes
//! `a * b + c + d` in double-width arithmetic and returns the result split
//! into `(low_half, high_half)`.
//!
//! Like the saturating intrinsics, we keep the call as a placeholder
//! `mir.call` here and expand it to LLVM dialect arithmetic during the
//! MIR-to-LLVM lowering (`mir-lower/src/convert/ops/call.rs`), where the
//! operand types (width and signedness) are directly available.

use super::super::helpers;
use crate::error::TranslationResult;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_mir::rust_intrinsics;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::location::Location;
use pliron::operation::Operation;
use rustc_public::mir;

/// Bigint helper intrinsic from libcore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RustBigIntIntrinsic {
    /// `core::intrinsics::carrying_mul_add`: computes
    /// `multiplier * multiplicand + addend + carry` in double-width
    /// arithmetic, returned as a `(low, high)` tuple.
    CarryingMulAdd,
}

impl RustBigIntIntrinsic {
    /// Recognize the libcore intrinsic path that survived into MIR.
    pub fn from_core_path(name: &str) -> Option<Self> {
        match name {
            "core::intrinsics::carrying_mul_add" | "std::intrinsics::carrying_mul_add" => {
                Some(Self::CarryingMulAdd)
            }
            _ => None,
        }
    }

    /// Return the internal placeholder name used until MIR-to-LLVM lowering.
    pub fn placeholder_callee(self) -> &'static str {
        match self {
            Self::CarryingMulAdd => rust_intrinsics::CALLEE_CARRYING_MUL_ADD,
        }
    }
}

/// Emit a placeholder `mir.call` for a rustc bigint helper intrinsic.
#[allow(clippy::too_many_arguments)]
pub fn emit_rust_bigint_intrinsic(
    ctx: &mut Context,
    body: &mir::Body,
    intrinsic: RustBigIntIntrinsic,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let return_type = types::translate_destination_type(ctx, body, destination, &loc)?;
    helpers::emit_function_call(
        ctx,
        body,
        intrinsic.placeholder_callee(),
        args,
        destination,
        return_type,
        None,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
    )
}
