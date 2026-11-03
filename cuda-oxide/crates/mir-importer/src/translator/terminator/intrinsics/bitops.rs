/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rust compiler bit-manipulation intrinsics.
//!
//! These are `core::intrinsics::*` calls emitted by libcore for primitive
//! integer methods such as `rotate_left`, `count_ones`, and `swap_bytes`.
//! They lower to target-independent LLVM intrinsics during MIR -> LLVM lowering.

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

/// Bit intrinsic calls from libcore that lower cleanly to LLVM integer intrinsics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RustBitIntrinsic {
    /// `core::intrinsics::rotate_left`.
    RotateLeft,
    /// `core::intrinsics::rotate_right`.
    RotateRight,
    /// `core::intrinsics::ctpop`.
    Ctpop,
    /// `core::intrinsics::ctlz`; `zero_undef` is true for `ctlz_nonzero`.
    Ctlz { zero_undef: bool },
    /// `core::intrinsics::cttz`; `zero_undef` is true for `cttz_nonzero`.
    Cttz { zero_undef: bool },
    /// `core::intrinsics::bswap`.
    Bswap,
    /// `core::intrinsics::bitreverse`.
    Bitreverse,
}

impl RustBitIntrinsic {
    /// Recognize the libcore intrinsic path that survived into MIR.
    pub fn from_core_path(name: &str) -> Option<Self> {
        match name {
            "core::intrinsics::rotate_left" | "std::intrinsics::rotate_left" => {
                Some(Self::RotateLeft)
            }
            "core::intrinsics::rotate_right" | "std::intrinsics::rotate_right" => {
                Some(Self::RotateRight)
            }
            "core::intrinsics::ctpop" | "std::intrinsics::ctpop" => Some(Self::Ctpop),
            "core::intrinsics::ctlz" | "std::intrinsics::ctlz" => {
                Some(Self::Ctlz { zero_undef: false })
            }
            "core::intrinsics::ctlz_nonzero" | "std::intrinsics::ctlz_nonzero" => {
                Some(Self::Ctlz { zero_undef: true })
            }
            "core::intrinsics::cttz" | "std::intrinsics::cttz" => {
                Some(Self::Cttz { zero_undef: false })
            }
            "core::intrinsics::cttz_nonzero" | "std::intrinsics::cttz_nonzero" => {
                Some(Self::Cttz { zero_undef: true })
            }
            "core::intrinsics::bswap" | "std::intrinsics::bswap" => Some(Self::Bswap),
            "core::intrinsics::bitreverse" | "std::intrinsics::bitreverse" => {
                Some(Self::Bitreverse)
            }
            _ => None,
        }
    }

    /// Return the internal placeholder name used until MIR-to-LLVM lowering.
    pub fn placeholder_callee(self) -> &'static str {
        match self {
            Self::RotateLeft => rust_intrinsics::CALLEE_ROTATE_LEFT,
            Self::RotateRight => rust_intrinsics::CALLEE_ROTATE_RIGHT,
            Self::Ctpop => rust_intrinsics::CALLEE_CTPOP,
            Self::Ctlz { zero_undef: false } => rust_intrinsics::CALLEE_CTLZ,
            Self::Ctlz { zero_undef: true } => rust_intrinsics::CALLEE_CTLZ_NONZERO,
            Self::Cttz { zero_undef: false } => rust_intrinsics::CALLEE_CTTZ,
            Self::Cttz { zero_undef: true } => rust_intrinsics::CALLEE_CTTZ_NONZERO,
            Self::Bswap => rust_intrinsics::CALLEE_BSWAP,
            Self::Bitreverse => rust_intrinsics::CALLEE_BITREVERSE,
        }
    }
}

/// Emit a placeholder `mir.call` for a rustc bitop intrinsic.
///
/// The real LLVM intrinsic is chosen later, after MIR type conversion knows the
/// exact integer width to use for the overloaded `llvm.*.iN` name.
#[allow(clippy::too_many_arguments)]
pub fn emit_rust_bit_intrinsic(
    ctx: &mut Context,
    body: &mir::Body,
    intrinsic: RustBitIntrinsic,
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
