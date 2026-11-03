/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rust compiler saturating integer intrinsics.

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

/// Saturating arithmetic intrinsic from libcore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RustSaturatingIntrinsic {
    /// `core::intrinsics::saturating_add`.
    Add,
    /// `core::intrinsics::saturating_sub`.
    Sub,
}

impl RustSaturatingIntrinsic {
    /// Recognize the libcore intrinsic path that survived into MIR.
    pub fn from_core_path(name: &str) -> Option<Self> {
        match name {
            "core::intrinsics::saturating_add" | "std::intrinsics::saturating_add" => {
                Some(Self::Add)
            }
            "core::intrinsics::saturating_sub" | "std::intrinsics::saturating_sub" => {
                Some(Self::Sub)
            }
            _ => None,
        }
    }

    /// Return the internal placeholder name used until MIR-to-LLVM lowering.
    pub fn placeholder_callee(self) -> &'static str {
        match self {
            Self::Add => rust_intrinsics::CALLEE_SATURATING_ADD,
            Self::Sub => rust_intrinsics::CALLEE_SATURATING_SUB,
        }
    }
}

/// Emit a placeholder `mir.call` for a rustc saturating arithmetic intrinsic.
#[allow(clippy::too_many_arguments)]
pub fn emit_rust_saturating_intrinsic(
    ctx: &mut Context,
    body: &mir::Body,
    intrinsic: RustSaturatingIntrinsic,
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
