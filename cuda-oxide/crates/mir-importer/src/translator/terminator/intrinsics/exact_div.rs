/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rust compiler `exact_div` intrinsic.
//!
//! `core::intrinsics::exact_div` is division carrying a caller promise: the
//! divisor is non-zero and divides the dividend with no remainder. Violating
//! either is undefined behaviour, which is what lets it lower to a plain
//! division with no zero check and no remainder fixup.
//!
//! It matters out of proportion to its size because `slice::as_chunks` and its
//! relatives compute their chunk count with `exact_div(self.len(), N)`
//! (`core/src/slice/mod.rs:1345`). Without this intrinsic those functions fail
//! to translate, so the idiomatic safe way to read N adjacent elements is
//! unavailable in device code.

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

/// The `exact_div` intrinsic from libcore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RustExactDivIntrinsic {
    /// `core::intrinsics::exact_div`.
    ExactDiv,
}

impl RustExactDivIntrinsic {
    /// Recognize the libcore intrinsic path that survived into MIR.
    pub fn from_core_path(name: &str) -> Option<Self> {
        match name {
            "core::intrinsics::exact_div" | "std::intrinsics::exact_div" => Some(Self::ExactDiv),
            _ => None,
        }
    }

    /// Return the internal placeholder name used until MIR-to-LLVM lowering.
    pub fn placeholder_callee(self) -> &'static str {
        match self {
            Self::ExactDiv => rust_intrinsics::CALLEE_EXACT_DIV,
        }
    }
}

/// Emit a placeholder `mir.call` for `core::intrinsics::exact_div`.
#[allow(clippy::too_many_arguments)]
pub fn emit_rust_exact_div_intrinsic(
    ctx: &mut Context,
    body: &mir::Body,
    intrinsic: RustExactDivIntrinsic,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_core_path_recognizes_exact_div() {
        for path in ["core::intrinsics::exact_div", "std::intrinsics::exact_div"] {
            assert_eq!(
                RustExactDivIntrinsic::from_core_path(path),
                Some(RustExactDivIntrinsic::ExactDiv),
                "expected `{path}` to be recognized"
            );
        }
    }

    /// `unchecked_div` carries a weaker promise (non-zero divisor only, the
    /// remainder is discarded), so it must not be folded into this intrinsic.
    #[test]
    fn from_core_path_rejects_neighbouring_division_intrinsics() {
        for path in [
            "core::intrinsics::unchecked_div",
            "core::intrinsics::exact_divide",
            "core::intrinsics::div_exact",
            "core::intrinsics::saturating_add",
        ] {
            assert_eq!(
                RustExactDivIntrinsic::from_core_path(path),
                None,
                "expected `{path}` not to be recognized"
            );
        }
    }

    #[test]
    fn placeholder_callee_matches_the_dialect_constant() {
        assert_eq!(
            RustExactDivIntrinsic::ExactDiv.placeholder_callee(),
            rust_intrinsics::CALLEE_EXACT_DIV
        );
    }
}
