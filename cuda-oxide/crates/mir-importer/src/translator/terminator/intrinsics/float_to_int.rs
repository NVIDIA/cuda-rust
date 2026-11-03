/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rust compiler `float_to_int_unchecked` intrinsic.
//!
//! `core::intrinsics::float_to_int_unchecked` converts a float to an integer
//! with a caller promise: the value is not NaN or infinite, and it fits the
//! destination type once truncated toward zero. Violating that is undefined
//! behaviour, which is what lets it lower to a plain `fptosi`/`fptoui` with
//! none of the clamping and NaN handling the saturating `as` cast needs.
//!
//! `f32::to_int_unchecked` and `f64::to_int_unchecked` are the stable entry
//! points; they reach the intrinsic through `FloatToInt::to_int_unchecked`
//! (`core/src/convert/num.rs`). Without support for the intrinsic they fail to
//! translate, so device code had no unchecked float-to-integer conversion.

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

/// The `float_to_int_unchecked` intrinsic from libcore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RustFloatToIntIntrinsic {
    /// `core::intrinsics::float_to_int_unchecked`.
    FloatToIntUnchecked,
}

impl RustFloatToIntIntrinsic {
    /// Recognize the libcore intrinsic path that survived into MIR.
    pub fn from_core_path(name: &str) -> Option<Self> {
        match name {
            "core::intrinsics::float_to_int_unchecked"
            | "std::intrinsics::float_to_int_unchecked" => Some(Self::FloatToIntUnchecked),
            _ => None,
        }
    }

    /// Return the internal placeholder name used until MIR-to-LLVM lowering.
    pub fn placeholder_callee(self) -> &'static str {
        match self {
            Self::FloatToIntUnchecked => rust_intrinsics::CALLEE_FLOAT_TO_INT_UNCHECKED,
        }
    }
}

/// Emit a placeholder `mir.call` for `core::intrinsics::float_to_int_unchecked`.
#[allow(clippy::too_many_arguments)]
pub fn emit_rust_float_to_int_intrinsic(
    ctx: &mut Context,
    body: &mir::Body,
    intrinsic: RustFloatToIntIntrinsic,
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
    fn from_core_path_recognizes_float_to_int_unchecked() {
        for path in [
            "core::intrinsics::float_to_int_unchecked",
            "std::intrinsics::float_to_int_unchecked",
        ] {
            assert_eq!(
                RustFloatToIntIntrinsic::from_core_path(path),
                Some(RustFloatToIntIntrinsic::FloatToIntUnchecked),
                "expected `{path}` to be recognized"
            );
        }
    }

    /// Only the intrinsic itself matches. The methods that wrap it have MIR
    /// bodies of their own, which the importer translates as ordinary calls.
    #[test]
    fn from_core_path_rejects_neighbouring_conversion_paths() {
        for path in [
            "core::convert::num::FloatToInt::to_int_unchecked",
            "f32::to_int_unchecked",
            "core::intrinsics::float_to_int",
            "core::intrinsics::exact_div",
        ] {
            assert_eq!(
                RustFloatToIntIntrinsic::from_core_path(path),
                None,
                "expected `{path}` not to be recognized"
            );
        }
    }

    #[test]
    fn placeholder_callee_matches_the_dialect_constant() {
        assert_eq!(
            RustFloatToIntIntrinsic::FloatToIntUnchecked.placeholder_callee(),
            rust_intrinsics::CALLEE_FLOAT_TO_INT_UNCHECKED
        );
    }
}
