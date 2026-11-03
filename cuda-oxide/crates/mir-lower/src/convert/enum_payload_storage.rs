/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Target-stable physical storage for enum payload values.
//!
//! Rust treats CUDA pointers as one logical pointer-sized value, while modern
//! NVVM uses a 32-bit physical representation for address-space-3 pointers.
//! Enum storage therefore cannot retain a semantic shared pointer directly.
//! Direct and struct/tuple-nested shared pointers are represented as CUDA
//! generic pointers in the enum and converted at construction/extraction
//! boundaries. Rust `bool` leaves are represented by canonical `i8` bytes at
//! the same boundary.
//!
//! Arrays containing shared pointers are rebuilt recursively only when the
//! payload's total array-expanded shared-pointer leaves stay within an
//! explicit code-shape bound. The bound is enforced once at the payload root,
//! so one array of 17 leaves and a struct of two 9-leaf arrays are rejected by
//! the same contract. Pointer vectors remain fail-closed because they require
//! separate ABI and address-space-cast semantics.

use crate::convert::target_stable_storage::{
    StorageRewriteOptions, coerce_target_stable_value, target_stable_storage_type,
};
use pliron::context::Context;
use pliron::irbuild::dialect_conversion::DialectConversionRewriter;
use pliron::result::Result;
use pliron::r#type::TypeHandle;
use pliron::value::Value;

/// Maximum number of array-expanded shared-pointer leaves one payload rewrite
/// may produce, totalled across the whole payload type.
///
/// Construction and extraction rebuild arrays in SSA, so every shared-pointer
/// leaf produces a pair of aggregate operations around one address-space cast.
/// Struct nesting stays unbounded because its leaf count is proportional to
/// the source text, while `[&shared; N]` expands from three tokens into `N`
/// rebuild sequences. The same constant bounds the pointer-overlap walk in
/// `build_enum_slot_map`, keeping one contract for every payload shape.
pub(crate) const MAX_ENUM_PAYLOAD_ARRAY_REWRITE_LEAVES: u64 = 16;

/// Return the physical LLVM type used to store one semantic enum payload.
///
/// Shared pointers are genericized and bool leaves are canonicalized to bytes.
/// The implementation is shared with other target-stable physical ABI carriers,
/// while this wrapper retains the enum-specific bounded-array contract.
pub(crate) fn enum_payload_storage_type(
    ctx: &mut Context,
    semantic_ty: TypeHandle,
) -> std::result::Result<TypeHandle, anyhow::Error> {
    let rewrite = target_stable_storage_type(
        ctx,
        semantic_ty,
        StorageRewriteOptions {
            canonicalize_bool: true,
        },
        "enum payload storage",
    )?;
    if rewrite.array_shared_pointer_leaves > MAX_ENUM_PAYLOAD_ARRAY_REWRITE_LEAVES {
        return Err(anyhow::anyhow!(
            "enum payload storage: arrays containing shared-memory pointers are not supported above the bounded rewrite limit; rewrite requires {} pointer conversions, supported bound is {MAX_ENUM_PAYLOAD_ARRAY_REWRITE_LEAVES}",
            rewrite.array_shared_pointer_leaves
        ));
    }
    Ok(rewrite.ty)
}

/// Convert an enum payload value between its semantic and physical types.
///
/// This is symmetric and recursively rebuilds aggregates, inserting explicit
/// address-space casts for shared pointer leaves and widening/narrowing bool
/// leaves to/from their canonical storage byte representation.
pub(crate) fn coerce_enum_payload_value(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: Value,
    target_ty: TypeHandle,
) -> Result<Value> {
    coerce_target_stable_value(ctx, rewriter, value, target_ty, "enum payload storage")
}
