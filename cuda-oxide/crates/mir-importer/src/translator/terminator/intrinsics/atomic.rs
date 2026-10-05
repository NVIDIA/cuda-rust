/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Atomic operation intrinsic handlers.
//!
//! Translates atomic method calls into NVVM atomic dialect operations.
//! Supports two front-ends:
//!
//! 1. **`cuda_device::atomic::*`** — custom GPU atomic types with explicit scope
//! 2. **`core::sync::atomic::*`** — standard library atomics (via `std::intrinsics::atomic_*`)
//!
//! Both front-ends emit the same NVVM ops and share the entire lowering pipeline
//! (mir-lower fence splitting → LLVM dialect → export → llc → PTX).
//!
//! # cuda_device Path — Type Resolution
//!
//! The atomic type name encodes scope and element type:
//!
//! ```text
//! BlockAtomicI64::fetch_add
//! ─────┬────────  ────┬────
//!   scope prefix    method
//!       └── AtomicI64 = 64-bit signed integer
//! ```
//!
//! | Prefix            | Scope   | PTX    |
//! |-------------------|---------|--------|
//! | `DeviceAtomic*`   | Device  | `.gpu` |
//! | `BlockAtomic*`    | Block   | `.cta` |
//! | `SystemAtomic*`   | System  | `.sys` |
//!
//! # cuda_device Path — Method → RMW Kind Mapping
//!
//! | Method       | Integer RMW Kind   | Float RMW Kind |
//! |--------------|--------------------|----------------|
//! | `fetch_add`  | `Add`              | `FAdd`         |
//! | `fetch_sub`  | `Sub`              | `FAdd(-x)`     |
//! | `fetch_and`  | `And`              | —              |
//! | `fetch_or`   | `Or`               | —              |
//! | `fetch_xor`  | `Xor`              | —              |
//! | `fetch_min`  | `Min` / `UMin` `[*]` | —            |
//! | `fetch_max`  | `Max` / `UMax` `[*]` | —            |
//! | `swap`       | `Xchg`             | `Xchg`         |
//!
//! `[*]` `fetch_min`/`fetch_max` use signed (`Min`/`Max`) for `I32`/`I64`,
//!     unsigned (`UMin`/`UMax`) for `U32`/`U64`.
//!
//! # core::sync::atomic Path
//!
//! Standard library atomics compile down to `std::intrinsics::atomic_*` (or
//! `core::intrinsics::atomic_*` in `#![no_std]`).  These are generic intrinsics
//! whose ordering is a **const generic**, not a runtime argument:
//!
//! ```text
//! std::intrinsics::atomic_xadd::<u32, u32, AtomicOrdering::Relaxed>(ptr, val)
//! ─────────────────────┬─────    ──┬──      ────────┬───────────── ──┬──  ─┬─
//!                 intrinsic name   type          ordering           ptr   val
//! ```
//!
//! All `core::sync::atomic` operations are lowered with **system scope** (`.sys`)
//! for safe host-device coherence, matching CUDA C++ `cuda::atomic<T>` defaults.

use super::super::helpers;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::values::ValueMap;
use crate::translator::{facts, rvalue, types};

use dialect_nvvm::ops::InlinePtxOp;
use dialect_nvvm::ops::atomic::{
    AtomicOrdering, AtomicRmwKind, AtomicScope, NvvmAtomicCmpxchgOp, NvvmAtomicFenceOp,
    NvvmAtomicLoadOp, NvvmAtomicRmwOp, NvvmAtomicStoreOp,
};

use dialect_mir::ops::{MirConstructTupleOp, MirEqOp, MirNegOp};
use dialect_mir::types::MirFP16Type;
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::Typed;
use pliron::{input_err, input_error_noloc};
use rustc_public::mir;
use rustc_public::ty::{GenericArgKind, RigidTy, TyConst, TyConstKind, TyKind};
// =============================================================================
// Type info — extracted from a named atomic type or a core MIR element type
// =============================================================================

/// Describes the element type and scope used to lower an atomic operation.
///
/// Named `cuda_device::atomic` types populate this from their path, while core
/// atomic intrinsics populate it from their MIR element type.
///
/// Example: `BlockAtomicI64` →
/// `{ bit_width: 64, is_float: false, is_signed: true, is_pointer: false, scope: Block }`
pub struct AtomicTypeInfo {
    pub bit_width: u32,
    pub is_float: bool,
    pub is_signed: bool,
    pub is_pointer: bool,
    pub scope: AtomicScope,
}

impl AtomicTypeInfo {
    /// Get the pliron result type for this atomic's element.
    fn element_type(&self, ctx: &mut Context) -> pliron::r#type::TypeHandle {
        if self.is_pointer {
            unreachable!("pointer atomic element types must be preserved from MIR");
        }

        if self.is_float {
            match self.bit_width {
                // Rust `f16` is represented by dialect-mir's own `mir.fp16` (apfloat::Half);
                // f32/f64 reuse the pliron builtin float types.
                16 => MirFP16Type::get(ctx).into(),
                32 => FP32Type::get(ctx).into(),
                64 => FP64Type::get(ctx).into(),
                _ => unreachable!("unsupported float atomic width: {}", self.bit_width),
            }
        } else {
            let signedness = if self.is_signed {
                Signedness::Signed
            } else {
                Signedness::Unsigned
            };
            IntegerType::get(ctx, self.bit_width, signedness).to_handle()
        }
    }
}

/// Parse an atomic type name (e.g., `"DeviceAtomicU32"`, `"BlockAtomicI64"`) into type info.
///
/// Device scope uses the `DeviceAtomic*` prefix to avoid name collision with
/// `core::sync::atomic::Atomic*`. Returns `None` if the name doesn't match.
fn parse_atomic_type_name(type_name: &str) -> Option<AtomicTypeInfo> {
    // Extract scope prefix and base type suffix. Try longer prefixes first.
    let (scope, base) = if let Some(rest) = type_name.strip_prefix("BlockAtomic") {
        (AtomicScope::Block, rest)
    } else if let Some(rest) = type_name.strip_prefix("SystemAtomic") {
        (AtomicScope::System, rest)
    } else {
        let rest = type_name.strip_prefix("DeviceAtomic")?;
        (AtomicScope::Device, rest)
    };

    let (bit_width, is_float, is_signed) = match base {
        "U32" => (32, false, false),
        "I32" => (32, false, true),
        "U64" => (64, false, false),
        "I64" => (64, false, true),
        "F16" => (16, true, false),
        "F32" => (32, true, false),
        "F64" => (64, true, false),
        _ => return None,
    };

    Some(AtomicTypeInfo {
        bit_width,
        is_float,
        is_signed,
        is_pointer: false,
        scope,
    })
}

/// Check whether a call path refers to a known atomic type.
///
/// Used as a guard in the `try_dispatch_intrinsic` match arm.
pub fn is_atomic_path(path: &str) -> bool {
    parse_atomic_path(path).is_some()
}

/// Parse a full call path into (type_info, method_name).
///
/// Example: `"cuda_device::atomic::AtomicU32::fetch_add"` → `(AtomicTypeInfo{..}, "fetch_add")`
fn parse_atomic_path(path: &str) -> Option<(AtomicTypeInfo, &str)> {
    let mut parts = path.rsplit("::");
    let method = parts.next()?;
    let type_name = parts.next()?;
    let info = parse_atomic_type_name(type_name)?;
    Some((info, method))
}

// =============================================================================
// RMW kind resolution
// =============================================================================

/// Map a method name to the appropriate `AtomicRmwKind`.
///
/// For `fetch_min`/`fetch_max`, signedness matters:
/// - Unsigned types (U32, U64) → `UMin`/`UMax`
/// - Signed types (I32, I64) → `Min`/`Max`
///
/// For float `fetch_add`, use `FAdd`; float `fetch_sub` is handled as
/// `FAdd(-x)` at emission time so LLVM can use native PTX add atomics.
fn method_to_rmw_kind(method: &str, info: &AtomicTypeInfo) -> Option<AtomicRmwKind> {
    match method {
        "fetch_add" => {
            if info.is_float {
                Some(AtomicRmwKind::FAdd)
            } else {
                Some(AtomicRmwKind::Add)
            }
        }
        "fetch_sub" => {
            if info.is_float {
                Some(AtomicRmwKind::FAdd)
            } else {
                Some(AtomicRmwKind::Sub)
            }
        }
        "fetch_and" => Some(AtomicRmwKind::And),
        "fetch_or" => Some(AtomicRmwKind::Or),
        "fetch_xor" => Some(AtomicRmwKind::Xor),
        "fetch_min" => {
            if info.is_signed {
                Some(AtomicRmwKind::Min)
            } else {
                Some(AtomicRmwKind::UMin)
            }
        }
        "fetch_max" => {
            if info.is_signed {
                Some(AtomicRmwKind::Max)
            } else {
                Some(AtomicRmwKind::UMax)
            }
        }
        "swap" => Some(AtomicRmwKind::Xchg),
        _ => None,
    }
}

// =============================================================================
// Ordering extraction from MIR constants
// =============================================================================

/// Extract an `AtomicOrdering` from a MIR operand that represents
/// a `cuda_device::atomic::AtomicOrdering` enum value.
///
/// The ordering decides which memory fences the emitted PTX carries, so
/// there is no safe guess: matching is by variant NAME (layout drift in
/// cuda-device cannot silently remap orderings) and anything that is not a
/// readable constant variant is a hard error.
fn extract_ordering(operand: &mir::Operand, loc: &Location) -> TranslationResult<AtomicOrdering> {
    let mir::Operand::Constant(constant) = operand else {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "atomic ordering must be a compile-time constant (a literal AtomicOrdering \
                 variant)"
                    .to_string()
            )
        );
    };
    let (_idx, variant_name) = facts::extract_enum_variant(&constant.const_, loc)?;
    match variant_name.as_str() {
        "Relaxed" => Ok(AtomicOrdering::Relaxed),
        "Acquire" => Ok(AtomicOrdering::Acquire),
        "Release" => Ok(AtomicOrdering::Release),
        "AcqRel" => Ok(AtomicOrdering::AcqRel),
        "SeqCst" => Ok(AtomicOrdering::SeqCst),
        other => input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!("unknown AtomicOrdering variant `{other}`"))
        ),
    }
}

// =============================================================================
// Top-level dispatch — called from terminator/mod.rs
// =============================================================================

/// Dispatch an atomic intrinsic call to the appropriate emit function.
///
/// Returns `Ok(Some(op))` if the method was handled, `Ok(None)` if the
/// method is not an intrinsic (e.g., `new()`), or `Err` on failure.
#[allow(clippy::too_many_arguments)]
pub fn dispatch(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    path: &str,
) -> TranslationResult<Option<Ptr<Operation>>> {
    let (type_info, method) = match parse_atomic_path(path) {
        Some(parsed) => parsed,
        None => return Ok(None),
    };

    match method {
        "load" => Ok(Some(emit_atomic_load(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            &type_info,
        )?)),

        "store" => Ok(Some(emit_atomic_store(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            &type_info,
        )?)),

        "fetch_add" | "fetch_sub" | "fetch_and" | "fetch_or" | "fetch_xor" | "fetch_min"
        | "fetch_max" | "swap" => {
            let rmw_kind = method_to_rmw_kind(method, &type_info).unwrap();
            let negate_value = type_info.is_float && method == "fetch_sub";
            Ok(Some(emit_atomic_rmw(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
                &type_info,
                rmw_kind,
                negate_value,
            )?))
        }

        "compare_exchange_raw" => Ok(Some(emit_atomic_compare_exchange(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            &type_info,
        )?)),

        // new() is a const fn compiled normally; compare_exchange() is an
        // #[inline(always)] wrapper around compare_exchange_raw — both are
        // handled by regular MIR translation, not intrinsic dispatch.
        _ => Ok(None),
    }
}

// =============================================================================
// Emit functions
// =============================================================================

/// Emit an atomic load.
///
/// MIR args: `[self_ptr, ordering]`
#[allow(clippy::too_many_arguments)]
fn emit_atomic_load(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    type_info: &AtomicTypeInfo,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "atomic load expects 2 arguments (self, ordering), got {}",
                args.len()
            ))
        );
    }

    let ordering = extract_ordering(&args[1], &loc)?;
    let result_ty = type_info.element_type(ctx);

    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let (prepared_destination, last_op) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let nvvm_op =
        NvvmAtomicLoadOp::build(ctx, ptr_val, result_ty, ordering, type_info.scope.clone());
    let op_ptr = nvvm_op.get_operation();
    op_ptr.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op_ptr.insert_after(ctx, prev);
    } else {
        op_ptr.insert_at_front(block_ptr, ctx);
    }

    let result_value = op_ptr.deref(ctx).get_result(0);
    helpers::emit_prepared_result_and_goto(
        ctx,
        prepared_destination,
        result_value,
        target,
        block_ptr,
        op_ptr,
        value_map,
        block_map,
        loc,
        "atomic load call without target block",
    )
}

/// Emit an atomic store.
///
/// MIR args: `[self_ptr, val, ordering]`
#[allow(clippy::too_many_arguments)]
fn emit_atomic_store(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    type_info: &AtomicTypeInfo,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "atomic store expects 3 arguments (self, val, ordering), got {}",
                args.len()
            ))
        );
    }

    let ordering = extract_ordering(&args[2], &loc)?;

    // Get the value to store (arg 1)
    let (val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Get the pointer (arg 0) -- self
    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let (prepared_destination, last_op) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let nvvm_op = NvvmAtomicStoreOp::build(ctx, val, ptr_val, ordering, type_info.scope.clone());
    let op_ptr = nvvm_op.get_operation();
    op_ptr.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op_ptr.insert_after(ctx, prev);
    } else {
        op_ptr.insert_at_front(block_ptr, ctx);
    }

    // Store returns unit -- set destination to a unit value
    let unit_ty = dialect_mir::types::MirTupleType::get(ctx, vec![]);
    let unit_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructTupleOp::get_concrete_op_info(),
        vec![unit_ty.into()],
        vec![],
        vec![],
        0,
    );
    unit_op.deref_mut(ctx).set_loc(loc.clone());
    unit_op.insert_after(ctx, op_ptr);
    let unit_val = unit_op.deref(ctx).get_result(0);
    helpers::emit_prepared_result_and_goto(
        ctx,
        prepared_destination,
        unit_val,
        target,
        block_ptr,
        unit_op,
        value_map,
        block_map,
        loc,
        "atomic store call without target block",
    )
}

/// Emit an atomic read-modify-write operation.
///
/// Handles all RMW methods: `fetch_add`, `fetch_sub`, `fetch_and`, `fetch_or`,
/// `fetch_xor`, `fetch_min`, `fetch_max`, `swap`.
///
/// MIR args: `[self_ptr, val, ordering]`
#[allow(clippy::too_many_arguments)]
fn emit_atomic_rmw(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    type_info: &AtomicTypeInfo,
    rmw_kind: AtomicRmwKind,
    negate_value: bool,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "atomic RMW expects 3 arguments (self, val, ordering), got {}",
                args.len()
            ))
        );
    }

    let ordering = extract_ordering(&args[2], &loc)?;
    let result_ty = type_info.element_type(ctx);

    // Get the pointer (arg 0)
    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Get the value operand (arg 1)
    let (val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let (val, last_op) = if negate_value {
        let neg_op = Operation::new(
            ctx,
            MirNegOp::get_concrete_op_info(),
            vec![val.get_type(ctx)],
            vec![val],
            vec![],
            0,
        );
        neg_op.deref_mut(ctx).set_loc(loc.clone());
        if let Some(prev) = last_op {
            neg_op.insert_after(ctx, prev);
        } else {
            neg_op.insert_at_front(block_ptr, ctx);
        }
        (neg_op.deref(ctx).get_result(0), Some(neg_op))
    } else {
        (val, last_op)
    };

    let (prepared_destination, last_op) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let nvvm_op = NvvmAtomicRmwOp::build(
        ctx,
        ptr_val,
        val,
        result_ty,
        rmw_kind,
        ordering,
        type_info.scope.clone(),
    );
    let op_ptr = nvvm_op.get_operation();
    op_ptr.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op_ptr.insert_after(ctx, prev);
    } else {
        op_ptr.insert_at_front(block_ptr, ctx);
    }

    let result_value = op_ptr.deref(ctx).get_result(0);
    helpers::emit_prepared_result_and_goto(
        ctx,
        prepared_destination,
        result_value,
        target,
        block_ptr,
        op_ptr,
        value_map,
        block_map,
        loc,
        "atomic RMW call without target block",
    )
}

/// Emit an atomic compare-and-exchange.
///
/// Only valid for integer types. Float types do not support CAS in PTX.
///
/// MIR args: `[self_ptr, current, new, success_ordering, failure_ordering]`
#[allow(clippy::too_many_arguments)]
fn emit_atomic_compare_exchange(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    type_info: &AtomicTypeInfo,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 5 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "atomic compare_exchange expects 5 arguments (self, current, new, success, failure), got {}",
                args.len()
            ))
        );
    }

    let success_ordering = extract_ordering(&args[3], &loc)?;
    let failure_ordering = extract_ordering(&args[4], &loc)?;
    let result_ty = type_info.element_type(ctx);

    // Get the pointer (arg 0)
    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Get the expected (current) value (arg 1)
    let (cmp_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    // Get the new value (arg 2)
    let (new_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let (prepared_destination, last_op) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let nvvm_op = NvvmAtomicCmpxchgOp::build(
        ctx,
        ptr_val,
        cmp_val,
        new_val,
        result_ty,
        success_ordering,
        failure_ordering,
        type_info.scope.clone(),
    );
    let op_ptr = nvvm_op.get_operation();
    op_ptr.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op_ptr.insert_after(ctx, prev);
    } else {
        op_ptr.insert_at_front(block_ptr, ctx);
    }

    let result_value = op_ptr.deref(ctx).get_result(0);
    helpers::emit_prepared_result_and_goto(
        ctx,
        prepared_destination,
        result_value,
        target,
        block_ptr,
        op_ptr,
        value_map,
        block_map,
        loc,
        "atomic compare_exchange call without target block",
    )
}

// =============================================================================
// core::sync::atomic support — std::intrinsics::atomic_* dispatch
// =============================================================================

/// Check whether a call path is a core/std atomic intrinsic.
///
/// Matches `std::intrinsics::atomic_*` and `core::intrinsics::atomic_*`.
pub fn is_core_atomic_intrinsic(path: &str) -> bool {
    path.starts_with("std::intrinsics::atomic_") || path.starts_with("core::intrinsics::atomic_")
}

/// Extract the operation name from a core atomic intrinsic path.
///
/// Example: `"std::intrinsics::atomic_xadd"` → `Some("xadd")`
fn parse_core_intrinsic_op(path: &str) -> Option<&str> {
    path.strip_prefix("std::intrinsics::atomic_")
        .or_else(|| path.strip_prefix("core::intrinsics::atomic_"))
}

/// Map `std::intrinsics::AtomicOrdering` discriminant to our `AtomicOrdering`.
///
/// **Important**: The discriminant layout differs from `cuda_device::AtomicOrdering`:
///
/// | Discriminant | `std::intrinsics::AtomicOrdering` | `cuda_device::AtomicOrdering` |
/// |--------------|-----------------------------------|-----------------------|
/// |            0 | Relaxed                             | Relaxed               |
/// |            1 | **Release**                         | **Acquire**           |
/// |            2 | **Acquire**                         | **Release**           |
/// |            3 | AcqRel                              | AcqRel                |
/// |            4 | SeqCst                              | SeqCst                |
fn intrinsic_ordering_from_discriminant(discr: u64) -> Option<AtomicOrdering> {
    Some(match discr {
        0 => AtomicOrdering::Relaxed,
        1 => AtomicOrdering::Release, // std has Release=1, unlike cuda_device Acquire=1
        2 => AtomicOrdering::Acquire, // std has Acquire=2, unlike cuda_device Release=2
        3 => AtomicOrdering::AcqRel,
        4 => AtomicOrdering::SeqCst,
        _ => return None,
    })
}

/// Build `AtomicTypeInfo` from a rustc type, with system scope.
///
/// Core atomics always use system scope for safe host-device coherence.
fn type_info_from_mir_ty(ty: &rustc_public::ty::Ty) -> Option<AtomicTypeInfo> {
    let (bit_width, is_float, is_signed, is_pointer) = match ty.kind() {
        TyKind::RigidTy(RigidTy::Uint(uint_ty)) => {
            use rustc_public::ty::UintTy;
            let width = match uint_ty {
                UintTy::U8 => 8,
                UintTy::U16 => 16,
                UintTy::U32 => 32,
                UintTy::U64 => 64,
                UintTy::U128 => 128,
                // usize is target-dependent (32-bit on nvptx, 64-bit on nvptx64).
                // We only target nvptx64 today; making this configurable via
                // `PipelineConfig::target_pointer_width` is a future change.
                UintTy::Usize => 64,
            };
            (width, false, false, false)
        }
        TyKind::RigidTy(RigidTy::Int(int_ty)) => {
            use rustc_public::ty::IntTy;
            let width = match int_ty {
                IntTy::I8 => 8,
                IntTy::I16 => 16,
                IntTy::I32 => 32,
                IntTy::I64 => 64,
                IntTy::I128 => 128,
                // isize is target-dependent (32-bit on nvptx, 64-bit on nvptx64).
                // We only target nvptx64 today; making this configurable via
                // `PipelineConfig::target_pointer_width` is a future change.
                IntTy::Isize => 64,
            };
            (width, false, true, false)
        }
        TyKind::RigidTy(RigidTy::Float(float_ty)) => {
            use rustc_public::ty::FloatTy;
            let width = match float_ty {
                FloatTy::F16 => 16,
                FloatTy::F32 => 32,
                FloatTy::F64 => 64,
                FloatTy::F128 => 128,
            };
            (width, true, false, false)
        }
        TyKind::RigidTy(RigidTy::RawPtr(_, _)) => {
            // Core AtomicPtr values are pointer-width on nvptx64; the NVVM verifier
            // restricts lowered pointer values to the 64-bit generic address space.
            (64, false, false, true)
        }
        _ => return None,
    };

    Some(AtomicTypeInfo {
        bit_width,
        is_float,
        is_signed,
        is_pointer,
        scope: AtomicScope::System, // core atomics always use system scope
    })
}

/// Map a core intrinsic operation name to an `AtomicRmwKind`.
///
/// | Intrinsic op | RMW Kind              |
/// |--------------|-----------------------|
/// | `xadd`       | `Add` / `FAdd`        |
/// | `xsub`       | `Sub`                 |
/// | `and`        | `And`                 |
/// | `or`         | `Or`                  |
/// | `xor`        | `Xor`                 |
/// | `min`        | `Min` (signed)        |
/// | `umin`       | `UMin` (unsigned)     |
/// | `max`        | `Max` (signed)        |
/// | `umax`       | `UMax` (unsigned)     |
/// | `xchg`       | `Xchg`                |
fn intrinsic_op_to_rmw_kind(op: &str, info: &AtomicTypeInfo) -> Option<AtomicRmwKind> {
    if info.is_pointer {
        return match op {
            "xchg" => Some(AtomicRmwKind::Xchg),
            _ => None,
        };
    }

    match op {
        "xadd" => {
            if info.is_float {
                Some(AtomicRmwKind::FAdd)
            } else {
                Some(AtomicRmwKind::Add)
            }
        }
        "xsub" => Some(AtomicRmwKind::Sub),
        "and" => Some(AtomicRmwKind::And),
        "or" => Some(AtomicRmwKind::Or),
        "xor" => Some(AtomicRmwKind::Xor),
        "min" => Some(AtomicRmwKind::Min),
        "umin" => Some(AtomicRmwKind::UMin),
        "max" => Some(AtomicRmwKind::Max),
        "umax" => Some(AtomicRmwKind::UMax),
        "xchg" => Some(AtomicRmwKind::Xchg),
        _ => None,
    }
}

fn extract_core_ordering(c: &TyConst) -> Option<AtomicOrdering> {
    let discr = match c.kind() {
        TyConstKind::Value(_, alloc) => u64::try_from(alloc.read_uint().ok()?).ok()?,
        _ => c.eval_target_usize().ok()?,
    };
    intrinsic_ordering_from_discriminant(discr)
}

/// Extract ordering consts without assuming how many type generics precede
/// them, plus the `VOLATILE` flag when present.
///
/// nightly-2026-08-28 added a `const VOLATILE: bool` tail generic to
/// `atomic_load`/`atomic_store`. Ordering consts are the
/// `AtomicOrdering`-typed ones; a `bool` const is the volatile flag.
/// Returns `None` when any ordering const cannot be evaluated.
fn extract_orderings_from_generics(
    substs: &rustc_public::ty::GenericArgs,
) -> Option<(Vec<AtomicOrdering>, bool)> {
    let mut orderings = Vec::new();
    let mut volatile = false;
    for arg in substs.0.iter() {
        let GenericArgKind::Const(c) = arg else {
            continue;
        };
        let is_bool = matches!(
            c.kind(),
            TyConstKind::Value(ty, _) if matches!(ty.kind(), TyKind::RigidTy(RigidTy::Bool))
        );
        if is_bool {
            let TyConstKind::Value(_, alloc) = c.kind() else {
                unreachable!("bool const kind re-matched");
            };
            volatile = alloc.read_uint().ok()? != 0;
        } else {
            orderings.push(extract_core_ordering(c)?);
        }
    }
    Some((orderings, volatile))
}

/// Extract the element type from the first generic type arg.
fn extract_type_info_from_generics(
    substs: &rustc_public::ty::GenericArgs,
) -> Option<AtomicTypeInfo> {
    substs.0.iter().find_map(|arg| match arg {
        GenericArgKind::Type(ty) => type_info_from_mir_ty(ty),
        _ => None,
    })
}

/// The route a core atomic intrinsic takes through [`dispatch_core_intrinsic`].
///
/// The two fences carry no element-type generic, only an ordering const, so
/// they must be picked out by name before the type extraction shared by the
/// load/store/RMW/CAS paths. A fence that misses this routing is reported as
/// a missing element type, which is not the problem (issue #781 was exactly
/// that for `compiler_fence`).
#[derive(Debug, PartialEq, Eq)]
enum CoreIntrinsicRoute {
    /// `atomic_singlethreadfence` (`core::sync::atomic::compiler_fence`):
    /// constrains only the optimizer and must emit no hardware barrier.
    CompilerFence,
    /// `atomic_fence` (`core::sync::atomic::fence`): a hardware barrier at
    /// system scope.
    HardwareFence,
    /// Everything else: element-typed load/store/RMW/CAS dispatch.
    Typed,
}

/// Classify a core atomic intrinsic op name (see [`parse_core_intrinsic_op`]).
fn route_core_intrinsic(op_name: &str) -> CoreIntrinsicRoute {
    match op_name {
        "singlethreadfence" => CoreIntrinsicRoute::CompilerFence,
        "fence" => CoreIntrinsicRoute::HardwareFence,
        _ => CoreIntrinsicRoute::Typed,
    }
}

/// Dispatch a `std::intrinsics::atomic_*` / `core::intrinsics::atomic_*` call.
///
/// Extracts the generic args (type, ordering) from the `func` operand and
/// routes to the appropriate emit function.  All operations use **system scope**.
///
/// Returns `Ok(Some(op))` if handled, `Err` on failure.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_core_intrinsic(
    ctx: &mut Context,
    body: &mir::Body,
    func: &mir::Operand,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    path: &str,
) -> TranslationResult<Ptr<Operation>> {
    let op_name = parse_core_intrinsic_op(path).unwrap_or("");

    // The ordering-only fences must be routed by name before the common type
    // extraction used by load/store/RMW/CAS (see [`CoreIntrinsicRoute`]).
    match route_core_intrinsic(op_name) {
        CoreIntrinsicRoute::CompilerFence => {
            let orderings = extract_core_intrinsic_orderings(func, &loc, 1)?;
            return emit_core_compiler_fence(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
                orderings[0].clone(),
            );
        }
        CoreIntrinsicRoute::HardwareFence => {
            let orderings = extract_core_intrinsic_orderings(func, &loc, 1)?;
            return emit_core_atomic_fence(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
                orderings[0].clone(),
            );
        }
        CoreIntrinsicRoute::Typed => {}
    }

    // Extract generic args from the func operand.
    let is_cmpxchg = op_name == "cxchg" || op_name == "cxchgweak";
    let expected_orderings = if is_cmpxchg { 2 } else { 1 };
    let (type_info, orderings) =
        extract_core_intrinsic_generics(func, &loc, expected_orderings, op_name)?;
    let ordering = orderings[0].clone();

    // Route by operation name
    if op_name == "load" {
        emit_core_atomic_load(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            &type_info,
            ordering,
        )
    } else if op_name == "store" {
        emit_core_atomic_store(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            &type_info,
            ordering,
        )
    } else if is_cmpxchg {
        emit_core_atomic_cmpxchg(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            &type_info,
            ordering,
            orderings[1].clone(),
        )
    } else if let Some(rmw_kind) = intrinsic_op_to_rmw_kind(op_name, &type_info) {
        emit_core_atomic_rmw(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            &type_info,
            ordering,
            rmw_kind,
        )
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!("unsupported core atomic intrinsic: {path}"))
        )
    }
}

/// Extract and validate ordering const generics from generic arguments.
fn extract_core_intrinsic_orderings_from_generics(
    substs: &rustc_public::ty::GenericArgs,
    loc: &Location,
    expected_orderings: usize,
) -> TranslationResult<Vec<AtomicOrdering>> {
    let Some((orderings, volatile)) = extract_orderings_from_generics(substs) else {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported("could not evaluate core atomic ordering generics")
        );
    };

    if volatile {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "volatile core atomics (`VOLATILE = true`) are not supported in device code"
            )
        );
    }

    if orderings.len() != expected_orderings {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "core atomic intrinsic requires {expected_orderings} ordering generic(s), found {}",
                orderings.len()
            ))
        );
    }

    Ok(orderings)
}

/// Extract only ordering const generics from a core atomic intrinsic.
///
/// This path is used by `atomic_fence`, whose generic arguments do not contain
/// an element type.
fn extract_core_intrinsic_orderings(
    func: &mir::Operand,
    loc: &Location,
    expected_orderings: usize,
) -> TranslationResult<Vec<AtomicOrdering>> {
    if let mir::Operand::Constant(const_op) = func
        && let TyKind::RigidTy(RigidTy::FnDef(_, substs)) = const_op.const_.ty().kind()
    {
        return extract_core_intrinsic_orderings_from_generics(&substs, loc, expected_orderings);
    }

    input_err!(
        loc.clone(),
        TranslationErr::unsupported(
            "core atomic intrinsic: could not extract generics from func operand"
        )
    )
}

/// Extract type info and ordering from a core atomic intrinsic's generic args.
fn extract_core_intrinsic_generics(
    func: &mir::Operand,
    loc: &Location,
    expected_orderings: usize,
    op_name: &str,
) -> TranslationResult<(AtomicTypeInfo, Vec<AtomicOrdering>)> {
    if let mir::Operand::Constant(const_op) = func
        && let TyKind::RigidTy(RigidTy::FnDef(_, substs)) = const_op.const_.ty().kind()
    {
        let Some(type_info) = extract_type_info_from_generics(&substs) else {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "unsupported core atomic operation `{op_name}`: no element type in its \
                         generics, and it is not one of the ordering-only fences"
                ))
            );
        };

        if !core_atomic_width_is_supported(type_info.bit_width) {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "{}-bit core atomics are not supported; use 32-bit or 64-bit",
                    type_info.bit_width
                ))
            );
        }

        let orderings =
            extract_core_intrinsic_orderings_from_generics(&substs, loc, expected_orderings)?;

        return Ok((type_info, orderings));
    }

    input_err!(
        loc.clone(),
        TranslationErr::unsupported(
            "core atomic intrinsic: could not extract generics from func operand"
        )
    )
}
fn core_atomic_width_is_supported(bit_width: u32) -> bool {
    matches!(bit_width, 32 | 64)
}

// =============================================================================
// Core intrinsic emit functions
//
// These handle the MIR arg layout for std::intrinsics::atomic_* which differs
// from cuda_device (no ordering arg, different arg count).  They build the same
// NVVM ops as the cuda_device emit functions.
// =============================================================================

/// Build the barrier op for a core compiler fence.
///
/// `core::sync::atomic::compiler_fence` constrains only the compiler: it forbids
/// reordering memory operations across itself and emits no hardware instruction,
/// so unlike `fence` it must not become a PTX `fence` or `membar`. The encoding
/// is an empty side-effecting inline PTX block carrying a `~{memory}` clobber --
/// the mechanism the first-class fence routes already use, minus the instruction
/// text. That survives the libNVVM-safe route, which rejects the LLVM `fence`
/// instruction outright.
///
/// `Relaxed` is refused. Safe Rust cannot build it: `compiler_fence(Relaxed)`
/// panics in core. Only a direct nightly intrinsic call reaches here, and a
/// relaxed compiler fence orders nothing, so refuse it rather than emit a
/// barrier that silently means something else.
fn build_compiler_fence_barrier(
    ctx: &mut Context,
    loc: &Location,
    ordering: &AtomicOrdering,
) -> TranslationResult<Ptr<Operation>> {
    if matches!(ordering, AtomicOrdering::Relaxed) {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "core compiler fence cannot use Relaxed ordering".to_owned()
            )
        );
    }

    // No results, and therefore no `=` output constraints, which is what the
    // inline-PTX verifier checks the two against.
    Ok(InlinePtxOp::build(
        ctx,
        vec![],
        vec![],
        "",
        "~{memory}",
        true,
        false,
    ))
}

/// Emit a core compiler fence (see [`build_compiler_fence_barrier`] for the
/// encoding and the `Relaxed` refusal).
///
/// MIR args: none; ordering is carried by a const generic.
#[allow(clippy::too_many_arguments)]
fn emit_core_compiler_fence(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    ordering: AtomicOrdering,
) -> TranslationResult<Ptr<Operation>> {
    if !args.is_empty() {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "core compiler fence expects no arguments, got {}",
                args.len()
            ))
        );
    }

    let (prepared_destination, prev_op) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let barrier = build_compiler_fence_barrier(ctx, &loc, &ordering)?;
    barrier.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = prev_op {
        barrier.insert_after(ctx, prev);
    } else {
        barrier.insert_at_front(block_ptr, ctx);
    }

    // `core::sync::atomic::compiler_fence` returns unit.
    let unit_ty = dialect_mir::types::MirTupleType::get(ctx, vec![]);
    let unit_op = Operation::new(
        ctx,
        MirConstructTupleOp::get_concrete_op_info(),
        vec![unit_ty.into()],
        vec![],
        vec![],
        0,
    );
    unit_op.deref_mut(ctx).set_loc(loc.clone());
    unit_op.insert_after(ctx, barrier);
    let unit_val = unit_op.deref(ctx).get_result(0);

    helpers::emit_prepared_result_and_goto(
        ctx,
        prepared_destination,
        unit_val,
        target,
        block_ptr,
        unit_op,
        value_map,
        block_map,
        loc,
        "core compiler fence call without target block",
    )
}

/// Emit a core atomic fence.
///
/// MIR args: none; ordering is carried by a const generic. Core fences use
/// system scope, matching the rest of the `core::sync::atomic` importer.
#[allow(clippy::too_many_arguments)]
fn emit_core_atomic_fence(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    ordering: AtomicOrdering,
) -> TranslationResult<Ptr<Operation>> {
    if !args.is_empty() {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "core atomic fence expects no arguments, got {}",
                args.len()
            ))
        );
    }

    let (prepared_destination, prev_op) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let fence = NvvmAtomicFenceOp::build(ctx, ordering, AtomicScope::System);
    let fence_op = fence.get_operation();
    fence_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = prev_op {
        fence_op.insert_after(ctx, prev);
    } else {
        fence_op.insert_at_front(block_ptr, ctx);
    }

    // `core::sync::atomic::fence` returns unit.
    let unit_ty = dialect_mir::types::MirTupleType::get(ctx, vec![]);
    let unit_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructTupleOp::get_concrete_op_info(),
        vec![unit_ty.into()],
        vec![],
        vec![],
        0,
    );
    unit_op.deref_mut(ctx).set_loc(loc.clone());
    unit_op.insert_after(ctx, fence_op);
    let unit_val = unit_op.deref(ctx).get_result(0);

    helpers::emit_prepared_result_and_goto(
        ctx,
        prepared_destination,
        unit_val,
        target,
        block_ptr,
        unit_op,
        value_map,
        block_map,
        loc,
        "core atomic fence call without target block",
    )
}

/// Emit a core atomic load.
///
/// MIR args: `[ptr]` -- 1 arg, ordering from const generic.
#[allow(clippy::too_many_arguments)]
fn emit_core_atomic_load(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    type_info: &AtomicTypeInfo,
    ordering: AtomicOrdering,
) -> TranslationResult<Ptr<Operation>> {
    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Pointer atomics must preserve the translated MIR pointer type rather than
    // reconstructing the value as an integer of pointer width.
    let result_ty = if type_info.is_pointer {
        let ptr_ty = ptr_val.get_type(ctx);
        let ptr_ty = ptr_ty.deref(ctx);
        let Some(ptr_ty) = ptr_ty.downcast_ref::<dialect_mir::types::MirPtrType>() else {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(
                    "core atomic load address is not a MIR pointer".to_string()
                )
            );
        };

        ptr_ty.pointee
    } else {
        type_info.element_type(ctx)
    };

    let (prepared_destination, last_op) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let nvvm_op =
        NvvmAtomicLoadOp::build(ctx, ptr_val, result_ty, ordering, type_info.scope.clone());
    let op_ptr = nvvm_op.get_operation();
    op_ptr.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op_ptr.insert_after(ctx, prev);
    } else {
        op_ptr.insert_at_front(block_ptr, ctx);
    }

    let result_value = op_ptr.deref(ctx).get_result(0);
    helpers::emit_prepared_result_and_goto(
        ctx,
        prepared_destination,
        result_value,
        target,
        block_ptr,
        op_ptr,
        value_map,
        block_map,
        loc,
        "core atomic load call without target block",
    )
}

/// Emit a core atomic store.
///
/// MIR args: `[ptr, val]` -- 2 args, ordering from const generic.
#[allow(clippy::too_many_arguments)]
fn emit_core_atomic_store(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    type_info: &AtomicTypeInfo,
    ordering: AtomicOrdering,
) -> TranslationResult<Ptr<Operation>> {
    // Get the value to store (arg 1)
    let (val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Get the pointer (arg 0)
    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let (prepared_destination, last_op) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let nvvm_op = NvvmAtomicStoreOp::build(ctx, val, ptr_val, ordering, type_info.scope.clone());
    let op_ptr = nvvm_op.get_operation();
    op_ptr.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op_ptr.insert_after(ctx, prev);
    } else {
        op_ptr.insert_at_front(block_ptr, ctx);
    }

    // Store returns unit
    let unit_ty = dialect_mir::types::MirTupleType::get(ctx, vec![]);
    let unit_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructTupleOp::get_concrete_op_info(),
        vec![unit_ty.into()],
        vec![],
        vec![],
        0,
    );
    unit_op.deref_mut(ctx).set_loc(loc.clone());
    unit_op.insert_after(ctx, op_ptr);
    let unit_val = unit_op.deref(ctx).get_result(0);
    helpers::emit_prepared_result_and_goto(
        ctx,
        prepared_destination,
        unit_val,
        target,
        block_ptr,
        unit_op,
        value_map,
        block_map,
        loc,
        "core atomic store call without target block",
    )
}

/// Emit a core atomic read-modify-write operation.
///
/// MIR args: `[ptr, val]` -- 2 args, ordering from const generic.
#[allow(clippy::too_many_arguments)]
fn emit_core_atomic_rmw(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    type_info: &AtomicTypeInfo,
    ordering: AtomicOrdering,
    rmw_kind: AtomicRmwKind,
) -> TranslationResult<Ptr<Operation>> {
    // Get the pointer (arg 0)
    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Get the value operand (arg 1)
    let (val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let result_ty = if type_info.is_pointer {
        val.get_type(ctx)
    } else {
        type_info.element_type(ctx)
    };

    let (prepared_destination, last_op) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let nvvm_op = NvvmAtomicRmwOp::build(
        ctx,
        ptr_val,
        val,
        result_ty,
        rmw_kind,
        ordering,
        type_info.scope.clone(),
    );
    let op_ptr = nvvm_op.get_operation();
    op_ptr.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op_ptr.insert_after(ctx, prev);
    } else {
        op_ptr.insert_at_front(block_ptr, ctx);
    }

    let result_value = op_ptr.deref(ctx).get_result(0);
    helpers::emit_prepared_result_and_goto(
        ctx,
        prepared_destination,
        result_value,
        target,
        block_ptr,
        op_ptr,
        value_map,
        block_map,
        loc,
        "core atomic RMW call without target block",
    )
}

/// Emit a core atomic compare-and-exchange.
///
/// MIR args: `[ptr, old, new]` -- 3 args, orderings from const generics.
/// Returns `(old_val, bool)` tuple (LLVM cmpxchg semantics).
#[allow(clippy::too_many_arguments)]
fn emit_core_atomic_cmpxchg(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    type_info: &AtomicTypeInfo,
    success_ordering: AtomicOrdering,
    failure_ordering: AtomicOrdering,
) -> TranslationResult<Ptr<Operation>> {
    // Get the pointer (arg 0)
    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Get the expected (current) value (arg 1)
    let (cmp_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let result_ty = if type_info.is_pointer {
        cmp_val.get_type(ctx)
    } else {
        type_info.element_type(ctx)
    };

    // Get the new value (arg 2)
    let (new_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let (prepared_destination, last_op) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let nvvm_op = NvvmAtomicCmpxchgOp::build(
        ctx,
        ptr_val,
        cmp_val,
        new_val,
        result_ty,
        success_ordering,
        failure_ordering,
        type_info.scope.clone(),
    );
    let op_ptr = nvvm_op.get_operation();
    op_ptr.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op_ptr.insert_after(ctx, prev);
    } else {
        op_ptr.insert_at_front(block_ptr, ctx);
    }

    let result_value = op_ptr.deref(ctx).get_result(0);
    let bool_ty = types::get_bool_type(ctx).to_handle();
    let success_op = Operation::new(
        ctx,
        MirEqOp::get_concrete_op_info(),
        vec![bool_ty],
        vec![result_value, cmp_val],
        vec![],
        0,
    );
    success_op.deref_mut(ctx).set_loc(loc.clone());
    success_op.insert_after(ctx, op_ptr);

    let success_value = success_op.deref(ctx).get_result(0);
    // The destination place is typed `(T, bool)` in MIR; translate that
    // rustc type so the constructed tuple uniques with the destination's
    // layout-carrying tuple type.
    let dest_tuple_ty = destination.ty(body.locals()).map_err(|e| {
        input_error_noloc!(TranslationErr::unsupported(format!(
            "Failed to query atomic cmpxchg destination type: {:?}",
            e
        )))
    })?;
    let tuple_ty = crate::translator::types::translate_type(ctx, &dest_tuple_ty)?;
    let tuple_op = Operation::new(
        ctx,
        MirConstructTupleOp::get_concrete_op_info(),
        vec![tuple_ty],
        vec![result_value, success_value],
        vec![],
        0,
    );
    tuple_op.deref_mut(ctx).set_loc(loc.clone());
    tuple_op.insert_after(ctx, success_op);

    let tuple_value = tuple_op.deref(ctx).get_result(0);
    helpers::emit_prepared_result_and_goto(
        ctx,
        prepared_destination,
        tuple_value,
        target,
        block_ptr,
        tuple_op,
        value_map,
        block_map,
        loc,
        "core atomic cmpxchg call without target block",
    )
}

#[cfg(test)]
mod tests {
    use super::{
        AtomicTypeInfo, CoreIntrinsicRoute, build_compiler_fence_barrier,
        core_atomic_width_is_supported, intrinsic_op_to_rmw_kind,
        intrinsic_ordering_from_discriminant, parse_core_intrinsic_op, route_core_intrinsic,
    };
    use dialect_nvvm::ops::atomic::{AtomicRmwKind, AtomicScope};
    use dialect_nvvm::ops::{AtomicOrdering, InlinePtxOp};
    use pliron::common_traits::Verify;
    use pliron::context::Context;
    use pliron::location::Location;

    fn test_context() -> Context {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_nvvm::register(&mut ctx);
        ctx
    }

    /// The regression gate for issue #781: `singlethreadfence` (the intrinsic
    /// behind `core::sync::atomic::compiler_fence`) must take the
    /// compiler-fence route through `dispatch_core_intrinsic`. Without this
    /// routing it falls through to the element-typed dispatch and any kernel
    /// calling `compiler_fence` fails to compile with a missing-element-type
    /// diagnostic.
    #[test]
    fn compiler_fence_takes_the_compiler_fence_route() {
        assert_eq!(
            route_core_intrinsic("singlethreadfence"),
            CoreIntrinsicRoute::CompilerFence
        );
    }

    /// `fence` stays on the hardware-barrier route and the element-typed ops
    /// stay on the typed dispatch: the compiler-fence routing must not widen.
    #[test]
    fn fence_and_typed_ops_keep_their_routes() {
        assert_eq!(
            route_core_intrinsic("fence"),
            CoreIntrinsicRoute::HardwareFence
        );
        for op in ["load", "store", "xadd", "cxchg", "cxchgweak", ""] {
            assert_eq!(
                route_core_intrinsic(op),
                CoreIntrinsicRoute::Typed,
                "{op:?}"
            );
        }
    }

    /// `compiler_fence(Relaxed)` panics in core, so only a direct nightly
    /// intrinsic call can carry Relaxed here. The importer refuses it instead
    /// of emitting a barrier that silently means something else.
    #[test]
    fn compiler_fence_refuses_relaxed_ordering() {
        let mut ctx = test_context();
        let err =
            build_compiler_fence_barrier(&mut ctx, &Location::Unknown, &AtomicOrdering::Relaxed)
                .expect_err("Relaxed must be refused");
        assert!(
            err.to_string().contains("Relaxed"),
            "diagnostic must name the refused ordering: {err}"
        );
    }

    /// Every ordering safe Rust can pass to `compiler_fence` encodes as an
    /// empty, volatile, non-convergent inline-PTX block whose only content is
    /// the `~{memory}` clobber: no instruction text (so no hardware `fence` or
    /// `membar`), no results, and it satisfies the inline-PTX verifier (zero
    /// results paired against zero `=` output constraints).
    #[test]
    fn compiler_fence_barrier_is_an_empty_memory_clobber() {
        let mut ctx = test_context();
        for ordering in [
            AtomicOrdering::Acquire,
            AtomicOrdering::Release,
            AtomicOrdering::AcqRel,
            AtomicOrdering::SeqCst,
        ] {
            let op = build_compiler_fence_barrier(&mut ctx, &Location::Unknown, &ordering)
                .unwrap_or_else(|e| panic!("{ordering:?} must be accepted: {e}"));
            let barrier = InlinePtxOp::new(op);
            assert_eq!(
                barrier
                    .get_attr_ptx_template(&ctx)
                    .map(|s| String::from((*s).clone()))
                    .as_deref(),
                Some(""),
                "{ordering:?}: the barrier must emit no PTX instruction"
            );
            assert_eq!(
                barrier
                    .get_attr_ptx_constraints(&ctx)
                    .map(|s| String::from((*s).clone()))
                    .as_deref(),
                Some("~{memory}"),
                "{ordering:?}: the barrier must clobber memory"
            );
            assert!(
                barrier
                    .get_attr_ptx_sideeffect(&ctx)
                    .is_some_and(|b| bool::from((*b).clone())),
                "{ordering:?}: the barrier must be side-effecting"
            );
            assert!(
                barrier
                    .get_attr_ptx_convergent(&ctx)
                    .is_some_and(|b| !bool::from((*b).clone())),
                "{ordering:?}: the barrier must not be convergent"
            );
            assert_eq!(
                op.deref(&ctx).get_num_results(),
                0,
                "{ordering:?}: the barrier has no results"
            );
            barrier
                .verify(&ctx)
                .unwrap_or_else(|e| panic!("{ordering:?}: verifier must accept the barrier: {e}"));
        }
    }

    /// Both ordering-only fences must be recognised by name. `singlethreadfence`
    /// carries no element type, so if it is not matched here it falls through to
    /// the load/store/RMW generics extraction and is reported as a missing
    /// element type -- which is what issue #781 was.
    #[test]
    fn ordering_only_fences_are_recognised_by_name() {
        for (path, expected) in [
            ("core::intrinsics::atomic_fence", "fence"),
            ("std::intrinsics::atomic_fence", "fence"),
            (
                "core::intrinsics::atomic_singlethreadfence",
                "singlethreadfence",
            ),
            (
                "std::intrinsics::atomic_singlethreadfence",
                "singlethreadfence",
            ),
        ] {
            assert_eq!(parse_core_intrinsic_op(path), Some(expected), "{path}");
        }
    }

    /// `singlethreadfence` must not be confused with `fence`: they take
    /// different routes, one emitting a hardware barrier and one emitting none.
    #[test]
    fn compiler_fence_is_not_parsed_as_a_hardware_fence() {
        assert_ne!(
            parse_core_intrinsic_op("core::intrinsics::atomic_singlethreadfence"),
            parse_core_intrinsic_op("core::intrinsics::atomic_fence")
        );
    }

    #[test]
    fn core_atomics_accept_only_current_backend_widths() {
        assert!(core_atomic_width_is_supported(32));
        assert!(core_atomic_width_is_supported(64));
        for width in [8, 16, 128] {
            assert!(!core_atomic_width_is_supported(width));
        }
    }

    #[test]
    fn core_pointer_atomics_only_allow_exchange_rmw() {
        let info = AtomicTypeInfo {
            bit_width: 64,
            is_float: false,
            is_signed: false,
            is_pointer: true,
            scope: AtomicScope::System,
        };

        assert_eq!(
            intrinsic_op_to_rmw_kind("xchg", &info),
            Some(AtomicRmwKind::Xchg)
        );

        for op in [
            "xadd", "xsub", "and", "or", "xor", "min", "umin", "max", "umax",
        ] {
            assert_eq!(
                intrinsic_op_to_rmw_kind(op, &info),
                None,
                "pointer RMW `{op}` must remain unsupported"
            );
        }
    }

    #[test]
    fn core_atomic_ordering_discriminants_match_rustc() {
        assert_eq!(
            intrinsic_ordering_from_discriminant(0),
            Some(AtomicOrdering::Relaxed)
        );
        assert_eq!(
            intrinsic_ordering_from_discriminant(1),
            Some(AtomicOrdering::Release)
        );
        assert_eq!(
            intrinsic_ordering_from_discriminant(2),
            Some(AtomicOrdering::Acquire)
        );
        assert_eq!(
            intrinsic_ordering_from_discriminant(3),
            Some(AtomicOrdering::AcqRel)
        );
        assert_eq!(
            intrinsic_ordering_from_discriminant(4),
            Some(AtomicOrdering::SeqCst)
        );
        assert_eq!(intrinsic_ordering_from_discriminant(5), None);
    }
}
