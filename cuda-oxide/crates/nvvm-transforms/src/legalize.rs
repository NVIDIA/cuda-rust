/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Converts modern LLVM operations to the LLVM 7 forms accepted by
//! pre-Blackwell libNVVM.
//!
//! The pass runs after MIR-to-LLVM lowering. It preserves operation semantics
//! and reports an error when no equivalent legacy form is available.

use std::num::NonZeroUsize;

use llvm_export::{
    attributes::{
        AtomicOrderingAttr, AtomicRmwKindAttr, FCmpPredicateAttr, FastmathFlagsAttr,
        ICmpPredicateAttr, IntegerOverflowFlagsAttr, SyncScopeAttr,
    },
    op_interfaces::{
        BinArithOp, CastOpInterface, CastOpWithNNegInterface, FastMathFlags,
        IntBinArithOpWithOverflowFlag, SyncScopeInterface,
    },
    ops as llvm,
    ops::{AsmKind, InlineAsmOpExt},
    types as llvm_types,
};
use pliron::{
    attribute::Attribute,
    builtin::{
        attributes::{BoolAttr, FPDoubleAttr, FPSingleAttr, IntegerAttr, TypeAttr},
        op_interfaces::{CallOpCallable, CallOpInterface, SymbolOpInterface},
        type_interfaces::FunctionTypeInterface,
        types::{FP16Type, FP32Type, FP64Type, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    identifier::Identifier,
    linked_list::ContainsLinkedList,
    location::Located,
    op::{Op, op_cast},
    operation::Operation,
    result::Result,
    r#type::{TypeHandle, Typed},
    utils::apint::APInt,
    value::Value,
};

use crate::helpers;

const NNEG_ATTR: &str = "llvm_nneg_flag";

/// Rewrite a lowered LLVM module to the LLVM 7 subset used by legacy NVVM IR.
///
/// Floating-point atomic add is rewritten to the NVVM intrinsic accepted by
/// LLVM 7. Integer `atomicrmw` and `cmpxchg` operations remain native when
/// legacy libNVVM can preserve their semantics exactly. Scoped integer atomics
/// and compare-exchange operations with stronger failure ordering are rewritten
/// to inline PTX instead of weakening scope or ordering. Other atomic and fence
/// operations return an error instead of being emitted with unverified
/// semantics.
pub(crate) fn legalize_for_legacy_nvvm(
    ctx: &mut Context,
    module: Ptr<Operation>,
    capability: u32,
) -> Result<()> {
    let mut ops = Vec::new();
    collect_ops(ctx, module, &mut ops);

    // Validate the complete module before changing it. Some unsupported
    // ordering and scope settings are otherwise ignored by libNVVM.
    let mut float_atomic_adds = Vec::new();
    let mut scalar_rmw_rewrites = Vec::new();
    let mut scalar_cmpxchg_rewrites = Vec::new();
    for &op in &ops {
        reject_nonportable_f16_types(ctx, op)?;
        reject_unsupported_op(ctx, op)?;
        validate_rewrite_candidate(ctx, op)?;
        if let Some(rewrite) = validate_scalar_atomic_rmw(ctx, op, capability)? {
            scalar_rmw_rewrites.push(rewrite);
        }
        if let Some(rewrite) = validate_scalar_cmpxchg(ctx, op, capability)? {
            scalar_cmpxchg_rewrites.push(rewrite);
        }
        if let Some(intrinsic) = validate_float_atomic_add(ctx, op, capability)? {
            float_atomic_adds.push((op, intrinsic));
        }
    }

    let mut obsolete_declarations = Vec::new();
    for op in ops {
        remove_nneg(ctx, op);

        if let Some(rewrite) = scalar_rmw_rewrites.iter().find(|rewrite| rewrite.op == op) {
            rewrite_scalar_atomic_rmw(ctx, rewrite)?;
            continue;
        }

        if let Some(rewrite) = scalar_cmpxchg_rewrites
            .iter()
            .find(|rewrite| rewrite.op == op)
        {
            rewrite_scalar_cmpxchg_with_ptx(ctx, rewrite)?;
            continue;
        }

        if let Some(intrinsic) = float_atomic_adds
            .iter()
            .find_map(|(candidate, intrinsic)| (*candidate == op).then_some(intrinsic))
        {
            rewrite_float_atomic_add(ctx, op, intrinsic)?;
            continue;
        }

        if Operation::get_op::<llvm::FNegOp>(op, ctx).is_some() {
            rewrite_fneg(ctx, op)?;
            continue;
        }

        let Some(call) = Operation::get_op::<llvm::CallOp>(op, ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(ctx) else {
            continue;
        };
        let name = callee.to_string();

        if parse_integer_sat_name(&name).is_some() {
            rewrite_integer_sat(ctx, op, &name)?;
            obsolete_declarations.push(name);
        } else if parse_float_to_int_sat_name(&name).is_some() {
            rewrite_float_to_int_sat(ctx, op, &name)?;
            obsolete_declarations.push(name);
        } else if legacy_bit_rewrite(&name).is_some() {
            rewrite_legacy_bit_intrinsic(ctx, op, &name)?;
            obsolete_declarations.push(name);
        } else if shuffle_mode(&name).is_some() {
            rewrite_shuffle(ctx, op, &name)?;
            obsolete_declarations.push(name);
        } else if vote_mode(&name).is_some() {
            rewrite_vote(ctx, op, &name)?;
            obsolete_declarations.push(name);
        } else if let Some(legacy_name) = legacy_memory_intrinsic_name(&name) {
            rewrite_renamed_call(ctx, op, &legacy_name)?;
            obsolete_declarations.push(name);
        }
    }

    remove_obsolete_declarations(ctx, module, &obsolete_declarations);
    verify_legacy_subset(ctx, module, capability)
}

/// Rewrite bit-manipulation intrinsics whose integer widths NVVM rejects in
/// both textual dialects.
///
/// Unlike [`legalize_for_legacy_nvvm`], this leaves modern operations, flags,
/// f16, atomics, and intrinsic signatures unchanged. Blackwell and newer
/// targets accept those forms, but NVVM bit intrinsics still stop at i64.
pub(crate) fn legalize_nvvm_bit_intrinsics(
    ctx: &mut Context,
    module: Ptr<Operation>,
) -> Result<()> {
    let mut ops = Vec::new();
    collect_ops(ctx, module, &mut ops);

    // Validate every candidate before changing the module.
    for &op in &ops {
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(ctx) else {
            continue;
        };
        let name = callee.to_string();
        if legacy_bit_rewrite(&name).is_some() {
            validate_legacy_bit_call(ctx, op, &name)?;
        }
    }

    let mut obsolete_declarations = Vec::new();
    for op in ops {
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(ctx) else {
            continue;
        };
        let name = callee.to_string();
        if legacy_bit_rewrite(&name).is_some() {
            rewrite_legacy_bit_intrinsic(ctx, op, &name)?;
            obsolete_declarations.push(name);
        }
    }

    remove_obsolete_declarations(ctx, module, &obsolete_declarations);
    let mut rewritten_ops = Vec::new();
    collect_ops(ctx, module, &mut rewritten_ops);
    for op in rewritten_ops {
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(ctx) else {
            continue;
        };
        if legacy_bit_rewrite(callee.as_ref()).is_some() {
            return pliron::input_err!(
                op.deref(ctx).loc(),
                "NVVM bit-intrinsic legalization left unsupported call @{callee} behind"
            );
        }
    }
    Ok(())
}

/// CUDA 12's LLVM 7 NVVM dialect does not support the scalar `half` type.
///
/// This checks operands, results, function and global signatures, block
/// arguments, aggregate elements, and operation type attributes. Named
/// recursive structs are guarded by the `seen` set.
fn reject_nonportable_f16_types(ctx: &Context, op: Ptr<Operation>) -> Result<()> {
    let has_half = op
        .deref(ctx)
        .operands()
        .any(|value| type_contains_f16(ctx, value.get_type(ctx), &mut Vec::new()))
        || op
            .deref(ctx)
            .results()
            .any(|value| type_contains_f16(ctx, value.get_type(ctx), &mut Vec::new()))
        || op.deref(ctx).attributes.0.values().any(|attr| {
            attr.downcast_ref::<TypeAttr>()
                .is_some_and(|ty| type_contains_f16(ctx, ty.get_type(ctx), &mut Vec::new()))
        })
        || op.deref(ctx).regions().any(|region| {
            region.deref(ctx).iter(ctx).any(|block| {
                block
                    .deref(ctx)
                    .arguments()
                    .any(|argument| type_contains_f16(ctx, argument.get_type(ctx), &mut Vec::new()))
            })
        })
        || Operation::get_op::<llvm::FuncOp>(op, ctx)
            .is_some_and(|func| type_contains_f16(ctx, func.get_type(ctx).into(), &mut Vec::new()))
        || Operation::get_op::<llvm::GlobalOp>(op, ctx)
            .is_some_and(|global| type_contains_f16(ctx, global.get_type(ctx), &mut Vec::new()));

    if has_half {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "legacy NVVM IR cannot contain scalar f16 because cuda-oxide supports the CUDA 12 LLVM 7 dialect; use f32/f64 or the modern NVVM/PTX path"
        );
    }
    Ok(())
}

fn type_contains_f16(ctx: &Context, ty: TypeHandle, seen: &mut Vec<TypeHandle>) -> bool {
    if ty.deref(ctx).is::<FP16Type>() {
        return true;
    }
    if seen.contains(&ty) {
        return false;
    }
    seen.push(ty);

    let ty_ref = ty.deref(ctx);
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
        !struct_ty.is_opaque()
            && struct_ty
                .fields()
                .any(|field| type_contains_f16(ctx, field, seen))
    } else if let Some(array_ty) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
        type_contains_f16(ctx, array_ty.elem_type(), seen)
    } else if let Some(vector_ty) = ty_ref.downcast_ref::<llvm_types::VectorType>() {
        type_contains_f16(ctx, vector_ty.elem_type(), seen)
    } else if let Some(func_ty) = ty_ref.downcast_ref::<llvm_types::FuncType>() {
        type_contains_f16(ctx, func_ty.result_type(), seen)
            || func_ty
                .arg_types()
                .into_iter()
                .any(|arg| type_contains_f16(ctx, arg, seen))
    } else {
        false
    }
}

fn collect_ops(ctx: &Context, root: Ptr<Operation>, out: &mut Vec<Ptr<Operation>>) {
    out.push(root);
    let regions: Vec<_> = root.deref(ctx).regions().collect();
    for region in regions {
        let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
        for block in blocks {
            let children: Vec<_> = block.deref(ctx).iter(ctx).collect();
            for child in children {
                collect_ops(ctx, child, out);
            }
        }
    }
}

fn reject_unsupported_op(ctx: &Context, op: Ptr<Operation>) -> Result<()> {
    let reason = if Operation::get_op::<llvm::AtomicLoadOp>(op, ctx).is_some() {
        Some("atomic loads")
    } else if Operation::get_op::<llvm::AtomicStoreOp>(op, ctx).is_some() {
        Some("atomic stores")
    } else if let Some(rmw) = Operation::get_op::<llvm::AtomicRmwOp>(op, ctx) {
        let value_ty = op.deref(ctx).get_operand(1).get_type(ctx);
        if float_width(ctx, value_ty).is_some() {
            match rmw.get_attr_llvm_rmw_kind(ctx).as_deref() {
                Some(AtomicRmwKindAttr::FAdd) => None,
                _ => Some("floating-point atomic read-modify-write operations other than add"),
            }
        } else {
            // The scalar atomic validator owns integer/pointer type, width,
            // operation, address-space, ordering and scope validation.
            None
        }
    } else if Operation::get_op::<llvm::AtomicCmpxchgOp>(op, ctx).is_some() {
        // Complete compare-exchange validation follows below.
        None
    } else if Operation::get_op::<llvm::FenceOp>(op, ctx).is_some() {
        Some("LLVM fences")
    } else if Operation::get_op::<llvm::DebugValueOp>(op, ctx).is_some() {
        Some("LLVM debug-value records")
    } else {
        None
    };

    if let Some(reason) = reason {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "cuda-oxide has not yet legalized {reason} for legacy NVVM IR; use ordinary PTX output or a Blackwell NVVM target"
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LegacyAtomicScope {
    Device,
    Block,
    System,
}

impl LegacyAtomicScope {
    fn from_syncscope(scope: &SyncScopeAttr) -> Option<Self> {
        match scope {
            SyncScopeAttr::NamedScope(name) if name.as_str() == "device" => Some(Self::Device),
            SyncScopeAttr::NamedScope(name) if name.as_str() == "block" => Some(Self::Block),
            SyncScopeAttr::System => Some(Self::System),
            _ => None,
        }
    }

    fn ptx(self) -> &'static str {
        match self {
            Self::Device => "gpu",
            Self::Block => "cta",
            Self::System => "sys",
        }
    }

    fn needs_scoped_rewrite(self) -> bool {
        !matches!(self, Self::Device)
    }
}

struct ScalarRmwRewrite {
    op: Ptr<Operation>,
    width: u32,
    kind: AtomicRmwKindAttr,
    scope: LegacyAtomicScope,
}

struct ScalarCmpxchgRewrite {
    op: Ptr<Operation>,
    extract: Ptr<Operation>,
    width: u32,
    success: AtomicOrderingAttr,
    scope: LegacyAtomicScope,
}

fn legacy_atomic_scope(
    ctx: &Context,
    op: Ptr<Operation>,
    syncscope: &SyncScopeAttr,
    operation: &str,
) -> Result<LegacyAtomicScope> {
    LegacyAtomicScope::from_syncscope(syncscope).ok_or_else(|| {
        pliron::input_error!(
            op.deref(ctx).loc(),
            "legacy NVVM {operation} has an unsupported synchronization scope"
        )
    })
}

fn generic_atomic_pointer(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    ptr: Value,
    address_space: u32,
) -> Value {
    if address_space == 0 {
        return ptr;
    }

    let generic_ty: TypeHandle = llvm_types::PointerType::get(ctx, 0).into();
    let cast = llvm::AddrSpaceCastOp::new(ctx, ptr, generic_ty);
    insert_before(ctx, anchor, cast.get_operation())
}

fn integer_atomic_reg_constraint(width: u32) -> &'static str {
    match width {
        32 => "r",
        64 => "l",
        _ => unreachable!("validated legacy integer atomic width"),
    }
}

fn is_generic_pointer(ctx: &Context, ty: TypeHandle) -> bool {
    ty.deref(ctx)
        .downcast_ref::<llvm_types::PointerType>()
        .is_some_and(|pointer| pointer.address_space() == 0)
}

fn atomic_scalar_width(ctx: &Context, ty: TypeHandle) -> Option<u32> {
    if is_generic_pointer(ctx, ty) {
        // Device compilation targets nvptx64. Preserve the pointer type;
        // only the PTX instruction/register representation needs this width.
        Some(64)
    } else {
        scalar_integer_width(ctx, ty)
    }
}

fn integer_rmw_ptx_opcode(kind: &AtomicRmwKindAttr, width: u32) -> Option<String> {
    let suffix = match kind {
        AtomicRmwKindAttr::Xchg => format!("exch.b{width}"),
        AtomicRmwKindAttr::Add | AtomicRmwKindAttr::Sub => format!("add.u{width}"),
        AtomicRmwKindAttr::And => format!("and.b{width}"),
        AtomicRmwKindAttr::Or => format!("or.b{width}"),
        AtomicRmwKindAttr::Xor => format!("xor.b{width}"),
        AtomicRmwKindAttr::Max => format!("max.s{width}"),
        AtomicRmwKindAttr::Min => format!("min.s{width}"),
        AtomicRmwKindAttr::UMax => format!("max.u{width}"),
        AtomicRmwKindAttr::UMin => format!("min.u{width}"),
        _ => return None,
    };
    Some(suffix)
}

fn valid_cmpxchg_ordering_pair(
    _success: &AtomicOrderingAttr,
    failure: &AtomicOrderingAttr,
) -> bool {
    // Rust and current LLVM permit failure to be stronger than success.
    // Failure is a load, so release and acq_rel remain invalid.
    matches!(
        failure,
        AtomicOrderingAttr::Monotonic | AtomicOrderingAttr::Acquire | AtomicOrderingAttr::SeqCst
    )
}

fn combined_cmpxchg_ordering(
    success: &AtomicOrderingAttr,
    failure: &AtomicOrderingAttr,
) -> AtomicOrderingAttr {
    use AtomicOrderingAttr::*;
    match (success, failure) {
        (SeqCst, _) | (_, SeqCst) => SeqCst,
        (Monotonic, Acquire) => Acquire,
        (Release, Acquire) => AcqRel,
        _ => success.clone(),
    }
}

fn cmpxchg_ptx_semantics(success: &AtomicOrderingAttr) -> (&'static str, bool) {
    match success {
        AtomicOrderingAttr::Monotonic => ("relaxed", false),
        AtomicOrderingAttr::Acquire => ("acquire", false),
        AtomicOrderingAttr::Release => ("release", false),
        AtomicOrderingAttr::AcqRel => ("acq_rel", false),
        // PTX has no seq_cst atom semantic. The PTX atomic ABI maps a
        // sequentially consistent RMW to a preceding fence.sc plus an
        // acquire RMW at the same scope.
        AtomicOrderingAttr::SeqCst => ("acquire", true),
    }
}

fn find_cmpxchg_old_value_extract(
    ctx: &Context,
    cas_op: Ptr<Operation>,
    value_ty: TypeHandle,
) -> Result<Ptr<Operation>> {
    let aggregate = cas_op.deref(ctx).get_result(0);
    let uses = aggregate.uses(ctx);

    if uses.len() != 1 {
        return pliron::input_err!(
            cas_op.deref(ctx).loc(),
            "legacy NVVM compare-exchange PTX rewrite requires exactly one canonical old-value extract"
        );
    }

    let old_value_use = &uses[0];
    let extract = old_value_use.user_op();

    if old_value_use.find_index(ctx) != 0
        || Operation::get_op::<llvm::ExtractValueOp>(extract, ctx).is_none()
        || extract.deref(ctx).get_num_results() != 1
        || extract.deref(ctx).get_result(0).get_type(ctx) != value_ty
    {
        return pliron::input_err!(
            cas_op.deref(ctx).loc(),
            "legacy NVVM compare-exchange PTX rewrite requires the canonical old-value extract"
        );
    }

    Ok(extract)
}

fn atomic_pointer_address_space(
    ctx: &Context,
    op: Ptr<Operation>,
    ptr: Value,
    operation: &str,
) -> Result<u32> {
    ptr.get_type(ctx)
        .deref(ctx)
        .downcast_ref::<llvm_types::PointerType>()
        .map(llvm_types::PointerType::address_space)
        .ok_or_else(|| {
            pliron::input_error!(
                op.deref(ctx).loc(),
                "legacy NVVM {operation} requires a pointer operand"
            )
        })
}

/// Validate integer and generic-pointer `atomicrmw` operations for legacy NVVM.
///
/// Device-scoped operations that are directly representable in LLVM 7 remain
/// native LLVM operations. Pointer exchange and block/system-scoped operations are scheduled
/// for an inline-PTX rewrite so their synchronization scope is not weakened by
/// legacy libNVVM. Strong Rust orderings have already been split into fences by
/// MIR lowering, so the LLVM `atomicrmw` itself must always be monotonic here.
fn validate_scalar_atomic_rmw(
    ctx: &Context,
    op: Ptr<Operation>,
    capability: u32,
) -> Result<Option<ScalarRmwRewrite>> {
    let Some(rmw) = Operation::get_op::<llvm::AtomicRmwOp>(op, ctx) else {
        return Ok(None);
    };
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let (ptr, value) = (operands[0], operands[1]);
    let value_ty = value.get_type(ctx);
    if float_width(ctx, value_ty).is_some() {
        return Ok(None);
    }

    let pointer_value = is_generic_pointer(ctx, value_ty);
    let width = atomic_scalar_width(ctx, value_ty).ok_or_else(|| {
        pliron::input_error!(
            op.deref(ctx).loc(),
            "legacy NVVM atomic RMW requires a scalar integer or generic pointer value"
        )
    })?;
    if !matches!(width, 32 | 64) {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "legacy NVVM integer atomic RMW supports only i32 and i64, not i{width}"
        );
    }

    let kind = rmw
        .get_attr_llvm_rmw_kind(ctx)
        .as_deref()
        .cloned()
        .ok_or_else(|| {
            pliron::input_error!(
                op.deref(ctx).loc(),
                "legacy NVVM integer atomic RMW is missing its operation kind"
            )
        })?;
    if pointer_value && kind != AtomicRmwKindAttr::Xchg {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "legacy NVVM pointer atomic RMW supports only exchange"
        );
    }
    if integer_rmw_ptx_opcode(&kind, width).is_none() {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "legacy NVVM integer atomic RMW does not support this operation kind"
        );
    }

    if rmw.get_attr_llvm_rmw_ordering(ctx).as_deref() != Some(&AtomicOrderingAttr::Monotonic) {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "legacy NVVM integer atomic RMW requires monotonic ordering after fence splitting"
        );
    }

    let address_space = atomic_pointer_address_space(ctx, op, ptr, "integer atomic RMW")?;
    if !matches!(address_space, 0 | 1 | 3) {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "legacy NVVM integer atomic RMW does not support address space {address_space}"
        );
    }

    let syncscope = rmw.syncscope(ctx);
    let scope = legacy_atomic_scope(ctx, op, &syncscope, "integer atomic RMW")?;

    // LLVM 7 rejects pointer-valued atomicrmw even at device scope. The
    // existing PTX route accepts pointer registers without integer casts.
    if !pointer_value && !scope.needs_scoped_rewrite() {
        return Ok(None);
    }
    if capability < 70 {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "legacy scoped integer atomic RMW requires sm_70 or newer; the build targets sm_{capability}"
        );
    }

    Ok(Some(ScalarRmwRewrite {
        op,
        width,
        kind,
        scope,
    }))
}

fn rewrite_scalar_atomic_rmw(ctx: &mut Context, rewrite: &ScalarRmwRewrite) -> Result<()> {
    let operands: Vec<_> = rewrite.op.deref(ctx).operands().collect();
    let (ptr, value) = (operands[0], operands[1]);
    let address_space = atomic_pointer_address_space(ctx, rewrite.op, ptr, "integer atomic RMW")?;
    let ptr = generic_atomic_pointer(ctx, rewrite.op, ptr, address_space);

    // PTX has no atomic subtraction opcode. Modular integer subtraction is
    // exactly atomic addition of the two's-complement negation.
    let value = if matches!(rewrite.kind, AtomicRmwKindAttr::Sub) {
        let zero = integer_const(ctx, rewrite.op, rewrite.width, 0);
        let negated = llvm::SubOp::new_with_overflow_flag(
            ctx,
            zero,
            value,
            IntegerOverflowFlagsAttr::default(),
        )
        .get_operation();
        insert_before(ctx, rewrite.op, negated)
    } else {
        value
    };

    let opcode = integer_rmw_ptx_opcode(&rewrite.kind, rewrite.width)
        .expect("validated legacy integer atomic RMW kind");
    let reg = integer_atomic_reg_constraint(rewrite.width);
    let template = format!(
        "atom.relaxed.{}.{} $0, [$1], $2;",
        rewrite.scope.ptx(),
        opcode
    );
    let asm = llvm::InlineAsmOp::build(
        ctx,
        value.get_type(ctx),
        vec![ptr, value],
        &template,
        &format!("={reg},l,{reg},~{{memory}}"),
        AsmKind::SideEffect,
    );
    let result = insert_before(ctx, rewrite.op, asm.get_operation());
    replace_one_result(ctx, rewrite.op, result)
}

/// Validate integer and generic-pointer compare-exchange operations for legacy NVVM.
///
/// Integer device scope plus monotonic success and failure ordering remains a native
/// LLVM `cmpxchg`, matching the bare-atomicrmw/cmpxchg subset the legacy
/// LLVM 7 dialect handles directly. Block/system scope or any ordered success
/// or failure ordering is rewritten to scoped inline PTX because legacy
/// libNVVM accepts LLVM's cmpxchg ordering fields but lowers ordered forms to
/// a bare unordered `atom.cas`. Pointer values share this PTX route at every scope.
fn validate_scalar_cmpxchg(
    ctx: &Context,
    op: Ptr<Operation>,
    capability: u32,
) -> Result<Option<ScalarCmpxchgRewrite>> {
    let Some(cas) = Operation::get_op::<llvm::AtomicCmpxchgOp>(op, ctx) else {
        return Ok(None);
    };
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let (ptr, compare, new_value) = (operands[0], operands[1], operands[2]);
    let value_ty = compare.get_type(ctx);
    let pointer_value = is_generic_pointer(ctx, value_ty);
    let width = atomic_scalar_width(ctx, value_ty).ok_or_else(|| {
        pliron::input_error!(
            op.deref(ctx).loc(),
            "legacy NVVM compare-exchange requires a scalar integer or generic pointer value"
        )
    })?;
    if !matches!(width, 32 | 64) {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "legacy NVVM compare-exchange supports only i32 and i64, not i{width}"
        );
    }
    if new_value.get_type(ctx) != value_ty {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "legacy NVVM compare-exchange compare and replacement values must have the same type"
        );
    }

    let success = cas
        .get_attr_llvm_cas_success_ordering(ctx)
        .as_deref()
        .cloned()
        .ok_or_else(|| {
            pliron::input_error!(
                op.deref(ctx).loc(),
                "legacy NVVM compare-exchange is missing its success ordering"
            )
        })?;
    let failure = cas
        .get_attr_llvm_cas_failure_ordering(ctx)
        .as_deref()
        .cloned()
        .ok_or_else(|| {
            pliron::input_error!(
                op.deref(ctx).loc(),
                "legacy NVVM compare-exchange is missing its failure ordering"
            )
        })?;
    if !valid_cmpxchg_ordering_pair(&success, &failure) {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "legacy NVVM compare-exchange has an invalid success/failure ordering pair"
        );
    }

    let address_space = atomic_pointer_address_space(ctx, op, ptr, "compare-exchange")?;
    if !matches!(address_space, 0 | 1 | 3) {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "legacy NVVM compare-exchange does not support address space {address_space}"
        );
    }

    let syncscope = cas.syncscope(ctx);
    let scope = legacy_atomic_scope(ctx, op, &syncscope, "compare-exchange")?;
    // PTX CAS has one ordering for both outcomes. Join the two requested
    // orderings rather than dropping stronger failure semantics. This also
    // avoids emitting pairs that the older LLVM 7 verifier would reject.
    let combined_ordering = combined_cmpxchg_ordering(&success, &failure);
    let needs_rewrite = pointer_value
        || scope.needs_scoped_rewrite()
        || !matches!(combined_ordering, AtomicOrderingAttr::Monotonic);

    if !needs_rewrite {
        return Ok(None);
    }
    if capability < 70 {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "legacy compare-exchange PTX rewrite requires sm_70 or newer; the build targets sm_{capability}"
        );
    }

    let extract = find_cmpxchg_old_value_extract(ctx, op, value_ty)?;

    Ok(Some(ScalarCmpxchgRewrite {
        op,
        extract,
        width,
        success: combined_ordering,
        scope,
    }))
}

fn rewrite_scalar_cmpxchg_with_ptx(
    ctx: &mut Context,
    rewrite: &ScalarCmpxchgRewrite,
) -> Result<()> {
    let operands: Vec<_> = rewrite.op.deref(ctx).operands().collect();
    let (ptr, compare, new_value) = (operands[0], operands[1], operands[2]);
    let address_space = atomic_pointer_address_space(ctx, rewrite.op, ptr, "compare-exchange")?;
    let ptr = generic_atomic_pointer(ctx, rewrite.op, ptr, address_space);

    let (sem, seq_cst) = cmpxchg_ptx_semantics(&rewrite.success);
    let scope = rewrite.scope.ptx();
    let mut template = String::new();
    if seq_cst {
        template.push_str(&format!("fence.sc.{scope}; "));
    }
    template.push_str(&format!(
        "atom.{sem}.{scope}.cas.b{} $0, [$1], $2, $3;",
        rewrite.width
    ));

    let reg = integer_atomic_reg_constraint(rewrite.width);
    let asm = llvm::InlineAsmOp::build(
        ctx,
        compare.get_type(ctx),
        vec![ptr, compare, new_value],
        &template,
        &format!("={reg},l,{reg},{reg},~{{memory}}"),
        AsmKind::SideEffect,
    );
    let old_value = insert_before(ctx, rewrite.op, asm.get_operation());

    // The MIR lowering produces `cmpxchg` followed by `extractvalue ..., 0`.
    // The PTX CAS already returns the old scalar value, so replace the extract
    // result and remove both aggregate-producing operations.
    let extracted = rewrite.extract.deref(ctx).get_result(0);
    extracted.replace_all_uses_with(ctx, &old_value);
    rewrite.extract.unlink(ctx);
    rewrite.op.unlink(ctx);
    Ok(())
}

/// Validate the exact semantics and declaration ABI needed by the legacy NVVM
/// floating-point atomic-add intrinsic. This runs for the complete module
/// before any rewrite so an error cannot leave earlier operations changed.
fn validate_float_atomic_add(
    ctx: &Context,
    op: Ptr<Operation>,
    capability: u32,
) -> Result<Option<String>> {
    let Some(rmw) = Operation::get_op::<llvm::AtomicRmwOp>(op, ctx) else {
        return Ok(None);
    };
    if rmw.get_attr_llvm_rmw_kind(ctx).as_deref() != Some(&AtomicRmwKindAttr::FAdd) {
        return Ok(None);
    }

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let (ptr, value) = (operands[0], operands[1]);
    let width = float_width(ctx, value.get_type(ctx)).ok_or_else(|| {
        pliron::input_error!(
            op.deref(ctx).loc(),
            "floating-point atomic add requires an f32 or f64 value"
        )
    })?;
    if !matches!(width, 32 | 64) {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "legacy NVVM floating-point atomic add supports only f32 and f64"
        );
    }
    // PTX `atom.add.f64` needs sm_60 (Pascal) hardware; reject here so the
    // failure is a clear cuda-oxide diagnostic before any rewrite, not a
    // downstream libNVVM or ptxas error on the emitted intrinsic.
    if width == 64 && capability < 60 {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "legacy NVVM f64 atomic add requires sm_60 or newer; the build targets sm_{capability}"
        );
    }
    if rmw.get_attr_llvm_rmw_ordering(ctx).as_deref() != Some(&AtomicOrderingAttr::Monotonic) {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "legacy NVVM floating-point atomic add requires monotonic ordering"
        );
    }
    let syncscope = rmw.syncscope(ctx);
    if syncscope.to_name() != "device" {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "legacy NVVM floating-point atomic add requires device synchronization scope"
        );
    }

    let address_space = ptr
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<llvm_types::PointerType>()
        .map(llvm_types::PointerType::address_space)
        .ok_or_else(|| {
            pliron::input_error!(
                op.deref(ctx).loc(),
                "floating-point atomic add requires a pointer operand"
            )
        })?;
    if !matches!(address_space, 0 | 1 | 3) {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "legacy NVVM floating-point atomic add does not support address space {address_space}"
        );
    }

    let intrinsic = format!("llvm_nvvm_atomic_load_add_f{width}_p{address_space}f{width}");
    let parent_block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(op.deref(ctx).loc(), "atomic RMW has no parent block")
    })?;
    let parent_function = parent_block.deref(ctx).get_parent_op(ctx).ok_or_else(|| {
        pliron::input_error!(op.deref(ctx).loc(), "atomic RMW block has no parent")
    })?;
    let module = parent_function
        .deref(ctx)
        .get_parent_op(ctx)
        .ok_or_else(|| {
            pliron::input_error!(op.deref(ctx).loc(), "atomic RMW function has no module")
        })?;
    let module_block = module
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .iter(ctx)
        .next()
        .ok_or_else(|| pliron::input_error!(op.deref(ctx).loc(), "atomic RMW module is empty"))?;
    for existing_op in module_block.deref(ctx).iter(ctx) {
        let existing_dyn = Operation::get_op_dyn(existing_op, ctx);
        let Some(existing_symbol) = op_cast::<dyn SymbolOpInterface>(existing_dyn.as_ref()) else {
            continue;
        };
        if existing_symbol.get_symbol_name(ctx).to_string() != intrinsic {
            continue;
        }
        let Some(existing) = Operation::get_op::<llvm::FuncOp>(existing_op, ctx) else {
            return pliron::input_err!(
                op.deref(ctx).loc(),
                "legacy NVVM atomic intrinsic `{intrinsic}` conflicts with an existing non-function symbol"
            );
        };
        if existing.get_operation().deref(ctx).regions().count() != 0 {
            return pliron::input_err!(
                op.deref(ctx).loc(),
                "legacy NVVM atomic intrinsic `{intrinsic}` conflicts with an existing function definition; an intrinsic must be a declaration"
            );
        }
        let existing_ty = existing.get_type(ctx);
        let existing_ref = existing_ty.deref(ctx);
        let expected_args = [ptr.get_type(ctx), value.get_type(ctx)];
        if existing_ref.result_type() != value.get_type(ctx)
            || existing_ref.arg_types() != expected_args
            || existing_ref.is_var_arg()
        {
            return pliron::input_err!(
                op.deref(ctx).loc(),
                "legacy NVVM atomic intrinsic `{intrinsic}` has an incompatible existing declaration"
            );
        }
    }

    Ok(Some(intrinsic))
}

/// LLVM 7 predates floating-point `atomicrmw`, while NVVM provides atomic-add
/// intrinsics for generic, global, and shared pointers. All fallible semantic
/// validation is completed by [`validate_float_atomic_add`] before this runs.
fn rewrite_float_atomic_add(ctx: &mut Context, op: Ptr<Operation>, intrinsic: &str) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let (ptr, value) = (operands[0], operands[1]);
    let func_ty = llvm_types::FuncType::get(
        ctx,
        value.get_type(ctx),
        vec![ptr.get_type(ctx), value.get_type(ctx)],
        false,
    );
    let parent_block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(op.deref(ctx).loc(), "atomic RMW has no parent block")
    })?;
    helpers::ensure_intrinsic_declared(ctx, parent_block, intrinsic, func_ty)
        .map_err(|error| pliron::input_error!(op.deref(ctx).loc(), "{error}"))?;
    let call = llvm::CallOp::new(
        ctx,
        CallOpCallable::Direct(intrinsic.try_into().map_err(|error| {
            pliron::input_error!(
                op.deref(ctx).loc(),
                "invalid atomic intrinsic name: {error}"
            )
        })?),
        func_ty,
        vec![ptr, value],
    );
    let result = insert_before(ctx, op, call.get_operation());
    replace_one_result(ctx, op, result)
}

fn validate_rewrite_candidate(ctx: &Context, op: Ptr<Operation>) -> Result<()> {
    if Operation::get_op::<llvm::FNegOp>(op, ctx).is_some() {
        let ty = op.deref(ctx).get_operand(0).get_type(ctx);
        if !matches!(float_width(ctx, ty), Some(32) | Some(64)) {
            return pliron::input_err!(
                op.deref(ctx).loc(),
                "legacy NVVM fneg legalization supports only scalar f32 and f64; scalar f16 is not portable to the supported CUDA 12 legacy dialect"
            );
        }
    }

    let Some(call) = Operation::get_op::<llvm::CallOp>(op, ctx) else {
        return Ok(());
    };
    let CallOpCallable::Direct(callee) = call.callee(ctx) else {
        return Ok(());
    };
    let name = callee.to_string();
    if parse_integer_sat_name(&name).is_some() {
        validate_integer_sat_call(ctx, op, &name)?;
    } else if parse_float_to_int_sat_name(&name).is_some() {
        validate_float_to_int_sat_call(ctx, op, &name)?;
    } else if legacy_bit_rewrite(&name).is_some() {
        validate_legacy_bit_call(ctx, op, &name)?;
    } else if shuffle_mode(&name).is_some() {
        validate_shuffle_call(ctx, op, &name)?;
    } else if vote_mode(&name).is_some() {
        validate_vote_call(ctx, op, &name)?;
    }
    Ok(())
}

fn remove_nneg(ctx: &mut Context, op: Ptr<Operation>) {
    let key: Identifier = NNEG_ATTR
        .try_into()
        .expect("static attribute name is valid");
    // `ZExtOp` and `UIToFPOp` implement pliron-LLVM's NNegFlag interface,
    // whose verifier requires the attribute to exist even when it is false.
    // LLVM 7 cannot parse the textual `nneg` keyword, so clear the semantic
    // bit while retaining the dialect invariant; the exporter omits false.
    if op.deref(ctx).attributes.0.contains_key(&key) {
        op.deref_mut(ctx).attributes.set(key, BoolAttr::new(false));
    }
}

fn insert_before(ctx: &mut Context, anchor: Ptr<Operation>, op: Ptr<Operation>) -> Value {
    op.insert_before(ctx, anchor);
    op.deref(ctx).get_result(0)
}

fn replace_one_result(ctx: &mut Context, old_op: Ptr<Operation>, replacement: Value) -> Result<()> {
    if old_op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(
            old_op.deref(ctx).loc(),
            "legacy legalization expected exactly one result"
        );
    }
    let old = old_op.deref(ctx).get_result(0);
    if old.get_type(ctx) != replacement.get_type(ctx) {
        return pliron::input_err!(
            old_op.deref(ctx).loc(),
            "legacy legalization produced a replacement with a different type"
        );
    }
    old.replace_all_uses_with(ctx, &replacement);
    old_op.unlink(ctx);
    Ok(())
}

fn integer_const(ctx: &mut Context, anchor: Ptr<Operation>, width: u32, bits: u128) -> Value {
    let ty = IntegerType::get(ctx, width, Signedness::Signless);
    let attr = IntegerAttr::new(
        ty,
        APInt::from_u128(
            bits,
            NonZeroUsize::new(width as usize).expect("integer width is non-zero"),
        ),
    );
    let op = llvm::ConstantOp::new(ctx, Box::new(attr)).get_operation();
    insert_before(ctx, anchor, op)
}

fn bool_binop<T: Op + BinArithOp>(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    lhs: Value,
    rhs: Value,
) -> Value {
    let op = T::new(ctx, lhs, rhs).get_operation();
    insert_before(ctx, anchor, op)
}

fn shl(ctx: &mut Context, anchor: Ptr<Operation>, lhs: Value, rhs: Value) -> Value {
    let op = llvm::ShlOp::new(ctx, lhs, rhs).get_operation();
    op.deref_mut(ctx).attributes.set(
        llvm_export::op_interfaces::ATTR_KEY_INTEGER_OVERFLOW_FLAGS.clone(),
        IntegerOverflowFlagsAttr::default(),
    );
    insert_before(ctx, anchor, op)
}

fn icmp(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    predicate: ICmpPredicateAttr,
    lhs: Value,
    rhs: Value,
) -> Value {
    let op = llvm::ICmpOp::new(ctx, predicate, lhs, rhs).get_operation();
    insert_before(ctx, anchor, op)
}

fn select(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    condition: Value,
    if_true: Value,
    if_false: Value,
) -> Value {
    let op = llvm::SelectOp::new(ctx, condition, if_true, if_false).get_operation();
    insert_before(ctx, anchor, op)
}

fn rewrite_fneg(ctx: &mut Context, op: Ptr<Operation>) -> Result<()> {
    let value = op.deref(ctx).get_operand(0);
    let float_ty = value.get_type(ctx);
    let width = float_width(ctx, float_ty).ok_or_else(|| {
        pliron::input_error!(
            op.deref(ctx).loc(),
            "legacy fneg legalization supports only scalar f32 and f64"
        )
    })?;
    let int_ty: TypeHandle = IntegerType::get(ctx, width, Signedness::Signless).into();

    // IEEE negation is exactly a sign-bit toggle, including for NaNs and
    // signed zero.  This is stronger than `0.0 - x`, which changes NaNs and
    // mishandles +0.0.
    let as_int_op = llvm::BitcastOp::new(ctx, value, int_ty).get_operation();
    let as_int = insert_before(ctx, op, as_int_op);
    let sign_mask = integer_const(ctx, op, width, 1_u128 << (width - 1));
    let toggled_op = llvm::XorOp::new(ctx, as_int, sign_mask).get_operation();
    let toggled = insert_before(ctx, op, toggled_op);
    let result_op = llvm::BitcastOp::new(ctx, toggled, float_ty).get_operation();
    let result = insert_before(ctx, op, result_op);
    replace_one_result(ctx, op, result)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IntegerSatKind {
    SignedAdd,
    SignedSub,
    UnsignedAdd,
    UnsignedSub,
}

fn parse_integer_sat_name(name: &str) -> Option<(IntegerSatKind, u32)> {
    let (kind, width) = if let Some(rest) = name.strip_prefix("llvm_sadd_sat_i") {
        (IntegerSatKind::SignedAdd, rest)
    } else if let Some(rest) = name.strip_prefix("llvm_ssub_sat_i") {
        (IntegerSatKind::SignedSub, rest)
    } else if let Some(rest) = name.strip_prefix("llvm_uadd_sat_i") {
        (IntegerSatKind::UnsignedAdd, rest)
    } else {
        let rest = name.strip_prefix("llvm_usub_sat_i")?;
        (IntegerSatKind::UnsignedSub, rest)
    };
    let width = width.parse().ok()?;
    (width > 0 && width <= 128).then_some((kind, width))
}

fn validate_integer_sat_call(ctx: &Context, op: Ptr<Operation>, name: &str) -> Result<()> {
    let (_, expected_width) = parse_integer_sat_name(name).expect("validated intrinsic name");
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 || op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} must have two operands and one result"
        );
    }
    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let width = scalar_integer_width(ctx, result_ty);
    if width != Some(expected_width) || operands.iter().any(|v| v.get_type(ctx) != result_ty) {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} has a signature inconsistent with its overload suffix"
        );
    }
    Ok(())
}

fn rewrite_integer_sat(ctx: &mut Context, op: Ptr<Operation>, name: &str) -> Result<()> {
    let (kind, width) = parse_integer_sat_name(name).expect("validated intrinsic name");
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let (lhs, rhs) = (operands[0], operands[1]);
    let flags = IntegerOverflowFlagsAttr::default();
    let arithmetic = match kind {
        IntegerSatKind::SignedAdd | IntegerSatKind::UnsignedAdd => {
            llvm::AddOp::new_with_overflow_flag(ctx, lhs, rhs, flags).get_operation()
        }
        IntegerSatKind::SignedSub | IntegerSatKind::UnsignedSub => {
            llvm::SubOp::new_with_overflow_flag(ctx, lhs, rhs, flags).get_operation()
        }
    };
    let value = insert_before(ctx, op, arithmetic);

    let result = match kind {
        IntegerSatKind::UnsignedAdd => {
            let overflow = icmp(ctx, op, ICmpPredicateAttr::ULT, value, lhs);
            let max = integer_const(ctx, op, width, mask_for_width(width));
            select(ctx, op, overflow, max, value)
        }
        IntegerSatKind::UnsignedSub => {
            let underflow = icmp(ctx, op, ICmpPredicateAttr::ULT, lhs, rhs);
            let zero = integer_const(ctx, op, width, 0);
            select(ctx, op, underflow, zero, value)
        }
        IntegerSatKind::SignedAdd | IntegerSatKind::SignedSub => {
            let zero = integer_const(ctx, op, width, 0);
            let lhs_negative = icmp(ctx, op, ICmpPredicateAttr::SLT, lhs, zero);
            let rhs_negative = icmp(ctx, op, ICmpPredicateAttr::SLT, rhs, zero);
            let result_negative = icmp(ctx, op, ICmpPredicateAttr::SLT, value, zero);
            let lhs_rhs_relation = icmp(
                ctx,
                op,
                if kind == IntegerSatKind::SignedAdd {
                    ICmpPredicateAttr::EQ
                } else {
                    ICmpPredicateAttr::NE
                },
                lhs_negative,
                rhs_negative,
            );
            let sign_changed = icmp(
                ctx,
                op,
                ICmpPredicateAttr::NE,
                lhs_negative,
                result_negative,
            );
            let overflow = bool_binop::<llvm::AndOp>(ctx, op, lhs_rhs_relation, sign_changed);
            let min = integer_const(ctx, op, width, 1_u128 << (width - 1));
            let max = integer_const(ctx, op, width, (1_u128 << (width - 1)) - 1);
            let clamp = select(ctx, op, lhs_negative, min, max);
            select(ctx, op, overflow, clamp, value)
        }
    };
    replace_one_result(ctx, op, result)
}

fn mask_for_width(width: u32) -> u128 {
    if width == 128 {
        u128::MAX
    } else {
        (1_u128 << width) - 1
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LegacyBitRewrite {
    InvalidBswapI8,
    FunnelLeftI128,
    FunnelRightI128,
    BswapI128,
    BitreverseI128,
    CtpopI128,
    CtlzI128,
    CttzI128,
}

/// NVVM only accepts bit-manipulation intrinsics through i64. Rust exposes the
/// same operations for i128, so split them into exact i64 operations before
/// handing the module to libNVVM. `llvm.bswap.i8` is not valid LLVM IR at all;
/// recognize it only so malformed input fails during validation rather than in
/// the textual LLVM verifier. Rust's `u8::swap_bytes` is lowered to an identity
/// before this pass.
fn legacy_bit_rewrite(name: &str) -> Option<LegacyBitRewrite> {
    match name {
        "llvm_bswap_i8" => Some(LegacyBitRewrite::InvalidBswapI8),
        "llvm_fshl_i128" => Some(LegacyBitRewrite::FunnelLeftI128),
        "llvm_fshr_i128" => Some(LegacyBitRewrite::FunnelRightI128),
        "llvm_bswap_i128" => Some(LegacyBitRewrite::BswapI128),
        "llvm_bitreverse_i128" => Some(LegacyBitRewrite::BitreverseI128),
        "llvm_ctpop_i128" => Some(LegacyBitRewrite::CtpopI128),
        "llvm_ctlz_i128" => Some(LegacyBitRewrite::CtlzI128),
        "llvm_cttz_i128" => Some(LegacyBitRewrite::CttzI128),
        _ => None,
    }
}

fn validate_legacy_bit_call(ctx: &Context, op: Ptr<Operation>, name: &str) -> Result<()> {
    let kind = legacy_bit_rewrite(name).expect("validated legacy bit intrinsic name");
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} must produce exactly one result"
        );
    }
    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let i1_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
    let expected_width = if kind == LegacyBitRewrite::InvalidBswapI8 {
        8
    } else {
        128
    };
    if scalar_integer_width(ctx, result_ty) != Some(expected_width) {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} result type does not match its overload suffix"
        );
    }

    let valid = match kind {
        LegacyBitRewrite::FunnelLeftI128 | LegacyBitRewrite::FunnelRightI128 => {
            operands.len() == 3
                && operands
                    .iter()
                    .all(|value| value.get_type(ctx) == result_ty)
        }
        LegacyBitRewrite::CtlzI128 | LegacyBitRewrite::CttzI128 => {
            operands.len() == 2
                && operands[0].get_type(ctx) == result_ty
                && operands[1].get_type(ctx) == i1_ty
        }
        _ => operands.len() == 1 && operands[0].get_type(ctx) == result_ty,
    };
    if !valid {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} has a signature inconsistent with its overload suffix"
        );
    }

    if kind == LegacyBitRewrite::InvalidBswapI8 {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} is not a valid LLVM intrinsic: byte-swap widths must be a multiple of 16 bits"
        );
    }

    if matches!(
        kind,
        LegacyBitRewrite::CtlzI128 | LegacyBitRewrite::CttzI128
    ) && literal_i1_constant(ctx, operands[1]).is_none()
    {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} is_zero_poison flag must be an immediate i1 constant (0 or 1)"
        );
    }
    Ok(())
}

fn literal_i1_constant(ctx: &Context, value: Value) -> Option<bool> {
    let defining_op = value.defining_op()?;
    let constant = Operation::get_op::<llvm::ConstantOp>(defining_op, ctx)?;
    let attr = constant.get_value(ctx);
    let integer = (&*attr as &dyn Attribute).downcast_ref::<IntegerAttr>()?;
    match integer.value().to_u64() {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    }
}

fn split_i128(ctx: &mut Context, anchor: Ptr<Operation>, value: Value) -> (Value, Value) {
    let i64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Signless).into();
    let low_op = llvm::TruncOp::new(ctx, value, i64_ty).get_operation();
    let low = insert_before(ctx, anchor, low_op);
    let shift = integer_const(ctx, anchor, 128, 64);
    let shifted_op = llvm::LShrOp::new(ctx, value, shift).get_operation();
    let shifted = insert_before(ctx, anchor, shifted_op);
    let high_op = llvm::TruncOp::new(ctx, shifted, i64_ty).get_operation();
    let high = insert_before(ctx, anchor, high_op);
    (low, high)
}

fn zext_to_i128(ctx: &mut Context, anchor: Ptr<Operation>, value: Value) -> Value {
    let i128_ty: TypeHandle = IntegerType::get(ctx, 128, Signedness::Signless).into();
    let extend = llvm::ZExtOp::new_with_nneg(ctx, value, i128_ty, false).get_operation();
    insert_before(ctx, anchor, extend)
}

fn call_i64_intrinsic(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    name: &str,
    args: Vec<Value>,
) -> Result<Value> {
    let i64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Signless).into();
    let param_types = args.iter().map(|value| value.get_type(ctx)).collect();
    let func_ty = llvm_types::FuncType::get(ctx, i64_ty, param_types, false);
    let parent_block = anchor.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(
            anchor.deref(ctx).loc(),
            "intrinsic call has no parent block"
        )
    })?;
    helpers::ensure_intrinsic_declared(ctx, parent_block, name, func_ty)
        .map_err(|error| pliron::input_error!(anchor.deref(ctx).loc(), "{error}"))?;
    let call = llvm::CallOp::new(
        ctx,
        CallOpCallable::Direct(name.try_into().map_err(|error| {
            pliron::input_error!(anchor.deref(ctx).loc(), "invalid intrinsic name: {error}")
        })?),
        func_ty,
        args,
    );
    Ok(insert_before(ctx, anchor, call.get_operation()))
}

fn combine_reversed_i64_halves(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    low: Value,
    high: Value,
) -> Value {
    // Reversing bytes or bits across the complete i128 swaps the transformed
    // halves: transformed(low) becomes the high half of the result.
    let low = zext_to_i128(ctx, anchor, low);
    let high = zext_to_i128(ctx, anchor, high);
    let shift = integer_const(ctx, anchor, 128, 64);
    let low_as_high = shl(ctx, anchor, low, shift);
    let combined = llvm::OrOp::new(ctx, low_as_high, high).get_operation();
    insert_before(ctx, anchor, combined)
}

fn rewrite_legacy_bit_intrinsic(ctx: &mut Context, op: Ptr<Operation>, name: &str) -> Result<()> {
    let kind = legacy_bit_rewrite(name).expect("validated legacy bit intrinsic name");
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let result = match kind {
        LegacyBitRewrite::InvalidBswapI8 => {
            return pliron::input_err!(
                op.deref(ctx).loc(),
                "{name} reached rewriting despite being rejected during validation"
            );
        }
        LegacyBitRewrite::FunnelLeftI128 | LegacyBitRewrite::FunnelRightI128 => {
            let mask = integer_const(ctx, op, 128, 127);
            let shift = bool_binop::<llvm::AndOp>(ctx, op, operands[2], mask);
            let zero = integer_const(ctx, op, 128, 0);
            let negative_shift_op = llvm::SubOp::new_with_overflow_flag(
                ctx,
                zero,
                shift,
                IntegerOverflowFlagsAttr::default(),
            )
            .get_operation();
            let negative_shift = insert_before(ctx, op, negative_shift_op);
            let reverse_shift = bool_binop::<llvm::AndOp>(ctx, op, negative_shift, mask);
            let is_zero = icmp(ctx, op, ICmpPredicateAttr::EQ, shift, zero);
            let (left, right, zero_result) = if kind == LegacyBitRewrite::FunnelLeftI128 {
                (operands[0], operands[1], operands[0])
            } else {
                (operands[0], operands[1], operands[1])
            };
            let left_shift = if kind == LegacyBitRewrite::FunnelLeftI128 {
                shift
            } else {
                reverse_shift
            };
            let right_shift = if kind == LegacyBitRewrite::FunnelLeftI128 {
                reverse_shift
            } else {
                shift
            };
            let left = shl(ctx, op, left, left_shift);
            let right_op = llvm::LShrOp::new(ctx, right, right_shift).get_operation();
            let right = insert_before(ctx, op, right_op);
            let combined_op = llvm::OrOp::new(ctx, left, right).get_operation();
            let combined = insert_before(ctx, op, combined_op);
            select(ctx, op, is_zero, zero_result, combined)
        }
        LegacyBitRewrite::BswapI128 | LegacyBitRewrite::BitreverseI128 => {
            let (low, high) = split_i128(ctx, op, operands[0]);
            let intrinsic = if kind == LegacyBitRewrite::BswapI128 {
                "llvm_bswap_i64"
            } else {
                "llvm_bitreverse_i64"
            };
            let low = call_i64_intrinsic(ctx, op, intrinsic, vec![low])?;
            let high = call_i64_intrinsic(ctx, op, intrinsic, vec![high])?;
            combine_reversed_i64_halves(ctx, op, low, high)
        }
        LegacyBitRewrite::CtpopI128 => {
            let (low, high) = split_i128(ctx, op, operands[0]);
            let low = call_i64_intrinsic(ctx, op, "llvm_ctpop_i64", vec![low])?;
            let high = call_i64_intrinsic(ctx, op, "llvm_ctpop_i64", vec![high])?;
            let low = zext_to_i128(ctx, op, low);
            let high = zext_to_i128(ctx, op, high);
            let sum = llvm::AddOp::new_with_overflow_flag(
                ctx,
                low,
                high,
                IntegerOverflowFlagsAttr::default(),
            )
            .get_operation();
            insert_before(ctx, op, sum)
        }
        LegacyBitRewrite::CtlzI128 | LegacyBitRewrite::CttzI128 => {
            let (low, high) = split_i128(ctx, op, operands[0]);
            let flag = operands[1];
            let (primary, secondary, intrinsic) = if kind == LegacyBitRewrite::CtlzI128 {
                (high, low, "llvm_ctlz_i64")
            } else {
                (low, high, "llvm_cttz_i64")
            };
            let primary_count = call_i64_intrinsic(ctx, op, intrinsic, vec![primary, flag])?;
            let secondary_count = call_i64_intrinsic(ctx, op, intrinsic, vec![secondary, flag])?;
            let zero64 = integer_const(ctx, op, 64, 0);
            let primary_is_zero = icmp(ctx, op, ICmpPredicateAttr::EQ, primary, zero64);
            let sixty_four = integer_const(ctx, op, 64, 64);
            let extended_count = llvm::AddOp::new_with_overflow_flag(
                ctx,
                sixty_four,
                secondary_count,
                IntegerOverflowFlagsAttr::default(),
            )
            .get_operation();
            let secondary_count = insert_before(ctx, op, extended_count);
            let count = select(ctx, op, primary_is_zero, secondary_count, primary_count);
            zext_to_i128(ctx, op, count)
        }
    };
    replace_one_result(ctx, op, result)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FloatToIntSatKind {
    Signed,
    Unsigned,
}

fn parse_float_to_int_sat_name(name: &str) -> Option<(FloatToIntSatKind, u32, u32)> {
    let (kind, rest) = if let Some(rest) = name.strip_prefix("llvm_fptosi_sat_i") {
        (FloatToIntSatKind::Signed, rest)
    } else {
        let rest = name.strip_prefix("llvm_fptoui_sat_i")?;
        (FloatToIntSatKind::Unsigned, rest)
    };
    let (int_width, float_width) = rest.split_once("_f")?;
    let int_width: u32 = int_width.parse().ok()?;
    let float_width: u32 = float_width.parse().ok()?;
    (int_width > 0 && int_width <= 128 && matches!(float_width, 16 | 32 | 64)).then_some((
        kind,
        int_width,
        float_width,
    ))
}

fn validate_float_to_int_sat_call(ctx: &Context, op: Ptr<Operation>, name: &str) -> Result<()> {
    let (_, expected_int_width, expected_float_width) =
        parse_float_to_int_sat_name(name).expect("validated intrinsic name");
    if expected_float_width == 16 {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "legacy NVVM saturating conversion from f16 is not portable to the supported CUDA 12 legacy dialect"
        );
    }
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 || op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} must have one operand and one result"
        );
    }
    let result_width = scalar_integer_width(ctx, op.deref(ctx).get_result(0).get_type(ctx));
    let source_width = float_width(ctx, operands[0].get_type(ctx));
    if result_width != Some(expected_int_width) || source_width != Some(expected_float_width) {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} has a signature inconsistent with its overload suffix"
        );
    }
    Ok(())
}

fn rewrite_float_to_int_sat(ctx: &mut Context, op: Ptr<Operation>, name: &str) -> Result<()> {
    let (kind, width, source_width) =
        parse_float_to_int_sat_name(name).expect("validated intrinsic name");
    let mut source = op.deref(ctx).get_operand(0);

    // If f16 support is added later, extending to f32 preserves every f16
    // value exactly and allows the same clamp algorithm to be reused.
    let compare_width = if source_width == 16 {
        let f32_ty: TypeHandle = FP32Type::get(ctx).into();
        let extend = llvm::FPExtOp::new(ctx, source, f32_ty);
        extend.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
        source = insert_before(ctx, op, extend.get_operation());
        32
    } else {
        source_width
    };

    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let raw = match kind {
        FloatToIntSatKind::Signed => llvm::FPToSIOp::new(ctx, source, result_ty).get_operation(),
        FloatToIntSatKind::Unsigned => llvm::FPToUIOp::new(ctx, source, result_ty).get_operation(),
    };
    let raw = insert_before(ctx, op, raw);

    let (low_float, high_float) = match kind {
        FloatToIntSatKind::Signed => (
            -(2.0_f64).powi(width as i32 - 1),
            (2.0_f64).powi(width as i32 - 1),
        ),
        FloatToIntSatKind::Unsigned => (0.0, (2.0_f64).powi(width as i32)),
    };
    let low = float_const(ctx, op, compare_width, low_float)?;
    let high = float_const(ctx, op, compare_width, high_float)?;
    let is_low = fcmp(ctx, op, FCmpPredicateAttr::OLE, source, low);
    let is_high = fcmp(ctx, op, FCmpPredicateAttr::OGE, source, high);
    let is_nan = fcmp(ctx, op, FCmpPredicateAttr::UNO, source, source);

    let zero = integer_const(ctx, op, width, 0);
    let (low_value, high_value) = match kind {
        FloatToIntSatKind::Signed => (
            integer_const(ctx, op, width, 1_u128 << (width - 1)),
            integer_const(ctx, op, width, (1_u128 << (width - 1)) - 1),
        ),
        FloatToIntSatKind::Unsigned => (zero, integer_const(ctx, op, width, mask_for_width(width))),
    };
    let high_or_raw = select(ctx, op, is_high, high_value, raw);
    let low_or_bounded = select(ctx, op, is_low, low_value, high_or_raw);
    let result = select(ctx, op, is_nan, zero, low_or_bounded);
    replace_one_result(ctx, op, result)
}

fn float_const(ctx: &mut Context, anchor: Ptr<Operation>, width: u32, value: f64) -> Result<Value> {
    let attr: Box<dyn pliron::builtin::attr_interfaces::TypedAttrInterface> = match width {
        32 => Box::new(FPSingleAttr::from(value as f32)),
        64 => Box::new(FPDoubleAttr::from(value)),
        _ => {
            return pliron::input_err!(
                anchor.deref(ctx).loc(),
                "legacy float clamp constants support only f32 and f64"
            );
        }
    };
    let const_op = llvm::ConstantOp::new(ctx, attr).get_operation();
    Ok(insert_before(ctx, anchor, const_op))
}

fn fcmp(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    predicate: FCmpPredicateAttr,
    lhs: Value,
    rhs: Value,
) -> Value {
    let compare = llvm::FCmpOp::new(ctx, predicate, lhs, rhs);
    compare.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
    insert_before(ctx, anchor, compare.get_operation())
}

fn float_width(ctx: &Context, ty: TypeHandle) -> Option<u32> {
    let ty = ty.deref(ctx);
    if ty.is::<FP16Type>() {
        Some(16)
    } else if ty.is::<FP32Type>() {
        Some(32)
    } else if ty.is::<FP64Type>() {
        Some(64)
    } else {
        None
    }
}

fn scalar_integer_width(ctx: &Context, ty: TypeHandle) -> Option<u32> {
    ty.deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|t| t.width())
}

fn shuffle_mode(name: &str) -> Option<u32> {
    let stem = name
        .strip_suffix("_i32")
        .or_else(|| name.strip_suffix("_f32"))?;
    match stem {
        "llvm_nvvm_shfl_sync_idx" => Some(0),
        "llvm_nvvm_shfl_sync_up" => Some(1),
        "llvm_nvvm_shfl_sync_down" => Some(2),
        "llvm_nvvm_shfl_sync_bfly" => Some(3),
        _ => None,
    }
}

fn validate_shuffle_call(ctx: &Context, op: Ptr<Operation>, name: &str) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 4 || op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} must have four operands and one result"
        );
    }
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let value_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let expected_value_ty = if name.ends_with("_f32") {
        FP32Type::get(ctx).into()
    } else {
        i32_ty
    };
    if value_ty != expected_value_ty
        || operands[1].get_type(ctx) != expected_value_ty
        || operands[0].get_type(ctx) != i32_ty
        || operands[2].get_type(ctx) != i32_ty
        || operands[3].get_type(ctx) != i32_ty
    {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} has an invalid legacy-shuffle signature"
        );
    }
    Ok(())
}

fn rewrite_shuffle(ctx: &mut Context, op: Ptr<Operation>, name: &str) -> Result<()> {
    let mode = shuffle_mode(name).expect("validated shuffle name");
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let i1_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
    let struct_ty = llvm_types::StructType::get_unnamed(
        ctx,
        (vec![i32_ty, i1_ty], llvm_types::StructLayout::Unpacked),
    );
    let func_ty = llvm_types::FuncType::get(
        ctx,
        struct_ty.into(),
        vec![i32_ty, i32_ty, i32_ty, i32_ty, i32_ty],
        false,
    );
    let parent_block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(op.deref(ctx).loc(), "shuffle call has no parent block")
    })?;
    helpers::ensure_intrinsic_declared(ctx, parent_block, "llvm_nvvm_shfl_sync_i32", func_ty)
        .map_err(|e| pliron::input_error!(op.deref(ctx).loc(), "{e}"))?;

    let mut value = operands[1];
    if name.ends_with("_f32") {
        let cast_op = llvm::BitcastOp::new(ctx, value, i32_ty).get_operation();
        value = insert_before(ctx, op, cast_op);
    }
    let mode = integer_const(ctx, op, 32, mode as u128);
    let callee = CallOpCallable::Direct(
        "llvm_nvvm_shfl_sync_i32"
            .try_into()
            .expect("static intrinsic name is valid"),
    );
    let call = llvm::CallOp::new(
        ctx,
        callee,
        func_ty,
        vec![operands[0], mode, value, operands[2], operands[3]],
    );
    let aggregate = insert_before(ctx, op, call.get_operation());
    let extracted = llvm::ExtractValueOp::new(ctx, aggregate, vec![0])
        .map_err(|e| pliron::input_error!(op.deref(ctx).loc(), "{e}"))?;
    let mut result = insert_before(ctx, op, extracted.get_operation());
    if name.ends_with("_f32") {
        let f32_ty: TypeHandle = FP32Type::get(ctx).into();
        let cast_op = llvm::BitcastOp::new(ctx, result, f32_ty).get_operation();
        result = insert_before(ctx, op, cast_op);
    }
    replace_one_result(ctx, op, result)
}

fn vote_mode(name: &str) -> Option<(u32, u32)> {
    match name {
        "llvm_nvvm_vote_all_sync" => Some((0, 1)),
        "llvm_nvvm_vote_any_sync" => Some((1, 1)),
        "llvm_nvvm_vote_ballot_sync" => Some((3, 0)),
        _ => None,
    }
}

fn validate_vote_call(ctx: &Context, op: Ptr<Operation>, name: &str) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 || op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} must have two operands and one result"
        );
    }
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let i1_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
    let expected_result = if name.ends_with("ballot_sync") {
        i32_ty
    } else {
        i1_ty
    };
    if operands[0].get_type(ctx) != i32_ty
        || operands[1].get_type(ctx) != i1_ty
        || op.deref(ctx).get_result(0).get_type(ctx) != expected_result
    {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} has an invalid legacy-vote signature"
        );
    }
    Ok(())
}

fn rewrite_vote(ctx: &mut Context, op: Ptr<Operation>, name: &str) -> Result<()> {
    let (mode, field) = vote_mode(name).expect("validated vote name");
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let i1_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
    let struct_ty = llvm_types::StructType::get_unnamed(
        ctx,
        (vec![i32_ty, i1_ty], llvm_types::StructLayout::Unpacked),
    );
    let func_ty =
        llvm_types::FuncType::get(ctx, struct_ty.into(), vec![i32_ty, i32_ty, i1_ty], false);
    let parent_block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(op.deref(ctx).loc(), "vote call has no parent block")
    })?;
    helpers::ensure_intrinsic_declared(ctx, parent_block, "llvm_nvvm_vote_sync", func_ty)
        .map_err(|e| pliron::input_error!(op.deref(ctx).loc(), "{e}"))?;
    let mode = integer_const(ctx, op, 32, mode as u128);
    let call = llvm::CallOp::new(
        ctx,
        CallOpCallable::Direct(
            "llvm_nvvm_vote_sync"
                .try_into()
                .expect("static intrinsic name is valid"),
        ),
        func_ty,
        vec![operands[0], mode, operands[1]],
    );
    let aggregate = insert_before(ctx, op, call.get_operation());
    let extract = llvm::ExtractValueOp::new(ctx, aggregate, vec![field])
        .map_err(|e| pliron::input_error!(op.deref(ctx).loc(), "{e}"))?;
    let result = insert_before(ctx, op, extract.get_operation());
    replace_one_result(ctx, op, result)
}

fn legacy_memory_intrinsic_name(name: &str) -> Option<String> {
    for stem in ["memcpy", "memmove"] {
        if let Some(rest) = name.strip_prefix(&format!("llvm_{stem}_p")) {
            // Modern opaque-pointer spelling:
            // llvm.{memcpy,memmove}.p<dst>.p<src>.i<bits>
            let (dst, rest) = rest.split_once("_p")?;
            let (src, bits) = rest.split_once("_i")?;
            dst.parse::<u32>().ok()?;
            src.parse::<u32>().ok()?;
            bits.parse::<u32>().ok()?;
            return Some(format!("llvm_{stem}_p{dst}i8_p{src}i8_i{bits}"));
        }
    }

    if let Some(rest) = name.strip_prefix("llvm_memset_p") {
        // Modern opaque-pointer spelling: llvm.memset.p<dst>.i<bits>
        let (dst, bits) = rest.split_once("_i")?;
        dst.parse::<u32>().ok()?;
        bits.parse::<u32>().ok()?;
        return Some(format!("llvm_memset_p{dst}i8_i{bits}"));
    }

    None
}

fn rewrite_renamed_call(
    ctx: &mut Context,
    op: Ptr<Operation>,
    replacement_name: &str,
) -> Result<()> {
    let call = Operation::get_op::<llvm::CallOp>(op, ctx).expect("validated call op");
    let args = call.args(ctx);
    let func_ty = call.callee_type(ctx);
    let func_ty = pliron::r#type::TypedHandle::from_handle(func_ty, ctx).map_err(|e| {
        pliron::input_error!(
            op.deref(ctx).loc(),
            "memory intrinsic call has invalid function type: {e}"
        )
    })?;
    let parent_block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(
            op.deref(ctx).loc(),
            "memory intrinsic call has no parent block"
        )
    })?;
    helpers::ensure_intrinsic_declared(ctx, parent_block, replacement_name, func_ty)
        .map_err(|e| pliron::input_error!(op.deref(ctx).loc(), "{e}"))?;
    let replacement = llvm::CallOp::new(
        ctx,
        CallOpCallable::Direct(replacement_name.try_into().map_err(|e| {
            pliron::input_error!(
                op.deref(ctx).loc(),
                "invalid legacy memory intrinsic name: {e}"
            )
        })?),
        func_ty,
        args,
    );
    let result = insert_before(ctx, op, replacement.get_operation());
    replace_one_result(ctx, op, result)
}

fn remove_obsolete_declarations(
    ctx: &mut Context,
    module: Ptr<Operation>,
    obsolete_names: &[String],
) {
    if obsolete_names.is_empty() {
        return;
    }
    let Some(region) = module.deref(ctx).regions().next() else {
        return;
    };
    let Some(block) = region.deref(ctx).iter(ctx).next() else {
        return;
    };
    let top_level: Vec<_> = block.deref(ctx).iter(ctx).collect();
    for op in top_level {
        let Some(func) = Operation::get_op::<llvm::FuncOp>(op, ctx) else {
            continue;
        };
        let name = func.get_symbol_name(ctx).to_string();
        if obsolete_names.iter().any(|old| old == &name) {
            op.unlink(ctx);
        }
    }
}

fn verify_legacy_subset(ctx: &Context, module: Ptr<Operation>, capability: u32) -> Result<()> {
    let mut ops = Vec::new();
    collect_ops(ctx, module, &mut ops);
    for op in ops {
        reject_unsupported_op(ctx, op)?;
        if validate_scalar_atomic_rmw(ctx, op, capability)?.is_some() {
            return pliron::input_err!(
                op.deref(ctx).loc(),
                "legacy NVVM legalization left a scoped integer atomic RMW behind"
            );
        }
        if validate_scalar_cmpxchg(ctx, op, capability)?.is_some() {
            return pliron::input_err!(
                op.deref(ctx).loc(),
                "legacy NVVM legalization left a compare-exchange requiring PTX rewrite behind"
            );
        }
        if Operation::get_op::<llvm::FNegOp>(op, ctx).is_some() {
            return pliron::input_err!(
                op.deref(ctx).loc(),
                "legacy NVVM legalization left an llvm.fneg behind"
            );
        }
        let nneg: Identifier = NNEG_ATTR
            .try_into()
            .expect("static attribute name is valid");
        let has_true_nneg = op
            .deref(ctx)
            .attributes
            .get::<BoolAttr>(&nneg)
            .is_some_and(|flag| bool::from(flag.clone()));
        if has_true_nneg {
            return pliron::input_err!(
                op.deref(ctx).loc(),
                "legacy NVVM legalization left a true LLVM nneg flag behind"
            );
        }
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(ctx) else {
            continue;
        };
        let name = callee.to_string();
        if parse_integer_sat_name(&name).is_some()
            || parse_float_to_int_sat_name(&name).is_some()
            || legacy_bit_rewrite(&name).is_some()
            || shuffle_mode(&name).is_some()
            || vote_mode(&name).is_some()
            || legacy_memory_intrinsic_name(&name).is_some()
        {
            return pliron::input_err!(
                op.deref(ctx).loc(),
                "legacy NVVM legalization left unsupported call @{name} behind"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvm_export::{
        export::{NvvmExportConfig, NvvmIrDialect, export_module_to_string_with_config},
        op_interfaces::{CastOpWithNNegInterface, NNegFlag},
        types::{FuncType, PointerType, VoidType},
    };
    use pliron::{
        basic_block::BasicBlock,
        builtin::{op_interfaces::SymbolOpInterface, ops::ModuleOp},
        common_traits::Verify,
        printable::Printable,
    };

    fn named_scope(name: &str) -> SyncScopeAttr {
        SyncScopeAttr::NamedScope(pliron::builtin::attributes::StringAttr::new(
            name.to_string(),
        ))
    }

    fn module_block(ctx: &Context, module: &ModuleOp) -> Ptr<BasicBlock> {
        module
            .get_operation()
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .iter(ctx)
            .next()
            .expect("module has an entry block")
    }

    fn function(
        ctx: &mut Context,
        module: &ModuleOp,
        name: &str,
        result: TypeHandle,
        params: Vec<TypeHandle>,
    ) -> (llvm::FuncOp, Ptr<BasicBlock>) {
        let ty = FuncType::get(ctx, result, params, false);
        let func = llvm::FuncOp::new(ctx, name.try_into().unwrap(), ty);
        let entry = func.get_or_create_entry_block(ctx);
        func.get_operation()
            .insert_at_back(module_block(ctx, module), ctx);
        (func, entry)
    }

    fn operation_ids(ctx: &Context, root: Ptr<Operation>) -> Vec<String> {
        let mut ops = Vec::new();
        collect_ops(ctx, root, &mut ops);
        ops.into_iter()
            .map(|op| Operation::get_opid(op, ctx).to_string())
            .collect()
    }

    #[test]
    fn fneg_is_an_exact_sign_bit_toggle_and_module_still_verifies() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "fneg".try_into().unwrap());
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let (_func, entry) = function(&mut ctx, &module, "neg", f32_ty, vec![f32_ty]);
        let input = entry.deref(&ctx).get_argument(0);
        let neg =
            llvm::FNegOp::new_with_fast_math_flags(&mut ctx, input, FastmathFlagsAttr::default());
        neg.get_operation().insert_at_back(entry, &ctx);
        let neg_result = neg.get_operation().deref(&ctx).get_result(0);
        llvm::ReturnOp::new(&mut ctx, Some(neg_result))
            .get_operation()
            .insert_at_back(entry, &ctx);

        crate::legalize_for_nvvm(
            &mut ctx,
            module.get_operation(),
            llvm_export::export::NvvmIrDialect::LegacyLlvm7,
            90,
        )
        .unwrap();

        let ids = operation_ids(&ctx, module.get_operation());
        assert!(!ids.iter().any(|id| id == "llvm.fneg"), "{ids:?}");
        assert_eq!(ids.iter().filter(|id| *id == "llvm.bitcast").count(), 2);
        assert_eq!(ids.iter().filter(|id| *id == "llvm.xor").count(), 1);
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();
    }

    #[test]
    fn nneg_is_cleared_but_required_dialect_attribute_is_retained() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "nneg".try_into().unwrap());
        let i8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Signless).into();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let (_func, entry) = function(&mut ctx, &module, "extend", i32_ty, vec![i8_ty]);
        let input = entry.deref(&ctx).get_argument(0);
        let zext = llvm::ZExtOp::new_with_nneg(&mut ctx, input, i32_ty, true);
        zext.get_operation().insert_at_back(entry, &ctx);
        let result = zext.get_operation().deref(&ctx).get_result(0);
        llvm::ReturnOp::new(&mut ctx, Some(result))
            .get_operation()
            .insert_at_back(entry, &ctx);

        legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap();

        assert!(!zext.nneg(&ctx));
        let key: Identifier = NNEG_ATTR.try_into().unwrap();
        assert!(
            zext.get_operation()
                .deref(&ctx)
                .attributes
                .get::<BoolAttr>(&key)
                .is_some(),
            "pliron-LLVM requires the false nneg attribute to remain"
        );
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();
    }

    fn direct_intrinsic_call(
        ctx: &mut Context,
        module: &ModuleOp,
        name: &str,
        result_ty: TypeHandle,
        param_tys: Vec<TypeHandle>,
    ) -> (llvm::CallOp, Ptr<BasicBlock>) {
        let void_ty: TypeHandle = VoidType::get(ctx).into();
        let (_func, entry) = function(ctx, module, "caller", void_ty, param_tys.clone());
        let func_ty = FuncType::get(ctx, result_ty, param_tys, false);
        helpers::ensure_intrinsic_declared(ctx, entry, name, func_ty).unwrap();
        let args = entry.deref(ctx).arguments().collect();
        let call = llvm::CallOp::new(
            ctx,
            CallOpCallable::Direct(name.try_into().unwrap()),
            func_ty,
            args,
        );
        call.get_operation().insert_at_back(entry, ctx);
        llvm::ReturnOp::new(ctx, None)
            .get_operation()
            .insert_at_back(entry, ctx);
        (call, entry)
    }

    #[test]
    fn integer_saturation_intrinsics_are_fully_expanded() {
        for name in [
            "llvm_sadd_sat_i32",
            "llvm_ssub_sat_i32",
            "llvm_uadd_sat_i32",
            "llvm_usub_sat_i32",
        ] {
            let mut ctx = Context::new();
            let module = ModuleOp::new(&mut ctx, name.try_into().unwrap());
            let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
            direct_intrinsic_call(&mut ctx, &module, name, i32_ty, vec![i32_ty, i32_ty]);

            legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap();

            let ids = operation_ids(&ctx, module.get_operation());
            assert!(ids.iter().any(|id| id == "llvm.select"), "{name}: {ids:?}");
            assert!(ids.iter().any(|id| id == "llvm.icmp"), "{name}: {ids:?}");
            let old_decl_remains = module_block(&ctx, &module)
                .deref(&ctx)
                .iter(&ctx)
                .any(|op| {
                    Operation::get_op::<llvm::FuncOp>(op, &ctx)
                        .is_some_and(|f| f.get_symbol_name(&ctx).to_string() == name)
                });
            assert!(!old_decl_remains, "obsolete declaration @{name} remains");
            module.get_operation().deref(&ctx).verify(&ctx).unwrap();
        }
    }

    #[test]
    fn unsupported_i128_bit_intrinsics_expand_to_legacy_i64_operations() {
        for (name, parameter_count, helper) in [
            ("llvm_fshl_i128", 3, None),
            ("llvm_fshr_i128", 3, None),
            ("llvm_bswap_i128", 1, Some("llvm_bswap_i64")),
            ("llvm_bitreverse_i128", 1, Some("llvm_bitreverse_i64")),
            ("llvm_ctpop_i128", 1, Some("llvm_ctpop_i64")),
        ] {
            let mut ctx = Context::new();
            let module = ModuleOp::new(&mut ctx, name.try_into().unwrap());
            let i128_ty: TypeHandle = IntegerType::get(&ctx, 128, Signedness::Signless).into();
            direct_intrinsic_call(
                &mut ctx,
                &module,
                name,
                i128_ty,
                vec![i128_ty; parameter_count],
            );

            legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap();

            let mut ops = Vec::new();
            collect_ops(&ctx, module.get_operation(), &mut ops);
            let calls: Vec<_> = ops
                .into_iter()
                .filter_map(|op| Operation::get_op::<llvm::CallOp>(op, &ctx))
                .filter_map(|call| match call.callee(&ctx) {
                    CallOpCallable::Direct(name) => Some(name.to_string()),
                    CallOpCallable::Indirect(_) => None,
                })
                .collect();
            assert!(!calls.iter().any(|callee| callee == name), "{calls:?}");
            if let Some(helper) = helper {
                assert!(calls.iter().any(|callee| callee == helper), "{calls:?}");
            }
            module.get_operation().deref(&ctx).verify(&ctx).unwrap();
        }

        for (name, helper) in [
            ("llvm_ctlz_i128", "llvm_ctlz_i64"),
            ("llvm_cttz_i128", "llvm_cttz_i64"),
        ] {
            for zero_poison in [false, true] {
                let mut ctx = Context::new();
                let module = ModuleOp::new(&mut ctx, name.try_into().unwrap());
                let i128_ty: TypeHandle = IntegerType::get(&ctx, 128, Signedness::Signless).into();
                let i1_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
                let void_ty: TypeHandle = VoidType::get(&ctx).into();
                let (_func, entry) = function(&mut ctx, &module, "caller", void_ty, vec![i128_ty]);
                let input = entry.deref(&ctx).get_argument(0);
                let flag = helpers::create_i1_constant(&mut ctx, entry, zero_poison).unwrap();
                let func_ty = FuncType::get(&ctx, i128_ty, vec![i128_ty, i1_ty], false);
                helpers::ensure_intrinsic_declared(&mut ctx, entry, name, func_ty).unwrap();
                let call = llvm::CallOp::new(
                    &mut ctx,
                    CallOpCallable::Direct(name.try_into().unwrap()),
                    func_ty,
                    vec![input, flag],
                );
                call.get_operation().insert_at_back(entry, &ctx);
                llvm::ReturnOp::new(&mut ctx, None)
                    .get_operation()
                    .insert_at_back(entry, &ctx);

                legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap();

                let mut ops = Vec::new();
                collect_ops(&ctx, module.get_operation(), &mut ops);
                let calls: Vec<_> = ops
                    .into_iter()
                    .filter_map(|op| Operation::get_op::<llvm::CallOp>(op, &ctx))
                    .filter_map(|call| match call.callee(&ctx) {
                        CallOpCallable::Direct(name) => Some(name.to_string()),
                        CallOpCallable::Indirect(_) => None,
                    })
                    .collect();
                assert!(!calls.iter().any(|callee| callee == name), "{calls:?}");
                assert!(calls.iter().any(|callee| callee == helper), "{calls:?}");
                module.get_operation().deref(&ctx).verify(&ctx).unwrap();
            }
        }
    }

    #[test]
    fn invalid_bswap_i8_is_rejected_without_mutating_the_module() {
        for common_only in [false, true] {
            let mut ctx = Context::new();
            let module = ModuleOp::new(&mut ctx, "bswap_i8".try_into().unwrap());
            let i8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Signless).into();
            direct_intrinsic_call(&mut ctx, &module, "llvm_bswap_i8", i8_ty, vec![i8_ty]);

            let before = module.get_operation().deref(&ctx).disp(&ctx).to_string();
            let error = if common_only {
                legalize_nvvm_bit_intrinsics(&mut ctx, module.get_operation()).unwrap_err()
            } else {
                legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap_err()
            };
            let after = module.get_operation().deref(&ctx).disp(&ctx).to_string();

            assert!(
                error
                    .disp(&ctx)
                    .to_string()
                    .contains("not a valid LLVM intrinsic")
            );
            assert_eq!(before, after);
        }
    }

    #[test]
    fn dynamic_count_zero_poison_flag_is_rejected_without_mutating_the_module() {
        for name in ["llvm_ctlz_i128", "llvm_cttz_i128"] {
            for common_only in [false, true] {
                let mut ctx = Context::new();
                let module = ModuleOp::new(&mut ctx, name.try_into().unwrap());
                let i128_ty: TypeHandle = IntegerType::get(&ctx, 128, Signedness::Signless).into();
                let i1_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
                direct_intrinsic_call(&mut ctx, &module, name, i128_ty, vec![i128_ty, i1_ty]);

                let before = module.get_operation().deref(&ctx).disp(&ctx).to_string();
                let error = if common_only {
                    legalize_nvvm_bit_intrinsics(&mut ctx, module.get_operation()).unwrap_err()
                } else {
                    legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap_err()
                };
                let after = module.get_operation().deref(&ctx).disp(&ctx).to_string();

                assert!(
                    error
                        .disp(&ctx)
                        .to_string()
                        .contains("immediate i1 constant"),
                    "{}",
                    error.disp(&ctx)
                );
                assert_eq!(before, after, "{name} changed before validation failed");
            }
        }
    }

    #[test]
    fn common_nvvm_bit_legalizer_leaves_modern_operations_untouched() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "modern_bits".try_into().unwrap());
        let i128_ty: TypeHandle = IntegerType::get(&ctx, 128, Signedness::Signless).into();
        direct_intrinsic_call(&mut ctx, &module, "llvm_bswap_i128", i128_ty, vec![i128_ty]);

        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let (_func, entry) = function(&mut ctx, &module, "modern_fneg", f32_ty, vec![f32_ty]);
        let input = entry.deref(&ctx).get_argument(0);
        let fneg =
            llvm::FNegOp::new_with_fast_math_flags(&mut ctx, input, FastmathFlagsAttr::default());
        fneg.get_operation().insert_at_back(entry, &ctx);
        let result = fneg.get_operation().deref(&ctx).get_result(0);
        llvm::ReturnOp::new(&mut ctx, Some(result))
            .get_operation()
            .insert_at_back(entry, &ctx);

        crate::legalize_for_nvvm(
            &mut ctx,
            module.get_operation(),
            llvm_export::export::NvvmIrDialect::Modern,
            90,
        )
        .unwrap();

        let ids = operation_ids(&ctx, module.get_operation());
        assert!(ids.iter().any(|id| id == "llvm.fneg"), "{ids:?}");
        let mut calls = Vec::new();
        let mut ops = Vec::new();
        collect_ops(&ctx, module.get_operation(), &mut ops);
        for op in ops {
            if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
                && let CallOpCallable::Direct(callee) = call.callee(&ctx)
            {
                calls.push(callee.to_string());
            }
        }
        assert!(!calls.iter().any(|name| name == "llvm_bswap_i128"));
        assert!(calls.iter().any(|name| name == "llvm_bswap_i64"));
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();
    }

    #[test]
    fn float_to_int_saturation_handles_bounds_and_nan_without_modern_intrinsic() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "fptosi_sat".try_into().unwrap());
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        direct_intrinsic_call(
            &mut ctx,
            &module,
            "llvm_fptosi_sat_i32_f32",
            i32_ty,
            vec![f32_ty],
        );

        legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap();

        let ids = operation_ids(&ctx, module.get_operation());
        assert_eq!(ids.iter().filter(|id| *id == "llvm.fcmp").count(), 3);
        assert_eq!(ids.iter().filter(|id| *id == "llvm.select").count(), 3);
        assert_eq!(ids.iter().filter(|id| *id == "llvm.fptosi").count(), 1);
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();
    }

    #[test]
    fn modern_shuffle_and_vote_calls_use_legacy_aggregate_intrinsics() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "warp".try_into().unwrap());
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        direct_intrinsic_call(
            &mut ctx,
            &module,
            "llvm_nvvm_shfl_sync_bfly_i32",
            i32_ty,
            vec![i32_ty, i32_ty, i32_ty, i32_ty],
        );

        // Use a distinct caller name so this second helper can coexist in the
        // same module without violating symbol uniqueness.
        let i1_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let vote_ty = FuncType::get(&ctx, i1_ty, vec![i32_ty, i1_ty], false);
        let void_ty: TypeHandle = VoidType::get(&ctx).into();
        let (_vote_func, vote_entry) = function(
            &mut ctx,
            &module,
            "vote_caller",
            void_ty,
            vec![i32_ty, i1_ty],
        );
        helpers::ensure_intrinsic_declared(
            &mut ctx,
            vote_entry,
            "llvm_nvvm_vote_any_sync",
            vote_ty,
        )
        .unwrap();
        let vote_args = vote_entry.deref(&ctx).arguments().collect();
        llvm::CallOp::new(
            &mut ctx,
            CallOpCallable::Direct("llvm_nvvm_vote_any_sync".try_into().unwrap()),
            vote_ty,
            vote_args,
        )
        .get_operation()
        .insert_at_back(vote_entry, &ctx);
        llvm::ReturnOp::new(&mut ctx, None)
            .get_operation()
            .insert_at_back(vote_entry, &ctx);

        legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap();

        let mut calls = Vec::new();
        let mut ops = Vec::new();
        collect_ops(&ctx, module.get_operation(), &mut ops);
        for op in ops {
            if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
                && let CallOpCallable::Direct(callee) = call.callee(&ctx)
            {
                calls.push(callee.to_string());
            }
        }
        assert!(calls.iter().any(|n| n == "llvm_nvvm_shfl_sync_i32"));
        assert!(calls.iter().any(|n| n == "llvm_nvvm_vote_sync"));
        assert!(!calls.iter().any(|n| shuffle_mode(n).is_some()));
        assert!(!calls.iter().any(|n| vote_mode(n).is_some()));
        let ids = operation_ids(&ctx, module.get_operation());
        assert_eq!(
            ids.iter().filter(|id| *id == "llvm.extract_value").count(),
            2
        );
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();
    }

    #[test]
    fn f16_function_signature_is_rejected_without_needing_an_fneg() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "half".try_into().unwrap());
        let f16_ty: TypeHandle = FP16Type::get(&ctx).into();
        let (_func, entry) = function(&mut ctx, &module, "identity_half", f16_ty, vec![f16_ty]);
        let input = entry.deref(&ctx).get_argument(0);
        llvm::ReturnOp::new(&mut ctx, Some(input))
            .get_operation()
            .insert_at_back(entry, &ctx);

        let error = legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap_err();
        let text = error.disp(&ctx).to_string();
        assert!(text.contains("f16"), "{text}");
        assert!(text.contains("CUDA 12"), "{text}");
    }

    #[test]
    fn opaque_memory_intrinsic_overloads_gain_legacy_i8_pointees() {
        assert_eq!(
            legacy_memory_intrinsic_name("llvm_memcpy_p1_p3_i64").as_deref(),
            Some("llvm_memcpy_p1i8_p3i8_i64")
        );
        assert_eq!(
            legacy_memory_intrinsic_name("llvm_memmove_p0_p1_i32").as_deref(),
            Some("llvm_memmove_p0i8_p1i8_i32")
        );
        assert_eq!(
            legacy_memory_intrinsic_name("llvm_memset_p3_i64").as_deref(),
            Some("llvm_memset_p3i8_i64")
        );
        assert_eq!(
            legacy_memory_intrinsic_name("llvm_memcpy_p0i8_p0i8_i64"),
            None,
            "an already-legacy overload must not be rewritten again"
        );
    }

    fn atomic_add_function(
        ctx: &mut Context,
        module: &ModuleOp,
        name: &str,
        address_space: u32,
        value_ty: TypeHandle,
        ordering: AtomicOrderingAttr,
        scope: SyncScopeAttr,
    ) -> llvm::FuncOp {
        let ptr_ty: TypeHandle = PointerType::get(ctx, address_space).into();
        let (func, entry) = function(ctx, module, name, value_ty, vec![ptr_ty, value_ty]);
        let ptr = entry.deref(ctx).get_argument(0);
        let value = entry.deref(ctx).get_argument(1);
        let rmw = llvm::AtomicRmwOp::new(ctx, ptr, value, AtomicRmwKindAttr::FAdd, ordering, scope);
        rmw.get_operation().insert_at_back(entry, ctx);
        let result = rmw.get_operation().deref(ctx).get_result(0);
        llvm::ReturnOp::new(ctx, Some(result))
            .get_operation()
            .insert_at_back(entry, ctx);
        func
    }

    #[allow(clippy::too_many_arguments)]
    fn integer_atomic_rmw_function(
        ctx: &mut Context,
        module: &ModuleOp,
        name: &str,
        address_space: u32,
        width: u32,
        kind: AtomicRmwKindAttr,
        ordering: AtomicOrderingAttr,
        scope: SyncScopeAttr,
    ) -> llvm::FuncOp {
        let value_ty: TypeHandle = IntegerType::get(ctx, width, Signedness::Signless).into();
        scalar_atomic_rmw_function(
            ctx,
            module,
            name,
            address_space,
            value_ty,
            kind,
            ordering,
            scope,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn scalar_atomic_rmw_function(
        ctx: &mut Context,
        module: &ModuleOp,
        name: &str,
        address_space: u32,
        value_ty: TypeHandle,
        kind: AtomicRmwKindAttr,
        ordering: AtomicOrderingAttr,
        scope: SyncScopeAttr,
    ) -> llvm::FuncOp {
        let ptr_ty: TypeHandle = PointerType::get(ctx, address_space).into();
        let (func, entry) = function(ctx, module, name, value_ty, vec![ptr_ty, value_ty]);
        let ptr = entry.deref(ctx).get_argument(0);
        let value = entry.deref(ctx).get_argument(1);
        let rmw = llvm::AtomicRmwOp::new(ctx, ptr, value, kind, ordering, scope);
        rmw.get_operation().insert_at_back(entry, ctx);
        let result = rmw.get_operation().deref(ctx).get_result(0);
        llvm::ReturnOp::new(ctx, Some(result))
            .get_operation()
            .insert_at_back(entry, ctx);
        func
    }

    #[allow(clippy::too_many_arguments)]
    fn integer_cmpxchg_function(
        ctx: &mut Context,
        module: &ModuleOp,
        name: &str,
        address_space: u32,
        width: u32,
        success_ordering: AtomicOrderingAttr,
        failure_ordering: AtomicOrderingAttr,
        scope: SyncScopeAttr,
    ) -> llvm::FuncOp {
        let value_ty: TypeHandle = IntegerType::get(ctx, width, Signedness::Signless).into();
        scalar_cmpxchg_function(
            ctx,
            module,
            name,
            address_space,
            value_ty,
            success_ordering,
            failure_ordering,
            scope,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn scalar_cmpxchg_function(
        ctx: &mut Context,
        module: &ModuleOp,
        name: &str,
        address_space: u32,
        value_ty: TypeHandle,
        success_ordering: AtomicOrderingAttr,
        failure_ordering: AtomicOrderingAttr,
        scope: SyncScopeAttr,
    ) -> llvm::FuncOp {
        let ptr_ty: TypeHandle = PointerType::get(ctx, address_space).into();
        let (func, entry) = function(
            ctx,
            module,
            name,
            value_ty,
            vec![ptr_ty, value_ty, value_ty],
        );
        let ptr = entry.deref(ctx).get_argument(0);
        let compare = entry.deref(ctx).get_argument(1);
        let new_value = entry.deref(ctx).get_argument(2);
        let cas = llvm::AtomicCmpxchgOp::new(
            ctx,
            ptr,
            compare,
            new_value,
            success_ordering,
            failure_ordering,
            scope,
        );
        cas.get_operation().insert_at_back(entry, ctx);
        let aggregate = cas.get_operation().deref(ctx).get_result(0);
        let extract = llvm::ExtractValueOp::new(ctx, aggregate, vec![0]).unwrap();
        extract.get_operation().insert_at_back(entry, ctx);
        let old_value = extract.get_operation().deref(ctx).get_result(0);
        llvm::ReturnOp::new(ctx, Some(old_value))
            .get_operation()
            .insert_at_back(entry, ctx);
        func
    }

    #[test]
    fn legacy_float_atomic_add_uses_nvvm_intrinsics_for_supported_types_and_spaces() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "atomic_add".try_into().unwrap());
        let value_types = [
            (32, TypeHandle::from(FP32Type::get(&ctx))),
            (64, TypeHandle::from(FP64Type::get(&ctx))),
        ];
        let mut functions = Vec::new();
        for (width, value_ty) in value_types {
            for address_space in [0, 1, 3] {
                let function = atomic_add_function(
                    &mut ctx,
                    &module,
                    &format!("atomic_add_f{width}_p{address_space}"),
                    address_space,
                    value_ty,
                    AtomicOrderingAttr::Monotonic,
                    named_scope("device"),
                );
                functions.push(function);
            }
        }

        legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap();

        let mut ops = Vec::new();
        collect_ops(&ctx, module.get_operation(), &mut ops);
        assert!(
            !ops.iter()
                .any(|op| Operation::get_op::<llvm::AtomicRmwOp>(*op, &ctx).is_some())
        );
        let mut calls = ops
            .iter()
            .filter_map(|op| Operation::get_op::<llvm::CallOp>(*op, &ctx))
            .filter_map(|call| match call.callee(&ctx) {
                CallOpCallable::Direct(name) => Some(name.to_string()),
                CallOpCallable::Indirect(_) => None,
            })
            .collect::<Vec<_>>();
        let mut expected_calls = Vec::new();
        for width in [32, 64] {
            for address_space in [0, 1, 3] {
                expected_calls.push(format!(
                    "llvm_nvvm_atomic_load_add_f{width}_p{address_space}f{width}"
                ));
            }
        }
        calls.sort();
        expected_calls.sort();
        assert_eq!(calls, expected_calls);
        for function in functions {
            let entry = function.get_entry_block(&ctx).unwrap();
            let operations = entry.deref(&ctx).iter(&ctx).collect::<Vec<_>>();
            let call = operations
                .iter()
                .find_map(|operation| Operation::get_op::<llvm::CallOp>(*operation, &ctx))
                .expect("legalized function contains the NVVM intrinsic call");
            let return_op = operations
                .iter()
                .find_map(|operation| Operation::get_op::<llvm::ReturnOp>(*operation, &ctx))
                .expect("atomic helper returns its old value");
            assert_eq!(
                return_op.get_operation().deref(&ctx).get_operand(0),
                call.get_operation().deref(&ctx).get_result(0),
                "the atomicrmw result must be replaced with the intrinsic's returned old value"
            );
        }
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();
        let ir = export_module_to_string_with_config(
            &ctx,
            &module,
            &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
        )
        .expect("legalized atomic adds export as LLVM 7 NVVM IR");
        for width in [32, 64] {
            for address_space in [0, 1, 3] {
                let name = format!("llvm.nvvm.atomic.load.add.f{width}.p{address_space}f{width}");
                let scalar = if width == 32 { "float" } else { "double" };
                let pointer = if address_space == 0 {
                    format!("{scalar}*")
                } else {
                    format!("{scalar} addrspace({address_space})*")
                };
                let declaration = format!("declare {scalar} @{name}({pointer}, {scalar})");
                assert!(
                    ir.contains(&declaration),
                    "missing exact `{declaration}`:\n{ir}"
                );
                let call = format!("call {scalar} @{name}({pointer} ");
                assert_eq!(
                    ir.matches(&call).count(),
                    1,
                    "expected one exact typed call beginning `{call}`:\n{ir}"
                );
            }
        }
        assert!(!ir.contains("atomicrmw fadd"), "{ir}");
    }

    #[test]
    fn legacy_float_atomic_add_rejects_semantics_the_intrinsic_cannot_preserve() {
        for (address_space, scope, expected) in [
            (7, named_scope("device"), "address space 7"),
            (1, named_scope("block"), "device synchronization scope"),
            (1, SyncScopeAttr::System, "device synchronization scope"),
        ] {
            let mut ctx = Context::new();
            let module = ModuleOp::new(&mut ctx, "atomic_add".try_into().unwrap());
            let f64_ty: TypeHandle = FP64Type::get(&ctx).into();
            atomic_add_function(
                &mut ctx,
                &module,
                "atomic_add_f64",
                address_space,
                f64_ty,
                AtomicOrderingAttr::Monotonic,
                scope,
            );

            let error = legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap_err();
            assert!(error.disp(&ctx).to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn legacy_float_atomic_add_rejects_non_monotonic_ordering_before_mutation() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "atomic_add".try_into().unwrap());
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let function = atomic_add_function(
            &mut ctx,
            &module,
            "acquire_add",
            1,
            f32_ty,
            AtomicOrderingAttr::Acquire,
            named_scope("device"),
        );

        let error = legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap_err();
        assert!(
            error.disp(&ctx).to_string().contains("monotonic ordering"),
            "{error}"
        );
        let entry = function.get_entry_block(&ctx).unwrap();
        assert!(
            entry
                .deref(&ctx)
                .iter(&ctx)
                .any(|op| { Operation::get_op::<llvm::AtomicRmwOp>(op, &ctx).is_some() })
        );
    }

    #[test]
    fn legacy_f64_atomic_add_rejects_pre_pascal_target_before_mutation() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "atomic_add".try_into().unwrap());
        let f64_ty: TypeHandle = FP64Type::get(&ctx).into();
        let function = atomic_add_function(
            &mut ctx,
            &module,
            "atomic_add_f64",
            1,
            f64_ty,
            AtomicOrderingAttr::Monotonic,
            named_scope("device"),
        );

        let error = legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 52).unwrap_err();
        assert!(
            error
                .disp(&ctx)
                .to_string()
                .contains("requires sm_60 or newer"),
            "{error}"
        );
        let entry = function.get_entry_block(&ctx).unwrap();
        assert!(
            entry
                .deref(&ctx)
                .iter(&ctx)
                .any(|op| { Operation::get_op::<llvm::AtomicRmwOp>(op, &ctx).is_some() })
        );
    }

    #[test]
    fn legacy_f32_atomic_add_accepts_pre_pascal_target() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "atomic_add".try_into().unwrap());
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        atomic_add_function(
            &mut ctx,
            &module,
            "atomic_add_f32",
            1,
            f32_ty,
            AtomicOrderingAttr::Monotonic,
            named_scope("device"),
        );

        legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 52).unwrap();
        let ir = export_module_to_string_with_config(
            &ctx,
            &module,
            &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
        )
        .expect("f32 atomic add legalizes on pre-Pascal targets");
        assert!(
            ir.contains("call float @llvm.nvvm.atomic.load.add.f32.p1f32(float addrspace(1)* "),
            "{ir}"
        );
        assert!(!ir.contains("atomicrmw fadd"), "{ir}");
    }

    #[test]
    fn legacy_float_atomic_add_rejects_incompatible_intrinsic_declaration_before_mutation() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "atomic_add".try_into().unwrap());
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let f64_ty: TypeHandle = FP64Type::get(&ctx).into();
        let ptr_ty: TypeHandle = PointerType::get(&ctx, 1).into();
        let wrong_ty = FuncType::get(&ctx, f64_ty, vec![ptr_ty, f32_ty], false);
        llvm::FuncOp::new(
            &mut ctx,
            "llvm_nvvm_atomic_load_add_f32_p1f32".try_into().unwrap(),
            wrong_ty,
        )
        .get_operation()
        .insert_at_back(module_block(&ctx, &module), &ctx);
        let function = atomic_add_function(
            &mut ctx,
            &module,
            "atomic_add_f32",
            1,
            f32_ty,
            AtomicOrderingAttr::Monotonic,
            named_scope("device"),
        );

        let error = legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap_err();
        assert!(
            error
                .disp(&ctx)
                .to_string()
                .contains("incompatible existing declaration"),
            "{error}"
        );
        let entry = function.get_entry_block(&ctx).unwrap();
        assert!(
            entry
                .deref(&ctx)
                .iter(&ctx)
                .any(|op| { Operation::get_op::<llvm::AtomicRmwOp>(op, &ctx).is_some() })
        );
    }

    #[test]
    fn legacy_float_atomic_add_rejects_intrinsic_definition_before_mutation() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "atomic_add".try_into().unwrap());
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let ptr_ty: TypeHandle = PointerType::get(&ctx, 1).into();
        let (_definition, entry) = function(
            &mut ctx,
            &module,
            "llvm_nvvm_atomic_load_add_f32_p1f32",
            f32_ty,
            vec![ptr_ty, f32_ty],
        );
        let passthrough = entry.deref(&ctx).get_argument(1);
        llvm::ReturnOp::new(&mut ctx, Some(passthrough))
            .get_operation()
            .insert_at_back(entry, &ctx);
        let function = atomic_add_function(
            &mut ctx,
            &module,
            "atomic_add_f32",
            1,
            f32_ty,
            AtomicOrderingAttr::Monotonic,
            named_scope("device"),
        );
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();

        let error = legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap_err();
        assert!(
            error.disp(&ctx).to_string().contains("function definition"),
            "{error}"
        );
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();
        let entry = function.get_entry_block(&ctx).unwrap();
        assert!(
            entry
                .deref(&ctx)
                .iter(&ctx)
                .any(|op| Operation::get_op::<llvm::AtomicRmwOp>(op, &ctx).is_some())
        );
    }

    #[test]
    fn legacy_float_atomic_add_rejects_non_function_symbol_collision_before_mutation() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "atomic_add".try_into().unwrap());
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        llvm::GlobalOp::new(
            &mut ctx,
            "llvm_nvvm_atomic_load_add_f32_p1f32".try_into().unwrap(),
            f32_ty,
        )
        .get_operation()
        .insert_at_back(module_block(&ctx, &module), &ctx);
        let function = atomic_add_function(
            &mut ctx,
            &module,
            "atomic_add_f32",
            1,
            f32_ty,
            AtomicOrderingAttr::Monotonic,
            named_scope("device"),
        );
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();

        let error = legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap_err();
        assert!(
            error
                .disp(&ctx)
                .to_string()
                .contains("existing non-function symbol"),
            "{error}"
        );
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();
        let entry = function.get_entry_block(&ctx).unwrap();
        assert!(
            entry
                .deref(&ctx)
                .iter(&ctx)
                .any(|op| Operation::get_op::<llvm::AtomicRmwOp>(op, &ctx).is_some())
        );
    }

    #[test]
    fn invalid_float_atomic_add_leaves_the_complete_module_unchanged() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "atomic_add".try_into().unwrap());
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        atomic_add_function(
            &mut ctx,
            &module,
            "valid_first",
            1,
            f32_ty,
            AtomicOrderingAttr::Monotonic,
            named_scope("device"),
        );
        atomic_add_function(
            &mut ctx,
            &module,
            "invalid_second",
            1,
            f32_ty,
            AtomicOrderingAttr::Monotonic,
            named_scope("block"),
        );

        let error = legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap_err();
        assert!(
            error
                .disp(&ctx)
                .to_string()
                .contains("device synchronization scope"),
            "{error}"
        );

        let mut ops = Vec::new();
        collect_ops(&ctx, module.get_operation(), &mut ops);
        assert_eq!(
            ops.iter()
                .filter(|op| Operation::get_op::<llvm::AtomicRmwOp>(**op, &ctx).is_some())
                .count(),
            2,
            "validation failure must not rewrite an earlier valid atomic"
        );
        assert!(
            !ops.iter()
                .any(|op| Operation::get_op::<llvm::CallOp>(*op, &ctx).is_some()),
            "validation failure must not insert intrinsic calls"
        );
    }

    #[test]
    fn legacy_integer_atomic_rmw_is_preserved_for_supported_llvm7_forms() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "integer_atomic_rmw".try_into().unwrap());

        for width in [32, 64] {
            for address_space in [0, 1, 3] {
                integer_atomic_rmw_function(
                    &mut ctx,
                    &module,
                    &format!("rmw_i{width}_p{address_space}"),
                    address_space,
                    width,
                    AtomicRmwKindAttr::Add,
                    AtomicOrderingAttr::Monotonic,
                    named_scope("device"),
                );
            }
        }

        legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap();
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();

        let mut ops = Vec::new();
        collect_ops(&ctx, module.get_operation(), &mut ops);
        assert_eq!(
            ops.iter()
                .filter(|op| Operation::get_op::<llvm::AtomicRmwOp>(**op, &ctx).is_some())
                .count(),
            6,
            "supported integer atomicrmw operations must remain native LLVM operations"
        );

        let ir = export_module_to_string_with_config(
            &ctx,
            &module,
            &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
        )
        .expect("supported integer atomicrmw exports as LLVM 7 NVVM IR");
        for scalar in ["i32", "i64"] {
            for address_space in [0, 1, 3] {
                let pointer = if address_space == 0 {
                    format!("{scalar}*")
                } else {
                    format!("{scalar} addrspace({address_space})*")
                };
                let prefix = format!("atomicrmw add {pointer} ");
                assert!(
                    ir.contains(&prefix),
                    "missing typed legacy atomicrmw beginning `{prefix}`:\n{ir}"
                );
            }
        }
        assert!(
            ir.contains("syncscope(\"device\") monotonic"),
            "legacy integer atomicrmw must keep device scope and monotonic ordering:\n{ir}"
        );
    }

    #[test]
    fn legacy_integer_atomic_rmw_accepts_the_supported_operation_set() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "integer_atomic_kinds".try_into().unwrap());
        for (index, kind) in [
            AtomicRmwKindAttr::Xchg,
            AtomicRmwKindAttr::Add,
            AtomicRmwKindAttr::Sub,
            AtomicRmwKindAttr::And,
            AtomicRmwKindAttr::Or,
            AtomicRmwKindAttr::Xor,
            AtomicRmwKindAttr::Max,
            AtomicRmwKindAttr::Min,
            AtomicRmwKindAttr::UMax,
            AtomicRmwKindAttr::UMin,
        ]
        .into_iter()
        .enumerate()
        {
            integer_atomic_rmw_function(
                &mut ctx,
                &module,
                &format!("rmw_kind_{index}"),
                1,
                32,
                kind,
                AtomicOrderingAttr::Monotonic,
                named_scope("device"),
            );
        }

        legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap();
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();
        let mut ops = Vec::new();
        collect_ops(&ctx, module.get_operation(), &mut ops);
        assert_eq!(
            ops.iter()
                .filter(|op| Operation::get_op::<llvm::AtomicRmwOp>(**op, &ctx).is_some())
                .count(),
            10
        );
    }

    #[test]
    fn legacy_integer_cmpxchg_is_preserved_for_supported_llvm7_forms() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "integer_cmpxchg".try_into().unwrap());

        for width in [32, 64] {
            for address_space in [0, 1, 3] {
                integer_cmpxchg_function(
                    &mut ctx,
                    &module,
                    &format!("cas_i{width}_p{address_space}"),
                    address_space,
                    width,
                    AtomicOrderingAttr::Monotonic,
                    AtomicOrderingAttr::Monotonic,
                    named_scope("device"),
                );
            }
        }

        legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap();
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();

        let mut ops = Vec::new();
        collect_ops(&ctx, module.get_operation(), &mut ops);
        assert_eq!(
            ops.iter()
                .filter(|op| Operation::get_op::<llvm::AtomicCmpxchgOp>(**op, &ctx).is_some())
                .count(),
            6,
            "supported cmpxchg operations must remain native LLVM operations"
        );
        assert_eq!(
            ops.iter()
                .filter(|op| Operation::get_op::<llvm::ExtractValueOp>(**op, &ctx).is_some())
                .count(),
            6,
            "cmpxchg old-value extraction must remain intact"
        );

        let ir = export_module_to_string_with_config(
            &ctx,
            &module,
            &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
        )
        .expect("supported cmpxchg exports as LLVM 7 NVVM IR");
        for scalar in ["i32", "i64"] {
            for address_space in [0, 1, 3] {
                let pointer = if address_space == 0 {
                    format!("{scalar}*")
                } else {
                    format!("{scalar} addrspace({address_space})*")
                };
                let prefix = format!("cmpxchg {pointer} ");
                assert!(
                    ir.contains(&prefix),
                    "missing typed legacy cmpxchg beginning `{prefix}`:\n{ir}"
                );
            }
        }
        assert!(
            ir.contains("syncscope(\"device\") monotonic monotonic"),
            "legacy cmpxchg must retain monotonic success and failure ordering:\n{ir}"
        );
    }

    #[test]
    fn legacy_scoped_integer_atomic_rmw_rewrites_to_inline_ptx() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "scoped_integer_rmw".try_into().unwrap());

        let mut expected = Vec::new();
        for width in [32, 64] {
            for (scope_name, scope, ptx_scope) in [
                ("block", named_scope("block"), "cta"),
                ("system", SyncScopeAttr::System, "sys"),
            ] {
                for address_space in [0, 1, 3] {
                    integer_atomic_rmw_function(
                        &mut ctx,
                        &module,
                        &format!("rmw_{scope_name}_i{width}_p{address_space}"),
                        address_space,
                        width,
                        AtomicRmwKindAttr::Add,
                        AtomicOrderingAttr::Monotonic,
                        scope.clone(),
                    );
                    expected.push(format!("atom.relaxed.{ptx_scope}.add.u{width}"));
                }
            }
        }

        legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap();
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();

        let mut ops = Vec::new();
        collect_ops(&ctx, module.get_operation(), &mut ops);
        assert_eq!(
            ops.iter()
                .filter(|op| Operation::get_op::<llvm::AtomicRmwOp>(**op, &ctx).is_some())
                .count(),
            0,
            "scoped integer RMWs must not remain native LLVM atomics"
        );
        assert_eq!(
            ops.iter()
                .filter(|op| Operation::get_op::<llvm::InlineAsmOp>(**op, &ctx).is_some())
                .count(),
            expected.len()
        );

        let ir = export_module_to_string_with_config(
            &ctx,
            &module,
            &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
        )
        .expect("scoped integer RMWs export as legacy-compatible inline PTX");
        for instruction in expected {
            assert!(ir.contains(&instruction), "missing `{instruction}`:\n{ir}");
        }
    }

    #[test]
    fn legacy_scoped_integer_atomic_rmw_rewrites_the_supported_operation_set() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "scoped_integer_rmw_kinds".try_into().unwrap());
        let cases = [
            (AtomicRmwKindAttr::Xchg, "exch.b32"),
            (AtomicRmwKindAttr::Add, "add.u32"),
            (AtomicRmwKindAttr::Sub, "add.u32"),
            (AtomicRmwKindAttr::And, "and.b32"),
            (AtomicRmwKindAttr::Or, "or.b32"),
            (AtomicRmwKindAttr::Xor, "xor.b32"),
            (AtomicRmwKindAttr::Max, "max.s32"),
            (AtomicRmwKindAttr::Min, "min.s32"),
            (AtomicRmwKindAttr::UMax, "max.u32"),
            (AtomicRmwKindAttr::UMin, "min.u32"),
        ];

        for (index, (kind, _)) in cases.iter().enumerate() {
            integer_atomic_rmw_function(
                &mut ctx,
                &module,
                &format!("rmw_kind_{index}"),
                1,
                32,
                (*kind).clone(),
                AtomicOrderingAttr::Monotonic,
                named_scope("block"),
            );
        }

        legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap();
        let ir = export_module_to_string_with_config(
            &ctx,
            &module,
            &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
        )
        .expect("supported scoped RMW kinds export as inline PTX");

        for (_, opcode) in cases {
            let instruction = format!("atom.relaxed.cta.{opcode}");
            assert!(ir.contains(&instruction), "missing `{instruction}`:\n{ir}");
        }
        assert!(
            ir.matches("atom.relaxed.cta.add.u32").count() >= 2,
            "add and sub must both lower through PTX add:\n{ir}"
        );
    }

    #[test]
    fn legacy_scoped_integer_atomic_rmw_requires_sm70() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "scoped_integer_rmw_floor".try_into().unwrap());
        integer_atomic_rmw_function(
            &mut ctx,
            &module,
            "rmw",
            1,
            32,
            AtomicRmwKindAttr::Add,
            AtomicOrderingAttr::Monotonic,
            named_scope("block"),
        );

        let before = module.get_operation().deref(&ctx).disp(&ctx).to_string();
        let error = legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 69).unwrap_err();
        let after = module.get_operation().deref(&ctx).disp(&ctx).to_string();
        assert!(
            error.disp(&ctx).to_string().contains("requires sm_70"),
            "{error}"
        );
        assert_eq!(before, after, "target-floor failure must be transactional");
    }

    #[test]
    fn legacy_pointer_exchange_and_cas_preserve_pointer_types_and_every_scope() {
        for address_space in [0, 1, 3] {
            for (scope, ptx_scope) in [
                (named_scope("device"), "gpu"),
                (named_scope("block"), "cta"),
                (SyncScopeAttr::System, "sys"),
            ] {
                let mut ctx = Context::new();
                let module = ModuleOp::new(&mut ctx, "pointer_atomics".try_into().unwrap());
                let value_ty: TypeHandle = PointerType::get(&ctx, 0).into();
                scalar_atomic_rmw_function(
                    &mut ctx,
                    &module,
                    "exchange",
                    address_space,
                    value_ty,
                    AtomicRmwKindAttr::Xchg,
                    AtomicOrderingAttr::Monotonic,
                    scope.clone(),
                );
                scalar_cmpxchg_function(
                    &mut ctx,
                    &module,
                    "compare",
                    address_space,
                    value_ty,
                    AtomicOrderingAttr::Release,
                    AtomicOrderingAttr::Acquire,
                    scope,
                );
                legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap();
                module.get_operation().deref(&ctx).verify(&ctx).unwrap();
                let ir = export_module_to_string_with_config(
                    &ctx,
                    &module,
                    &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
                )
                .unwrap();
                assert_eq!(
                    ir.matches(&format!("atom.relaxed.{ptx_scope}.exch.b64"))
                        .count(),
                    1,
                    "{ir}"
                );
                assert_eq!(
                    ir.matches(&format!("atom.acq_rel.{ptx_scope}.cas.b64"))
                        .count(),
                    1,
                    "{ir}"
                );
                assert_eq!(ir.matches("call i8* asm sideeffect").count(), 2, "{ir}");
                assert_eq!(ir.matches("ret i8*").count(), 2, "{ir}");
                assert!(!ir.contains("atomicrmw") && !ir.contains("cmpxchg"), "{ir}");
                assert!(!ir.contains("ptrtoint") && !ir.contains("inttoptr"), "{ir}");
                if address_space != 0 {
                    assert!(
                        ir.contains(&format!("addrspacecast i8 addrspace({address_space})*")),
                        "{ir}"
                    );
                }
            }
        }
    }

    #[test]
    fn legacy_pointer_atomics_reject_non_generic_values_arithmetic_and_old_targets() {
        for (value_space, kind, capability, expected) in [
            (1, AtomicRmwKindAttr::Xchg, 90, "generic pointer value"),
            (3, AtomicRmwKindAttr::Xchg, 90, "generic pointer value"),
            (0, AtomicRmwKindAttr::Add, 90, "supports only exchange"),
            (0, AtomicRmwKindAttr::Xchg, 69, "requires sm_70"),
        ] {
            let mut ctx = Context::new();
            let module = ModuleOp::new(&mut ctx, "invalid_pointer_rmw".try_into().unwrap());
            let value_ty: TypeHandle = PointerType::get(&ctx, value_space).into();
            scalar_atomic_rmw_function(
                &mut ctx,
                &module,
                "exchange",
                0,
                value_ty,
                kind,
                AtomicOrderingAttr::Monotonic,
                named_scope("device"),
            );
            let before = module.get_operation().deref(&ctx).disp(&ctx).to_string();
            let error =
                legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), capability).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
            assert_eq!(
                before,
                module.get_operation().deref(&ctx).disp(&ctx).to_string()
            );
        }
        for (value_space, capability, expected) in [
            (1, 90, "generic pointer value"),
            (3, 90, "generic pointer value"),
            (0, 69, "requires sm_70"),
        ] {
            let mut ctx = Context::new();
            let module = ModuleOp::new(&mut ctx, "invalid_pointer_cas".try_into().unwrap());
            let value_ty: TypeHandle = PointerType::get(&ctx, value_space).into();
            scalar_cmpxchg_function(
                &mut ctx,
                &module,
                "compare",
                0,
                value_ty,
                AtomicOrderingAttr::Monotonic,
                AtomicOrderingAttr::Monotonic,
                named_scope("device"),
            );
            let before = module.get_operation().deref(&ctx).disp(&ctx).to_string();
            let error =
                legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), capability).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
            assert_eq!(
                before,
                module.get_operation().deref(&ctx).disp(&ctx).to_string()
            );
        }
    }

    #[test]
    fn legacy_cmpxchg_covers_every_rust_ordering_pair_without_weakening_failure() {
        use AtomicOrderingAttr::*;
        let success_orderings = [Monotonic, Acquire, Release, AcqRel, SeqCst];
        let failure_orderings = [Monotonic, Acquire, SeqCst];
        // Each cell is the PTX semantic and whether a preceding SC fence is
        // needed. Rows are success orderings; columns are failure orderings.
        let expected = [
            [("relaxed", false), ("acquire", false), ("acquire", true)],
            [("acquire", false), ("acquire", false), ("acquire", true)],
            [("release", false), ("acq_rel", false), ("acquire", true)],
            [("acq_rel", false), ("acq_rel", false), ("acquire", true)],
            [("acquire", true), ("acquire", true), ("acquire", true)],
        ];
        for width in [32, 64] {
            for (scope, ptx_scope) in [
                (named_scope("device"), "gpu"),
                (named_scope("block"), "cta"),
                (SyncScopeAttr::System, "sys"),
            ] {
                for (row, success) in success_orderings.iter().enumerate() {
                    for (column, failure) in failure_orderings.iter().enumerate() {
                        let mut ctx = Context::new();
                        let module = ModuleOp::new(&mut ctx, "ordering_pair".try_into().unwrap());
                        integer_cmpxchg_function(
                            &mut ctx,
                            &module,
                            "cas",
                            1,
                            width,
                            success.clone(),
                            failure.clone(),
                            scope.clone(),
                        );
                        legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap();
                        module.get_operation().deref(&ctx).verify(&ctx).unwrap();
                        let ir = export_module_to_string_with_config(
                            &ctx,
                            &module,
                            &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
                        )
                        .unwrap();
                        if row == 0 && column == 0 && ptx_scope == "gpu" {
                            assert_eq!(ir.matches("cmpxchg").count(), 1, "{ir}");
                            assert!(!ir.contains("atom."), "{ir}");
                        } else {
                            let (sem, fence) = expected[row][column];
                            let instruction = format!("atom.{sem}.{ptx_scope}.cas.b{width}");
                            assert_eq!(ir.matches(&instruction).count(), 1, "{ir}");
                            assert_eq!(
                                ir.contains(&format!("fence.sc.{ptx_scope}")),
                                fence,
                                "{ir}"
                            );
                            assert!(!ir.contains("cmpxchg"), "{ir}");
                        }
                    }
                }
            }
        }
        for success in success_orderings {
            for failure in [Release, AcqRel] {
                let mut ctx = Context::new();
                let module = ModuleOp::new(&mut ctx, "invalid_pair".try_into().unwrap());
                integer_cmpxchg_function(
                    &mut ctx,
                    &module,
                    "cas",
                    1,
                    32,
                    success.clone(),
                    failure,
                    SyncScopeAttr::System,
                );
                let before = module.get_operation().deref(&ctx).disp(&ctx).to_string();
                let error =
                    legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains("invalid success/failure ordering pair"),
                    "{error}"
                );
                assert_eq!(
                    before,
                    module.get_operation().deref(&ctx).disp(&ctx).to_string()
                );
            }
        }
    }

    #[test]
    fn legacy_scalar_cmpxchg_rewrites_scopes_and_strong_failure_ordering() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "scoped_integer_cas".try_into().unwrap());

        integer_cmpxchg_function(
            &mut ctx,
            &module,
            "device_acquire_failure",
            1,
            32,
            AtomicOrderingAttr::AcqRel,
            AtomicOrderingAttr::Acquire,
            named_scope("device"),
        );
        integer_cmpxchg_function(
            &mut ctx,
            &module,
            "block_relaxed_failure",
            1,
            32,
            AtomicOrderingAttr::AcqRel,
            AtomicOrderingAttr::Monotonic,
            named_scope("block"),
        );
        integer_cmpxchg_function(
            &mut ctx,
            &module,
            "system_acquire_failure",
            1,
            64,
            AtomicOrderingAttr::Acquire,
            AtomicOrderingAttr::Acquire,
            SyncScopeAttr::System,
        );
        integer_cmpxchg_function(
            &mut ctx,
            &module,
            "device_seqcst_failure",
            1,
            32,
            AtomicOrderingAttr::SeqCst,
            AtomicOrderingAttr::SeqCst,
            named_scope("device"),
        );

        legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap();
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();

        let mut ops = Vec::new();
        collect_ops(&ctx, module.get_operation(), &mut ops);
        assert_eq!(
            ops.iter()
                .filter(|op| Operation::get_op::<llvm::AtomicCmpxchgOp>(**op, &ctx).is_some())
                .count(),
            0
        );
        assert_eq!(
            ops.iter()
                .filter(|op| Operation::get_op::<llvm::ExtractValueOp>(**op, &ctx).is_some())
                .count(),
            0,
            "PTX CAS returns the old scalar directly, so canonical extracts must disappear"
        );

        let ir = export_module_to_string_with_config(
            &ctx,
            &module,
            &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
        )
        .expect("rewritten compare-exchange exports as legacy-compatible inline PTX");
        for instruction in [
            "atom.acq_rel.gpu.cas.b32",
            "atom.acq_rel.cta.cas.b32",
            "atom.acquire.sys.cas.b64",
            "fence.sc.gpu; atom.acquire.gpu.cas.b32",
        ] {
            assert!(ir.contains(instruction), "missing `{instruction}`:\n{ir}");
        }
    }

    #[test]
    fn legacy_scalar_cmpxchg_rewrites_ordered_success_with_monotonic_failure() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "ordered_success_cas".try_into().unwrap());

        for (name, width, success) in [
            ("device_acquire_success", 32, AtomicOrderingAttr::Acquire),
            ("device_release_success", 64, AtomicOrderingAttr::Release),
            ("device_acq_rel_success", 32, AtomicOrderingAttr::AcqRel),
            ("device_seq_cst_success", 64, AtomicOrderingAttr::SeqCst),
        ] {
            integer_cmpxchg_function(
                &mut ctx,
                &module,
                name,
                1,
                width,
                success,
                AtomicOrderingAttr::Monotonic,
                named_scope("device"),
            );
        }

        legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap();
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();

        let mut ops = Vec::new();
        collect_ops(&ctx, module.get_operation(), &mut ops);
        assert_eq!(
            ops.iter()
                .filter(|op| Operation::get_op::<llvm::AtomicCmpxchgOp>(**op, &ctx).is_some())
                .count(),
            0,
            "ordered-success compare-exchange must route to the PTX rewrite lane"
        );

        let ir = export_module_to_string_with_config(
            &ctx,
            &module,
            &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
        )
        .expect("ordered-success compare-exchange exports as legacy-compatible inline PTX");
        for instruction in [
            "atom.acquire.gpu.cas.b32",
            "atom.release.gpu.cas.b64",
            "atom.acq_rel.gpu.cas.b32",
            "fence.sc.gpu; atom.acquire.gpu.cas.b64",
        ] {
            assert!(ir.contains(instruction), "missing `{instruction}`:\n{ir}");
        }
    }

    #[test]
    fn legacy_integer_cmpxchg_ptx_rewrite_requires_sm70() {
        for (scope, failure) in [
            (named_scope("block"), AtomicOrderingAttr::Monotonic),
            (named_scope("device"), AtomicOrderingAttr::Acquire),
            (named_scope("device"), AtomicOrderingAttr::Monotonic),
        ] {
            let mut ctx = Context::new();
            let module = ModuleOp::new(&mut ctx, "integer_cas_floor".try_into().unwrap());
            integer_cmpxchg_function(
                &mut ctx,
                &module,
                "cas",
                1,
                32,
                AtomicOrderingAttr::AcqRel,
                failure,
                scope,
            );

            let before = module.get_operation().deref(&ctx).disp(&ctx).to_string();
            let error = legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 69).unwrap_err();
            let after = module.get_operation().deref(&ctx).disp(&ctx).to_string();
            assert!(
                error.disp(&ctx).to_string().contains("requires sm_70"),
                "{error}"
            );
            assert_eq!(before, after, "target-floor failure must be transactional");
        }
    }

    #[test]
    fn legacy_integer_atomic_rmw_rejects_unrepresentable_forms() {
        for (width, address_space, ordering, scope, kind, expected) in [
            (
                16,
                1,
                AtomicOrderingAttr::Monotonic,
                named_scope("device"),
                AtomicRmwKindAttr::Add,
                "only i32 and i64",
            ),
            (
                128,
                1,
                AtomicOrderingAttr::Monotonic,
                named_scope("device"),
                AtomicRmwKindAttr::Add,
                "only i32 and i64",
            ),
            (
                32,
                7,
                AtomicOrderingAttr::Monotonic,
                named_scope("device"),
                AtomicRmwKindAttr::Add,
                "address space 7",
            ),
            (
                32,
                1,
                AtomicOrderingAttr::Acquire,
                named_scope("device"),
                AtomicRmwKindAttr::Add,
                "monotonic ordering",
            ),
            (
                32,
                1,
                AtomicOrderingAttr::Monotonic,
                named_scope("cluster"),
                AtomicRmwKindAttr::Add,
                "unsupported synchronization scope",
            ),
            (
                32,
                1,
                AtomicOrderingAttr::Monotonic,
                named_scope("device"),
                AtomicRmwKindAttr::Nand,
                "does not support this operation kind",
            ),
        ] {
            let mut ctx = Context::new();
            let module = ModuleOp::new(&mut ctx, "invalid_integer_rmw".try_into().unwrap());
            integer_atomic_rmw_function(
                &mut ctx,
                &module,
                "rmw",
                address_space,
                width,
                kind,
                ordering,
                scope,
            );
            let before = module.get_operation().deref(&ctx).disp(&ctx).to_string();
            let error = legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap_err();
            let after = module.get_operation().deref(&ctx).disp(&ctx).to_string();
            assert!(error.disp(&ctx).to_string().contains(expected), "{error}");
            assert_eq!(
                before, after,
                "failed validation must not mutate the module"
            );
        }
    }

    #[test]
    fn legacy_integer_cmpxchg_rejects_unrepresentable_forms() {
        for (width, address_space, success, failure, scope, expected) in [
            (
                16,
                1,
                AtomicOrderingAttr::AcqRel,
                AtomicOrderingAttr::Monotonic,
                named_scope("device"),
                "only i32 and i64",
            ),
            (
                128,
                1,
                AtomicOrderingAttr::AcqRel,
                AtomicOrderingAttr::Monotonic,
                named_scope("device"),
                "only i32 and i64",
            ),
            (
                32,
                7,
                AtomicOrderingAttr::AcqRel,
                AtomicOrderingAttr::Monotonic,
                named_scope("device"),
                "address space 7",
            ),
            (
                32,
                1,
                AtomicOrderingAttr::AcqRel,
                AtomicOrderingAttr::Monotonic,
                named_scope("cluster"),
                "unsupported synchronization scope",
            ),
            (
                32,
                1,
                AtomicOrderingAttr::Release,
                AtomicOrderingAttr::Release,
                named_scope("device"),
                "invalid success/failure ordering pair",
            ),
        ] {
            let mut ctx = Context::new();
            let module = ModuleOp::new(&mut ctx, "invalid_integer_cas".try_into().unwrap());
            integer_cmpxchg_function(
                &mut ctx,
                &module,
                "cas",
                address_space,
                width,
                success,
                failure,
                scope,
            );
            let before = module.get_operation().deref(&ctx).disp(&ctx).to_string();
            let error = legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap_err();
            let after = module.get_operation().deref(&ctx).disp(&ctx).to_string();
            assert!(error.disp(&ctx).to_string().contains(expected), "{error}");
            assert_eq!(
                before, after,
                "failed validation must not mutate the module"
            );
        }
    }

    #[test]
    fn invalid_integer_cmpxchg_leaves_an_earlier_valid_rmw_untouched() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "integer_atomic_transaction".try_into().unwrap());
        integer_atomic_rmw_function(
            &mut ctx,
            &module,
            "valid_rmw",
            1,
            32,
            AtomicRmwKindAttr::Add,
            AtomicOrderingAttr::Monotonic,
            named_scope("device"),
        );
        integer_cmpxchg_function(
            &mut ctx,
            &module,
            "invalid_cas",
            1,
            32,
            AtomicOrderingAttr::AcqRel,
            AtomicOrderingAttr::Monotonic,
            named_scope("cluster"),
        );

        let before = module.get_operation().deref(&ctx).disp(&ctx).to_string();
        let error = legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap_err();
        assert!(
            error
                .disp(&ctx)
                .to_string()
                .contains("unsupported synchronization scope"),
            "{error}"
        );
        let after = module.get_operation().deref(&ctx).disp(&ctx).to_string();
        assert_eq!(
            before, after,
            "validation failure must not mutate an earlier valid integer atomic"
        );
    }

    #[test]
    fn unsupported_atomic_load_fails_before_mutating_other_ops() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "atomic".try_into().unwrap());
        let ptr_ty: TypeHandle = PointerType::get(&ctx, 1).into();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let void_ty: TypeHandle = VoidType::get(&ctx).into();
        let (_func, entry) = function(&mut ctx, &module, "atomic_load", void_ty, vec![ptr_ty]);
        let ptr = entry.deref(&ctx).get_argument(0);
        let load = llvm::AtomicLoadOp::new(
            &mut ctx,
            ptr,
            i32_ty,
            AtomicOrderingAttr::Monotonic,
            SyncScopeAttr::System,
        );
        load.get_operation().insert_at_back(entry, &ctx);
        llvm::ReturnOp::new(&mut ctx, None)
            .get_operation()
            .insert_at_back(entry, &ctx);

        let error = legalize_for_legacy_nvvm(&mut ctx, module.get_operation(), 90).unwrap_err();
        assert!(error.disp(&ctx).to_string().contains("atomic loads"));
        assert!(Operation::get_op::<llvm::AtomicLoadOp>(load.get_operation(), &ctx).is_some());
    }
}
