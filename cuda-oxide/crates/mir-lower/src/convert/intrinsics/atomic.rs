/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Atomic operation conversion: NVVM atomic dialect → LLVM atomic instructions.
//!
//! Preserves atomic value, ordering and scope semantics across GPU storage
//! spaces. Generic storage is classified at run time; known storage spaces
//! bypass the dispatch. Global and shared storage retain atomic operations,
//! while thread-private local storage uses ordinary typed memory operations.
//! Standalone fences remain synchronization operations in every case.
//!
//! # Lowering Strategy
//!
//! For nonlocal storage, RMW and compare-exchange lower to standard LLVM IR
//! instructions. Loads
//! and stores lower to inline PTX, because libNVVM rejects `load atomic` /
//! `store atomic` outright ("Atomic loads/stores are not supported"):
//!
//! | NVVM Op                 | Lowered form                             |
//! |-------------------------|------------------------------------------|
//! | `NvvmAtomicLoadOp`      | inline PTX `ld.{sem}.{scope}.{ty}`       |
//! | `NvvmAtomicStoreOp`     | inline PTX `st.{sem}.{scope}.{ty}`       |
//! | `NvvmAtomicFenceOp`     | inline PTX or `llvm.nvvm.membar.*`       |
//! | `NvvmAtomicRmwOp`       | `atomicrmw ... syncscope("device")` `[*]`  |
//! | `NvvmAtomicCmpxchgOp`   | `cmpxchg ... syncscope("device")`        |
//!
//! `[*]` atomicrmw uses fence splitting workaround -- see below.
//!
//! # atomicrmw Fence Splitting Workaround
//!
//! LLVM's NVPTX backend silently drops orderings on `atomicrmw`
//! (fix is in LLVM 23 via PR #176015). Until then, we emit:
//!
//! ```text
//! Relaxed:  atomicrmw ... monotonic
//! Acquire:  atomicrmw ... monotonic  +  fence acquire
//! Release:  fence release  +  atomicrmw ... monotonic
//! AcqRel:   fence release  +  atomicrmw ... monotonic  +  fence acquire
//! SeqCst:   fence seq_cst  +  atomicrmw ... monotonic  +  fence seq_cst
//! ```
//!
//! All fences carry the same syncscope as the atomic op.
//!
//! # Scope → Syncscope Mapping
//!
//! | NVVM Scope | LLVM syncscope     | PTX scope |
//! |------------|--------------------|-----------|
//! | Device     | `"device"`         | `.gpu`    |
//! | Block      | `"block"`          | `.cta`    |
//! | System     | (default)          | `.sys`    |

use crate::convert::intrinsics::common;
use crate::convert::types::convert_type;

use dialect_nvvm::ops::atomic::{
    AtomicOrdering as NvvmOrdering, AtomicRmwKind as NvvmRmwKind, AtomicScope as NvvmScope,
    NvvmAtomicCmpxchgOp, NvvmAtomicFenceOp, NvvmAtomicLoadOp, NvvmAtomicOpInterface,
    NvvmAtomicRmwOp, NvvmAtomicStoreOp,
};
use llvm_export::attributes::{
    ICmpPredicateAttr, LlvmAtomicOrdering, LlvmAtomicRmwKind, SyncScopeAttr,
};
use llvm_export::op_interfaces::{
    BinArithOp, CastOpInterface, FloatBinArithOpWithFastMathFlags, IntBinArithOpWithOverflowFlag,
};
use llvm_export::ops as llvm;
use llvm_export::ops::{AsmKind, InlineAsmOpExt};
use llvm_export::types as llvm_types;

use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::StringAttr;
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::{BlockInsertionPoint, Inserter, OpInsertionPoint};
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;

// =============================================================================
// Scope / Ordering Mapping
// =============================================================================

/// Map an NVVM scope to the upstream [`SyncScopeAttr`]. Device and block are
/// named scopes in LLVM-IR (`syncscope("device")` / `syncscope("block")`);
/// system is LLVM's unnamed default.
fn map_scope(scope: &NvvmScope) -> SyncScopeAttr {
    match scope {
        NvvmScope::Device => SyncScopeAttr::NamedScope(StringAttr::new("device".to_string())),
        NvvmScope::Block => SyncScopeAttr::NamedScope(StringAttr::new("block".to_string())),
        NvvmScope::System => SyncScopeAttr::System,
    }
}

fn map_ordering(ord: &NvvmOrdering) -> LlvmAtomicOrdering {
    match ord {
        NvvmOrdering::Relaxed => LlvmAtomicOrdering::Monotonic,
        NvvmOrdering::Acquire => LlvmAtomicOrdering::Acquire,
        NvvmOrdering::Release => LlvmAtomicOrdering::Release,
        NvvmOrdering::AcqRel => LlvmAtomicOrdering::AcqRel,
        NvvmOrdering::SeqCst => LlvmAtomicOrdering::SeqCst,
    }
}

fn map_rmw_kind(kind: &NvvmRmwKind) -> LlvmAtomicRmwKind {
    match kind {
        NvvmRmwKind::Add => LlvmAtomicRmwKind::Add,
        NvvmRmwKind::Sub => LlvmAtomicRmwKind::Sub,
        NvvmRmwKind::And => LlvmAtomicRmwKind::And,
        NvvmRmwKind::Or => LlvmAtomicRmwKind::Or,
        NvvmRmwKind::Xor => LlvmAtomicRmwKind::Xor,
        NvvmRmwKind::Xchg => LlvmAtomicRmwKind::Xchg,
        NvvmRmwKind::Min => LlvmAtomicRmwKind::Min,
        NvvmRmwKind::Max => LlvmAtomicRmwKind::Max,
        NvvmRmwKind::UMin => LlvmAtomicRmwKind::UMin,
        NvvmRmwKind::UMax => LlvmAtomicRmwKind::UMax,
        NvvmRmwKind::FAdd => LlvmAtomicRmwKind::FAdd,
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// PTX type suffix, register constraint class, and (for floats) the integer
/// type the value is staged through.
///
/// The asm always uses an integer register class (`h`/`r`/`l`). A float
/// operand or result is bitcast to and from the same-width integer type at
/// the LLVM dialect level, so the asm operand type always agrees with its
/// constraint; handing llc a float value under an integer constraint is a
/// constraint mismatch.
fn ptx_type_and_reg(
    ctx: &Context,
    ty: pliron::r#type::TypeHandle,
) -> Option<(
    &'static str,
    &'static str,
    Option<pliron::r#type::TypeHandle>,
)> {
    let staging = |width: u32| -> pliron::r#type::TypeHandle {
        IntegerType::get(ctx, width, Signedness::Signless).into()
    };
    let ty_ref = ty.deref(ctx);
    if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        return match int_ty.width() {
            16 => Some(("b16", "h", None)),
            32 => Some(("b32", "r", None)),
            64 => Some(("b64", "l", None)),
            _ => None,
        };
    }
    // Generic pointers use 64-bit `l` registers on nvptx64. Address-space-specific
    // pointers are rejected because their representation may differ from the
    // generic pointer representation.
    // Keep the LLVM pointer type intact. The legacy legalizer likewise carries
    // typed pointers through its inline-PTX exchange and compare-exchange.
    if let Some(ptr_ty) = ty_ref.downcast_ref::<llvm_types::PointerType>() {
        return (ptr_ty.address_space() == llvm_types::address_space::GENERIC)
            .then_some(("b64", "l", None));
    }
    if ty_ref.is::<llvm_types::HalfType>() {
        return Some(("b16", "h", Some(staging(16))));
    }
    if ty_ref.is::<FP32Type>() {
        return Some(("b32", "r", Some(staging(32))));
    }
    if ty_ref.is::<FP64Type>() {
        return Some(("b64", "l", Some(staging(64))));
    }
    None
}

/// PTX scope qualifier.
fn ptx_scope(scope: &NvvmScope) -> &'static str {
    match scope {
        NvvmScope::Device => "gpu",
        NvvmScope::Block => "cta",
        NvvmScope::System => "sys",
    }
}

/// Inline-PTX template for an atomic load.
///
/// PTX has no sequentially consistent load instruction. libcu++ maps a SeqCst
/// load to `fence.sc.{scope}` followed by an acquire load at the same scope;
/// the same mapping is emitted here, fused into a single asm template so the
/// fence can never be separated from the access.
fn ptx_load_template(ord: &NvvmOrdering, scope: &str, ptx_ty: &str) -> Result<String> {
    match ord {
        NvvmOrdering::Relaxed => Ok(format!("ld.relaxed.{scope}.{ptx_ty} $0, [$1];")),
        NvvmOrdering::Acquire => Ok(format!("ld.acquire.{scope}.{ptx_ty} $0, [$1];")),
        NvvmOrdering::SeqCst => Ok(format!(
            "fence.sc.{scope}; ld.acquire.{scope}.{ptx_ty} $0, [$1];"
        )),
        other => pliron::input_err_noloc!(
            "atomic load cannot have {:?} ordering; use Relaxed, Acquire or SeqCst",
            other
        ),
    }
}

/// Inline-PTX template for an atomic store.
///
/// SeqCst mirrors the load mapping (libcu++'s): `fence.sc.{scope}` followed
/// by a release store at the same scope, fused into one template.
fn ptx_store_template(ord: &NvvmOrdering, scope: &str, ptx_ty: &str) -> Result<String> {
    match ord {
        NvvmOrdering::Relaxed => Ok(format!("st.relaxed.{scope}.{ptx_ty} [$0], $1;")),
        NvvmOrdering::Release => Ok(format!("st.release.{scope}.{ptx_ty} [$0], $1;")),
        NvvmOrdering::SeqCst => Ok(format!(
            "fence.sc.{scope}; st.release.{scope}.{ptx_ty} [$0], $1;"
        )),
        other => pliron::input_err_noloc!(
            "atomic store cannot have {:?} ordering; use Relaxed, Release or SeqCst",
            other
        ),
    }
}

/// Emit a memory fence.
///
/// libNVVM rejects the LLVM `fence` instruction outright:
///
/// ```text
/// context:   fence syncscope("block") release
///   Illegal instruction: fence
/// ```
///
/// so every AcqRel or SeqCst atomic was unbuildable under `--materialize-cubin`,
/// including the ones in the shipped `atomics` example.
///
/// Two routes, chosen by what the ordering actually needs.
///
/// **SeqCst goes through the typed NVVM intrinsic.** PTX defines `membar.level`
/// as a synonym for `fence.sc.level`, so `llvm.nvvm.membar.{cta,gl,sys}` is an
/// exact match, and it is the route the rest of this crate already uses for
/// fences: `cuda_device::fence::threadfence` is documented as lowering to
/// `llvm.nvvm.membar.gl`. Going through the intrinsic keeps the fence
/// something LLVM can reason about rather than opaque assembly.
///
/// **Acquire, Release and AcqRel go through inline PTX**, because no intrinsic
/// for them exists. The catalog carries only the three `membar` scopes plus the
/// special-purpose `fence.proxy` and `fence.mbarrier_init` forms. Emitting
/// `membar` for an AcqRel fence would be correct but would silently upgrade the
/// caller's request to sequential consistency, so `fence.acq_rel.{scope}` is
/// emitted directly instead. PTX has no separate acquire or release fence;
/// `fence.acq_rel` is the primitive both lower to.
fn emit_fence(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    ordering: LlvmAtomicOrdering,
    syncscope: &NvvmScope,
) -> Result<()> {
    let scope = match syncscope {
        NvvmScope::Device => "gl",
        NvvmScope::Block => "cta",
        NvvmScope::System => "sys",
    };

    if matches!(ordering, LlvmAtomicOrdering::SeqCst) {
        let void_ty = llvm_types::VoidType::get(ctx);
        let func_ty = llvm_types::FuncType::get(ctx, void_ty.into(), vec![], false);
        common::call_intrinsic(
            ctx,
            rewriter,
            op,
            &format!("llvm_nvvm_membar_{scope}"),
            func_ty,
            vec![],
        )?;
        return Ok(());
    }

    // Acquire, Release and AcqRel all lower to the same PTX fence.
    let ptx_scope = ptx_scope(syncscope);
    let void_ty = llvm_types::VoidType::get(ctx);
    let asm = llvm::InlineAsmOp::build(
        ctx,
        void_ty.into(),
        vec![],
        &format!("fence.acq_rel.{ptx_scope};"),
        "~{memory}",
        AsmKind::SideEffect,
    );
    rewriter.insert_operation(ctx, asm.get_operation());
    Ok(())
}

/// Insert an instruction while preserving the source atomic's location.
fn insert_atomic_instruction(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    source: Ptr<Operation>,
    instruction: Ptr<Operation>,
) {
    crate::convert::preserve_location(ctx, source, instruction);
    rewriter.insert_operation(ctx, instruction);
}

fn cast_storage(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    source: Ptr<Operation>,
    pointer: Value,
    address_space: u32,
) -> Value {
    let target = llvm_types::PointerType::get(ctx, address_space).into();
    if pointer.get_type(ctx) == target {
        return pointer;
    }
    let cast = llvm::AddrSpaceCastOp::new(ctx, pointer, target).get_operation();
    insert_atomic_instruction(ctx, rewriter, source, cast);
    cast.deref(ctx).get_result(0)
}

/// Select the storage semantics from the address space, not the source program's
/// allocation shape. A generic pointer may arrive through calls, selects or
/// block arguments, so its actual state space must be tested at run time.
///
/// PTX local storage is private to the executing thread. Its atomic operations
/// can therefore use ordinary typed memory operations (the same transformation
/// used by LLVM's LowerAtomic/NVPTXAtomicLower). They cannot synchronize with
/// another thread. Nonlocal paths retain the requested scopes and orderings.
///
/// The nonlocal arm keeps its original pointer address space: generic pointers
/// can address global, CTA-shared or cluster-shared storage. Reclassifying
/// everything outside the CTA shared window as global would break distributed
/// shared memory. The typed isspacep intrinsic lets optimization eliminate the
/// dispatch when the address space is known after inlining.
fn lower_for_storage(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    source: Ptr<Operation>,
    pointer: Value,
    mut emit: impl FnMut(
        &mut Context,
        &mut DialectConversionRewriter,
        Value,
        bool,
    ) -> Result<Vec<Value>>,
) -> Result<()> {
    let pointer_ty = pointer.get_type(ctx);
    let address_space = pointer_ty
        .deref(ctx)
        .downcast_ref::<llvm_types::PointerType>()
        .ok_or_else(|| pliron::input_error_noloc!("atomic storage must be a pointer"))?
        .address_space();
    if !matches!(address_space, 0 | 1 | 3 | 5 | 7) {
        return pliron::input_err_noloc!(
            "atomic storage address space {} is unsupported; expected generic, global, shared, local or cluster-shared",
            address_space
        );
    }
    if address_space != llvm_types::address_space::GENERIC {
        let values = emit(ctx, rewriter, pointer, address_space == 5)?;
        rewriter.replace_operation_with_values(ctx, source, values);
        return Ok(());
    }

    let block = source
        .deref(ctx)
        .get_parent_block()
        .expect("atomic has a block");
    let continuation =
        rewriter.split_block(ctx, block, OpInsertionPoint::BeforeOperation(source), None);
    let result_types: Vec<_> = source
        .deref(ctx)
        .results()
        .map(|v| v.get_type(ctx))
        .collect();
    for result_type in result_types {
        let result_type = convert_type(ctx, result_type)
            .map_err(|error| pliron::input_error_noloc!("{}", error))?;
        BasicBlock::push_argument(continuation, ctx, result_type);
    }
    let local = rewriter.create_block(
        ctx,
        BlockInsertionPoint::BeforeBlock(continuation),
        None,
        vec![],
    );
    let nonlocal = rewriter.create_block(
        ctx,
        BlockInsertionPoint::BeforeBlock(continuation),
        None,
        vec![],
    );
    let boolean = IntegerType::get(ctx, 1, Signedness::Signless);
    let query_type = llvm_types::FuncType::get(ctx, boolean.into(), vec![pointer_ty], false);
    rewriter.set_insertion_point_to_block_end(block);
    let query = common::call_intrinsic(
        ctx,
        rewriter,
        source,
        "llvm_nvvm_isspacep_local",
        query_type,
        vec![pointer],
    )?;
    let condition = query.deref(ctx).get_result(0);
    let branch =
        llvm::CondBrOp::new(ctx, condition, local, vec![], nonlocal, vec![]).get_operation();
    insert_atomic_instruction(ctx, rewriter, source, branch);

    for (path, private) in [(local, true), (nonlocal, false)] {
        rewriter.set_insertion_point_to_block_end(path);
        let storage = if private {
            cast_storage(ctx, rewriter, source, pointer, 5)
        } else {
            pointer
        };
        let values = emit(ctx, rewriter, storage, private)?;
        let branch = llvm::BrOp::new(ctx, continuation, values).get_operation();
        insert_atomic_instruction(ctx, rewriter, source, branch);
    }
    let values = continuation.deref(ctx).arguments().collect();
    rewriter.set_insertion_point_before_operation(source);
    rewriter.replace_operation_with_values(ctx, source, values);
    Ok(())
}

fn private_load(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    source: Ptr<Operation>,
    pointer: Value,
    value_type: TypeHandle,
) -> Value {
    let load = llvm::LoadOp::new(ctx, pointer, value_type).get_operation();
    insert_atomic_instruction(ctx, rewriter, source, load);
    load.deref(ctx).get_result(0)
}

fn private_store(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    source: Ptr<Operation>,
    pointer: Value,
    value: Value,
) {
    let store = llvm::StoreOp::new(ctx, value, pointer).get_operation();
    insert_atomic_instruction(ctx, rewriter, source, store);
}

/// Ordinary operations implement a private RMW without dropping any value
/// semantics: integer arithmetic wraps, extrema retain signedness, and floats
/// use ordinary fadd without introducing fast-math assumptions.
fn private_rmw(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    source: Ptr<Operation>,
    pointer: Value,
    value: Value,
    kind: &NvvmRmwKind,
) -> Value {
    let old = private_load(ctx, rewriter, source, pointer, value.get_type(ctx));
    let updated = match kind {
        NvvmRmwKind::Xchg => value,
        NvvmRmwKind::Min | NvvmRmwKind::Max | NvvmRmwKind::UMin | NvvmRmwKind::UMax => {
            let predicate = match kind {
                NvvmRmwKind::Min => ICmpPredicateAttr::SLT,
                NvvmRmwKind::Max => ICmpPredicateAttr::SGT,
                NvvmRmwKind::UMin => ICmpPredicateAttr::ULT,
                NvvmRmwKind::UMax => ICmpPredicateAttr::UGT,
                _ => unreachable!(),
            };
            let compare = llvm::ICmpOp::new(ctx, predicate, old, value).get_operation();
            insert_atomic_instruction(ctx, rewriter, source, compare);
            let condition = compare.deref(ctx).get_result(0);
            let selected = llvm::SelectOp::new(ctx, condition, old, value).get_operation();
            insert_atomic_instruction(ctx, rewriter, source, selected);
            selected.deref(ctx).get_result(0)
        }
        _ => {
            let arithmetic = match kind {
                NvvmRmwKind::Add => {
                    llvm::AddOp::new_with_overflow_flag(ctx, old, value, Default::default())
                        .get_operation()
                }
                NvvmRmwKind::Sub => {
                    llvm::SubOp::new_with_overflow_flag(ctx, old, value, Default::default())
                        .get_operation()
                }
                NvvmRmwKind::And => llvm::AndOp::new(ctx, old, value).get_operation(),
                NvvmRmwKind::Or => llvm::OrOp::new(ctx, old, value).get_operation(),
                NvvmRmwKind::Xor => llvm::XorOp::new(ctx, old, value).get_operation(),
                NvvmRmwKind::FAdd => {
                    llvm::FAddOp::new_with_fast_math_flags(ctx, old, value, Default::default())
                        .get_operation()
                }
                _ => unreachable!(),
            };
            insert_atomic_instruction(ctx, rewriter, source, arithmetic);
            arithmetic.deref(ctx).get_result(0)
        }
    };
    private_store(ctx, rewriter, source, pointer, updated);
    old
}

// =============================================================================
// Fence
// =============================================================================

pub(crate) fn convert_atomic_fence(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let fence = NvvmAtomicFenceOp::new(op);
    let ordering = fence.ordering(ctx);
    if matches!(ordering, NvvmOrdering::Relaxed) {
        return pliron::input_err_noloc!("atomic fence cannot use Relaxed ordering");
    }

    emit_fence(
        ctx,
        rewriter,
        op,
        map_ordering(&ordering),
        &fence.scope(ctx),
    )?;
    rewriter.erase_operation(ctx, op);
    Ok(())
}

// =============================================================================
// Load
// =============================================================================

pub(crate) fn convert_atomic_load(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let nvvm_op = NvvmAtomicLoadOp::new(op);
    let ordering = nvvm_op.ordering(ctx);
    let scope = ptx_scope(&nvvm_op.scope(ctx));
    let pointer = op.deref(ctx).get_operand(0);
    let result_type = op.deref(ctx).get_result(0).get_type(ctx);
    let result_type =
        convert_type(ctx, result_type).map_err(|error| pliron::input_error_noloc!("{}", error))?;
    let (ptx_type, register, staging_type) = ptx_type_and_reg(ctx, result_type)
        .ok_or_else(|| pliron::input_error_noloc!("atomic load of unsupported operand type"))?;
    let template = ptx_load_template(&ordering, scope, ptx_type)?;
    lower_for_storage(
        ctx,
        rewriter,
        op,
        pointer,
        |ctx, rewriter, storage, private| {
            if private {
                return Ok(vec![private_load(ctx, rewriter, op, storage, result_type)]);
            }
            // Qualified generic loads are valid only for global/shared storage. The
            // dispatch proves this before converting the typed address back to the
            // generic 64-bit register representation used by the PTX template.
            let storage = cast_storage(ctx, rewriter, op, storage, 0);
            let assembly = llvm::InlineAsmOp::build(
                ctx,
                staging_type.unwrap_or(result_type),
                vec![storage],
                &template,
                &format!("={register},l,~{{memory}}"),
                AsmKind::SideEffect,
            )
            .get_operation();
            insert_atomic_instruction(ctx, rewriter, op, assembly);
            let value = assembly.deref(ctx).get_result(0);
            if staging_type.is_some() {
                let bitcast = llvm::BitcastOp::new(ctx, value, result_type).get_operation();
                insert_atomic_instruction(ctx, rewriter, op, bitcast);
                Ok(vec![bitcast.deref(ctx).get_result(0)])
            } else {
                Ok(vec![value])
            }
        },
    )
}

// =============================================================================
// Store
// =============================================================================

pub(crate) fn convert_atomic_store(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let nvvm_op = NvvmAtomicStoreOp::new(op);
    let ordering = nvvm_op.ordering(ctx);
    let scope = ptx_scope(&nvvm_op.scope(ctx));
    let value = op.deref(ctx).get_operand(0);
    let pointer = op.deref(ctx).get_operand(1);
    let (ptx_type, register, staging_type) = ptx_type_and_reg(ctx, value.get_type(ctx))
        .ok_or_else(|| pliron::input_error_noloc!("atomic store of unsupported operand type"))?;
    let template = ptx_store_template(&ordering, scope, ptx_type)?;
    lower_for_storage(
        ctx,
        rewriter,
        op,
        pointer,
        |ctx, rewriter, storage, private| {
            if private {
                private_store(ctx, rewriter, op, storage, value);
                return Ok(vec![]);
            }
            let storage = cast_storage(ctx, rewriter, op, storage, 0);
            let value = match staging_type {
                Some(integer_type) => {
                    let bitcast = llvm::BitcastOp::new(ctx, value, integer_type).get_operation();
                    insert_atomic_instruction(ctx, rewriter, op, bitcast);
                    bitcast.deref(ctx).get_result(0)
                }
                None => value,
            };
            let void = llvm_types::VoidType::get(ctx);
            let assembly = llvm::InlineAsmOp::build(
                ctx,
                void.into(),
                vec![storage, value],
                &template,
                &format!("l,{register},~{{memory}}"),
                AsmKind::SideEffect,
            )
            .get_operation();
            insert_atomic_instruction(ctx, rewriter, op, assembly);
            Ok(vec![])
        },
    )
}

// =============================================================================
// Read-Modify-Write (with fence splitting workaround)
// =============================================================================

pub(crate) fn convert_atomic_rmw(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let nvvm_op = NvvmAtomicRmwOp::new(op);
    let ordering = nvvm_op.ordering(ctx);
    let scope = nvvm_op.scope(ctx);
    let kind = nvvm_op.rmw_kind(ctx);
    let pointer = op.deref(ctx).get_operand(0);
    let value = op.deref(ctx).get_operand(1);
    lower_for_storage(
        ctx,
        rewriter,
        op,
        pointer,
        |ctx, rewriter, storage, private| {
            if private {
                return Ok(vec![private_rmw(ctx, rewriter, op, storage, value, &kind)]);
            }
            match ordering {
                NvvmOrdering::Release | NvvmOrdering::AcqRel => {
                    emit_fence(ctx, rewriter, op, LlvmAtomicOrdering::Release, &scope)?;
                }
                NvvmOrdering::SeqCst => {
                    emit_fence(ctx, rewriter, op, LlvmAtomicOrdering::SeqCst, &scope)?;
                }
                NvvmOrdering::Relaxed | NvvmOrdering::Acquire => {}
            }
            let atomic = llvm::AtomicRmwOp::new(
                ctx,
                storage,
                value,
                map_rmw_kind(&kind),
                LlvmAtomicOrdering::Monotonic,
                map_scope(&scope),
            )
            .get_operation();
            insert_atomic_instruction(ctx, rewriter, op, atomic);
            match ordering {
                NvvmOrdering::Acquire | NvvmOrdering::AcqRel => {
                    emit_fence(ctx, rewriter, op, LlvmAtomicOrdering::Acquire, &scope)?;
                }
                NvvmOrdering::SeqCst => {
                    emit_fence(ctx, rewriter, op, LlvmAtomicOrdering::SeqCst, &scope)?;
                }
                NvvmOrdering::Relaxed | NvvmOrdering::Release => {}
            }
            Ok(vec![atomic.deref(ctx).get_result(0)])
        },
    )
}

// =============================================================================
// Compare-and-Exchange
// =============================================================================

pub(crate) fn convert_atomic_cmpxchg(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let nvvm_op = NvvmAtomicCmpxchgOp::new(op);
    let success_ordering = map_ordering(&nvvm_op.success_ordering(ctx));
    let failure_ordering = map_ordering(&nvvm_op.failure_ordering(ctx));
    let scope = nvvm_op.scope(ctx);
    let pointer = op.deref(ctx).get_operand(0);
    let expected = op.deref(ctx).get_operand(1);
    let replacement = op.deref(ctx).get_operand(2);
    lower_for_storage(
        ctx,
        rewriter,
        op,
        pointer,
        |ctx, rewriter, storage, private| {
            if private {
                let old = private_load(ctx, rewriter, op, storage, expected.get_type(ctx));
                let compare =
                    llvm::ICmpOp::new(ctx, ICmpPredicateAttr::EQ, old, expected).get_operation();
                insert_atomic_instruction(ctx, rewriter, op, compare);
                // Like LLVM LowerAtomic: in private storage, rewriting the old value
                // on failure has no competing observer and preserves the value.
                let condition = compare.deref(ctx).get_result(0);
                let selected =
                    llvm::SelectOp::new(ctx, condition, replacement, old).get_operation();
                insert_atomic_instruction(ctx, rewriter, op, selected);
                let selected_value = selected.deref(ctx).get_result(0);
                private_store(ctx, rewriter, op, storage, selected_value);
                return Ok(vec![old]);
            }
            let atomic = llvm::AtomicCmpxchgOp::new(
                ctx,
                storage,
                expected,
                replacement,
                success_ordering.clone(),
                failure_ordering.clone(),
                map_scope(&scope),
            )
            .get_operation();
            insert_atomic_instruction(ctx, rewriter, op, atomic);
            let atomic_result = atomic.deref(ctx).get_result(0);
            let extract = llvm::ExtractValueOp::new(ctx, atomic_result, vec![0])
                .map_err(|error| pliron::input_error_noloc!("{}", error))?
                .get_operation();
            insert_atomic_instruction(ctx, rewriter, op, extract);
            Ok(vec![extract.deref(ctx).get_result(0)])
        },
    )
}

// =============================================================================
// Packed Atomic Add (f16x2, bf16x2) -- inline PTX
// =============================================================================

/// Convert a packed atomic add op to inline PTX.
///
/// Constraints: `=r,l,r,~{memory}` -- output register, address pointer, input
/// register, memory clobber.
///
/// Uses `SideEffect` (not convergent): atomics are per-thread, not
/// warp-synchronous.
pub(crate) fn convert_packed_atom_add(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    ptx_type: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!(
            "packed atomic add requires 2 operands (address, addend), got {}",
            operands.len()
        );
    }
    let addr = operands[0];
    let val = operands[1];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        i32_ty.into(),
        vec![addr, val],
        &format!("atom.global.add.noftz.{ptx_type} $0, [$1], $2;"),
        "=r,l,r,~{memory}",
        AsmKind::SideEffect,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}
