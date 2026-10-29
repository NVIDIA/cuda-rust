/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Call operation conversion: `dialect-mir` → LLVM dialect.
//!
//! Handles function call lowering with ABI-level transformations:
//! - Slice arguments flattened to (ptr, len) pairs
//! - Struct arguments flattened to individual fields
//! - Unit return type becomes void
//! - Pointer arguments cast to generic address space (for ABI compatibility)
//!
//! These transformations match the function signature flattening done in
//! `convert_function_type`.
//!
//! # Device Extern Symbol Resolution
//!
//! When calling device extern functions (declared with `#[device] extern "C"`),
//! the MIR contains calls to prefixed symbols like `cuda_oxide_device_extern_foo`.
//! This prefix is added by the proc-macro for internal detection. However, the
//! external LTOIR (e.g., CCCL libraries) exports the original symbol name `foo`.
//!
//! We strip the prefix during lowering so the LLVM IR emits:
//! ```llvm
//! call @foo(...)  ; NOT @cuda_oxide_device_extern_foo
//! ```
//!
//! This allows nvJitLink to resolve the symbol against the external LTOIR.
//!
//! # Address Space Handling
//!
//! The call op's `func_type` must match the callee's declared signature
//! exactly, including the address space of every pointer parameter. We
//! achieve this by:
//!
//!   1. Looking up the callee's `llvm::FuncOp` declaration in the parent
//!      module, before flattening the arguments.
//!   2. Coercing each (post-flatten) argument to the corresponding declared
//!      parameter type — when the argument is a pointer in a different
//!      address space than the parameter, we insert `llvm.addrspacecast`
//!      to bridge them.
//!   3. Building the call op's `func_type` directly from the looked-up
//!      declaration, so a tightening of the declaration automatically
//!      flows to every call site.
//!
//! This single mechanism handles both directions of the addrspace dance:
//!
//!   * Path 1 (e.g. `block_reduce(*mut SharedArray<T,N>)`): callee param
//!     is `ptr addrspace(3)`, caller has `ptr addrspace(3)` → no cast.
//!     A caller that legitimately had a generic pointer would be cast UP
//!     to addrspace(3).
//!   * Path 3 (e.g. `extern "C" fn cublasdx_gemm(_: *mut i8)`): callee
//!     param is `ptr addrspace(0)`, caller has `ptr addrspace(3)` (from
//!     `DynamicSharedArray::get`) → cast DOWN to addrspace(0).
//!
//! The previous implementation always cast pointer arguments to addrspace(0)
//! regardless of the callee's declared signature. That worked for Path 3
//! but actively broke Path 1, because the resulting call op carried a
//! `func_type` whose pointer parameter types were `addrspace(0)` while the
//! callee declaration carried `addrspace(3)`. The verifier rejected the
//! mismatch.

use crate::convert::target_stable_storage::coerce_target_stable_value;
use crate::convert::types::{
    StructLayoutInfo, TransparentScalarAbiInfo, build_struct_slot_map, convert_function_type,
    convert_type, is_kernel_func, is_zero_sized_type, shared_return_abi_info,
    transparent_scalar_abi_info,
};
use crate::helpers;
use dialect_mir::ops::{MirCallOp, MirFuncOp};
use dialect_mir::rust_intrinsics::{self, RustFloatMathIntrinsic};
use dialect_mir::types::{MirDisjointSliceType, MirSliceType, MirStructType, MirTupleType};
use llvm_export::attributes::IntegerOverflowFlagsAttr;
use llvm_export::op_interfaces::{
    BinArithOp, CastOpInterface, CastOpWithNNegInterface, IntBinArithOpWithOverflowFlag,
};
use llvm_export::ops as llvm;
use llvm_export::types as llvm_types;
use llvm_export::types::PointerTypeExt;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::op_interfaces::{CallOpCallable, SymbolOpInterface};
use pliron::builtin::type_interfaces::FunctionTypeInterface;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};
use pliron::utils::apint::APInt;
use pliron::value::Value;
use std::num::NonZeroUsize;

/// Generic address space (can alias any memory).
const ADDRSPACE_GENERIC: u32 = 0;

// The `#[device] extern "C"` macro renames a foreign function `foo` to
// `cuda_oxide_device_extern_<hash>_foo` so the collector can find it. During
// MIR lowering we strip that prefix back off so the LLVM IR / LTOIR refers to
// the original symbol the user wrote. The prefix string itself, plus the
// `device_extern_base_name` extractor, lives in `reserved-oxide-symbols`.
use reserved_oxide_symbols::device_extern_base_name;

/// Internal placeholder for rustc bit intrinsics that need LLVM intrinsic calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RustBitIntrinsic {
    RotateLeft,
    RotateRight,
    Ctpop,
    Ctlz { zero_undef: bool },
    Cttz { zero_undef: bool },
    Bswap,
    Bitreverse,
}

impl RustBitIntrinsic {
    /// Convert an importer placeholder name back into the intrinsic it represents.
    fn from_placeholder_callee(callee: &str) -> Option<Self> {
        match callee {
            rust_intrinsics::CALLEE_ROTATE_LEFT => Some(Self::RotateLeft),
            rust_intrinsics::CALLEE_ROTATE_RIGHT => Some(Self::RotateRight),
            rust_intrinsics::CALLEE_CTPOP => Some(Self::Ctpop),
            rust_intrinsics::CALLEE_CTLZ => Some(Self::Ctlz { zero_undef: false }),
            rust_intrinsics::CALLEE_CTLZ_NONZERO => Some(Self::Ctlz { zero_undef: true }),
            rust_intrinsics::CALLEE_CTTZ => Some(Self::Cttz { zero_undef: false }),
            rust_intrinsics::CALLEE_CTTZ_NONZERO => Some(Self::Cttz { zero_undef: true }),
            rust_intrinsics::CALLEE_BSWAP => Some(Self::Bswap),
            rust_intrinsics::CALLEE_BITREVERSE => Some(Self::Bitreverse),
            _ => None,
        }
    }
}

/// Internal placeholder for rustc saturating arithmetic intrinsics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RustSaturatingIntrinsic {
    Add,
    Sub,
}

impl RustSaturatingIntrinsic {
    /// Convert an importer placeholder name back into the intrinsic it represents.
    fn from_placeholder_callee(callee: &str) -> Option<Self> {
        match callee {
            rust_intrinsics::CALLEE_SATURATING_ADD => Some(Self::Add),
            rust_intrinsics::CALLEE_SATURATING_SUB => Some(Self::Sub),
            _ => None,
        }
    }
}

/// Internal placeholder for `core::intrinsics::exact_div`.
///
/// The caller has promised the divisor is non-zero and divides the dividend
/// exactly, so this lowers to a plain `sdiv`/`udiv` with no zero check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RustExactDivIntrinsic;

impl RustExactDivIntrinsic {
    /// Convert an importer placeholder name back into the intrinsic it represents.
    fn from_placeholder_callee(callee: &str) -> Option<Self> {
        (callee == rust_intrinsics::CALLEE_EXACT_DIV).then_some(Self)
    }
}

/// Internal placeholder for rustc bigint helper intrinsics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RustBigIntIntrinsic {
    /// `core::intrinsics::carrying_mul_add`: double-width
    /// multiply-accumulate returning a `(low, high)` pair.
    CarryingMulAdd,
}

impl RustBigIntIntrinsic {
    /// Convert an importer placeholder name back into the intrinsic it represents.
    fn from_placeholder_callee(callee: &str) -> Option<Self> {
        match callee {
            rust_intrinsics::CALLEE_CARRYING_MUL_ADD => Some(Self::CarryingMulAdd),
            _ => None,
        }
    }
}

fn anyhow_to_pliron(e: anyhow::Error) -> pliron::result::Error {
    pliron::create_error!(
        pliron::location::Location::Unknown,
        pliron::result::ErrorKind::VerificationFailed,
        pliron::result::StringError(e.to_string())
    )
}

/// Convert `mir.call` to `llvm.call` with argument flattening.
///
/// Performs ABI-level transformations to match CUDA calling conventions:
/// - Slice arguments: flattened to `(ptr, len)` pairs
/// - Struct arguments: flattened to individual fields
/// - Unit return type: converted to void
/// - Callee name: `::` mangled to `__` for LLVM identifier validity
/// - Device extern calls: prefix stripped to use original symbol name
pub fn convert(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let callee_name: String = {
        let mir_call = MirCallOp::new(op);
        let callee_attr = match mir_call.get_attr_callee(ctx) {
            Some(a) => a,
            None => {
                return pliron::input_err!(
                    op.deref(ctx).loc(),
                    "MirCallOp missing callee attribute"
                );
            }
        };
        (*callee_attr).clone().into()
    };

    if let Some(intrinsic) = RustBitIntrinsic::from_placeholder_callee(&callee_name) {
        return convert_rust_bit_intrinsic(ctx, rewriter, op, intrinsic);
    }

    if let Some(intrinsic) = RustSaturatingIntrinsic::from_placeholder_callee(&callee_name) {
        return convert_rust_saturating_intrinsic(ctx, rewriter, op, operands_info, intrinsic);
    }

    if RustExactDivIntrinsic::from_placeholder_callee(&callee_name).is_some() {
        return convert_rust_exact_div(ctx, rewriter, op, operands_info);
    }

    if let Some(RustBigIntIntrinsic::CarryingMulAdd) =
        RustBigIntIntrinsic::from_placeholder_callee(&callee_name)
    {
        return convert_rust_carrying_mul_add(ctx, rewriter, op, operands_info);
    }

    if let Some(intrinsic) = RustFloatMathIntrinsic::from_placeholder_callee(&callee_name) {
        return super::float_math::convert_rust_float_math_intrinsic(ctx, rewriter, op, intrinsic);
    }

    if callee_name == rust_intrinsics::CALLEE_SELECT_UNPREDICTABLE {
        return convert_rust_select_unpredictable(ctx, rewriter, op);
    }

    let callee_ident: pliron::identifier::Identifier = {
        let resolved_name = resolve_device_extern_symbol(&callee_name);

        resolved_name
            .try_into()
            .expect("callee name should have been legalized during MIR import")
    };

    let args: Vec<Value> = op.deref(ctx).operands().collect();

    let has_result = op.deref(ctx).get_num_results() > 0;
    let mir_result_ty_ptr = if has_result {
        Some(op.deref(ctx).get_result(0).get_type(ctx))
    } else {
        None
    };

    let mut zst_replacement_type = None;
    let mut transparent_result_abi = None;
    let mut packed_shared_result_abi = None;
    let result_type = if let Some(mir_ty) = mir_result_ty_ptr {
        // Only the empty tuple `()` is the unit type. `is::<MirTupleType>()`
        // also matches `(T, U, ...)`, so we have to peek at the field count.
        // Non-empty tuples take the convert_type path and end up as an LLVM
        // struct return, same as named structs.
        let is_unit = mir_ty
            .deref(ctx)
            .downcast_ref::<MirTupleType>()
            .is_some_and(|t| t.get_types().is_empty());
        let is_transparent_scalar = {
            let ty_ref = mir_ty.deref(ctx);
            ty_ref
                .downcast_ref::<MirStructType>()
                .is_some_and(MirStructType::is_transparent_scalar)
        };
        let converted = if is_transparent_scalar {
            let abi = transparent_scalar_abi_info(ctx, mir_ty).map_err(anyhow_to_pliron)?;
            let scalar_ty = abi.scalar_ty;
            transparent_result_abi = Some(abi);
            scalar_ty
        } else if let Some(abi) = shared_return_abi_info(ctx, mir_ty).map_err(anyhow_to_pliron)? {
            let storage_ty = abi.storage_ty;
            packed_shared_result_abi = Some(abi);
            storage_ty
        } else {
            convert_type(ctx, mir_ty).map_err(anyhow_to_pliron)?
        };
        if is_unit || is_zero_sized_type(ctx, converted) {
            // NVPTX cannot carry a ZST in a function signature, so the call
            // itself returns void. Keep the converted ZST type so any MIR uses
            // of the result can be replaced with a typed undef below.
            zst_replacement_type = Some(converted);
            llvm_types::VoidType::get(ctx).into()
        } else {
            converted
        }
    } else {
        llvm_types::VoidType::get(ctx).into()
    };

    // Look up the callee's declared signature in the parent module. The
    // declaration may already have been lowered to `llvm::FuncOp` (typical
    // for device-extern decls inserted by the importer ahead of time, and
    // for callees whose conversion ran first), or it may still be a
    // `MirFuncOp` whose body hasn't been touched yet. In the second case
    // we run the same MIR-to-LLVM signature flattening that the function
    // converter will eventually apply, so the call site sees the exact
    // same parameter types the callee will be lowered to. This makes
    // intra-Rust calls into shared-memory-typed params and device-extern
    // calls with the generic ABI work uniformly.
    let callee_decl_arg_types = find_callee_arg_types(ctx, op, &callee_ident).unwrap_or_default();
    let expected_param_tys = if callee_decl_arg_types.is_empty() {
        None
    } else {
        Some(callee_decl_arg_types.as_slice())
    };

    let (flattened_args, flattened_arg_types) =
        flatten_arguments(ctx, rewriter, &args, operands_info, expected_param_tys)?;

    let func_type = llvm_types::FuncType::get(ctx, result_type, flattened_arg_types, false);

    // Direct libdevice calls (`__nv_*`) originate from hand-written externs
    // (e.g. khal_std's `Float::asin`/`acos` → `__nv_asinf`/`__nv_acosf`) that
    // have no MIR body and therefore no emitted declaration. Without a module
    // declaration the verifier rejects the call ("Symbol __nv_asinf not found").
    // Declare it here with the call's signature, mirroring the float-math
    // intrinsic path; the symbol is bound against libdevice at the cubin-link
    // stage just like `__nv_expf`/`__nv_sqrtf`.
    {
        let resolved_name = resolve_device_extern_symbol(&callee_name);
        let parent_block = op.deref(ctx).get_parent_block();
        if resolved_name.starts_with("__nv_")
            && let Some(parent_block) = parent_block
        {
            let loc = op.deref(ctx).loc().clone();
            helpers::ensure_intrinsic_declared(ctx, parent_block, &resolved_name, func_type)
                .map_err(|e| {
                    pliron::input_error!(
                        loc,
                        "Failed to declare libdevice extern {resolved_name}: {e}"
                    )
                })?;
        }
    }

    let llvm_call = llvm::CallOp::new(
        ctx,
        CallOpCallable::Direct(callee_ident),
        func_type,
        flattened_args,
    );
    crate::convert::preserve_location(ctx, op, llvm_call.get_operation());
    rewriter.insert_operation(ctx, llvm_call.get_operation());

    let is_void = result_type.deref(ctx).is::<llvm_types::VoidType>();
    if has_result && !is_void && llvm_call.get_operation().deref(ctx).get_num_results() > 0 {
        if let Some(abi) = transparent_result_abi {
            // The ABI call returns only the underlying scalar, but MIR users
            // still operate on the ordinary converted wrapper aggregate.
            // Rebuild nested transparent layers from inner to outer so the
            // replacement result has exactly the MIR op's converted type.
            let mut value = llvm_call.get_operation().deref(ctx).get_result(0);
            let mut replacement = llvm_call.get_operation();
            for layer in abi.layers.iter().rev() {
                let undef = llvm::UndefOp::new(ctx, layer.llvm_struct_ty);
                rewriter.insert_operation(ctx, undef.get_operation());
                let aggregate = undef.get_operation().deref(ctx).get_result(0);
                let insert =
                    llvm::InsertValueOp::new(ctx, aggregate, value, vec![layer.field_slot]);
                rewriter.insert_operation(ctx, insert.get_operation());
                value = insert.get_operation().deref(ctx).get_result(0);
                replacement = insert.get_operation();
            }
            rewriter.replace_operation(ctx, op, replacement);
        } else if let Some(abi) = packed_shared_result_abi {
            let value = llvm_call.get_operation().deref(ctx).get_result(0);
            let semantic = coerce_target_stable_value(
                ctx,
                rewriter,
                value,
                abi.semantic_ty,
                "packed shared internal ABI",
            )?;
            rewriter.replace_operation_with_values(ctx, op, vec![semantic]);
        } else {
            rewriter.replace_operation(ctx, op, llvm_call.get_operation());
        }
    } else if op.deref(ctx).has_use()
        && let Some(zst_type) = zst_replacement_type
    {
        // The ABI intentionally erases ZST returns, but MIR may still use the
        // typed result (return it again, pass it to another call, or project a
        // zero-sized field). Preserve the side-effecting void call and replace
        // only its value result with an undef of the converted ZST type.
        let undef = llvm::UndefOp::new(ctx, zst_type);
        rewriter.insert_operation(ctx, undef.get_operation());
        rewriter.replace_operation(ctx, op, undef.get_operation());
    } else {
        // The LLVM call has no usable result so the MIR op must be erased.
        // This is safe for a result-less call or an unused ZST result. Any
        // other live result means the result-type computation silently
        // dropped a real value; report that instead of letting pliron's
        // erase-with-uses invariant panic and escape into rustc as an ICE.
        if op.deref(ctx).has_use() {
            let loc = op.deref(ctx).loc().clone();
            return pliron::input_err!(
                loc,
                "mir.call lowering produced a void LLVM call but the MIR op \
                 still has live uses; the return type was likely misclassified \
                 (for example, a non-unit tuple treated as `()`)"
            );
        }
        rewriter.erase_operation(ctx, op);
    }

    Ok(())
}

/// Lower the placeholder call for `select_unpredictable` to an LLVM `select`.
///
/// The importer emits `select_unpredictable(cond, true_val, false_val)` as a
/// placeholder `mir.call` with exactly three operands. By this point the
/// operands carry their converted LLVM types, so the `select` is built
/// directly: its result type comes from `true_val` and matches both
/// `false_val` and the intrinsic's return type. The Rust `bool` condition
/// lowers to `i1`, which is what `SelectOp` requires. The `unpredictable`
/// branch-weight hint is an optimization hint with no device semantics and
/// is intentionally not represented.
fn convert_rust_select_unpredictable(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
) -> Result<()> {
    let loc = op.deref(ctx).loc().clone();
    if op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(loc, "select_unpredictable call must have one result");
    }
    let args: Vec<Value> = op.deref(ctx).operands().collect();
    let [cond, true_val, false_val] = args.as_slice() else {
        return pliron::input_err!(
            loc,
            "select_unpredictable expects exactly three operands (cond, true, false)"
        );
    };
    let select = llvm::SelectOp::new(ctx, *cond, *true_val, *false_val);
    rewriter.insert_operation(ctx, select.get_operation());
    rewriter.replace_operation(ctx, op, select.get_operation());
    Ok(())
}

/// Lower placeholder calls for rustc's integer bit intrinsics to LLVM intrinsics.
///
/// Rust methods like `u128::rotate_left` call `core::intrinsics::rotate_left`
/// in libcore. The importer preserves that as a placeholder `mir.call`; here we
/// recover the concrete integer width and emit the corresponding overloaded
/// LLVM intrinsic.
fn convert_rust_bit_intrinsic(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    intrinsic: RustBitIntrinsic,
) -> Result<()> {
    let loc = op.deref(ctx).loc().clone();
    if op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(loc, "Rust bit intrinsic call must have one result");
    }

    let args: Vec<Value> = op.deref(ctx).operands().collect();
    let Some(&value) = args.first() else {
        return pliron::input_err!(loc, "Rust bit intrinsic call missing integer operand");
    };

    let value_ty = value.get_type(ctx);
    let value_width = integer_bit_width(ctx, value_ty, loc.clone())?;
    let mir_result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let result_type = convert_type(ctx, mir_result_ty).map_err(anyhow_to_pliron)?;

    if matches!(intrinsic, RustBitIntrinsic::Bswap) && value_width == 8 {
        // LLVM has no useful byte swap for a single byte; Rust's semantics are identity.
        //
        // The result type is checked first. `bitcast` requires both types to be
        // non-aggregate and of one size, so a result that is not an 8-bit
        // integer would leave this arm emitting IR that LLVM rejects. Every
        // other arm reaches `cast_integer_value_to_type` below, which performs
        // the same check; this one returns early and would otherwise skip it.
        let result_width = integer_bit_width(ctx, result_type, loc.clone())?;
        if result_width != value_width {
            return pliron::input_err!(
                loc,
                "bswap on an {value_width}-bit integer cannot produce a \
                 {result_width}-bit result"
            );
        }
        let bitcast = llvm::BitcastOp::new(ctx, value, result_type);
        rewriter.insert_operation(ctx, bitcast.get_operation());
        rewriter.replace_operation(ctx, op, bitcast.get_operation());
        return Ok(());
    }

    let (intrinsic_name, intrinsic_args, intrinsic_result_ty) = match intrinsic {
        RustBitIntrinsic::RotateLeft | RustBitIntrinsic::RotateRight => {
            if args.len() != 2 {
                return pliron::input_err!(
                    loc,
                    "rotate intrinsic requires value and shift operands"
                );
            }
            let (shift, _) =
                cast_integer_value_to_type(ctx, rewriter, args[1], value_ty, loc.clone())?;
            let suffix = match intrinsic {
                RustBitIntrinsic::RotateLeft => "fshl",
                RustBitIntrinsic::RotateRight => "fshr",
                _ => unreachable!(),
            };
            (
                format!("llvm_{suffix}_i{value_width}"),
                vec![value, value, shift],
                value_ty,
            )
        }
        RustBitIntrinsic::Ctpop => (format!("llvm_ctpop_i{value_width}"), vec![value], value_ty),
        RustBitIntrinsic::Ctlz { zero_undef } => {
            let zero_undef = create_i1_constant(ctx, rewriter, zero_undef);
            (
                format!("llvm_ctlz_i{value_width}"),
                vec![value, zero_undef],
                value_ty,
            )
        }
        RustBitIntrinsic::Cttz { zero_undef } => {
            let zero_undef = create_i1_constant(ctx, rewriter, zero_undef);
            (
                format!("llvm_cttz_i{value_width}"),
                vec![value, zero_undef],
                value_ty,
            )
        }
        RustBitIntrinsic::Bswap => (format!("llvm_bswap_i{value_width}"), vec![value], value_ty),
        RustBitIntrinsic::Bitreverse => (
            format!("llvm_bitreverse_i{value_width}"),
            vec![value],
            value_ty,
        ),
    };

    let arg_types = intrinsic_args
        .iter()
        .map(|arg| arg.get_type(ctx))
        .collect::<Vec<_>>();
    let func_ty = llvm_types::FuncType::get(ctx, intrinsic_result_ty, arg_types, false);
    let parent_block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(loc.clone(), "Rust bit intrinsic call has no parent block")
    })?;
    helpers::ensure_intrinsic_declared(ctx, parent_block, &intrinsic_name, func_ty)
        .map_err(|e| pliron::input_error!(loc.clone(), "Failed to declare intrinsic: {e}"))?;

    let sym_name: pliron::identifier::Identifier = intrinsic_name
        .as_str()
        .try_into()
        .map_err(|e| pliron::input_error!(loc.clone(), "Invalid intrinsic name: {:?}", e))?;
    let llvm_call = llvm::CallOp::new(
        ctx,
        CallOpCallable::Direct(sym_name),
        func_ty,
        intrinsic_args,
    );
    rewriter.insert_operation(ctx, llvm_call.get_operation());

    let call_result = llvm_call.get_operation().deref(ctx).get_result(0);
    let (_, final_op) =
        cast_integer_value_to_type(ctx, rewriter, call_result, result_type, loc.clone())?;
    let replacement = final_op.unwrap_or_else(|| llvm_call.get_operation());
    rewriter.replace_operation(ctx, op, replacement);

    Ok(())
}

/// Lower placeholder calls for rustc's saturating integer intrinsics.
///
/// Rust preserves signedness in the original MIR type. The converted LLVM value
/// is signless, so this uses `operands_info` to choose `sadd/ssub` versus
/// `uadd/usub`.
fn convert_rust_saturating_intrinsic(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
    intrinsic: RustSaturatingIntrinsic,
) -> Result<()> {
    let loc = op.deref(ctx).loc().clone();
    if op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(loc, "Rust saturating intrinsic call must have one result");
    }

    let args: Vec<Value> = op.deref(ctx).operands().collect();
    if args.len() != 2 {
        return pliron::input_err!(
            loc,
            "Rust saturating intrinsic requires left and right operands"
        );
    }

    let lhs = args[0];
    let rhs = args[1];
    let lhs_ty = lhs.get_type(ctx);
    let width = integer_bit_width(ctx, lhs_ty, loc.clone())?;
    let is_signed =
        if let Some(int_ty) = operands_info.lookup_most_recent_of_type::<IntegerType>(ctx, lhs) {
            int_ty.signedness() == Signedness::Signed
        } else {
            return pliron::input_err!(loc, "expected integer type for Rust saturating intrinsic");
        };

    let (rhs, _) = cast_integer_value_to_type(ctx, rewriter, rhs, lhs_ty, loc.clone())?;
    let op_stem = match (is_signed, intrinsic) {
        (true, RustSaturatingIntrinsic::Add) => "sadd",
        (false, RustSaturatingIntrinsic::Add) => "uadd",
        (true, RustSaturatingIntrinsic::Sub) => "ssub",
        (false, RustSaturatingIntrinsic::Sub) => "usub",
    };
    let intrinsic_name = format!("llvm_{op_stem}_sat_i{width}");
    let func_ty = llvm_types::FuncType::get(ctx, lhs_ty, vec![lhs_ty, lhs_ty], false);
    let parent_block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(
            loc.clone(),
            "Rust saturating intrinsic call has no parent block"
        )
    })?;
    helpers::ensure_intrinsic_declared(ctx, parent_block, &intrinsic_name, func_ty)
        .map_err(|e| pliron::input_error!(loc.clone(), "Failed to declare intrinsic: {e}"))?;

    let sym_name: pliron::identifier::Identifier = intrinsic_name
        .as_str()
        .try_into()
        .map_err(|e| pliron::input_error!(loc.clone(), "Invalid intrinsic name: {:?}", e))?;
    let llvm_call = llvm::CallOp::new(
        ctx,
        CallOpCallable::Direct(sym_name),
        func_ty,
        vec![lhs, rhs],
    );
    rewriter.insert_operation(ctx, llvm_call.get_operation());

    let result_mir_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let result_ty = convert_type(ctx, result_mir_ty).map_err(anyhow_to_pliron)?;
    let call_result = llvm_call.get_operation().deref(ctx).get_result(0);
    let (_, final_op) =
        cast_integer_value_to_type(ctx, rewriter, call_result, result_ty, loc.clone())?;
    let replacement = final_op.unwrap_or_else(|| llvm_call.get_operation());
    rewriter.replace_operation(ctx, op, replacement);

    Ok(())
}

/// Convert `core::intrinsics::exact_div` to `llvm.sdiv` or `llvm.udiv`.
///
/// The intrinsic's contract is that the divisor is non-zero and divides the
/// dividend with no remainder; both are the caller's obligation, and violating
/// either is undefined behaviour. That is exactly the contract of a bare LLVM
/// division, so no zero check or remainder correction is emitted.
///
/// Signedness comes from the pre-conversion MIR operand type, the same source
/// `mir.div` uses, because the LLVM integer type is signless by the time the
/// operand is rewritten.
fn convert_rust_exact_div(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let loc = op.deref(ctx).loc().clone();
    if op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(loc, "exact_div call must have one result");
    }

    let args: Vec<Value> = op.deref(ctx).operands().collect();
    let [dividend, divisor] = args[..] else {
        return pliron::input_err!(loc, "exact_div requires a dividend and a divisor");
    };

    let is_signed = if let Some(int_ty) =
        operands_info.lookup_most_recent_of_type::<IntegerType>(ctx, dividend)
    {
        int_ty.signedness() == Signedness::Signed
    } else {
        let live = dividend.get_type(ctx);
        let live_ref = live.deref(ctx);
        match live_ref.downcast_ref::<IntegerType>() {
            // Signless lowers as unsigned, matching `mir.div`.
            Some(int_ty) => int_ty.signedness() == Signedness::Signed,
            None => return pliron::input_err!(loc, "expected integer type for exact_div"),
        }
    };

    // Defensive only: rustc's `exact_div` signature is `(T, T) -> T`, so both
    // operands already have the same width and this widen path never fires.
    let dividend_ty = dividend.get_type(ctx);
    let (divisor, _) = cast_integer_value_to_type(ctx, rewriter, divisor, dividend_ty, loc)?;

    let llvm_op = if is_signed {
        llvm::SDivOp::new(ctx, dividend, divisor).get_operation()
    } else {
        llvm::UDivOp::new(ctx, dividend, divisor).get_operation()
    };
    rewriter.insert_operation(ctx, llvm_op);
    rewriter.replace_operation(ctx, op, llvm_op);
    Ok(())
}

/// Lower the placeholder call for rustc's `carrying_mul_add` bigint intrinsic.
///
/// `core::intrinsics::carrying_mul_add(a, b, c, d)` computes `a * b + c + d`
/// without losing any bits and returns the exact result split into a
/// `(low_half, high_half)` tuple. The integer methods `carrying_mul_add`,
/// `carrying_mul`, and `widening_mul` all funnel into this one intrinsic.
/// An N-bit multiply-accumulate always fits in 2*N bits:
/// even for the largest unsigned inputs,
/// `(2^N - 1)^2 + 2 * (2^N - 1) == 2^(2N) - 1`.
///
/// The lowering widens all four operands to 2*N bits (zero-extending for
/// unsigned types, sign-extending for signed types, matching the `as` casts
/// in core's fallback implementation), computes the product-sum in 2*N-bit
/// arithmetic, and splits the wide value:
///
/// ```text
/// wide = ext(a) * ext(b) + ext(c) + ext(d)  : i2N
/// low  = trunc(wide)                        : iN
/// high = trunc(wide >> N)                   : iN
/// result = { low, high }
/// ```
///
/// A logical shift (`lshr`) is used for the high half even for signed types:
/// core's fallback uses an arithmetic shift there, but the shifted value is
/// immediately truncated to N bits, and bits [N, 2N) of the wide value are
/// identical under either shift. The NVPTX backend pattern-matches this
/// ext/mul/shift idiom into `mul.lo` / `mul.hi` / `mad` instructions, so the
/// generated PTX is good.
///
/// 128-bit integers are rejected with a diagnostic: their lowering would
/// need 256-bit intermediate arithmetic, which NVPTX cannot legalize.
fn convert_rust_carrying_mul_add(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let loc = op.deref(ctx).loc().clone();
    if op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(loc, "Rust carrying_mul_add intrinsic must have one result");
    }

    let args: Vec<Value> = op.deref(ctx).operands().collect();
    if args.len() != 4 {
        return pliron::input_err!(
            loc,
            "Rust carrying_mul_add intrinsic requires four operands \
             (multiplier, multiplicand, addend, carry)"
        );
    }

    let elem_ty = args[0].get_type(ctx);
    let width = integer_bit_width(ctx, elem_ty, loc.clone())?;
    if width > 64 {
        return pliron::input_err!(
            loc,
            "carrying_mul_add on {width}-bit integers is not yet supported on the device: \
             the lowering needs {}-bit intermediate arithmetic, which NVPTX cannot legalize",
            width * 2
        );
    }

    // Rust preserves signedness in the original MIR type; the converted LLVM
    // value is signless, so recover it from the pre-conversion operand type.
    let is_signed = if let Some(int_ty) =
        operands_info.lookup_most_recent_of_type::<IntegerType>(ctx, args[0])
    {
        int_ty.signedness() == Signedness::Signed
    } else {
        return pliron::input_err!(
            loc,
            "expected integer type for Rust carrying_mul_add intrinsic"
        );
    };

    let wide_ty: TypeHandle = IntegerType::get(ctx, width * 2, Signedness::Signless).into();

    // Widen every operand to 2*N bits with the signedness-appropriate extension.
    let mut wide_args = Vec::with_capacity(4);
    for &arg in &args {
        let ext_op = if is_signed {
            llvm::SExtOp::new(ctx, arg, wide_ty).get_operation()
        } else {
            // `nneg` is a poison-introducing optimization flag ("the operand
            // is known non-negative"); we assert nothing and leave it unset.
            llvm::ZExtOp::new_with_nneg(ctx, arg, wide_ty, false).get_operation()
        };
        rewriter.insert_operation(ctx, ext_op);
        wide_args.push(ext_op.deref(ctx).get_result(0));
    }

    // wide = a * b + c + d, computed in 2*N bits (cannot overflow).
    let flags = IntegerOverflowFlagsAttr::default();
    let mul = llvm::MulOp::new_with_overflow_flag(ctx, wide_args[0], wide_args[1], flags.clone())
        .get_operation();
    rewriter.insert_operation(ctx, mul);
    let product = mul.deref(ctx).get_result(0);
    let add_c = llvm::AddOp::new_with_overflow_flag(ctx, product, wide_args[2], flags.clone())
        .get_operation();
    rewriter.insert_operation(ctx, add_c);
    let sum_c = add_c.deref(ctx).get_result(0);
    let add_d =
        llvm::AddOp::new_with_overflow_flag(ctx, sum_c, wide_args[3], flags).get_operation();
    rewriter.insert_operation(ctx, add_d);
    let wide = add_d.deref(ctx).get_result(0);

    // low = trunc(wide); high = trunc(wide >> N).
    let low_op = llvm::TruncOp::new(ctx, wide, elem_ty).get_operation();
    rewriter.insert_operation(ctx, low_op);
    let low = low_op.deref(ctx).get_result(0);

    let shift_amount = {
        let wide_width = NonZeroUsize::new((width * 2) as usize).expect("width is non-zero");
        let attr = IntegerAttr::new(
            IntegerType::get(ctx, width * 2, Signedness::Signless),
            APInt::from_u64(u64::from(width), wide_width),
        );
        let const_op = llvm::ConstantOp::new(ctx, Box::new(attr));
        rewriter.insert_operation(ctx, const_op.get_operation());
        const_op.get_operation().deref(ctx).get_result(0)
    };
    let shr = llvm::LShrOp::new(ctx, wide, shift_amount).get_operation();
    rewriter.insert_operation(ctx, shr);
    let shifted = shr.deref(ctx).get_result(0);
    let high_op = llvm::TruncOp::new(ctx, shifted, elem_ty).get_operation();
    rewriter.insert_operation(ctx, high_op);
    let high = high_op.deref(ctx).get_result(0);

    // Pack the (low, high) tuple into the converted result struct.
    let result_mir_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let result_ty = convert_type(ctx, result_mir_ty).map_err(anyhow_to_pliron)?;
    let undef = llvm::UndefOp::new(ctx, result_ty);
    rewriter.insert_operation(ctx, undef.get_operation());
    let struct_val = undef.get_operation().deref(ctx).get_result(0);

    let insert_low = llvm::InsertValueOp::new(ctx, struct_val, low, vec![0]);
    rewriter.insert_operation(ctx, insert_low.get_operation());
    let struct_with_low = insert_low.get_operation().deref(ctx).get_result(0);

    let insert_high = llvm::InsertValueOp::new(ctx, struct_with_low, high, vec![1]);
    rewriter.insert_operation(ctx, insert_high.get_operation());

    rewriter.replace_operation(ctx, op, insert_high.get_operation());
    Ok(())
}

/// Read the width from an integer type, or report a useful lowering error.
fn integer_bit_width(
    ctx: &Context,
    ty: TypeHandle,
    loc: pliron::location::Location,
) -> Result<u32> {
    let ty_ref = ty.deref(ctx);
    let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() else {
        return pliron::input_err!(loc, "expected integer type for Rust bit intrinsic");
    };
    Ok(int_ty.width())
}

/// Insert the `i1` flag operand used by `llvm.ctlz` and `llvm.cttz`.
fn create_i1_constant(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: bool,
) -> Value {
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let width = NonZeroUsize::new(1).expect("1 is non-zero");
    let apint = APInt::from_u64(u64::from(value), width);
    let attr = IntegerAttr::new(i1_ty, apint);
    let const_op = llvm::ConstantOp::new(ctx, Box::new(attr));
    rewriter.insert_operation(ctx, const_op.get_operation());
    const_op.get_operation().deref(ctx).get_result(0)
}

/// Cast an integer value to the target width when Rust and LLVM disagree.
///
/// This is needed for count/zero intrinsics: LLVM returns `iN`, while Rust's
/// public methods return `u32`.
fn cast_integer_value_to_type(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: Value,
    target_ty: TypeHandle,
    loc: pliron::location::Location,
) -> Result<(Value, Option<Ptr<Operation>>)> {
    let source_width = integer_bit_width(ctx, value.get_type(ctx), loc.clone())?;
    let target_width = integer_bit_width(ctx, target_ty, loc)?;

    if source_width == target_width {
        return Ok((value, None));
    }

    let cast_op = if source_width < target_width {
        let zext = llvm::ZExtOp::new(ctx, value, target_ty);
        let nneg_key: pliron::identifier::Identifier = "llvm_nneg_flag".try_into().unwrap();
        zext.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(nneg_key, pliron::builtin::attributes::BoolAttr::new(false));
        zext.get_operation()
    } else {
        llvm::TruncOp::new(ctx, value, target_ty).get_operation()
    };
    rewriter.insert_operation(ctx, cast_op);
    Ok((cast_op.deref(ctx).get_result(0), Some(cast_op)))
}

/// Recover the transparent scalar ABI projection required by the callee's
/// declared LLVM parameter type.
///
/// A `mir.call` to a `#[device] extern` can already carry the final link symbol,
/// so the reserved source-level extern prefix is not a reliable discriminator at
/// this stage. Instead, use the callee declaration as the ABI authority: when an
/// operand's MIR type history contains a rustc-proven transparent scalar wrapper
/// whose final scalar type exactly matches the next declared LLVM parameter, that
/// wrapper must cross this call boundary as the scalar.
///
/// Nested wrapper histories can contain both `Outer` and `Inner`. Prefer the
/// candidate with the most projection layers so the live outer aggregate is
/// fully unwrapped instead of stopping after the innermost recorded wrapper.
fn transparent_scalar_abi_matching_param(
    ctx: &mut Context,
    arg: Value,
    operands_info: &OperandsInfo,
    expected_ty: Option<TypeHandle>,
) -> Result<Option<TransparentScalarAbiInfo>> {
    let Some(expected_ty) = expected_ty else {
        return Ok(None);
    };

    let mut best: Option<TransparentScalarAbiInfo> = None;
    for mir_ty in operands_info.lookup_operand_history(arg) {
        let is_transparent_scalar = {
            let ty_ref = mir_ty.deref(ctx);
            ty_ref
                .downcast_ref::<MirStructType>()
                .is_some_and(MirStructType::is_transparent_scalar)
        };
        if !is_transparent_scalar {
            continue;
        }

        let abi = transparent_scalar_abi_info(ctx, mir_ty).map_err(anyhow_to_pliron)?;
        if abi.scalar_ty != expected_ty {
            continue;
        }

        let replace = best
            .as_ref()
            .is_none_or(|current| abi.layers.len() > current.layers.len());
        if replace {
            best = Some(abi);
        }
    }

    Ok(best)
}

/// Flatten arguments according to ABI rules and coerce each one to the
/// callee's expected parameter type.
///
/// - Slice types → (ptr, len) pair
/// - Struct types → individual field values (in MEMORY ORDER)
/// - `repr(transparent)` scalar structs at device-extern boundaries → underlying scalar
/// - Other types → pass through
///
/// `expected_param_tys`, when present, is the LLVM-level (post-flatten)
/// parameter signature of the callee. Each emitted (post-flatten) argument
/// is coerced to its corresponding entry — in practice this means inserting
/// an `addrspacecast` whenever a pointer arg's address space differs from
/// the declared parameter's. When `expected_param_tys` is `None` (the
/// callee declaration could not be located), each pointer falls back to
/// being cast down to the generic address space, preserving the legacy
/// behavior so device-extern calls (which always declare generic params)
/// keep working.
fn flatten_arguments(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    args: &[Value],
    operands_info: &OperandsInfo,
    expected_param_tys: Option<&[TypeHandle]>,
) -> Result<(Vec<Value>, Vec<TypeHandle>)> {
    let mut flattened_args = Vec::new();
    let mut flattened_arg_types = Vec::new();

    // Helper: pull the next expected param type, if any.
    let take_expected = |flattened_arg_types: &Vec<TypeHandle>| -> Option<TypeHandle> {
        let idx = flattened_arg_types.len();
        expected_param_tys.and_then(|tys| tys.get(idx).copied())
    };

    for arg in args.iter() {
        let arg_ty = arg.get_type(ctx);

        enum FlattenKind {
            /// `(ptr, len)`, then one argument per index-space layout field
            /// (empty for `&[T]` and for slices over type-fixed spaces).
            Slice {
                space_tys: Vec<TypeHandle>,
            },
            Struct {
                layout: StructLayoutInfo,
            },
            TransparentScalar(TransparentScalarAbiInfo),
            None,
        }

        let transparent_abi = transparent_scalar_abi_matching_param(
            ctx,
            *arg,
            operands_info,
            take_expected(&flattened_arg_types),
        )?;

        let flatten_kind = if let Some(abi) = transparent_abi {
            FlattenKind::TransparentScalar(abi)
        } else if let Some(mir_ty) = operands_info.lookup_most_recent_type(*arg) {
            let ty_ref = mir_ty.deref(ctx);
            if ty_ref.is::<MirSliceType>() {
                FlattenKind::Slice {
                    space_tys: Vec::new(),
                }
            } else if let Some(slice_ty) = ty_ref.downcast_ref::<MirDisjointSliceType>() {
                FlattenKind::Slice {
                    space_tys: slice_ty.space_tys.clone(),
                }
            } else if let Some(struct_ty) = ty_ref.downcast_ref::<MirStructType>() {
                FlattenKind::Struct {
                    layout: StructLayoutInfo::of_struct(struct_ty),
                }
            } else {
                FlattenKind::None
            }
        } else {
            FlattenKind::None
        };

        match flatten_kind {
            FlattenKind::Slice { space_tys } => {
                let ptr_ty = llvm_types::PointerType::get_generic(ctx);
                let len_ty = IntegerType::get(ctx, 64, Signedness::Signless);

                let extract_ptr = llvm::ExtractValueOp::new(ctx, *arg, vec![0])?;
                rewriter.insert_operation(ctx, extract_ptr.get_operation());
                let ptr_val = extract_ptr.get_operation().deref(ctx).get_result(0);

                let extract_len = llvm::ExtractValueOp::new(ctx, *arg, vec![1])?;
                rewriter.insert_operation(ctx, extract_len.get_operation());
                let len_val = extract_len.get_operation().deref(ctx).get_result(0);

                let (ptr_val, ptr_ty) = coerce_arg_to_param_ty(
                    ctx,
                    rewriter,
                    ptr_val,
                    ptr_ty.into(),
                    take_expected(&flattened_arg_types),
                )?;
                flattened_args.push(ptr_val);
                flattened_arg_types.push(ptr_ty);

                let (len_val, len_ty) = coerce_arg_to_param_ty(
                    ctx,
                    rewriter,
                    len_val,
                    len_ty.into(),
                    take_expected(&flattened_arg_types),
                )?;
                flattened_args.push(len_val);
                flattened_arg_types.push(len_ty);

                // Index-space layout fields (e.g. a runtime row width) follow
                // at aggregate slots `2 + i`. `convert_function_type` gave the
                // callee one parameter per non-zero-sized field, so the call
                // must supply them symmetrically or the callee would read a
                // garbage row width from a missing argument.
                for (i, space_ty) in space_tys.into_iter().enumerate() {
                    let converted = convert_type(ctx, space_ty).map_err(|e| {
                        pliron::input_error_noloc!(
                            "failed to convert disjoint-slice index-space field type: {e}"
                        )
                    })?;
                    // A zero-sized field contributes no parameter, matching
                    // the signature conversion.
                    if is_zero_sized_type(ctx, converted) {
                        continue;
                    }
                    let slot = u32::try_from(2 + i).expect("index-space field slot fits u32");
                    let extract_space = llvm::ExtractValueOp::new(ctx, *arg, vec![slot])?;
                    rewriter.insert_operation(ctx, extract_space.get_operation());
                    let space_val = extract_space.get_operation().deref(ctx).get_result(0);

                    let (space_val, space_ty) = coerce_arg_to_param_ty(
                        ctx,
                        rewriter,
                        space_val,
                        converted,
                        take_expected(&flattened_arg_types),
                    )?;
                    flattened_args.push(space_val);
                    flattened_arg_types.push(space_ty);
                }
            }
            FlattenKind::TransparentScalar(abi) => {
                let mut scalar = *arg;

                // Layers are recorded outer-to-inner. Extract through every
                // wrapper so nested transparent ADTs reach the exact scalar
                // type declared by the device extern.
                for layer in &abi.layers {
                    let extract = llvm::ExtractValueOp::new(ctx, scalar, vec![layer.field_slot])?;
                    rewriter.insert_operation(ctx, extract.get_operation());
                    scalar = extract.get_operation().deref(ctx).get_result(0);
                }

                let (scalar, scalar_ty) = coerce_arg_to_param_ty(
                    ctx,
                    rewriter,
                    scalar,
                    abi.scalar_ty,
                    take_expected(&flattened_arg_types),
                )?;
                flattened_args.push(scalar);
                flattened_arg_types.push(scalar_ty);
            }
            FlattenKind::Struct { layout } => {
                // Walk in memory order (the order `convert_function_type`
                // flattens params in), extracting each non-ZST field from
                // the slot the type converter placed it in, NOT from a
                // running non-ZST count, which would land on `[N x i8]`
                // padding slots for padded structs (issue #128).
                let map = build_struct_slot_map(ctx, &layout).map_err(anyhow_to_pliron)?;
                for &decl_idx in &layout.mem_to_decl {
                    let Some(slot) = map.decl_to_llvm[decl_idx] else {
                        continue; // ZST field: not passed.
                    };
                    let llvm_field_ty = map.field_llvm_types[decl_idx];
                    let extract_op = llvm::ExtractValueOp::new(ctx, *arg, vec![slot])?;
                    rewriter.insert_operation(ctx, extract_op.get_operation());
                    let field_val = extract_op.get_operation().deref(ctx).get_result(0);

                    let (field_val, field_ty) = coerce_arg_to_param_ty(
                        ctx,
                        rewriter,
                        field_val,
                        llvm_field_ty,
                        take_expected(&flattened_arg_types),
                    )?;
                    flattened_args.push(field_val);
                    flattened_arg_types.push(field_ty);
                }
            }
            FlattenKind::None => {
                if is_zero_sized_type(ctx, arg_ty) {
                    continue;
                }
                let (final_arg, final_ty) = coerce_arg_to_param_ty(
                    ctx,
                    rewriter,
                    *arg,
                    arg_ty,
                    take_expected(&flattened_arg_types),
                )?;
                flattened_args.push(final_arg);
                flattened_arg_types.push(final_ty);
            }
        }
    }

    Ok((flattened_args, flattened_arg_types))
}

/// Coerce a single (post-flatten) argument so that it satisfies the callee's
/// declared parameter type.
///
/// When `expected_ty` is supplied, this is the principled coercion used for
/// every kind of call: pointer arguments whose address space differs from
/// the declared parameter's are bridged with an `llvm.addrspacecast` to
/// the declared address space (in either direction — generic ↔ shared,
/// shared ↔ global, etc.). Non-pointer mismatches are left for the verifier
/// to surface as a real type error.
///
/// When `expected_ty` is `None` (callee declaration not located), we fall
/// back to the legacy "always generic" policy. That fallback exists because
/// historically every call site cast pointer args down to addrspace(0) on
/// the assumption that all callees declared their parameters as generic.
/// Device-extern functions (`#[device] extern "C"`) still rely on that
/// assumption and benefit from the fallback when their declaration hasn't
/// landed in the module yet during conversion.
fn coerce_arg_to_param_ty(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    arg: Value,
    arg_ty: TypeHandle,
    expected_ty: Option<TypeHandle>,
) -> Result<(Value, TypeHandle)> {
    if let Some(expected_ty) = expected_ty {
        if arg_ty == expected_ty {
            return Ok((arg, arg_ty));
        }
        let arg_addrspace = pointer_addrspace(ctx, arg_ty);
        let expected_addrspace = pointer_addrspace(ctx, expected_ty);
        if let (Some(src_as), Some(dst_as)) = (arg_addrspace, expected_addrspace)
            && src_as != dst_as
        {
            let cast_ty = llvm_types::PointerType::get(ctx, dst_as).into();
            let cast_op = llvm::AddrSpaceCastOp::new(ctx, arg, cast_ty);
            rewriter.insert_operation(ctx, cast_op.get_operation());
            let casted_val = cast_op.get_operation().deref(ctx).get_result(0);
            return Ok((casted_val, expected_ty));
        }
        return Ok((arg, arg_ty));
    }

    // Legacy fallback: cast any non-generic pointer down to addrspace(0).
    let arg_addrspace = pointer_addrspace(ctx, arg_ty);
    if let Some(addrspace) = arg_addrspace
        && addrspace != ADDRSPACE_GENERIC
    {
        let cast_ty = llvm_types::PointerType::get(ctx, ADDRSPACE_GENERIC).into();
        let cast_op = llvm::AddrSpaceCastOp::new(ctx, arg, cast_ty);
        rewriter.insert_operation(ctx, cast_op.get_operation());
        let casted_val = cast_op.get_operation().deref(ctx).get_result(0);
        let generic_ptr_ty = llvm_types::PointerType::get_generic(ctx);
        return Ok((casted_val, generic_ptr_ty.into()));
    }

    Ok((arg, arg_ty))
}

/// Return the address space of `ty` if it is an LLVM pointer type.
fn pointer_addrspace(ctx: &Context, ty: TypeHandle) -> Option<u32> {
    ty.deref(ctx)
        .downcast_ref::<llvm_types::PointerType>()
        .map(|ptr_ty| ptr_ty.address_space())
}

/// Look up the LLVM-level parameter signature of `callee_ident` by walking
/// from `op` up to the parent module and scanning its top-level function
/// declarations.
///
/// Two callee shapes are recognised:
///
///   * `llvm::FuncOp` — already lowered. We return its `arg_types()` directly.
///     This covers device-extern declarations (inserted by the importer
///     ahead of conversion) and any callee whose conversion happened to run
///     before the current call.
///
///   * `MirFuncOp` — still in MIR form. We run `convert_function_type` to
///     project its declared MIR signature into the same LLVM-level
///     (slice/struct-flattened) parameter list that the function lowerer
///     will eventually emit. This guarantees the call site sees the exact
///     same parameter types the callee will end up with — including the
///     address space of every pointer parameter.
///
/// Returns `None` when neither form is present (e.g. the callee lives in a
/// different module). The caller then falls back to its legacy coercion.
fn find_callee_arg_types(
    ctx: &mut Context,
    op: Ptr<Operation>,
    callee_ident: &pliron::identifier::Identifier,
) -> Option<Vec<TypeHandle>> {
    let block = op.deref(ctx).get_parent_block()?;
    let func_op = block.deref(ctx).get_parent_op(ctx)?;
    let module_op = func_op.deref(ctx).get_parent_op(ctx)?;

    let region = module_op.deref(ctx).get_region(0);
    let module_block = region.deref(ctx).iter(ctx).next()?;

    // First pass: walk the module's top-level ops once and extract whichever
    // declaration shape (LLVM or MIR) carries the matching symbol name.
    // `Ptr<Operation>` is `Copy`, so we can collect candidates without
    // borrowing the context past this loop.
    let mut llvm_decl_op: Option<Ptr<Operation>> = None;
    let mut mir_decl_op: Option<Ptr<Operation>> = None;
    for existing_op in module_block.deref(ctx).iter(ctx) {
        if let Some(existing_func) = Operation::get_op::<llvm::FuncOp>(existing_op, ctx)
            && &existing_func.get_symbol_name(ctx) == callee_ident
        {
            llvm_decl_op = Some(existing_op);
            break;
        }
        if let Some(existing_func) = MirFuncOp::wrap(ctx, existing_op)
            && &existing_func.get_symbol_name(ctx) == callee_ident
        {
            mir_decl_op = Some(existing_op);
            // Keep scanning — an `llvm::FuncOp` would still take precedence.
        }
    }

    if let Some(existing_op) = llvm_decl_op {
        let func = Operation::get_op::<llvm::FuncOp>(existing_op, ctx)?;
        let func_ty = func.get_type(ctx);
        return Some(func_ty.deref(ctx).arg_types().clone());
    }

    if let Some(existing_op) = mir_decl_op {
        let func = MirFuncOp::wrap(ctx, existing_op)?;
        let mir_func_ty = func.get_type(ctx);
        // Match the callee's own boundary kind: a kernel callee (rare; not
        // produced by Rust code today, but cheap to keep correct) keeps its
        // params as byval, internal callees stay flattened. Reading
        // `is_kernel_func` here keeps `find_callee_arg_types` consistent
        // with what `lowering.rs::convert_func` would produce for the
        // same callee.
        let callee_is_kernel = is_kernel_func(ctx, existing_op);
        let llvm_func_ty = convert_function_type(ctx, mir_func_ty, callee_is_kernel).ok()?;
        return Some(llvm_func_ty.deref(ctx).arg_types().clone());
    }

    None
}

/// Resolve a device-extern symbol name by stripping the internal prefix.
///
/// When calling `#[device] extern "C"` functions, the macro-expanded Rust code
/// calls symbols like `cuda_oxide_device_extern_<hash>_foo`. We strip the
/// reserved prefix here so the emitted LLVM IR refers to the original symbol
/// `foo` that the linked LTOIR provides.
///
/// # Limitations
///
/// This approach derives the original name by stripping the prefix, which means:
/// - Custom `#[link_name = "..."]` attributes are NOT honored (edge case).
/// - If the original function is `bar` but the user wrote `#[link_name = "custom"]`,
///   we'd emit `bar`, not `custom`. Acceptable for current call sites.
fn resolve_device_extern_symbol(callee_name: &str) -> String {
    device_extern_base_name(callee_name)
        .map(str::to_string)
        .unwrap_or_else(|| callee_name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_transparent_arg_uses_scalar_when_callee_param_is_scalar() {
        use dialect_mir::types::StructAbiKind;

        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        crate::register(&mut ctx);

        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let inner: TypeHandle = MirStructType::get_with_full_layout_and_abi(
            &mut ctx,
            "Inner".into(),
            vec!["value".into()],
            vec![u32_ty],
            vec![0],
            vec![0],
            4,
            4,
            StructAbiKind::TransparentScalar,
        )
        .into();
        let outer: TypeHandle = MirStructType::get_with_full_layout_and_abi(
            &mut ctx,
            "Outer".into(),
            vec!["inner".into()],
            vec![inner],
            vec![0],
            vec![0],
            4,
            4,
            StructAbiKind::TransparentScalar,
        )
        .into();

        // Model the real conversion pipeline: the live value gets its LLVM
        // type from ordinary aggregate conversion, while OperandsInfo retains
        // the MIR wrapper history separately.
        let live_outer_ty = convert_type(&mut ctx, outer).unwrap();
        let undef = llvm::UndefOp::new(&mut ctx, live_outer_ty);
        let arg = undef.get_operation().deref(&ctx).get_result(0);
        let operands_info = OperandsInfo::new(vec![(arg, vec![outer, inner])]);

        let outer_abi = transparent_scalar_abi_info(&mut ctx, outer).unwrap();
        let selected = transparent_scalar_abi_matching_param(
            &mut ctx,
            arg,
            &operands_info,
            Some(outer_abi.scalar_ty),
        )
        .unwrap()
        .expect("scalar callee parameter must select the outer transparent ABI");
        assert_eq!(selected.layers.len(), 2);
        assert_eq!(selected.scalar_ty, outer_abi.scalar_ty);

        // An ordinary Rust device-function boundary for `Outer(Inner(T))`
        // still expects the inner aggregate after one-level struct flattening,
        // so the transparent scalar projection must not activate there.
        let inner_aggregate_ty = convert_type(&mut ctx, inner).unwrap();
        assert!(
            transparent_scalar_abi_matching_param(
                &mut ctx,
                arg,
                &operands_info,
                Some(inner_aggregate_ty),
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn test_resolve_device_extern_symbol() {
        use reserved_oxide_symbols::{device_extern_symbol, kernel_symbol};

        // Plain form: the hash-suffixed prefix is stripped to leave the
        // user-facing base name.
        assert_eq!(
            resolve_device_extern_symbol(&device_extern_symbol("dot_product")),
            "dot_product"
        );

        // FQDN form (cross-crate): the extractor skips the crate qualifier
        // and the hash-suffixed prefix.
        let fqdn = format!("device_ffi_test::{}", device_extern_symbol("foo"));
        assert_eq!(resolve_device_extern_symbol(&fqdn), "foo");

        // Non-extern symbols pass through unchanged — this is what lets
        // ordinary device-function calls reach the LLVM backend without
        // the device-extern detour.
        assert_eq!(
            resolve_device_extern_symbol("my_module::regular_function"),
            "my_module::regular_function"
        );

        // A kernel symbol must NOT be confused for a device-extern symbol
        // (mutual-exclusion guarantee from reserved-oxide-symbols).
        let kernel_name = kernel_symbol("my_kernel");
        assert_eq!(resolve_device_extern_symbol(&kernel_name), kernel_name);
    }

    /// The `exact_div` placeholder is recognized, and nothing else is.
    ///
    /// The negative half matters: `unchecked_div` carries a weaker promise (the
    /// divisor is non-zero, but the remainder is discarded), so folding it into
    /// this path would silently drop the exactness obligation.
    #[test]
    fn test_exact_div_placeholder_round_trip() {
        assert!(
            RustExactDivIntrinsic::from_placeholder_callee(rust_intrinsics::CALLEE_EXACT_DIV)
                .is_some()
        );

        for other in [
            rust_intrinsics::CALLEE_SATURATING_ADD,
            rust_intrinsics::CALLEE_SATURATING_SUB,
            rust_intrinsics::CALLEE_CARRYING_MUL_ADD,
            rust_intrinsics::CALLEE_SQRT_F32,
            "core::intrinsics::unchecked_div",
        ] {
            assert!(
                RustExactDivIntrinsic::from_placeholder_callee(other).is_none(),
                "`{other}` must not be treated as exact_div"
            );
        }
    }
}
