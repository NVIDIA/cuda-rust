/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cast operation conversion: `dialect-mir` → LLVM dialect.
//!
//! Dispatches on `MirCastKindAttr` (preserved from Rust MIR) to select the
//! correct LLVM instruction. This avoids guessing cast semantics from types.
//!
//! # Cast Dispatch
//!
//! | MirCastKindAttr                | LLVM Operation                                         |
//! |--------------------------------|--------------------------------------------------------|
//! | Transmute                      | exact-size bit cast or aligned memory round-trip       |
//! | IntToInt (wider, signed)       | `sext`                                                 |
//! | IntToInt (wider, unsigned)     | `zext`                                                 |
//! | IntToInt (narrower)            | `trunc`                                                |
//! | IntToInt (same width)          | `bitcast`                                              |
//! | IntToFloat                     | `sitofp` or `uitofp`                                   |
//! | FloatToInt                     | `llvm.fptosi.sat` / `llvm.fptoui.sat` (Rust semantics) |
//! | FloatToFloat                   | `fpext` or `fptrunc`                                   |
//! | PtrToPtr / FnPtrToPtr          | `emit_pointer_cast` (see below)                        |
//! | PointerCoercionUnsize          | `emit_unsize_cast` → `emit_pointer_cast` (see below)   |
//! | PointerCoercion* (other)       | `emit_pointer_cast` (see below)                        |
//! | PointerExposeAddress           | genericize named-space pointers, then `ptrtoint`       |
//! | PointerWithExposedProvenance   | `inttoptr`                                             |
//!
//! ## `emit_unsize_cast` handles array→slice unsizing:
//! | Source → Dest                  | LLVM Operation                                  |
//! |--------------------------------|-------------------------------------------------|
//! | ptr-to-array → struct (slice)  | `insertvalue` ptr + `insertvalue` len into undef |
//! | other                          | falls through to `emit_pointer_cast`             |
//!
//! ## `emit_pointer_cast` handles semantic pointer coercions:
//! | Source → Dest                                  | LLVM Operation                              |
//! |------------------------------------------------|---------------------------------------------|
//! | struct → ptr (fat→thin)                        | `extractvalue` field 0                      |
//! | ptr → struct (thin→fat, no niche)              | `insertvalue` into undef                    |
//! | ptr → integer                                  | genericize named-space pointers, `ptrtoint` |
//! | integer → ptr                                  | `inttoptr`                                  |
//! | slice-shaped struct → slice-shaped struct     | fieldwise SSA; `addrspacecast` data as needed |
//! | other struct → struct                          | `alloca` + `store` + `load`                 |
//! | ptr → ptr (diff addrspace)                     | `addrspacecast`                             |
//! | struct → integer, equal size                   | `alloca` + `store` + `load`                 |
//! | struct → integer, mismatched size              | cuda-oxide error (see issue #21)            |
//! | array ↔ anything, equal size                   | `alloca` + `store` + `load`                 |
//! | array ↔ anything, mismatched size              | cuda-oxide error (see issue #125)           |
//! | otherwise                                      | `bitcast`                                   |
//!
//! Transmute never uses those semantic field-zero shortcuts. When an enum or
//! another aggregate participates, its already-physical LLVM bytes are copied
//! through an exactly-sized, explicitly aligned stack slot.

use crate::convert::types::{
    convert_type, llvm_type_contains_pointer_in_address_space, llvm_type_is_byte_faithful,
    llvm_type_size_align,
};
use crate::helpers;
use dialect_mir::attributes::MirCastKindAttr;
use dialect_mir::ops::MirCastOp;
use dialect_mir::types::{MirArrayType, MirPtrType, address_space};
use llvm_export::op_interfaces::{CastOpInterface, CastOpWithNNegInterface};
use llvm_export::ops as llvm;
use llvm_export::types::{FuncType, PointerType, PointerTypeExt};
use pliron::builtin::op_interfaces::CallOpCallable;
use pliron::builtin::type_interfaces::FloatTypeInterface;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;
use pliron::result::Result;
use pliron::r#type::{Typed, type_cast};

/// Convert a MIR cast operation to the appropriate LLVM cast instruction.
///
/// Dispatches on the `cast_kind` attribute to determine semantics, then uses
/// source/destination types for the specific instruction selection within each kind.
pub fn convert(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let loc = op.deref(ctx).loc();
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    let val = match operands.as_slice() {
        [val] => *val,
        _ => return pliron::input_err!(loc, "Cast requires exactly 1 operand"),
    };

    let cast_op = MirCastOp::new(op);
    let cast_kind_ref = cast_op.get_attr_cast_kind(ctx).ok_or_else(|| {
        pliron::input_error!(loc.clone(), "MirCastOp missing cast_kind attribute")
    })?;
    let cast_kind = cast_kind_ref.clone();
    drop(cast_kind_ref);

    // Pre-conversion MIR operand type — preserves signedness info from Rust's type system
    let mir_opd = op.deref(ctx).get_operand(0);
    let mir_opd_ty = operands_info
        .lookup_most_recent_type(mir_opd)
        .unwrap_or_else(|| mir_opd.get_type(ctx));
    // Pre-conversion MIR result type — preserves signedness (LLVM types are signless)
    let mir_result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let llvm_ty = convert_type(ctx, mir_result_ty).map_err(|e| pliron::input_error!(loc, "{e}"))?;
    let val_ty = val.get_type(ctx);

    // Rust pointer/reference kind is a MIR semantic distinction, not an LLVM
    // representation distinction. With opaque LLVM pointers (and the canonical
    // `{ptr, i64}` slice layout), changing only pointer kind can therefore
    // produce identical lowered source/destination types. Forward the value
    // directly instead of manufacturing a bitcast or a struct memory
    // round-trip. Address-space-changing casts still take the normal path.
    if val_ty == llvm_ty
        && matches!(
            &cast_kind,
            MirCastKindAttr::PtrToPtr
                | MirCastKindAttr::PointerCoercionMutToConst
                | MirCastKindAttr::Subtype
        )
    {
        rewriter.replace_operation_with_values(ctx, op, vec![val]);
        return Ok(());
    }

    let llvm_op = match &cast_kind {
        MirCastKindAttr::Transmute => emit_transmute(ctx, rewriter, val, val_ty, llvm_ty)?,

        MirCastKindAttr::IntToInt => {
            let src_w = val_ty
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .map(|t| t.width())
                .ok_or_else(|| pliron::input_error_noloc!("IntToInt: source is not an integer"))?;
            let dst_w = llvm_ty
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .map(|t| t.width())
                .ok_or_else(|| {
                    pliron::input_error_noloc!("IntToInt: destination is not an integer")
                })?;
            convert_int_to_int(ctx, rewriter, val, llvm_ty, src_w, dst_w, mir_opd_ty)?
        }

        MirCastKindAttr::IntToFloat => {
            convert_int_to_float(ctx, rewriter, op, val, val_ty, llvm_ty, mir_opd_ty)?
        }

        MirCastKindAttr::FloatToInt => {
            convert_float_to_int(ctx, rewriter, op, val, llvm_ty, mir_result_ty)?
        }

        MirCastKindAttr::FloatToFloat => {
            convert_float_to_float(ctx, rewriter, val, llvm_ty, val_ty)?
        }

        MirCastKindAttr::PointerCoercionUnsize => {
            emit_unsize_cast(ctx, rewriter, op, val, val_ty, llvm_ty, mir_opd_ty)?
        }

        MirCastKindAttr::PtrToPtr
        | MirCastKindAttr::FnPtrToPtr
        | MirCastKindAttr::PointerCoercionMutToConst
        | MirCastKindAttr::PointerCoercionReifyFnPointer
        | MirCastKindAttr::PointerCoercionUnsafeFnPointer
        | MirCastKindAttr::PointerCoercionClosureFnPointer
        | MirCastKindAttr::PointerCoercionArrayToPointer
        | MirCastKindAttr::Subtype => emit_pointer_cast(ctx, rewriter, op, val, val_ty, llvm_ty)?,

        MirCastKindAttr::PointerExposeAddress => {
            emit_ptr_to_int(ctx, rewriter, val, val_ty, llvm_ty)
        }

        MirCastKindAttr::PointerWithExposedProvenance => {
            emit_int_to_ptr(ctx, rewriter, val, llvm_ty)
        }
    };

    rewriter.insert_operation(ctx, llvm_op);
    rewriter.replace_operation(ctx, op, llvm_op);

    Ok(())
}

/// Integer → integer: extension, truncation, or same-width bitcast.
fn convert_int_to_int(
    ctx: &mut Context,
    _rewriter: &mut DialectConversionRewriter,
    val: pliron::value::Value,
    llvm_ty: pliron::r#type::TypeHandle,
    src_w: u32,
    dst_w: u32,
    mir_opd_ty: pliron::r#type::TypeHandle,
) -> Result<Ptr<Operation>> {
    if dst_w > src_w {
        let is_signed = {
            let ty_obj = mir_opd_ty.deref(ctx);
            ty_obj
                .downcast_ref::<IntegerType>()
                .ok_or_else(|| {
                    pliron::input_error_noloc!("IntToInt: MIR operand type is not an integer")
                })?
                .signedness()
                == Signedness::Signed
        };

        if is_signed {
            Ok(llvm::SExtOp::new(ctx, val, llvm_ty).get_operation())
        } else {
            let zext = llvm::ZExtOp::new(ctx, val, llvm_ty);
            let nneg_key: pliron::identifier::Identifier = "llvm_nneg_flag".try_into().unwrap();
            zext.get_operation()
                .deref_mut(ctx)
                .attributes
                .set(nneg_key, pliron::builtin::attributes::BoolAttr::new(false));
            Ok(zext.get_operation())
        }
    } else if dst_w < src_w {
        Ok(llvm::TruncOp::new(ctx, val, llvm_ty).get_operation())
    } else {
        Ok(llvm::BitcastOp::new(ctx, val, llvm_ty).get_operation())
    }
}

/// Integer → float: signed or unsigned conversion.
fn convert_int_to_float(
    ctx: &mut Context,
    _rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    val: pliron::value::Value,
    val_ty: pliron::r#type::TypeHandle,
    llvm_ty: pliron::r#type::TypeHandle,
    mir_opd_ty: pliron::r#type::TypeHandle,
) -> Result<Ptr<Operation>> {
    let is_signed = {
        let ty_obj = mir_opd_ty.deref(ctx);
        ty_obj
            .downcast_ref::<IntegerType>()
            .ok_or_else(|| {
                pliron::input_error_noloc!("IntToFloat: MIR operand type is not an integer")
            })?
            .signedness()
            == Signedness::Signed
    };

    let int_width = val_ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|t| t.width())
        .ok_or_else(|| {
            pliron::input_error!(
                op.deref(ctx).loc(),
                "IntToFloat: operand type is not an integer"
            )
        })?;
    if int_width > 64 {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "IntToFloat: integer source widths wider than 64 bits are not yet supported on the device"
        );
    }

    if is_signed {
        Ok(llvm::SIToFPOp::new(ctx, val, llvm_ty).get_operation())
    } else {
        let uitofp = llvm::UIToFPOp::new(ctx, val, llvm_ty);
        let nneg_key: pliron::identifier::Identifier = "llvm_nneg_flag".try_into().unwrap();
        uitofp
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(nneg_key, pliron::builtin::attributes::BoolAttr::new(false));
        Ok(uitofp.get_operation())
    }
}

/// Float → integer: signed or unsigned conversion (saturating, Rust semantics).
///
/// Uses LLVM's `llvm.fptosi.sat` / `llvm.fptoui.sat` intrinsics so that
/// out-of-range values saturate to T::MIN/T::MAX and NaN → 0, matching Rust.
/// Uses the **MIR** result type for signedness — the LLVM integer type is signless.
fn convert_float_to_int(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    val: pliron::value::Value,
    llvm_ty: pliron::r#type::TypeHandle,
    mir_result_ty: pliron::r#type::TypeHandle,
) -> Result<Ptr<Operation>> {
    let val_ty = val.get_type(ctx);
    let is_signed = {
        let ty_obj = mir_result_ty.deref(ctx);
        ty_obj
            .downcast_ref::<IntegerType>()
            .ok_or_else(|| {
                pliron::input_error_noloc!("FloatToInt: MIR result type is not an integer")
            })?
            .signedness()
            == Signedness::Signed
    };

    let int_width = llvm_ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|t| t.width())
        .ok_or_else(|| {
            pliron::input_error!(
                op.deref(ctx).loc(),
                "FloatToInt: result type is not an integer"
            )
        })?;
    if int_width > 64 {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "FloatToInt: integer destinations wider than 64 bits are not yet supported on the device"
        );
    }

    let int_suffix = format!("i{}", int_width);

    let float_suffix = match float_bit_width(ctx, val_ty) {
        Ok(16) => "f16",
        Ok(32) => "f32",
        Ok(64) => "f64",
        Ok(bits) => {
            return pliron::input_err!(
                op.deref(ctx).loc(),
                "FloatToInt: unsupported source float width {bits}"
            );
        }
        Err(err) => return Err(err),
    };

    let intrinsic_name = if is_signed {
        format!("llvm_fptosi_sat_{}_{}", int_suffix, float_suffix)
    } else {
        format!("llvm_fptoui_sat_{}_{}", int_suffix, float_suffix)
    };

    let func_ty = FuncType::get(ctx, llvm_ty, vec![val_ty], false);

    // Navigate from op to its containing block for intrinsic declaration
    let llvm_block = op
        .deref(ctx)
        .get_parent_block()
        .ok_or_else(|| pliron::input_error!(op.deref(ctx).loc(), "Cast op has no parent block"))?;
    helpers::ensure_intrinsic_declared(ctx, llvm_block, &intrinsic_name, func_ty).map_err(|e| {
        pliron::input_error!(op.deref(ctx).loc(), "Failed to declare intrinsic: {e}")
    })?;

    let sym_name: pliron::identifier::Identifier =
        intrinsic_name.as_str().try_into().map_err(|e| {
            pliron::input_error!(op.deref(ctx).loc(), "Invalid intrinsic name: {:?}", e)
        })?;
    let callee = CallOpCallable::Direct(sym_name);
    let llvm_call = llvm::CallOp::new(ctx, callee, func_ty, vec![val]);

    // The call op is the final replacement, but we need intermediate ops inserted by rewriter.
    // Don't insert here — the caller handles insert + replace.
    let _ = &rewriter;
    Ok(llvm_call.get_operation())
}

/// Return the element count of the sized array that becomes a slice tail
/// during an unsize coercion.
///
/// The array may be the direct pointee (`&[T; N] -> &[T]`) or may sit behind
/// one or more trailing struct fields (`&Outer<Inner<[T; N]>> ->
/// &Outer<Inner<[T]>>`). Rust DST unsizing follows that trailing-field chain,
/// so mirror it here instead of assuming the array is only one field deep.
fn unsize_array_tail_len(ctx: &Context, mut ty: pliron::r#type::TypeHandle) -> Option<u64> {
    loop {
        let next = {
            let ty_ref = ty.deref(ctx);

            if let Some(array_ty) = ty_ref.downcast_ref::<MirArrayType>() {
                return Some(array_ty.size());
            }

            let struct_ty = ty_ref.downcast_ref::<dialect_mir::types::MirStructType>()?;
            let field_types = struct_ty.field_types();
            let last_decl_idx = match struct_ty.memory_order().last().copied() {
                Some(idx) => idx,
                None => field_types.len().checked_sub(1)?,
            };

            *field_types.get(last_decl_idx)?
        };

        ty = next;
    }
}

/// Emit an Unsize coercion: `&[T; N]` → `&[T]` (or `*[T; N]` → `[T]`).
///
/// When the MIR source is a pointer to an array and the LLVM destination is a
/// fat-pointer struct `{ ptr, i64 }`, we construct the full slice by inserting
/// both the data pointer (field 0) and the array length (field 1).
///
/// For other Unsize coercions (e.g., trait objects), falls through to
/// `emit_pointer_cast`.
fn emit_unsize_cast(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    val: pliron::value::Value,
    val_ty: pliron::r#type::TypeHandle,
    llvm_ty: pliron::r#type::TypeHandle,
    mir_opd_ty: pliron::r#type::TypeHandle,
) -> Result<Ptr<Operation>> {
    let array_len = {
        let mir_ref = mir_opd_ty.deref(ctx);
        mir_ref
            .downcast_ref::<MirPtrType>()
            .and_then(|ptr_ty| unsize_array_tail_len(ctx, ptr_ty.pointee))
    };

    if let Some(len) = array_len {
        let dst_is_struct = llvm_ty.deref(ctx).is::<llvm_export::types::StructType>();

        if dst_is_struct {
            // Coerce the source data pointer to the slice's field-0 address
            // space. When the array lives in shared memory (addrspace 3, e.g.
            // a `&[T; N]` row of a nested `SharedArray<[T; N], M>` borrowed
            // via indexing — see `examples/shared_slice_unsize`), forming a
            // `&[T]`/`&mut [T]` fat pointer over it yields a
            // `ptr addrspace(3)`, but the canonical slice type stores
            // `ptr addrspace(0)` in field 0. Insert an addrspacecast so the
            // insert_value types match (PTX lowers this to `cvta.shared`).
            let field0_as = llvm_ty
                .deref(ctx)
                .downcast_ref::<llvm_export::types::StructType>()
                .map(|st| st.field_type(0))
                .and_then(|f0| {
                    f0.deref(ctx)
                        .downcast_ref::<llvm_export::types::PointerType>()
                        .map(|pt| pt.address_space())
                });
            let val_as = val
                .get_type(ctx)
                .deref(ctx)
                .downcast_ref::<llvm_export::types::PointerType>()
                .map(|pt| pt.address_space());
            let val = match (val_as, field0_as) {
                (Some(s_as), Some(d_as)) if s_as != d_as => {
                    let cast_ty = llvm_export::types::PointerType::get(ctx, d_as).into();
                    let c = llvm::AddrSpaceCastOp::new(ctx, val, cast_ty);
                    rewriter.insert_operation(ctx, c.get_operation());
                    c.get_operation().deref(ctx).get_result(0)
                }
                _ => val,
            };

            let undef = llvm::UndefOp::new(ctx, llvm_ty);
            rewriter.insert_operation(ctx, undef.get_operation());
            let undef_val = undef.get_operation().deref(ctx).get_result(0);

            let insert_ptr = llvm::InsertValueOp::new(ctx, undef_val, val, vec![0]);
            rewriter.insert_operation(ctx, insert_ptr.get_operation());
            let with_ptr = insert_ptr.get_operation().deref(ctx).get_result(0);

            let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
            let len_apint = pliron::utils::apint::APInt::from_i64(
                len as i64,
                std::num::NonZeroUsize::new(64).unwrap(),
            );
            let len_attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, len_apint);
            let len_const = llvm::ConstantOp::new(ctx, Box::new(len_attr));
            rewriter.insert_operation(ctx, len_const.get_operation());
            let len_val = len_const.get_operation().deref(ctx).get_result(0);

            return Ok(llvm::InsertValueOp::new(ctx, with_ptr, len_val, vec![1]).get_operation());
        }
    }

    emit_pointer_cast(ctx, rewriter, op, val, val_ty, llvm_ty)
}

/// Return the data-pointer and metadata field types for the canonical slice-fat-pointer
/// LLVM shape `{ ptr, integer }`.
///
/// This recognizer is intentionally narrow. In the semantic pointer-cast path,
/// `{ ptr, integer }` is the slice fat-pointer representation, while trait-object
/// fat pointers use pointer metadata and therefore do not match. Requiring the
/// metadata type to remain identical in the caller prevents this path from
/// becoming a general aggregate coercion.
fn slice_fat_pointer_fields(
    ctx: &Context,
    ty: pliron::r#type::TypeHandle,
) -> Option<(pliron::r#type::TypeHandle, pliron::r#type::TypeHandle)> {
    let ty_ref = ty.deref(ctx);
    let struct_ty = ty_ref.downcast_ref::<llvm_export::types::StructType>()?;
    if struct_ty.num_fields() != 2 {
        return None;
    }

    let data_ty = struct_ty.field_type(0);
    let metadata_ty = struct_ty.field_type(1);
    let data_is_pointer = data_ty.deref(ctx).is::<llvm_export::types::PointerType>();
    let metadata_is_integer = metadata_ty.deref(ctx).is::<IntegerType>();
    (data_is_pointer && metadata_is_integer).then_some((data_ty, metadata_ty))
}

/// Lower a slice-fat-pointer cast entirely in SSA.
///
/// The data-pointer half carries address-space semantics and is converted with
/// `addrspacecast` when the source and destination spaces differ. The integer
/// metadata half is copied unchanged. No aggregate bytes are materialized in
/// memory, so an addrspace(3) pointer is never exposed as its target-dependent
/// physical representation.
fn try_emit_slice_fat_pointer_cast(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    val: pliron::value::Value,
    val_ty: pliron::r#type::TypeHandle,
    llvm_ty: pliron::r#type::TypeHandle,
) -> Result<Option<Ptr<Operation>>> {
    let Some((src_data_ty, src_metadata_ty)) = slice_fat_pointer_fields(ctx, val_ty) else {
        return Ok(None);
    };
    let Some((dst_data_ty, dst_metadata_ty)) = slice_fat_pointer_fields(ctx, llvm_ty) else {
        return Ok(None);
    };

    // Slice casts preserve metadata exactly. Refuse to reinterpret, widen, or
    // truncate the metadata field here; a non-identical metadata type is not
    // the canonical slice-fat-pointer coercion this helper recognizes.
    if src_metadata_ty != dst_metadata_ty {
        return Ok(None);
    }

    let src_data_as = src_data_ty
        .deref(ctx)
        .downcast_ref::<PointerType>()
        .expect("slice fat-pointer data field must be a pointer")
        .address_space();
    let dst_data_as = dst_data_ty
        .deref(ctx)
        .downcast_ref::<PointerType>()
        .expect("slice fat-pointer data field must be a pointer")
        .address_space();

    // Cluster-shared pointers have distinct lowering semantics and are intentionally
    // outside this legalization. Preserve the existing aggregate fallback for AS7.
    if src_data_as == address_space::CLUSTER_SHARED || dst_data_as == address_space::CLUSTER_SHARED
    {
        return Ok(None);
    }

    let extract_data = llvm::ExtractValueOp::new(ctx, val, vec![0])
        .map_err(|e| pliron::input_error_noloc!("slice pointer cast data extraction: {e}"))?;
    rewriter.insert_operation(ctx, extract_data.get_operation());
    let data = extract_data.get_operation().deref(ctx).get_result(0);

    let extract_metadata = llvm::ExtractValueOp::new(ctx, val, vec![1])
        .map_err(|e| pliron::input_error_noloc!("slice pointer cast metadata extraction: {e}"))?;
    rewriter.insert_operation(ctx, extract_metadata.get_operation());
    let metadata = extract_metadata.get_operation().deref(ctx).get_result(0);

    let data = if src_data_as != dst_data_as {
        let cast = llvm::AddrSpaceCastOp::new(ctx, data, dst_data_ty);
        rewriter.insert_operation(ctx, cast.get_operation());
        cast.get_operation().deref(ctx).get_result(0)
    } else {
        data
    };

    let undef = llvm::UndefOp::new(ctx, llvm_ty);
    rewriter.insert_operation(ctx, undef.get_operation());
    let undef_value = undef.get_operation().deref(ctx).get_result(0);

    let insert_data = llvm::InsertValueOp::new(ctx, undef_value, data, vec![0]);
    rewriter.insert_operation(ctx, insert_data.get_operation());
    let with_data = insert_data.get_operation().deref(ctx).get_result(0);

    Ok(Some(
        llvm::InsertValueOp::new(ctx, with_data, metadata, vec![1]).get_operation(),
    ))
}

/// Emit a pointer-compatible cast, handling the struct↔ptr patterns that arise
/// because our type system represents fat pointers (slices) as `{ ptr, i64 }` structs.
///
/// LLVM does not allow `bitcast` between structs and scalars/pointers, so:
/// - struct → ptr: `extractvalue` field 0 (extract data pointer from fat pointer)
/// - ptr → struct: `insertvalue` into undef at field 0 (wrap thin ptr in fat pointer)
/// - ptr → ptr (different address space): `addrspacecast`
/// - slice-shaped struct → slice-shaped struct: extract data + metadata,
///   address-space-cast the data pointer when needed, then rebuild in SSA
/// - array ↔ anything: memory round-trip (`alloca` + `store` + `load`),
///   because `bitcast` is only defined between non-aggregate first-class
///   types (e.g. `u32::from_ne_bytes` transmutes `[u8; 4]` → `u32`)
/// - otherwise: `bitcast`
fn emit_pointer_cast(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    _op: Ptr<Operation>,
    val: pliron::value::Value,
    val_ty: pliron::r#type::TypeHandle,
    llvm_ty: pliron::r#type::TypeHandle,
) -> Result<Ptr<Operation>> {
    let src_is_struct = val_ty.deref(ctx).is::<llvm_export::types::StructType>();
    let dst_is_struct = llvm_ty.deref(ctx).is::<llvm_export::types::StructType>();
    let src_as = val_ty
        .deref(ctx)
        .downcast_ref::<llvm_export::types::PointerType>()
        .map(|pt| pt.address_space());
    let dst_as = llvm_ty
        .deref(ctx)
        .downcast_ref::<llvm_export::types::PointerType>()
        .map(|pt| pt.address_space());
    let dst_is_ptr = dst_as.is_some();
    let src_is_ptr = src_as.is_some();
    let src_is_int = val_ty.deref(ctx).is::<IntegerType>();
    let src_is_array = val_ty.deref(ctx).is::<llvm_export::types::ArrayType>();
    let dst_is_array = llvm_ty.deref(ctx).is::<llvm_export::types::ArrayType>();

    if src_is_struct && dst_is_ptr {
        Ok(llvm::ExtractValueOp::new(ctx, val, vec![0])
            .map_err(|e| pliron::input_error_noloc!("pointer cast ExtractValueOp: {e}"))?
            .get_operation())
    } else if src_is_ptr && dst_is_struct {
        let undef = llvm::UndefOp::new(ctx, llvm_ty);
        rewriter.insert_operation(ctx, undef.get_operation());
        let undef_val = undef.get_operation().deref(ctx).get_result(0);
        Ok(llvm::InsertValueOp::new(ctx, undef_val, val, vec![0]).get_operation())
    } else if src_is_ptr && llvm_ty.deref(ctx).is::<IntegerType>() {
        Ok(emit_ptr_to_int(ctx, rewriter, val, val_ty, llvm_ty))
    } else if src_is_int && dst_is_ptr {
        Ok(emit_int_to_ptr(ctx, rewriter, val, llvm_ty))
    } else if src_is_struct && dst_is_struct {
        if let Some(cast) = try_emit_slice_fat_pointer_cast(ctx, rewriter, val, val_ty, llvm_ty)? {
            Ok(cast)
        } else {
            emit_transmute_via_memory(ctx, rewriter, val, val_ty, llvm_ty)
        }
    } else if let (Some(s), Some(d)) = (src_as, dst_as) {
        if s != d {
            let cast_ty = llvm_export::types::PointerType::get(ctx, d).into();
            Ok(llvm::AddrSpaceCastOp::new(ctx, val, cast_ty).get_operation())
        } else {
            Ok(llvm::BitcastOp::new(ctx, val, llvm_ty).get_operation())
        }
    } else if (src_is_int && dst_is_struct)
        || (src_is_struct && llvm_ty.deref(ctx).is::<IntegerType>())
        || src_is_array
        || dst_is_array
    {
        // Array on either side (e.g. `u32::from_ne_bytes` is a
        // `[u8; 4]` → `u32` Transmute, `u32::to_ne_bytes` the reverse).
        // LLVM's `bitcast` is only defined between non-aggregate
        // first-class types, so an aggregate must go through memory.
        emit_transmute_via_memory(ctx, rewriter, val, val_ty, llvm_ty)
    } else {
        Ok(llvm::BitcastOp::new(ctx, val, llvm_ty).get_operation())
    }
}

/// Lower Rust `Transmute` without assigning semantic meaning to aggregate
/// fields. Equal-size aggregates use memory; equal-bit-size scalar values use
/// LLVM's bit-preserving casts. This is what lets an integer transmute to a
/// physically represented enum (and a pointer to an Option-like enum) without
/// inventing or extracting a field-zero payload.
fn emit_transmute(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    val: pliron::value::Value,
    val_ty: pliron::r#type::TypeHandle,
    llvm_ty: pliron::r#type::TypeHandle,
) -> Result<Ptr<Operation>> {
    let src_ptr_as = val_ty
        .deref(ctx)
        .downcast_ref::<PointerType>()
        .map(PointerType::address_space);
    let dst_ptr_as = llvm_ty
        .deref(ctx)
        .downcast_ref::<PointerType>()
        .map(PointerType::address_space);
    let src_is_int = val_ty.deref(ctx).is::<IntegerType>();
    let dst_is_int = llvm_ty.deref(ctx).is::<IntegerType>();

    // `ptr::addr()` is implemented in the sysroot as a pointer-to-usize
    // transmute. For a named CUDA address space, Rust's logical address is
    // the generic CUDA address rather than the local offset in that space:
    // static shared memory can validly have offset zero without being null.
    // Genericize before ptrtoint so the memory-space identity participates in
    // the integer value. This also gives modern NVVM's 32-bit shared pointer
    // the 64-bit Rust `usize` representation expected by the MIR.
    if src_ptr_as.is_some() && dst_is_int {
        return Ok(emit_ptr_to_int(ctx, rewriter, val, val_ty, llvm_ty));
    }

    // A scalar transmute involving a shared-memory pointer bridges through
    // the generic address space, so the value Rust observes is the 64-bit
    // generic address, never a raw space-local offset (a valid static shared
    // allocation can have shared offset zero without being null). The scalar
    // arms below implement the bridge: integer → shared enters through
    // `emit_int_to_ptr` (inttoptr to generic, then addrspacecast), and
    // pointer ↔ pointer across spaces is a direct `addrspacecast`. Rust
    // layout sizes these (every pointer is 8 bytes on nvptx64), identically
    // under every backend and dialect mode.
    //
    // An aggregate transmute has no such bridge: its memory round-trip would
    // store the pointer's PHYSICAL bytes, whose width is target-mode
    // dependent (64-bit PTX/legacy, 32-bit modern NVVM `p3:32`) and not yet
    // chosen when this lowering runs. Those forms are rejected below, at the
    // points that would otherwise dispatch to the memory round-trip.
    let shared = llvm_export::types::address_space::SHARED;
    let involves_shared_pointer = llvm_type_contains_pointer_in_address_space(ctx, val_ty, shared)
        || llvm_type_contains_pointer_in_address_space(ctx, llvm_ty, shared);

    let (src_bytes, _) = llvm_type_size_align(ctx, val_ty).ok_or_else(|| {
        pliron::input_error_noloc!(
            "Transmute: cannot determine source layout for {}",
            val_ty.disp(ctx)
        )
    })?;
    let (dst_bytes, _) = llvm_type_size_align(ctx, llvm_ty).ok_or_else(|| {
        pliron::input_error_noloc!(
            "Transmute: cannot determine destination layout for {}",
            llvm_ty.disp(ctx)
        )
    })?;
    if src_bytes != dst_bytes {
        return pliron::input_err_noloc!(
            "Transmute size mismatch: source {} is {} bytes, destination {} is {} bytes",
            val_ty.disp(ctx),
            src_bytes,
            llvm_ty.disp(ctx),
            dst_bytes
        );
    }

    let is_aggregate = |ty: pliron::r#type::TypeHandle, ctx: &Context| {
        let ty = ty.deref(ctx);
        ty.is::<llvm_export::types::StructType>() || ty.is::<llvm_export::types::ArrayType>()
    };
    if is_aggregate(val_ty, ctx) || is_aggregate(llvm_ty, ctx) {
        if involves_shared_pointer {
            return reject_shared_pointer_memory_transmute(ctx, val_ty, llvm_ty);
        }
        return emit_transmute_via_memory(ctx, rewriter, val, val_ty, llvm_ty);
    }

    let src_bits = scalar_bit_width(ctx, val_ty).ok_or_else(|| {
        pliron::input_error_noloc!("Transmute: unsupported scalar source {}", val_ty.disp(ctx))
    })?;
    let dst_bits = scalar_bit_width(ctx, llvm_ty).ok_or_else(|| {
        pliron::input_error_noloc!(
            "Transmute: unsupported scalar destination {}",
            llvm_ty.disp(ctx)
        )
    })?;
    if src_bits != dst_bits {
        // `bool` is the important case: it is one byte in Rust memory but i1
        // in SSA. Its physical memory byte is the zero-extended 0 or 1, so
        // make that boundary explicit instead of relying on a type-punned
        // i1 store/load to define the rest of the byte.
        let src_integer_width = val_ty
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .map(IntegerType::width);
        let dst_integer_width = llvm_ty
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .map(IntegerType::width);
        if src_integer_width == Some(1) && dst_integer_width == Some(8) {
            return Ok(llvm::ZExtOp::new_with_nneg(ctx, val, llvm_ty, false).get_operation());
        }
        if src_integer_width == Some(8) && dst_integer_width == Some(1) {
            return Ok(llvm::TruncOp::new(ctx, val, llvm_ty).get_operation());
        }
        if involves_shared_pointer {
            return reject_shared_pointer_memory_transmute(ctx, val_ty, llvm_ty);
        }
        return emit_transmute_via_memory(ctx, rewriter, val, val_ty, llvm_ty);
    }

    match (src_ptr_as, dst_ptr_as) {
        (Some(source), Some(destination)) if source != destination => {
            Ok(llvm::AddrSpaceCastOp::new(ctx, val, llvm_ty).get_operation())
        }
        (Some(_), Some(_)) => Ok(llvm::BitcastOp::new(ctx, val, llvm_ty).get_operation()),
        (Some(_), None) if dst_is_int => Ok(emit_ptr_to_int(ctx, rewriter, val, val_ty, llvm_ty)),
        (None, Some(_)) if src_is_int => Ok(emit_int_to_ptr(ctx, rewriter, val, llvm_ty)),
        (Some(_), None) => {
            let integer_ty: pliron::r#type::TypeHandle =
                IntegerType::get(ctx, src_bits, Signedness::Signless).into();
            // Bridge through [`emit_ptr_to_int`] so a named-space pointer's
            // integer form is its generic address, not a raw local offset.
            let ptr_to_int = emit_ptr_to_int(ctx, rewriter, val, val_ty, integer_ty);
            rewriter.insert_operation(ctx, ptr_to_int);
            let integer = ptr_to_int.deref(ctx).get_result(0);
            Ok(llvm::BitcastOp::new(ctx, integer, llvm_ty).get_operation())
        }
        (None, Some(_)) => {
            let integer_ty: pliron::r#type::TypeHandle =
                IntegerType::get(ctx, dst_bits, Signedness::Signless).into();
            let bitcast = llvm::BitcastOp::new(ctx, val, integer_ty);
            rewriter.insert_operation(ctx, bitcast.get_operation());
            let integer = bitcast.get_operation().deref(ctx).get_result(0);
            Ok(emit_int_to_ptr(ctx, rewriter, integer, llvm_ty))
        }
        (None, None) => Ok(llvm::BitcastOp::new(ctx, val, llvm_ty).get_operation()),
    }
}

/// Reject a Transmute whose lowering would round-trip shared-pointer bytes
/// through memory.
///
/// A scalar shared pointer transmutes soundly because it bridges through the
/// generic address space, where its Rust-visible form is the 64-bit generic
/// address. A memory round-trip has no such bridge: it would store the
/// pointer's physical representation, whose width is target-mode dependent
/// (64-bit PTX/legacy, 32-bit modern NVVM `p3:32`) and not yet chosen when
/// this lowering runs, and the stored bytes would expose raw shared-local
/// offsets to Rust. Fail loudly rather than miscompile.
fn reject_shared_pointer_memory_transmute(
    ctx: &Context,
    val_ty: pliron::r#type::TypeHandle,
    llvm_ty: pliron::r#type::TypeHandle,
) -> Result<Ptr<Operation>> {
    pliron::input_err_noloc!(
        "Transmute from {} to {} would round-trip shared-memory (addrspace(3)) pointer bytes \
         through memory. Aggregate transmutes containing shared-memory pointers are \
         intentionally unsupported: the pointer's physical width is target-mode dependent \
         (64-bit PTX/legacy, 32-bit modern NVVM), and raw shared-local offsets must never \
         become Rust-visible bytes. Transmute the pointer as a scalar instead; it converts \
         through its 64-bit generic address.",
        val_ty.disp(ctx),
        llvm_ty.disp(ctx)
    )
}

fn scalar_bit_width(ctx: &Context, ty: pliron::r#type::TypeHandle) -> Option<u32> {
    let ty_ref = ty.deref(ctx);
    if let Some(integer) = ty_ref.downcast_ref::<IntegerType>() {
        return Some(integer.width());
    }
    if let Some(float) = type_cast::<dyn FloatTypeInterface>(&*ty_ref) {
        return u32::try_from(float.get_semantics().bits).ok();
    }
    if ty_ref.is::<llvm_export::types::PointerType>() {
        return Some(64);
    }
    drop(ty_ref);
    llvm_type_size_align(ctx, ty).and_then(|(bytes, _)| u32::try_from(bytes.checked_mul(8)?).ok())
}

/// Expose a pointer's CUDA generic address as an integer.
///
/// Named CUDA address spaces use offsets local to that space. In particular,
/// a valid static shared-memory allocation may have shared offset zero. Rust
/// raw-pointer APIs treat that allocation as non-null, so exposing the named
/// offset directly would make `ptr::is_null` and `ptr::as_mut` misclassify it.
/// Convert through the generic address space first, where CUDA encodes the
/// memory-space identity as part of the pointer value.
///
/// This is the Rust pointer-address boundary. Target intrinsics that explicitly
/// consume native shared offsets (for example, inline-PTX `ldmatrix`) keep
/// their direct address-space-specific conversion.
fn emit_ptr_to_int(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    val: pliron::value::Value,
    val_ty: pliron::r#type::TypeHandle,
    int_ty: pliron::r#type::TypeHandle,
) -> Ptr<Operation> {
    let Some(address_space) = val_ty
        .deref(ctx)
        .downcast_ref::<PointerType>()
        .map(PointerType::address_space)
    else {
        return llvm::PtrToIntOp::new(ctx, val, int_ty).get_operation();
    };

    let val = if address_space == 0 {
        val
    } else {
        let generic_ptr_ty = PointerType::get_generic(ctx);
        let cast = llvm::AddrSpaceCastOp::new(ctx, val, generic_ptr_ty.into());
        rewriter.insert_operation(ctx, cast.get_operation());
        cast.get_operation().deref(ctx).get_result(0)
    };

    llvm::PtrToIntOp::new(ctx, val, int_ty).get_operation()
}

/// Materialize an integer as a pointer, entering named address spaces
/// through the CUDA generic space.
///
/// The inverse boundary of [`emit_ptr_to_int`]: an exposed pointer address
/// is the CUDA generic address, so an integer cast back into a named
/// address space must `inttoptr` to generic first and then `addrspacecast`
/// into that space. A bare `inttoptr` into a named space would reinterpret
/// the generic address as a space-local offset and break the
/// expose/with-exposed-provenance round trip.
fn emit_int_to_ptr(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    val: pliron::value::Value,
    ptr_ty: pliron::r#type::TypeHandle,
) -> Ptr<Operation> {
    let address_space = ptr_ty
        .deref(ctx)
        .downcast_ref::<PointerType>()
        .map(PointerType::address_space);
    match address_space {
        None | Some(0) => llvm::IntToPtrOp::new(ctx, val, ptr_ty).get_operation(),
        Some(_) => {
            let generic_ptr_ty = PointerType::get_generic(ctx);
            let to_generic = llvm::IntToPtrOp::new(ctx, val, generic_ptr_ty.into());
            rewriter.insert_operation(ctx, to_generic.get_operation());
            let generic = to_generic.get_operation().deref(ctx).get_result(0);
            llvm::AddrSpaceCastOp::new(ctx, generic, ptr_ty).get_operation()
        }
    }
}

fn const_i64(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    n: i64,
) -> pliron::value::Value {
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let apint = pliron::utils::apint::APInt::from_i64(n, std::num::NonZeroUsize::new(64).unwrap());
    let attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, apint);
    let c = llvm::ConstantOp::new(ctx, Box::new(attr));
    rewriter.insert_operation(ctx, c.get_operation());
    c.get_operation().deref(ctx).get_result(0)
}

/// Equal-size Transmute through memory: `alloca` a stack slot, `store` the
/// source value into it, then `load` it back as the destination type.
///
/// This is the only valid lowering when either side is an aggregate,
/// because LLVM's `bitcast` is restricted to non-aggregate first-class
/// types (an aggregate bitcast such as `bitcast [4 x i8] %v to i32` is
/// rejected by `llc` with "invalid cast opcode"). The `opt -O2` middle
/// end folds the round-trip away, so no real stack traffic survives.
///
/// Guarded by a total-byte-size equality check so a size-mismatched
/// transmute fails loudly at compile time instead of silently truncating
/// the source or loading bytes that were never stored.
///
/// The stack slot is aligned to the larger of the two types' ABI
/// alignments. For `[u8; 4]` → `u32` the byte array alone would give the
/// slot align 1, making the 4-byte integer load under-aligned; raising
/// the slot to align 4 keeps both accesses natural. The chosen alignment
/// is stamped explicitly on all three ops so the textual exporter does
/// not fall back to each type's own (possibly smaller) natural alignment.
fn emit_transmute_via_memory(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    val: pliron::value::Value,
    val_ty: pliron::r#type::TypeHandle,
    llvm_ty: pliron::r#type::TypeHandle,
) -> Result<Ptr<Operation>> {
    let Some((src_bytes, src_align)) = llvm_type_size_align(ctx, val_ty) else {
        return pliron::input_err_noloc!(
            "Transmute via memory round-trip: cannot compute the total size of source type {}. \
             Refusing to lower (see issue #125).",
            val_ty.disp(ctx)
        );
    };
    let Some((dst_bytes, dst_align)) = llvm_type_size_align(ctx, llvm_ty) else {
        return pliron::input_err_noloc!(
            "Transmute via memory round-trip: cannot compute the total size of destination \
             type {}. Refusing to lower (see issue #125).",
            llvm_ty.disp(ctx)
        );
    };
    if src_bytes != dst_bytes {
        return pliron::input_err_noloc!(
            "aggregate Transmute size mismatch: source {} is {} bytes, destination {} is {} \
             bytes. Refusing the memory round-trip that would silently miscompile \
             (see issue #125).",
            val_ty.disp(ctx),
            src_bytes,
            llvm_ty.disp(ctx),
            dst_bytes
        );
    }

    // A memory round-trip can expose every stored byte through a different
    // destination type. Refuse a non-byte-faithful aggregate source: the
    // important example is `{ i1 }`, where LLVM may leave the upper seven
    // bits of Rust's one-byte `bool` storage undefined. The scalar-bool path
    // above explicitly zero-extends to i8, but recursively normalizing bools
    // and implicit padding inside arbitrary aggregates is not implemented.
    // A representation-identical aggregate-to-aggregate transmute remains
    // safe because the destination observes the same typed fields rather
    // than reinterpreting those unspecified bits.
    let source_is_aggregate = {
        let source = val_ty.deref(ctx);
        source.is::<llvm_export::types::StructType>()
            || source.is::<llvm_export::types::ArrayType>()
    };
    if source_is_aggregate && val_ty != llvm_ty && !llvm_type_is_byte_faithful(ctx, val_ty) {
        return pliron::input_err_noloc!(
            "Transmute via memory round-trip: source aggregate {} is not byte-faithful (for example, it may contain bool/i1 or implicit padding). Refusing to expose unspecified storage bits through destination {}.",
            val_ty.disp(ctx),
            llvm_ty.disp(ctx)
        );
    }

    // The slot must satisfy whichever side needs the stricter alignment.
    let align = u32::try_from(src_align.max(dst_align)).map_err(|_| {
        pliron::input_error_noloc!("Transmute alignment does not fit LLVM alignment metadata")
    })?;

    let is_i1 = |ty: pliron::r#type::TypeHandle, ctx: &Context| {
        ty.deref(ctx)
            .downcast_ref::<IntegerType>()
            .is_some_and(|integer| integer.width() == 1)
    };
    let source_is_i1 = is_i1(val_ty, ctx);
    let destination_is_i1 = is_i1(llvm_ty, ctx);

    let byte_ty: pliron::r#type::TypeHandle = IntegerType::get(ctx, 8, Signedness::Signless).into();

    // An SSA i1 has one byte of Rust storage, whose valid physical values are
    // exactly 0 and 1. Always cross that memory boundary as i8. This also
    // covers bool <-> one-byte aggregate transmutes.
    let stored_value = if source_is_i1 {
        let zext = llvm::ZExtOp::new_with_nneg(ctx, val, byte_ty, false);
        rewriter.insert_operation(ctx, zext.get_operation());
        zext.get_operation().deref(ctx).get_result(0)
    } else {
        val
    };
    let storage_ty = if source_is_i1 { byte_ty } else { val_ty };

    let one = const_i64(ctx, rewriter, 1);
    let alloca = llvm::AllocaOp::new(ctx, storage_ty, one, 0);
    llvm_export::ops::set_op_alignment(ctx, alloca.get_operation(), align);
    rewriter.insert_operation(ctx, alloca.get_operation());
    let ptr = alloca.get_operation().deref(ctx).get_result(0);

    let store = llvm::StoreOp::new(ctx, stored_value, ptr);
    llvm_export::ops::set_op_alignment(ctx, store.get_operation(), align);
    rewriter.insert_operation(ctx, store.get_operation());

    let loaded_ty = if destination_is_i1 { byte_ty } else { llvm_ty };
    let load = llvm::LoadOp::new(ctx, ptr, loaded_ty);
    llvm_export::ops::set_op_alignment(ctx, load.get_operation(), align);
    if destination_is_i1 {
        rewriter.insert_operation(ctx, load.get_operation());
        let byte = load.get_operation().deref(ctx).get_result(0);
        Ok(llvm::TruncOp::new(ctx, byte, llvm_ty).get_operation())
    } else {
        Ok(load.get_operation())
    }
}

/// Float → float: extend or truncate precision.
fn convert_float_to_float(
    ctx: &mut Context,
    _rewriter: &mut DialectConversionRewriter,
    val: pliron::value::Value,
    llvm_ty: pliron::r#type::TypeHandle,
    val_ty: pliron::r#type::TypeHandle,
) -> Result<Ptr<Operation>> {
    let src_width = float_bit_width(ctx, val_ty)?;
    let dst_width = float_bit_width(ctx, llvm_ty)?;

    let flags_key: pliron::identifier::Identifier = "llvm_fast_math_flags".try_into().unwrap();
    let flags = llvm_export::attributes::FastmathFlagsAttr::default();

    if src_width < dst_width {
        let op = llvm::FPExtOp::new(ctx, val, llvm_ty);
        op.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(flags_key, flags);
        Ok(op.get_operation())
    } else if src_width > dst_width {
        let op = llvm::FPTruncOp::new(ctx, val, llvm_ty);
        op.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(flags_key, flags);
        Ok(op.get_operation())
    } else {
        Ok(llvm::BitcastOp::new(ctx, val, llvm_ty).get_operation())
    }
}

fn float_bit_width(ctx: &Context, ty: pliron::r#type::TypeHandle) -> Result<usize> {
    let ty_ref = ty.deref(ctx);
    let Some(float_ty) = type_cast::<dyn FloatTypeInterface>(&*ty_ref) else {
        return pliron::input_err_noloc!("expected floating-point type");
    };
    Ok(float_ty.get_semantics().bits)
}

#[cfg(test)]
// Tests build kinded fixture types directly; production minting lives in mir-importer's facts.rs.
#[allow(clippy::disallowed_methods)]
mod tests {
    use crate::convert::ops::test_util::*;
    use dialect_mir::attributes::{MirCastKindAttr, MirPointerKindAuthorityAttr};
    use dialect_mir::ops as mir;
    use dialect_mir::types::{
        EnumCarrierKind, EnumEncoding, EnumLayoutKind, EnumVariant, MirEnumType, MirPointerKind,
        MirPtrType, MirStructType,
    };
    use llvm_export::ops as llvm;
    use pliron::builtin::op_interfaces::{CallOpCallable, CallOpInterface, SymbolOpInterface};
    use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
    use pliron::context::{Context, Ptr};
    use pliron::linked_list::ContainsLinkedList;
    use pliron::op::Op;
    use pliron::operation::Operation;
    use pliron::r#type::{TypeHandle, Typed};

    fn int_ty(ctx: &mut Context, width: u32, signedness: Signedness) -> TypeHandle {
        IntegerType::get(ctx, width, signedness).into()
    }

    fn build_single_cast(
        ctx: &mut Context,
        src_ty: TypeHandle,
        dst_ty: TypeHandle,
        kind: MirCastKindAttr,
    ) -> Ptr<Operation> {
        let (module_ptr, block) = build_kernel(ctx, vec![src_ty], vec![dst_ty]);
        let arg = block.deref(ctx).get_argument(0);

        let cast_op = Operation::new(
            ctx,
            mir::MirCastOp::get_concrete_op_info(),
            vec![dst_ty],
            vec![arg],
            vec![],
            0,
        );
        let cast = mir::MirCastOp::new(cast_op);
        cast.set_attr_cast_kind(ctx, kind.clone());
        if dialect_mir::types::type_contains_concrete_pointer_kind(ctx, dst_ty)
            || (kind == MirCastKindAttr::Transmute
                && !dialect_mir::types::pointer_kinds_in_type(ctx, dst_ty).is_empty())
        {
            cast.set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::RustCast);
        }
        cast_op.insert_at_back(block, ctx);

        let cast_result = cast_op.deref(ctx).get_result(0);
        append_mir_return(ctx, block, vec![cast_result]);

        module_ptr
    }

    fn lower_single_cast(
        ctx: &mut Context,
        src_ty: TypeHandle,
        dst_ty: TypeHandle,
        kind: MirCastKindAttr,
    ) -> Ptr<Operation> {
        let module_ptr = build_single_cast(ctx, src_ty, dst_ty, kind);
        crate::lower_mir_to_llvm(ctx, module_ptr).expect("lowering failed");
        module_ptr
    }

    fn assert_cast_lowered_to<T: Op>(ctx: &Context, module_ptr: Ptr<Operation>, expected: &str) {
        let body = kernel_blocks(ctx, module_ptr);
        assert_eq!(
            count_ops::<T>(ctx, &body),
            1,
            "expected exactly one {expected}"
        );
        assert_eq!(
            count_ops::<mir::MirCastOp>(ctx, &body),
            0,
            "mir.cast must be replaced during lowering"
        );
    }

    fn module_has_llvm_func(ctx: &Context, module_ptr: Ptr<Operation>, symbol: &str) -> bool {
        let top = module_top_block(ctx, module_ptr);
        top.deref(ctx)
            .iter(ctx)
            .filter_map(|op| Operation::get_op::<llvm::FuncOp>(op, ctx))
            .any(|func| func.get_symbol_name(ctx).to_string() == symbol)
    }

    fn assert_single_direct_intrinsic_call(
        ctx: &Context,
        module_ptr: Ptr<Operation>,
        symbol: &str,
        description: &str,
    ) {
        let calls = find_all::<llvm::CallOp>(ctx, &kernel_blocks(ctx, module_ptr));
        let [call] = calls.as_slice() else {
            panic!("{description} must lower to exactly one llvm.call");
        };
        let CallOpCallable::Direct(callee) = call.callee(ctx) else {
            panic!("{description} must use a direct intrinsic call");
        };
        assert_eq!(
            callee.to_string(),
            symbol,
            "{description} must call {symbol}"
        );
        assert!(
            module_has_llvm_func(ctx, module_ptr, symbol),
            "{description} must declare {symbol}"
        );
    }

    fn pointer_addrspace(ctx: &Context, ty: TypeHandle) -> u32 {
        ty.deref(ctx)
            .downcast_ref::<llvm_export::types::PointerType>()
            .expect("expected LLVM pointer type")
            .address_space()
    }

    fn slice_like_pointer_struct(
        ctx: &mut Context,
        name: &str,
        element_width: u32,
        address_space: u32,
    ) -> TypeHandle {
        let element = int_ty(ctx, element_width, Signedness::Unsigned);
        let data: TypeHandle = MirPtrType::get(ctx, element, true, address_space).into();
        let metadata = int_ty(ctx, 64, Signedness::Unsigned);
        MirStructType::get_with_full_layout(
            ctx,
            name.into(),
            vec!["data".into(), "len".into()],
            vec![data, metadata],
            vec![0, 1],
            vec![0, 8],
            16,
            8,
        )
        .into()
    }

    fn integer_niche_enum(ctx: &mut Context) -> TypeHandle {
        let u8_ty: TypeHandle = int_ty(ctx, 8, Signedness::Unsigned);
        let u32_ty: TypeHandle = int_ty(ctx, 32, Signedness::Unsigned);
        MirEnumType::get_with_encoding(
            ctx,
            "MaybeNonZero".into(),
            u8_ty,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![u32_ty], vec![0], vec![4]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 4,
                abi_align: 4,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Integer,
                carrier_width: 32,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into()
    }

    fn pointer_niche_enum(ctx: &mut Context, pointer: TypeHandle) -> TypeHandle {
        let u8_ty: TypeHandle = int_ty(ctx, 8, Signedness::Unsigned);
        MirEnumType::get_with_encoding(
            ctx,
            "MaybeRef".into(),
            u8_ty,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![pointer], vec![0], vec![8]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 8,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Pointer,
                carrier_width: 64,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into()
    }

    fn assert_physical_transmute_round_trip(ctx: &Context, module: Ptr<Operation>) {
        let body = kernel_blocks(ctx, module);
        assert_eq!(count_ops::<llvm::AllocaOp>(ctx, &body), 1);
        assert_eq!(count_ops::<llvm::StoreOp>(ctx, &body), 1);
        assert_eq!(count_ops::<llvm::LoadOp>(ctx, &body), 1);
        assert_eq!(count_ops::<llvm::ExtractValueOp>(ctx, &body), 0);
        assert_eq!(count_ops::<llvm::InsertValueOp>(ctx, &body), 0);
    }

    #[test]
    fn transmute_between_integer_and_niche_enum_preserves_physical_bytes() {
        for integer_to_enum in [true, false] {
            let mut ctx = make_ctx();
            let integer = int_ty(&mut ctx, 32, Signedness::Unsigned);
            let enum_ty = integer_niche_enum(&mut ctx);
            let (source, destination) = if integer_to_enum {
                (integer, enum_ty)
            } else {
                (enum_ty, integer)
            };
            let module =
                lower_single_cast(&mut ctx, source, destination, MirCastKindAttr::Transmute);
            assert_physical_transmute_round_trip(&ctx, module);
        }
    }

    #[test]
    fn transmute_between_pointer_and_reference_niche_enum_is_not_field_coercion() {
        for pointer_to_enum in [true, false] {
            let mut ctx = make_ctx();
            let pointee = int_ty(&mut ctx, 32, Signedness::Unsigned);
            let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, pointee, false).into();
            let enum_ty = pointer_niche_enum(&mut ctx, pointer);
            let (source, destination) = if pointer_to_enum {
                (pointer, enum_ty)
            } else {
                (enum_ty, pointer)
            };
            let module =
                lower_single_cast(&mut ctx, source, destination, MirCastKindAttr::Transmute);
            assert_physical_transmute_round_trip(&ctx, module);
        }
    }

    #[test]
    fn pointer_kind_only_coercion_has_no_llvm_instruction() {
        let mut ctx = make_ctx();
        let pointee = int_ty(&mut ctx, 32, Signedness::Unsigned);
        let raw_mut: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, pointee, true, MirPointerKind::RawMut)
                .into();
        let raw_const: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, pointee, false, MirPointerKind::RawConst)
                .into();

        let module = lower_single_cast(
            &mut ctx,
            raw_mut,
            raw_const,
            MirCastKindAttr::PointerCoercionMutToConst,
        );
        let body = kernel_blocks(&ctx, module);

        assert_eq!(count_ops::<mir::MirCastOp>(&ctx, &body), 0);
        assert_eq!(count_ops::<llvm::BitcastOp>(&ctx, &body), 0);
        assert_eq!(count_ops::<llvm::AddrSpaceCastOp>(&ctx, &body), 0);
        assert_eq!(count_ops::<llvm::AllocaOp>(&ctx, &body), 0);
        assert_eq!(count_ops::<llvm::StoreOp>(&ctx, &body), 0);
        assert_eq!(count_ops::<llvm::LoadOp>(&ctx, &body), 0);
    }

    #[test]
    fn lowering_rejects_pointer_kind_laundering_before_erasure() {
        let mut ctx = make_ctx();
        let pointee = int_ty(&mut ctx, 32, Signedness::Unsigned);
        let erased: TypeHandle = MirPtrType::get_generic(&mut ctx, pointee, true).into();
        let unique: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, pointee, true, MirPointerKind::UniqueRef)
                .into();
        let (module, block) = build_kernel(&mut ctx, vec![erased], vec![unique]);
        let source = block.deref(&ctx).get_argument(0);
        let cast_op = Operation::new(
            &mut ctx,
            mir::MirCastOp::get_concrete_op_info(),
            vec![unique],
            vec![source],
            vec![],
            0,
        );
        mir::MirCastOp::new(cast_op).set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
        cast_op.insert_at_back(block, &ctx);
        let cast_result = cast_op.deref(&ctx).get_result(0);
        append_mir_return(&mut ctx, block, vec![cast_result]);

        let error = crate::lower_mir_to_llvm(&mut ctx, module)
            .expect_err("lowering must verify pointer kinds before erasing them");
        assert!(
            error
                .to_string()
                .contains("without an explicit Rust pointer-kind authority"),
            "unexpected diagnostic: {error}"
        );
    }

    #[test]
    fn transmute_between_bool_and_byte_uses_explicit_integer_casts() {
        for bool_to_byte in [true, false] {
            let mut ctx = make_ctx();
            let boolean = int_ty(&mut ctx, 1, Signedness::Signless);
            let byte = int_ty(&mut ctx, 8, Signedness::Unsigned);
            let (source, destination) = if bool_to_byte {
                (boolean, byte)
            } else {
                (byte, boolean)
            };

            let module =
                lower_single_cast(&mut ctx, source, destination, MirCastKindAttr::Transmute);
            let body = kernel_blocks(&ctx, module);
            assert_eq!(count_ops::<llvm::AllocaOp>(&ctx, &body), 0);
            assert_eq!(count_ops::<llvm::StoreOp>(&ctx, &body), 0);
            assert_eq!(count_ops::<llvm::LoadOp>(&ctx, &body), 0);
            assert_eq!(
                count_ops::<llvm::ZExtOp>(&ctx, &body),
                usize::from(bool_to_byte)
            );
            assert_eq!(
                count_ops::<llvm::TruncOp>(&ctx, &body),
                usize::from(!bool_to_byte)
            );
        }
    }

    #[test]
    fn transmute_between_bool_and_byte_array_materializes_the_physical_byte() {
        for bool_to_bytes in [true, false] {
            let mut ctx = make_ctx();
            let boolean = int_ty(&mut ctx, 1, Signedness::Signless);
            let byte = int_ty(&mut ctx, 8, Signedness::Unsigned);
            let bytes: TypeHandle = dialect_mir::types::MirArrayType::get(&mut ctx, byte, 1).into();
            let (source, destination) = if bool_to_bytes {
                (boolean, bytes)
            } else {
                (bytes, boolean)
            };

            let module =
                lower_single_cast(&mut ctx, source, destination, MirCastKindAttr::Transmute);
            let body = kernel_blocks(&ctx, module);
            assert_eq!(count_ops::<llvm::AllocaOp>(&ctx, &body), 1);
            assert_eq!(count_ops::<llvm::StoreOp>(&ctx, &body), 1);
            assert_eq!(count_ops::<llvm::LoadOp>(&ctx, &body), 1);
            assert_eq!(
                count_ops::<llvm::ZExtOp>(&ctx, &body),
                usize::from(bool_to_bytes)
            );
            assert_eq!(
                count_ops::<llvm::TruncOp>(&ctx, &body),
                usize::from(!bool_to_bytes)
            );

            let stores = find_all::<llvm::StoreOp>(&ctx, &body);
            let stored_width = stores[0]
                .get_operand_value(&ctx)
                .get_type(&ctx)
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .map(IntegerType::width);
            if bool_to_bytes {
                assert_eq!(stored_width, Some(8), "the memory store must be i8");
            }
            let loads = find_all::<llvm::LoadOp>(&ctx, &body);
            let loaded_width = loads[0]
                .get_operation()
                .deref(&ctx)
                .get_result(0)
                .get_type(&ctx)
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .map(IntegerType::width);
            if !bool_to_bytes {
                assert_eq!(loaded_width, Some(8), "the memory load must be i8");
            }
        }
    }

    #[test]
    fn transmute_from_bool_wrapper_rejects_unspecified_upper_byte_bits() {
        let mut ctx = make_ctx();
        let boolean = int_ty(&mut ctx, 1, Signedness::Signless);
        let wrapper: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "BoolWrapper".into(),
            vec!["value".into()],
            vec![boolean],
            vec![0],
            vec![0],
            1,
            1,
        )
        .into();
        let byte = int_ty(&mut ctx, 8, Signedness::Unsigned);
        let module = build_single_cast(&mut ctx, wrapper, byte, MirCastKindAttr::Transmute);

        let error = crate::lower_mir_to_llvm(&mut ctx, module)
            .expect_err("aggregate bool storage must not be exposed as raw bytes");
        assert!(
            error.to_string().contains("not byte-faithful"),
            "unexpected diagnostic: {error}"
        );
    }

    #[test]
    fn integer_to_shared_pointer_transmute_enters_through_generic_space() {
        let mut ctx = make_ctx();
        let pointee = int_ty(&mut ctx, 32, Signedness::Unsigned);
        let shared_pointer: TypeHandle = MirPtrType::get(&mut ctx, pointee, false, 3).into();
        let usize_ty = int_ty(&mut ctx, 64, Signedness::Unsigned);

        let module = lower_single_cast(
            &mut ctx,
            usize_ty,
            shared_pointer,
            MirCastKindAttr::Transmute,
        );

        // The exposed integer is a generic address, so re-entering the shared
        // space must go inttoptr → generic, then addrspacecast → shared. A
        // bare inttoptr into addrspace(3) would reinterpret the generic
        // address as a space-local offset.
        assert_cast_lowered_to::<llvm::AddrSpaceCastOp>(&ctx, module, "llvm.addrspacecast");
        let body = kernel_blocks(&ctx, module);
        let int_to_ptrs = find_all::<llvm::IntToPtrOp>(&ctx, &body);
        let [int_to_ptr] = int_to_ptrs.as_slice() else {
            panic!("integer-to-shared transmute must lower to exactly one llvm.inttoptr");
        };
        let generic = int_to_ptr.get_operation().deref(&ctx).get_result(0);
        assert_eq!(
            pointer_addrspace(&ctx, generic.get_type(&ctx)),
            0,
            "inttoptr must land in the generic address space"
        );
        let casts = find_all::<llvm::AddrSpaceCastOp>(&ctx, &body);
        let recovered = casts[0].get_operation().deref(&ctx).get_result(0);
        assert_eq!(
            pointer_addrspace(&ctx, recovered.get_type(&ctx)),
            3,
            "addrspacecast must restore the shared address space"
        );
    }

    #[test]
    fn shared_pointer_transmutes_between_address_spaces_use_addrspacecast() {
        for shared_is_source in [true, false] {
            let mut ctx = make_ctx();
            let pointee = int_ty(&mut ctx, 32, Signedness::Unsigned);
            let shared_pointer: TypeHandle = MirPtrType::get(&mut ctx, pointee, false, 3).into();
            let generic_pointer: TypeHandle =
                MirPtrType::get_generic(&mut ctx, pointee, false).into();
            let (source, destination) = if shared_is_source {
                (shared_pointer, generic_pointer)
            } else {
                (generic_pointer, shared_pointer)
            };

            let module =
                lower_single_cast(&mut ctx, source, destination, MirCastKindAttr::Transmute);

            assert_cast_lowered_to::<llvm::AddrSpaceCastOp>(&ctx, module, "llvm.addrspacecast");
            let body = kernel_blocks(&ctx, module);
            assert_eq!(
                count_ops::<llvm::AllocaOp>(&ctx, &body),
                0,
                "pointer-to-pointer transmute must not round-trip through memory"
            );
            let casts = find_all::<llvm::AddrSpaceCastOp>(&ctx, &body);
            let result = casts[0].get_operation().deref(&ctx).get_result(0);
            let expected_space = if shared_is_source { 0 } else { 3 };
            assert_eq!(
                pointer_addrspace(&ctx, result.get_type(&ctx)),
                expected_space,
                "addrspacecast must produce the destination address space"
            );
        }
    }

    #[test]
    fn aggregate_transmutes_containing_shared_pointers_are_rejected() {
        for (wrap_pointer, pointer_is_source) in [(true, false), (true, true), (false, true)] {
            let mut ctx = make_ctx();
            let pointee = int_ty(&mut ctx, 32, Signedness::Unsigned);
            let shared_pointer: TypeHandle = MirPtrType::get(&mut ctx, pointee, false, 3).into();
            let pointer_shape: TypeHandle = if wrap_pointer {
                MirStructType::get_with_full_layout(
                    &mut ctx,
                    "SharedPointerWrapper".into(),
                    vec!["pointer".into()],
                    vec![shared_pointer],
                    vec![0],
                    vec![0],
                    8,
                    8,
                )
                .into()
            } else {
                shared_pointer
            };
            let byte = int_ty(&mut ctx, 8, Signedness::Unsigned);
            let other_side: TypeHandle = if wrap_pointer {
                // Wrapped pointer against a plain integer: the wrapper is the
                // aggregate that forces the memory round-trip.
                int_ty(&mut ctx, 64, Signedness::Unsigned)
            } else {
                // Bare shared pointer against a byte array: the array is the
                // aggregate that forces the memory round-trip.
                dialect_mir::types::MirArrayType::get(&mut ctx, byte, 8).into()
            };
            let (source, destination) = if pointer_is_source {
                (pointer_shape, other_side)
            } else {
                (other_side, pointer_shape)
            };
            let module =
                build_single_cast(&mut ctx, source, destination, MirCastKindAttr::Transmute);

            let error = crate::lower_mir_to_llvm(&mut ctx, module).expect_err(
                "a transmute that would memory-round-trip shared-pointer bytes must fail",
            );
            assert!(
                error.to_string().contains("intentionally unsupported"),
                "unexpected diagnostic: {error}"
            );
        }
    }

    #[test]
    fn shared_pointer_to_integer_transmute_genericizes_before_ptr_to_int() {
        let mut ctx = make_ctx();
        let pointee = int_ty(&mut ctx, 32, Signedness::Unsigned);
        let shared_pointer: TypeHandle = MirPtrType::get(&mut ctx, pointee, false, 3).into();
        let usize_ty = int_ty(&mut ctx, 64, Signedness::Unsigned);

        let module = lower_single_cast(
            &mut ctx,
            shared_pointer,
            usize_ty,
            MirCastKindAttr::Transmute,
        );

        assert_cast_lowered_to::<llvm::PtrToIntOp>(&ctx, module, "llvm.ptrtoint");
        assert_eq!(
            count_ops::<llvm::AddrSpaceCastOp>(&ctx, &kernel_blocks(&ctx, module)),
            1,
            "shared pointers must become generic before ptr::addr() observes them"
        );
    }

    #[test]
    fn int_to_int_signed_widen_lowers_to_s_ext() {
        let mut ctx = make_ctx();
        let i8_ty = int_ty(&mut ctx, 8, Signedness::Signed);
        let i32_ty = int_ty(&mut ctx, 32, Signedness::Signed);

        let module_ptr = lower_single_cast(&mut ctx, i8_ty, i32_ty, MirCastKindAttr::IntToInt);

        assert_cast_lowered_to::<llvm::SExtOp>(&ctx, module_ptr, "llvm.sext");
    }

    #[test]
    fn int_to_int_unsigned_widen_lowers_to_z_ext() {
        let mut ctx = make_ctx();
        let u8_ty = int_ty(&mut ctx, 8, Signedness::Unsigned);
        let u32_ty = int_ty(&mut ctx, 32, Signedness::Unsigned);

        let module_ptr = lower_single_cast(&mut ctx, u8_ty, u32_ty, MirCastKindAttr::IntToInt);

        assert_cast_lowered_to::<llvm::ZExtOp>(&ctx, module_ptr, "llvm.zext");
    }

    #[test]
    fn int_to_int_narrow_lowers_to_trunc() {
        let mut ctx = make_ctx();
        let i32_ty = int_ty(&mut ctx, 32, Signedness::Signed);
        let i8_ty = int_ty(&mut ctx, 8, Signedness::Signed);

        let module_ptr = lower_single_cast(&mut ctx, i32_ty, i8_ty, MirCastKindAttr::IntToInt);

        assert_cast_lowered_to::<llvm::TruncOp>(&ctx, module_ptr, "llvm.trunc");
    }

    #[test]
    fn int_to_float_unsigned_lowers_to_ui_to_fp() {
        let mut ctx = make_ctx();
        let u32_ty = int_ty(&mut ctx, 32, Signedness::Unsigned);
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();

        let module_ptr = lower_single_cast(&mut ctx, u32_ty, f32_ty, MirCastKindAttr::IntToFloat);

        assert_cast_lowered_to::<llvm::UIToFPOp>(&ctx, module_ptr, "llvm.uitofp");
    }

    #[test]
    fn float_to_int_signed_lowers_to_saturating_intrinsic_call() {
        let mut ctx = make_ctx();
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let i32_ty = int_ty(&mut ctx, 32, Signedness::Signed);

        let module_ptr = lower_single_cast(&mut ctx, f32_ty, i32_ty, MirCastKindAttr::FloatToInt);

        assert_cast_lowered_to::<llvm::CallOp>(&ctx, module_ptr, "llvm.call");
        assert_single_direct_intrinsic_call(
            &ctx,
            module_ptr,
            "llvm_fptosi_sat_i32_f32",
            "f32 -> i32 signed cast",
        );
    }

    #[test]
    fn pointer_expose_address_lowers_to_ptr_to_int() {
        let mut ctx = make_ctx();
        let pointee_ty = int_ty(&mut ctx, 32, Signedness::Signless);
        let ptr_ty: TypeHandle = MirPtrType::get(&mut ctx, pointee_ty, false, 0).into();
        let usize_ty = int_ty(&mut ctx, 64, Signedness::Unsigned);

        let module_ptr = lower_single_cast(
            &mut ctx,
            ptr_ty,
            usize_ty,
            MirCastKindAttr::PointerExposeAddress,
        );

        assert_cast_lowered_to::<llvm::PtrToIntOp>(&ctx, module_ptr, "llvm.ptrtoint");
        assert_eq!(
            count_ops::<llvm::AddrSpaceCastOp>(&ctx, &kernel_blocks(&ctx, module_ptr)),
            0,
            "generic pointers need no address-space conversion"
        );
    }

    #[test]
    fn named_space_pointer_expose_address_genericizes_before_ptr_to_int() {
        for address_space in [
            llvm_export::types::address_space::GLOBAL,
            llvm_export::types::address_space::SHARED,
            llvm_export::types::address_space::CONSTANT,
            llvm_export::types::address_space::LOCAL,
        ] {
            let mut ctx = make_ctx();
            let pointee_ty = int_ty(&mut ctx, 32, Signedness::Signless);
            let ptr_ty: TypeHandle =
                MirPtrType::get(&mut ctx, pointee_ty, false, address_space).into();
            let usize_ty = int_ty(&mut ctx, 64, Signedness::Unsigned);

            let module_ptr = lower_single_cast(
                &mut ctx,
                ptr_ty,
                usize_ty,
                MirCastKindAttr::PointerExposeAddress,
            );

            assert_cast_lowered_to::<llvm::PtrToIntOp>(&ctx, module_ptr, "llvm.ptrtoint");
            assert_eq!(
                count_ops::<llvm::AddrSpaceCastOp>(&ctx, &kernel_blocks(&ctx, module_ptr)),
                1,
                "address space {address_space} must become generic before exposing its address"
            );
        }
    }

    #[test]
    fn int_to_int_same_width_lowers_to_bitcast() {
        let mut ctx = make_ctx();
        let i32_ty = int_ty(&mut ctx, 32, Signedness::Signed);
        let u32_ty = int_ty(&mut ctx, 32, Signedness::Unsigned);

        let module_ptr = lower_single_cast(&mut ctx, i32_ty, u32_ty, MirCastKindAttr::IntToInt);

        assert_cast_lowered_to::<llvm::BitcastOp>(&ctx, module_ptr, "llvm.bitcast");
    }

    #[test]
    fn int_to_float_signed_lowers_to_si_to_fp() {
        let mut ctx = make_ctx();
        let i32_ty = int_ty(&mut ctx, 32, Signedness::Signed);
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();

        let module_ptr = lower_single_cast(&mut ctx, i32_ty, f32_ty, MirCastKindAttr::IntToFloat);

        assert_cast_lowered_to::<llvm::SIToFPOp>(&ctx, module_ptr, "llvm.sitofp");
    }

    #[test]
    fn int_to_float_rejects_integer_source_wider_than_64_bits() {
        let mut ctx = make_ctx();
        let i128_ty = int_ty(&mut ctx, 128, Signedness::Signed);
        let f64_ty: TypeHandle = FP64Type::get(&ctx).into();

        let module = build_single_cast(&mut ctx, i128_ty, f64_ty, MirCastKindAttr::IntToFloat);
        let error = crate::lower_mir_to_llvm(&mut ctx, module).expect_err(
            "i128 -> f64 cast must be rejected before an unsupported soft-float helper is emitted",
        );
        assert!(
            error.to_string().contains("wider than 64 bits"),
            "unexpected diagnostic: {error}"
        );
    }

    #[test]
    fn int_to_float_rejects_unsigned_integer_source_wider_than_64_bits() {
        let mut ctx = make_ctx();
        let u128_ty = int_ty(&mut ctx, 128, Signedness::Unsigned);
        let f64_ty: TypeHandle = FP64Type::get(&ctx).into();

        let module = build_single_cast(&mut ctx, u128_ty, f64_ty, MirCastKindAttr::IntToFloat);
        let error = crate::lower_mir_to_llvm(&mut ctx, module).expect_err(
            "u128 -> f64 cast must be rejected before an unsupported soft-float helper is emitted",
        );
        assert!(
            error.to_string().contains("wider than 64 bits"),
            "unexpected diagnostic: {error}"
        );
    }

    #[test]
    fn float_to_int_unsigned_lowers_to_unsigned_saturating_intrinsic_call() {
        let mut ctx = make_ctx();
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let u32_ty = int_ty(&mut ctx, 32, Signedness::Unsigned);

        let module_ptr = lower_single_cast(&mut ctx, f32_ty, u32_ty, MirCastKindAttr::FloatToInt);

        assert_cast_lowered_to::<llvm::CallOp>(&ctx, module_ptr, "llvm.call");
        assert_single_direct_intrinsic_call(
            &ctx,
            module_ptr,
            "llvm_fptoui_sat_i32_f32",
            "f32 -> u32 unsigned cast",
        );
    }

    #[test]
    fn float_to_int_rejects_integer_destination_wider_than_64_bits() {
        let mut ctx = make_ctx();
        let f64_ty: TypeHandle = FP64Type::get(&ctx).into();
        let i128_ty = int_ty(&mut ctx, 128, Signedness::Signed);

        let module = build_single_cast(&mut ctx, f64_ty, i128_ty, MirCastKindAttr::FloatToInt);
        let error = crate::lower_mir_to_llvm(&mut ctx, module).expect_err(
            "f64 -> i128 saturating cast must be rejected before an unsupported intrinsic is emitted",
        );
        assert!(
            error.to_string().contains("wider than 64 bits"),
            "unexpected diagnostic: {error}"
        );
    }

    #[test]
    fn float_to_int_rejects_unsigned_integer_destination_wider_than_64_bits() {
        let mut ctx = make_ctx();
        let f64_ty: TypeHandle = FP64Type::get(&ctx).into();
        let u128_ty = int_ty(&mut ctx, 128, Signedness::Unsigned);

        let module = build_single_cast(&mut ctx, f64_ty, u128_ty, MirCastKindAttr::FloatToInt);
        let error = crate::lower_mir_to_llvm(&mut ctx, module).expect_err(
            "f64 -> u128 saturating cast must be rejected before an unsupported intrinsic is emitted",
        );
        assert!(
            error.to_string().contains("wider than 64 bits"),
            "unexpected diagnostic: {error}"
        );
    }

    #[test]
    fn float_to_float_widen_lowers_to_fp_ext() {
        let mut ctx = make_ctx();
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let f64_ty: TypeHandle = FP64Type::get(&ctx).into();

        let module_ptr = lower_single_cast(&mut ctx, f32_ty, f64_ty, MirCastKindAttr::FloatToFloat);

        assert_cast_lowered_to::<llvm::FPExtOp>(&ctx, module_ptr, "llvm.fpext");
    }

    #[test]
    fn float_to_float_narrow_lowers_to_fp_trunc() {
        let mut ctx = make_ctx();
        let f64_ty: TypeHandle = FP64Type::get(&ctx).into();
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();

        let module_ptr = lower_single_cast(&mut ctx, f64_ty, f32_ty, MirCastKindAttr::FloatToFloat);

        assert_cast_lowered_to::<llvm::FPTruncOp>(&ctx, module_ptr, "llvm.fptrunc");
    }

    #[test]
    fn pointer_with_exposed_provenance_lowers_to_int_to_ptr() {
        let mut ctx = make_ctx();
        let usize_ty = int_ty(&mut ctx, 64, Signedness::Unsigned);
        let pointee_ty = int_ty(&mut ctx, 32, Signedness::Signless);
        let ptr_ty: TypeHandle =
            MirPtrType::get_with_kind(&mut ctx, pointee_ty, false, 0, MirPointerKind::RawConst)
                .into();

        let module_ptr = lower_single_cast(
            &mut ctx,
            usize_ty,
            ptr_ty,
            MirCastKindAttr::PointerWithExposedProvenance,
        );

        assert_cast_lowered_to::<llvm::IntToPtrOp>(&ctx, module_ptr, "llvm.inttoptr");
        assert_eq!(
            count_ops::<llvm::AddrSpaceCastOp>(&ctx, &kernel_blocks(&ctx, module_ptr)),
            0,
            "generic pointers need no address-space conversion"
        );
    }

    #[test]
    fn named_space_pointer_with_exposed_provenance_enters_through_generic() {
        for address_space in [
            llvm_export::types::address_space::GLOBAL,
            llvm_export::types::address_space::SHARED,
            llvm_export::types::address_space::CONSTANT,
            llvm_export::types::address_space::LOCAL,
        ] {
            let mut ctx = make_ctx();
            let usize_ty = int_ty(&mut ctx, 64, Signedness::Unsigned);
            let pointee_ty = int_ty(&mut ctx, 32, Signedness::Signless);
            let ptr_ty: TypeHandle = MirPtrType::get_with_kind(
                &mut ctx,
                pointee_ty,
                false,
                address_space,
                MirPointerKind::RawConst,
            )
            .into();

            let module_ptr = lower_single_cast(
                &mut ctx,
                usize_ty,
                ptr_ty,
                MirCastKindAttr::PointerWithExposedProvenance,
            );

            // The exposed integer is a generic address, so re-entry must be
            // inttoptr-to-generic followed by addrspacecast into the space,
            // mirroring the PointerExposeAddress direction.
            assert_cast_lowered_to::<llvm::AddrSpaceCastOp>(&ctx, module_ptr, "llvm.addrspacecast");
            assert_eq!(
                count_ops::<llvm::IntToPtrOp>(&ctx, &kernel_blocks(&ctx, module_ptr)),
                1,
                "address space {address_space} must re-enter through a generic inttoptr"
            );
            let casts = find_all::<llvm::AddrSpaceCastOp>(&ctx, &kernel_blocks(&ctx, module_ptr));
            let [cast] = casts.as_slice() else {
                panic!("expected exactly one llvm.addrspacecast");
            };
            let result_ty = cast
                .get_operation()
                .deref(&ctx)
                .get_result(0)
                .get_type(&ctx);
            assert_eq!(
                pointer_addrspace(&ctx, result_ty),
                address_space,
                "the addrspacecast must land in the destination space"
            );
        }
    }

    #[test]
    fn ptr_to_ptr_different_addrspace_lowers_to_addrspace_cast() {
        let mut ctx = make_ctx();
        let pointee_ty = int_ty(&mut ctx, 32, Signedness::Signless);
        let generic_ptr_ty: TypeHandle = MirPtrType::get(&mut ctx, pointee_ty, false, 0).into();
        let shared_ptr_ty: TypeHandle = MirPtrType::get(&mut ctx, pointee_ty, false, 3).into();

        let module_ptr = lower_single_cast(
            &mut ctx,
            generic_ptr_ty,
            shared_ptr_ty,
            MirCastKindAttr::PtrToPtr,
        );

        assert_cast_lowered_to::<llvm::AddrSpaceCastOp>(&ctx, module_ptr, "llvm.addrspacecast");

        let casts = find_all::<llvm::AddrSpaceCastOp>(&ctx, &kernel_blocks(&ctx, module_ptr));
        let [cast] = casts.as_slice() else {
            panic!("ptr -> ptr addrspace cast must lower to exactly one llvm.addrspacecast");
        };
        let result_ty = cast
            .get_operation()
            .deref(&ctx)
            .get_result(0)
            .get_type(&ctx);
        assert_eq!(
            pointer_addrspace(&ctx, result_ty),
            3,
            "ptr -> ptr addrspace cast must produce an addrspace(3) pointer"
        );
    }

    #[test]
    fn slice_fat_pointer_cast_with_shared_data_rebuilds_in_ssa() {
        for shared_is_source in [true, false] {
            let mut ctx = make_ctx();
            let shared = llvm_export::types::address_space::SHARED;
            let generic = llvm_export::types::address_space::GENERIC;
            let (source_space, destination_space) = if shared_is_source {
                (shared, generic)
            } else {
                (generic, shared)
            };
            let source =
                slice_like_pointer_struct(&mut ctx, "SourceSliceFatPointer", 32, source_space);
            let destination = slice_like_pointer_struct(
                &mut ctx,
                "DestinationSliceFatPointer",
                16,
                destination_space,
            );

            let module =
                lower_single_cast(&mut ctx, source, destination, MirCastKindAttr::PtrToPtr);
            let body = kernel_blocks(&ctx, module);

            assert_eq!(
                count_ops::<llvm::AllocaOp>(&ctx, &body),
                0,
                "slice fat-pointer casts must stay in SSA"
            );
            assert_eq!(
                count_ops::<llvm::StoreOp>(&ctx, &body),
                0,
                "slice fat-pointer casts must not expose aggregate bytes"
            );
            assert_eq!(
                count_ops::<llvm::LoadOp>(&ctx, &body),
                0,
                "slice fat-pointer casts must not reload target-dependent pointer bytes"
            );
            assert!(
                count_ops::<llvm::ExtractValueOp>(&ctx, &body) >= 2,
                "slice fat-pointer casts must extract data and metadata"
            );
            assert!(
                count_ops::<llvm::InsertValueOp>(&ctx, &body) >= 2,
                "slice fat-pointer casts must rebuild data and metadata"
            );

            let expected_space = destination_space;
            let casts = find_all::<llvm::AddrSpaceCastOp>(&ctx, &body);
            assert!(
                casts.iter().any(|cast| {
                    let result = cast.get_operation().deref(&ctx).get_result(0);
                    pointer_addrspace(&ctx, result.get_type(&ctx)) == expected_space
                }),
                "slice data pointer must be addrspacecast into the destination space"
            );
        }
    }

    #[test]
    fn generic_slice_fat_pointer_cast_avoids_memory_round_trip() {
        let mut ctx = make_ctx();
        let generic = llvm_export::types::address_space::GENERIC;
        let source = slice_like_pointer_struct(&mut ctx, "SourceSliceFatPointer", 32, generic);
        let destination =
            slice_like_pointer_struct(&mut ctx, "DestinationSliceFatPointer", 16, generic);

        let module = lower_single_cast(&mut ctx, source, destination, MirCastKindAttr::PtrToPtr);
        let body = kernel_blocks(&ctx, module);

        assert_eq!(count_ops::<llvm::AllocaOp>(&ctx, &body), 0);
        assert_eq!(count_ops::<llvm::StoreOp>(&ctx, &body), 0);
        assert_eq!(count_ops::<llvm::LoadOp>(&ctx, &body), 0);
    }

    #[test]
    fn cluster_shared_slice_fat_pointer_cast_keeps_existing_fallback() {
        for cluster_shared_is_source in [true, false] {
            let mut ctx = make_ctx();
            let cluster_shared = dialect_mir::types::address_space::CLUSTER_SHARED;
            let generic = dialect_mir::types::address_space::GENERIC;
            let (source_space, destination_space) = if cluster_shared_is_source {
                (cluster_shared, generic)
            } else {
                (generic, cluster_shared)
            };
            let source =
                slice_like_pointer_struct(&mut ctx, "SourceSliceFatPointer", 32, source_space);
            let destination = slice_like_pointer_struct(
                &mut ctx,
                "DestinationSliceFatPointer",
                16,
                destination_space,
            );

            let module =
                lower_single_cast(&mut ctx, source, destination, MirCastKindAttr::PtrToPtr);
            let body = kernel_blocks(&ctx, module);

            assert!(
                count_ops::<llvm::AllocaOp>(&ctx, &body) > 0,
                "cluster-shared slice-shaped casts must keep the existing aggregate fallback"
            );
        }
    }

    /// `&mut [T; N]` in shared memory (addrspace 3) unsized to `&mut [T]`:
    /// the slice's field-0 pointer slot is generic (addrspace 0), so the data
    /// pointer must be `addrspacecast` before the `insert_value` (PTX
    /// `cvta.shared`). Without the coercion the lowered module fails
    /// verification with "Value being inserted / extracted does not match the
    /// type of the indexed aggregate".
    #[test]
    fn shared_array_unsize_coerces_data_pointer_to_slice_addrspace() {
        let mut ctx = make_ctx();
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let arr_ty: TypeHandle = dialect_mir::types::MirArrayType::get(&mut ctx, f32_ty, 16).into();
        let src_ty: TypeHandle = MirPtrType::get_shared(&mut ctx, arr_ty, true).into();
        let dst_ty: TypeHandle =
            dialect_mir::types::MirSliceType::get_with_mutability(&mut ctx, f32_ty, true).into();

        let module_ptr = lower_single_cast(
            &mut ctx,
            src_ty,
            dst_ty,
            MirCastKindAttr::PointerCoercionUnsize,
        );

        let body = kernel_blocks(&ctx, module_ptr);
        assert_eq!(
            count_ops::<llvm::AddrSpaceCastOp>(&ctx, &body),
            1,
            "shared -> generic data pointer must be addrspacecast before insertion"
        );
        assert_eq!(
            count_ops::<llvm::InsertValueOp>(&ctx, &body),
            2,
            "fat pointer construction must insert data pointer and length"
        );
        assert_eq!(
            count_ops::<mir::MirCastOp>(&ctx, &body),
            0,
            "mir.cast must be replaced during lowering"
        );
    }

    /// The same unsize from a generic (addrspace 0) array must NOT insert a
    /// spurious addrspacecast.
    #[test]
    fn generic_array_unsize_needs_no_addrspace_coercion() {
        let mut ctx = make_ctx();
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let arr_ty: TypeHandle = dialect_mir::types::MirArrayType::get(&mut ctx, f32_ty, 16).into();
        let src_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, arr_ty, true).into();
        let dst_ty: TypeHandle =
            dialect_mir::types::MirSliceType::get_with_mutability(&mut ctx, f32_ty, true).into();

        let module_ptr = lower_single_cast(
            &mut ctx,
            src_ty,
            dst_ty,
            MirCastKindAttr::PointerCoercionUnsize,
        );

        let body = kernel_blocks(&ctx, module_ptr);
        assert_eq!(
            count_ops::<llvm::AddrSpaceCastOp>(&ctx, &body),
            0,
            "matching address spaces must not introduce an addrspacecast"
        );
        assert_eq!(
            count_ops::<llvm::InsertValueOp>(&ctx, &body),
            2,
            "fat pointer construction must insert data pointer and length"
        );
    }
}
