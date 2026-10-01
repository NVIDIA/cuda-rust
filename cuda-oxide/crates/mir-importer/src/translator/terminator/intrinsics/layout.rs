/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rust dynamic-layout intrinsics for slices, `str`, and slice-tailed structs.
//!
//! `core::mem::size_of_val` and `core::mem::align_of_val` inline to the
//! corresponding `core::intrinsics::*` calls when their argument is a DST.
//! CUDA Oxide already models `[T]` and `str` pointers as `MirSliceType`, whose
//! field 0 is the data pointer and field 1 is the runtime length metadata.
//!
//! For the DST shapes covered here, rustc's backend semantics are:
//!
//! - `[T]`: size = `len * size_of::<T>()`, align = `align_of::<T>()`
//! - `str`: size = byte length, align = 1
//! - `struct S { prefix: ..., tail: [T] }`: size and alignment are computed
//!   from rustc's field offset/layout plus the runtime tail length metadata
//!
//! Sized types are also accepted as a defensive fallback for low-MIR-opt
//! builds, where the intrinsic may survive even though optimized MIR normally
//! folds it to a constant. Other DST metadata shapes remain unsupported.

use super::super::helpers;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::values::ValueMap;
use crate::translator::{rvalue, types};
use dialect_mir::attributes::FieldIndexAttr;
use dialect_mir::ops::{MirAddOp, MirConstantOp, MirDivOp, MirExtractFieldOp, MirMulOp};
use dialect_mir::types::MirSliceType;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::IntegerAttr;
use pliron::context::{Context, Ptr};
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::Typed;
use pliron::utils::apint::APInt;
use pliron::value::Value;
use pliron::{input_err, input_error};
use rustc_public::CrateDefType;
use rustc_public::mir;
use rustc_public::ty::{RigidTy, Ty, TyKind};
use std::num::NonZeroUsize;

/// `core::intrinsics::{size_of_val, align_of_val}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RustLayoutIntrinsic {
    /// Return the runtime size of a possibly-unsized pointee.
    SizeOfVal,
    /// Return the ABI alignment of a possibly-unsized pointee.
    AlignOfVal,
}

impl RustLayoutIntrinsic {
    /// Recognize the libcore intrinsic path that survived into MIR.
    pub fn from_core_path(name: &str) -> Option<Self> {
        match name {
            "core::intrinsics::size_of_val" | "std::intrinsics::size_of_val" => {
                Some(Self::SizeOfVal)
            }
            "core::intrinsics::align_of_val" | "std::intrinsics::align_of_val" => {
                Some(Self::AlignOfVal)
            }
            _ => None,
        }
    }

    fn display_name(self) -> &'static str {
        match self {
            Self::SizeOfVal => "size_of_val",
            Self::AlignOfVal => "align_of_val",
        }
    }
}

fn rust_layout_shape(
    ty: &Ty,
    what: &str,
    loc: &Location,
) -> TranslationResult<rustc_public::abi::LayoutShape> {
    let layout = ty.layout().map_err(|error| {
        input_error!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "Failed to query rustc layout for {what}: {error:?}"
            ))
        )
    })?;
    Ok(layout.shape())
}

fn size_to_u64(bytes: usize, what: &str, loc: &Location) -> TranslationResult<u64> {
    u64::try_from(bytes).map_err(|_| {
        input_error!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{what} size does not fit the cuda-oxide u64 usize model: {bytes}"
            ))
        )
    })
}

/// Materialize a `usize` constant and insert it after `prev_op`.
fn emit_usize_constant(
    ctx: &mut Context,
    value: u64,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> (Value, Ptr<Operation>) {
    let usize_ty = types::get_usize_type(ctx);
    let width = NonZeroUsize::new(
        usize::try_from(usize_ty.deref(ctx).width()).expect("usize width must fit usize"),
    )
    .expect("usize integer width must be non-zero");
    let value_attr = IntegerAttr::new(usize_ty, APInt::from_u64(value, width));

    let op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![usize_ty.to_handle()],
        vec![],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc);
    MirConstantOp::new(op).set_attr_value(ctx, value_attr);
    match prev_op {
        Some(prev) => op.insert_after(ctx, prev),
        None => op.insert_at_front(block_ptr, ctx),
    }

    (op.deref(ctx).get_result(0), op)
}

/// Extract field 1, the runtime length, from a slice-shaped fat value.
fn emit_slice_len(
    ctx: &mut Context,
    slice_value: Value,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Ptr<Operation>)> {
    let value_ty = slice_value.get_type(ctx);
    if value_ty.deref(ctx).downcast_ref::<MirSliceType>().is_none() {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "DST layout intrinsic expected a slice-shaped fat-pointer operand".to_string()
            )
        );
    }

    let usize_ty = types::get_usize_type(ctx);
    let op = Operation::new(
        ctx,
        MirExtractFieldOp::get_concrete_op_info(),
        vec![usize_ty.to_handle()],
        vec![slice_value],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc);
    MirExtractFieldOp::new(op).set_attr_index(ctx, FieldIndexAttr(1));
    match prev_op {
        Some(prev) => op.insert_after(ctx, prev),
        None => op.insert_at_front(block_ptr, ctx),
    }

    Ok((op.deref(ctx).get_result(0), op))
}

/// Multiply a runtime `usize` by a compile-time `usize` constant.
fn emit_usize_mul_constant(
    ctx: &mut Context,
    lhs: Value,
    rhs: u64,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Ptr<Operation>,
    loc: Location,
) -> (Value, Ptr<Operation>) {
    if rhs == 1 {
        return (lhs, prev_op);
    }

    let (rhs_value, rhs_op) = emit_usize_constant(ctx, rhs, block_ptr, Some(prev_op), loc.clone());
    let usize_ty = types::get_usize_type(ctx);
    let mul = Operation::new(
        ctx,
        MirMulOp::get_concrete_op_info(),
        vec![usize_ty.to_handle()],
        vec![lhs, rhs_value],
        vec![],
        0,
    );
    mul.deref_mut(ctx).set_loc(loc);
    mul.insert_after(ctx, rhs_op);
    (mul.deref(ctx).get_result(0), mul)
}

/// Add a compile-time `usize` constant to a runtime `usize`.
fn emit_usize_add_constant(
    ctx: &mut Context,
    lhs: Value,
    rhs: u64,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Ptr<Operation>,
    loc: Location,
) -> (Value, Ptr<Operation>) {
    if rhs == 0 {
        return (lhs, prev_op);
    }

    let (rhs_value, rhs_op) = emit_usize_constant(ctx, rhs, block_ptr, Some(prev_op), loc.clone());
    let usize_ty = types::get_usize_type(ctx);
    let add = Operation::new(
        ctx,
        MirAddOp::get_concrete_op_info(),
        vec![usize_ty.to_handle()],
        vec![lhs, rhs_value],
        vec![],
        0,
    );
    add.deref_mut(ctx).set_loc(loc);
    add.insert_after(ctx, rhs_op);
    (add.deref(ctx).get_result(0), add)
}

/// Round a runtime `usize` up to a compile-time power-of-two alignment.
fn emit_usize_align_up(
    ctx: &mut Context,
    value: Value,
    alignment: u64,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Ptr<Operation>,
    loc: Location,
) -> (Value, Ptr<Operation>) {
    if alignment <= 1 {
        return (value, prev_op);
    }

    debug_assert!(alignment.is_power_of_two());

    let (biased, add_op) =
        emit_usize_add_constant(ctx, value, alignment - 1, block_ptr, prev_op, loc.clone());
    let (align_value, align_op) =
        emit_usize_constant(ctx, alignment, block_ptr, Some(add_op), loc.clone());

    let usize_ty = types::get_usize_type(ctx);
    let div = Operation::new(
        ctx,
        MirDivOp::get_concrete_op_info(),
        vec![usize_ty.to_handle()],
        vec![biased, align_value],
        vec![],
        0,
    );
    div.deref_mut(ctx).set_loc(loc.clone());
    div.insert_after(ctx, align_op);
    let quotient = div.deref(ctx).get_result(0);

    let mul = Operation::new(
        ctx,
        MirMulOp::get_concrete_op_info(),
        vec![usize_ty.to_handle()],
        vec![quotient, align_value],
        vec![],
        0,
    );
    mul.deref_mut(ctx).set_loc(loc);
    mul.insert_after(ctx, div);
    (mul.deref(ctx).get_result(0), mul)
}

/// Return the trailing field of a slice-tailed struct.
fn trailing_struct_field(ty: &Ty, loc: &Location) -> TranslationResult<(Ty, usize, Option<u64>)> {
    let TyKind::RigidTy(RigidTy::Adt(adt_def, substs)) = ty.kind() else {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "expected a struct DST while computing dynamic layout; got {:?}",
                ty.kind()
            ))
        );
    };

    if !matches!(adt_def.kind(), rustc_public::ty::AdtKind::Struct) {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "dynamic slice-tail layout requires a struct; got {:?}",
                adt_def.kind()
            ))
        );
    }

    let variants = adt_def.variants();
    let variant = variants.first().ok_or_else(|| {
        input_error!(
            loc.clone(),
            TranslationErr::unsupported("slice-tailed struct has no field variant".to_string())
        )
    })?;
    let fields = variant.fields();
    let last_index = fields.len().checked_sub(1).ok_or_else(|| {
        input_error!(
            loc.clone(),
            TranslationErr::unsupported("slice-tailed struct has no fields".to_string())
        )
    })?;
    let last_ty = fields[last_index].ty_with_args(&substs);

    Ok((last_ty, last_index, adt_def.repr().pack))
}

/// Compute the ABI alignment of a slice-tailed struct DST.
///
/// The tail metadata is a length, so alignment is fully determined by the
/// monomorphized prefix, the trailing element type, and any `repr(packed)` cap.
fn slice_tail_alignment(value_ty: &Ty, loc: &Location) -> TranslationResult<u64> {
    match value_ty.kind() {
        TyKind::RigidTy(RigidTy::Slice(element_ty)) => {
            Ok(rust_layout_shape(&element_ty, "slice-tail element", loc)?.abi_align)
        }
        TyKind::RigidTy(RigidTy::Str) => Ok(1),
        TyKind::RigidTy(RigidTy::Adt(..)) => {
            let (last_ty, _last_index, packed) = trailing_struct_field(value_ty, loc)?;
            let shape = rust_layout_shape(value_ty, "slice-tailed struct", loc)?;
            let mut tail_align = slice_tail_alignment(&last_ty, loc)?;

            if let Some(packed) = packed {
                tail_align = tail_align.min(packed);
            }

            Ok(shape.abi_align.max(tail_align))
        }
        _ => input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "unsupported unsized tail while computing alignment: {:?}",
                value_ty.kind()
            ))
        ),
    }
}

/// Compute runtime size and static ABI alignment for a slice-tailed DST.
///
/// This mirrors rustc's DST layout rule for aggregate tails:
///
/// `size = align_up(last_field_offset + dynamic_tail_size, full_alignment)`.
///
/// The same runtime metadata value is threaded recursively for nested
/// slice-tailed structs because their metadata is the ultimate slice length.
#[allow(clippy::too_many_arguments)]
fn emit_slice_tail_size_and_align(
    ctx: &mut Context,
    value_ty: &Ty,
    metadata_len: Value,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Ptr<Operation>,
    loc: Location,
) -> TranslationResult<(Value, u64, Ptr<Operation>)> {
    match value_ty.kind() {
        TyKind::RigidTy(RigidTy::Slice(element_ty)) => {
            let element_layout = rust_layout_shape(&element_ty, "slice-tail element", &loc)?;
            let element_size =
                size_to_u64(element_layout.size.bytes(), "slice-tail element", &loc)?;
            let (size, size_op) =
                emit_usize_mul_constant(ctx, metadata_len, element_size, block_ptr, prev_op, loc);
            Ok((size, element_layout.abi_align, size_op))
        }
        TyKind::RigidTy(RigidTy::Str) => Ok((metadata_len, 1, prev_op)),
        TyKind::RigidTy(RigidTy::Adt(..)) => {
            let (last_ty, last_index, packed) = trailing_struct_field(value_ty, &loc)?;
            let shape = rust_layout_shape(value_ty, "slice-tailed struct", &loc)?;
            let tail_offset = match &shape.fields {
                rustc_public::abi::FieldsShape::Arbitrary { offsets } => offsets
                    .get(last_index)
                    .map(|offset| offset.bytes())
                    .ok_or_else(|| {
                        input_error!(
                            loc.clone(),
                            TranslationErr::unsupported(format!(
                                "slice-tailed struct layout has {} offsets for field {}",
                                offsets.len(),
                                last_index
                            ))
                        )
                    })?,
                other => {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "slice-tailed struct has unsupported field layout: {other:?}"
                        ))
                    );
                }
            };
            let tail_offset = size_to_u64(tail_offset, "slice-tailed struct tail offset", &loc)?;

            let (tail_size, mut tail_align, tail_op) = emit_slice_tail_size_and_align(
                ctx,
                &last_ty,
                metadata_len,
                block_ptr,
                prev_op,
                loc.clone(),
            )?;

            if let Some(packed) = packed {
                tail_align = tail_align.min(packed);
            }

            let full_align = shape.abi_align.max(tail_align);
            let (unrounded_size, add_op) = emit_usize_add_constant(
                ctx,
                tail_size,
                tail_offset,
                block_ptr,
                tail_op,
                loc.clone(),
            );
            let (full_size, align_op) =
                emit_usize_align_up(ctx, unrounded_size, full_align, block_ptr, add_op, loc);

            Ok((full_size, full_align, align_op))
        }
        _ => input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "unsupported unsized tail while computing size: {:?}",
                value_ty.kind()
            ))
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_size_of_val(
    ctx: &mut Context,
    body: &mir::Body,
    arg: &mir::Operand,
    value_ty: &Ty,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    loc: Location,
) -> TranslationResult<(Value, Ptr<Operation>)> {
    match value_ty.kind() {
        TyKind::RigidTy(RigidTy::Slice(element_ty)) => {
            let (fat_value, after_operand) = rvalue::translate_operand(
                ctx,
                body,
                arg,
                value_map,
                block_ptr,
                prev_op,
                loc.clone(),
            )?;
            let (len, after_len) =
                emit_slice_len(ctx, fat_value, block_ptr, after_operand, loc.clone())?;

            let element_layout = rust_layout_shape(&element_ty, "slice element", &loc)?;
            let element_size = size_to_u64(element_layout.size.bytes(), "slice element", &loc)?;
            Ok(emit_usize_mul_constant(
                ctx,
                len,
                element_size,
                block_ptr,
                after_len,
                loc,
            ))
        }
        TyKind::RigidTy(RigidTy::Str) => {
            let (fat_value, after_operand) = rvalue::translate_operand(
                ctx,
                body,
                arg,
                value_map,
                block_ptr,
                prev_op,
                loc.clone(),
            )?;
            emit_slice_len(ctx, fat_value, block_ptr, after_operand, loc)
        }
        TyKind::RigidTy(RigidTy::Adt(..)) if types::slice_tail_element_ty(value_ty).is_some() => {
            let (fat_value, after_operand) = rvalue::translate_operand(
                ctx,
                body,
                arg,
                value_map,
                block_ptr,
                prev_op,
                loc.clone(),
            )?;
            let (len, after_len) =
                emit_slice_len(ctx, fat_value, block_ptr, after_operand, loc.clone())?;
            let (size, _align, after_layout) =
                emit_slice_tail_size_and_align(ctx, value_ty, len, block_ptr, after_len, loc)?;
            Ok((size, after_layout))
        }
        _ => {
            let shape = rust_layout_shape(value_ty, "size_of_val pointee", &loc)?;
            if !shape.is_sized() {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "size_of_val currently supports slices, str, slice-tailed structs, and Sized pointees; got {:?}",
                        value_ty.kind()
                    ))
                );
            }
            let size = size_to_u64(shape.size.bytes(), "size_of_val pointee", &loc)?;
            Ok(emit_usize_constant(ctx, size, block_ptr, prev_op, loc))
        }
    }
}

fn emit_align_of_val(
    ctx: &mut Context,
    value_ty: &Ty,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Ptr<Operation>)> {
    let alignment = match value_ty.kind() {
        TyKind::RigidTy(RigidTy::Slice(element_ty)) => {
            rust_layout_shape(&element_ty, "slice element", &loc)?.abi_align
        }
        TyKind::RigidTy(RigidTy::Str) => 1,
        TyKind::RigidTy(RigidTy::Adt(..)) if types::slice_tail_element_ty(value_ty).is_some() => {
            slice_tail_alignment(value_ty, &loc)?
        }
        _ => {
            let shape = rust_layout_shape(value_ty, "align_of_val pointee", &loc)?;
            if !shape.is_sized() {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "align_of_val currently supports slices, str, slice-tailed structs, and Sized pointees; got {:?}",
                        value_ty.kind()
                    ))
                );
            }
            shape.abi_align
        }
    };

    Ok(emit_usize_constant(ctx, alignment, block_ptr, prev_op, loc))
}

/// Lower `core::intrinsics::{size_of_val, align_of_val}` for supported DSTs.
#[allow(clippy::too_many_arguments)]
pub fn emit_rust_layout_intrinsic(
    ctx: &mut Context,
    body: &mir::Body,
    intrinsic: RustLayoutIntrinsic,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    type_substs: &[Ty],
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "{} expects exactly one pointer argument, got {}",
                intrinsic.display_name(),
                args.len()
            ))
        );
    }

    let Some(value_ty) = type_substs.first() else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "{} is missing its pointee type substitution",
                intrinsic.display_name()
            ))
        );
    };

    let target_idx = match target {
        Some(index) => *index,
        None => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "{} call without target block",
                    intrinsic.display_name()
                ))
            );
        }
    };

    let (prepared_destination, prev_op) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let (result, result_op) = match intrinsic {
        RustLayoutIntrinsic::SizeOfVal => emit_size_of_val(
            ctx,
            body,
            &args[0],
            value_ty,
            block_ptr,
            prev_op,
            value_map,
            loc.clone(),
        )?,
        RustLayoutIntrinsic::AlignOfVal => {
            emit_align_of_val(ctx, value_ty, block_ptr, prev_op, loc.clone())?
        }
    };

    let store_op = helpers::finish_destination_write(
        ctx,
        prepared_destination,
        result,
        value_map,
        block_ptr,
        result_op,
        loc.clone(),
    )?;

    Ok(helpers::emit_goto(
        ctx, target_idx, store_op, block_map, loc,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_size_and_align_of_val_paths() {
        for path in [
            "core::intrinsics::size_of_val",
            "std::intrinsics::size_of_val",
        ] {
            assert_eq!(
                RustLayoutIntrinsic::from_core_path(path),
                Some(RustLayoutIntrinsic::SizeOfVal)
            );
        }

        for path in [
            "core::intrinsics::align_of_val",
            "std::intrinsics::align_of_val",
        ] {
            assert_eq!(
                RustLayoutIntrinsic::from_core_path(path),
                Some(RustLayoutIntrinsic::AlignOfVal)
            );
        }
    }

    #[test]
    fn rejects_neighboring_layout_operations() {
        for path in [
            "core::mem::size_of_val",
            "core::mem::align_of_val",
            "core::intrinsics::size_of",
            "core::intrinsics::align_of",
            "core::intrinsics::size_of_val_raw",
        ] {
            assert_eq!(RustLayoutIntrinsic::from_core_path(path), None);
        }
    }
}
