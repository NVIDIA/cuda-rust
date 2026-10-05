/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thread and block indexing intrinsics.
//!
//! Handles translation of position-related intrinsics that query thread/block
//! identity and compute global indices.
//!
//! # Intrinsic Table
//!
//! | Intrinsic                  | NVVM Op                 | Description                                          |
//! |----------------------------|-------------------------|------------------------------------------------------|
//! | `threadIdx_x/y/z`          | `ReadPtxSregTidX/Y/Z`   | Thread ID within block                               |
//! | `blockIdx_x/y/z`           | `ReadPtxSregCtaidX/Y/Z` | Block ID within grid                                 |
//! | `blockDim_x/y/z`           | `ReadPtxSregNtidX/Y/Z`  | Block dimensions                                     |
//! | `index_1d()`               | Normal function call    | Global 1D thread index                               |
//! | `index_2d_row/col()`       | Normal function call    | 2D row/column indices                                |
//! | `index_2d::<S>()`          | Normal function call    | Const-stride 2D index (returns `Option<ThreadIndex>`)|
//! | `index_2d_runtime(&slice)` | Normal function call    | Runtime-stride 2D index (row width read from slice)  |
//! | `len()`                    | `MirExtractFieldOp`     | Slice length extraction                              |
//!
//! # Index Formulas
//!
//! - `index_1d() = blockIdx.x * blockDim.x + threadIdx.x`
//! - `index_2d_row() = blockIdx.y * blockDim.y + threadIdx.y`
//! - `index_2d_col() = blockIdx.x * blockDim.x + threadIdx.x`
//! - `index_2d::<S>() = if col < S { Some(row * S + col) } else { None }`
//! - `index_2d_runtime(&slice)`: normal function call into `cuda_device`;
//!   the witness packs `(row, col)` and the addressed slice resolves them
//!   against its own host-bound row width at the access site

use super::super::helpers;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::types;
use crate::translator::values::ValueMap;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::Typed;
use pliron::value::Value;
use rustc_public::mir;

/// Load the `DisjointSlice` value behind a method receiver.
///
/// `DisjointSlice::len` has an `&self` receiver, so fully monomorphized MIR
/// passes a `mir.ptr<mir.disjoint_slice<T>>`. Keep that source-level contract
/// explicit: accept exactly one pointer layer whose pointee is the compiler's
/// `MirDisjointSliceType`, and reject every other shape instead of guessing.
fn load_disjoint_slice_receiver(
    ctx: &mut Context,
    receiver: Value,
    block_ptr: Ptr<BasicBlock>,
    last_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Ptr<Operation>)> {
    let receiver_ty = receiver.get_type(ctx);
    let pointee = {
        let receiver_ty_obj = receiver_ty.deref(ctx);
        let pointee = receiver_ty_obj
            .downcast_ref::<dialect_mir::types::MirPtrType>()
            .map(|ptr_ty| ptr_ty.pointee);
        match pointee {
            Some(pointee)
                if pointee
                    .deref(ctx)
                    .downcast_ref::<dialect_mir::types::MirDisjointSliceType>()
                    .is_some() =>
            {
                pointee
            }
            _ => {
                return input_err!(
                    loc,
                    TranslationErr::type_error(
                        "DisjointSlice::len receiver must be a pointer to mir.disjoint_slice"
                            .to_string(),
                    )
                );
            }
        }
    };

    let load_op = Operation::new(
        ctx,
        dialect_mir::ops::MirLoadOp::get_concrete_op_info(),
        vec![pointee],
        vec![receiver],
        vec![],
        0,
    );
    load_op.deref_mut(ctx).set_loc(loc);
    match last_op {
        Some(prev) => load_op.insert_after(ctx, prev),
        None => load_op.insert_at_front(block_ptr, ctx),
    }

    let loaded_val = load_op.deref(ctx).get_result(0);
    Ok((loaded_val, load_op))
}

/// Emits `DisjointSlice::len()`: Extract the length field from a DisjointSlice.
///
/// # DisjointSlice Layout
///
/// ```text
/// struct DisjointSlice<T> {
///     ptr: *mut T,        // field 0
///     len: usize,         // field 1 ← extracted
///     _marker: PhantomData // field 2
/// }
/// ```
///
/// # Arguments
///
/// - `args[0]`: `&DisjointSlice<T>` - Reference to the slice
///
/// # Returns
///
/// `usize` - Number of elements in the slice
#[allow(clippy::too_many_arguments)]
pub fn emit_len(
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
) -> TranslationResult<Ptr<Operation>> {
    // Args should be: [&DisjointSlice]
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!("len expects 1 argument, got {}", args.len()))
        );
    }

    // Get the DisjointSlice value (arg 0)
    let (disjoint_slice_val, last_op) = match &args[0] {
        mir::Operand::Copy(place) | mir::Operand::Move(place) => {
            rvalue::translate_place(ctx, body, place, value_map, block_ptr, prev_op, loc.clone())?
        }
        _ => {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported("Constant DisjointSlice not supported".to_string(),)
            );
        }
    };

    let (disjoint_slice_val, load_op) =
        load_disjoint_slice_receiver(ctx, disjoint_slice_val, block_ptr, last_op, loc.clone())?;
    let last_op = Some(load_op);

    let (prepared_destination, last_op) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    // Extract len field (field 1) from DisjointSlice
    // DisjointSlice layout: { ptr: *mut T, len: usize, _marker: PhantomData }
    // We need the result type (usize). In MIR lowering we map usize to i64 usually.
    let usize_ty = types::get_usize_type(ctx);

    let extract_len_op = Operation::new(
        ctx,
        dialect_mir::ops::MirExtractFieldOp::get_concrete_op_info(),
        vec![usize_ty.into()],
        vec![disjoint_slice_val],
        vec![],
        0,
    );
    extract_len_op.deref_mut(ctx).set_loc(loc.clone());

    let extract_len = dialect_mir::ops::MirExtractFieldOp::new(extract_len_op);
    extract_len.set_attr_index(ctx, dialect_mir::attributes::FieldIndexAttr(1));

    if let Some(prev) = last_op {
        extract_len.get_operation().insert_after(ctx, prev);
    } else {
        extract_len.get_operation().insert_at_front(block_ptr, ctx);
    }
    let len_val = extract_len.get_operation().deref(ctx).get_result(0);
    let prev = extract_len.get_operation();
    helpers::emit_prepared_result_and_goto(
        ctx,
        prepared_destination,
        len_val,
        target,
        block_ptr,
        prev,
        value_map,
        block_map,
        loc,
        "len call without target block",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::ops::MirLoadOp;
    use dialect_mir::types::{MirDisjointSliceType, MirPtrType, MirSliceType};
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::common_traits::Verify;
    use pliron::linked_list::ContainsLinkedList;

    #[test]
    fn disjoint_slice_len_receiver_loads_exactly_one_typed_pointer_layer() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let element_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned).to_handle();
        let disjoint_ty: pliron::r#type::TypeHandle =
            MirDisjointSliceType::get(&mut ctx, element_ty).into();
        let receiver_ty = MirPtrType::get_generic(&mut ctx, disjoint_ty, false);
        let block = BasicBlock::new(&mut ctx, None, vec![receiver_ty.into()]);
        let receiver = block.deref(&ctx).get_argument(0);

        let (loaded, load_op) =
            load_disjoint_slice_receiver(&mut ctx, receiver, block, None, Location::Unknown)
                .expect("a pointer to MirDisjointSliceType is the len receiver shape");

        assert_eq!(loaded.get_type(&ctx), disjoint_ty);
        assert_eq!(block.deref(&ctx).iter(&ctx).count(), 1);
        let load = MirLoadOp::new(load_op);
        assert_eq!(load.address_opd(&ctx), receiver);
        assert!(load.verify(&ctx).is_ok());
    }

    #[test]
    fn disjoint_slice_len_receiver_rejects_near_miss_shapes() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let element_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned).to_handle();
        let disjoint_ty: pliron::r#type::TypeHandle =
            MirDisjointSliceType::get(&mut ctx, element_ty).into();
        let receiver_ty: pliron::r#type::TypeHandle =
            MirPtrType::get_generic(&mut ctx, disjoint_ty, false).into();

        // `len(&self)` always supplies one pointer layer. A direct fat value
        // or another pointer layer indicates a broken caller/translation and
        // must not be accepted by recursively guessing at the representation.
        for near_miss_ty in [disjoint_ty, {
            MirPtrType::get_generic(&mut ctx, receiver_ty, false).into()
        }] {
            let block = BasicBlock::new(&mut ctx, None, vec![near_miss_ty]);
            let receiver = block.deref(&ctx).get_argument(0);
            assert!(
                load_disjoint_slice_receiver(&mut ctx, receiver, block, None, Location::Unknown,)
                    .is_err()
            );
            assert_eq!(block.deref(&ctx).iter(&ctx).count(), 0);
        }

        // An ordinary Rust slice is also a `(ptr, len)` carrier, but it is not
        // a DisjointSlice receiver and must not pass a shape-only check.
        let ordinary_slice_ty: pliron::r#type::TypeHandle =
            MirSliceType::get(&mut ctx, element_ty).into();
        let ordinary_receiver_ty = MirPtrType::get_generic(&mut ctx, ordinary_slice_ty, false);
        let block = BasicBlock::new(&mut ctx, None, vec![ordinary_receiver_ty.into()]);
        let receiver = block.deref(&ctx).get_argument(0);
        assert!(
            load_disjoint_slice_receiver(&mut ctx, receiver, block, None, Location::Unknown,)
                .is_err()
        );
        assert_eq!(block.deref(&ctx).iter(&ctx).count(), 0);
    }

    /// The sreg reads behind `index_1d` are dispatched from `generated/sreg.rs`,
    /// which stamps each op with a marker literal. Assert the three markers
    /// against the target table the backend reads back at verification time,
    /// so a renumbering that reached only one of the two generated artifacts
    /// is caught.
    #[test]
    fn index_1d_sreg_ops_carry_their_exact_generated_markers() {
        for (dialect_op, expected) in [
            ("nvvm.read_ptx_sreg_tid_x", "v1:i0001"),
            ("nvvm.read_ptx_sreg_ctaid_x", "v1:i0002"),
            ("nvvm.read_ptx_sreg_ntid_x", "v1:i0003"),
        ] {
            assert_eq!(
                cuda_oxide_codegen::__private::generated_intrinsic_marker_by_op_name(dialect_op),
                Some(expected),
            );
        }
    }
}
