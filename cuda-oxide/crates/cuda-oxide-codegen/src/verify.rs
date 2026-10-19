/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::error::PipelineError;
use pliron::common_traits::Verify;
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::operation::Operation;
use pliron::printable::Printable;

/// Recursively verifies an operation and all nested operations.
///
/// On failure, attempts to find the innermost failing operation for better
/// error messages.
// Verification gate shared by this crate's own pipeline stages (prep.rs,
// pipeline.rs) and, through the `__private` re-export in lib.rs, by
// mir-importer's per-function post-translation check. That cross-crate hook
// is why this is `pub` + `#[doc(hidden)]`; it is not part of the
// standalone frontend contract.
#[doc(hidden)]
pub fn verify_operation(
    ctx: &Context,
    op_ptr: Ptr<Operation>,
    name: &str,
) -> Result<(), PipelineError> {
    if let Err(error) = dialect_mir::verification::verify_pointer_kind_producers(ctx, op_ptr) {
        return Err(PipelineError::Verification {
            name: name.to_string(),
            message: error.disp(ctx).to_string(),
            operation: None,
        });
    }

    if let Err(e) = op_ptr.deref(ctx).verify(ctx) {
        // Try to find specific failing operation
        if let Some((err_op, err_msg)) = find_inner_verification_error(ctx, op_ptr) {
            return Err(PipelineError::Verification {
                name: name.to_string(),
                message: err_msg,
                operation: Some(err_op.deref(ctx).disp(ctx).to_string()),
            });
        }

        // Use .disp(ctx) to get full error with location and backtrace
        return Err(PipelineError::Verification {
            name: name.to_string(),
            message: e.disp(ctx).to_string(),
            operation: None,
        });
    }
    Ok(())
}

/// Finds the innermost operation that failed verification.
///
/// Helps produce better error messages by pointing to the specific failing
/// operation rather than just the containing module/function.
///
/// Iterative, not recursive: recursion depth here would track REGION
/// nesting, and this walk assumes nothing about it at all. The gate runs
/// on more than one shape of tree (the `dialect-mir` module before
/// lowering and the LLVM dialect module after it, where `mir.func` and
/// `llvm.func` each carry a region), and there is no reason a compile
/// error, of all things, should depend on an unstated depth bound to
/// avoid becoming a stack overflow instead.
///
/// Two passes over an explicit stack, not one: `to_visit` drains in the
/// tree's natural (root-to-leaf, left-to-right) order, which is exactly
/// backwards from what finding an INNERMOST failure needs. Collecting into
/// `post_order` first and draining THAT reverses it back to
/// children-before-parent, matching the original recursive function's
/// contract (which checked every descendant of an operation before
/// checking the operation itself) exactly.
fn find_inner_verification_error(
    ctx: &Context,
    op_ptr: Ptr<Operation>,
) -> Option<(Ptr<Operation>, String)> {
    let mut to_visit = vec![op_ptr];
    let mut post_order = Vec::new();
    while let Some(op) = to_visit.pop() {
        post_order.push(op);
        for region in op.deref(ctx).regions() {
            let region_ref = region.deref(ctx);
            for block in region_ref.iter(ctx) {
                let block_ref = block.deref(ctx);
                to_visit.extend(block_ref.iter(ctx));
            }
        }
    }

    while let Some(op) = post_order.pop() {
        if let Err(e) = op.deref(ctx).verify(ctx) {
            // Use .disp(ctx) to get full error with location and backtrace
            return Some((op, e.disp(ctx).to_string()));
        }
    }

    None
}

#[cfg(test)]
// Tests build kinded fixture types directly; production minting lives in mir-importer's facts.rs.
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use dialect_mir::attributes::FieldIndexAttr;
    use dialect_mir::ops::{MirFieldAddrOp, MirFuncOp};
    use dialect_mir::types::{MirPtrType, MirStructType};
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::{IntegerAttr, TypeAttr};
    use pliron::builtin::op_interfaces::{OneRegionInterface, SymbolOpInterface};
    use pliron::builtin::ops::ConstantOp;
    use pliron::builtin::ops::ModuleOp;
    use pliron::builtin::types::{FunctionType, IntegerType, Signedness};
    use pliron::op::Op;
    use pliron::r#type::TypeHandle;
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    /// A `struct { a: u8, b: u32 }` type and its two field pointer types,
    /// precomputed so `struct_field_addr` can hand an in-bounds op exactly
    /// the result type `MirFieldAddrOp::verify` expects: a pointer whose
    /// pointee is the struct's recorded field type. Bundling them is
    /// convenience, not an identity requirement; pliron uniques types
    /// globally, so constructing the same `u32` twice yields the same
    /// `TypeHandle` anyway. A struct rather than a tuple deliberately:
    /// struct pointees are supported by `MirFieldAddrOp` on any revision
    /// of this codebase, so this test does not depend on #709's
    /// tuple-pointee support landing first.
    struct StructTypes {
        struct_ptr: TypeHandle,
        field_ptr: [TypeHandle; 2],
    }

    fn struct_types(ctx: &mut Context) -> StructTypes {
        let u8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let struct_ty: TypeHandle = MirStructType::get(
            ctx,
            "S".to_string(),
            vec!["a".to_string(), "b".to_string()],
            vec![u8_ty, u32_ty],
        )
        .into();
        StructTypes {
            struct_ptr: MirPtrType::get_generic(ctx, struct_ty, false).into(),
            field_ptr: [
                MirPtrType::get_generic(ctx, u8_ty, false).into(),
                MirPtrType::get_generic(ctx, u32_ty, false).into(),
            ],
        }
    }

    /// A `mir.field_addr` into a `struct { a: u8, b: u32 }` pointer (`block`'s
    /// only argument) at `field_index`. A leaf op (no regions, no
    /// successors), so its own verifier is the only thing that can make it
    /// pass or fail: 0 or 1 are in bounds and get their real field pointer
    /// type as the result type; anything else (2 here) is out of bounds and,
    /// since no real field type exists to give it, keeps the struct's own
    /// pointer type as a deliberately wrong result type -- MirFieldAddrOp::
    /// verify must reject it on the bounds check before it would ever reach
    /// a result pointee-type comparison.
    fn struct_field_addr(
        ctx: &mut Context,
        types: &StructTypes,
        block: Ptr<BasicBlock>,
        field_index: u32,
    ) -> Ptr<Operation> {
        let result_ty = types
            .field_ptr
            .get(field_index as usize)
            .copied()
            .unwrap_or(types.struct_ptr);
        let base = block.deref(ctx).get_argument(0);
        // The verifier requires the stamped aggregate fact to equal the
        // operand pointee, so the fixture stamps the real struct type; the
        // out-of-bounds case must still fail on the bounds check, not here.
        let aggregate_ty = {
            let ptr_ref = types.struct_ptr.deref(ctx);
            ptr_ref
                .downcast_ref::<MirPtrType>()
                .expect("fixture struct_ptr is a MirPtrType")
                .pointee
        };
        let op = Operation::new(
            ctx,
            MirFieldAddrOp::get_concrete_op_info(),
            vec![result_ty],
            vec![base],
            vec![],
            0,
        );
        MirFieldAddrOp::new(op).set_attr_field_index(ctx, FieldIndexAttr(field_index));
        MirFieldAddrOp::new(op).set_attr_aggregate_ty(ctx, TypeAttr::new(aggregate_ty));
        op
    }

    /// A module whose single block takes one struct-pointer argument,
    /// replacing `ModuleOp::new`'s own argument-less entry block.
    fn module_with_one_block(
        ctx: &mut Context,
        types: &StructTypes,
    ) -> (ModuleOp, Ptr<BasicBlock>) {
        let module = ModuleOp::new(ctx, "m".try_into().unwrap());
        let region = module.get_region(ctx);
        let old_block = region
            .deref(ctx)
            .iter(ctx)
            .next()
            .expect("ModuleOp::new creates one block");
        BasicBlock::erase(old_block, ctx);
        let block = BasicBlock::new(ctx, None, vec![types.struct_ptr]);
        block.insert_at_front(region, ctx);
        (module, block)
    }

    #[test]
    fn verify_operation_passes_through_a_fully_valid_tree() {
        let mut ctx = Context::new();
        let types = struct_types(&mut ctx);
        let (module, block) = module_with_one_block(&mut ctx, &types);
        struct_field_addr(&mut ctx, &types, block, 1).insert_at_back(block, &ctx);

        assert!(verify_operation(&ctx, module.get_operation(), "m").is_ok());
    }

    #[test]
    fn verify_operation_rejects_builtin_constant_pointer_results() {
        use dialect_mir::types::MirPointerKind;

        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
        let pointer_ty = MirPtrType::get_generic_with_kind(
            &mut ctx,
            u32_ty.into(),
            true,
            MirPointerKind::UniqueRef,
        );
        let value = APInt::from_u64(0, NonZeroUsize::new(32).unwrap());
        let constant = ConstantOp::new(&mut ctx, Box::new(IntegerAttr::new(u32_ty, value)));
        let result = constant.get_operation().deref(&ctx).get_result(0);
        result.set_type(&ctx, pointer_ty.into());

        let types = struct_types(&mut ctx);
        let (module, block) = module_with_one_block(&mut ctx, &types);
        constant.get_operation().insert_at_back(block, &ctx);

        let error = verify_operation(&ctx, module.get_operation(), "m").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("builtin.constant cannot produce")
        );
    }

    #[test]
    fn verify_operation_reports_a_direct_child_failure() {
        let mut ctx = Context::new();
        let types = struct_types(&mut ctx);
        let (module, block) = module_with_one_block(&mut ctx, &types);
        struct_field_addr(&mut ctx, &types, block, 2).insert_at_back(block, &ctx);

        let err = verify_operation(&ctx, module.get_operation(), "m").unwrap_err();
        let PipelineError::Verification { operation, .. } = err else {
            panic!("expected a Verification error, got {err:?}");
        };
        assert!(
            operation.unwrap().contains("mir.field_addr"),
            "must name the failing mir.field_addr operation, not just the module"
        );
    }

    #[test]
    fn verify_operation_finds_the_innermost_failure_among_several_siblings() {
        // Three siblings in one block: valid, INVALID, valid. The malformed
        // middle one is what verify_operation must name, not the first
        // sibling checked, not the module, and not silently the wrong one
        // because a later valid sibling masked it.
        let mut ctx = Context::new();
        let types = struct_types(&mut ctx);
        let (module, block) = module_with_one_block(&mut ctx, &types);

        struct_field_addr(&mut ctx, &types, block, 0).insert_at_back(block, &ctx);
        let bad_op = struct_field_addr(&mut ctx, &types, block, 2);
        bad_op.insert_at_back(block, &ctx);
        struct_field_addr(&mut ctx, &types, block, 1).insert_at_back(block, &ctx);

        let (found_op, _) = find_inner_verification_error(&ctx, module.get_operation())
            .expect("one of the three siblings is malformed");
        assert_eq!(
            found_op, bad_op,
            "must return the specific malformed sibling, not any other operation"
        );
    }

    #[test]
    fn verify_operation_reports_a_failure_several_region_levels_below_the_root() {
        // Module -> mir.func -> entry block -> malformed mir.field_addr:
        // the failure sits two region levels below the root passed to
        // verify_operation. `Operation::verify` recurses into regions, so
        // the module AND the function both fail their own verify calls;
        // the innermost contract requires naming the leaf, not the first
        // region-bearing ancestor that happens to fail.
        let mut ctx = Context::new();
        let types = struct_types(&mut ctx);
        let module = ModuleOp::new(&mut ctx, "m".try_into().unwrap());
        let module_block = module
            .get_region(&ctx)
            .deref(&ctx)
            .iter(&ctx)
            .next()
            .expect("ModuleOp::new creates one block");

        let func_ty = FunctionType::get(&ctx, vec![types.struct_ptr], vec![]);
        let func_op = Operation::new(
            &mut ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let func = MirFuncOp::new(&mut ctx, func_op, TypeAttr::new(func_ty.into()));
        func.set_symbol_name(&mut ctx, "f".try_into().unwrap());
        func_op.insert_at_back(module_block, &ctx);

        let func_region = func.get_region(&ctx);
        let entry = BasicBlock::new(&mut ctx, None, vec![types.struct_ptr]);
        entry.insert_at_front(func_region, &ctx);
        let bad_op = struct_field_addr(&mut ctx, &types, entry, 2);
        bad_op.insert_at_back(entry, &ctx);

        let (found_op, _) = find_inner_verification_error(&ctx, module.get_operation())
            .expect("the nested mir.field_addr is malformed");
        assert_eq!(
            found_op, bad_op,
            "must name the malformed leaf, not the enclosing function or module"
        );
    }

    #[test]
    fn verify_operation_reports_the_leftmost_of_two_malformed_siblings() {
        // Two malformed siblings in one block. The recursive contract
        // checked siblings in block order, so the FIRST one is what a
        // diagnostic must name; a traversal that flipped sibling order
        // would still "find an error", which is why this asserts on the
        // specific op rather than on finding any op at all.
        let mut ctx = Context::new();
        let types = struct_types(&mut ctx);
        let (module, block) = module_with_one_block(&mut ctx, &types);

        let first_bad = struct_field_addr(&mut ctx, &types, block, 2);
        first_bad.insert_at_back(block, &ctx);
        let second_bad = struct_field_addr(&mut ctx, &types, block, 3);
        second_bad.insert_at_back(block, &ctx);

        let (found_op, _) = find_inner_verification_error(&ctx, module.get_operation())
            .expect("both siblings are malformed");
        assert_eq!(
            found_op, first_bad,
            "must report the leftmost malformed sibling, not the later one"
        );
    }
}
