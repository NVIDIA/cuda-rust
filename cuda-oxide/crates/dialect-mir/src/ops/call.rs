/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR call operations.
//!
//! This module defines function call operations for the MIR dialect.

use crate::{
    attributes::MirPointerKindAuthorityAttr,
    rust_intrinsics,
    types::{MirTupleType, pointer_kinds_in_type},
};

use pliron::{
    attribute::attr_cast,
    builtin::{
        attr_interfaces::TypedAttrInterface,
        attributes::{StringAttr, TypeAttr},
        op_interfaces::SymbolOpInterface,
        type_interfaces::FunctionTypeInterface,
        types::{FunctionType, IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::{TypeHandle, Typed, type_cast},
    verify_err,
};
use pliron_derive::pliron_op;

// ============================================================================
// MirCallOp
// ============================================================================

/// MIR call operation.
///
/// Represents a function call.
///
/// # Attributes
///
/// ```text
/// | Name                     | Type                          | Description |
/// |--------------------------|-------------------------------|-------------|
/// | `callee`                 | StringAttr                    | Name of the called function. |
/// | `external_callee_type`   | TypeAttr                      | Exact Rust-level signature of a foreign item with no `mir.func`. |
/// | `call_pointer_kind_authority` | MirPointerKindAuthorityAttr | Must be `AbiBoundary` when `external_callee_type` is present. |
/// ```
///
/// # Operands
///
/// Variadic operands matching the callee's argument types.
///
/// # Results
///
/// Variadic results matching the callee's return types.
///
/// # Verification
///
/// - Must have `callee` attribute.
/// - Once the call and callee have both been attached to the same module,
///   operand and result types must match the callee's `mir.func` signature.
/// - An attached foreign call with no `mir.func` must carry both an exact
///   `external_callee_type` and `AbiBoundary`; the call must match that type.
/// - Internal Rust-intrinsic placeholders use exact, name-specific rules and
///   may not claim external ABI authority.
#[pliron_op(
    name = "mir.call",
    format,
    attributes = (
        callee: StringAttr,
        external_callee_type: TypeAttr,
        call_pointer_kind_authority: MirPointerKindAuthorityAttr
    )
)]
pub struct MirCallOp;

impl MirCallOp {
    /// Create a new MirCallOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirCallOp { op }
    }

    /// Mark a call to a resolved Rust foreign item and retain its exact
    /// source-level signature independently from the call operands/results.
    pub fn set_external_callee_signature(&self, ctx: &mut Context, signature: TypeHandle) {
        self.set_attr_external_callee_type(ctx, TypeAttr::new(signature));
        self.set_attr_call_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::AbiBoundary);
    }
}

impl Verify for MirCallOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        let Some(callee_attr) = self.get_attr_callee(ctx) else {
            return verify_err!(op.loc(), "MirCallOp must have a callee attribute");
        };
        let callee_name = String::from((*callee_attr).clone());
        let external_signature = external_callee_signature(ctx, self)?;

        // Internal placeholders are not external symbols. They have a closed
        // schema here because no `mir.func` declaration independently types
        // them. In particular, select_unpredictable is generic and can carry
        // pointers, so preserving full type equality is essential.
        if rust_intrinsics::is_known_placeholder(&callee_name) {
            if external_signature.is_some() {
                return verify_err!(
                    op.loc(),
                    "internal Rust-intrinsic placeholder must not claim external ABI authority"
                );
            }
            return verify_known_placeholder(ctx, self, &callee_name);
        }

        // The prefix is reserved for the importer/lowerer contract. An exact
        // allow-list prevents a typo or future pseudo-call from falling
        // through to the external-call escape hatch.
        if callee_name.starts_with(rust_intrinsics::PLACEHOLDER_PREFIX) {
            return verify_err!(
                op.loc(),
                "unknown internal Rust-intrinsic placeholder `{}`",
                callee_name
            );
        }

        match resolve_call(ctx, self.get_operation(), &callee_name) {
            CallResolution::AttachedResolved(callee) => {
                if external_signature.is_some() {
                    return verify_err!(
                        op.loc(),
                        "call to an in-module mir.func must not claim external ABI authority"
                    );
                }
                verify_call_signature(ctx, self, callee.get_type(ctx).into())
            }
            CallResolution::AttachedUnresolved => {
                let Some(signature) = external_signature else {
                    return verify_err!(
                        op.loc(),
                        "attached MirCallOp `{}` has no in-module mir.func or independently typed foreign ABI signature",
                        callee_name
                    );
                };
                verify_call_signature(ctx, self, signature)
            }
            CallResolution::Detached => {
                // Importer functions are verified before they are inserted in
                // the module, so ordinary symbol lookup genuinely is not
                // possible yet. Still validate an explicit foreign claim now;
                // otherwise the final whole-module verifier decides whether
                // the call resolves or must carry such a claim.
                if let Some(signature) = external_signature {
                    verify_call_signature(ctx, self, signature)
                } else {
                    Ok(())
                }
            }
        }
    }
}

/// A call's placement and symbol-table state, kept distinct so the importer
/// construction phase cannot be confused with an unresolved final module.
enum CallResolution {
    Detached,
    AttachedResolved(super::MirFuncOp),
    AttachedUnresolved,
}

fn resolve_call(ctx: &Context, call: Ptr<Operation>, callee_name: &str) -> CallResolution {
    let mut containing_op = call;
    loop {
        let Some(parent_block) = containing_op.deref(ctx).get_parent_block() else {
            return CallResolution::Detached;
        };
        let Some(parent_op) = parent_block.deref(ctx).get_parent_op(ctx) else {
            return CallResolution::Detached;
        };
        if let Some(function) = super::MirFuncOp::wrap(ctx, parent_op) {
            let Some(symbol_block) = function.get_operation().deref(ctx).get_parent_block() else {
                // Importer functions are verified before insertion into their
                // module, so this is the one legitimate attached-to-a-body but
                // not-yet-attached-to-a-symbol-table construction state.
                return CallResolution::Detached;
            };
            return resolve_in_symbol_block(ctx, symbol_block, callee_name);
        }
        if Operation::get_op::<pliron::builtin::ops::ModuleOp>(parent_op, ctx).is_some() {
            // A call can be malformedly placed directly in a module, or can
            // live in an already-lowered/non-MIR function in a mixed module.
            // Both are final attached state, not importer construction state.
            return resolve_in_symbol_block(ctx, parent_block, callee_name);
        }
        containing_op = parent_op;
    }
}

fn resolve_in_symbol_block(
    ctx: &Context,
    symbol_block: Ptr<pliron::basic_block::BasicBlock>,
    callee_name: &str,
) -> CallResolution {
    symbol_block
        .deref(ctx)
        .iter(ctx)
        .find_map(|candidate| {
            let function = super::MirFuncOp::wrap(ctx, candidate)?;
            (function.get_symbol_name(ctx).to_string() == callee_name).then_some(function)
        })
        .map_or(
            CallResolution::AttachedUnresolved,
            CallResolution::AttachedResolved,
        )
}

/// Read and validate the two-part foreign-call capability.
///
/// `AbiBoundary` alone would merely bless the type invented by the call. The
/// `TypeAttr` is translated independently from rustc's resolved foreign-item
/// declaration and is therefore the authoritative source signature.
fn external_callee_signature(ctx: &Context, call: &MirCallOp) -> Result<Option<TypeHandle>, Error> {
    let op = call.get_operation().deref(ctx);
    let type_attr = call.get_attr_external_callee_type(ctx);
    let authority = call.get_attr_call_pointer_kind_authority(ctx);

    match (type_attr, authority) {
        (None, None) => Ok(None),
        (Some(_), None) => verify_err!(
            op.loc(),
            "external_callee_type requires AbiBoundary pointer-kind authority"
        ),
        (None, Some(_)) => verify_err!(
            op.loc(),
            "external-call AbiBoundary authority requires external_callee_type"
        ),
        (Some(type_attr), Some(authority)) => {
            if *authority != MirPointerKindAuthorityAttr::AbiBoundary {
                return verify_err!(
                    op.loc(),
                    "external_callee_type requires AbiBoundary pointer-kind authority"
                );
            }
            let signature = attr_cast::<dyn TypedAttrInterface>(&*type_attr)
                .expect("TypeAttr must implement TypedAttrInterface")
                .get_type(ctx);
            if signature
                .deref(ctx)
                .downcast_ref::<FunctionType>()
                .is_none()
            {
                return verify_err!(
                    op.loc(),
                    "external_callee_type must contain a builtin source-level function type"
                );
            }
            Ok(Some(signature))
        }
    }
}

fn verify_call_signature(
    ctx: &Context,
    call: &MirCallOp,
    signature: TypeHandle,
) -> Result<(), Error> {
    let op = call.get_operation().deref(ctx);
    let signature_ref = signature.deref(ctx);
    let Some(signature) = type_cast::<dyn FunctionTypeInterface>(&*signature_ref) else {
        return verify_err!(
            op.loc(),
            "callee type does not implement FunctionTypeInterface"
        );
    };
    let expected_arguments = signature.arg_types();
    let expected_results = signature.res_types();

    if op.get_num_operands() != expected_arguments.len() {
        return verify_err!(
            op.loc(),
            "MirCallOp argument count does not match callee signature"
        );
    }
    for (index, expected) in expected_arguments.iter().enumerate() {
        if op.get_operand(index).get_type(ctx) != *expected {
            return verify_err!(
                op.loc(),
                "MirCallOp argument {} type does not match callee signature",
                index
            );
        }
    }

    // The importer models a call returning Rust `()` as one empty-tuple SSA
    // result so it can flow through the caller's destination place, while a
    // function signature omits unit from its result list.
    let call_returns_logical_unit = op.get_num_results() == 1
        && op
            .get_result(0)
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<MirTupleType>()
            .is_some_and(|tuple| tuple.get_types().is_empty());
    if expected_results.is_empty() && call_returns_logical_unit {
        return Ok(());
    }

    if op.get_num_results() != expected_results.len() {
        return verify_err!(
            op.loc(),
            "MirCallOp result count does not match callee signature"
        );
    }
    for (index, expected) in expected_results.iter().enumerate() {
        if op.get_result(index).get_type(ctx) != *expected {
            return verify_err!(
                op.loc(),
                "MirCallOp result {} type does not match callee signature",
                index
            );
        }
    }

    Ok(())
}

fn verify_known_placeholder(
    ctx: &Context,
    call: &MirCallOp,
    callee_name: &str,
) -> Result<(), Error> {
    let op = call.get_operation().deref(ctx);

    if callee_name == rust_intrinsics::CALLEE_SELECT_UNPREDICTABLE {
        if op.get_num_operands() != 3 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "select_unpredictable placeholder requires three operands and one result"
            );
        }
        let condition_type = op.get_operand(0).get_type(ctx);
        let condition_type = condition_type.deref(ctx);
        let valid_condition = condition_type
            .downcast_ref::<IntegerType>()
            .is_some_and(|integer| {
                integer.width() == 1 && integer.signedness() == Signedness::Signless
            });
        if !valid_condition {
            return verify_err!(
                op.loc(),
                "select_unpredictable placeholder condition must be signless i1"
            );
        }

        let selected_type = op.get_operand(1).get_type(ctx);
        if op.get_operand(2).get_type(ctx) != selected_type
            || op.get_result(0).get_type(ctx) != selected_type
        {
            return verify_err!(
                op.loc(),
                "select_unpredictable placeholder value operands and result must have exactly the same type"
            );
        }
        return Ok(());
    }

    // Every other currently supported placeholder is a numeric intrinsic.
    // Detailed width/arity constraints remain in the lowering that owns each
    // operation, but none may carry a pointer-kind claim (including one nested
    // inside a MIR aggregate).
    for index in 0..op.get_num_operands() {
        if !pointer_kinds_in_type(ctx, op.get_operand(index).get_type(ctx)).is_empty() {
            return verify_err!(
                op.loc(),
                "numeric Rust-intrinsic placeholder operand {} must not carry a pointer type",
                index
            );
        }
    }
    for index in 0..op.get_num_results() {
        if !pointer_kinds_in_type(ctx, op.get_result(index).get_type(ctx)).is_empty() {
            return verify_err!(
                op.loc(),
                "numeric Rust-intrinsic placeholder result {} must not carry a pointer type",
                index
            );
        }
    }
    Ok(())
}

/// Register call operations into the given context.
pub fn register(ctx: &mut Context) {
    MirCallOp::register(ctx);
}

#[cfg(test)]
// Tests build kinded fixture types directly; production minting lives in mir-importer's facts.rs.
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::{
        ops::{MirFuncOp, MirReturnOp},
        types::{MirPointerKind, MirPtrType, MirTupleType},
    };
    use pliron::{
        basic_block::BasicBlock,
        builtin::{
            attributes::TypeAttr,
            ops::ModuleOp,
            types::{FunctionType, IntegerType, Signedness},
        },
        printable::Printable,
        r#type::TypeHandle,
    };

    fn module_top_block(ctx: &mut Context, module: &ModuleOp) -> Ptr<BasicBlock> {
        let region = module.get_operation().deref(ctx).get_region(0);
        if let Some(block) = region.deref(ctx).iter(ctx).next() {
            return block;
        }
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(region, ctx);
        block
    }

    fn append_function(
        ctx: &mut Context,
        module_block: Ptr<BasicBlock>,
        name: &str,
        arguments: Vec<TypeHandle>,
        results: Vec<TypeHandle>,
    ) -> (MirFuncOp, Ptr<BasicBlock>) {
        let function_type = FunctionType::get(ctx, arguments.clone(), results);
        let operation = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function = MirFuncOp::new(ctx, operation, TypeAttr::new(function_type.into()));
        function.set_symbol_name(ctx, name.try_into().unwrap());
        let entry = BasicBlock::new(ctx, None, arguments);
        entry.insert_at_back(function.get_operation().deref(ctx).get_region(0), ctx);
        function.get_operation().insert_at_back(module_block, ctx);
        (function, entry)
    }

    fn append_call(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        callee: &str,
        arguments: Vec<pliron::value::Value>,
        results: Vec<TypeHandle>,
    ) -> MirCallOp {
        let call = new_call(ctx, callee, arguments, results);
        call.get_operation().insert_at_back(block, ctx);
        call
    }

    fn new_call(
        ctx: &mut Context,
        callee: &str,
        arguments: Vec<pliron::value::Value>,
        results: Vec<TypeHandle>,
    ) -> MirCallOp {
        let operation = Operation::new(
            ctx,
            MirCallOp::get_concrete_op_info(),
            results,
            arguments,
            vec![],
            0,
        );
        let call = MirCallOp::new(operation);
        call.set_attr_callee(ctx, StringAttr::new(callee.to_string()));
        call
    }

    fn append_return(ctx: &mut Context, block: Ptr<BasicBlock>) {
        Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        )
        .insert_at_back(block, ctx);
    }

    #[test]
    fn in_module_calls_reject_pointer_kind_and_mutability_mismatches() {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let module = ModuleOp::new(&mut ctx, "call_pointer_kinds".try_into().unwrap());
        let module_block = module_top_block(&mut ctx, &module);
        let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();

        for (index, actual_kind, actual_mutable, expected_kind, expected_mutable) in [
            (
                0,
                MirPointerKind::Erased,
                false,
                MirPointerKind::SharedRef,
                false,
            ),
            (
                1,
                MirPointerKind::Erased,
                true,
                MirPointerKind::UniqueRef,
                true,
            ),
            (
                2,
                MirPointerKind::Erased,
                true,
                MirPointerKind::RawMut,
                true,
            ),
            (
                3,
                MirPointerKind::RawMut,
                true,
                MirPointerKind::UniqueRef,
                true,
            ),
            (
                4,
                MirPointerKind::Erased,
                false,
                MirPointerKind::Erased,
                true,
            ),
        ] {
            let actual: TypeHandle =
                MirPtrType::get_generic_with_kind(&mut ctx, pointee, actual_mutable, actual_kind)
                    .into();
            let expected: TypeHandle = MirPtrType::get_generic_with_kind(
                &mut ctx,
                pointee,
                expected_mutable,
                expected_kind,
            )
            .into();
            let callee_name = format!("pointer_callee_{index}");
            let (_, callee_entry) =
                append_function(&mut ctx, module_block, &callee_name, vec![expected], vec![]);
            append_return(&mut ctx, callee_entry);
            let caller_name = format!("pointer_caller_{index}");
            let (_, caller_entry) = append_function(
                &mut ctx,
                module_block,
                &caller_name,
                vec![actual, expected],
                vec![],
            );

            let mismatched_argument = caller_entry.deref(&ctx).get_argument(0);
            let mismatched = append_call(
                &mut ctx,
                caller_entry,
                &callee_name,
                vec![mismatched_argument],
                vec![],
            );
            assert!(
                mismatched.verify(&ctx).is_err(),
                "{actual_kind:?} (mutable={actual_mutable}) must not satisfy an in-module \
                 {expected_kind:?} (mutable={expected_mutable}) parameter"
            );

            let exact_argument = caller_entry.deref(&ctx).get_argument(1);
            let exact = append_call(
                &mut ctx,
                caller_entry,
                &callee_name,
                vec![exact_argument],
                vec![],
            );
            assert!(
                exact.verify(&ctx).is_ok(),
                "the exact {expected_kind:?} argument must verify"
            );
            append_return(&mut ctx, caller_entry);
        }

        let module_error = module
            .get_operation()
            .deref(&ctx)
            .verify(&ctx)
            .expect_err("the module must reject its nested mismatched calls");
        let module_error = module_error.disp(&ctx).to_string();
        assert!(
            module_error.contains("MirCallOp argument 0 type does not match callee signature"),
            "the mandatory whole-module gate must fail specifically on the nested call signature; got: {module_error}"
        );
    }

    #[test]
    fn in_module_call_accepts_the_logical_unit_result_bridge() {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let module = ModuleOp::new(&mut ctx, "unit_call".try_into().unwrap());
        let module_block = module_top_block(&mut ctx, &module);
        let (_, callee_entry) =
            append_function(&mut ctx, module_block, "drop_like", vec![], vec![]);
        append_return(&mut ctx, callee_entry);
        let (_, caller_entry) = append_function(&mut ctx, module_block, "caller", vec![], vec![]);
        let unit: TypeHandle = MirTupleType::get(&mut ctx, vec![]).into();
        let call = append_call(&mut ctx, caller_entry, "drop_like", vec![], vec![unit]);
        append_return(&mut ctx, caller_entry);

        assert!(
            call.verify(&ctx).is_ok(),
            "a Rust unit call result is the documented empty-signature bridge"
        );
    }

    #[test]
    fn unresolved_calls_require_an_independently_typed_foreign_signature() {
        let mut ctx = Context::new();
        crate::register(&mut ctx);

        let detached = new_call(&mut ctx, "ordinary_function", vec![], vec![]);
        assert!(
            detached.verify(&ctx).is_ok(),
            "construction-time verification must defer module symbol lookup"
        );

        let detached_function_type = FunctionType::get(&ctx, vec![], vec![]);
        let detached_function_operation = Operation::new(
            &mut ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let detached_function = MirFuncOp::new(
            &mut ctx,
            detached_function_operation,
            TypeAttr::new(detached_function_type.into()),
        );
        detached_function.set_symbol_name(&mut ctx, "detached_caller".try_into().unwrap());
        let detached_entry = BasicBlock::new(&mut ctx, None, vec![]);
        detached_entry.insert_at_back(detached_function_operation.deref(&ctx).get_region(0), &ctx);
        let importer_construction_call = append_call(
            &mut ctx,
            detached_entry,
            "ordinary_function",
            vec![],
            vec![],
        );
        assert!(
            importer_construction_call.verify(&ctx).is_ok(),
            "a call inside an as-yet-uninserted importer function must remain detached"
        );

        let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let raw_mut: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, pointee, true, MirPointerKind::RawMut)
                .into();
        let unique_ref: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, pointee, true, MirPointerKind::UniqueRef)
                .into();

        let top_level_module = ModuleOp::new(&mut ctx, "top_level_call".try_into().unwrap());
        let top_level_block = module_top_block(&mut ctx, &top_level_module);
        let attached_top_level = new_call(&mut ctx, "ordinary_function", vec![], vec![unique_ref]);
        attached_top_level
            .get_operation()
            .insert_at_back(top_level_block, &ctx);
        assert!(
            attached_top_level.verify(&ctx).is_err(),
            "a call directly in a final module must not claim detached-construction deferral"
        );

        let module = ModuleOp::new(&mut ctx, "foreign_calls".try_into().unwrap());
        let module_block = module_top_block(&mut ctx, &module);
        let (_, caller_entry) = append_function(
            &mut ctx,
            module_block,
            "caller",
            vec![raw_mut, unique_ref],
            vec![],
        );
        let raw_value = caller_entry.deref(&ctx).get_argument(0);
        let unique_value = caller_entry.deref(&ctx).get_argument(1);
        let signature: TypeHandle = FunctionType::get(&ctx, vec![raw_mut], vec![raw_mut]).into();

        let unresolved = append_call(
            &mut ctx,
            caller_entry,
            "external_raw",
            vec![raw_value],
            vec![raw_mut],
        );
        assert!(
            unresolved.verify(&ctx).is_err(),
            "an attached unresolved call must not pass merely because its pointer bits lower"
        );

        let exact = append_call(
            &mut ctx,
            caller_entry,
            "external_raw",
            vec![raw_value],
            vec![raw_mut],
        );
        exact.set_external_callee_signature(&mut ctx, signature);
        assert!(exact.verify(&ctx).is_ok());

        let wrong_argument = append_call(
            &mut ctx,
            caller_entry,
            "external_raw",
            vec![unique_value],
            vec![raw_mut],
        );
        wrong_argument.set_external_callee_signature(&mut ctx, signature);
        assert!(
            wrong_argument.verify(&ctx).is_err(),
            "AbiBoundary must not turn UniqueRef into the declared RawMut argument"
        );

        let wrong_result = append_call(
            &mut ctx,
            caller_entry,
            "external_raw",
            vec![raw_value],
            vec![unique_ref],
        );
        wrong_result.set_external_callee_signature(&mut ctx, signature);
        assert!(
            wrong_result.verify(&ctx).is_err(),
            "AbiBoundary must not turn the declared RawMut result into UniqueRef"
        );

        let unpaired_signature = append_call(
            &mut ctx,
            caller_entry,
            "external_raw",
            vec![raw_value],
            vec![raw_mut],
        );
        unpaired_signature.set_attr_external_callee_type(&ctx, TypeAttr::new(signature));
        assert!(
            unpaired_signature.verify(&ctx).is_err(),
            "a signature without the explicit ABI authority is incomplete"
        );

        let wrong_authority = append_call(
            &mut ctx,
            caller_entry,
            "external_raw",
            vec![raw_value],
            vec![raw_mut],
        );
        wrong_authority.set_attr_external_callee_type(&ctx, TypeAttr::new(signature));
        wrong_authority
            .set_attr_call_pointer_kind_authority(&ctx, MirPointerKindAuthorityAttr::Reborrow);
        assert!(
            wrong_authority.verify(&ctx).is_err(),
            "a non-ABI pointer authority must not bless an external signature"
        );

        let (_, local_entry) = append_function(
            &mut ctx,
            module_block,
            "local_raw",
            vec![raw_mut],
            vec![raw_mut],
        );
        append_return(&mut ctx, local_entry);
        let falsely_external = append_call(
            &mut ctx,
            caller_entry,
            "local_raw",
            vec![raw_value],
            vec![raw_mut],
        );
        falsely_external.set_external_callee_signature(&mut ctx, signature);
        assert!(
            falsely_external.verify(&ctx).is_err(),
            "an in-module Rust callee must be typed only by its mir.func"
        );
    }

    #[test]
    fn select_unpredictable_preserves_its_exact_generic_type() {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let module = ModuleOp::new(&mut ctx, "select_calls".try_into().unwrap());
        let module_block = module_top_block(&mut ctx, &module);
        let bool_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let raw_mut: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, pointee, true, MirPointerKind::RawMut)
                .into();
        let unique_ref: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, pointee, true, MirPointerKind::UniqueRef)
                .into();
        let (_, caller_entry) = append_function(
            &mut ctx,
            module_block,
            "caller",
            vec![bool_ty, raw_mut, unique_ref],
            vec![],
        );
        let condition = caller_entry.deref(&ctx).get_argument(0);
        let raw = caller_entry.deref(&ctx).get_argument(1);
        let unique = caller_entry.deref(&ctx).get_argument(2);

        let exact = append_call(
            &mut ctx,
            caller_entry,
            rust_intrinsics::CALLEE_SELECT_UNPREDICTABLE,
            vec![condition, raw, raw],
            vec![raw_mut],
        );
        assert!(exact.verify(&ctx).is_ok());

        let invented_result = append_call(
            &mut ctx,
            caller_entry,
            rust_intrinsics::CALLEE_SELECT_UNPREDICTABLE,
            vec![condition, raw, raw],
            vec![unique_ref],
        );
        assert!(
            invented_result.verify(&ctx).is_err(),
            "select must not launder RawMut operands into a UniqueRef result"
        );

        let mixed_branches = append_call(
            &mut ctx,
            caller_entry,
            rust_intrinsics::CALLEE_SELECT_UNPREDICTABLE,
            vec![condition, raw, unique],
            vec![raw_mut],
        );
        assert!(
            mixed_branches.verify(&ctx).is_err(),
            "both select alternatives must carry the same exact Rust type"
        );

        let nested_raw: TypeHandle = MirTupleType::get(&mut ctx, vec![raw_mut]).into();
        let nested_unique: TypeHandle = MirTupleType::get(&mut ctx, vec![unique_ref]).into();
        let (_, nested_entry) = append_function(
            &mut ctx,
            module_block,
            "nested_caller",
            vec![bool_ty, nested_raw],
            vec![],
        );
        let nested_condition = nested_entry.deref(&ctx).get_argument(0);
        let nested_value = nested_entry.deref(&ctx).get_argument(1);
        let nested_mismatch = append_call(
            &mut ctx,
            nested_entry,
            rust_intrinsics::CALLEE_SELECT_UNPREDICTABLE,
            vec![nested_condition, nested_value, nested_value],
            vec![nested_unique],
        );
        assert!(
            nested_mismatch.verify(&ctx).is_err(),
            "exact equality must recurse through aggregate pointer carriers"
        );
    }

    #[test]
    fn placeholder_names_are_closed_and_numeric_placeholders_reject_pointers() {
        let mut ctx = Context::new();
        crate::register(&mut ctx);

        let unknown = new_call(&mut ctx, "__cuda_oxide_rust_intrinsic_typo", vec![], vec![]);
        assert!(
            unknown.verify(&ctx).is_err(),
            "the reserved namespace must use an exact allow-list even while detached"
        );

        let module = ModuleOp::new(&mut ctx, "numeric_placeholder".try_into().unwrap());
        let module_block = module_top_block(&mut ctx, &module);
        let pointee: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let pointer: TypeHandle =
            MirPtrType::get_generic_with_kind(&mut ctx, pointee, true, MirPointerKind::RawMut)
                .into();
        let (_, caller_entry) =
            append_function(&mut ctx, module_block, "caller", vec![pointer], vec![]);
        let value = caller_entry.deref(&ctx).get_argument(0);
        let numeric = append_call(
            &mut ctx,
            caller_entry,
            rust_intrinsics::CALLEE_CTPOP,
            vec![value],
            vec![pointer],
        );
        assert!(
            numeric.verify(&ctx).is_err(),
            "numeric placeholders cannot serve as an untyped pointer escape hatch"
        );
    }
}
