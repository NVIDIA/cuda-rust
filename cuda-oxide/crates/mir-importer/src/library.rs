// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Retain authenticated native and storage operations in MIR.

use std::cell::RefCell;
use std::collections::BTreeMap;

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::terminator::helpers;
use crate::translator::values::ValueMap;
use crate::translator::{rvalue, types};
pub use dialect_mir::library::{LibraryCallPlan, LibraryCallSite, LibraryCallTargetAttr};
use dialect_mir::library::{
    LibraryNativeMetadata, LibraryRoleAttr, LibrarySourceEffectAttr, LibraryWriteSourceAttr,
};
use dialect_mir::ops::{MirLibraryOp, MirUndefOp};
use dialect_mir::types::MirTupleType;
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::FunctionType;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::Location;
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

/// Internal call targets, keyed by emitted caller and original collected callee
/// export name. Values name the same Rust body in that caller's root context.
pub type CallTargetRedirects = BTreeMap<String, BTreeMap<String, String>>;

/// Source semantics authenticated from the genuine definition and concrete types.
///
/// These inputs are independent from finalized provider plan flags.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LibraryCallMetadata {
    pub marker: String,
    pub role: LibraryRoleAttr,
}

#[derive(Default)]
struct SourceCallContext {
    effect: Option<LibrarySourceEffectAttr>,
    destination_write: Option<LibraryWriteSourceAttr>,
}

thread_local! {
    static SOURCE_CALL: RefCell<SourceCallContext> = const { RefCell::new(SourceCallContext { effect: None, destination_write: None }) };
    static SELECTED_CALL_TARGETS: RefCell<BTreeMap<LibraryCallSite, LibraryCallTargetAttr>> = const { RefCell::new(BTreeMap::new()) };
    static SEMANTICS: RefCell<BTreeMap<LibraryCallSite, LibraryCallMetadata>> = const { RefCell::new(BTreeMap::new()) };
}

/// Import-only lexical context for the one source call currently expanding.
/// The retained event observes the existing producer before its destination store.
pub(crate) struct SourceEffectGuard(SourceCallContext);
impl SourceEffectGuard {
    pub(crate) fn new(
        effect: Option<LibrarySourceEffectAttr>,
        destination_write: Option<LibraryWriteSourceAttr>,
    ) -> Self {
        Self(SOURCE_CALL.with(|current| {
            current.replace(SourceCallContext {
                effect,
                destination_write,
            })
        }))
    }
    pub(crate) fn finish(&self, ctx: &mut Context, continuation: Ptr<Operation>) {
        if let Some(effect) = SOURCE_CALL.with(|current| current.borrow_mut().effect.take()) {
            let marker = dialect_mir::ops::MirLibrarySourceOp::build(ctx, effect, None);
            let event_loc = continuation.deref(ctx).loc().clone();
            marker.deref_mut(ctx).set_loc(event_loc);
            marker.insert_before(ctx, continuation);
        }
    }
}
impl Drop for SourceEffectGuard {
    fn drop(&mut self) {
        SOURCE_CALL.with(|current| *current.borrow_mut() = std::mem::take(&mut self.0));
    }
}

/// Capture the current call's source write independently of its result effect.
pub(crate) fn current_destination_write_source() -> Option<LibraryWriteSourceAttr> {
    SOURCE_CALL.with(|current| current.borrow().destination_write.clone())
}

pub(crate) fn retain_source_result(
    ctx: &mut Context,
    value: pliron::value::Value,
    previous: Ptr<Operation>,
    loc: Location,
) -> Ptr<Operation> {
    let Some(effect) = SOURCE_CALL.with(|current| current.borrow_mut().effect.take()) else {
        return previous;
    };
    // Checked participation rejects inline assembly independently of its result.
    // Do not add a use that blocks forwarding its compiler-owned result bundle.
    let result = (!matches!(
        effect.kind,
        dialect_mir::library::LibrarySourceEffectKind::InlineAssembly
    ))
    .then_some(value);
    let marker = dialect_mir::ops::MirLibrarySourceOp::build(ctx, effect, result);
    marker.deref_mut(ctx).set_loc(loc);
    marker.insert_after(ctx, previous);
    marker
}

pub(crate) fn set_selected_call_targets(targets: BTreeMap<LibraryCallSite, LibraryCallTargetAttr>) {
    SELECTED_CALL_TARGETS.with(|current| *current.borrow_mut() = targets);
}

pub(crate) fn get_selected_call_target(site: &LibraryCallSite) -> Option<LibraryCallTargetAttr> {
    SELECTED_CALL_TARGETS.with(|current| current.borrow().get(site).cloned())
}

pub(crate) fn attach_call_source(ctx: &mut Context, op: Ptr<Operation>, site: LibraryCallSite) {
    let call = dialect_mir::ops::MirCallOp::new(op);
    if let Some(target) = get_selected_call_target(&site) {
        call.set_attr_library_call_target(ctx, target);
    }
    call.set_attr_library_source_site(ctx, site);
}

pub(crate) fn set_semantics(semantics: BTreeMap<LibraryCallSite, LibraryCallMetadata>) {
    SEMANTICS.with(|current| *current.borrow_mut() = semantics);
}

pub(crate) fn get_semantics(site: &LibraryCallSite) -> Option<LibraryCallMetadata> {
    SEMANTICS.with(|calls| calls.borrow().get(site).cloned())
}

/// Runtime argument positions in the genuine source marker, before ABI lowering.
fn runtime_indices(role: LibraryRoleAttr) -> &'static [usize] {
    use LibraryNativeMetadata as Native;
    match role {
        LibraryRoleAttr::Native(Native::Raw { .. }) => &[1],
        LibraryRoleAttr::Native(Native::Callback { .. }) => &[1, 3],
        LibraryRoleAttr::Native(Native::CheckedBlock { .. }) => &[1, 2],
        LibraryRoleAttr::Native(Native::CheckedBlockResource { .. }) => &[1, 2, 3],
        LibraryRoleAttr::Native(Native::CheckedBlockStorage { .. }) => &[1, 2, 3, 4],
        LibraryRoleAttr::StorageNew(_) => &[],
        LibraryRoleAttr::StorageReconfigure { .. }
        | LibraryRoleAttr::StorageReserve { .. }
        | LibraryRoleAttr::StorageRelease => &[0],
        LibraryRoleAttr::NativeOperandBoundary => unreachable!("boundary is importer-owned"),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn translate_call(
    ctx: &mut Context,
    body: &mir::Body,
    metadata: &LibraryCallMetadata,
    site: LibraryCallSite,
    instance: rustc_public::mir::mono::Instance,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block: Ptr<BasicBlock>,
    previous: Option<Ptr<Operation>>,
    values: &mut ValueMap,
    blocks: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    use rustc_public::ty::{Abi, RigidTy, TyKind};
    let signature = instance
        .ty()
        .kind()
        .fn_sig()
        .ok_or_else(|| {
            pliron::input_error!(
                loc.clone(),
                TranslationErr::invalid_op("library marker has no resolved signature")
            )
        })?
        .skip_binder();
    if instance.mangled_name() != metadata.marker
        || signature.abi != Abi::Rust
        || signature.c_variadic
        || signature.inputs().len() != args.len()
    {
        return input_err!(
            loc,
            TranslationErr::invalid_op("library marker signature changed after authentication")
        );
    }
    let mut previous = previous;
    if matches!(metadata.role, LibraryRoleAttr::Native(_)) {
        // The finalized before-reuse barrier must precede source operand loads,
        // not merely the eventual native call. This position has no completion
        // authority of its own; the late emitter consumes it with the call plan.
        let boundary = MirLibraryOp::build(
            ctx,
            LibraryRoleAttr::NativeOperandBoundary,
            site.clone(),
            metadata.marker.clone(),
        );
        boundary.deref_mut(ctx).set_loc(loc.clone());
        helpers::insert_op(ctx, boundary, block, previous);
        previous = Some(boundary);
    }
    let indices = runtime_indices(metadata.role);
    let mut operands = Vec::with_capacity(indices.len());
    let mut declared = Vec::with_capacity(indices.len());
    for &index in indices {
        let (Some(operand), Some(ty)) = (args.get(index), signature.inputs().get(index)) else {
            return input_err!(
                loc,
                TranslationErr::invalid_op(
                    "library marker operand count changed after authentication"
                )
            );
        };
        declared.push(types::translate_type(ctx, ty)?);
        let (value, last) =
            rvalue::translate_operand(ctx, body, operand, values, block, previous, loc.clone())?;
        operands.push(value);
        previous = last;
    }
    let unit = matches!(signature.output().kind(), TyKind::RigidTy(RigidTy::Tuple(fields)) if fields.is_empty());
    let results = if unit {
        vec![]
    } else {
        vec![types::translate_type(ctx, &signature.output())?]
    };
    let signature = FunctionType::get(ctx, declared, results);
    let (prepared, previous) =
        if target.is_none() && matches!(metadata.role, LibraryRoleAttr::Native(_)) {
            (None, previous)
        } else {
            let (prepared, previous) = helpers::prepare_destination_write(
                ctx,
                body,
                destination,
                values,
                block,
                previous,
                loc.clone(),
            )?;
            (Some(prepared), previous)
        };
    let retained = MirLibraryOp::build_call(
        ctx,
        metadata.role,
        site,
        metadata.marker.clone(),
        signature,
        operands,
    );
    retained.deref_mut(ctx).set_loc(loc.clone());
    helpers::insert_op(ctx, retained, block, previous);
    let Some(prepared) = prepared else {
        return Ok(crate::translator::terminator::emit_unreachable_after(
            ctx,
            block,
            Some(retained),
            loc,
        ));
    };
    let (result, previous) = if unit {
        let ty = MirTupleType::get(ctx, vec![]).into();
        let op = MirUndefOp::new(ctx, ty).get_operation();
        helpers::insert_op(ctx, op, block, Some(retained));
        (op.deref(ctx).get_result(0), op)
    } else {
        (retained.deref(ctx).get_result(0), retained)
    };
    helpers::emit_prepared_result_and_goto(
        ctx,
        prepared,
        result,
        target,
        block,
        previous,
        values,
        blocks,
        loc,
        "library marker requires a continuation",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::library::{LibrarySourceEffectKind, LibraryWriteKind};
    use dialect_mir::ops::{MirLibrarySourceOp, MirReturnOp};
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::linked_list::ContainsLinkedList;

    #[test]
    fn source_call_guard_restores_independent_effect_and_destination() {
        let site = LibraryCallSite {
            caller: "source".to_string(),
            block: 3,
        };
        let outer_effect = LibrarySourceEffectAttr {
            site: site.clone(),
            kind: LibrarySourceEffectKind::UnitHint,
        };
        let inner_effect = LibrarySourceEffectAttr {
            site: site.clone(),
            kind: LibrarySourceEffectKind::BlockQuery,
        };
        let write = LibraryWriteSourceAttr {
            site,
            statement: 2,
            kind: LibraryWriteKind::Indirect {
                nominal_protected: false,
            },
        };
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);
        let ty = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        let producer = Operation::new(
            &mut ctx,
            MirUndefOp::get_concrete_op_info(),
            vec![ty],
            vec![],
            vec![],
            0,
        );
        producer.insert_at_back(block, &ctx);
        let value = producer.deref(&ctx).get_result(0);
        let continuation = Operation::new(
            &mut ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![value],
            vec![],
            0,
        );
        continuation.insert_after(&ctx, producer);
        assert!(current_destination_write_source().is_none());
        {
            let outer = SourceEffectGuard::new(Some(outer_effect.clone()), Some(write.clone()));
            {
                let inner = SourceEffectGuard::new(Some(inner_effect.clone()), Some(write.clone()));
                let marker = retain_source_result(&mut ctx, value, producer, Location::Unknown);
                assert_eq!(marker.deref(&ctx).get_operand(0), value);
                assert_eq!(
                    *MirLibrarySourceOp::new(marker)
                        .get_attr_library_source_effect(&ctx)
                        .unwrap(),
                    inner_effect
                );
                assert_eq!(current_destination_write_source(), Some(write.clone()));
                assert!(SOURCE_CALL.with(|current| current.borrow().effect.is_none()));
                inner.finish(&mut ctx, continuation);
                assert_eq!(current_destination_write_source(), Some(write.clone()));
            }
            assert_eq!(
                SOURCE_CALL.with(|current| current.borrow().effect.clone()),
                Some(outer_effect.clone())
            );
            let early_return = || -> Result<(), ()> {
                let _inner = SourceEffectGuard::new(None, None);
                Err(())?;
                Ok(())
            };
            assert!(early_return().is_err());
            assert_eq!(current_destination_write_source(), Some(write));
            assert_eq!(
                SOURCE_CALL.with(|current| current.borrow().effect.clone()),
                Some(outer_effect)
            );
            outer.finish(&mut ctx, continuation);
            assert!(SOURCE_CALL.with(|current| current.borrow().effect.is_none()));
            assert!(current_destination_write_source().is_some());
        }
        assert!(current_destination_write_source().is_none());
        assert!(SOURCE_CALL.with(|current| current.borrow().effect.is_none()));
        assert_eq!(
            block
                .deref(&ctx)
                .iter(&ctx)
                .filter(|&op| Operation::get_op::<MirLibrarySourceOp>(op, &ctx).is_some())
                .count(),
            2,
            "result and unit effects are each retained once"
        );
    }
}
