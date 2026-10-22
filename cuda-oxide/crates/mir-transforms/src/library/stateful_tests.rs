/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;
use dialect_mir::{
    library::{LibraryFunctionSourceAttr, LibraryScalar, LibrarySourceEffectAttr, NativeSpec},
    types::{MirPointerKind, MirPtrType, MirTupleType},
};
use pliron::{
    builtin::{
        attributes::TypeAttr,
        op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::{FunctionType, IntegerType, Signedness},
    },
    op::Op,
    r#type::TypeHandle,
};

fn function(ctx: &mut Context) -> (Ptr<Operation>, Ptr<Operation>, Ptr<BasicBlock>) {
    dialect_mir::register(ctx);
    let module = ModuleOp::new(ctx, "stateful".try_into().unwrap());
    let signature = FunctionType::get(ctx, vec![], vec![]);
    let function = Operation::new(
        ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let f = MirFuncOp::new(ctx, function, TypeAttr::new(signature.into()));
    f.set_symbol_name(ctx, "root".try_into().unwrap());
    f.set_library_function_source(
        ctx,
        LibraryFunctionSourceAttr {
            export: "root".into(),
            mangled: "root_marker".into(),
            diagnostic: "root".into(),
            checked_ancestor: true,
        },
    );
    module.append_operation(ctx, function, 0);
    let block = BasicBlock::new(ctx, None, vec![]);
    block.insert_at_back(function.deref(ctx).get_region(0), ctx);
    (module.get_operation(), function, block)
}
fn emit<O: Op>(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    types: Vec<TypeHandle>,
    operands: Vec<Value>,
) -> Ptr<Operation> {
    let op = Operation::new(ctx, O::get_concrete_op_info(), types, operands, vec![], 0);
    op.insert_at_back(block, ctx);
    op
}
fn site(block: usize) -> LibraryCallSite {
    LibraryCallSite {
        caller: "root".into(),
        block,
    }
}
fn value(ctx: &mut Context, block: Ptr<BasicBlock>, ty: TypeHandle) -> Value {
    emit::<MirUndefOp>(ctx, block, vec![ty], vec![])
        .deref(ctx)
        .get_result(0)
}
#[allow(clippy::disallowed_methods)]
fn native(ctx: &mut Context, block: Ptr<BasicBlock>, control: Value, index: usize) -> Role {
    let scalar: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signed).into();
    let input: TypeHandle =
        MirPtrType::get_generic_with_kind(ctx, scalar, false, MirPointerKind::RawConst).into();
    let output: TypeHandle =
        MirPtrType::get_generic_with_kind(ctx, scalar, true, MirPointerKind::RawMut).into();
    let memory = value(ctx, block, input);
    MirLibrarySourceOp::build(
        ctx,
        LibrarySourceEffectAttr {
            site: site(index),
            kind: SourceKind::BlockQuery,
        },
        Some(memory),
    )
    .insert_at_back(block, ctx);
    let tuple: TypeHandle = MirTupleType::get(ctx, vec![input, scalar]).into();
    let controls = emit::<MirConstructTupleOp>(ctx, block, vec![tuple], vec![memory, control])
        .deref(ctx)
        .get_result(0);
    let payload: TypeHandle = MirTupleType::get(ctx, vec![output]).into();
    let payload_value = value(ctx, block, payload);
    let signature = FunctionType::get(ctx, vec![tuple, payload], vec![]);
    let role = Role::Native(Native::CheckedBlock {
        spec: NativeSpec {
            scalar: LibraryScalar::Signed(32),
            items: 1,
        },
    });
    MirLibraryOp::build_call(
        ctx,
        role,
        site(index),
        "checked_marker".into(),
        signature,
        vec![controls, payload_value],
    )
    .insert_at_back(block, ctx);
    role
}
fn source_value(ctx: &mut Context, block: Ptr<BasicBlock>, kind: SourceKind, slot: bool) -> Value {
    let scalar: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signed).into();
    let result = value(ctx, block, scalar);
    MirLibrarySourceOp::build(
        ctx,
        LibrarySourceEffectAttr {
            site: site(0),
            kind,
        },
        Some(result),
    )
    .insert_at_back(block, ctx);
    if slot {
        let pointer = MirPtrType::get_generic(ctx, scalar, true).into();
        let address = emit::<MirAllocaOp>(ctx, block, vec![pointer], vec![])
            .deref(ctx)
            .get_result(0);
        emit::<MirStoreOp>(ctx, block, vec![], vec![address, result]);
        emit::<MirLoadOp>(ctx, block, vec![scalar], vec![address])
            .deref(ctx)
            .get_result(0)
    } else {
        result
    }
}
fn analyze(
    ctx: &Context,
    module: Ptr<Operation>,
    manager: &mut AnalysisManager,
    roles: &BTreeMap<LibraryCallSite, Role>,
) -> pliron::result::Result<Validation> {
    analyze_libraries(
        module,
        ctx,
        manager,
        &BTreeMap::from([("root".into(), BTreeSet::from(["root".into()]))]),
        &BTreeMap::from([("root".into(), "root".into())]),
        roles,
        &BTreeMap::from([("root".into(), [64, 1, 1])]),
    )
}

#[test]
fn query_result_before_destination_store_has_the_same_ssa_and_full_slot_proof() {
    for kind in [SourceKind::BlockQuery, SourceKind::LaneQuery] {
        let ctx = &mut Context::new();
        let (module, _, block) = function(ctx);
        let control = source_value(ctx, block, kind, true);
        let role = native(ctx, block, control, 1);
        emit::<MirReturnOp>(ctx, block, vec![], vec![]);
        module.deref(ctx).verify(ctx).unwrap();
        let roles = BTreeMap::from([(site(1), role)]);
        let manager = &mut AnalysisManager::default();
        let full = analyze(ctx, module, manager, &roles);
        assert_eq!(full.is_ok(), kind == SourceKind::BlockQuery);
        if let Err(error) = full {
            assert!(error.to_string().contains("not proved block-uniform"));
        }
        pliron::opts::mem2reg::mem2reg(module, ctx, manager).unwrap();
        let mut invalidated = pliron::pass::PassResult::default();
        invalidated.ir_changed = pliron::irbuild::IRStatus::Changed;
        manager.retain_preserved(&invalidated);
        module.deref(ctx).verify(ctx).unwrap();
        assert!(
            !super::super::admission::operations(ctx, module)
                .iter()
                .any(|&op| Operation::get_op::<MirAllocaOp>(op, ctx).is_some())
        );
        let ssa = analyze(ctx, module, manager, &roles);
        assert_eq!(ssa.is_ok(), kind == SourceKind::BlockQuery);
        if let Err(error) = ssa {
            assert!(error.to_string().contains("not proved block-uniform"));
        }
    }
}

#[test]
fn separate_helper_returns_keep_selection_taint_after_promotion() {
    use dialect_mir::library::{LibraryCallKind, LibraryCallTargetAttr};
    use pliron::builtin::{
        attributes::{OperandSegmentSizesAttr, StringAttr},
        op_interfaces::ATTR_KEY_OPERAND_SEGMENT_SIZES,
    };

    for kind in [SourceKind::LaneQuery, SourceKind::BlockQuery] {
        let ctx = &mut Context::new();
        let (module, _, root) = function(ctx);
        let scalar: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signed).into();
        let bit: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
        let condition = value(ctx, root, bit);
        MirLibrarySourceOp::build(
            ctx,
            LibrarySourceEffectAttr {
                site: site(0),
                kind,
            },
            Some(condition),
        )
        .insert_at_back(root, ctx);
        let first = source_value(ctx, root, SourceKind::BlockQuery, false);
        let second = source_value(ctx, root, SourceKind::BlockQuery, false);
        assert_ne!(first, second);

        let signature = FunctionType::get(ctx, vec![bit, scalar, scalar], vec![scalar]);
        let helper = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let helper_op = MirFuncOp::new(ctx, helper, TypeAttr::new(signature.into()));
        helper_op.set_symbol_name(ctx, "choose_return".try_into().unwrap());
        helper_op.set_library_function_source(
            ctx,
            LibraryFunctionSourceAttr {
                export: "choose_return".into(),
                mangled: "choose_return_marker".into(),
                diagnostic: "choose_return".into(),
                checked_ancestor: false,
            },
        );
        Operation::get_op::<ModuleOp>(module, ctx)
            .unwrap()
            .append_operation(ctx, helper, 0);
        let entry = BasicBlock::new(ctx, None, vec![bit, scalar, scalar]);
        let left = BasicBlock::new(ctx, None, vec![]);
        let right = BasicBlock::new(ctx, None, vec![]);
        for block in [entry, left, right] {
            block.insert_at_back(helper.deref(ctx).get_region(0), ctx);
        }
        let arguments: Vec<_> = (0..3).map(|i| entry.deref(ctx).get_argument(i)).collect();
        let pointer = MirPtrType::get_generic(ctx, scalar, true).into();
        let address = emit::<MirAllocaOp>(ctx, entry, vec![pointer], vec![])
            .deref(ctx)
            .get_result(0);
        let branch = Operation::new(
            ctx,
            MirCondBranchOp::get_concrete_op_info(),
            vec![],
            vec![arguments[0]],
            vec![left, right],
            0,
        );
        branch.deref_mut(ctx).attributes.set(
            ATTR_KEY_OPERAND_SEGMENT_SIZES.clone(),
            OperandSegmentSizesAttr(vec![1, 0, 0]),
        );
        branch.insert_at_back(entry, ctx);
        let mut returns = Vec::new();
        for (block, argument) in [(left, arguments[1]), (right, arguments[2])] {
            // The importer materializes each source return through local _0.
            emit::<MirStoreOp>(ctx, block, vec![], vec![address, argument]);
            let loaded = emit::<MirLoadOp>(ctx, block, vec![scalar], vec![address])
                .deref(ctx)
                .get_result(0);
            returns.push(emit::<MirReturnOp>(ctx, block, vec![], vec![loaded]));
        }
        let call = emit::<MirCallOp>(ctx, root, vec![scalar], vec![condition, first, second]);
        let call_op = MirCallOp::new(call);
        call_op.set_attr_callee(ctx, StringAttr::new("choose_return".into()));
        call_op.set_attr_library_source_site(ctx, site(0));
        call_op.set_attr_library_call_target(
            ctx,
            LibraryCallTargetAttr {
                kind: LibraryCallKind::Direct,
                export: "choose_return".into(),
                marker: "choose_return_marker".into(),
            },
        );
        let control = call.deref(ctx).get_result(0);
        let role = native(ctx, root, control, 1);
        emit::<MirReturnOp>(ctx, root, vec![], vec![]);
        let roles = BTreeMap::from([(site(1), role)]);
        let manager = &mut AnalysisManager::default();
        for promoted in [false, true] {
            if promoted {
                pliron::opts::mem2reg::mem2reg(module, ctx, manager).unwrap();
                let mut invalidated = pliron::pass::PassResult::default();
                invalidated.ir_changed = pliron::irbuild::IRStatus::Changed;
                manager.retain_preserved(&invalidated);
                assert!(
                    !super::super::admission::operations(ctx, module)
                        .iter()
                        .any(|&op| Operation::get_op::<MirAllocaOp>(op, ctx).is_some())
                );
                for (&ret, &argument) in returns.iter().zip(&arguments[1..]) {
                    assert!(Operation::get_op::<MirReturnOp>(ret, ctx).is_some());
                    assert_eq!(ret.deref(ctx).get_operand(0), argument);
                }
            }
            module.deref(ctx).verify(ctx).unwrap();
            let result = analyze_libraries(
                module,
                ctx,
                manager,
                &BTreeMap::from([(
                    "root".into(),
                    BTreeSet::from(["root".into(), "choose_return".into()]),
                )]),
                &BTreeMap::from([
                    ("root".into(), "root".into()),
                    ("choose_return".into(), "choose_return".into()),
                ]),
                &roles,
                &BTreeMap::from([("root".into(), [64, 1, 1])]),
            );
            assert_eq!(
                result.is_ok(),
                kind == SourceKind::BlockQuery,
                "promoted={promoted}"
            );
            if let Err(error) = result {
                assert!(error.to_string().contains("not proved block-uniform"));
            }
        }
    }
}

#[test]
fn same_successor_arguments_keep_their_lane_varying_selection() {
    use pliron::builtin::{
        attributes::OperandSegmentSizesAttr, op_interfaces::ATTR_KEY_OPERAND_SEGMENT_SIZES,
    };
    for identical in [false, true] {
        let ctx = &mut Context::new();
        let (module, f, entry) = function(ctx);
        let bit: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
        let condition = value(ctx, entry, bit);
        MirLibrarySourceOp::build(
            ctx,
            LibrarySourceEffectAttr {
                site: site(0),
                kind: SourceKind::LaneQuery,
            },
            Some(condition),
        )
        .insert_at_back(entry, ctx);
        let a = source_value(ctx, entry, SourceKind::BlockQuery, false);
        let b = if identical {
            a
        } else {
            source_value(ctx, entry, SourceKind::BlockQuery, false)
        };
        let join = BasicBlock::new(ctx, None, vec![a.get_type(ctx)]);
        join.insert_at_back(f.deref(ctx).get_region(0), ctx);
        let branch = Operation::new(
            ctx,
            MirCondBranchOp::get_concrete_op_info(),
            vec![],
            vec![condition, a, b],
            vec![join, join],
            0,
        );
        branch.deref_mut(ctx).attributes.set(
            ATTR_KEY_OPERAND_SEGMENT_SIZES.clone(),
            OperandSegmentSizesAttr(vec![1, 1, 1]),
        );
        branch.insert_at_back(entry, ctx);
        let argument = join.deref(ctx).get_argument(0);
        let role = native(ctx, join, argument, 4);
        emit::<MirReturnOp>(ctx, join, vec![], vec![]);
        module.deref(ctx).verify(ctx).unwrap();
        let result = analyze(
            ctx,
            module,
            &mut AnalysisManager::default(),
            &BTreeMap::from([(site(4), role)]),
        );
        assert_eq!(result.is_ok(), identical);
        if let Err(error) = result {
            assert!(error.to_string().contains("not proved block-uniform"));
        }
    }
}

fn test_constant(ctx: &mut Context, block: Ptr<BasicBlock>, width: u32, n: u64) -> Value {
    use pliron::{builtin::attributes::IntegerAttr, utils::apint::APInt};
    let ty = IntegerType::get(ctx, width, Signedness::Unsigned);
    let op = emit::<MirConstantOp>(ctx, block, vec![ty.into()], vec![]);
    MirConstantOp::new(op).set_attr_value(
        ctx,
        IntegerAttr::new(
            ty,
            APInt::from_u64(n, std::num::NonZeroUsize::new(width as usize).unwrap()),
        ),
    );
    op.deref(ctx).get_result(0)
}

#[allow(clippy::disallowed_methods)]
fn test_slot(ctx: &mut Context, block: Ptr<BasicBlock>, initial: Value) -> Value {
    let pointer = MirPtrType::get_generic(ctx, initial.get_type(ctx), true).into();
    let address = emit::<MirAllocaOp>(ctx, block, vec![pointer], vec![])
        .deref(ctx)
        .get_result(0);
    emit::<MirStoreOp>(ctx, block, vec![], vec![address, initial]);
    address
}

fn test_helper(
    ctx: &mut Context,
    module: Ptr<Operation>,
    inputs: Vec<TypeHandle>,
    outputs: Vec<TypeHandle>,
) -> Ptr<BasicBlock> {
    let signature = FunctionType::get(ctx, inputs.clone(), outputs);
    let op = Operation::new(
        ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let f = MirFuncOp::new(ctx, op, TypeAttr::new(signature.into()));
    f.set_symbol_name(ctx, "helper".try_into().unwrap());
    f.set_library_function_source(
        ctx,
        LibraryFunctionSourceAttr {
            export: "helper".into(),
            mangled: "helper_marker".into(),
            diagnostic: "helper".into(),
            checked_ancestor: false,
        },
    );
    Operation::get_op::<ModuleOp>(module, ctx)
        .unwrap()
        .append_operation(ctx, op, 0);
    let block = BasicBlock::new(ctx, None, inputs);
    block.insert_at_back(op.deref(ctx).get_region(0), ctx);
    block
}

fn test_call(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    inputs: Vec<Value>,
    outputs: Vec<TypeHandle>,
) -> Ptr<Operation> {
    use dialect_mir::library::LibraryCallTargetAttr;
    use pliron::builtin::attributes::StringAttr;
    let op = emit::<MirCallOp>(ctx, block, outputs, inputs);
    let call = MirCallOp::new(op);
    call.set_attr_callee(ctx, StringAttr::new("helper".into()));
    call.set_attr_library_source_site(ctx, site(0));
    call.set_attr_library_call_target(
        ctx,
        LibraryCallTargetAttr {
            kind: LibraryCallKind::Direct,
            export: "helper".into(),
            marker: "helper_marker".into(),
        },
    );
    op
}

fn slot_and_promoted_verdicts(
    ctx: &mut Context,
    module: Ptr<Operation>,
    roles: &BTreeMap<LibraryCallSite, Role>,
) -> Vec<Result<(), String>> {
    let manager = &mut AnalysisManager::default();
    [false, true]
        .into_iter()
        .map(|promoted| {
            if promoted {
                pliron::opts::mem2reg::mem2reg(module, ctx, manager).unwrap();
                let mut changed = pliron::pass::PassResult::default();
                changed.ir_changed = pliron::irbuild::IRStatus::Changed;
                manager.retain_preserved(&changed);
            }
            module.deref(ctx).verify(ctx).unwrap();
            analyze_libraries(
                module,
                ctx,
                manager,
                &BTreeMap::from([(
                    "root".into(),
                    BTreeSet::from(["root".into(), "helper".into()]),
                )]),
                &BTreeMap::from([
                    ("root".into(), "root".into()),
                    ("helper".into(), "helper".into()),
                ]),
                roles,
                &BTreeMap::from([("root".into(), [64, 1, 1])]),
            )
            .map(|_| ())
            .map_err(|error| error.to_string())
        })
        .collect()
}

#[test]
fn engine_review_merged_slot_alias_invalidates_only_exposed_controls() {
    use pliron::builtin::{
        attributes::OperandSegmentSizesAttr, op_interfaces::ATTR_KEY_OPERAND_SEGMENT_SIZES,
    };
    for read_escaped in [false, true] {
        let ctx = &mut Context::new();
        let (module, f, entry) = function(ctx);
        let uniform = source_value(ctx, entry, SourceKind::BlockQuery, false);
        let lane = source_value(ctx, entry, SourceKind::LaneQuery, false);
        let a = test_slot(ctx, entry, uniform);
        let b = test_slot(ctx, entry, uniform);
        let untouched = test_slot(ctx, entry, uniform);
        // A proof-only address observation must not make ordinary locals borrowed.
        MirLibraryWriteCheckOp::build(
            ctx,
            library::LibraryWriteSourceAttr {
                site: site(0),
                statement: 0,
                kind: LibraryWriteKind::Indirect {
                    nominal_protected: false,
                },
            },
            vec![untouched],
        )
        .insert_at_back(entry, ctx);
        let bit = IntegerType::get(ctx, 1, Signedness::Signless).into();
        let condition = value(ctx, entry, bit);
        MirLibrarySourceOp::build(
            ctx,
            LibrarySourceEffectAttr {
                site: site(0),
                kind: SourceKind::BlockQuery,
            },
            Some(condition),
        )
        .insert_at_back(entry, ctx);
        let left = BasicBlock::new(ctx, None, vec![]);
        let right = BasicBlock::new(ctx, None, vec![]);
        let join = BasicBlock::new(ctx, None, vec![a.get_type(ctx)]);
        for block in [left, right, join] {
            block.insert_at_back(f.deref(ctx).get_region(0), ctx);
        }
        let branch = Operation::new(
            ctx,
            MirCondBranchOp::get_concrete_op_info(),
            vec![],
            vec![condition],
            vec![left, right],
            0,
        );
        branch.deref_mut(ctx).attributes.set(
            ATTR_KEY_OPERAND_SEGMENT_SIZES.clone(),
            OperandSegmentSizesAttr(vec![1, 0, 0]),
        );
        branch.insert_at_back(entry, ctx);
        for (block, pointer) in [(left, a), (right, b)] {
            Operation::new(
                ctx,
                MirGotoOp::get_concrete_op_info(),
                vec![],
                vec![pointer],
                vec![join],
                0,
            )
            .insert_at_back(block, ctx);
        }
        let selected = join.deref(ctx).get_argument(0);
        emit::<MirStoreOp>(ctx, join, vec![], vec![selected, lane]);
        let address = if read_escaped { a } else { untouched };
        let control = emit::<MirLoadOp>(ctx, join, vec![uniform.get_type(ctx)], vec![address])
            .deref(ctx)
            .get_result(0);
        let role = native(ctx, join, control, 4);
        emit::<MirReturnOp>(ctx, join, vec![], vec![]);
        let verdicts = slot_and_promoted_verdicts(ctx, module, &BTreeMap::from([(site(4), role)]));
        assert_eq!(
            verdicts.iter().map(Result::is_ok).collect::<Vec<_>>(),
            vec![!read_escaped; 2],
            "{verdicts:?}"
        );
        if read_escaped {
            assert!(
                verdicts
                    .iter()
                    .all(|v| v.as_ref().unwrap_err().contains("not proved block-uniform"))
            );
        }
    }
}

#[test]
fn engine_review_aggregate_slot_alias_survives_helper_invalidation() {
    use dialect_mir::attributes::FieldIndexAttr;
    for read_escaped in [false, true] {
        let ctx = &mut Context::new();
        let (module, _, root) = function(ctx);
        let uniform = source_value(ctx, root, SourceKind::BlockQuery, false);
        let lane = source_value(ctx, root, SourceKind::LaneQuery, false);
        let scalar = uniform.get_type(ctx);
        let address = test_slot(ctx, root, uniform);
        let untouched = test_slot(ctx, root, uniform);
        let tuple = MirTupleType::get(ctx, vec![address.get_type(ctx)]).into();
        let aggregate = emit::<MirConstructTupleOp>(ctx, root, vec![tuple], vec![address])
            .deref(ctx)
            .get_result(0);
        let captured = test_slot(ctx, root, aggregate);
        let helper = test_helper(ctx, module, vec![captured.get_type(ctx), scalar], vec![]);
        let capture = helper.deref(ctx).get_argument(0);
        let loaded = emit::<MirLoadOp>(ctx, helper, vec![tuple], vec![capture])
            .deref(ctx)
            .get_result(0);
        let field =
            emit::<MirExtractFieldOp>(ctx, helper, vec![address.get_type(ctx)], vec![loaded]);
        MirExtractFieldOp::new(field).set_attr_index(ctx, FieldIndexAttr(0));
        let field = field.deref(ctx).get_result(0);
        let written = helper.deref(ctx).get_argument(1);
        emit::<MirStoreOp>(ctx, helper, vec![], vec![field, written]);
        emit::<MirReturnOp>(ctx, helper, vec![], vec![]);
        test_call(ctx, root, vec![captured, lane], vec![]);
        let control = emit::<MirLoadOp>(
            ctx,
            root,
            vec![scalar],
            vec![if read_escaped { address } else { untouched }],
        )
        .deref(ctx)
        .get_result(0);
        let role = native(ctx, root, control, 1);
        emit::<MirReturnOp>(ctx, root, vec![], vec![]);
        let verdicts = slot_and_promoted_verdicts(ctx, module, &BTreeMap::from([(site(1), role)]));
        assert_eq!(
            verdicts.iter().map(Result::is_ok).collect::<Vec<_>>(),
            vec![!read_escaped; 2],
            "{verdicts:?}"
        );
        if read_escaped {
            assert!(
                verdicts
                    .iter()
                    .all(|v| v.as_ref().unwrap_err().contains("not proved block-uniform"))
            );
        }
    }
}

#[test]
#[allow(clippy::disallowed_methods)]
fn engine_review_array_element_store_does_not_replace_whole_fact() {
    use dialect_mir::{
        attributes::{FieldIndexAttr, MirCastKindAttr, MirPointerKindAuthorityAttr},
        types::MirArrayType,
    };
    #[derive(Clone, Copy, Debug)]
    enum CastPath {
        Direct,
        ZeroOffset,
        Tuple,
    }
    let mut failures = Vec::new();
    for (overwritten, cast_path) in [
        (None, None),
        (Some(1), None),
        (Some(0), None),
        (Some(0), Some(CastPath::Direct)),
        (Some(0), Some(CastPath::ZeroOffset)),
        (Some(0), Some(CastPath::Tuple)),
    ] {
        let ctx = &mut Context::new();
        let (module, _, root) = function(ctx);
        let uniform = source_value(ctx, root, SourceKind::BlockQuery, false);
        let item = if overwritten.is_some() {
            source_value(ctx, root, SourceKind::LaneQuery, false)
        } else {
            uniform
        };
        let scalar = item.get_type(ctx);
        let array = MirArrayType::get(ctx, scalar, 4).into();
        let items = emit::<MirConstructArrayOp>(ctx, root, vec![array], vec![item; 4])
            .deref(ctx)
            .get_result(0);
        let address = test_slot(ctx, root, items);
        // Pointee-preserving pointer-kind boundaries retain the same slot.
        let raw_array =
            MirPtrType::get_generic_with_kind(ctx, array, true, MirPointerKind::RawMut).into();
        let cast = emit::<MirCastOp>(ctx, root, vec![raw_array], vec![address]);
        MirCastOp::new(cast).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
        MirCastOp::new(cast)
            .set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::RawAddress);
        let address = cast.deref(ctx).get_result(0);
        if let Some(index) = overwritten {
            let index = test_constant(ctx, root, 32, index);
            let element =
                MirPtrType::get_generic_with_kind(ctx, scalar, true, MirPointerKind::RawMut).into();
            let element = if matches!(cast_path, Some(CastPath::Tuple)) {
                let source_tuple = MirTupleType::get(ctx, vec![raw_array]).into();
                let target_tuple = MirTupleType::get(ctx, vec![element]).into();
                let tuple =
                    emit::<MirConstructTupleOp>(ctx, root, vec![source_tuple], vec![address])
                        .deref(ctx)
                        .get_result(0);
                let cast = emit::<MirCastOp>(ctx, root, vec![target_tuple], vec![tuple]);
                MirCastOp::new(cast).set_attr_cast_kind(ctx, MirCastKindAttr::Transmute);
                MirCastOp::new(cast)
                    .set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::RustCast);
                let tuple = cast.deref(ctx).get_result(0);
                let extracted = emit::<MirExtractFieldOp>(ctx, root, vec![element], vec![tuple]);
                MirExtractFieldOp::new(extracted).set_attr_index(ctx, FieldIndexAttr(0));
                extracted.deref(ctx).get_result(0)
            } else if let Some(path) = cast_path {
                let cast = emit::<MirCastOp>(ctx, root, vec![element], vec![address]);
                MirCastOp::new(cast)
                    .set_attr_cast_kind(ctx, MirCastKindAttr::PointerCoercionArrayToPointer);
                MirCastOp::new(cast)
                    .set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::RustCast);
                let decayed = cast.deref(ctx).get_result(0);
                if matches!(path, CastPath::ZeroOffset) {
                    emit::<MirPtrOffsetOp>(ctx, root, vec![element], vec![decayed, index])
                        .deref(ctx)
                        .get_result(0)
                } else {
                    decayed
                }
            } else {
                emit::<MirArrayElementAddrOp>(ctx, root, vec![element], vec![address, index])
                    .deref(ctx)
                    .get_result(0)
            };
            emit::<MirStoreOp>(ctx, root, vec![], vec![element, uniform]);
        }
        let helper = test_helper(ctx, module, vec![array], vec![scalar]);
        let index = test_constant(ctx, helper, 32, 1);
        let argument = helper.deref(ctx).get_argument(0);
        let extracted =
            emit::<MirExtractArrayElementOp>(ctx, helper, vec![scalar], vec![argument, index])
                .deref(ctx)
                .get_result(0);
        emit::<MirReturnOp>(ctx, helper, vec![], vec![extracted]);
        let loaded = emit::<MirLoadOp>(ctx, root, vec![array], vec![address])
            .deref(ctx)
            .get_result(0);
        let call = test_call(ctx, root, vec![loaded], vec![scalar]);
        let result = call.deref(ctx).get_result(0);
        let role = native(ctx, root, result, 1);
        emit::<MirReturnOp>(ctx, root, vec![], vec![]);
        let verdicts = slot_and_promoted_verdicts(ctx, module, &BTreeMap::from([(site(1), role)]));
        if verdicts.iter().map(Result::is_ok).collect::<Vec<_>>() != vec![overwritten.is_none(); 2]
        {
            failures.push(format!(
                "index={overwritten:?}, cast_path={cast_path:?}: {verdicts:?}"
            ));
            continue;
        }
        if overwritten.is_some() {
            assert!(
                verdicts
                    .iter()
                    .all(|v| v.as_ref().unwrap_err().contains("not proved block-uniform"))
            );
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn bulk_transfers_invalidate_escaped_slots_and_sibling_fields() {
    #[derive(Clone, Copy, Debug)]
    enum Destination {
        Scalar,
        AdjacentField,
        Unrelated,
    }
    fn field(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        address: Value,
        scalar: TypeHandle,
        index: u32,
    ) -> Value {
        let pointer = MirPtrType::get_generic(ctx, scalar, true).into();
        let op = MirFieldAddrOp::build(ctx, address, pointer, index).unwrap();
        op.insert_at_back(block, ctx);
        op.deref(ctx).get_result(0)
    }
    let mut failures = Vec::new();
    for moving in [false, true] {
        for destination in [
            Destination::Scalar,
            Destination::AdjacentField,
            Destination::Unrelated,
        ] {
            let ctx = &mut Context::new();
            let (module, _, root) = function(ctx);
            let uniform = source_value(ctx, root, SourceKind::BlockQuery, false);
            let lane = source_value(ctx, root, SourceKind::LaneQuery, false);
            let scalar = uniform.get_type(ctx);
            let (dst, src, observed, count) = if matches!(destination, Destination::AdjacentField) {
                let pair = MirTupleType::get(ctx, vec![scalar, scalar]).into();
                let initial = emit::<MirConstructTupleOp>(ctx, root, vec![pair], vec![uniform; 2])
                    .deref(ctx)
                    .get_result(0);
                let replacement = emit::<MirConstructTupleOp>(ctx, root, vec![pair], vec![lane; 2])
                    .deref(ctx)
                    .get_result(0);
                let dst = test_slot(ctx, root, initial);
                let src = test_slot(ctx, root, replacement);
                let observed = field(ctx, root, dst, scalar, 1);
                // Two scalar elements overwrite both adjacent fields. Invalidating
                // just the first field would leave the observed sibling stale.
                (
                    field(ctx, root, dst, scalar, 0),
                    field(ctx, root, src, scalar, 0),
                    observed,
                    2,
                )
            } else {
                let dst = test_slot(ctx, root, uniform);
                let src = test_slot(ctx, root, lane);
                let observed = if matches!(destination, Destination::Unrelated) {
                    test_slot(ctx, root, uniform)
                } else {
                    dst
                };
                (dst, src, observed, 1)
            };
            let count = test_constant(ctx, root, 64, count);
            let transfer = if moving {
                MirMemmoveOp::build(ctx, dst, src, count)
            } else {
                MirMemcpyOp::build(ctx, dst, src, count)
            }
            .unwrap();
            transfer.insert_at_back(root, ctx);
            let control = emit::<MirLoadOp>(ctx, root, vec![scalar], vec![observed])
                .deref(ctx)
                .get_result(0);
            let role = native(ctx, root, control, 1);
            emit::<MirReturnOp>(ctx, root, vec![], vec![]);
            let verdicts =
                slot_and_promoted_verdicts(ctx, module, &BTreeMap::from([(site(1), role)]));
            let accepted = matches!(destination, Destination::Unrelated);
            if verdicts.iter().map(Result::is_ok).collect::<Vec<_>>() != vec![accepted; 2] {
                failures.push(format!(
                    "moving={moving}, destination={destination:?}: {verdicts:?}"
                ));
            } else if !accepted {
                assert!(
                    verdicts
                        .iter()
                        .all(|v| v.as_ref().unwrap_err().contains("not proved block-uniform"))
                );
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn unchecked_indexing_is_rejected_in_roots_and_ordinary_helpers() {
    for marked in [None, Some("root"), Some("helper")] {
        let ctx = &mut Context::new();
        let (module, root_op, root) = function(ctx);
        let control = source_value(ctx, root, SourceKind::BlockQuery, true);
        let scalar = control.get_type(ctx);
        let helper = test_helper(ctx, module, vec![scalar], vec![scalar]);
        let argument = helper.deref(ctx).get_argument(0);
        emit::<MirReturnOp>(ctx, helper, vec![], vec![argument]);
        if let Some(name) = marked {
            let op = if name == "root" {
                root_op
            } else {
                super::super::admission::operations(ctx, module)
                    .into_iter()
                    .find(|&op| {
                        Operation::get_op::<MirFuncOp>(op, ctx)
                            .is_some_and(|f| f.get_symbol_name(ctx).to_string() == "helper")
                    })
                    .unwrap()
            };
            MirFuncOp::wrap(ctx, op)
                .unwrap()
                .set_library_unchecked_indexing(ctx, true);
        }
        let call = test_call(ctx, root, vec![control], vec![scalar]);
        let result = call.deref(ctx).get_result(0);
        let role = native(ctx, root, result, 1);
        emit::<MirReturnOp>(ctx, root, vec![], vec![]);
        let verdicts = slot_and_promoted_verdicts(ctx, module, &BTreeMap::from([(site(1), role)]));
        assert_eq!(
            verdicts.iter().map(Result::is_ok).collect::<Vec<_>>(),
            vec![marked.is_none(); 2],
            "marked={marked:?}: {verdicts:?}"
        );
        if let Some(name) = marked {
            for verdict in verdicts {
                let diagnostic = verdict.unwrap_err();
                assert!(diagnostic.contains(
                    "checked block collectives do not support unchecked indexing; retain runtime bounds checks"
                ));
                assert!(diagnostic.contains(&format!("{name} MIR block 0")));
            }
        }
    }
}

#[test]
#[allow(clippy::disallowed_methods)]
fn resource_access_requires_completion_and_retains_original_witness_origin() {
    use dialect_mir::{library::LibraryWitness, types::MirDisjointSliceType};
    for completed in [false, true] {
        let ctx = &mut Context::new();
        dialect_mir::register(ctx);
        let module = ModuleOp::new(ctx, "resource".try_into().unwrap());
        let scalar: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signed).into();
        let carrier: TypeHandle = MirDisjointSliceType::get(ctx, scalar).into();
        let signature = FunctionType::get(ctx, vec![carrier], vec![]);
        let function = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let f = MirFuncOp::new(ctx, function, TypeAttr::new(signature.into()));
        f.set_symbol_name(ctx, "root".try_into().unwrap());
        f.set_library_root_argument(ctx, 0, LibraryRootArgumentClass::ResourceCandidate);
        module.append_operation(ctx, function, 0);
        let block = BasicBlock::new(ctx, None, vec![carrier]);
        block.insert_at_back(function.deref(ctx).get_region(0), ctx);
        let witness_ty =
            MirPtrType::get_generic_with_kind(ctx, carrier, true, MirPointerKind::UniqueRef).into();
        let argument = block.deref(ctx).get_argument(0);
        let witness = emit::<MirRefOp>(ctx, block, vec![witness_ty], vec![argument]);
        MirRefOp::new(witness).set_mutable(ctx, true);
        MirRefOp::new(witness).set_pointer_kind_authority(
            ctx,
            dialect_mir::attributes::MirPointerKindAuthorityAttr::Reborrow,
        );
        let witness = witness.deref(ctx).get_result(0);
        let pointer: TypeHandle =
            MirPtrType::get_generic_with_kind(ctx, scalar, true, MirPointerKind::RawMut).into();
        let memory = value(ctx, block, pointer);
        MirLibrarySourceOp::build(
            ctx,
            LibrarySourceEffectAttr {
                site: site(0),
                kind: SourceKind::BlockQuery,
            },
            Some(memory),
        )
        .insert_at_back(block, ctx);
        let tuple: TypeHandle = MirTupleType::get(ctx, vec![pointer]).into();
        let uniform = emit::<MirConstructTupleOp>(ctx, block, vec![tuple], vec![memory])
            .deref(ctx)
            .get_result(0);
        let payload = value(ctx, block, tuple);
        let call_ty = FunctionType::get(ctx, vec![witness_ty, tuple, tuple], vec![]);
        let mut roles = BTreeMap::new();
        for (index, access) in [(1, LibraryAccess::Read), (2, LibraryAccess::Write)] {
            if index == 2 && completed {
                MirLibrarySourceOp::build(
                    ctx,
                    LibrarySourceEffectAttr {
                        site: site(3),
                        kind: SourceKind::BlockSync,
                    },
                    None,
                )
                .insert_at_back(block, ctx);
            }
            let role = Role::Native(Native::CheckedBlockResource {
                spec: NativeSpec {
                    scalar: LibraryScalar::Signed(32),
                    items: 1,
                },
                access,
                witness: LibraryWitness::CollectiveCarrier(1),
            });
            MirLibraryOp::build_call(
                ctx,
                role,
                site(index),
                "resource_marker".into(),
                call_ty,
                vec![witness, uniform, payload],
            )
            .insert_at_back(block, ctx);
            roles.insert(site(index), role);
        }
        emit::<MirReturnOp>(ctx, block, vec![], vec![]);
        let verdicts = slot_and_promoted_verdicts(ctx, module.get_operation(), &roles);
        assert_eq!(
            verdicts.iter().map(Result::is_ok).collect::<Vec<_>>(),
            vec![completed; 2],
            "{verdicts:?}"
        );
        if !completed {
            assert!(verdicts.iter().all(|v| {
                v.as_ref()
                    .unwrap_err()
                    .contains("requires a full-block barrier")
            }));
        }
    }
}

#[test]
fn varying_cycles_stop_at_reconvergence() {
    use pliron::builtin::{
        attributes::OperandSegmentSizesAttr, op_interfaces::ATTR_KEY_OPERAND_SEGMENT_SIZES,
    };

    // The second cycle never returns to the varying branch itself. The last
    // case reconverges its varying choice before a uniform outer backedge.
    let direct = vec![vec![1], vec![2, 3], vec![1], vec![]];
    let nested = vec![vec![1], vec![2, 4], vec![3, 4], vec![2], vec![]];
    let reconvergent = vec![
        vec![1],
        vec![2, 6],
        vec![3, 4],
        vec![5],
        vec![5],
        vec![1],
        vec![],
    ];
    let cases = [
        ("varying backedge", &direct, 1, SourceKind::LaneQuery, false),
        ("uniform backedge", &direct, 1, SourceKind::BlockQuery, true),
        (
            "varying entry to inner cycle",
            &nested,
            1,
            SourceKind::LaneQuery,
            false,
        ),
        (
            "uniform entry to inner cycle",
            &nested,
            1,
            SourceKind::BlockQuery,
            true,
        ),
        (
            "reconvergent inner choice",
            &reconvergent,
            2,
            SourceKind::LaneQuery,
            true,
        ),
    ];
    let mut mismatches = Vec::new();
    for (name, edges, selected, kind, accepted) in cases {
        let ctx = &mut Context::new();
        let (module, f, entry) = function(ctx);
        let bit = IntegerType::get(ctx, 1, Signedness::Signless).into();
        let mut conditions = Vec::new();
        for source_kind in [kind, SourceKind::BlockQuery] {
            let condition = value(ctx, entry, bit);
            MirLibrarySourceOp::build(
                ctx,
                LibrarySourceEffectAttr {
                    site: site(0),
                    kind: source_kind,
                },
                Some(condition),
            )
            .insert_at_back(entry, ctx);
            conditions.push(test_slot(ctx, entry, condition));
        }
        let control = source_value(ctx, entry, SourceKind::BlockQuery, false);
        let mut blocks = vec![entry];
        for _ in 1..edges.len() {
            let block = BasicBlock::new(ctx, None, vec![]);
            block.insert_at_back(f.deref(ctx).get_region(0), ctx);
            blocks.push(block);
        }
        let last = edges.len() - 1;
        let mut roles = BTreeMap::new();
        for (index, targets) in edges.iter().enumerate() {
            let block = blocks[index];
            if targets.is_empty() {
                roles.insert(site(last), native(ctx, block, control, last));
                emit::<MirReturnOp>(ctx, block, vec![], vec![]);
            } else if targets.len() == 1 {
                Operation::new(
                    ctx,
                    MirGotoOp::get_concrete_op_info(),
                    vec![],
                    vec![],
                    vec![blocks[targets[0]]],
                    0,
                )
                .insert_at_back(block, ctx);
            } else {
                let address = conditions[usize::from(index != selected)];
                let condition = emit::<MirLoadOp>(ctx, block, vec![bit], vec![address])
                    .deref(ctx)
                    .get_result(0);
                let branch = Operation::new(
                    ctx,
                    MirCondBranchOp::get_concrete_op_info(),
                    vec![],
                    vec![condition],
                    targets.iter().map(|&i| blocks[i]).collect(),
                    0,
                );
                branch.deref_mut(ctx).attributes.set(
                    ATTR_KEY_OPERAND_SEGMENT_SIZES.clone(),
                    OperandSegmentSizesAttr(vec![1, 0, 0]),
                );
                branch.insert_at_back(block, ctx);
            }
        }
        let verdicts = slot_and_promoted_verdicts(ctx, module, &roles);
        println!("{name}: {verdicts:?}");
        for verdict in verdicts {
            if verdict.is_ok() != accepted {
                mismatches.push(format!("{name}: {verdict:?}"));
            }
            if let Err(error) = verdict {
                assert!(
                    error.contains("lane-varying cycle cannot prove collective participation"),
                    "{name}: {error}"
                );
            }
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}
