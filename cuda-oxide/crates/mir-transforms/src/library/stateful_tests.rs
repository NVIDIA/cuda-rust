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
        None,
    )
}

#[test]
#[allow(clippy::disallowed_methods)]
fn scalar_callbacks_remain_raw_and_cannot_join_checked_roots() {
    for checked_root in [false, true] {
        let ctx = &mut Context::new();
        let (module, _, block) = function(ctx);
        let scalar: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signed).into();
        let pointer =
            MirPtrType::get_generic_with_kind(ctx, scalar, true, MirPointerKind::RawMut).into();
        let tuple = MirTupleType::get(ctx, vec![pointer, pointer]).into();
        let environment =
            MirPtrType::get_generic_with_kind(ctx, scalar, false, MirPointerKind::SharedRef).into();
        let operands = vec![value(ctx, block, tuple), value(ctx, block, environment)];
        let signature = FunctionType::get(ctx, vec![tuple, environment], vec![]);
        let role = Role::Native(Native::Callback {
            spec: NativeSpec {
                scalar: LibraryScalar::Signed(32),
                items: 1,
            },
            block_threads: 64,
        });
        MirLibraryOp::build_call(
            ctx,
            role,
            site(1),
            "callback_marker".into(),
            signature,
            operands,
        )
        .insert_at_back(block, ctx);
        let mut roles = BTreeMap::from([(site(1), role)]);
        if checked_root {
            let control = source_value(ctx, block, SourceKind::BlockQuery, false);
            roles.insert(site(2), native(ctx, block, control, 2));
        }
        emit::<MirReturnOp>(ctx, block, vec![], vec![]);
        module.deref(ctx).verify(ctx).unwrap();
        let result = analyze(ctx, module, &mut AnalysisManager::default(), &roles);
        if checked_root {
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("raw and checked library calls")
            );
        } else {
            assert!(result.unwrap().shapes.is_empty());
        }
    }
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
                None,
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

#[test]
fn membership_selects_only_the_exact_member_edge() {
    let targets = [(1, false), (2, true)];
    let predicate = Fact::Member {
        width: 17,
        inverted: false,
        boolean: false,
    };
    let branches = branch_controls(predicate.clone(), &targets, Control::FULL);
    assert!(!branches[0].1.admits(Domain::Prefix(17)));
    let member = branches[1].1;
    assert!(member.admits(Domain::Prefix(17)));
    assert!(!member.admits(Domain::Prefix(31)));
    assert!(!member.admits(Domain::Complete(Scope::Warp(32))));
    assert!(!Control::FULL.admits(Domain::Prefix(17)));
    let repeated = branch_controls(predicate, &targets, member);
    assert_eq!(repeated, vec![(2, member)]);
    let inverted = Fact::Member {
        width: 17,
        inverted: true,
        boolean: true,
    };
    let branches = branch_controls(inverted, &targets, Control::FULL);
    assert!(branches[0].1.admits(Domain::Prefix(17)));
    assert!(!branches[1].1.admits(Domain::Prefix(17)));
    let divergent = member.narrow(Domain::LANE);
    assert!(!divergent.admits(Domain::Prefix(17)));
}

#[test]
fn member_assignments_do_not_become_full_warp_controls_at_a_join() {
    let prior = Fact::Uniform(Domain::BLOCK);
    let assigned = prior.clone().taint(Domain::Prefix(17));
    let joined = assigned.merge(&prior);
    assert!(joined.uniform_at(Domain::Prefix(17)));
    assert!(!joined.uniform_at(Domain::Complete(Scope::Warp(32))));
    let returned = return_fact(Fact::Fields(vec![joined]));
    assert!(returned.field(0).uniform_at(Domain::Prefix(17)));
    assert!(!returned.field(0).uniform_at(Domain::Prefix(31)));
    let mut borrowed = Fact::Referent(Box::new(returned));
    borrowed.invalidate_borrow();
    assert_eq!(borrowed, Fact::Varying);
}

#[test]
fn prefix_completion_does_not_cover_global_or_sibling_pool_effects() {
    let p17 = Region::Implicit(Domain::Prefix(17));
    let p31 = Region::Implicit(Domain::Prefix(31));
    let lane_pool = Region::Implicit(Domain::Complete(Scope::LANE));
    let warp_write = ("buffer".into(), Scope::Warp(32), Access::Write);
    let block_read = ("buffer".into(), Scope::Block, Access::Read);
    let mut state = State {
        values: FxHashMap::default(),
        slots: vec![],
        borrowed: BTreeSet::new(),
        pending: BTreeMap::from([
            (warp_write.clone(), Scope::LANE),
            (block_read.clone(), Scope::LANE),
        ]),
        epochs: Epochs {
            dirty: BTreeSet::from([p17.clone(), p31.clone(), lane_pool.clone()]),
            ..Epochs::default()
        },
    };
    complete(&mut state, Domain::Prefix(17));
    assert!(!state.epochs.dirty.contains(&p17));
    assert!(state.epochs.dirty.contains(&p31));
    assert!(state.epochs.dirty.contains(&lane_pool));
    assert_eq!(state.pending[&warp_write], Scope::LANE);
    assert_eq!(state.pending[&block_read], Scope::LANE);
    complete(&mut state, Domain::Complete(Scope::Warp(32)));
    assert!(state.epochs.dirty.is_empty());
    assert_eq!(state.pending[&warp_write], Scope::Warp(32));
    assert_eq!(state.pending[&block_read], Scope::LANE);
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
    let roots = BTreeMap::from([(
        "root".into(),
        BTreeSet::from(["root".into(), "helper".into()]),
    )]);
    let symbols = BTreeMap::from([
        ("root".into(), "root".into()),
        ("helper".into(), "helper".into()),
    ]);
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
            let first = analyze_libraries(
                module,
                ctx,
                manager,
                &roots,
                &symbols,
                roles,
                &BTreeMap::from([("root".into(), [64, 1, 1])]),
                None,
            )
            .map_err(|error| error.to_string())?;
            // Warp roots now use the same layout-aware proof, even for zero-byte
            // providers. Such a plan must not invent reuse-barrier credit.
            let natives = BTreeMap::from([(
                "root".into(),
                roles
                    .iter()
                    .filter(|(_, role)| matches!(role, Role::Native(_)))
                    .map(|(site, _)| {
                        (
                            site.clone(),
                            dialect_mir::library::LibraryCallPlan::internal(
                                dialect_mir::library::LibraryOperation::Native,
                                "marker".into(),
                            ),
                        )
                    })
                    .collect(),
            )]);
            let mut plans = super::super::storage::allocate_library_storage(&first, &natives)?;
            if plans.is_empty() {
                return Ok(());
            }
            analyze_libraries(
                module,
                ctx,
                manager,
                &roots,
                &symbols,
                roles,
                &first.shapes,
                Some(&mut plans),
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

#[allow(clippy::disallowed_methods)]
fn resource_function(ctx: &mut Context) -> (Ptr<Operation>, Ptr<BasicBlock>, Value) {
    use dialect_mir::types::MirDisjointSliceType;
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
    (module.get_operation(), block, witness)
}

#[allow(clippy::disallowed_methods)]
fn scoped_native(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    witness: Value,
    control: Value,
    scope: Domain,
    index: usize,
) -> Role {
    use dialect_mir::library::LibraryWitness;
    let scalar = IntegerType::get(ctx, 32, Signedness::Signed).into();
    let pointer =
        MirPtrType::get_generic_with_kind(ctx, scalar, false, MirPointerKind::RawConst).into();
    let memory = value(ctx, block, pointer);
    MirLibrarySourceOp::build(
        ctx,
        LibrarySourceEffectAttr {
            site: site(index),
            kind: SourceKind::BlockQuery,
        },
        Some(memory),
    )
    .insert_at_back(block, ctx);
    let uniform_ty = MirTupleType::get(ctx, vec![pointer, control.get_type(ctx)]).into();
    let uniform = emit::<MirConstructTupleOp>(ctx, block, vec![uniform_ty], vec![memory, control])
        .deref(ctx)
        .get_result(0);
    let output =
        MirPtrType::get_generic_with_kind(ctx, scalar, true, MirPointerKind::RawMut).into();
    let payload_ty = MirTupleType::get(ctx, vec![output]).into();
    let payload = value(ctx, block, payload_ty);
    let signature = FunctionType::get(
        ctx,
        vec![witness.get_type(ctx), uniform_ty, payload_ty],
        vec![],
    );
    let spec = NativeSpec {
        scalar: LibraryScalar::Signed(32),
        items: 1,
    };
    let kind = LibraryWitness::CollectiveCarrier(1);
    let role = Role::Native(if let Domain::Complete(Scope::Warp(width)) = scope {
        Native::CheckedWarp {
            spec,
            width,
            access: LibraryAccess::Read,
            witness: kind,
        }
    } else {
        Native::CheckedBlockResource {
            spec,
            access: LibraryAccess::Read,
            witness: kind,
        }
    });
    MirLibraryOp::build_call(
        ctx,
        role,
        site(index),
        "scoped_marker".into(),
        signature,
        vec![witness, uniform, payload],
    )
    .insert_at_back(block, ctx);
    role
}

#[test]
fn genuine_queries_keep_their_exact_scope_through_slots_and_arithmetic() {
    for width in [1, 2, 4, 8, 16, 32] {
        let ctx = &mut Context::new();
        let (module, block, witness) = resource_function(ctx);
        let query = MirLibraryOp::build_query(ctx, site(10), "query".into(), width);
        query.insert_at_back(block, ctx);
        let result = query.deref(ctx).get_result(0);
        let address = test_slot(ctx, block, result);
        let result = emit::<MirLoadOp>(ctx, block, vec![result.get_type(ctx)], vec![address])
            .deref(ctx)
            .get_result(0);
        let offset = test_constant(ctx, block, 32, 1);
        let result = emit::<MirAddOp>(ctx, block, vec![result.get_type(ctx)], vec![result, offset])
            .deref(ctx)
            .get_result(0);
        let role = scoped_native(
            ctx,
            block,
            witness,
            result,
            Domain::Complete(Scope::Warp(32)),
            11,
        );
        emit::<MirReturnOp>(ctx, block, vec![], vec![]);
        let roles = BTreeMap::from([(site(10), Role::WarpGroupIndex { width }), (site(11), role)]);
        let verdicts = slot_and_promoted_verdicts(ctx, module, &roles);
        assert_eq!(
            verdicts.iter().map(Result::is_ok).collect::<Vec<_>>(),
            vec![width == 32; 2],
            "width {width}: {verdicts:?}"
        );
        if width == 32 {
            let proof = analyze(ctx, module, &mut AnalysisManager::default(), &roles).unwrap();
            assert_eq!(proof.traces["root"].events.len(), 1);
            assert!(proof.traces["root"].events.contains_key(&vec![11]));
        } else {
            assert!(
                verdicts
                    .iter()
                    .all(|v| v.as_ref().unwrap_err().contains("not proved Warp-uniform"))
            );
        }
    }
}

#[test]
fn terminal_tile_completion_still_requires_its_participating_control() {
    use pliron::builtin::{
        attributes::OperandSegmentSizesAttr, op_interfaces::ATTR_KEY_OPERAND_SEGMENT_SIZES,
    };
    for width in [8, 16] {
        let ctx = &mut Context::new();
        let (module, entry, witness) = resource_function(ctx);
        let region = entry.deref(ctx).get_parent_region().unwrap();
        let yes = BasicBlock::new(ctx, None, vec![]);
        let no = BasicBlock::new(ctx, None, vec![]);
        yes.insert_at_back(region, ctx);
        no.insert_at_back(region, ctx);
        let control = source_value(ctx, entry, SourceKind::BlockQuery, false);
        let native = scoped_native(
            ctx,
            entry,
            witness,
            control,
            Domain::Complete(Scope::Warp(8)),
            1,
        );
        let query = MirLibraryOp::build_query(ctx, site(2), "query".into(), 8);
        query.insert_at_back(entry, ctx);
        let result = query.deref(ctx).get_result(0);
        let zero = test_constant(ctx, entry, 32, 0);
        let bit = IntegerType::get(ctx, 1, Signedness::Signless).into();
        let condition = emit::<MirNeOp>(ctx, entry, vec![bit], vec![result, zero])
            .deref(ctx)
            .get_result(0);
        let address = test_slot(ctx, entry, condition);
        let condition = emit::<MirLoadOp>(ctx, entry, vec![bit], vec![address])
            .deref(ctx)
            .get_result(0);
        let branch = Operation::new(
            ctx,
            MirCondBranchOp::get_concrete_op_info(),
            vec![],
            vec![condition],
            vec![yes, no],
            0,
        );
        branch.deref_mut(ctx).attributes.set(
            ATTR_KEY_OPERAND_SEGMENT_SIZES.clone(),
            OperandSegmentSizesAttr(vec![1, 0, 0]),
        );
        branch.insert_at_back(entry, ctx);
        let sync = Role::SyncWarp { width };
        MirLibraryOp::build(ctx, sync, site(3), "sync".into()).insert_at_back(yes, ctx);
        emit::<MirReturnOp>(ctx, yes, vec![], vec![]);
        emit::<MirReturnOp>(ctx, no, vec![], vec![]);
        let roles = BTreeMap::from([
            (site(1), native),
            (site(2), Role::WarpGroupIndex { width: 8 }),
            (site(3), sync),
        ]);
        let verdicts = slot_and_promoted_verdicts(ctx, module, &roles);
        assert_eq!(
            verdicts.iter().map(Result::is_ok).collect::<Vec<_>>(),
            vec![width == 8; 2]
        );
        if width == 16 {
            assert!(verdicts.iter().all(|v| {
                v.as_ref()
                    .unwrap_err()
                    .contains("WarpTile barrier is reached through control narrower than its width")
            }));
        }
    }
}

#[test]
fn helper_cycles_reconverge_without_widening_returned_values() {
    use pliron::builtin::{
        attributes::OperandSegmentSizesAttr, op_interfaces::ATTR_KEY_OPERAND_SEGMENT_SIZES,
    };
    for (width, consumer, carry_result, accepted) in [
        (32, Domain::Complete(Scope::Warp(32)), false, true),
        (32, Domain::BLOCK, false, true),
        (16, Domain::Complete(Scope::Warp(32)), false, true),
        (1, Domain::Complete(Scope::Warp(32)), false, true),
        (16, Domain::Complete(Scope::Warp(32)), true, false),
        (1, Domain::BLOCK, true, false),
    ] {
        let ctx = &mut Context::new();
        let (module, root, witness) = resource_function(ctx);
        let scalar = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let helper = test_helper(ctx, module, vec![], vec![scalar]);
        let region = helper.deref(ctx).get_parent_region().unwrap();
        let header = BasicBlock::new(ctx, None, vec![]);
        header.insert_at_back(region, ctx);
        let body = BasicBlock::new(ctx, None, vec![]);
        let exit = BasicBlock::new(ctx, None, vec![]);
        body.insert_at_back(region, ctx);
        exit.insert_at_back(region, ctx);
        let query_site = LibraryCallSite {
            caller: "helper".into(),
            block: 0,
        };
        let query = MirLibraryOp::build_query(ctx, query_site.clone(), "query".into(), width);
        query.insert_at_back(helper, ctx);
        let result = query.deref(ctx).get_result(0);
        let zero = test_constant(ctx, helper, 32, 0);
        let bit = IntegerType::get(ctx, 1, Signedness::Signless).into();
        let condition = emit::<MirNeOp>(ctx, helper, vec![bit], vec![result, zero])
            .deref(ctx)
            .get_result(0);
        let address = test_slot(ctx, helper, condition);
        Operation::new(
            ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![header],
            0,
        )
        .insert_at_back(helper, ctx);
        let condition = emit::<MirLoadOp>(ctx, header, vec![bit], vec![address])
            .deref(ctx)
            .get_result(0);
        let branch = Operation::new(
            ctx,
            MirCondBranchOp::get_concrete_op_info(),
            vec![],
            vec![condition],
            vec![body, exit],
            0,
        );
        branch.deref_mut(ctx).attributes.set(
            ATTR_KEY_OPERAND_SEGMENT_SIZES.clone(),
            OperandSegmentSizesAttr(vec![1, 0, 0]),
        );
        branch.insert_at_back(header, ctx);
        Operation::new(
            ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![header],
            0,
        )
        .insert_at_back(body, ctx);
        emit::<MirReturnOp>(ctx, exit, vec![], vec![result]);
        let call = test_call(ctx, root, vec![], vec![scalar]);
        // Structural participation reconverges, but the query returned by the
        // helper keeps its original precision when used as a later control.
        let control = if carry_result {
            call.deref(ctx).get_result(0)
        } else {
            source_value(ctx, root, SourceKind::BlockQuery, false)
        };
        let role = scoped_native(ctx, root, witness, control, consumer, 1);
        emit::<MirReturnOp>(ctx, root, vec![], vec![]);
        let roles = BTreeMap::from([
            (query_site, Role::WarpGroupIndex { width }),
            (site(1), role),
        ]);
        let verdicts = slot_and_promoted_verdicts(ctx, module, &roles);
        assert_eq!(
            verdicts.iter().map(Result::is_ok).collect::<Vec<_>>(),
            vec![accepted; 2],
            "width {width}, {consumer:?}: {verdicts:?}"
        );
        if !accepted {
            assert!(
                verdicts
                    .iter()
                    .all(|v| { v.as_ref().unwrap_err().contains("not proved") })
            );
        }
    }
}

#[test]
#[allow(clippy::disallowed_methods)]
fn resource_access_requires_completion_and_retains_original_witness_origin() {
    use dialect_mir::library::LibraryWitness;
    for (scopes, accesses, completion, accepted) in [
        (
            [Domain::BLOCK, Domain::BLOCK],
            [LibraryAccess::Read, LibraryAccess::Write],
            None,
            false,
        ),
        (
            [Domain::BLOCK, Domain::BLOCK],
            [LibraryAccess::Read, LibraryAccess::Write],
            Some(Domain::BLOCK),
            true,
        ),
        (
            [
                Domain::Complete(Scope::Warp(32)),
                Domain::Complete(Scope::Warp(32)),
            ],
            [LibraryAccess::Read, LibraryAccess::Read],
            None,
            true,
        ),
        (
            [Domain::Complete(Scope::Warp(32)), Domain::BLOCK],
            [LibraryAccess::Write, LibraryAccess::Read],
            None,
            false,
        ),
        (
            [Domain::BLOCK, Domain::Complete(Scope::Warp(32))],
            [LibraryAccess::Read, LibraryAccess::Write],
            None,
            false,
        ),
        (
            [Domain::Complete(Scope::Warp(32)), Domain::BLOCK],
            [LibraryAccess::Write, LibraryAccess::Read],
            Some(Domain::BLOCK),
            true,
        ),
        (
            [
                Domain::Complete(Scope::Warp(8)),
                Domain::Complete(Scope::Warp(8)),
            ],
            [LibraryAccess::Write, LibraryAccess::Read],
            Some(Domain::Complete(Scope::Warp(8))),
            true,
        ),
        (
            [
                Domain::Complete(Scope::Warp(8)),
                Domain::Complete(Scope::Warp(16)),
            ],
            [LibraryAccess::Write, LibraryAccess::Read],
            Some(Domain::Complete(Scope::Warp(8))),
            false,
        ),
        (
            [
                Domain::Complete(Scope::Warp(8)),
                Domain::Complete(Scope::Warp(16)),
            ],
            [LibraryAccess::Write, LibraryAccess::Read],
            Some(Domain::Complete(Scope::Warp(16))),
            true,
        ),
        (
            [
                Domain::Complete(Scope::Warp(16)),
                Domain::Complete(Scope::Warp(8)),
            ],
            [LibraryAccess::Write, LibraryAccess::Read],
            Some(Domain::Complete(Scope::Warp(8))),
            false,
        ),
        (
            [Domain::Complete(Scope::Warp(32)), Domain::BLOCK],
            [LibraryAccess::Write, LibraryAccess::Read],
            Some(Domain::Complete(Scope::Warp(32))),
            false,
        ),
    ] {
        let ctx = &mut Context::new();
        let (module, block, witness) = resource_function(ctx);
        let scalar: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signed).into();
        let witness_ty = witness.get_type(ctx);
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
        for (position, (scope, access)) in scopes.into_iter().zip(accesses).enumerate() {
            let index = position + 1;
            if index == 2
                && let Some(scope) = completion
            {
                if let Domain::Complete(Scope::Warp(width)) = scope {
                    let role = Role::SyncWarp { width };
                    MirLibraryOp::build(ctx, role, site(3), "sync".into())
                        .insert_at_back(block, ctx);
                    roles.insert(site(3), role);
                } else {
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
            }
            let spec = NativeSpec {
                scalar: LibraryScalar::Signed(32),
                items: 1,
            };
            let witness_kind = LibraryWitness::CollectiveCarrier(1);
            let role = Role::Native(if let Domain::Complete(Scope::Warp(width)) = scope {
                Native::CheckedWarp {
                    spec,
                    width,
                    access,
                    witness: witness_kind,
                }
            } else {
                Native::CheckedBlockResource {
                    spec,
                    access,
                    witness: witness_kind,
                }
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
        let verdicts = slot_and_promoted_verdicts(ctx, module, &roles);
        assert_eq!(
            verdicts.iter().map(Result::is_ok).collect::<Vec<_>>(),
            vec![accepted; 2],
            "{verdicts:?}"
        );
        if !accepted {
            assert!(verdicts.iter().all(|v| {
                v.as_ref()
                    .unwrap_err()
                    .contains("requires a barrier covering both accesses")
            }));
        }
    }
}

#[test]
fn cycles_use_structural_control_after_reconvergence() {
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
        ("varying backedge", &direct, 1, SourceKind::LaneQuery, true),
        ("uniform backedge", &direct, 1, SourceKind::BlockQuery, true),
        (
            "varying entry to inner cycle",
            &nested,
            1,
            SourceKind::LaneQuery,
            true,
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
                    error.contains("operation is reached through lane-varying control flow"),
                    "{name}: {error}"
                );
            }
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

#[test]
#[allow(clippy::disallowed_methods)]
fn planned_storage_reuses_the_same_proof_for_completion_and_global_resource_conflicts() {
    use dialect_mir::{library::LibraryWitness, types::MirDisjointSliceType};
    for (policy, completion, bytes, expected) in [
        (None, None, 16, None),
        (
            None,
            None,
            0,
            Some("conflicting collective resource access"),
        ),
        (
            Some(Policy::default()),
            None,
            16,
            Some("explicit storage reuse"),
        ),
        (Some(Policy::default()), Some(Domain::BLOCK), 16, None),
        (
            Some(Policy {
                auto_sync: true,
                ..Default::default()
            }),
            None,
            16,
            None,
        ),
        (
            Some(Policy::default()),
            Some(Domain::Complete(Scope::Warp(32))),
            16,
            Some("explicit storage reuse"),
        ),
        (
            None,
            Some(Domain::Complete(Scope::Warp(32))),
            0,
            Some("conflicting collective resource access"),
        ),
    ] {
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
        let mut roles = BTreeMap::new();
        let (storage, storage_arg) = if let Some(policy) = policy {
            let byte = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
            let descriptor = dialect_mir::types::MirStructType::get(
                ctx,
                "descriptor".into(),
                vec!["private".into()],
                vec![byte],
            )
            .into();
            let signature = FunctionType::get(ctx, vec![], vec![descriptor]);
            let role = Role::StorageNew(policy);
            let new =
                MirLibraryOp::build_call(ctx, role, site(10), "new".into(), signature, vec![]);
            new.insert_at_back(block, ctx);
            roles.insert(site(10), role);
            let original = new.deref(ctx).get_result(0);
            let address = test_slot(ctx, block, original);
            let loaded = emit::<MirLoadOp>(ctx, block, vec![descriptor], vec![address])
                .deref(ctx)
                .get_result(0);
            let borrow =
                MirPtrType::get_generic_with_kind(ctx, descriptor, true, MirPointerKind::UniqueRef)
                    .into();
            let reference = emit::<MirRefOp>(ctx, block, vec![borrow], vec![loaded]);
            MirRefOp::new(reference).set_mutable(ctx, true);
            MirRefOp::new(reference).set_pointer_kind_authority(
                ctx,
                dialect_mir::attributes::MirPointerKindAuthorityAttr::Reborrow,
            );
            (
                reference.deref(ctx).get_result(0),
                LibraryStorageArg::Explicit(policy),
            )
        } else {
            let empty = MirTupleType::get(ctx, vec![]).into();
            (value(ctx, block, empty), LibraryStorageArg::Implicit)
        };
        let call_ty = FunctionType::get(
            ctx,
            vec![witness_ty, storage.get_type(ctx), tuple, tuple],
            vec![],
        );
        for (index, access) in [(1, LibraryAccess::Read), (2, LibraryAccess::Write)] {
            // A real payload role with zero native scratch cannot complete the
            // pending global resource access, even in a planned-storage root.
            if index == 2 {
                roles.insert(site(11), local_payload(ctx, block, vec![], None, None, 11));
            }
            if index == 2
                && let Some(scope) = completion
            {
                if let Domain::Complete(Scope::Warp(width)) = scope {
                    let role = Role::SyncWarp { width };
                    MirLibraryOp::build(ctx, role, site(3), "sync".into())
                        .insert_at_back(block, ctx);
                    roles.insert(site(3), role);
                } else {
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
            }
            let role = Role::Native(Native::CheckedBlockStorage {
                spec: NativeSpec {
                    scalar: LibraryScalar::Signed(32),
                    items: 1,
                },
                access,
                witness: LibraryWitness::CollectiveCarrier(1),
                storage: storage_arg,
            });
            MirLibraryOp::build_call(
                ctx,
                role,
                site(index),
                "resource_marker".into(),
                call_ty,
                vec![witness, storage, uniform, payload],
            )
            .insert_at_back(block, ctx);
            roles.insert(site(index), role);
        }
        emit::<MirReturnOp>(ctx, block, vec![], vec![]);
        let module = module.get_operation();
        module.deref(ctx).verify(ctx).unwrap();
        let manager = &mut AnalysisManager::default();
        for promoted in [false, true] {
            if promoted {
                pliron::opts::mem2reg::mem2reg(module, ctx, manager).unwrap();
                let mut changed = pliron::pass::PassResult::default();
                changed.ir_changed = pliron::irbuild::IRStatus::Changed;
                manager.retain_preserved(&changed);
                module.deref(ctx).verify(ctx).unwrap();
            }
            let first = analyze(ctx, module, manager, &roles).unwrap();
            let natives = super::super::storage::RootedNativePlans::from([(
                "root".into(),
                [1, 2]
                    .into_iter()
                    .map(|index| {
                        let mut plan = dialect_mir::library::LibraryCallPlan::internal(
                            dialect_mir::library::LibraryOperation::Native,
                            "resource_marker".into(),
                        );
                        plan.scratch_bytes = bytes;
                        plan.scratch_alignment = 4;
                        (site(index), plan)
                    })
                    .collect(),
            )]);
            let mut natives = natives;
            natives.get_mut("root").unwrap().insert(
                site(11),
                dialect_mir::library::LibraryCallPlan::internal(
                    dialect_mir::library::LibraryOperation::Native,
                    "payload_marker".into(),
                ),
            );
            let mut plans =
                super::super::storage::allocate_library_storage(&first, &natives).unwrap();
            assert!(manager.try_get_analysis::<PreparedGraph>(module).is_some());
            let second = analyze_libraries(
                module,
                ctx,
                manager,
                &BTreeMap::from([("root".into(), BTreeSet::from(["root".into()]))]),
                &BTreeMap::from([("root".into(), "root".into())]),
                &roles,
                &first.shapes,
                Some(&mut plans),
            );
            if let Some(expected) = expected {
                assert!(second.unwrap_err().to_string().contains(expected));
            } else {
                second.unwrap();
                assert_eq!(plans["root"].bytes, bytes);
                assert_eq!(plans["root"].sites[&vec![2]].before, policy.is_none());
                assert_eq!(
                    plans["root"].sites[&vec![1]].after,
                    policy.is_some_and(|p| p.auto_sync)
                );
            }
        }
    }
}

#[test]
#[allow(clippy::disallowed_methods)]
fn reservations_require_release_and_completion_or_disjoint_exclusive_placement() {
    use dialect_mir::{library::LibraryReserveLayout, types::MirStructType};
    // A barrier does not release a token, and releasing a token does not complete
    // its reads. Exclusive sites instead receive distinct aligned regions.
    for (release, completed, exclusive, capacity, expected) in [
        (false, false, false, 0, Some("overlaps a live reservation")),
        (false, true, false, 0, Some("overlaps a live reservation")),
        (true, false, false, 0, Some("explicit storage reuse")),
        (true, true, false, 0, None),
        (false, false, true, 0, None),
        (false, false, true, 19, Some("including exclusive padding")),
    ] {
        let ctx = &mut Context::new();
        let (module, _, block) = function(ctx);
        let byte = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        let scalar = IntegerType::get(ctx, 32, Signedness::Signed).into();
        let descriptor =
            MirStructType::get(ctx, "descriptor".into(), vec!["private".into()], vec![byte]).into();
        let pointer =
            MirPtrType::get_generic_with_kind(ctx, scalar, true, MirPointerKind::RawMut).into();
        let token_type =
            MirStructType::get(ctx, "token".into(), vec!["private".into()], vec![pointer]).into();
        let policy = Policy {
            exclusive,
            capacity,
            ..Default::default()
        };
        let new_role = Role::StorageNew(policy);
        let signature = FunctionType::get(ctx, vec![], vec![descriptor]);
        let new =
            MirLibraryOp::build_call(ctx, new_role, site(10), "new".into(), signature, vec![]);
        new.insert_at_back(block, ctx);
        let initial = new.deref(ctx).get_result(0);
        let address = test_slot(ctx, block, initial);
        let loaded = emit::<MirLoadOp>(ctx, block, vec![descriptor], vec![address])
            .deref(ctx)
            .get_result(0);
        let borrow_type =
            MirPtrType::get_generic_with_kind(ctx, descriptor, true, MirPointerKind::UniqueRef)
                .into();
        let borrow = emit::<MirRefOp>(ctx, block, vec![borrow_type], vec![loaded]);
        MirRefOp::new(borrow).set_mutable(ctx, true);
        MirRefOp::new(borrow).set_pointer_kind_authority(
            ctx,
            dialect_mir::attributes::MirPointerKindAuthorityAttr::Reborrow,
        );
        let borrow = borrow.deref(ctx).get_result(0);
        let mut roles = BTreeMap::from([(site(10), new_role)]);
        for index in [1, 2] {
            let role = Role::StorageReserve {
                policy,
                layout: LibraryReserveLayout {
                    scalar: LibraryScalar::Signed(32),
                    count: 1,
                    bytes: 4,
                    alignment: if index == 1 { 4 } else { 16 },
                },
            };
            let signature = FunctionType::get(ctx, vec![borrow_type], vec![token_type]);
            let reserve = MirLibraryOp::build_call(
                ctx,
                role,
                site(index),
                "reserve".into(),
                signature,
                vec![borrow],
            );
            reserve.insert_at_back(block, ctx);
            roles.insert(site(index), role);
            if index == 1 {
                if release {
                    let token = reserve.deref(ctx).get_result(0);
                    let address = test_slot(ctx, block, token);
                    let token = emit::<MirLoadOp>(ctx, block, vec![token_type], vec![address])
                        .deref(ctx)
                        .get_result(0);
                    let signature = FunctionType::get(ctx, vec![token_type], vec![]);
                    MirLibraryOp::build_call(
                        ctx,
                        Role::StorageRelease,
                        site(3),
                        "release".into(),
                        signature,
                        vec![token],
                    )
                    .insert_at_back(block, ctx);
                    roles.insert(site(3), Role::StorageRelease);
                }
                if completed {
                    MirLibrarySourceOp::build(
                        ctx,
                        LibrarySourceEffectAttr {
                            site: site(4),
                            kind: SourceKind::BlockSync,
                        },
                        None,
                    )
                    .insert_at_back(block, ctx);
                }
            }
        }
        emit::<MirReturnOp>(ctx, block, vec![], vec![]);
        module.deref(ctx).verify(ctx).unwrap();
        let manager = &mut AnalysisManager::default();
        for promoted in [false, true] {
            if promoted {
                pliron::opts::mem2reg::mem2reg(module, ctx, manager).unwrap();
                let mut changed = pliron::pass::PassResult::default();
                changed.ir_changed = pliron::irbuild::IRStatus::Changed;
                manager.retain_preserved(&changed);
                module.deref(ctx).verify(ctx).unwrap();
            }
            let first = analyze(ctx, module, manager, &roles).unwrap();
            assert_eq!(first.traces["root"].events[&vec![2]].origin, Some(vec![10]));
            let plans = super::super::storage::allocate_library_storage(&first, &BTreeMap::new());
            if capacity != 0 {
                assert!(plans.err().unwrap().contains(expected.unwrap()));
                continue;
            }
            let mut plans = plans.unwrap();
            assert_eq!(plans["root"].bytes, if exclusive { 20 } else { 4 });
            assert_eq!(
                plans["root"].sites[&vec![2]].offset,
                if exclusive { 16 } else { 0 }
            );
            let second = analyze_libraries(
                module,
                ctx,
                manager,
                &BTreeMap::from([("root".into(), BTreeSet::from(["root".into()]))]),
                &BTreeMap::from([("root".into(), "root".into())]),
                &roles,
                &first.shapes,
                Some(&mut plans),
            );
            if let Some(expected) = expected {
                assert!(second.unwrap_err().to_string().contains(expected));
            } else {
                second.unwrap();
            }
        }
    }
}

#[test]
fn warp_storage_reuse_completes_only_covered_accesses_in_both_preparations() {
    for (widths, bytes, accepted) in [
        (vec![8, 8], 148, true),
        (vec![8, 8], 0, false),
        (vec![8, 16], 148, false),
        (vec![8, 16, 8], 148, false),
    ] {
        let ctx = &mut Context::new();
        let (module, block, witness) = resource_function(ctx);
        let control = source_value(ctx, block, SourceKind::BlockQuery, true);
        let mut roles = BTreeMap::new();
        for (position, &width) in widths.iter().enumerate() {
            let index = position + 1;
            let mut role = scoped_native(
                ctx,
                block,
                witness,
                control,
                Domain::Complete(Scope::Warp(width)),
                index,
            );
            if position + 1 == widths.len() {
                let Role::Native(Native::CheckedWarp { access, .. }) = &mut role else {
                    unreachable!()
                };
                *access = LibraryAccess::Write;
                let op = block.deref(ctx).iter(ctx).last().unwrap();
                MirLibraryOp::new(op).set_attr_role(ctx, role);
            }
            roles.insert(site(index), role);
        }
        emit::<MirReturnOp>(ctx, block, vec![], vec![]);
        let manager = &mut AnalysisManager::default();
        for promoted in [false, true] {
            if promoted {
                pliron::opts::mem2reg::mem2reg(module, ctx, manager).unwrap();
                let mut changed = pliron::pass::PassResult::default();
                changed.ir_changed = pliron::irbuild::IRStatus::Changed;
                manager.retain_preserved(&changed);
            }
            module.deref(ctx).verify(ctx).unwrap();
            let first = analyze(ctx, module, manager, &roles).unwrap();
            let natives = BTreeMap::from([(
                "root".into(),
                roles
                    .keys()
                    .map(|site| {
                        let mut plan = dialect_mir::library::LibraryCallPlan::internal(
                            dialect_mir::library::LibraryOperation::Native,
                            "scoped_marker".into(),
                        );
                        plan.scratch_bytes = bytes;
                        plan.scratch_alignment = 4;
                        (site.clone(), plan)
                    })
                    .collect(),
            )]);
            let mut plans =
                super::super::storage::allocate_library_storage(&first, &natives).unwrap();
            assert!(plans.contains_key("root"));
            if bytes == 0 {
                assert_eq!(plans["root"].bytes, 0);
            }
            let result = analyze_libraries(
                module,
                ctx,
                manager,
                &BTreeMap::from([("root".into(), BTreeSet::from(["root".into()]))]),
                &BTreeMap::from([("root".into(), "root".into())]),
                &roles,
                &first.shapes,
                Some(&mut plans),
            );
            assert_eq!(
                result.is_ok(),
                accepted,
                "{widths:?}, bytes={bytes}, promoted={promoted}: {result:?}"
            );
            if accepted {
                assert_eq!(plans["root"].bytes, 1184);
                assert_eq!(plans["root"].sites[&vec![1]].stride, 148);
                assert!(!plans["root"].sites[&vec![1]].before);
                assert!(plans["root"].sites[&vec![2]].before);
                assert!(!plans["root"].sites[&vec![2]].after);
            } else {
                assert!(
                    result
                        .unwrap_err()
                        .to_string()
                        .contains("conflicting collective resource access")
                );
            }
        }
    }
}

#[allow(clippy::disallowed_methods)]
fn local_payload(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    controls: Vec<Value>,
    output: Option<Value>,
    width: Option<u32>,
    index: usize,
) -> Role {
    let scalar: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signed).into();
    let input_ty =
        MirPtrType::get_generic_with_kind(ctx, scalar, false, MirPointerKind::RawConst).into();
    let output_ty =
        MirPtrType::get_generic_with_kind(ctx, scalar, true, MirPointerKind::RawMut).into();
    let input = value(ctx, block, input_ty);
    let output = output.unwrap_or_else(|| value(ctx, block, output_ty));
    let payload_ty = MirTupleType::get(ctx, vec![input_ty, output.get_type(ctx)]).into();
    let payload = emit::<MirConstructTupleOp>(ctx, block, vec![payload_ty], vec![input, output])
        .deref(ctx)
        .get_result(0);
    let controls_ty =
        MirTupleType::get(ctx, controls.iter().map(|v| v.get_type(ctx)).collect()).into();
    let controls = emit::<MirConstructTupleOp>(ctx, block, vec![controls_ty], controls)
        .deref(ctx)
        .get_result(0);
    let spec = NativeSpec {
        scalar: LibraryScalar::Signed(32),
        items: 1,
    };
    let (native, operands) = if let Some(width) = width {
        (
            Native::CheckedWarpPayload { spec, width },
            vec![controls, payload],
        )
    } else {
        let unit = MirTupleType::get(ctx, vec![]).into();
        let storage = value(ctx, block, unit);
        (
            Native::CheckedBlockPayload {
                spec,
                storage: LibraryStorageArg::Implicit,
            },
            vec![storage, controls, payload],
        )
    };
    let signature = FunctionType::get(
        ctx,
        operands.iter().map(|v| v.get_type(ctx)).collect(),
        vec![],
    );
    let role = Role::Native(native);
    MirLibraryOp::build_call(
        ctx,
        role,
        site(index),
        "payload_marker".into(),
        signature,
        operands,
    )
    .insert_at_back(block, ctx);
    role
}

#[test]
fn prefix_membership_requires_the_true_edge_on_every_entry_path() {
    use pliron::builtin::{
        attributes::OperandSegmentSizesAttr, op_interfaces::ATTR_KEY_OPERAND_SEGMENT_SIZES,
    };
    let cases = [
        (
            "guarded",
            vec![vec![1, 3], vec![2], vec![3], vec![]],
            0,
            2,
            true,
        ),
        (
            "backward entry",
            vec![vec![1], vec![2], vec![1, 3], vec![]],
            2,
            1,
            false,
        ),
        (
            "critical edge",
            vec![vec![1, 2], vec![3, 4], vec![3], vec![4], vec![]],
            1,
            3,
            false,
        ),
        (
            "uniform outer loop",
            vec![vec![1], vec![2, 5], vec![3, 4], vec![4], vec![1], vec![]],
            2,
            3,
            true,
        ),
    ];
    for (name, edges, guard, call, accepted) in cases {
        let ctx = &mut Context::new();
        let (module, f, entry) = function(ctx);
        let bit = IntegerType::get(ctx, 1, Signedness::Signless).into();
        let byte = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        let query = MirLibraryOp::build_prefix_query(ctx, site(0), "prefix".into(), 17);
        query.insert_at_back(entry, ctx);
        let member = query.deref(ctx).get_result(0);
        let member = test_slot(ctx, entry, member);
        let uniform = value(ctx, entry, bit);
        MirLibrarySourceOp::build(
            ctx,
            LibrarySourceEffectAttr {
                site: site(1),
                kind: SourceKind::BlockQuery,
            },
            Some(uniform),
        )
        .insert_at_back(entry, ctx);
        let uniform = test_slot(ctx, entry, uniform);
        let mut blocks = vec![entry];
        for _ in 1..edges.len() {
            let block = BasicBlock::new(ctx, None, vec![]);
            block.insert_at_back(f.deref(ctx).get_region(0), ctx);
            blocks.push(block);
        }
        let mut roles = BTreeMap::from([(site(0), Role::WarpPrefixMember { width: 17 })]);
        for (index, targets) in edges.iter().enumerate() {
            let block = blocks[index];
            if index == call {
                roles.insert(
                    site(2),
                    local_payload(ctx, block, vec![], None, Some(17), 2),
                );
            }
            if targets.is_empty() {
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
                let condition = if index == guard {
                    let loaded = emit::<MirLoadOp>(ctx, block, vec![byte], vec![member])
                        .deref(ctx)
                        .get_result(0);
                    let zero = test_constant(ctx, block, 8, 0);
                    emit::<MirNeOp>(ctx, block, vec![bit], vec![loaded, zero])
                        .deref(ctx)
                        .get_result(0)
                } else {
                    emit::<MirLoadOp>(ctx, block, vec![bit], vec![uniform])
                        .deref(ctx)
                        .get_result(0)
                };
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
        assert_eq!(
            verdicts.iter().map(Result::is_ok).collect::<Vec<_>>(),
            vec![accepted; 2],
            "{name}: {verdicts:?}"
        );
    }
}

#[test]
#[allow(clippy::disallowed_methods)]
fn payload_controls_and_native_written_outputs_keep_the_slot_and_promoted_proof() {
    use dialect_mir::attributes::{MirCastKindAttr, MirPointerKindAuthorityAttr};
    for width in [None, Some(8)] {
        for (kind, written, accepted) in [
            (None, false, true),
            (Some(SourceKind::BlockQuery), false, true),
            (Some(SourceKind::LaneQuery), false, false),
            (Some(SourceKind::BlockQuery), true, false),
        ] {
            let ctx = &mut Context::new();
            let (module, _, block) = function(ctx);
            let mut roles = BTreeMap::new();
            let controls = if let Some(kind) = kind {
                let initial = source_value(ctx, block, kind, true);
                let control = if written {
                    let address = test_slot(ctx, block, initial);
                    let raw = MirPtrType::get_generic_with_kind(
                        ctx,
                        initial.get_type(ctx),
                        true,
                        MirPointerKind::RawMut,
                    )
                    .into();
                    let cast = emit::<MirCastOp>(ctx, block, vec![raw], vec![address]);
                    MirCastOp::new(cast).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
                    MirCastOp::new(cast)
                        .set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::RawAddress);
                    let output = cast.deref(ctx).get_result(0);
                    roles.insert(
                        site(1),
                        local_payload(ctx, block, vec![], Some(output), width, 1),
                    );
                    emit::<MirLoadOp>(ctx, block, vec![initial.get_type(ctx)], vec![address])
                        .deref(ctx)
                        .get_result(0)
                } else {
                    initial
                };
                vec![control]
            } else {
                vec![]
            };
            roles.insert(site(2), local_payload(ctx, block, controls, None, width, 2));
            emit::<MirReturnOp>(ctx, block, vec![], vec![]);
            for verdict in slot_and_promoted_verdicts(ctx, module, &roles) {
                assert_eq!(
                    verdict.is_ok(),
                    accepted,
                    "{width:?}/{kind:?}/written={written}: {verdict:?}"
                );
                if let Err(error) = verdict {
                    assert!(error.contains("not proved"), "{error}");
                }
            }
        }
    }
}

#[test]
fn zero_layout_payload_roots_retain_planning_and_reject_legacy_mixing() {
    for width in [None, Some(8)] {
        for legacy in [false, true] {
            let ctx = &mut Context::new();
            let (module, _, block) = function(ctx);
            let role = local_payload(ctx, block, vec![], None, width, 1);
            let mut roles = BTreeMap::from([(site(1), role)]);
            if legacy {
                let control = source_value(ctx, block, SourceKind::BlockQuery, false);
                roles.insert(site(2), native(ctx, block, control, 2));
            }
            emit::<MirReturnOp>(ctx, block, vec![], vec![]);
            for verdict in slot_and_promoted_verdicts(ctx, module, &roles) {
                if legacy {
                    assert!(
                        verdict
                            .unwrap_err()
                            .contains("untracked and planned-storage")
                    );
                } else {
                    verdict.unwrap();
                }
            }
            if !legacy {
                let validation =
                    analyze(ctx, module, &mut AnalysisManager::default(), &roles).unwrap();
                let mut plan = dialect_mir::library::LibraryCallPlan::internal(
                    dialect_mir::library::LibraryOperation::Native,
                    "payload_marker".into(),
                );
                plan.payload_only = true;
                plan.checked_block = true;
                plan.warp_width = width;
                let native = super::super::storage::RootedNativePlans::from([(
                    "root".into(),
                    BTreeMap::from([(site(1), plan)]),
                )]);
                let plans =
                    super::super::storage::allocate_library_storage(&validation, &native).unwrap();
                assert_eq!(plans["root"].bytes, 0);
                assert!(
                    plans["root"].sites.is_empty(),
                    "zero layout adds no completion binding"
                );
            }
        }
    }
}

#[test]
#[allow(clippy::disallowed_methods)]
fn membership_byte_not_is_not_logical_not() {
    use dialect_mir::attributes::MirCastKindAttr;
    use pliron::builtin::{
        attributes::OperandSegmentSizesAttr, op_interfaces::ATTR_KEY_OPERAND_SEGMENT_SIZES,
    };
    for (width, negate) in [(8, false), (1, true), (8, true)] {
        let ctx = &mut Context::new();
        let (module, f, root) = function(ctx);
        let query = MirLibraryOp::build_prefix_query(ctx, site(0), "prefix_member".into(), 17);
        query.insert_at_back(root, ctx);
        let zero = test_constant(ctx, root, 8, 0);
        let bit: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
        let member = query.deref(ctx).get_result(0);
        let predicate = emit::<MirNeOp>(ctx, root, vec![bit], vec![member, zero])
            .deref(ctx)
            .get_result(0);
        let target = IntegerType::get(ctx, width, Signedness::Unsigned).into();
        let cast = emit::<MirCastOp>(ctx, root, vec![target], vec![predicate]);
        MirCastOp::new(cast).set_attr_cast_kind(ctx, MirCastKindAttr::IntToInt);
        let byte = cast.deref(ctx).get_result(0);
        let one = test_constant(ctx, root, width, 1);
        let condition = if negate {
            let inverse = emit::<MirNotOp>(ctx, root, vec![target], vec![byte])
                .deref(ctx)
                .get_result(0);
            emit::<MirNeOp>(ctx, root, vec![bit], vec![inverse, one])
                .deref(ctx)
                .get_result(0)
        } else {
            emit::<MirEqOp>(ctx, root, vec![bit], vec![byte, one])
                .deref(ctx)
                .get_result(0)
        };
        let active = BasicBlock::new(ctx, None, vec![]);
        let exit = BasicBlock::new(ctx, None, vec![]);
        for block in [active, exit] {
            block.insert_at_back(f.deref(ctx).get_region(0), ctx);
        }
        let branch = Operation::new(
            ctx,
            MirCondBranchOp::get_concrete_op_info(),
            vec![],
            vec![condition],
            vec![active, exit],
            0,
        );
        branch.deref_mut(ctx).attributes.set(
            ATTR_KEY_OPERAND_SEGMENT_SIZES.clone(),
            OperandSegmentSizesAttr(vec![1, 0, 0]),
        );
        branch.insert_at_back(root, ctx);
        let role = local_payload(ctx, active, vec![], None, Some(17), 1);
        Operation::new(
            ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![exit],
            0,
        )
        .insert_at_back(active, ctx);
        emit::<MirReturnOp>(ctx, exit, vec![], vec![]);
        let verdicts = slot_and_promoted_verdicts(
            ctx,
            module,
            &BTreeMap::from([
                (site(0), Role::WarpPrefixMember { width: 17 }),
                (site(1), role),
            ]),
        );
        assert_eq!(
            verdicts.iter().map(Result::is_ok).collect::<Vec<_>>(),
            vec![width == 1 || !negate; 2],
            "width={width}, negate={negate}: {verdicts:?}"
        );
        if width == 8 && negate {
            assert!(verdicts.iter().all(|v| {
                v.as_ref()
                    .unwrap_err()
                    .contains("lane-varying control flow")
            }));
        }
    }
}

#[test]
#[allow(clippy::disallowed_methods)]
fn checked_callback_payload_keeps_output_taint_and_planned_root_mixing() {
    use dialect_mir::attributes::{MirCastKindAttr, MirPointerKindAuthorityAttr};
    for (written, legacy, accepted) in [
        (false, false, true),
        (true, false, false),
        (false, true, false),
    ] {
        let ctx = &mut Context::new();
        let (module, _, block) = function(ctx);
        let scalar: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signed).into();
        let byte: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        let unit = MirTupleType::get(ctx, vec![]).into();
        let input_ty =
            MirPtrType::get_generic_with_kind(ctx, scalar, false, MirPointerKind::RawConst).into();
        let output_ty =
            MirPtrType::get_generic_with_kind(ctx, scalar, true, MirPointerKind::RawMut).into();
        let input_bits_ty =
            MirPtrType::get_generic_with_kind(ctx, byte, false, MirPointerKind::RawConst).into();
        let output_bits_ty =
            MirPtrType::get_generic_with_kind(ctx, byte, true, MirPointerKind::RawMut).into();
        let initial = source_value(ctx, block, SourceKind::BlockQuery, true);
        let slot = test_slot(ctx, block, initial);
        let output = if written {
            let cast = emit::<MirCastOp>(ctx, block, vec![output_ty], vec![slot]);
            MirCastOp::new(cast).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
            MirCastOp::new(cast)
                .set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::RawAddress);
            cast.deref(ctx).get_result(0)
        } else {
            value(ctx, block, output_ty)
        };
        let input = value(ctx, block, input_ty);
        let input_bits = value(ctx, block, input_bits_ty);
        let output_bits = value(ctx, block, output_bits_ty);
        let tuple = MirTupleType::get(
            ctx,
            vec![input_ty, output_ty, input_bits_ty, output_bits_ty],
        )
        .into();
        let payload = emit::<MirConstructTupleOp>(
            ctx,
            block,
            vec![tuple],
            vec![input, output, input_bits, output_bits],
        )
        .deref(ctx)
        .get_result(0);
        let environment_ty =
            MirPtrType::get_generic_with_kind(ctx, unit, false, MirPointerKind::SharedRef).into();
        let storage = value(ctx, block, unit);
        let controls = value(ctx, block, unit);
        let environment = value(ctx, block, environment_ty);
        let role = Role::Native(Native::CheckedBlockPayloadCallback {
            spec: NativeSpec {
                scalar: LibraryScalar::Signed(32),
                items: 1,
            },
            storage: LibraryStorageArg::Implicit,
        });
        let signature = FunctionType::get(ctx, vec![unit, unit, tuple, environment_ty], vec![]);
        MirLibraryOp::build_call(
            ctx,
            role,
            site(1),
            "callback".into(),
            signature,
            vec![storage, controls, payload, environment],
        )
        .insert_at_back(block, ctx);
        let control = emit::<MirLoadOp>(ctx, block, vec![scalar], vec![slot])
            .deref(ctx)
            .get_result(0);
        let second = if legacy {
            native(ctx, block, control, 2)
        } else {
            local_payload(ctx, block, vec![control], None, None, 2)
        };
        let roles = BTreeMap::from([(site(1), role), (site(2), second)]);
        emit::<MirReturnOp>(ctx, block, vec![], vec![]);
        for verdict in slot_and_promoted_verdicts(ctx, module, &roles) {
            assert_eq!(
                verdict.is_ok(),
                accepted,
                "written={written}, legacy={legacy}: {verdict:?}"
            );
            if let Err(error) = verdict {
                assert!(
                    error.contains(if legacy {
                        "untracked and planned-storage"
                    } else {
                        "not proved"
                    }),
                    "{error}"
                );
            }
        }
    }
}
