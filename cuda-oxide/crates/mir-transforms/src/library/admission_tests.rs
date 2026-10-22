/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::admission::*;
use super::context_plans::*;
use super::contexts::*;
use super::storage::{RootedNativePlans, Validation};
use dialect_mir::{
    library::{
        LibraryCallPlan, LibraryCallSite, LibraryFunctionSourceAttr,
        LibraryNativeMetadata as Native, LibraryRoleAttr, LibraryScalar, NativeSpec,
    },
    ops::*,
    types::{MirPointerKind, MirPtrType, MirTupleType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, StringAttr, TypeAttr},
        op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::{FunctionType, IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    graph::dominance::DomInfo,
    irbuild::{
        listener::DummyListener,
        rewriter::{IRRewriter, Rewriter},
    },
    location::Location,
    op::Op,
    operation::Operation,
    pass::AnalysisManager,
    r#type::TypeHandle,
    utils::apint::APInt,
    value::Value,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
};
fn module(ctx: &mut Context) -> Ptr<Operation> {
    dialect_mir::register(ctx);
    ModuleOp::new(ctx, "admission".try_into().unwrap()).get_operation()
}
fn function(
    ctx: &mut Context,
    module: Ptr<Operation>,
    name: &str,
    source: &str,
    results: Vec<TypeHandle>,
) -> (Ptr<Operation>, Ptr<BasicBlock>) {
    let ty = FunctionType::get(ctx, vec![], results);
    let op = Operation::new(
        ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let f = MirFuncOp::new(ctx, op, TypeAttr::new(ty.into()));
    f.set_symbol_name(ctx, name.try_into().unwrap());
    f.set_library_function_source(
        ctx,
        LibraryFunctionSourceAttr {
            export: source.into(),
            mangled: format!("mangled_{source}"),
            diagnostic: source.into(),
            checked_ancestor: true,
        },
    );
    Operation::get_op::<ModuleOp>(module, ctx)
        .unwrap()
        .append_operation(ctx, op, 0);
    let block = BasicBlock::new(ctx, None, vec![]);
    block.insert_at_back(op.deref(ctx).get_region(0), ctx);
    (op, block)
}
fn ret(ctx: &mut Context, block: Ptr<BasicBlock>, values: Vec<Value>) {
    Operation::new(
        ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        values,
        vec![],
        0,
    )
    .insert_at_back(block, ctx);
}
fn shape(ctx: &mut Context, function: Ptr<Operation>, dimensions: [u32; 3]) {
    let ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
    for (key, value) in ["reqntid_x", "reqntid_y", "reqntid_z"]
        .into_iter()
        .zip(dimensions)
    {
        function.deref_mut(ctx).attributes.set(
            key.try_into().unwrap(),
            IntegerAttr::new(
                ty,
                APInt::from_u64(value.into(), NonZeroUsize::new(32).unwrap()),
            ),
        );
    }
}
fn site(caller: &str, block: usize) -> LibraryCallSite {
    LibraryCallSite {
        caller: caller.into(),
        block,
    }
}
const SPEC: NativeSpec = NativeSpec {
    scalar: LibraryScalar::Signed(32),
    items: 4,
};
#[allow(clippy::disallowed_methods)]
fn native(ctx: &mut Context, block: Ptr<BasicBlock>, caller: &str, kind: Native) -> Ptr<Operation> {
    let scalar: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signed).into();
    let pointer =
        MirPtrType::get_generic_with_kind(ctx, scalar, true, MirPointerKind::RawMut).into();
    let tuple = MirTupleType::get(ctx, vec![pointer]).into();
    let args = match kind {
        Native::Raw { .. } => vec![tuple],
        Native::Callback { .. } => vec![
            tuple,
            MirPtrType::get_generic_with_kind(ctx, scalar, false, MirPointerKind::SharedRef).into(),
        ],
        _ => vec![tuple, tuple],
    };
    let values = args
        .iter()
        .map(|&ty| {
            let op = Operation::new(
                ctx,
                MirUndefOp::get_concrete_op_info(),
                vec![ty],
                vec![],
                vec![],
                0,
            );
            op.insert_at_back(block, ctx);
            op.deref(ctx).get_result(0)
        })
        .collect();
    let signature = FunctionType::get(ctx, args, vec![]);
    let op = MirLibraryOp::build_call(
        ctx,
        LibraryRoleAttr::Native(kind),
        site(caller, 7),
        "genuine_marker".into(),
        signature,
        values,
    );
    op.insert_at_back(block, ctx);
    op
}
#[test]
fn admission_keeps_shared_roots_separate_and_original_effects_after_preparation() {
    let ctx = &mut Context::new();
    let module = module(ctx);
    let (a, ab) = function(ctx, module, "root_a", "root_a", vec![]);
    ret(ctx, ab, vec![]);
    let (b, bb) = function(ctx, module, "root_b", "root_b", vec![]);
    ret(ctx, bb, vec![]);
    let (_, hb) = function(ctx, module, "physical_helper", "source::helper", vec![]);
    let native = native(
        ctx,
        hb,
        "source::helper",
        Native::CheckedBlock { spec: SPEC },
    );
    ret(ctx, hb, vec![]);
    let symbols = BTreeMap::from([
        ("root_a".into(), "root_a".into()),
        ("root_b".into(), "root_b".into()),
        ("source::helper".into(), "physical_helper".into()),
    ]);
    let roots = BTreeMap::from([
        (
            "root_a".into(),
            BTreeSet::from(["root_a".into(), "source::helper".into()]),
        ),
        (
            "root_b".into(),
            BTreeSet::from(["root_b".into(), "source::helper".into()]),
        ),
    ]);
    let roles = capture_library_roles(module, ctx, &symbols).unwrap();
    shape(ctx, a, [8, 4, 2]);
    assert!(admit_library_roots(module, ctx, &roots, &symbols, &roles).is_err());
    shape(ctx, b, [33, 1, 1]);
    assert_eq!(
        admit_library_roots(module, ctx, &roots, &symbols, &roles).unwrap(),
        BTreeMap::from([("root_a".into(), [8, 4, 2]), ("root_b".into(), [33, 1, 1])])
    );
    IRRewriter::<DummyListener>::default().erase_operation(ctx, native);
    shape(ctx, b, [0, 1, 1]);
    assert!(admit_library_roots(module, ctx, &roots, &symbols, &roles).is_err());
}
#[test]
fn raw_contract_keeps_its_explicit_shape_and_authenticates_surviving_effects() {
    let ctx = &mut Context::new();
    let module = module(ctx);
    let (_, block) = function(ctx, module, "raw", "raw", vec![]);
    let op = native(
        ctx,
        block,
        "raw",
        Native::Raw {
            spec: SPEC,
            block_threads: 48,
        },
    );
    ret(ctx, block, vec![]);
    let symbols = BTreeMap::from([("raw".into(), "raw".into())]);
    let roots = BTreeMap::from([("raw".into(), BTreeSet::from(["raw".into()]))]);
    let roles = capture_library_roles(module, ctx, &symbols).unwrap();
    assert!(
        admit_library_roots(module, ctx, &roots, &symbols, &roles)
            .unwrap()
            .is_empty()
    );
    MirLibraryOp::new(op).set_attr_role(
        ctx,
        LibraryRoleAttr::Native(Native::Raw {
            spec: SPEC,
            block_threads: 64,
        }),
    );
    assert!(admit_library_roots(module, ctx, &roots, &symbols, &roles).is_err());
}

#[test]
fn warp_admission_requires_complete_physical_blocks_but_queries_need_no_contract() {
    use dialect_mir::library::{LibraryAccess, LibraryWitness};
    let ctx = &mut Context::new();
    let module = module(ctx);
    let (root, block) = function(ctx, module, "root", "root", vec![]);
    let query = MirLibraryOp::build_query(ctx, site("root", 1), "query".into(), 16);
    query.insert_at_back(block, ctx);
    ret(ctx, block, vec![]);
    let symbols = BTreeMap::from([("root".into(), "root".into())]);
    let roots = BTreeMap::from([("root".into(), BTreeSet::from(["root".into()]))]);
    let mut roles = capture_library_roles(module, ctx, &symbols).unwrap();
    assert!(
        admit_library_roots(module, ctx, &roots, &symbols, &roles)
            .unwrap()
            .is_empty()
    );
    // Even a prepared-away native site retains its original geometry obligation.
    for (dimensions, width, accepted) in [
        ([64, 1, 1], 32, true),
        ([4, 4, 6], 32, true),
        ([48, 1, 1], 32, false),
        ([64, 1, 1], 16, false),
    ] {
        shape(ctx, root, dimensions);
        roles.insert(
            site("root", 2),
            LibraryRoleAttr::Native(Native::CheckedWarp {
                spec: SPEC,
                width,
                access: LibraryAccess::Read,
                witness: LibraryWitness::ReadOnlySlice,
            }),
        );
        let result = admit_library_roots(module, ctx, &roots, &symbols, &roles);
        assert_eq!(result.is_ok(), accepted);
        if let Err(error) = result {
            assert!(error.to_string().contains("complete physical warps"));
        }
    }
}

#[test]
fn callbacks_preserve_erased_source_obligations_and_actual_thunk_identity() {
    use super::callback::admit_library_callbacks;
    use dialect_mir::library::{
        LibraryCallbackCall, LibraryCallbackDefinition as D, LibraryCallbackDispatch as Dispatch,
        LibraryCallbackFunction, LibraryCallbackSources,
    };
    let ctx = &mut Context::new();
    let module = module(ctx);
    let (_, root) = function(ctx, module, "root", "root", vec![]);
    let callback = native(
        ctx,
        root,
        "root",
        Native::Callback {
            spec: SPEC,
            block_threads: 64,
        },
    );
    ret(ctx, root, vec![]);
    let (thunk, body) = function(ctx, module, "physical_thunk", "source::thunk", vec![]);
    ret(ctx, body, vec![]);
    let identity = "mangled_source::thunk".to_string();
    let symbols = BTreeMap::from([
        ("root".into(), "root".into()),
        ("source::thunk".into(), "physical_thunk".into()),
    ]);
    let mut sources = LibraryCallbackSources {
        roots: BTreeMap::from([(site("root", 7), identity.clone())]),
        functions: BTreeMap::from([(
            identity.clone(),
            LibraryCallbackFunction {
                diagnostic: "source::thunk".into(),
                export: Some("source::thunk".into()),
                definition: D::Rust,
                dispatch: Dispatch::Direct,
                has_body: true,
                cuda_local_types: vec![],
                calls: vec![],
                callees: vec![],
            },
        )]),
    };
    assert!(
        admit_library_callbacks(module, ctx, &LibraryCallbackSources::default(), &symbols).is_err()
    );
    assert_eq!(
        admit_library_callbacks(module, ctx, &sources, &symbols).unwrap(),
        BTreeMap::from([(identity.clone(), "physical_thunk".into())])
    );
    // Preparation may remove the original call, but not its source obligations.
    Operation::erase(callback, ctx);
    sources
        .functions
        .get_mut(&identity)
        .unwrap()
        .cuda_local_types
        .push("cuda_device::TempStorage".into());
    assert!(
        admit_library_callbacks(module, ctx, &sources, &symbols)
            .unwrap_err()
            .contains("cannot contain CUDA resource type")
    );
    sources
        .functions
        .get_mut(&identity)
        .unwrap()
        .cuda_local_types
        .clear();
    // An expanded scalar intrinsic may leave no ordinary call in prepared MIR.
    sources
        .functions
        .get_mut(&identity)
        .unwrap()
        .calls
        .push(LibraryCallbackCall::Direct {
            definition: D::Intrinsic,
            diagnostic: "core::intrinsics::opaque".into(),
        });
    assert!(
        admit_library_callbacks(module, ctx, &sources, &symbols)
            .unwrap_err()
            .contains("unqualified compiler intrinsic")
    );
    sources.functions.get_mut(&identity).unwrap().calls.clear();
    assert!(admit_library_callbacks(module, ctx, &sources, &symbols).is_ok());
    let f = MirFuncOp::wrap(ctx, thunk).unwrap();
    let mut source = f.library_function_source(ctx).unwrap();
    source.mangled = "different_instance".into();
    f.set_library_function_source(ctx, source);
    assert!(
        admit_library_callbacks(module, ctx, &sources, &symbols)
            .unwrap_err()
            .contains("does not match its actual definition")
    );
}

#[test]
fn callbacks_follow_receiver_edges_and_reject_missing_or_unavailable_bodies() {
    use super::callback::admit_library_callbacks;
    use dialect_mir::library::{
        LibraryCallbackDefinition as D, LibraryCallbackDispatch as Dispatch,
        LibraryCallbackFunction, LibraryCallbackSources,
    };
    let ctx = &mut Context::new();
    let module = module(ctx);
    let (_, body) = function(ctx, module, "physical_thunk", "thunk", vec![]);
    ret(ctx, body, vec![]);
    let class = || LibraryCallbackFunction {
        diagnostic: "helper".into(),
        export: None,
        definition: D::Rust,
        dispatch: Dispatch::Direct,
        has_body: true,
        cuda_local_types: vec![],
        calls: vec![],
        callees: vec![],
    };
    let mut thunk = class();
    thunk.export = Some("thunk".into());
    // Actual receiver and shim are parallel collector edges from the caller.
    thunk.callees = vec!["shim".into(), "receiver".into()];
    let mut shim = class();
    shim.dispatch = Dispatch::FunctionItemShim;
    shim.has_body = false;
    let mut sources = LibraryCallbackSources {
        roots: BTreeMap::from([(site("root", 7), "mangled_thunk".into())]),
        functions: BTreeMap::from([
            ("mangled_thunk".into(), thunk),
            ("shim".into(), shim),
            ("receiver".into(), class()),
        ]),
    };
    let symbols = BTreeMap::from([("thunk".into(), "physical_thunk".into())]);
    assert!(admit_library_callbacks(module, ctx, &sources, &symbols).is_ok());
    sources.functions.get_mut("shim").unwrap().dispatch = Dispatch::ClosureOnceShim;
    assert!(admit_library_callbacks(module, ctx, &sources, &symbols).is_ok());
    sources.functions.get_mut("receiver").unwrap().definition = D::Cuda;
    assert!(
        admit_library_callbacks(module, ctx, &sources, &symbols)
            .unwrap_err()
            .contains("CUDA resource or collective operation")
    );
    sources.functions.remove("receiver");
    assert!(
        admit_library_callbacks(module, ctx, &sources, &symbols)
            .unwrap_err()
            .contains("classification is missing")
    );
    sources.functions.insert("receiver".into(), class());
    sources.functions.get_mut("shim").unwrap().dispatch = Dispatch::Direct;
    assert!(
        admit_library_callbacks(module, ctx, &sources, &symbols)
            .unwrap_err()
            .contains("body is unavailable")
    );
    sources.functions.get_mut("shim").unwrap().dispatch = Dispatch::Indirect;
    assert!(
        admit_library_callbacks(module, ctx, &sources, &symbols)
            .unwrap_err()
            .contains("indirect call")
    );
}
#[test]
#[allow(clippy::disallowed_methods)]
fn contexts_preserve_original_calls_statics_and_cross_block_values() {
    let ctx = &mut Context::new();
    let module = module(ctx);
    let word: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    let (helper, hb) = function(ctx, module, "physical_helper", "source::helper", vec![word]);
    let loc = Location::Named {
        name: "source_helper".into(),
        child_loc: Box::new(Location::Unknown),
    };
    helper.deref_mut(ctx).set_loc(loc.clone());
    let global_ptr = MirPtrType::get_global(ctx, word, true).into();
    let global = Operation::new(
        ctx,
        MirGlobalAllocOp::get_concrete_op_info(),
        vec![global_ptr],
        vec![],
        vec![],
        0,
    );
    MirGlobalAllocOp::new(global).set_attr_global_type(ctx, TypeAttr::new(word));
    MirGlobalAllocOp::new(global)
        .set_attr_global_key(ctx, StringAttr::new("original_static_identity".into()));
    global.insert_at_back(hb, ctx);
    native(
        ctx,
        hb,
        "source::helper",
        Native::CheckedBlock { spec: SPEC },
    )
    .deref_mut(ctx)
    .set_loc(loc.clone());
    let value = Operation::new(
        ctx,
        MirUndefOp::get_concrete_op_info(),
        vec![word],
        vec![],
        vec![],
        0,
    );
    value.insert_at_back(hb, ctx);
    let successor = BasicBlock::new(ctx, None, vec![]);
    successor.insert_at_back(helper.deref(ctx).get_region(0), ctx);
    Operation::new(
        ctx,
        MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![successor],
        0,
    )
    .insert_at_back(hb, ctx);
    let cross_block_value = value.deref(ctx).get_result(0);
    ret(ctx, successor, vec![cross_block_value]);
    let mut calls = Vec::new();
    let mut roots = BTreeMap::new();
    let mut symbols = BTreeMap::from([("source::helper".into(), "physical_helper".into())]);
    let mut targets = BTreeMap::new();
    let mut native_plans = RootedNativePlans::new();
    for name in ["a", "b"] {
        let (_, block) = function(ctx, module, name, name, vec![]);
        let call = Operation::new(
            ctx,
            MirCallOp::get_concrete_op_info(),
            vec![word],
            vec![],
            vec![],
            0,
        );
        MirCallOp::new(call).set_attr_callee(ctx, StringAttr::new("physical_helper".into()));
        MirCallOp::new(call).set_attr_library_source_site(ctx, site(name, 1));
        call.insert_at_back(block, ctx);
        ret(ctx, block, vec![]);
        calls.push(call);
        roots.insert(
            name.into(),
            BTreeSet::from([name.into(), "source::helper".into()]),
        );
        symbols.insert(name.into(), name.into());
        targets.insert(
            site(name, 1),
            BTreeSet::from(["source::helper".into(), "callable_shim".into()]),
        );
        native_plans.insert(
            name.into(),
            BTreeMap::from([(
                site("source::helper", 7),
                LibraryCallPlan {
                    symbol: format!("native_{name}"),
                    marker: "genuine_marker".into(),
                    checked_block: true,
                    resource_witness: false,
                    scratch_bytes: 16,
                    scratch_alignment: 4,
                    ..LibraryCallPlan::internal(
                        dialect_mir::library::LibraryOperation::Native,
                        "genuine_marker".into(),
                    )
                },
            )]),
        );
    }
    let validation = Validation {
        shapes: BTreeMap::from([("a".into(), [64, 1, 1]), ("b".into(), [33, 1, 1])]),
        ..Default::default()
    };
    let assembly = assemble_library_plans(
        module,
        ctx,
        &roots,
        &targets,
        &validation,
        &BTreeMap::new(),
        &native_plans,
        BTreeMap::new(),
    )
    .unwrap();
    for emitted in assembly.contexts.keys() {
        symbols
            .entry(emitted.clone())
            .or_insert_with(|| emitted.replace(':', "_"));
    }
    assert_eq!(
        assembly
            .contexts
            .values()
            .filter(|v| v.as_str() == "source::helper")
            .count(),
        2
    );
    let mut analyses = AnalysisManager::default();
    analyses.compute_analysis::<DomInfo>(helper, ctx).unwrap();
    let invalid = BTreeMap::from([(site("a", 1), BTreeSet::from(["wrong".into()]))]);
    assert!(
        materialize_library_contexts(
            module,
            ctx,
            &mut analyses,
            &assembly.contexts,
            &symbols,
            &invalid,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &assembly.redirects
        )
        .is_err()
    );
    assert!(analyses.try_get_analysis::<DomInfo>(helper).is_some());
    assert!(functions(ctx, module).contains_key("physical_helper"));
    materialize_library_contexts(
        module,
        ctx,
        &mut analyses,
        &assembly.contexts,
        &symbols,
        &targets,
        &BTreeMap::new(),
        &BTreeMap::new(),
        &assembly.redirects,
    )
    .unwrap();
    assert!(analyses.try_get_analysis::<DomInfo>(helper).is_none());
    assert!(!functions(ctx, module).contains_key("physical_helper"));
    for (name, call) in ["a", "b"].into_iter().zip(calls) {
        let emitted = &assembly.redirects[name]["source::helper"];
        let physical = &symbols[emitted];
        let clone = functions(ctx, module)[physical];
        assert_eq!(clone.deref(ctx).loc(), &loc);
        assert_eq!(
            String::from((*MirCallOp::new(call).get_attr_callee(ctx).unwrap()).clone()),
            *physical
        );
        assert_eq!(
            *MirCallOp::new(call)
                .get_attr_library_source_site(ctx)
                .unwrap(),
            site(name, 1)
        );
        for op in operations(ctx, clone) {
            if let Some(library) = Operation::get_op::<MirLibraryOp>(op, ctx) {
                assert_eq!(*library.get_attr_site(ctx).unwrap(), site(emitted, 7));
            }
            if let Some(global) = Operation::get_op::<MirGlobalAllocOp>(op, ctx) {
                assert_eq!(
                    String::from((*global.get_attr_global_key(ctx).unwrap()).clone()),
                    "original_static_identity"
                );
            }
        }
    }
    assert_eq!(
        resolve_library_call_target(
            None,
            "a",
            "foreign",
            &symbols,
            &targets,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &assembly.redirects
        )
        .unwrap(),
        "foreign"
    );
    module.deref(ctx).verify(ctx).unwrap();
}
