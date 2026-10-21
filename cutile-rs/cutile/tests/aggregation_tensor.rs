/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-only coverage for the aggregation capabilities and their launch
//! obligations. Assertions inspect the structural Tile IR module.

use std::collections::BTreeMap;

use cutile::compile_api::KernelCompiler;
use cutile::tile_kernel::validate_launch_checks;
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::{Attribute, Module, OpId, Operation, Type};

mod common;

#[cutile::module]
mod aggregation_tensor_kernels {
    use cutile::core::*;

    /// Split-K matmul through the atomic capability.
    #[cutile::entry()]
    fn split_k_atomic(
        out: &AddTensor<f32, { [-1, -1] }>,
        lhs: &Tensor<f32, { [-1, -1] }>,
        rhs: &Tensor<f32, { [-1, -1] }>,
    ) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let m: i32 = bid.0;
        let n: i32 = bid.1;
        let k: i32 = bid.2;

        let lhs_part: Partition<f32, { [16, 32] }> = lhs.partition(shape![16, 32]);
        let rhs_part: Partition<f32, { [32, 16] }> = rhs.partition(shape![32, 16]);
        let a: Tile<f32, { [16, 32] }> = lhs_part.load([m, k]);
        let b: Tile<f32, { [32, 16] }> = rhs_part.load([k, n]);
        let zero: Tile<f32, { [16, 16] }> = constant(0.0f32, shape![16, 16]);
        let partial: Tile<f32, { [16, 16] }> = mma(a, b, zero);

        let acc: AddPartition<f32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _done: Token = acc.atomic_add([m, n], partial);
    }

    #[cutile::entry()]
    fn add_f16(out: &AddTensor<f16, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<f16, { [16, 16] }> = constant(f16::ONE, shape![16, 16]);
        let acc: AddPartition<f16, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_add([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn add_bf16(out: &AddTensor<bf16, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<bf16, { [16, 16] }> = constant(bf16::ONE, shape![16, 16]);
        let acc: AddPartition<bf16, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_add([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn add_i32(out: &AddTensor<i32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i32, { [16, 16] }> = constant(1i32, shape![16, 16]);
        let acc: AddPartition<i32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_add([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn add_u32(out: &AddTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [16, 16] }> = constant(1u32, shape![16, 16]);
        let acc: AddPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_add([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn accumulate_rank1(out: &AddTensor<f32, { [-1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<f32, { [64] }> = constant(1.0f32, shape![64]);
        let acc: AddPartition<f32, { [64] }> = out.partition(shape![64]);
        let _t: Token = acc.atomic_add([bid.0], v);
    }

    #[cutile::entry()]
    fn accumulate_rank3(out: &AddTensor<f32, { [-1, -1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<f32, { [4, 8, 16] }> = constant(1.0f32, shape![4, 8, 16]);
        let acc: AddPartition<f32, { [4, 8, 16] }> = out.partition(shape![4, 8, 16]);
        let _t: Token = acc.atomic_add([bid.0, bid.1, bid.2], v);
    }

    #[cutile::entry()]
    fn accumulate_const_generic<const BM: i32, const BN: i32>(out: &AddTensor<f32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<f32, { [BM, BN] }> = constant(1.0f32, shape![BM, BN]);
        let acc: AddPartition<f32, { [BM, BN] }> = out.partition(shape![BM, BN]);
        let _t: Token = acc.atomic_add([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn static_ragged(out: &AddTensor<f32, { [20, 48] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<f32, { [16, 16] }> = constant(1.0f32, shape![16, 16]);
        let acc: AddPartition<f32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_add([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn static_divisible(out: &AddTensor<f32, { [32, 48] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<f32, { [16, 16] }> = constant(1.0f32, shape![16, 16]);
        let acc: AddPartition<f32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_add([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn add_f64(out: &AddTensor<f64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<f64, { [16, 16] }> = constant(1.0f64, shape![16, 16]);
        let acc: AddPartition<f64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_add([bid.0, bid.1], v);
    }
}

#[cutile::module]
mod raw_intrinsic_kernels {
    use cutile::core::*;
    use cutile::tileir::*;

    #[cutile::entry()]
    unsafe fn split_k_raw(
        out: &Tensor<f32, { [-1, -1] }>,
        lhs: &Tensor<f32, { [-1, -1] }>,
        rhs: &Tensor<f32, { [-1, -1] }>,
    ) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let m: i32 = bid.0;
        let n: i32 = bid.1;
        let k: i32 = bid.2;

        let lhs_part: Partition<f32, { [16, 32] }> = lhs.partition(shape![16, 32]);
        let rhs_part: Partition<f32, { [32, 16] }> = rhs.partition(shape![32, 16]);
        let a: Tile<f32, { [16, 32] }> = lhs_part.load([m, k]);
        let b: Tile<f32, { [32, 16] }> = rhs_part.load([k, n]);
        let zero: Tile<f32, { [16, 16] }> = constant(0.0f32, shape![16, 16]);
        let partial: Tile<f32, { [16, 16] }> = mma(a, b, zero);

        let mut view: PartitionMut<f32, { [16, 16] }> =
            unsafe { out.partition_full_mut(shape![16, 16]) };
        let _done: Token = unsafe {
            atomic_red_view_tko(
                &mut view,
                partial,
                [m, n],
                atomic::AddF,
                ordering::Relaxed,
                scope::Device,
            )
        };
    }
}

// ── Host binding ────────────────────────────────────────────────────────────

/// Pins a launcher's result type, so the recovery assertions below are real
/// type equalities rather than inference.
fn assert_launch_output<O, D: cutile::prelude::DeviceOp<Output = O>>(_: &D) {}

/// Compile-only: the host side of the capability typechecks end to end.
///
/// Never called, since there is no GPU here, but building the launcher makes
/// rustc check the `KernelAtomic` bound on the exclusive borrow, the ordinary
/// `KernelInput` bound on the two shared inputs beside it, the explicit
/// split-K grid, and that the atomic parameter recovers as the backing borrow
/// rather than as the `Atomic<..>` wrapper. The negative direction (a shared
/// `Arc<Tensor<_>>` bound atomically) is a compile-fail case in `tests/ui/`.
#[allow(dead_code)]
fn borrowed_host_binding_typechecks<'a>(
    out: &'a mut cutile::tensor::Tensor<f32>,
    lhs: std::sync::Arc<cutile::tensor::Tensor<f32>>,
    rhs: std::sync::Arc<cutile::tensor::Tensor<f32>>,
) {
    use cutile::tile_kernel::TileKernel;
    let launch = aggregation_tensor_kernels::split_k_atomic(out.atomic(), lhs, rhs).grid((2, 2, 4));
    assert_launch_output::<
        (
            &'a mut cutile::tensor::Tensor<f32>,
            std::sync::Arc<cutile::tensor::Tensor<f32>>,
            std::sync::Arc<cutile::tensor::Tensor<f32>>,
        ),
        _,
    >(&launch);
}

/// Compile-only: an owned tensor is the other exclusive authority, and it
/// recovers as the plain `Tensor<f32>` it was.
#[allow(dead_code)]
fn owned_host_binding_typechecks(
    out: cutile::tensor::Tensor<f32>,
    lhs: std::sync::Arc<cutile::tensor::Tensor<f32>>,
    rhs: std::sync::Arc<cutile::tensor::Tensor<f32>>,
) {
    use cutile::tile_kernel::TileKernel;
    let launch =
        aggregation_tensor_kernels::split_k_atomic(cutile::tensor::Atomic::owned(out), lhs, rhs)
            .grid((2, 2, 4));
    assert_launch_output::<
        (
            cutile::tensor::Tensor<f32>,
            std::sync::Arc<cutile::tensor::Tensor<f32>>,
            std::sync::Arc<cutile::tensor::Tensor<f32>>,
        ),
        _,
    >(&launch);
}

// ── Tile IR enum values asserted below ──────────────────────────────────────
//
const MODE_ADD_F: i64 = 4;
const MODE_ADD: i64 = 3;
const MODE_AND: i64 = 0;
const MODE_OR: i64 = 1;
const MODE_XOR: i64 = 2;
const MODE_MAX: i64 = 5;
const MODE_MIN: i64 = 6;
const MODE_UMAX: i64 = 7;
const MODE_UMIN: i64 = 8;
const ORDERING_RELAXED: i64 = 1;
const SCOPE_DEVICE: i64 = 1;

const ROW_MAJOR_1D: &[i32] = &[1];
const ROW_MAJOR_2D: &[i32] = &[-1, 1];
const ROW_MAJOR_3D: &[i32] = &[-1, -1, 1];

const MATMUL_STRIDES: &[(&str, &[i32])] = &[
    ("out", ROW_MAJOR_2D),
    ("lhs", ROW_MAJOR_2D),
    ("rhs", ROW_MAJOR_2D),
];

fn compiler(
    module_ast: fn() -> cutile_compiler::ast::Module,
    module: &str,
    func: &str,
    strides: &[(&str, &[i32])],
) -> KernelCompiler<fn() -> cutile_compiler::ast::Module> {
    KernelCompiler::new(module_ast, module, func)
        .target("sm_120")
        .strides(strides)
}

fn compile(
    module_ast: fn() -> cutile_compiler::ast::Module,
    module: &str,
    func: &str,
    strides: &[(&str, &[i32])],
) -> Module {
    compile_with_counts(module_ast, module, func, strides).0
}

fn compile_with_counts(
    module_ast: fn() -> cutile_compiler::ast::Module,
    module: &str,
    func: &str,
    strides: &[(&str, &[i32])],
) -> (Module, cutile_compiler::compile_api::CheckPlacementCounts) {
    let artifacts = compiler(module_ast, module, func, strides)
        .compile()
        .expect("compilation failed");
    let counts = artifacts.check_counts();
    (artifacts.into_module(), counts)
}

fn atomic_module() -> Module {
    compile(
        aggregation_tensor_kernels::__module_ast_self,
        "aggregation_tensor_kernels",
        "split_k_atomic",
        MATMUL_STRIDES,
    )
}

fn raw_module() -> Module {
    compile(
        raw_intrinsic_kernels::__module_ast_self,
        "raw_intrinsic_kernels",
        "split_k_raw",
        MATMUL_STRIDES,
    )
}

fn ops(module: &Module) -> Vec<OpId> {
    fn walk(module: &Module, op: OpId, out: &mut Vec<OpId>) {
        out.push(op);
        for &region in &module.op(op).regions {
            for &block in &module.region(region).blocks {
                for &inner in &module.block(block).ops {
                    walk(module, inner, out);
                }
            }
        }
    }
    let mut out = Vec::new();
    for &function in &module.functions {
        walk(module, function, &mut out);
    }
    out
}

fn ops_with(module: &Module, opcode: Opcode) -> Vec<&Operation> {
    ops(module)
        .into_iter()
        .map(|id| module.op(id))
        .filter(|op| op.opcode == opcode)
        .collect()
}

/// The module's single atomic reduction.
fn atomic_reduction(module: &Module) -> &Operation {
    let found = ops_with(module, Opcode::AtomicRedViewTko);
    assert_eq!(
        found.len(),
        1,
        "expected exactly one atomic reduction, got {}",
        found.len()
    );
    found[0]
}

fn int_attr(op: &Operation, name: &str) -> i64 {
    match op
        .attributes
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value)
    {
        Some(Attribute::Integer(value, _)) => *value,
        other => panic!("expected an integer `{name}` attribute, got {other:?}"),
    }
}

#[test]
fn atomic_add_emits_the_expected_atomic_reduction() {
    common::with_test_stack(|| {
        let module = atomic_module();
        let op = atomic_reduction(&module);

        assert_eq!(int_attr(op, "mode"), MODE_ADD_F, "floating-add mode");
        assert_eq!(
            int_attr(op, "memory_ordering_semantics"),
            ORDERING_RELAXED,
            "relaxed ordering"
        );
        assert_eq!(int_attr(op, "memory_scope"), SCOPE_DEVICE, "device scope");

        // No old-value result: the completion token is the only result.
        assert_eq!(
            op.result_types,
            vec![Type::Token],
            "the reduction must discard old values and yield only its token"
        );

        // The view operand is an unpadded partition view whose tile shape is
        // the contributed tile's.
        let view = module
            .value_type(op.operands[0])
            .without_pointer_attribute();
        let Type::PartitionView(view) = view else {
            panic!("expected a partition view operand, got {view:?}");
        };
        assert_eq!(view.tile_shape, vec![16, 16], "tile geometry from the tile");
        assert_eq!(
            view.padding_value, None,
            "the atomic view must be unpadded, got {:?}",
            view.padding_value
        );

        // Nothing stored the contribution ordinarily.
        assert!(
            ops_with(&module, Opcode::StoreViewTko).is_empty(),
            "an ordinary store must not stand in for the atomic reduction"
        );

        // The capability itself lowers to a plain tensor view.
        assert_eq!(
            ops_with(&module, Opcode::MakeTensorView).len(),
            3,
            "one tensor view per tensor parameter"
        );
    });
}

/// The attributes and operand/result arity of one op, which is the
/// "operation profile" the two kernels must agree on. Whole-module equality is
/// not used: SSA numbering and the entry symbol differ by construction.
fn op_shape(module: &Module, op: &Operation) -> String {
    let mut attrs: BTreeMap<&str, String> = BTreeMap::new();
    for (key, value) in &op.attributes {
        attrs.insert(key.as_str(), format!("{value:?}"));
    }
    let operands: Vec<String> = op
        .operands
        .iter()
        .map(|&v| format!("{:?}", module.value_type(v)))
        .collect();
    format!(
        "{:?} operands={:?} results={:?} attrs={:?}",
        op.opcode, operands, op.result_types, attrs
    )
}

fn opcode_profile(module: &Module) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for id in ops(module) {
        *counts
            .entry(format!("{:?}", module.op(id).opcode))
            .or_insert(0) += 1;
    }
    counts
}

#[test]
fn atomic_add_matches_the_raw_intrinsic_kernel() {
    common::with_test_stack(|| {
        let atomic = atomic_module();
        let raw = raw_module();

        // Repeatability first: comparing two kernels by a profile proves
        // nothing unless the profile is stable for one kernel.
        assert_eq!(
            opcode_profile(&atomic),
            opcode_profile(&atomic_module()),
            "the opcode profile is not reproducible across compiles"
        );
        assert_eq!(
            op_shape(&atomic, atomic_reduction(&atomic)),
            op_shape(&atomic, atomic_reduction(&atomic_module())),
            "the reduction's profile is not reproducible across compiles"
        );

        // `atomic_add` bounds-checks its index where the raw intrinsic does
        // not, so the profiles are compared modulo `Constant`: the discharged
        // proof leaves a few folded constants behind. Every other opcode must
        // match, and in particular there must be no `Assert` — a device-side
        // bounds check would show up here.
        let drop_constants = |m: BTreeMap<String, usize>| -> BTreeMap<String, usize> {
            m.into_iter().filter(|(k, _)| k != "Constant").collect()
        };
        assert_eq!(
            drop_constants(opcode_profile(&atomic)),
            drop_constants(opcode_profile(&raw)),
            "`atomic_add` must emit the same operations as the raw intrinsic, \
             constants aside"
        );
        assert_eq!(
            op_shape(&atomic, atomic_reduction(&atomic)),
            op_shape(&raw, atomic_reduction(&raw)),
            "the reduction itself must be identical in operands, results and \
             attributes to the raw intrinsic's"
        );
    });
}

#[test]
fn shape_is_type_metadata_not_a_device_read() {
    common::with_test_stack(|| {
        let module = compile(
            aggregation_tensor_kernels::__module_ast_self,
            "aggregation_tensor_kernels",
            "split_k_atomic",
            MATMUL_STRIDES,
        );
        assert!(
            ops_with(&module, Opcode::GetTensorShape).is_empty(),
            "the view's shape metadata must not emit a device shape query"
        );
    });
}

#[test]
fn every_float_element_reaches_the_same_reduction() {
    common::with_test_stack(|| {
        for func in ["add_f16", "add_bf16", "add_f64"] {
            let module = compile(
                aggregation_tensor_kernels::__module_ast_self,
                "aggregation_tensor_kernels",
                func,
                &[("out", ROW_MAJOR_2D)],
            );
            let op = atomic_reduction(&module);
            assert_eq!(
                int_attr(op, "mode"),
                MODE_ADD_F,
                "{func}: floating-add mode"
            );
            assert_eq!(
                int_attr(op, "memory_ordering_semantics"),
                ORDERING_RELAXED,
                "{func}: relaxed"
            );
            assert_eq!(int_attr(op, "memory_scope"), SCOPE_DEVICE, "{func}: device");
            assert_eq!(op.result_types, vec![Type::Token], "{func}: token only");
        }
    });
}

#[test]
fn every_rank_and_a_const_generic_tile_reach_the_same_reduction() {
    common::with_test_stack(|| {
        // (entry, stride hints, generic arguments, expected tile geometry)
        let cases: Vec<(&str, &[(&str, &[i32])], Vec<String>, Vec<i32>)> = vec![
            (
                "accumulate_rank1",
                &[("out", ROW_MAJOR_1D)],
                vec![],
                vec![64],
            ),
            (
                "accumulate_rank3",
                &[("out", ROW_MAJOR_3D)],
                vec![],
                vec![4, 8, 16],
            ),
            (
                "accumulate_const_generic",
                &[("out", ROW_MAJOR_2D)],
                vec!["16".into(), "32".into()],
                vec![16, 32],
            ),
        ];

        for (func, strides, generics, tile) in cases {
            let module = compiler(
                aggregation_tensor_kernels::__module_ast_self,
                "aggregation_tensor_kernels",
                func,
                strides,
            )
            .generics(generics)
            .compile()
            .unwrap_or_else(|error| panic!("{func} failed to compile: {error}"))
            .into_module();

            let op = atomic_reduction(&module);
            assert_eq!(int_attr(op, "mode"), MODE_ADD_F, "{func}: floating-add");
            assert_eq!(
                int_attr(op, "memory_ordering_semantics"),
                ORDERING_RELAXED,
                "{func}: relaxed"
            );
            assert_eq!(int_attr(op, "memory_scope"), SCOPE_DEVICE, "{func}: device");
            assert_eq!(op.result_types, vec![Type::Token], "{func}: token only");

            let view = module
                .value_type(op.operands[0])
                .without_pointer_attribute();
            let Type::PartitionView(view) = view else {
                panic!("{func}: expected a partition view operand, got {view:?}");
            };
            assert_eq!(view.tile_shape, tile, "{func}: tile geometry");
            assert_eq!(view.padding_value, None, "{func}: unpadded");

            assert_eq!(
                op.operands.len(),
                tile.len() + 3,
                "{func}: one index component per axis"
            );
        }
    });
}

/// `atomic_add` is bounds-checked, and for a `tile_block_id` coordinate the
/// check costs nothing on the device: the block-id axiom rung reduces it to a
/// claim over launch-known operands. This asserts the placement rather than
/// assuming it — the counters are the compiler's own report, and an `Assert`
/// op in the module would mean a per-thread branch instead.
#[test]
fn a_block_id_index_discharges_its_bounds_check() {
    common::with_test_stack(|| {
        let (module, counts) = compile_with_counts(
            aggregation_tensor_kernels::__module_ast_self,
            "aggregation_tensor_kernels",
            "split_k_atomic",
            MATMUL_STRIDES,
        );
        assert!(
            counts.in_place == 0,
            "expected no device-side check, got {counts:?}"
        );
        assert!(
            counts.discharged > 0,
            "expected the index proof to be discharged, got {counts:?}"
        );
        assert!(
            ops_with(&module, Opcode::Assert).is_empty(),
            "a discharged check must leave no `Assert` in the module"
        );
    });
}

/// The divisibility obligations the compiler evacuated for parameter
/// `param`, as `axis -> divisor`.
///
/// Reads the [`Predicate`] structurally, not the cause string. The same
/// kernel also evacuates grid-bound checks stated over the *same* extents
/// (`num_tile_blocks(k) <= ceil(extent(out, k)/16)`), so "a launch check
/// exists" is not evidence that the tiling obligation was emitted; only a
/// `DivisibleBy` whose term is exactly `1 * Dim { param, axis }` is.
fn divisibility_obligations(
    checks: &[cuda_async::predicate::LaunchCheck],
    param: usize,
) -> BTreeMap<usize, i64> {
    use cuda_async::predicate::{Atom, Predicate};
    let mut found = BTreeMap::new();
    for check in checks {
        let Predicate::DivisibleBy { term, divisor } = &check.predicate else {
            continue;
        };
        let coeffs = term.coeffs();
        if coeffs.len() != 1 || term.constant_part() != 0 {
            continue;
        }
        let (&atom, &coeff) = coeffs.iter().next().expect("one coefficient");
        let Atom::Dim { param: p, axis } = atom else {
            continue;
        };
        if p != param || coeff != 1 {
            continue;
        }
        let previous = found.insert(axis, *divisor);
        assert!(
            previous.is_none(),
            "two divisibility obligations for param {param} axis {axis}"
        );
    }
    found
}

/// Compiles `split_k_atomic` and hands back its hoisted launch checks.
fn split_k_launch_checks() -> Vec<cuda_async::predicate::LaunchCheck> {
    compiler(
        aggregation_tensor_kernels::__module_ast_self,
        "aggregation_tensor_kernels",
        "split_k_atomic",
        MATMUL_STRIDES,
    )
    .compile()
    .expect("compilation failed")
    .launch_checks()
    .to_vec()
}

/// A ragged destination must be rejected before any GPU work.
///
/// A Tile IR partition index space is a ceiling division, so a 20-row tensor
/// tiled by 16 has a legitimate index 1 selecting a *partial* tile. Tile IR
/// documents masking for ordinary stores to partial tiles but does not grant
/// the same to `atomic_red_view_tko`, whose view must be unpadded — so a safe
/// reduction requires an exact tiling. `partition` emits that as a launch
/// obligation per axis; this asserts the obligations are there on *both*
/// output axes, and then evaluates the real emitted checks against a shape
/// that is ragged on each axis in turn.
#[test]
fn a_ragged_destination_is_rejected_before_launch() {
    common::with_test_stack(|| {
        let checks = split_k_launch_checks();

        // `out` is parameter 0 and the kernel tiles it [16, 16]. Both axes
        // must carry the obligation: one would leave the other edge ragged.
        assert_eq!(
            divisibility_obligations(&checks, 0),
            BTreeMap::from([(0usize, 16i64), (1usize, 16i64)]),
            "both output axes must carry a divisibility obligation naming the \
             tile extent; got {checks:#?}"
        );
        // And the causes must explain themselves, since a failing launch
        // shows the user nothing else.
        for check in &checks {
            if matches!(
                check.predicate,
                cuda_async::predicate::Predicate::DivisibleBy { .. }
            ) {
                assert!(
                    check
                        .cause
                        .contains("multiple of the aggregation tile extent 16"),
                    "unhelpful cause: {}",
                    check.cause
                );
            }
        }

        // The inputs are read through ordinary partitions, which tolerate a
        // partial edge tile; no divisibility obligation belongs to them.
        assert!(
            divisibility_obligations(&checks, 1).is_empty()
                && divisibility_obligations(&checks, 2).is_empty(),
            "only the aggregation destination needs an exact tiling: {checks:#?}"
        );

        // Now evaluate the checks the compiler actually emitted. The grid is
        // (2, 3, 4) throughout, and every `out` shape below satisfies the
        // grid-bound checks, so the only thing that can reject a launch here
        // is the tiling.
        let grid = (2u32, 3u32, 4u32);
        let shapes = |out: Vec<i32>| vec![out, vec![32, 128], vec![128, 48]];

        let exact = shapes(vec![32, 48]);
        validate_launch_checks(&checks, &exact, &exact, grid)
            .expect("a tiling that divides exactly must be accepted");

        for (ragged, axis) in [(vec![20, 48], 0), (vec![32, 50], 1)] {
            let ragged_shapes = shapes(ragged.clone());
            let Err(err) = validate_launch_checks(&checks, &ragged_shapes, &ragged_shapes, grid)
            else {
                panic!("a destination ragged on axis {axis} ({ragged:?}) must be rejected");
            };
            let message = err.to_string();
            assert!(
                message.contains(&format!("axis {axis} must be a multiple of")),
                "the rejection must name the ragged axis ({axis}): {message}"
            );
        }
    });
}

/// The converse of the static case: an extent the signature pins to a value
/// the tile does not divide is an impossibility, so the JIT rejects the kernel
/// outright rather than emitting a launch check that could never pass.
#[test]
fn a_statically_ragged_destination_fails_to_compile() {
    common::with_test_stack(|| {
        let result = compiler(
            aggregation_tensor_kernels::__module_ast_self,
            "aggregation_tensor_kernels",
            "static_ragged",
            &[("out", ROW_MAJOR_2D)],
        )
        .compile();
        let Err(error) = result else {
            panic!("a [20, 48] destination tiled [16, 16] must not compile");
        };
        let message = error.to_string();
        assert!(
            message.contains("declared extent 20") && message.contains("multiple of"),
            "the error must name the declared extent and the requirement: {message}"
        );
    });
}

/// Exactly-divisible compile-time extents cost nothing.
///
/// The obligation is the same one, but the signature pins the extents, so the
/// compiler decides it and leaves no divisibility check for the host to run
/// per launch. Asserted against the dynamic kernel, which does leave two: the
/// difference is the claim.
#[test]
fn an_exactly_divisible_static_shape_leaves_no_launch_check() {
    common::with_test_stack(|| {
        let artifacts = compiler(
            aggregation_tensor_kernels::__module_ast_self,
            "aggregation_tensor_kernels",
            "static_divisible",
            &[("out", ROW_MAJOR_2D)],
        )
        .compile()
        .expect("compilation failed");
        let checks = artifacts.launch_checks();

        assert!(
            divisibility_obligations(checks, 0).is_empty(),
            "a [32, 48] destination tiled [16, 16] divides exactly on both axes, \
             so nothing should be left for the host to re-check: {checks:#?}"
        );
        // The reduction is still there — the check was discharged, not the
        // kernel silently reshaped.
        assert_eq!(
            atomic_reduction(artifacts.module()).result_types,
            vec![Type::Token]
        );
        // Control: the same tiling over `{ [-1, -1] }` does leave two.
        assert_eq!(
            divisibility_obligations(&split_k_launch_checks(), 0).len(),
            2
        );
    });
}

#[test]
fn the_capability_is_not_partitioned_by_the_entry_wrapper() {
    common::with_test_stack(|| {
        let module = atomic_module();
        // Three partition views: `lhs`, `rhs`, and the one the reduction
        // builds internally. A fourth would mean the entry wrapper
        // partitioned the capability the way it partitions a `&mut Tensor`.
        assert_eq!(
            ops_with(&module, Opcode::MakePartitionView).len(),
            3,
            "expected one view per input partition plus the reduction's own"
        );
    });
}

/// One method spelling, two Tile IR modes, selected by the element type.
///
/// `atomic_add` is eight concrete-element impls rather than one generic impl,
/// because the JIT reads the mode from the literal spelled at the call to
/// `atomic_red_view_raw`. A generic body cannot supply it, and two impls with
/// disjoint bounds collide as duplicate definitions. This asserts the split
/// actually lands: integers reach `Add`, floats reach `AddF`.
#[test]
fn an_integer_element_reaches_the_integer_add_mode() {
    common::with_test_stack(|| {
        for func in ["add_i32", "add_u32"] {
            let module = compile(
                aggregation_tensor_kernels::__module_ast_self,
                "aggregation_tensor_kernels",
                func,
                &[("out", ROW_MAJOR_2D)],
            );
            let op = atomic_reduction(&module);
            assert_eq!(int_attr(op, "mode"), MODE_ADD, "{func}: integer add mode");
            assert_ne!(
                int_attr(op, "mode"),
                MODE_ADD_F,
                "{func}: must not reach the floating-add mode"
            );
        }
    });
}

/// Kernels spelling the semantic comparison markers at the raw op, one per
/// signedness. The marker is resolved by the compiler from the Rust element
/// type, which is the only place signedness survives: the Tile IR scalar for
/// `u32` is `i32`.
#[cutile::module]
mod semantic_marker_kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn max_i32(out: &AddTensor<i32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i32, { [16, 16] }> = constant(1i32, shape![16, 16]);
        let acc: AddPartition<i32, { [16, 16] }> = out.partition(shape![16, 16]);
        // SAFETY: probe kernel; the index is a block id inside the grid and
        // `partition` established exact tiling.
        let _t: Token = unsafe {
            atomic_red_view_raw(
                &acc,
                v,
                [bid.0, bid.1],
                atomic::Maximum,
                ordering::Relaxed,
                scope::Device,
                None,
            )
        };
    }

    #[cutile::entry()]
    fn max_u32(out: &AddTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [16, 16] }> = constant(1u32, shape![16, 16]);
        let acc: AddPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        // SAFETY: probe kernel; the index is a block id inside the grid and
        // `partition` established exact tiling.
        let _t: Token = unsafe {
            atomic_red_view_raw(
                &acc,
                v,
                [bid.0, bid.1],
                atomic::Maximum,
                ordering::Relaxed,
                scope::Device,
                None,
            )
        };
    }

    #[cutile::entry()]
    fn max_i64(out: &AddTensor<i64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i64, { [16, 16] }> = constant(1i64, shape![16, 16]);
        let acc: AddPartition<i64, { [16, 16] }> = out.partition(shape![16, 16]);
        // SAFETY: probe kernel; the index is a block id inside the grid and
        // `partition` established exact tiling.
        let _t: Token = unsafe {
            atomic_red_view_raw(
                &acc,
                v,
                [bid.0, bid.1],
                atomic::Maximum,
                ordering::Relaxed,
                scope::Device,
                None,
            )
        };
    }

    #[cutile::entry()]
    fn max_u64(out: &AddTensor<u64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u64, { [16, 16] }> = constant(1u64, shape![16, 16]);
        let acc: AddPartition<u64, { [16, 16] }> = out.partition(shape![16, 16]);
        // SAFETY: probe kernel; the index is a block id inside the grid and
        // `partition` established exact tiling.
        let _t: Token = unsafe {
            atomic_red_view_raw(
                &acc,
                v,
                [bid.0, bid.1],
                atomic::Maximum,
                ordering::Relaxed,
                scope::Device,
                None,
            )
        };
    }

    #[cutile::entry()]
    fn min_i32(out: &AddTensor<i32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i32, { [16, 16] }> = constant(1i32, shape![16, 16]);
        let acc: AddPartition<i32, { [16, 16] }> = out.partition(shape![16, 16]);
        // SAFETY: probe kernel; the index is a block id inside the grid and
        // `partition` established exact tiling.
        let _t: Token = unsafe {
            atomic_red_view_raw(
                &acc,
                v,
                [bid.0, bid.1],
                atomic::Minimum,
                ordering::Relaxed,
                scope::Device,
                None,
            )
        };
    }

    #[cutile::entry()]
    fn min_u32(out: &AddTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [16, 16] }> = constant(1u32, shape![16, 16]);
        let acc: AddPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        // SAFETY: probe kernel; the index is a block id inside the grid and
        // `partition` established exact tiling.
        let _t: Token = unsafe {
            atomic_red_view_raw(
                &acc,
                v,
                [bid.0, bid.1],
                atomic::Minimum,
                ordering::Relaxed,
                scope::Device,
                None,
            )
        };
    }

    #[cutile::entry()]
    fn min_i64(out: &AddTensor<i64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i64, { [16, 16] }> = constant(1i64, shape![16, 16]);
        let acc: AddPartition<i64, { [16, 16] }> = out.partition(shape![16, 16]);
        // SAFETY: probe kernel; the index is a block id inside the grid and
        // `partition` established exact tiling.
        let _t: Token = unsafe {
            atomic_red_view_raw(
                &acc,
                v,
                [bid.0, bid.1],
                atomic::Minimum,
                ordering::Relaxed,
                scope::Device,
                None,
            )
        };
    }

    #[cutile::entry()]
    fn min_u64(out: &AddTensor<u64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u64, { [16, 16] }> = constant(1u64, shape![16, 16]);
        let acc: AddPartition<u64, { [16, 16] }> = out.partition(shape![16, 16]);
        // SAFETY: probe kernel; the index is a block id inside the grid and
        // `partition` established exact tiling.
        let _t: Token = unsafe {
            atomic_red_view_raw(
                &acc,
                v,
                [bid.0, bid.1],
                atomic::Minimum,
                ordering::Relaxed,
                scope::Device,
                None,
            )
        };
    }

    #[cutile::entry()]
    fn max_f32(out: &AddTensor<f32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<f32, { [16, 16] }> = constant(1.0f32, shape![16, 16]);
        let acc: AddPartition<f32, { [16, 16] }> = out.partition(shape![16, 16]);
        // SAFETY: probe kernel; the index is a block id inside the grid and
        // `partition` established exact tiling.
        let _t: Token = unsafe {
            atomic_red_view_raw(
                &acc,
                v,
                [bid.0, bid.1],
                atomic::Maximum,
                ordering::Relaxed,
                scope::Device,
                None,
            )
        };
    }
    /// The destination view and contribution tile deliberately disagree in
    /// signedness: the raw signature ties nothing together, and both erase to
    /// the same IR scalar, so only the compiler's cross-check can see it.
    #[cutile::entry()]
    fn max_mismatched(out: &MaxTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i32, { [16, 16] }> = constant(1i32, shape![16, 16]);
        let acc: MaxPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        // SAFETY: probe kernel; never reaches the device.
        let _t: Token = unsafe {
            atomic_red_view_raw(
                &acc,
                v,
                [bid.0, bid.1],
                atomic::Maximum,
                ordering::Relaxed,
                scope::Device,
                None,
            )
        };
    }
}

fn semantic_marker_mode(func: &str) -> i64 {
    let module = compile(
        semantic_marker_kernels::__module_ast_self,
        "semantic_marker_kernels",
        func,
        &[("out", ROW_MAJOR_2D)],
    );
    int_attr(atomic_reduction(&module), "mode")
}

/// `Maximum` and `Minimum` resolve the comparison from the element's
/// signedness: signed integers reach `Max`/`Min`, unsigned reach
/// `Umax`/`Umin`. Both directions are asserted, because the failure mode is
/// a wrong comparison, not a compile error: a signed mode applied to `u32`
/// values straddling the high bit silently picks the smaller value.
#[test]
fn comparison_markers_resolve_by_signedness() {
    common::with_test_stack(|| {
        for (func, expect, wrong) in [
            ("max_i32", MODE_MAX, MODE_UMAX),
            ("max_u32", MODE_UMAX, MODE_MAX),
            ("max_i64", MODE_MAX, MODE_UMAX),
            ("max_u64", MODE_UMAX, MODE_MAX),
            ("min_i32", MODE_MIN, MODE_UMIN),
            ("min_u32", MODE_UMIN, MODE_MIN),
            ("min_i64", MODE_MIN, MODE_UMIN),
            ("min_u64", MODE_UMIN, MODE_MIN),
        ] {
            let mode = semantic_marker_mode(func);
            assert_eq!(mode, expect, "{func}");
            assert_ne!(mode, wrong, "{func}: wrong signedness selected");
        }
    });
}

/// Tile IR has no atomic floating-point maximum, so `Maximum` over a float
/// element is rejected in the JIT with a message naming both.
#[test]
fn a_float_comparison_marker_is_rejected() {
    common::with_test_stack(|| {
        let msg = match compiler(
            semantic_marker_kernels::__module_ast_self,
            "semantic_marker_kernels",
            "max_f32",
            &[("out", ROW_MAJOR_2D)],
        )
        .compile()
        {
            Ok(_) => panic!("a float atomic maximum must not compile"),
            Err(err) => err.to_string(),
        };
        assert!(
            msg.contains("`Maximum` has no atomic reduction mode for `f32`"),
            "unexpected message: {msg}"
        );
    });
}

/// The partition view an operation addresses must be unpadded: Tile IR
/// rejects a padded view for an atomic reduction, and the safe constructors
/// never create one.
fn assert_unpadded_view(module: &Module, func: &str) {
    let views = ops_with(module, Opcode::MakePartitionView);
    assert!(!views.is_empty(), "{func}: expected a partition view");
    for op in views {
        for ty in &op.result_types {
            if let Type::PartitionView(view) = ty {
                assert!(
                    view.padding_value.is_none(),
                    "{func}: the view must be unpadded"
                );
            }
        }
    }
}

/// One kernel per legal operation/element pair: the full 28-cell matrix the
/// three element traits admit, every kernel identical except the capability
/// type, the method, and the element.
#[cutile::module]
mod op_matrix_kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn add_f16(out: &AddTensor<f16, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<f16, { [16, 16] }> = constant(f16::ONE, shape![16, 16]);
        let acc: AddPartition<f16, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_add([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn add_bf16(out: &AddTensor<bf16, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<bf16, { [16, 16] }> = constant(bf16::ONE, shape![16, 16]);
        let acc: AddPartition<bf16, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_add([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn add_f32(out: &AddTensor<f32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<f32, { [16, 16] }> = constant(1.0f32, shape![16, 16]);
        let acc: AddPartition<f32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_add([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn add_f64(out: &AddTensor<f64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<f64, { [16, 16] }> = constant(1.0f64, shape![16, 16]);
        let acc: AddPartition<f64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_add([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn add_i32(out: &AddTensor<i32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i32, { [16, 16] }> = constant(1i32, shape![16, 16]);
        let acc: AddPartition<i32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_add([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn add_u32(out: &AddTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [16, 16] }> = constant(1u32, shape![16, 16]);
        let acc: AddPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_add([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn add_i64(out: &AddTensor<i64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i64, { [16, 16] }> = constant(1i64, shape![16, 16]);
        let acc: AddPartition<i64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_add([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn add_u64(out: &AddTensor<u64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u64, { [16, 16] }> = constant(1u64, shape![16, 16]);
        let acc: AddPartition<u64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_add([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn and_i32(out: &AndTensor<i32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i32, { [16, 16] }> = constant(1i32, shape![16, 16]);
        let acc: AndPartition<i32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_and([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn and_u32(out: &AndTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [16, 16] }> = constant(1u32, shape![16, 16]);
        let acc: AndPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_and([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn and_i64(out: &AndTensor<i64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i64, { [16, 16] }> = constant(1i64, shape![16, 16]);
        let acc: AndPartition<i64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_and([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn and_u64(out: &AndTensor<u64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u64, { [16, 16] }> = constant(1u64, shape![16, 16]);
        let acc: AndPartition<u64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_and([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn or_i32(out: &OrTensor<i32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i32, { [16, 16] }> = constant(1i32, shape![16, 16]);
        let acc: OrPartition<i32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_or([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn or_u32(out: &OrTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [16, 16] }> = constant(1u32, shape![16, 16]);
        let acc: OrPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_or([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn or_i64(out: &OrTensor<i64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i64, { [16, 16] }> = constant(1i64, shape![16, 16]);
        let acc: OrPartition<i64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_or([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn or_u64(out: &OrTensor<u64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u64, { [16, 16] }> = constant(1u64, shape![16, 16]);
        let acc: OrPartition<u64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_or([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn xor_i32(out: &XorTensor<i32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i32, { [16, 16] }> = constant(1i32, shape![16, 16]);
        let acc: XorPartition<i32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_xor([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn xor_u32(out: &XorTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [16, 16] }> = constant(1u32, shape![16, 16]);
        let acc: XorPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_xor([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn xor_i64(out: &XorTensor<i64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i64, { [16, 16] }> = constant(1i64, shape![16, 16]);
        let acc: XorPartition<i64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_xor([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn xor_u64(out: &XorTensor<u64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u64, { [16, 16] }> = constant(1u64, shape![16, 16]);
        let acc: XorPartition<u64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_xor([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn max_i32(out: &MaxTensor<i32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i32, { [16, 16] }> = constant(1i32, shape![16, 16]);
        let acc: MaxPartition<i32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_max([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn max_u32(out: &MaxTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [16, 16] }> = constant(1u32, shape![16, 16]);
        let acc: MaxPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_max([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn max_i64(out: &MaxTensor<i64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i64, { [16, 16] }> = constant(1i64, shape![16, 16]);
        let acc: MaxPartition<i64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_max([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn max_u64(out: &MaxTensor<u64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u64, { [16, 16] }> = constant(1u64, shape![16, 16]);
        let acc: MaxPartition<u64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_max([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn min_i32(out: &MinTensor<i32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i32, { [16, 16] }> = constant(1i32, shape![16, 16]);
        let acc: MinPartition<i32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_min([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn min_u32(out: &MinTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [16, 16] }> = constant(1u32, shape![16, 16]);
        let acc: MinPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_min([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn min_i64(out: &MinTensor<i64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<i64, { [16, 16] }> = constant(1i64, shape![16, 16]);
        let acc: MinPartition<i64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_min([bid.0, bid.1], v);
    }

    #[cutile::entry()]
    fn min_u64(out: &MinTensor<u64, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u64, { [16, 16] }> = constant(1u64, shape![16, 16]);
        let acc: MinPartition<u64, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_min([bid.0, bid.1], v);
    }
}

/// Every supported operation/element pair compiles and emits exactly the
/// expected Tile IR: the right mode for the pair (signedness included),
/// relaxed ordering, device scope, an unpadded view, and a completion token
/// as the only result.
#[test]
fn every_operation_element_pair_reaches_its_mode() {
    common::with_test_stack(|| {
        for (func, expected_mode) in [
            ("add_f16", MODE_ADD_F),
            ("add_bf16", MODE_ADD_F),
            ("add_f32", MODE_ADD_F),
            ("add_f64", MODE_ADD_F),
            ("add_i32", MODE_ADD),
            ("add_u32", MODE_ADD),
            ("add_i64", MODE_ADD),
            ("add_u64", MODE_ADD),
            ("and_i32", MODE_AND),
            ("and_u32", MODE_AND),
            ("and_i64", MODE_AND),
            ("and_u64", MODE_AND),
            ("or_i32", MODE_OR),
            ("or_u32", MODE_OR),
            ("or_i64", MODE_OR),
            ("or_u64", MODE_OR),
            ("xor_i32", MODE_XOR),
            ("xor_u32", MODE_XOR),
            ("xor_i64", MODE_XOR),
            ("xor_u64", MODE_XOR),
            ("max_i32", MODE_MAX),
            ("max_u32", MODE_UMAX),
            ("max_i64", MODE_MAX),
            ("max_u64", MODE_UMAX),
            ("min_i32", MODE_MIN),
            ("min_u32", MODE_UMIN),
            ("min_i64", MODE_MIN),
            ("min_u64", MODE_UMIN),
        ] {
            let module = compile(
                op_matrix_kernels::__module_ast_self,
                "op_matrix_kernels",
                func,
                &[("out", ROW_MAJOR_2D)],
            );
            let op = atomic_reduction(&module);
            assert_eq!(int_attr(op, "mode"), expected_mode, "{func}: mode");
            assert_eq!(
                int_attr(op, "memory_ordering_semantics"),
                ORDERING_RELAXED,
                "{func}: ordering"
            );
            assert_eq!(int_attr(op, "memory_scope"), SCOPE_DEVICE, "{func}: scope");
            assert_eq!(op.result_types, vec![Type::Token], "{func}: result");
            assert_unpadded_view(&module, func);
        }
    });
}

/// A module-local type alias names a capability the same way it names a
/// `Tensor`, and element inference reads through it.
#[cutile::module]
mod alias_kernels {
    use cutile::core::*;

    type Bins = AddTensor<u32, { [-1] }>;

    #[cutile::entry()]
    fn histogram(out: &Bins) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [64] }> = constant(1u32, shape![64]);
        let acc: AddPartition<u32, { [64] }> = out.partition(shape![64]);
        let _t: Token = acc.atomic_add([bid.0], v);
    }
}

/// The capability reached through a module-local alias lowers identically:
/// integer add mode through the alias, with the same view and token shape.
#[test]
fn a_type_alias_reaches_the_same_reduction() {
    common::with_test_stack(|| {
        let module = compile(
            alias_kernels::__module_ast_self,
            "alias_kernels",
            "histogram",
            &[("out", &[1])],
        );
        let op = atomic_reduction(&module);
        assert_eq!(
            int_attr(op, "mode"),
            MODE_ADD,
            "integer add through the alias"
        );
        assert_eq!(op.result_types, vec![Type::Token]);
    });
}

/// A statically ragged destination fails for every family, not only addition:
/// the divisibility obligation is emitted by each capability's `partition`.
#[test]
fn a_statically_ragged_max_destination_fails_to_compile() {
    common::with_test_stack(|| {
        let msg = match compiler(
            ragged_max_kernels::__module_ast_self,
            "ragged_max_kernels",
            "static_ragged_max",
            &[("out", ROW_MAJOR_2D)],
        )
        .compile()
        {
            Ok(_) => panic!("a statically ragged MaxTensor must not compile"),
            Err(err) => err.to_string(),
        };
        assert!(
            msg.contains("not a multiple of the aggregation tile extent"),
            "unexpected message: {msg}"
        );
    });
}

#[cutile::module]
mod ragged_max_kernels {
    use cutile::core::*;

    /// 20 is not divisible by 16: index 1 would select a partial tile.
    #[cutile::entry()]
    fn static_ragged_max(out: &MaxTensor<i32, { [20, 48] }>) {
        let v: Tile<i32, { [16, 16] }> = constant(1i32, shape![16, 16]);
        let acc: MaxPartition<i32, { [16, 16] }> = out.partition(shape![16, 16]);
        let _t: Token = acc.atomic_max([0i32, 0i32], v);
    }
}

/// A semantic marker resolves from the *destination* view, and a
/// contribution tile of a different element is rejected by name: both erase
/// to the same IR scalar, so resolving from the contribution would silently
/// select the wrong comparison.
#[test]
fn a_mismatched_contribution_element_is_rejected() {
    common::with_test_stack(|| {
        let msg = match compiler(
            semantic_marker_kernels::__module_ast_self,
            "semantic_marker_kernels",
            "max_mismatched",
            &[("out", ROW_MAJOR_2D)],
        )
        .compile()
        {
            Ok(_) => panic!("a u32 destination with an i32 contribution must not compile"),
            Err(err) => err.to_string(),
        };
        assert!(
            msg.contains("`Maximum` destination is `u32`, but contribution tile is `i32`"),
            "unexpected message: {msg}"
        );
    });
}

/// The token a checked operation returns is the reduction's own completion
/// token: consumed as a raw store's ordering operand, that operand's producer
/// is the `AtomicRedViewTko` itself and its type is `Token`. The weaker
/// `result_types == [Token]` cannot tell the op's token from any other.
#[test]
fn the_returned_token_is_the_reductions_own() {
    common::with_test_stack(|| {
        for func in [
            "consume_add",
            "consume_and",
            "consume_or",
            "consume_xor",
            "consume_max",
            "consume_min",
        ] {
            let module = compile(
                token_consumer_kernels::__module_ast_self,
                "token_consumer_kernels",
                func,
                &[("out", ROW_MAJOR_2D)],
            );
            let reductions = ops_with(&module, Opcode::AtomicRedViewTko);
            assert_eq!(reductions.len(), 2, "{func}: expected both reductions");
            let token = *module
                .op(ops(&module)
                    .into_iter()
                    .filter(|&id| module.op(id).opcode == Opcode::AtomicRedViewTko)
                    .nth(1)
                    .expect("second reduction"))
                .operands
                .last()
                .expect("a reduction has operands");
            let producer = match module.value_data(token).producer {
                cutile_ir::ir::ValueProducer::OpResult { op, .. } => Some(module.op(op).opcode),
                cutile_ir::ir::ValueProducer::BlockArg { .. } => None,
            };
            assert_eq!(
                producer,
                Some(Opcode::AtomicRedViewTko),
                "{func}: the consumed token must be the first reduction's result"
            );
            assert_eq!(module.value_type(token), &Type::Token, "{func}");
        }
    });
}

#[cutile::module]
mod token_consumer_kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn consume_add(out: &AddTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [16, 16] }> = constant(1u32, shape![16, 16]);
        let acc: AddPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        let done: Token = acc.atomic_add([bid.0, bid.1], v);
        // SAFETY: same view, in-grid index, exact tiling from `partition`.
        let _t: Token = unsafe {
            atomic_red_view_raw(
                &acc,
                v,
                [bid.0, bid.1],
                atomic::Sum,
                ordering::Relaxed,
                scope::Device,
                Some(done),
            )
        };
    }
    #[cutile::entry()]
    fn consume_and(out: &AndTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [16, 16] }> = constant(1u32, shape![16, 16]);
        let acc: AndPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        let done: Token = acc.atomic_and([bid.0, bid.1], v);
        // SAFETY: same view, in-grid index, exact tiling from `partition`.
        let _t: Token = unsafe {
            atomic_red_view_raw(
                &acc,
                v,
                [bid.0, bid.1],
                atomic::And,
                ordering::Relaxed,
                scope::Device,
                Some(done),
            )
        };
    }
    #[cutile::entry()]
    fn consume_or(out: &OrTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [16, 16] }> = constant(1u32, shape![16, 16]);
        let acc: OrPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        let done: Token = acc.atomic_or([bid.0, bid.1], v);
        // SAFETY: same view, in-grid index, exact tiling from `partition`.
        let _t: Token = unsafe {
            atomic_red_view_raw(
                &acc,
                v,
                [bid.0, bid.1],
                atomic::Or,
                ordering::Relaxed,
                scope::Device,
                Some(done),
            )
        };
    }
    #[cutile::entry()]
    fn consume_xor(out: &XorTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [16, 16] }> = constant(1u32, shape![16, 16]);
        let acc: XorPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        let done: Token = acc.atomic_xor([bid.0, bid.1], v);
        // SAFETY: same view, in-grid index, exact tiling from `partition`.
        let _t: Token = unsafe {
            atomic_red_view_raw(
                &acc,
                v,
                [bid.0, bid.1],
                atomic::Xor,
                ordering::Relaxed,
                scope::Device,
                Some(done),
            )
        };
    }
    #[cutile::entry()]
    fn consume_max(out: &MaxTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [16, 16] }> = constant(1u32, shape![16, 16]);
        let acc: MaxPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        let done: Token = acc.atomic_max([bid.0, bid.1], v);
        // SAFETY: same view, in-grid index, exact tiling from `partition`.
        let _t: Token = unsafe {
            atomic_red_view_raw(
                &acc,
                v,
                [bid.0, bid.1],
                atomic::Maximum,
                ordering::Relaxed,
                scope::Device,
                Some(done),
            )
        };
    }
    #[cutile::entry()]
    fn consume_min(out: &MinTensor<u32, { [-1, -1] }>) {
        let bid: (i32, i32, i32) = get_tile_block_id();
        let v: Tile<u32, { [16, 16] }> = constant(1u32, shape![16, 16]);
        let acc: MinPartition<u32, { [16, 16] }> = out.partition(shape![16, 16]);
        let done: Token = acc.atomic_min([bid.0, bid.1], v);
        // SAFETY: same view, in-grid index, exact tiling from `partition`.
        let _t: Token = unsafe {
            atomic_red_view_raw(
                &acc,
                v,
                [bid.0, bid.1],
                atomic::Minimum,
                ordering::Relaxed,
                scope::Device,
                Some(done),
            )
        };
    }
}
