/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression tests for CuTe operation, type, and semantic verification contracts.

use cute_layout::{
    ComposedLayout, IntTuple, Layout, MAX_STATIC_VALIDATION_BYTES, OffsetUnit, Swizzle,
};
use dialect_mir::{
    ops::MirUndefOp,
    types::{MirArrayType, MirFP16Type, MirPtrType, address_space},
};
use pliron::{
    builtin::types::{FP32Type, IntegerType, Signedness},
    common_traits::Verify,
    context::Context,
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
    value::Value,
};

use super::*;
use crate::attributes::{CuteDivisibilityAttr, CuteMatrixRoleAttr};

/// Create a typed placeholder SSA result so a verifier test needs no runtime input.
fn undef(ctx: &mut Context, ty: TypeHandle) -> Value {
    MirUndefOp::new(ctx, ty)
        .get_operation()
        .deref(ctx)
        .get_result(0)
}

/// Build a matrix-load fixture with a selected operand role and otherwise valid shared carriers.
fn matrix_load(
    ctx: &mut Context,
    role: CuteMatrixRoleAttr,
    source_space: u32,
    malformed: &str,
) -> CuteLdmatrixOp {
    let elem: TypeHandle = if malformed == "element" {
        FP32Type::get(ctx).into()
    } else {
        MirFP16Type::get(ctx).into()
    };
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    let source: TypeHandle = MirPtrType::get(ctx, elem, false, source_space).into();
    let words = if role == CuteMatrixRoleAttr::B { 2 } else { 4 };
    let registers: TypeHandle = MirArrayType::get(ctx, u32_ty, words).into();
    let slot: TypeHandle = MirPtrType::get_generic(ctx, registers, malformed != "slot").into();
    let operands = [
        undef(ctx, source),
        undef(ctx, if malformed == "row" { u32_ty } else { u64_ty }),
        undef(
            ctx,
            if malformed == "column" {
                u32_ty
            } else {
                u64_ty
            },
        ),
        undef(ctx, if malformed == "lane" { u64_ty } else { u32_ty }),
        undef(ctx, slot),
    ];
    let layout = ComposedLayout::new(
        Swizzle::new(3, 3, 3),
        0,
        "(64,64):(64,1)".parse().unwrap(),
        OffsetUnit::Elements,
    )
    .unwrap();
    CuteLdmatrixOp::new(ctx, operands, role, layout, elem)
}

/// All supported A/B load roles accept their matching shared FP16 and register-carrier ABI.
#[test]
fn matrix_load_roles_accept_the_shared_f16_fragment_abi() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);
    for role in [
        CuteMatrixRoleAttr::A,
        CuteMatrixRoleAttr::B,
        CuteMatrixRoleAttr::BPair,
        CuteMatrixRoleAttr::BPairKMajor,
    ] {
        for space in [address_space::SHARED, address_space::GENERIC] {
            matrix_load(&mut ctx, role, space, "").verify(&ctx).unwrap();
        }
    }
}

/// The non-transposed K-major B-pair variant still checks pointer roles and scalar carrier widths.
#[test]
fn k_major_b_pair_keeps_matrix_load_operand_checks() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);
    for (malformed, message) in [
        ("element", "supports f16 elements only"),
        ("row", "warp tile row must be an unsigned 64-bit integer"),
        (
            "column",
            "warp tile column must be an unsigned 64-bit integer",
        ),
        ("lane", "lane id must be an unsigned 32-bit integer"),
        ("slot", "fragment slot must be mutable"),
    ] {
        let load = matrix_load(
            &mut ctx,
            CuteMatrixRoleAttr::BPairKMajor,
            address_space::SHARED,
            malformed,
        );
        let error = load.verify(&ctx).unwrap_err().to_string();
        assert!(error.contains(message), "{malformed}: {error}");
    }
    let load = matrix_load(
        &mut ctx,
        CuteMatrixRoleAttr::BPairKMajor,
        address_space::GLOBAL,
        "",
    );
    let error = load.verify(&ctx).unwrap_err().to_string();
    assert!(
        error.contains("source must point at shared/generic f16"),
        "{error}"
    );
}

/// Return mutually compatible thread, value, tile, and shared layouts for one cooperative-copy partition.
fn valid_layouts() -> (Layout, Layout, Layout, ComposedLayout) {
    // Six rows, four columns. Each row/thread owns one 16-byte f32 atom.
    //
    // thread 2 -> (2,0) (2,1) (2,2) (2,3)
    let thread: Layout = "(6,1):(1,0)".parse().unwrap();
    let value: Layout = "(1,4):(0,1)".parse().unwrap();
    let tile: Layout = "(6,4):(4,1)".parse().unwrap();
    // Match the decoder canary: an eight-element prefix plus a composed
    // swizzle. The tile is small enough that the source field is zero,
    // but the complete composition and byte recast are still exercised.
    let smem =
        ComposedLayout::new(Swizzle::new(3, 4, 3), 8, tile.clone(), OffsetUnit::Elements).unwrap();
    (thread, value, tile, smem)
}

/// Build a cooperative-copy fixture with controllable pointers, atom width, and static layouts.
#[allow(clippy::too_many_arguments)]
fn cooperative_copy(
    ctx: &mut Context,
    src_mutable: bool,
    src_address_space: u32,
    dst_mutable: bool,
    dst_address_space: u32,
    atom_bytes: u32,
    thread_layout: Layout,
    value_layout: Layout,
    tile_layout: Layout,
    smem_layout: ComposedLayout,
) -> CuteCopyG2SOp {
    cooperative_copy_with_integer_abi(
        ctx,
        src_mutable,
        src_address_space,
        dst_mutable,
        dst_address_space,
        64,
        Signedness::Unsigned,
        32,
        Signedness::Unsigned,
        atom_bytes,
        thread_layout,
        value_layout,
        tile_layout,
        smem_layout,
    )
}

/// Build the copy fixture with independently selected index/lane integer types for negative ABI cases.
#[allow(clippy::too_many_arguments)]
fn cooperative_copy_with_integer_abi(
    ctx: &mut Context,
    src_mutable: bool,
    src_address_space: u32,
    dst_mutable: bool,
    dst_address_space: u32,
    usize_width: u32,
    usize_signedness: Signedness,
    tidx_width: u32,
    tidx_signedness: Signedness,
    atom_bytes: u32,
    thread_layout: Layout,
    value_layout: Layout,
    tile_layout: Layout,
    smem_layout: ComposedLayout,
) -> CuteCopyG2SOp {
    let elem: TypeHandle = FP32Type::get(ctx).into();
    let src_ty: TypeHandle = MirPtrType::get(ctx, elem, src_mutable, src_address_space).into();
    let dst_ty: TypeHandle = MirPtrType::get(ctx, elem, dst_mutable, dst_address_space).into();
    let usize_ty: TypeHandle = IntegerType::get(ctx, usize_width, usize_signedness).into();
    let tidx_ty: TypeHandle = IntegerType::get(ctx, tidx_width, tidx_signedness).into();
    let operands = [
        undef(ctx, src_ty),
        undef(ctx, usize_ty),
        undef(ctx, usize_ty),
        undef(ctx, usize_ty),
        undef(ctx, usize_ty),
        undef(ctx, usize_ty),
        undef(ctx, dst_ty),
        undef(ctx, usize_ty),
        undef(ctx, tidx_ty),
    ];
    CuteCopyG2SOp::new(
        ctx,
        operands,
        atom_bytes,
        thread_layout,
        value_layout,
        tile_layout,
        smem_layout,
        4,
        elem,
    )
}

/// Capture a failed local verification diagnostic for a malformed fixture operation.
fn verifier_error(copy: &CuteCopyG2SOp, ctx: &Context, expected: &str) {
    let error = copy.verify(ctx).unwrap_err().to_string();
    assert!(
        error.contains(expected),
        "expected `{expected}` in `{error}`"
    );
}

/// Build a raw TMA load fixture with the supplied destination and completion barrier.
fn tma_copy(ctx: &mut Context, dst_address_space: u32) -> CuteTmaLoad2dOp {
    let elem: TypeHandle = FP32Type::get(ctx).into();
    let u8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let destination: TypeHandle = MirPtrType::get(ctx, elem, true, dst_address_space).into();
    let barrier: TypeHandle = MirPtrType::get(ctx, u64_ty, true, address_space::SHARED).into();
    let tensor_map: TypeHandle = MirPtrType::get_generic(ctx, u8_ty, false).into();
    let operands = [
        undef(ctx, destination),
        undef(ctx, barrier),
        undef(ctx, tensor_map),
        undef(ctx, u64_ty),
        undef(ctx, u64_ty),
    ];
    let smem = ComposedLayout::from_layout("(4,4):(4,1)".parse().unwrap(), OffsetUnit::Elements);
    CuteTmaLoad2dOp::new(ctx, operands, smem, elem)
}

/// Build a raw TMA store fixture with the supplied shared source.
fn tma_store(ctx: &mut Context, src_address_space: u32) -> CuteTmaStore2dOp {
    let elem: TypeHandle = FP32Type::get(ctx).into();
    let u8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let source: TypeHandle = MirPtrType::get(ctx, elem, true, src_address_space).into();
    let tensor_map: TypeHandle = MirPtrType::get_generic(ctx, u8_ty, false).into();
    let operands = [
        undef(ctx, source),
        undef(ctx, tensor_map),
        undef(ctx, u64_ty),
        undef(ctx, u64_ty),
    ];
    let smem = ComposedLayout::from_layout("(4,4):(4,1)".parse().unwrap(), OffsetUnit::Elements);
    CuteTmaStore2dOp::new(ctx, operands, smem, elem)
}

/// Malformed operand/result counts report errors instead of indexing missing values and panicking.
#[test]
fn custom_verifiers_check_cardinality_before_indexing() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);

    let malformed = |ctx: &mut Context, info| Operation::new(ctx, info, vec![], vec![], vec![], 0);

    let g2s = CuteCopyG2SOp::wrap(malformed(&mut ctx, CuteCopyG2SOp::get_concrete_op_info()));
    verifier_error(&g2s, &ctx, "exactly 9 operands and 0 results");

    let copy = CuteCopyOp::wrap(malformed(&mut ctx, CuteCopyOp::get_concrete_op_info()));
    let error = copy.verify(&ctx).unwrap_err().to_string();
    assert!(
        error.contains("exactly 2 operands and 0 results"),
        "{error}"
    );

    let assume =
        CuteAssumeDivOp::wrap(malformed(&mut ctx, CuteAssumeDivOp::get_concrete_op_info()));
    let error = assume.verify(&ctx).unwrap_err().to_string();
    assert!(error.contains("exactly 1 operand and 1 result"), "{error}");

    let tma_load =
        CuteTmaLoad2dOp::wrap(malformed(&mut ctx, CuteTmaLoad2dOp::get_concrete_op_info()));
    let error = tma_load.verify(&ctx).unwrap_err().to_string();
    assert!(
        error.contains("exactly 5 operands and 0 results"),
        "{error}"
    );

    let tma_store = CuteTmaStore2dOp::wrap(malformed(
        &mut ctx,
        CuteTmaStore2dOp::get_concrete_op_info(),
    ));
    let error = tma_store.verify(&ctx).unwrap_err().to_string();
    assert!(
        error.contains("exactly 4 operands and 0 results"),
        "{error}"
    );
}

/// A complete, non-overlapping thread/value partition satisfies the cooperative-copy contract.
#[test]
fn cooperative_abi_accepts_one_exact_well_formed_partition() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);

    let (thread_layout, value_layout, tile_layout, smem_layout) = valid_layouts();

    let copy = cooperative_copy(
        &mut ctx,
        false,
        address_space::GLOBAL,
        true,
        address_space::SHARED,
        16,
        thread_layout,
        value_layout,
        tile_layout,
        smem_layout,
    );
    copy.verify(&ctx).unwrap();
}

/// Generic pointers cannot stand in for a proven CTA-local shared TMA destination.
#[test]
fn tma_abi_requires_an_explicit_cta_shared_destination() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);

    tma_copy(&mut ctx, address_space::SHARED)
        .verify(&ctx)
        .unwrap();
    let generic = tma_copy(&mut ctx, address_space::GENERIC);
    let error = generic.verify(&ctx).unwrap_err().to_string();
    assert!(error.contains("mutable CTA-shared elem memory"), "{error}");
}

/// TMA stores require a proven CTA-local shared source rather than an ambiguous generic pointer.
#[test]
fn tma_store_abi_requires_an_explicit_cta_shared_source() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);

    tma_store(&mut ctx, address_space::SHARED)
        .verify(&ctx)
        .unwrap();
    let generic = tma_store(&mut ctx, address_space::GENERIC);
    let error = generic.verify(&ctx).unwrap_err().to_string();
    assert!(error.contains("CTA-shared elem memory"), "{error}");
}

/// Row-pitch divisibility must preserve alignment for every vector copy atom.
#[test]
fn cooperative_abi_rejects_a_pitch_promise_weaker_than_the_atom() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);

    let (thread, value, tile, smem) = valid_layouts();
    let copy = cooperative_copy(
        &mut ctx,
        false,
        address_space::GLOBAL,
        true,
        address_space::SHARED,
        16,
        thread,
        value,
        tile,
        smem,
    );

    // f32 x a two-element pitch promise protects only 8 bytes:
    //
    // row start ── 8-byte guarantee ──X── 16-byte cp.async
    copy.set_attr_leading_dim_divisor(&ctx, CuteDivisibilityAttr(2));
    verifier_error(&copy, &ctx, "promise is weaker than its copy atom");
}

/// A zero-width copy atom fails validation before any alignment division occurs.
#[test]
fn cooperative_abi_rejects_zero_atom_before_alignment_arithmetic() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);

    let (thread, value, tile, smem) = valid_layouts();
    let copy = cooperative_copy(
        &mut ctx,
        false,
        address_space::GLOBAL,
        true,
        address_space::SHARED,
        0,
        thread,
        value,
        tile,
        smem,
    );
    verifier_error(&copy, &ctx, "atom must be 4, 8, or 16 bytes");
}

/// Global/shared pointer roles and destination writability are verified independently of layout shape.
#[test]
fn cooperative_abi_rejects_pointer_role_and_mutability_errors() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);

    let (thr, val, tile, smem) = valid_layouts();
    let shared_source = cooperative_copy(
        &mut ctx,
        false,
        address_space::SHARED,
        true,
        address_space::SHARED,
        16,
        thr,
        val,
        tile,
        smem,
    );
    verifier_error(
        &shared_source,
        &ctx,
        "global source has the wrong address space",
    );

    let (thr, val, tile, smem) = valid_layouts();
    let global_destination = cooperative_copy(
        &mut ctx,
        false,
        address_space::GLOBAL,
        true,
        address_space::GLOBAL,
        16,
        thr,
        val,
        tile,
        smem,
    );
    verifier_error(
        &global_destination,
        &ctx,
        "shared destination must be mutable shared/generic memory",
    );

    let (thr, val, tile, smem) = valid_layouts();
    let immutable_destination = cooperative_copy(
        &mut ctx,
        false,
        address_space::GLOBAL,
        false,
        address_space::SHARED,
        16,
        thr,
        val,
        tile,
        smem,
    );
    verifier_error(
        &immutable_destination,
        &ctx,
        "shared destination must be mutable shared/generic memory",
    );
}

/// Lane and index carriers must use the exact unsigned widths expected by lowering.
#[test]
fn cooperative_abi_rejects_integer_width_or_signedness_drift() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);

    let make = |ctx: &mut Context, usize_width, usize_signedness, tidx_width, tidx_signedness| {
        let (thread, value, tile, smem) = valid_layouts();
        cooperative_copy_with_integer_abi(
            ctx,
            false,
            address_space::GLOBAL,
            true,
            address_space::SHARED,
            usize_width,
            usize_signedness,
            tidx_width,
            tidx_signedness,
            16,
            thread,
            value,
            tile,
            smem,
        )
    };

    let narrow_usize = make(&mut ctx, 32, Signedness::Unsigned, 32, Signedness::Unsigned);
    verifier_error(
        &narrow_usize,
        &ctx,
        "row count must be an unsigned 64-bit integer",
    );

    let signed_usize = make(&mut ctx, 64, Signedness::Signed, 32, Signedness::Unsigned);
    verifier_error(
        &signed_usize,
        &ctx,
        "row count must be an unsigned 64-bit integer",
    );

    let signed_tidx = make(&mut ctx, 64, Signedness::Unsigned, 32, Signedness::Signed);
    verifier_error(
        &signed_tidx,
        &ctx,
        "thread index must be an unsigned 32-bit integer",
    );
}

/// Thread/value coverage cannot contain overlap, gaps, or an incompatible atom grouping.
#[test]
fn cooperative_abi_rejects_bad_tv_and_atom_partitions() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);

    let sparse_threads = cooperative_copy(
        &mut ctx,
        false,
        address_space::GLOBAL,
        true,
        address_space::SHARED,
        16,
        "(2,2):(0,1)".parse().unwrap(),
        "(1,4):(0,1)".parse().unwrap(),
        "(4,4):(4,1)".parse().unwrap(),
        ComposedLayout::from_layout("(4,4):(4,1)".parse().unwrap(), OffsetUnit::Elements),
    );
    verifier_error(&sparse_threads, &ctx, "invalid thread layout");

    let partial_atom = cooperative_copy(
        &mut ctx,
        false,
        address_space::GLOBAL,
        true,
        address_space::SHARED,
        16,
        "(6,1):(1,0)".parse().unwrap(),
        "(1,2):(0,1)".parse().unwrap(),
        "(6,2):(2,1)".parse().unwrap(),
        ComposedLayout::from_layout("(6,2):(2,1)".parse().unwrap(), OffsetUnit::Elements),
    );
    verifier_error(&partial_atom, &ctx, "invalid copy atom");

    let incomplete_tile = cooperative_copy(
        &mut ctx,
        false,
        address_space::GLOBAL,
        true,
        address_space::SHARED,
        16,
        "(6,1):(1,0)".parse().unwrap(),
        "(1,4):(0,1)".parse().unwrap(),
        "(6,3):(3,1)".parse().unwrap(),
        ComposedLayout::from_layout("(6,3):(3,1)".parse().unwrap(), OffsetUnit::Elements),
    );
    verifier_error(&incomplete_tile, &ctx, "assignments");

    let aliased_tile = cooperative_copy(
        &mut ctx,
        false,
        address_space::GLOBAL,
        true,
        address_space::SHARED,
        16,
        "(6,1):(1,0)".parse().unwrap(),
        "(1,4):(0,1)".parse().unwrap(),
        "(6,4):(0,1)".parse().unwrap(),
        ComposedLayout::from_layout("(6,4):(4,1)".parse().unwrap(), OffsetUnit::Elements),
    );
    verifier_error(&aliased_tile, &ctx, "invalid tile layout");

    let (thread, value, _, _) = valid_layouts();
    let flat_tile = cooperative_copy(
        &mut ctx,
        false,
        address_space::GLOBAL,
        true,
        address_space::SHARED,
        16,
        thread,
        value,
        Layout::contiguous(24),
        ComposedLayout::from_layout(Layout::contiguous(24), OffsetUnit::Elements),
    );
    verifier_error(&flat_tile, &ctx, "exactly two row/column tile modes");
}

/// Each instruction must address one contiguous aligned atom after shared swizzling.
#[test]
fn cooperative_abi_rejects_scattered_or_swizzle_split_atoms() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);

    let scattered_source = cooperative_copy(
        &mut ctx,
        false,
        address_space::GLOBAL,
        true,
        address_space::SHARED,
        16,
        "(6,1):(1,0)".parse().unwrap(),
        "(1,4):(0,1)".parse().unwrap(),
        "(6,4):(1,6)".parse().unwrap(),
        ComposedLayout::from_layout("(6,4):(4,1)".parse().unwrap(), OffsetUnit::Elements),
    );
    verifier_error(
        &scattered_source,
        &ctx,
        "tile layout is not canonical row-major",
    );

    let split_destination = cooperative_copy(
        &mut ctx,
        false,
        address_space::GLOBAL,
        true,
        address_space::SHARED,
        16,
        "(6,1):(1,0)".parse().unwrap(),
        "(1,4):(0,1)".parse().unwrap(),
        "(6,4):(4,1)".parse().unwrap(),
        ComposedLayout::new(
            Swizzle::new(1, 3, 3),
            64,
            "(6,4):(16,4)".parse().unwrap(),
            OffsetUnit::Bytes,
        )
        .unwrap(),
    );
    verifier_error(
        &split_destination,
        &ctx,
        "shared-memory copy atom is not physically contiguous",
    );

    let aliased_destination = cooperative_copy(
        &mut ctx,
        false,
        address_space::GLOBAL,
        true,
        address_space::SHARED,
        16,
        "(2,1):(1,0)".parse().unwrap(),
        "(1,4):(0,1)".parse().unwrap(),
        "(2,4):(4,1)".parse().unwrap(),
        ComposedLayout::new(
            Swizzle::IDENTITY,
            0,
            "(2,4):(0,1)".parse().unwrap(),
            OffsetUnit::Elements,
        )
        .unwrap(),
    );
    verifier_error(
        &aliased_destination,
        &ctx,
        "invalid physical shared-memory map",
    );

    let negative_destination = cooperative_copy(
        &mut ctx,
        false,
        address_space::GLOBAL,
        true,
        address_space::SHARED,
        16,
        "(2,1):(1,0)".parse().unwrap(),
        "(1,4):(0,1)".parse().unwrap(),
        "(2,4):(4,1)".parse().unwrap(),
        ComposedLayout::new(
            Swizzle::IDENTITY,
            -4,
            "(2,4):(4,1)".parse().unwrap(),
            OffsetUnit::Elements,
        )
        .unwrap(),
    );
    verifier_error(&negative_destination, &ctx, "negative smem offset");
}

/// A vector atom cannot straddle logical rows even when its flattened offsets look contiguous.
#[test]
fn source_atoms_cannot_cross_a_logical_tile_row() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);

    // The four f32 values are 16 contiguous bytes in a dense tile, but
    // they span two logical matrix rows:
    //
    // row 0: [ A B ]
    // row 1: [ C D ]  <- one atom must not bridge this boundary
    let tile: Layout = "(2,2):(2,1)".parse().unwrap();
    let crossing = cooperative_copy(
        &mut ctx,
        false,
        address_space::GLOBAL,
        true,
        address_space::SHARED,
        16,
        "(1,1):(1,1)".parse().unwrap(),
        "(2,2):(2,1)".parse().unwrap(),
        tile.clone(),
        ComposedLayout::from_layout(tile, OffsetUnit::Elements),
    );
    verifier_error(&crossing, &ctx, "global copy atom crosses a tile row");
}

/// Excessive enumeration sizes and overflowing address calculations fail conservatively.
#[test]
fn cooperative_abi_bounds_exhaustive_work_and_checked_offsets() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);

    let elements = MAX_STATIC_VALIDATION_BYTES / 4 + 1;
    let shape = IntTuple::Tuple(vec![IntTuple::Leaf(elements), IntTuple::Leaf(1)]);
    let thread = Layout::new(
        shape.clone(),
        IntTuple::Tuple(vec![IntTuple::Leaf(1), IntTuple::Leaf(0)]),
    );
    let value: Layout = "(1,1):(0,0)".parse().unwrap();
    let tile = Layout::new(
        shape,
        IntTuple::Tuple(vec![IntTuple::Leaf(1), IntTuple::Leaf(0)]),
    );
    let oversized = cooperative_copy(
        &mut ctx,
        false,
        address_space::GLOBAL,
        true,
        address_space::SHARED,
        4,
        thread,
        value,
        tile.clone(),
        ComposedLayout::from_layout(tile, OffsetUnit::Elements),
    );
    verifier_error(&oversized, &ctx, "static byte-map limit");

    let (thread, value, tile, _) = valid_layouts();
    let overflowing_smem = cooperative_copy(
        &mut ctx,
        false,
        address_space::GLOBAL,
        true,
        address_space::SHARED,
        16,
        thread,
        value,
        tile,
        ComposedLayout::new(
            Swizzle::IDENTITY,
            i64::MAX - 15,
            "(6,4):(16,4)".parse().unwrap(),
            OffsetUnit::Bytes,
        )
        .unwrap(),
    );
    verifier_error(&overflowing_smem, &ctx, "composed offset");
}
