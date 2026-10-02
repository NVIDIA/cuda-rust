/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! cute dialect ops.
//!
//! The op set is intentionally small. `cute.copy` and `cute.copy_g2s` are
//! zero-result memory ops; `cute.assume_div` is an identity value that
//! preserves one compiler fact.

use cute_layout::{ComposedLayout, Layout, validate_cooperative_copy_plan};
use dialect_mir::types::{MirFP16Type, MirPtrType, address_space};
use pliron::{
    builtin::{
        attributes::TypeAttr,
        op_interfaces::{NOpdsInterface, NResultsInterface, OneOpdInterface, OneResultInterface},
        types::IntegerType,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::{TypeHandle, Typed},
    value::Value,
    verify_err,
};
use pliron_derive::pliron_op;

use crate::attributes::{
    CuteComposedLayoutAttr, CuteCopyAtomAttr, CuteDivisibilityAttr, CuteLayoutAttr,
    CuteMatrixRoleAttr,
};

/// Frozen zero-result ABI for a block-wide global-to-shared copy.
///
/// Runtime operands are flattened so backend lowering never depends on Rust
/// struct field layout or traces a value back to a constructor:
///
/// ```text
/// 0 gmem base       3 leading dim      6 smem base
/// 1 rows            4 tile row         7 smem capacity
/// 2 columns         5 tile column      8 thread index
/// ```
///
/// Static layouts and the atom stay in typed attributes. Boundary handling
/// is fixed to uniform `cp.async` source-size predication; commit/wait/barrier
/// remain explicit operations outside this op. Operands 1..=5 and 7 are
/// NVPTX64 Rust `usize` (`u64`); operand 8 is Rust `u32`. The leading-dimension
/// divisor is also an attribute, so M3 never has to rediscover an alignment
/// promise hidden inside the source carrier.
#[pliron_op(
    name = "cute.copy_g2s",
    format,
    interfaces = [NOpdsInterface<9>, NResultsInterface<0>],
    attributes = (
        atom_bytes: CuteCopyAtomAttr,
        thread_layout: CuteLayoutAttr,
        value_layout: CuteLayoutAttr,
        tile_layout: CuteLayoutAttr,
        smem_layout: CuteComposedLayoutAttr,
        leading_dim_divisor: CuteDivisibilityAttr,
        copy_g2s_elem: TypeAttr
    )
)]
pub struct CuteCopyG2SOp;

impl CuteCopyG2SOp {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &mut Context,
        operands: [Value; 9],
        atom_bytes: u32,
        thread_layout: Layout,
        value_layout: Layout,
        tile_layout: Layout,
        smem_layout: ComposedLayout,
        leading_dim_divisor: u64,
        elem: TypeHandle,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            operands.to_vec(),
            vec![],
            0,
        );
        let copy = Self { op };
        copy.set_attr_atom_bytes(ctx, CuteCopyAtomAttr(atom_bytes));
        copy.set_attr_thread_layout(ctx, CuteLayoutAttr(thread_layout));
        copy.set_attr_value_layout(ctx, CuteLayoutAttr(value_layout));
        copy.set_attr_tile_layout(ctx, CuteLayoutAttr(tile_layout));
        copy.set_attr_smem_layout(ctx, CuteComposedLayoutAttr(smem_layout));
        copy.set_attr_leading_dim_divisor(ctx, CuteDivisibilityAttr(leading_dim_divisor));
        copy.set_attr_copy_g2s_elem(ctx, TypeAttr::new(elem));
        copy
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

fn cooperative_elem_bytes(ctx: &Context, elem: TypeHandle) -> Option<i64> {
    let ty = elem.deref(ctx);
    if let Some(integer) = ty.downcast_ref::<IntegerType>() {
        let width = i64::from(integer.width());
        return (width > 0 && width % 8 == 0).then_some(width / 8);
    }
    if ty
        .downcast_ref::<pliron::builtin::types::FP32Type>()
        .is_some()
    {
        return Some(4);
    }
    if ty
        .downcast_ref::<pliron::builtin::types::FP64Type>()
        .is_some()
    {
        return Some(8);
    }
    ty.downcast_ref::<MirFP16Type>().is_some().then_some(2)
}

impl Verify for CuteCopyG2SOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        if op.get_num_operands() != 9 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.copy_g2s needs exactly 9 operands and 0 results, got {} and {}",
                op.get_num_operands(),
                op.get_num_results()
            );
        }
        let Some(atom_bytes) = self.get_attr_atom_bytes(ctx).map(|attr| attr.0) else {
            return verify_err!(op.loc(), "cute.copy_g2s must have atom_bytes");
        };
        if !matches!(atom_bytes, 4 | 8 | 16) {
            return verify_err!(op.loc(), "cute.copy_g2s atom must be 4, 8, or 16 bytes");
        }
        let Some(thread_layout) = self.get_attr_thread_layout(ctx).map(|attr| attr.0.clone())
        else {
            return verify_err!(op.loc(), "cute.copy_g2s must have a thread layout");
        };
        let Some(value_layout) = self.get_attr_value_layout(ctx).map(|attr| attr.0.clone()) else {
            return verify_err!(op.loc(), "cute.copy_g2s must have a value layout");
        };
        let Some(tile_layout) = self.get_attr_tile_layout(ctx).map(|attr| attr.0.clone()) else {
            return verify_err!(op.loc(), "cute.copy_g2s must have a tile layout");
        };
        let Some(smem_layout) = self.get_attr_smem_layout(ctx).map(|attr| attr.0.clone()) else {
            return verify_err!(op.loc(), "cute.copy_g2s must have a shared-memory layout");
        };
        let Some(leading_dim_divisor) = self.get_attr_leading_dim_divisor(ctx).map(|attr| attr.0)
        else {
            return verify_err!(
                op.loc(),
                "cute.copy_g2s must have a leading-dimension divisor"
            );
        };
        if leading_dim_divisor == 0 {
            return verify_err!(
                op.loc(),
                "cute.copy_g2s leading-dimension divisor must be positive"
            );
        }
        let Some(elem): Option<TypeHandle> = self
            .get_attr_copy_g2s_elem(ctx)
            .map(|attr| attr.clone().into())
        else {
            return verify_err!(op.loc(), "cute.copy_g2s must have an element type");
        };
        let Some(elem_bytes) = cooperative_elem_bytes(ctx, elem) else {
            return verify_err!(op.loc(), "unsupported cute.copy_g2s element type");
        };
        let Some(leading_dim_alignment) = leading_dim_divisor.checked_mul(elem_bytes as u64) else {
            return verify_err!(
                op.loc(),
                "cute.copy_g2s leading-dimension alignment overflows u64"
            );
        };
        if leading_dim_alignment % u64::from(atom_bytes) != 0 {
            return verify_err!(
                op.loc(),
                "cute.copy_g2s leading-dimension promise is weaker than its copy atom"
            );
        }
        // One shared pure validator feeds both importer recognition and this
        // op. M3 can consume the returned TV map and capacity without
        // re-deriving any layout fact.
        let _plan = validate_cooperative_copy_plan(
            atom_bytes,
            &thread_layout,
            &value_layout,
            &tile_layout,
            &smem_layout,
            elem_bytes,
        )
        .map_err(|error| {
            pliron::input_error!(op.loc(), "invalid cooperative copy plan: {error}")
        })?;

        for (index, role) in [(0usize, "global source"), (6usize, "shared destination")] {
            let ty = op.get_operand(index).get_type(ctx);
            let ty_ref = ty.deref(ctx);
            let Some(pointer) = ty_ref.downcast_ref::<MirPtrType>() else {
                return verify_err!(op.loc(), "{role} must be a MIR pointer");
            };
            if pointer.pointee != elem {
                return verify_err!(op.loc(), "{role} pointee must match elem");
            }
            if index == 0
                && ![address_space::GENERIC, address_space::GLOBAL].contains(&pointer.address_space)
            {
                return verify_err!(op.loc(), "global source has the wrong address space");
            }
            if index == 6
                && (!pointer.is_mutable
                    || ![address_space::GENERIC, address_space::SHARED]
                        .contains(&pointer.address_space))
            {
                return verify_err!(
                    op.loc(),
                    "shared destination must be mutable shared/generic memory"
                );
            }
        }

        for (index, role) in [
            (1usize, "row count"),
            (2usize, "column count"),
            (3usize, "leading dimension"),
            (4usize, "tile row"),
            (5usize, "tile column"),
            (7usize, "shared-memory capacity"),
        ] {
            let ty = op.get_operand(index).get_type(ctx);
            if ty
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .is_none_or(|integer| integer.width() != 64 || !integer.is_unsigned())
            {
                return verify_err!(op.loc(), "{role} must be an unsigned 64-bit integer");
            }
        }
        let tidx_ty = op.get_operand(8).get_type(ctx);
        if tidx_ty
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .is_none_or(|integer| integer.width() != 32 || !integer.is_unsigned())
        {
            return verify_err!(op.loc(), "thread index must be an unsigned 32-bit integer");
        }
        Ok(())
    }
}

/// Preserve the promise `value % divisor == 0` in SSA form.
///
/// ```text
/// Rust                       cute IR                       lowering
/// ─────                      ───────                       ────────
/// y = assume_div::<16>(x) -> y = cute.assume_div x {16} -> y uses x
///                                  │
///                                  └─ direct SSA consumers read the fact
/// ```
///
/// The operation does not change the runtime value. Its result exists so the
/// fact follows that exact value through ordinary SSA uses; a detached
/// zero-result annotation would require guessing which later value it meant.
#[pliron_op(
    name = "cute.assume_div",
    format,
    interfaces = [
        NOpdsInterface<1>,
        NResultsInterface<1>,
        OneOpdInterface,
        OneResultInterface
    ],
    attributes = (divisor: CuteDivisibilityAttr)
)]
pub struct CuteAssumeDivOp;

impl CuteAssumeDivOp {
    pub fn new(ctx: &mut Context, value: Value, divisor: u64) -> Self {
        let result_ty = value.get_type(ctx);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty],
            vec![value],
            vec![],
            0,
        );
        let assume = Self { op };
        assume.set_attr_divisor(ctx, CuteDivisibilityAttr(divisor));
        assume
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    pub fn divisor(&self, ctx: &Context) -> Option<u64> {
        self.get_attr_divisor(ctx).map(|attr| attr.0)
    }
}

impl Verify for CuteAssumeDivOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.assume_div needs exactly 1 operand and 1 result, got {} and {}",
                op.get_num_operands(),
                op.get_num_results()
            );
        }
        let Some(divisor) = self.divisor(ctx) else {
            return verify_err!(op.loc(), "cute.assume_div must have a divisor attribute");
        };
        if divisor == 0 {
            return verify_err!(
                op.loc(),
                "cute.assume_div divisor must be greater than zero"
            );
        }

        let input_ty = op.get_operand(0).get_type(ctx);
        if input_ty.deref(ctx).downcast_ref::<IntegerType>().is_none() {
            return verify_err!(op.loc(), "cute.assume_div input must be an integer");
        }
        if op.get_result(0).get_type(ctx) != input_ty {
            return verify_err!(
                op.loc(),
                "cute.assume_div result must have the same type as its input"
            );
        }
        Ok(())
    }
}

/// CuTe tile copy: copy the tile described by `layout` (element type `elem`)
/// from the memory at `src` to the memory at `dst`.
///
/// # Operands
///
/// ```text
/// | Name  | Type       | Description                        |
/// |-------|------------|------------------------------------|
/// | `src` | MirPtrType | Source pointer (pointee == elem)   |
/// | `dst` | MirPtrType | Destination pointer (pointee == elem) |
/// ```
///
/// # Attributes
///
/// ```text
/// | Name     | Type           | Description                       |
/// |----------|----------------|-----------------------------------|
/// | `layout` | CuteLayoutAttr | Tile layout, e.g. (4):(1)         |
/// | `elem`   | TypeAttr       | Element type                      |
/// ```
///
/// # Contract
///
/// Both pointers are naturally aligned for the whole tile (like CUDA's
/// `float4`): for an identity-map layout of `n` elements of size `s`, the
/// alignment is `n * s` bytes. The device API documents this; backend
/// lowering relies on it to emit a single vectorized access.
#[pliron_op(
    name = "cute.copy",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
    attributes = (layout: CuteLayoutAttr, elem: TypeAttr)
)]
pub struct CuteCopyOp;

impl CuteCopyOp {
    /// Create a `cute.copy`, computing nothing: the op carries the layout
    /// structurally for the selected backend continuation.
    pub fn new(
        ctx: &mut Context,
        src: Value,
        dst: Value,
        layout: Layout,
        elem: TypeHandle,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![src, dst],
            vec![],
            0,
        );
        let copy = CuteCopyOp { op };
        copy.set_attr_layout(ctx, CuteLayoutAttr(layout));
        copy.set_attr_elem(ctx, TypeAttr::new(elem));
        copy
    }

    /// Wrap an existing `cute.copy` operation (caller must have matched the
    /// OpId; used by backend continuations).
    pub fn wrap(op: Ptr<Operation>) -> Self {
        CuteCopyOp { op }
    }

    /// The tile layout (cloned out of the attribute).
    pub fn layout(&self, ctx: &Context) -> Option<Layout> {
        self.get_attr_layout(ctx).map(|a| a.0.clone())
    }

    /// The element type.
    pub fn elem(&self, ctx: &Context) -> Option<TypeHandle> {
        self.get_attr_elem(ctx).map(|a| a.clone().into())
    }
}

impl Verify for CuteCopyOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.copy needs exactly 2 operands and 0 results, got {} and {}",
                op.get_num_operands(),
                op.get_num_results()
            );
        }
        let elem: TypeHandle = match self.get_attr_elem(ctx) {
            Some(a) => a.clone().into(),
            None => return verify_err!(op.loc(), "cute.copy must have an elem attribute"),
        };
        if self.get_attr_layout(ctx).is_none() {
            return verify_err!(op.loc(), "cute.copy must have a layout attribute");
        }
        if self
            .get_attr_layout(ctx)
            .is_none_or(|attr| attr.0.checked_size().is_none())
        {
            return verify_err!(
                op.loc(),
                "cute.copy layout needs positive extents and a non-overflowing size"
            );
        }
        for (i, name) in [(0usize, "src"), (1usize, "dst")] {
            let ty = op.get_operand(i).get_type(ctx);
            let ty_ref = ty.deref(ctx);
            let Some(ptr) = ty_ref.downcast_ref::<MirPtrType>() else {
                return verify_err!(op.loc(), "cute.copy {} operand must be a mir.ptr", name);
            };
            if ptr.pointee != elem {
                return verify_err!(
                    op.loc(),
                    "cute.copy {} operand pointee must match the elem attribute",
                    name
                );
            }
            if i == 1 && !ptr.is_mutable {
                return verify_err!(op.loc(), "cute.copy destination pointer must be mutable");
            }
        }
        Ok(())
    }
}

/// Frozen zero-result ABI for one warp-cooperative `ldmatrix` fragment load.
///
/// ```text
/// 0 smem base (f16)   2 warp tile column   4 fragment slot (u32 array)
/// 1 warp tile row     3 lane id
/// ```
///
/// The shared layout (with any swizzle) and the operand role stay in typed
/// attributes. Backend lowering computes each lane's row address through the
/// composed byte map and selects the target load, storing the returned values
/// into the fragment slot. Operands 1..=2 are `u64`; operand 3 is `u32`.
#[pliron_op(
    name = "cute.ldmatrix",
    format,
    interfaces = [NOpdsInterface<5>, NResultsInterface<0>],
    attributes = (
        matrix_role: CuteMatrixRoleAttr,
        ldmatrix_smem_layout: CuteComposedLayoutAttr,
        ldmatrix_elem: TypeAttr
    )
)]
pub struct CuteLdmatrixOp;

impl CuteLdmatrixOp {
    pub fn new(
        ctx: &mut Context,
        operands: [Value; 5],
        role: CuteMatrixRoleAttr,
        smem_layout: ComposedLayout,
        elem: TypeHandle,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            operands.to_vec(),
            vec![],
            0,
        );
        let load = Self { op };
        load.set_attr_matrix_role(ctx, role);
        load.set_attr_ldmatrix_smem_layout(ctx, CuteComposedLayoutAttr(smem_layout));
        load.set_attr_ldmatrix_elem(ctx, TypeAttr::new(elem));
        load
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for CuteLdmatrixOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let Some(role) = self.get_attr_matrix_role(ctx).map(|attr| *attr) else {
            return verify_err!(op.loc(), "cute.ldmatrix must have a matrix role");
        };
        let Some(smem_layout) = self
            .get_attr_ldmatrix_smem_layout(ctx)
            .map(|attr| attr.0.clone())
        else {
            return verify_err!(op.loc(), "cute.ldmatrix must have a shared-memory layout");
        };
        let Some(elem): Option<TypeHandle> = self
            .get_attr_ldmatrix_elem(ctx)
            .map(|attr| attr.clone().into())
        else {
            return verify_err!(op.loc(), "cute.ldmatrix must have an element type");
        };
        if elem.deref(ctx).downcast_ref::<MirFP16Type>().is_none() {
            return verify_err!(op.loc(), "cute.ldmatrix v0 supports f16 elements only");
        }
        let _ = role;
        let modes = smem_layout.inner().modes();
        if modes.len() != 2 {
            return verify_err!(op.loc(), "cute.ldmatrix shared layout needs two modes");
        }
        let (Some(rows), Some(columns)) = (modes[0].checked_size(), modes[1].checked_size()) else {
            return verify_err!(op.loc(), "cute.ldmatrix shared layout extents are invalid");
        };
        let byte_layout = match smem_layout.to_byte_offsets(2) {
            Ok(layout) => layout,
            Err(error) => {
                return verify_err!(
                    op.loc(),
                    "cute.ldmatrix shared layout has no byte form: {error}"
                );
            }
        };
        if let Err(error) = cute_layout::validate_ldmatrix_source(&byte_layout, rows, columns) {
            return verify_err!(op.loc(), "cute.ldmatrix source is not loadable: {error}");
        }

        let smem_ty = op.get_operand(0).get_type(ctx);
        let smem_ref = smem_ty.deref(ctx);
        let Some(pointer) = smem_ref.downcast_ref::<MirPtrType>() else {
            return verify_err!(op.loc(), "cute.ldmatrix source must be a MIR pointer");
        };
        if pointer.pointee != elem
            || ![address_space::GENERIC, address_space::SHARED].contains(&pointer.address_space)
        {
            return verify_err!(
                op.loc(),
                "cute.ldmatrix source must point at shared/generic f16"
            );
        }
        for (index, role_name) in [(1usize, "warp tile row"), (2usize, "warp tile column")] {
            let ty = op.get_operand(index).get_type(ctx);
            if ty
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .is_none_or(|integer| integer.width() != 64 || !integer.is_unsigned())
            {
                return verify_err!(op.loc(), "{role_name} must be an unsigned 64-bit integer");
            }
        }
        let lane_ty = op.get_operand(3).get_type(ctx);
        if lane_ty
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .is_none_or(|integer| integer.width() != 32 || !integer.is_unsigned())
        {
            return verify_err!(op.loc(), "lane id must be an unsigned 32-bit integer");
        }
        let slot_ty = op.get_operand(4).get_type(ctx);
        let slot_ref = slot_ty.deref(ctx);
        let Some(slot) = slot_ref.downcast_ref::<MirPtrType>() else {
            return verify_err!(op.loc(), "fragment slot must be a MIR pointer");
        };
        if !slot.is_mutable {
            return verify_err!(op.loc(), "fragment slot must be mutable");
        }
        Ok(())
    }
}

/// Frozen zero-result ABI for one hardware (TMA) tile copy into CTA-local
/// shared memory.
///
/// ```text
/// 0 smem base (T)    2 tensor map        4 tile column index
/// 1 mbarrier         3 tile row index
/// ```
///
/// The shared layout stays in a typed attribute so the compiler can prove
/// TMA encodability at build time; the descriptor itself carries the same
/// layout in its Rust TYPE, so pairing errors never reach this op.
/// Barrier lifecycle (init, arrive_expect_tx, wait) stays caller-owned.
/// Remote or multicast cluster-shared copies require a distinct operation;
/// accepting a generic destination here would erase the proof needed to
/// select PTX's CTA-local TMA form.
#[pliron_op(
    name = "cute.copy_tma_2d",
    format,
    interfaces = [NOpdsInterface<5>, NResultsInterface<0>],
    attributes = (
        tma_smem_layout: CuteComposedLayoutAttr,
        tma_elem: TypeAttr
    )
)]
pub struct CuteTmaLoad2dOp;

impl CuteTmaLoad2dOp {
    pub fn new(
        ctx: &mut Context,
        operands: [Value; 5],
        smem_layout: ComposedLayout,
        elem: TypeHandle,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            operands.to_vec(),
            vec![],
            0,
        );
        let load = Self { op };
        load.set_attr_tma_smem_layout(ctx, CuteComposedLayoutAttr(smem_layout));
        load.set_attr_tma_elem(ctx, TypeAttr::new(elem));
        load
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for CuteTmaLoad2dOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        if op.get_num_operands() != 5 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_2d expects exactly 5 operands and 0 results"
            );
        }
        let Some(smem_layout) = self
            .get_attr_tma_smem_layout(ctx)
            .map(|attr| attr.0.clone())
        else {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_2d must have a shared-memory layout"
            );
        };
        let Some(elem): Option<TypeHandle> =
            self.get_attr_tma_elem(ctx).map(|attr| attr.clone().into())
        else {
            return verify_err!(op.loc(), "cute.copy_tma_2d must have an element type");
        };
        let Some(elem_bytes) = cooperative_elem_bytes(ctx, elem) else {
            return verify_err!(op.loc(), "unsupported cute.copy_tma_2d element type");
        };
        if let Err(error) = cute_layout::validate_tma_encodable(&smem_layout, elem_bytes) {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_2d layout is not encodable: {error}"
            );
        }
        let smem_ty = op.get_operand(0).get_type(ctx);
        let smem_ref = smem_ty.deref(ctx);
        let Some(pointer) = smem_ref.downcast_ref::<MirPtrType>() else {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_2d destination must be a MIR pointer"
            );
        };
        if pointer.pointee != elem
            || !pointer.is_mutable
            || pointer.address_space != address_space::SHARED
        {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_2d destination must be mutable CTA-shared elem memory"
            );
        }
        for index in [1usize, 2] {
            let ty = op.get_operand(index).get_type(ctx);
            if ty.deref(ctx).downcast_ref::<MirPtrType>().is_none() {
                return verify_err!(
                    op.loc(),
                    "cute.copy_tma_2d operand {index} must be a pointer"
                );
            }
        }
        for (index, what) in [(3usize, "tile row"), (4usize, "tile column")] {
            let ty = op.get_operand(index).get_type(ctx);
            if ty
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .is_none_or(|integer| integer.width() != 64 || !integer.is_unsigned())
            {
                return verify_err!(op.loc(), "{what} must be an unsigned 64-bit integer");
            }
        }
        Ok(())
    }
}

/// Frozen zero-result ABI for one hardware (TMA) tile copy from CTA-local
/// shared memory into global memory.
///
/// ```text
/// 0 smem source (T)   2 tile row index
/// 1 tensor map        3 tile column index
/// ```
///
/// The shared layout and element type remain typed attributes. The descriptor
/// carries the same witnesses at the Rust boundary, while this operation
/// preserves them for the selected backend continuation.
#[pliron_op(
    name = "cute.copy_tma_s2g_2d",
    format,
    interfaces = [NOpdsInterface<4>, NResultsInterface<0>],
    attributes = (
        tma_store_smem_layout: CuteComposedLayoutAttr,
        tma_store_elem: TypeAttr
    )
)]
pub struct CuteTmaStore2dOp;

impl CuteTmaStore2dOp {
    pub fn new(
        ctx: &mut Context,
        operands: [Value; 4],
        smem_layout: ComposedLayout,
        elem: TypeHandle,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            operands.to_vec(),
            vec![],
            0,
        );
        let store = Self { op };
        store.set_attr_tma_store_smem_layout(ctx, CuteComposedLayoutAttr(smem_layout));
        store.set_attr_tma_store_elem(ctx, TypeAttr::new(elem));
        store
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for CuteTmaStore2dOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        if op.get_num_operands() != 4 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_s2g_2d expects exactly 4 operands and 0 results"
            );
        }
        let Some(smem_layout) = self
            .get_attr_tma_store_smem_layout(ctx)
            .map(|attr| attr.0.clone())
        else {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_s2g_2d must have a shared-memory layout"
            );
        };
        let Some(elem): Option<TypeHandle> = self
            .get_attr_tma_store_elem(ctx)
            .map(|attr| attr.clone().into())
        else {
            return verify_err!(op.loc(), "cute.copy_tma_s2g_2d must have an element type");
        };
        let Some(elem_bytes) = cooperative_elem_bytes(ctx, elem) else {
            return verify_err!(op.loc(), "unsupported cute.copy_tma_s2g_2d element type");
        };
        if let Err(error) = cute_layout::validate_tma_encodable(&smem_layout, elem_bytes) {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_s2g_2d layout is not encodable: {error}"
            );
        }
        let smem_ty = op.get_operand(0).get_type(ctx);
        let smem_ref = smem_ty.deref(ctx);
        let Some(pointer) = smem_ref.downcast_ref::<MirPtrType>() else {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_s2g_2d source must be a MIR pointer"
            );
        };
        if pointer.pointee != elem || pointer.address_space != address_space::SHARED {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_s2g_2d source must be CTA-shared elem memory"
            );
        }
        let tensor_map_ty = op.get_operand(1).get_type(ctx);
        if tensor_map_ty
            .deref(ctx)
            .downcast_ref::<MirPtrType>()
            .is_none()
        {
            return verify_err!(
                op.loc(),
                "cute.copy_tma_s2g_2d tensor map must be a pointer"
            );
        }
        for (index, what) in [(2usize, "tile row"), (3usize, "tile column")] {
            let ty = op.get_operand(index).get_type(ctx);
            if ty
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .is_none_or(|integer| integer.width() != 64 || !integer.is_unsigned())
            {
                return verify_err!(op.loc(), "{what} must be an unsigned 64-bit integer");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cute_layout::{IntTuple, MAX_STATIC_VALIDATION_BYTES, OffsetUnit, Swizzle};
    use dialect_mir::ops::MirUndefOp;
    use pliron::builtin::types::{FP32Type, Signedness};

    fn undef(ctx: &mut Context, ty: TypeHandle) -> Value {
        MirUndefOp::new(ctx, ty)
            .get_operation()
            .deref(ctx)
            .get_result(0)
    }

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
            ComposedLayout::new(Swizzle::new(3, 4, 3), 8, tile.clone(), OffsetUnit::Elements)
                .unwrap();
        (thread, value, tile, smem)
    }

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

    fn verifier_error(copy: &CuteCopyG2SOp, ctx: &Context, expected: &str) {
        let error = copy.verify(ctx).unwrap_err().to_string();
        assert!(
            error.contains(expected),
            "expected `{expected}` in `{error}`"
        );
    }

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
        let smem =
            ComposedLayout::from_layout("(4,4):(4,1)".parse().unwrap(), OffsetUnit::Elements);
        CuteTmaLoad2dOp::new(ctx, operands, smem, elem)
    }

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
        let smem =
            ComposedLayout::from_layout("(4,4):(4,1)".parse().unwrap(), OffsetUnit::Elements);
        CuteTmaStore2dOp::new(ctx, operands, smem, elem)
    }

    #[test]
    fn custom_verifiers_check_cardinality_before_indexing() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        crate::register(&mut ctx);

        let malformed =
            |ctx: &mut Context, info| Operation::new(ctx, info, vec![], vec![], vec![], 0);

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

    #[test]
    fn cooperative_abi_rejects_integer_width_or_signedness_drift() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        crate::register(&mut ctx);

        let make =
            |ctx: &mut Context, usize_width, usize_signedness, tidx_width, tidx_signedness| {
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
}
