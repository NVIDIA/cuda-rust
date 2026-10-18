/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Direct, cooperative, descriptor-backed, and cache-aware copies.

mod global;
mod tma;

pub use global::{CuteGlobalStoreAttr, CuteStoreGlobalNoAllocateOp, global_store_element_bytes};
pub use tma::{CuteTmaLoad2dOp, CuteTmaStore2dOp};

use cute_layout::{ComposedLayout, Layout, validate_cooperative_copy_plan};
use dialect_mir::types::{MirFP16Type, MirPtrType, address_space};
use pliron::{
    builtin::{
        attributes::TypeAttr,
        op_interfaces::{NOpdsInterface, NResultsInterface},
        types::IntegerType,
    },
    common_traits::Verify,
    context::{Context, Ptr},
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
    /// Build a detached cooperative copy with validated-layout metadata and explicit pitch/index carriers.
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

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Return the supported scalar element width used by cooperative-copy validation.
pub(super) fn cooperative_elem_bytes(ctx: &Context, elem: TypeHandle) -> Option<i64> {
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
