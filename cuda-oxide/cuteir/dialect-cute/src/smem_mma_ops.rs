/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared-tensor and tiled-MMA meaning for the GEMM compute loop.
//!
//! Runtime values stay exactly as they are in MIR today. Typed attributes say
//! what those values mean:
//!
//! ```text
//! shared pointer + capacity ── overlay ─────────────────────────────┐
//!                                                                  │
//! lane ── tiled_mma_slice ──────────────────────────────────────┐   │
//!                                                              │   │
//! scales A/B ── load_scales ── slice K=64 ──────────────────┐   │   │
//! A shared tile ─────────────── load_a ───────────────────┐  │   │   │
//! B shared tile ─────────────── partition_b ───────────┐  │  │   │   │
//! zero / previous C ───────────────────────────────────┴──┴──┴───┴───┘
//!                                                       │
//!                                                       ▼
//!                                                  tiled_gemm
//! ```
//!
//! `partition_b` is intentionally lazy. It returns only B's shared pointer,
//! capacity, warp-N position, and K half. `tiled_gemm` can therefore load two
//! B fragments, use them immediately, and then load the next pair. No eager
//! eight-fragment B aggregate exists in this schema.
//!
//! Rust may temporarily store a shared view inside an ordinary aggregate.
//! Consumers therefore repeat its static `!cute.smem_tensor` TypeAttr instead
//! of requiring their pointer/capacity operands to come directly from the
//! overlay operation. Normal and no-inline MIR can flatten to the same ABI.
//!
//! Backend continuations consume the same ordinary carriers: they can erase
//! forwarding operations into native loads and MMA sites or group the values
//! into CuTe MLIR tensors and fragments.

use dialect_mir::types::{MirFP16Type, MirPtrType, address_space};
use pliron::builtin::{
    attributes::TypeAttr,
    op_interfaces::{NOpdsInterface, NResultsInterface, OneResultInterface},
    types::{FP32Type, IntegerType},
};
use pliron::common_traits::Verify;
use pliron::context::{Context, Ptr};
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Error;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use pliron::verify_err;
use pliron_derive::pliron_op;

use crate::attributes::{
    CuteMmaCarrierKindAttr, CuteTensorFormatAttr, CuteTensorLayoutAttr, CuteTensorRoleAttr,
    CuteTiledMmaPlanAttr, SM1XX_BLOCK_SCALE_ATOM_BYTES,
};
use crate::layout::{ComposedLayout, Layout, OffsetUnit, Swizzle};
use crate::types::{CuteSmemTensorType, MmaCarrierShape, mma_carrier_shape};

fn is_u32(ctx: &Context, value: Value) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 32 && integer.is_unsigned())
}

fn is_u64(ctx: &Context, value: Value) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 64 && integer.is_unsigned())
}

fn is_f32(ctx: &Context, value: Value) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<FP32Type>()
        .is_some()
}

fn smem_tensor_of_type(ctx: &Context, ty: TypeHandle) -> Option<CuteSmemTensorType> {
    ty.deref(ctx).downcast_ref::<CuteSmemTensorType>().cloned()
}

fn shared_pointer_matches(ctx: &Context, value: Value, storage: TypeHandle) -> bool {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .is_some_and(|pointer| {
            pointer.pointee == storage
                && pointer.is_mutable
                && pointer.address_space == address_space::SHARED
        })
}

fn shared_carriers_match_view(
    ctx: &Context,
    base: Value,
    capacity: Value,
    view: &CuteSmemTensorType,
) -> bool {
    view.tensor_view(ctx).is_some_and(|tensor| {
        shared_pointer_matches(ctx, base, tensor.storage) && is_u64(ctx, capacity)
    })
}

fn canonical_scale_placement() -> ComposedLayout {
    let inner: Layout = "(32,4):(4,1)"
        .parse()
        .expect("fixed SM120 scale placement is valid");
    ComposedLayout::new(Swizzle::IDENTITY, 0, inner, OffsetUnit::Elements)
        .expect("identity scale placement is valid")
}

fn canonical_data_placement() -> ComposedLayout {
    let inner: Layout = "(128,32):(32,1)"
        .parse()
        .expect("fixed SM120 data placement is valid");
    ComposedLayout::new(Swizzle::new(2, 3, 3), 0, inner, OffsetUnit::Elements)
        .expect("fixed SM120 data swizzle is valid")
}

fn plan_of_attr(
    plan: Option<std::cell::Ref<'_, CuteTiledMmaPlanAttr>>,
) -> Option<CuteTiledMmaPlanAttr> {
    plan.map(|plan| plan.clone())
}

fn check_plan(plan: &CuteTiledMmaPlanAttr, ctx: &Context) -> Result<(), &'static str> {
    plan.verify(ctx)
        .map_err(|_| "has an invalid tiled-MMA plan")?;
    if plan.atom != crate::attributes::CuteMmaAtomAttr::mxf4_m16n8k64()
        || plan.cta_m != 128
        || plan.cta_n != 128
        || plan.cta_k != 128
        || plan.warp_m != 4
        || plan.warp_n != 2
        || plan.b_load_group != 2
        || plan.shared_layout.0 != canonical_data_placement()
    {
        return Err("v0 supports the 128x128x128 SM120 MXFP4 plan");
    }
    let canonical = CuteTiledMmaPlanAttr::mxf4_128x128x128(plan.shared_layout.0.clone());
    if plan.m_ownership != canonical.m_ownership || plan.n_ownership != canonical.n_ownership {
        return Err("v0 needs the SM120 2x8 warp-cell ownership maps");
    }
    Ok(())
}

fn check_data_view(
    ctx: &Context,
    view: &CuteSmemTensorType,
    plan: &CuteTiledMmaPlanAttr,
    role: CuteTensorRoleAttr,
) -> Result<(), &'static str> {
    view.verify(ctx)
        .map_err(|_| "uses an invalid shared tensor type")?;
    let tensor = view.tensor_view(ctx).ok_or("must wrap a tensor view")?;
    if tensor.format != CuteTensorFormatAttr::E2M1
        || tensor.role != role
        || tensor.layout != CuteTensorLayoutAttr::KMajor
        || tensor.alignment.0 < 16
        || tensor
            .storage
            .deref(ctx)
            .downcast_ref::<MirFP16Type>()
            .is_none()
        || view.placement != plan.shared_layout
        || view.storage_elements() != Some(4096)
        || view.storage_bytes(ctx) != Some(8192)
    {
        return Err("must be the matching 128x128 packed-E2M1 shared tile");
    }
    Ok(())
}

fn check_scale_view(
    ctx: &Context,
    view: &CuteSmemTensorType,
    plan: &CuteTiledMmaPlanAttr,
    role: CuteTensorRoleAttr,
) -> Result<(), &'static str> {
    view.verify(ctx)
        .map_err(|_| "uses an invalid shared tensor type")?;
    let tensor = view.tensor_view(ctx).ok_or("must wrap a tensor view")?;
    let storage_is_u32 = tensor
        .storage
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 32 && integer.is_unsigned());
    if tensor.format != CuteTensorFormatAttr::UE8M0
        || tensor.role != role
        || tensor.layout != CuteTensorLayoutAttr::BlockScaleKMajor(plan.atom.values_per_scale)
        || tensor.alignment.0 < 4
        || !storage_is_u32
        || view.placement.0 != canonical_scale_placement()
        || view.storage_elements() != Some(SM1XX_BLOCK_SCALE_ATOM_BYTES / 4)
        || view.storage_bytes(ctx) != Some(SM1XX_BLOCK_SCALE_ATOM_BYTES)
    {
        return Err("must be the matching 128-row canonical UE8M0 shared scale atom");
    }
    Ok(())
}

fn check_carrier(
    ctx: &Context,
    value: Value,
    expected: MmaCarrierShape,
) -> Result<(), &'static str> {
    if mma_carrier_shape(ctx, value.get_type(ctx)) == Some(expected) {
        Ok(())
    } else {
        Err("has the wrong ordinary MIR carrier shape")
    }
}

/// Give a shared pointer/capacity pair typed tensor meaning without changing
/// either runtime value.
///
/// The `smem_overlay_view` attribute is a `!cute.smem_tensor` type. Results
/// have exactly the operand types, so backend lowering may erase this
/// operation to two identity replacements.
#[pliron_op(
    name = "cute.smem_tensor_overlay",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<2>],
    attributes = (smem_overlay_view: TypeAttr)
)]
pub struct CuteSmemTensorOverlayOp;

impl CuteSmemTensorOverlayOp {
    pub fn new(ctx: &mut Context, base: Value, capacity: Value, view: TypeHandle) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![base.get_type(ctx), capacity.get_type(ctx)],
                vec![base, capacity],
                vec![],
                0,
            ),
        };
        operation.set_attr_smem_overlay_view(ctx, TypeAttr::new(view));
        operation
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn input_base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    #[must_use]
    pub fn input_capacity(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    #[must_use]
    pub fn base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    #[must_use]
    pub fn capacity(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(1)
    }

    /// Static logical/storage/role/layout facts attached to this pair.
    #[must_use]
    pub fn view(&self, ctx: &Context) -> Option<CuteSmemTensorType> {
        let attr = self.get_attr_smem_overlay_view(ctx)?;
        smem_tensor_of_type(ctx, attr.get_type(ctx))
    }
}

impl Verify for CuteSmemTensorOverlayOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 2 {
            return verify_err!(
                op.loc(),
                "cute.smem_tensor_overlay needs 2 operands and 2 results"
            );
        }
        let Some(view) = self.view(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.smem_tensor_overlay must carry a cute.smem_tensor type"
            );
        };
        view.verify(ctx)?;
        let tensor = view.tensor_view(ctx).expect("verified shared tensor");
        if !shared_pointer_matches(ctx, self.input_base(ctx), tensor.storage) {
            return verify_err!(
                op.loc(),
                "cute.smem_tensor_overlay base must be a mutable CTA-shared storage pointer"
            );
        }
        if !is_u64(ctx, self.input_capacity(ctx)) {
            return verify_err!(
                op.loc(),
                "cute.smem_tensor_overlay capacity must be an unsigned 64-bit integer"
            );
        }
        if self.base(ctx).get_type(ctx) != self.input_base(ctx).get_type(ctx)
            || self.capacity(ctx).get_type(ctx) != self.input_capacity(ctx).get_type(ctx)
        {
            return verify_err!(
                op.loc(),
                "cute.smem_tensor_overlay results must preserve the pointer and capacity types"
            );
        }
        Ok(())
    }
}

/// Attach one tiled-MMA plan to the calling lane number.
///
/// The result remains an ordinary `u32`; it may safely cross loop edges.
#[pliron_op(
    name = "cute.tiled_mma_slice",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>, OneResultInterface],
    attributes = (tiled_mma_slice_plan: CuteTiledMmaPlanAttr)
)]
pub struct CuteTiledMmaSliceOp;

impl CuteTiledMmaSliceOp {
    pub fn new(ctx: &mut Context, lane: Value, plan: CuteTiledMmaPlanAttr) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![lane.get_type(ctx)],
                vec![lane],
                vec![],
                0,
            ),
        };
        operation.set_attr_tiled_mma_slice_plan(ctx, plan);
        operation
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn lane(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    #[must_use]
    pub fn sliced_lane(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    #[must_use]
    pub fn plan(&self, ctx: &Context) -> Option<CuteTiledMmaPlanAttr> {
        plan_of_attr(self.get_attr_tiled_mma_slice_plan(ctx))
    }
}

impl Verify for CuteTiledMmaSliceOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.tiled_mma_slice needs 1 operand and 1 result"
            );
        }
        if !is_u32(ctx, self.lane(ctx))
            || self.sliced_lane(ctx).get_type(ctx) != self.lane(ctx).get_type(ctx)
        {
            return verify_err!(
                op.loc(),
                "cute.tiled_mma_slice must preserve one unsigned u32 lane"
            );
        }
        let Some(plan) = self.plan(ctx) else {
            return verify_err!(op.loc(), "cute.tiled_mma_slice must carry a plan");
        };
        if let Err(message) = check_plan(&plan, ctx) {
            return verify_err!(op.loc(), "cute.tiled_mma_slice {message}");
        }
        Ok(())
    }
}

/// Fill one physical register carrier while retaining its fragment meaning.
///
/// V0 uses this for the 64-FP32 accumulator carrier only.
#[pliron_op(
    name = "cute.fragment_fill",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>, OneResultInterface],
    attributes = (
        fragment_fill_plan: CuteTiledMmaPlanAttr,
        fragment_fill_kind: CuteMmaCarrierKindAttr
    )
)]
pub struct CuteFragmentFillOp;

impl CuteFragmentFillOp {
    pub fn new(
        ctx: &mut Context,
        fill: Value,
        result_type: TypeHandle,
        plan: CuteTiledMmaPlanAttr,
        kind: CuteMmaCarrierKindAttr,
    ) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result_type],
                vec![fill],
                vec![],
                0,
            ),
        };
        operation.set_attr_fragment_fill_plan(ctx, plan);
        operation.set_attr_fragment_fill_kind(ctx, kind);
        operation
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn fill(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    #[must_use]
    pub fn fragment(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    #[must_use]
    pub fn plan(&self, ctx: &Context) -> Option<CuteTiledMmaPlanAttr> {
        plan_of_attr(self.get_attr_fragment_fill_plan(ctx))
    }

    #[must_use]
    pub fn kind(&self, ctx: &Context) -> Option<CuteMmaCarrierKindAttr> {
        self.get_attr_fragment_fill_kind(ctx).map(|kind| *kind)
    }
}

impl Verify for CuteFragmentFillOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(op.loc(), "cute.fragment_fill needs 1 operand and 1 result");
        }
        let Some(plan) = self.plan(ctx) else {
            return verify_err!(op.loc(), "cute.fragment_fill must carry a plan");
        };
        if let Err(message) = check_plan(&plan, ctx) {
            return verify_err!(op.loc(), "cute.fragment_fill {message}");
        }
        if self.kind(ctx) != Some(CuteMmaCarrierKindAttr::Accumulator)
            || !is_f32(ctx, self.fill(ctx))
        {
            return verify_err!(
                op.loc(),
                "cute.fragment_fill v0 fills an accumulator from one f32 value"
            );
        }
        let expected = MmaCarrierShape::f32(
            plan.accumulator_registers_per_lane()
                .expect("verified plan has an accumulator size"),
        );
        if let Err(message) = check_carrier(ctx, self.fragment(ctx), expected) {
            return verify_err!(op.loc(), "cute.fragment_fill result {message}");
        }
        Ok(())
    }
}

/// Load the ten packed scale words used by one warp for a K=128 stage.
#[pliron_op(
    name = "cute.mma_load_scales",
    format,
    interfaces = [NOpdsInterface<7>, NResultsInterface<1>, OneResultInterface],
    attributes = (
        mma_load_scales_plan: CuteTiledMmaPlanAttr,
        mma_load_scales_a_view: TypeAttr,
        mma_load_scales_b_view: TypeAttr
    )
)]
pub struct CuteMmaLoadScalesOp;

impl CuteMmaLoadScalesOp {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &mut Context,
        lane: Value,
        scale_a_base: Value,
        scale_a_capacity: Value,
        scale_b_base: Value,
        scale_b_capacity: Value,
        warp_m: Value,
        warp_n: Value,
        scale_a_view: TypeHandle,
        scale_b_view: TypeHandle,
        result_type: TypeHandle,
        plan: CuteTiledMmaPlanAttr,
    ) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result_type],
                vec![
                    lane,
                    scale_a_base,
                    scale_a_capacity,
                    scale_b_base,
                    scale_b_capacity,
                    warp_m,
                    warp_n,
                ],
                vec![],
                0,
            ),
        };
        operation.set_attr_mma_load_scales_plan(ctx, plan);
        operation.set_attr_mma_load_scales_a_view(ctx, TypeAttr::new(scale_a_view));
        operation.set_attr_mma_load_scales_b_view(ctx, TypeAttr::new(scale_b_view));
        operation
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn plan(&self, ctx: &Context) -> Option<CuteTiledMmaPlanAttr> {
        plan_of_attr(self.get_attr_mma_load_scales_plan(ctx))
    }

    /// Static M-role scale tensor facts for operands 1 and 2.
    #[must_use]
    pub fn scale_a_view(&self, ctx: &Context) -> Option<CuteSmemTensorType> {
        let attr = self.get_attr_mma_load_scales_a_view(ctx)?;
        smem_tensor_of_type(ctx, attr.get_type(ctx))
    }

    /// Static N-role scale tensor facts for operands 3 and 4.
    #[must_use]
    pub fn scale_b_view(&self, ctx: &Context) -> Option<CuteSmemTensorType> {
        let attr = self.get_attr_mma_load_scales_b_view(ctx)?;
        smem_tensor_of_type(ctx, attr.get_type(ctx))
    }

    #[must_use]
    pub fn scales(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }
}

impl Verify for CuteMmaLoadScalesOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 7 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.mma_load_scales needs 7 operands and 1 result"
            );
        }
        let Some(plan) = self.plan(ctx) else {
            return verify_err!(op.loc(), "cute.mma_load_scales must carry a plan");
        };
        if let Err(message) = check_plan(&plan, ctx) {
            return verify_err!(op.loc(), "cute.mma_load_scales {message}");
        }
        if !is_u32(ctx, op.get_operand(0))
            || !is_u64(ctx, op.get_operand(5))
            || !is_u64(ctx, op.get_operand(6))
        {
            return verify_err!(
                op.loc(),
                "cute.mma_load_scales needs a u32 lane and u64 warp-M/warp-N positions"
            );
        }
        let Some(scale_a) = self.scale_a_view(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.mma_load_scales must carry an M-role shared scale view"
            );
        };
        let Some(scale_b) = self.scale_b_view(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.mma_load_scales must carry an N-role shared scale view"
            );
        };
        if let Err(message) = check_scale_view(ctx, &scale_a, &plan, CuteTensorRoleAttr::Mkl) {
            return verify_err!(op.loc(), "cute.mma_load_scales A {message}");
        }
        if let Err(message) = check_scale_view(ctx, &scale_b, &plan, CuteTensorRoleAttr::Nkl) {
            return verify_err!(op.loc(), "cute.mma_load_scales B {message}");
        }
        if !shared_carriers_match_view(ctx, op.get_operand(1), op.get_operand(2), &scale_a)
            || !shared_carriers_match_view(ctx, op.get_operand(3), op.get_operand(4), &scale_b)
        {
            return verify_err!(
                op.loc(),
                "cute.mma_load_scales pointer/capacity carriers must match their typed scale views"
            );
        }
        let expected = MmaCarrierShape::u32(
            plan.scale_words_per_lane()
                .expect("verified plan has a scale carrier size"),
        );
        if let Err(message) = check_carrier(ctx, self.scales(ctx), expected) {
            return verify_err!(op.loc(), "cute.mma_load_scales result {message}");
        }
        Ok(())
    }
}

/// Select the low or high K=64 scale pair from every packed K=128 word.
#[pliron_op(
    name = "cute.fragment_slice_k",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface],
    attributes = (fragment_slice_k_plan: CuteTiledMmaPlanAttr)
)]
pub struct CuteFragmentSliceKOp;

impl CuteFragmentSliceKOp {
    pub fn new(
        ctx: &mut Context,
        stage_scales: Value,
        k_half: Value,
        result_type: TypeHandle,
        plan: CuteTiledMmaPlanAttr,
    ) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result_type],
                vec![stage_scales, k_half],
                vec![],
                0,
            ),
        };
        operation.set_attr_fragment_slice_k_plan(ctx, plan);
        operation
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn plan(&self, ctx: &Context) -> Option<CuteTiledMmaPlanAttr> {
        plan_of_attr(self.get_attr_fragment_slice_k_plan(ctx))
    }

    #[must_use]
    pub fn scales(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }
}

impl Verify for CuteFragmentSliceKOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.fragment_slice_k needs 2 operands and 1 result"
            );
        }
        let Some(plan) = self.plan(ctx) else {
            return verify_err!(op.loc(), "cute.fragment_slice_k must carry a plan");
        };
        if let Err(message) = check_plan(&plan, ctx) {
            return verify_err!(op.loc(), "cute.fragment_slice_k {message}");
        }
        if plan.cta_k != plan.atom.k * 2 || !is_u64(ctx, op.get_operand(1)) {
            return verify_err!(
                op.loc(),
                "cute.fragment_slice_k v0 selects one u64-indexed K=64 half of K=128"
            );
        }
        let expected = MmaCarrierShape::u32(
            plan.scale_words_per_lane()
                .expect("verified plan has a scale carrier size"),
        );
        if let Err(message) = check_carrier(ctx, op.get_operand(0), expected) {
            return verify_err!(op.loc(), "cute.fragment_slice_k input {message}");
        }
        if let Err(message) = check_carrier(ctx, self.scales(ctx), expected) {
            return verify_err!(op.loc(), "cute.fragment_slice_k result {message}");
        }
        Ok(())
    }
}

/// Load the two A fragments owned by one warp for one K=64 step.
#[pliron_op(
    name = "cute.mma_load_a",
    format,
    interfaces = [NOpdsInterface<5>, NResultsInterface<1>, OneResultInterface],
    attributes = (
        mma_load_a_plan: CuteTiledMmaPlanAttr,
        mma_load_a_view: TypeAttr
    )
)]
pub struct CuteMmaLoadAOp;

impl CuteMmaLoadAOp {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &mut Context,
        lane: Value,
        base: Value,
        capacity: Value,
        warp_m: Value,
        k_half: Value,
        view: TypeHandle,
        result_type: TypeHandle,
        plan: CuteTiledMmaPlanAttr,
    ) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result_type],
                vec![lane, base, capacity, warp_m, k_half],
                vec![],
                0,
            ),
        };
        operation.set_attr_mma_load_a_plan(ctx, plan);
        operation.set_attr_mma_load_a_view(ctx, TypeAttr::new(view));
        operation
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn plan(&self, ctx: &Context) -> Option<CuteTiledMmaPlanAttr> {
        plan_of_attr(self.get_attr_mma_load_a_plan(ctx))
    }

    /// Static M-role shared tensor facts for the pointer/capacity operands.
    #[must_use]
    pub fn view(&self, ctx: &Context) -> Option<CuteSmemTensorType> {
        let attr = self.get_attr_mma_load_a_view(ctx)?;
        smem_tensor_of_type(ctx, attr.get_type(ctx))
    }

    #[must_use]
    pub fn fragment(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }
}

impl Verify for CuteMmaLoadAOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 5 || op.get_num_results() != 1 {
            return verify_err!(op.loc(), "cute.mma_load_a needs 5 operands and 1 result");
        }
        let Some(plan) = self.plan(ctx) else {
            return verify_err!(op.loc(), "cute.mma_load_a must carry a plan");
        };
        if let Err(message) = check_plan(&plan, ctx) {
            return verify_err!(op.loc(), "cute.mma_load_a {message}");
        }
        if !is_u32(ctx, op.get_operand(0))
            || !is_u64(ctx, op.get_operand(3))
            || !is_u64(ctx, op.get_operand(4))
        {
            return verify_err!(
                op.loc(),
                "cute.mma_load_a needs a u32 lane and u64 warp-M/K-half positions"
            );
        }
        let Some(view) = self.view(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.mma_load_a must carry an M-role shared value view"
            );
        };
        if let Err(message) = check_data_view(ctx, &view, &plan, CuteTensorRoleAttr::Mkl) {
            return verify_err!(op.loc(), "cute.mma_load_a input {message}");
        }
        if !shared_carriers_match_view(ctx, op.get_operand(1), op.get_operand(2), &view) {
            return verify_err!(
                op.loc(),
                "cute.mma_load_a pointer/capacity carriers must match its typed A view"
            );
        }
        let expected = MmaCarrierShape::u32(
            plan.a_registers_per_lane()
                .expect("verified plan has an A carrier size"),
        );
        if let Err(message) = check_carrier(ctx, self.fragment(ctx), expected) {
            return verify_err!(op.loc(), "cute.mma_load_a result {message}");
        }
        Ok(())
    }
}

/// Keep B as a lazy shared-memory selection.
///
/// All four results preserve ordinary operand types. In particular, this op
/// has no register-fragment result and performs no `ldmatrix` load.
#[pliron_op(
    name = "cute.mma_partition_b",
    format,
    interfaces = [NOpdsInterface<4>, NResultsInterface<4>],
    attributes = (
        mma_partition_b_plan: CuteTiledMmaPlanAttr,
        mma_partition_b_view: TypeAttr
    )
)]
pub struct CuteMmaPartitionBOp;

impl CuteMmaPartitionBOp {
    pub fn new(
        ctx: &mut Context,
        base: Value,
        capacity: Value,
        warp_n: Value,
        k_half: Value,
        view: TypeHandle,
        plan: CuteTiledMmaPlanAttr,
    ) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![
                    base.get_type(ctx),
                    capacity.get_type(ctx),
                    warp_n.get_type(ctx),
                    k_half.get_type(ctx),
                ],
                vec![base, capacity, warp_n, k_half],
                vec![],
                0,
            ),
        };
        operation.set_attr_mma_partition_b_plan(ctx, plan);
        operation.set_attr_mma_partition_b_view(ctx, TypeAttr::new(view));
        operation
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn plan(&self, ctx: &Context) -> Option<CuteTiledMmaPlanAttr> {
        plan_of_attr(self.get_attr_mma_partition_b_plan(ctx))
    }

    /// Static N-role shared tensor facts for the lazy B selection.
    #[must_use]
    pub fn view(&self, ctx: &Context) -> Option<CuteSmemTensorType> {
        let attr = self.get_attr_mma_partition_b_view(ctx)?;
        smem_tensor_of_type(ctx, attr.get_type(ctx))
    }

    #[must_use]
    pub fn base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    #[must_use]
    pub fn capacity(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(1)
    }

    #[must_use]
    pub fn warp_n(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(2)
    }

    #[must_use]
    pub fn k_half(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(3)
    }
}

impl Verify for CuteMmaPartitionBOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 4 || op.get_num_results() != 4 {
            return verify_err!(
                op.loc(),
                "cute.mma_partition_b needs 4 operands and exactly 4 scalar results"
            );
        }
        let Some(plan) = self.plan(ctx) else {
            return verify_err!(op.loc(), "cute.mma_partition_b must carry a plan");
        };
        if let Err(message) = check_plan(&plan, ctx) {
            return verify_err!(op.loc(), "cute.mma_partition_b {message}");
        }
        let Some(view) = self.view(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.mma_partition_b must carry an N-role shared value view"
            );
        };
        if let Err(message) = check_data_view(ctx, &view, &plan, CuteTensorRoleAttr::Nkl) {
            return verify_err!(op.loc(), "cute.mma_partition_b input {message}");
        }
        if !shared_carriers_match_view(ctx, op.get_operand(0), op.get_operand(1), &view) {
            return verify_err!(
                op.loc(),
                "cute.mma_partition_b pointer/capacity carriers must match its typed B view"
            );
        }
        if !is_u64(ctx, op.get_operand(2)) || !is_u64(ctx, op.get_operand(3)) {
            return verify_err!(
                op.loc(),
                "cute.mma_partition_b warp-N and K-half must be unsigned u64 values"
            );
        }
        if (0..4)
            .any(|index| op.get_result(index).get_type(ctx) != op.get_operand(index).get_type(ctx))
        {
            return verify_err!(
                op.loc(),
                "cute.mma_partition_b must preserve pointer, capacity, warp-N, and K-half types"
            );
        }
        Ok(())
    }
}

/// Multiply one A K=64 bundle by one lazy B selection and add it to C.
///
/// B arrives as four scalar carriers, never as eight eager fragments. Backend
/// A must honor `plan.b_load_group`: load two B fragments, issue their four
/// MMA atoms, then move to the next pair.
#[pliron_op(
    name = "cute.tiled_gemm",
    format,
    interfaces = [NOpdsInterface<8>, NResultsInterface<1>, OneResultInterface],
    attributes = (
        tiled_gemm_plan: CuteTiledMmaPlanAttr,
        tiled_gemm_b_view: TypeAttr
    )
)]
pub struct CuteTiledGemmOp;

impl CuteTiledGemmOp {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &mut Context,
        lane: Value,
        a: Value,
        b_base: Value,
        b_capacity: Value,
        b_warp_n: Value,
        b_k_half: Value,
        scales: Value,
        accumulator: Value,
        b_view: TypeHandle,
        plan: CuteTiledMmaPlanAttr,
    ) -> Self {
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![accumulator.get_type(ctx)],
                vec![
                    lane,
                    a,
                    b_base,
                    b_capacity,
                    b_warp_n,
                    b_k_half,
                    scales,
                    accumulator,
                ],
                vec![],
                0,
            ),
        };
        operation.set_attr_tiled_gemm_plan(ctx, plan);
        operation.set_attr_tiled_gemm_b_view(ctx, TypeAttr::new(b_view));
        operation
    }

    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    #[must_use]
    pub fn plan(&self, ctx: &Context) -> Option<CuteTiledMmaPlanAttr> {
        plan_of_attr(self.get_attr_tiled_gemm_plan(ctx))
    }

    /// Static N-role shared tensor facts carried by the four lazy B scalars.
    #[must_use]
    pub fn b_view(&self, ctx: &Context) -> Option<CuteSmemTensorType> {
        let attr = self.get_attr_tiled_gemm_b_view(ctx)?;
        smem_tensor_of_type(ctx, attr.get_type(ctx))
    }

    #[must_use]
    pub fn accumulator(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }
}

impl Verify for CuteTiledGemmOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 8 || op.get_num_results() != 1 {
            return verify_err!(op.loc(), "cute.tiled_gemm needs 8 operands and 1 result");
        }
        let Some(plan) = self.plan(ctx) else {
            return verify_err!(op.loc(), "cute.tiled_gemm must carry a plan");
        };
        if let Err(message) = check_plan(&plan, ctx) {
            return verify_err!(op.loc(), "cute.tiled_gemm {message}");
        }
        if !is_u32(ctx, op.get_operand(0)) {
            return verify_err!(op.loc(), "cute.tiled_gemm lane must be an unsigned u32");
        }
        let expected_a = MmaCarrierShape::u32(
            plan.a_registers_per_lane()
                .expect("verified plan has an A carrier size"),
        );
        if let Err(message) = check_carrier(ctx, op.get_operand(1), expected_a) {
            return verify_err!(op.loc(), "cute.tiled_gemm A input {message}");
        }
        let Some(b_view) = self.b_view(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.tiled_gemm must carry an N-role shared view for lazy B"
            );
        };
        if let Err(message) = check_data_view(ctx, &b_view, &plan, CuteTensorRoleAttr::Nkl) {
            return verify_err!(op.loc(), "cute.tiled_gemm B input {message}");
        }
        if !shared_carriers_match_view(ctx, op.get_operand(2), op.get_operand(3), &b_view)
            || !is_u64(ctx, op.get_operand(4))
            || !is_u64(ctx, op.get_operand(5))
        {
            return verify_err!(
                op.loc(),
                "cute.tiled_gemm lazy B must be pointer, capacity, warp-N, and K-half scalars matching its typed view"
            );
        }
        let expected_scales = MmaCarrierShape::u32(
            plan.scale_words_per_lane()
                .expect("verified plan has a scale carrier size"),
        );
        if let Err(message) = check_carrier(ctx, op.get_operand(6), expected_scales) {
            return verify_err!(op.loc(), "cute.tiled_gemm scale input {message}");
        }
        let expected_accumulator = MmaCarrierShape::f32(
            plan.accumulator_registers_per_lane()
                .expect("verified plan has an accumulator size"),
        );
        if let Err(message) = check_carrier(ctx, op.get_operand(7), expected_accumulator) {
            return verify_err!(op.loc(), "cute.tiled_gemm accumulator input {message}");
        }
        if self.accumulator(ctx).get_type(ctx) != op.get_operand(7).get_type(ctx) {
            return verify_err!(
                op.loc(),
                "cute.tiled_gemm result must preserve the accumulator carrier type"
            );
        }
        if let Err(message) = check_carrier(ctx, self.accumulator(ctx), expected_accumulator) {
            return verify_err!(op.loc(), "cute.tiled_gemm result {message}");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::{CuteTensorAccessAttr, CuteTensorAddressSpaceAttr};
    use crate::types::CuteTensorViewType;
    use dialect_mir::ops::MirUndefOp;
    use pliron::builtin::types::Signedness;

    fn undef(ctx: &mut Context, ty: TypeHandle) -> Value {
        MirUndefOp::new(ctx, ty)
            .get_operation()
            .deref(ctx)
            .get_result(0)
    }

    fn data_placement() -> ComposedLayout {
        let inner: Layout = "(128,32):(32,1)".parse().unwrap();
        ComposedLayout::new(Swizzle::new(2, 3, 3), 0, inner, OffsetUnit::Elements).unwrap()
    }

    struct SmemTypeFacts {
        format: CuteTensorFormatAttr,
        role: CuteTensorRoleAttr,
        alignment: u64,
        layout: CuteTensorLayoutAttr,
        placement: ComposedLayout,
    }

    fn smem_type(
        ctx: &mut Context,
        logical: TypeHandle,
        storage: TypeHandle,
        facts: SmemTypeFacts,
    ) -> TypeHandle {
        let tensor: TypeHandle = CuteTensorViewType::get_with_facts(
            ctx,
            logical,
            storage,
            CuteTensorAddressSpaceAttr::Smem,
            CuteTensorAccessAttr::ReadOnly,
            facts.alignment,
            facts.format,
            facts.role,
            facts.layout,
        )
        .into();
        CuteSmemTensorType::get(ctx, tensor, facts.placement).into()
    }

    struct TestTypes {
        u32_ty: TypeHandle,
        u64_ty: TypeHandle,
        f32_ty: TypeHandle,
        a_ptr_ty: TypeHandle,
        b_ptr_ty: TypeHandle,
        scale_a_ptr_ty: TypeHandle,
        scale_b_ptr_ty: TypeHandle,
        a_view_ty: TypeHandle,
        b_view_ty: TypeHandle,
        scale_a_view_ty: TypeHandle,
        scale_b_view_ty: TypeHandle,
        a_carrier_ty: TypeHandle,
        scale_carrier_ty: TypeHandle,
        accumulator_ty: TypeHandle,
        plan: CuteTiledMmaPlanAttr,
    }

    fn setup() -> (Context, TestTypes) {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        crate::register(&mut ctx);
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let f16_ty: TypeHandle = MirFP16Type::get(&ctx).into();
        let a_ptr_ty: TypeHandle =
            MirPtrType::get(&mut ctx, f16_ty, true, address_space::SHARED).into();
        let b_ptr_ty = a_ptr_ty;
        let scale_a_ptr_ty: TypeHandle =
            MirPtrType::get(&mut ctx, u32_ty, true, address_space::SHARED).into();
        let scale_b_ptr_ty = scale_a_ptr_ty;
        let plan = CuteTiledMmaPlanAttr::mxf4_128x128x128(data_placement());
        let a_view_ty = smem_type(
            &mut ctx,
            f32_ty,
            f16_ty,
            SmemTypeFacts {
                format: CuteTensorFormatAttr::E2M1,
                role: CuteTensorRoleAttr::Mkl,
                alignment: 16,
                layout: CuteTensorLayoutAttr::KMajor,
                placement: data_placement(),
            },
        );
        let b_view_ty = smem_type(
            &mut ctx,
            f32_ty,
            f16_ty,
            SmemTypeFacts {
                format: CuteTensorFormatAttr::E2M1,
                role: CuteTensorRoleAttr::Nkl,
                alignment: 16,
                layout: CuteTensorLayoutAttr::KMajor,
                placement: data_placement(),
            },
        );
        let scale_a_view_ty = smem_type(
            &mut ctx,
            f32_ty,
            u32_ty,
            SmemTypeFacts {
                format: CuteTensorFormatAttr::UE8M0,
                role: CuteTensorRoleAttr::Mkl,
                alignment: 4,
                layout: CuteTensorLayoutAttr::BlockScaleKMajor(32),
                placement: canonical_scale_placement(),
            },
        );
        let scale_b_view_ty = smem_type(
            &mut ctx,
            f32_ty,
            u32_ty,
            SmemTypeFacts {
                format: CuteTensorFormatAttr::UE8M0,
                role: CuteTensorRoleAttr::Nkl,
                alignment: 4,
                layout: CuteTensorLayoutAttr::BlockScaleKMajor(32),
                placement: canonical_scale_placement(),
            },
        );
        let a_carrier_ty: TypeHandle =
            dialect_mir::types::MirArrayType::get(&mut ctx, u32_ty, 8).into();
        let scale_carrier_ty: TypeHandle =
            dialect_mir::types::MirArrayType::get(&mut ctx, u32_ty, 10).into();
        let accumulator_ty: TypeHandle =
            dialect_mir::types::MirArrayType::get(&mut ctx, f32_ty, 64).into();
        (
            ctx,
            TestTypes {
                u32_ty,
                u64_ty,
                f32_ty,
                a_ptr_ty,
                b_ptr_ty,
                scale_a_ptr_ty,
                scale_b_ptr_ty,
                a_view_ty,
                b_view_ty,
                scale_a_view_ty,
                scale_b_view_ty,
                a_carrier_ty,
                scale_carrier_ty,
                accumulator_ty,
                plan,
            },
        )
    }

    fn overlay(
        ctx: &mut Context,
        base_ty: TypeHandle,
        capacity_ty: TypeHandle,
        view_ty: TypeHandle,
    ) -> CuteSmemTensorOverlayOp {
        let base = undef(ctx, base_ty);
        let capacity = undef(ctx, capacity_ty);
        CuteSmemTensorOverlayOp::new(ctx, base, capacity, view_ty)
    }

    #[test]
    fn visual_chain_keeps_carriers_ordinary_and_b_lazy() {
        let (mut ctx, types) = setup();
        assert!(types.plan.verify(&ctx).is_ok());
        assert_eq!(types.plan.compute_warps(), Some(8));
        assert_eq!(types.plan.scale_words_per_lane(), Some(10));
        assert_eq!(types.plan.a_registers_per_lane(), Some(8));
        assert_eq!(types.plan.accumulator_registers_per_lane(), Some(64));

        let a = overlay(&mut ctx, types.a_ptr_ty, types.u64_ty, types.a_view_ty);
        let b = overlay(&mut ctx, types.b_ptr_ty, types.u64_ty, types.b_view_ty);
        let scale_a = overlay(
            &mut ctx,
            types.scale_a_ptr_ty,
            types.u64_ty,
            types.scale_a_view_ty,
        );
        let scale_b = overlay(
            &mut ctx,
            types.scale_b_ptr_ty,
            types.u64_ty,
            types.scale_b_view_ty,
        );
        for operation in [&a, &b, &scale_a, &scale_b] {
            assert!(operation.verify(&ctx).is_ok());
        }

        let lane = undef(&mut ctx, types.u32_ty);
        let warp_m = undef(&mut ctx, types.u64_ty);
        let warp_n = undef(&mut ctx, types.u64_ty);
        let k_half = undef(&mut ctx, types.u64_ty);
        let mma = CuteTiledMmaSliceOp::new(&mut ctx, lane, types.plan.clone());
        assert!(mma.verify(&ctx).is_ok());

        let zero = undef(&mut ctx, types.f32_ty);
        let fill = CuteFragmentFillOp::new(
            &mut ctx,
            zero,
            types.accumulator_ty,
            types.plan.clone(),
            CuteMmaCarrierKindAttr::Accumulator,
        );
        assert!(fill.verify(&ctx).is_ok());

        let sliced_lane = mma.sliced_lane(&ctx);
        let scale_a_base = scale_a.base(&ctx);
        let scale_a_capacity = scale_a.capacity(&ctx);
        let scale_b_base = scale_b.base(&ctx);
        let scale_b_capacity = scale_b.capacity(&ctx);
        let load_scales = CuteMmaLoadScalesOp::new(
            &mut ctx,
            sliced_lane,
            scale_a_base,
            scale_a_capacity,
            scale_b_base,
            scale_b_capacity,
            warp_m,
            warp_n,
            types.scale_a_view_ty,
            types.scale_b_view_ty,
            types.scale_carrier_ty,
            types.plan.clone(),
        );
        assert!(load_scales.verify(&ctx).is_ok());
        let stage_scales = load_scales.scales(&ctx);
        let slice_scales = CuteFragmentSliceKOp::new(
            &mut ctx,
            stage_scales,
            k_half,
            types.scale_carrier_ty,
            types.plan.clone(),
        );
        assert!(slice_scales.verify(&ctx).is_ok());

        let a_base = a.base(&ctx);
        let a_capacity = a.capacity(&ctx);
        let load_a = CuteMmaLoadAOp::new(
            &mut ctx,
            sliced_lane,
            a_base,
            a_capacity,
            warp_m,
            k_half,
            types.a_view_ty,
            types.a_carrier_ty,
            types.plan.clone(),
        );
        assert!(load_a.verify(&ctx).is_ok());

        let b_base = b.base(&ctx);
        let b_capacity = b.capacity(&ctx);
        let partition_b = CuteMmaPartitionBOp::new(
            &mut ctx,
            b_base,
            b_capacity,
            warp_n,
            k_half,
            types.b_view_ty,
            types.plan.clone(),
        );
        assert!(partition_b.verify(&ctx).is_ok());
        {
            let partition_operation = partition_b.get_operation().deref(&ctx);
            assert_eq!(partition_operation.get_num_results(), 4);
            assert!((0..4).all(|index| {
                let value = partition_operation.get_result(index);
                mma_carrier_shape(&ctx, value.get_type(&ctx)).is_none()
            }));
        }

        let a_fragment = load_a.fragment(&ctx);
        let partition_base = partition_b.base(&ctx);
        let partition_capacity = partition_b.capacity(&ctx);
        let partition_warp_n = partition_b.warp_n(&ctx);
        let partition_k_half = partition_b.k_half(&ctx);
        let selected_scales = slice_scales.scales(&ctx);
        let initial_accumulator = fill.fragment(&ctx);
        let gemm = CuteTiledGemmOp::new(
            &mut ctx,
            sliced_lane,
            a_fragment,
            partition_base,
            partition_capacity,
            partition_warp_n,
            partition_k_half,
            selected_scales,
            initial_accumulator,
            types.b_view_ty,
            types.plan,
        );
        assert!(gemm.verify(&ctx).is_ok());
        assert_eq!(
            mma_carrier_shape(&ctx, gemm.accumulator(&ctx).get_type(&ctx)),
            Some(MmaCarrierShape::f32(64))
        );
    }

    #[test]
    fn tiled_gemm_rejects_an_eager_or_untyped_b_fragment() {
        let (mut ctx, types) = setup();
        let lane = undef(&mut ctx, types.u32_ty);
        let a = undef(&mut ctx, types.a_carrier_ty);
        let eager_b = undef(&mut ctx, types.a_carrier_ty);
        let capacity = undef(&mut ctx, types.u64_ty);
        let warp_n = undef(&mut ctx, types.u64_ty);
        let k_half = undef(&mut ctx, types.u64_ty);
        let scales = undef(&mut ctx, types.scale_carrier_ty);
        let accumulator = undef(&mut ctx, types.accumulator_ty);
        let gemm = CuteTiledGemmOp::new(
            &mut ctx,
            lane,
            a,
            eager_b,
            capacity,
            warp_n,
            k_half,
            scales,
            accumulator,
            types.b_view_ty,
            types.plan,
        );
        assert!(
            gemm.verify(&ctx)
                .unwrap_err()
                .to_string()
                .contains("lazy B")
        );
    }

    #[test]
    fn typed_consumers_accept_flattened_carriers_after_aggregate_forwarding() {
        let (mut ctx, types) = setup();
        let lane = undef(&mut ctx, types.u32_ty);
        let warp_m = undef(&mut ctx, types.u64_ty);
        let warp_n = undef(&mut ctx, types.u64_ty);
        let k_half = undef(&mut ctx, types.u64_ty);
        let a_base = undef(&mut ctx, types.a_ptr_ty);
        let a_capacity = undef(&mut ctx, types.u64_ty);
        let load_a = CuteMmaLoadAOp::new(
            &mut ctx,
            lane,
            a_base,
            a_capacity,
            warp_m,
            k_half,
            types.a_view_ty,
            types.a_carrier_ty,
            types.plan.clone(),
        );
        assert!(load_a.verify(&ctx).is_ok());

        let a_fragment = load_a.fragment(&ctx);
        let b_base = undef(&mut ctx, types.b_ptr_ty);
        let b_capacity = undef(&mut ctx, types.u64_ty);
        let scales = undef(&mut ctx, types.scale_carrier_ty);
        let accumulator = undef(&mut ctx, types.accumulator_ty);
        let gemm = CuteTiledGemmOp::new(
            &mut ctx,
            lane,
            a_fragment,
            b_base,
            b_capacity,
            warp_n,
            k_half,
            scales,
            accumulator,
            types.b_view_ty,
            types.plan,
        );
        assert!(gemm.verify(&ctx).is_ok());
    }

    #[test]
    fn scale_load_rejects_swapped_m_and_n_roles() {
        let (mut ctx, types) = setup();
        let scale_a_wrong = overlay(
            &mut ctx,
            types.scale_a_ptr_ty,
            types.u64_ty,
            types.scale_b_view_ty,
        );
        let scale_b = overlay(
            &mut ctx,
            types.scale_b_ptr_ty,
            types.u64_ty,
            types.scale_b_view_ty,
        );
        let lane = undef(&mut ctx, types.u32_ty);
        let warp_m = undef(&mut ctx, types.u64_ty);
        let warp_n = undef(&mut ctx, types.u64_ty);
        let scale_a_base = scale_a_wrong.base(&ctx);
        let scale_a_capacity = scale_a_wrong.capacity(&ctx);
        let scale_b_base = scale_b.base(&ctx);
        let scale_b_capacity = scale_b.capacity(&ctx);
        let load = CuteMmaLoadScalesOp::new(
            &mut ctx,
            lane,
            scale_a_base,
            scale_a_capacity,
            scale_b_base,
            scale_b_capacity,
            warp_m,
            warp_n,
            types.scale_b_view_ty,
            types.scale_b_view_ty,
            types.scale_carrier_ty,
            types.plan,
        );
        assert!(
            load.verify(&ctx)
                .unwrap_err()
                .to_string()
                .contains("A must be")
        );
    }

    #[test]
    fn plan_and_carrier_verifiers_fail_closed() {
        let (mut ctx, types) = setup();
        let mut bad_plan = types.plan.clone();
        bad_plan.b_load_group = 3;
        assert!(
            bad_plan
                .verify(&ctx)
                .unwrap_err()
                .to_string()
                .contains("B load group")
        );

        let mut wrong_ownership = types.plan.clone();
        wrong_ownership.m_ownership = crate::attributes::CuteLayoutAttr(
            "(4,2):(2,1)".parse().expect("dense alternate map is valid"),
        );
        assert!(wrong_ownership.verify(&ctx).is_ok());
        let lane = undef(&mut ctx, types.u32_ty);
        let slice = CuteTiledMmaSliceOp::new(&mut ctx, lane, wrong_ownership);
        assert!(
            slice
                .verify(&ctx)
                .unwrap_err()
                .to_string()
                .contains("ownership maps")
        );

        let wrong_accumulator_ty: TypeHandle =
            dialect_mir::types::MirArrayType::get(&mut ctx, types.f32_ty, 63).into();
        let zero = undef(&mut ctx, types.f32_ty);
        let fill = CuteFragmentFillOp::new(
            &mut ctx,
            zero,
            wrong_accumulator_ty,
            types.plan,
            CuteMmaCarrierKindAttr::Accumulator,
        );
        assert!(
            fill.verify(&ctx)
                .unwrap_err()
                .to_string()
                .contains("carrier shape")
        );
    }
}
