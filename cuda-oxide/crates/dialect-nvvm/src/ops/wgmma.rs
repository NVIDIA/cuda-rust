/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warpgroup Matrix Multiply-Accumulate (WGMMA) operations for Hopper `sm_90a`.
//!
//! WGMMA provides tensor core operations that operate at the warpgroup level
//! (4 warps = 128 threads) for high-throughput matrix multiplication.
//!
//! The public importer first creates a pointer-form MMA operation. Before LLVM
//! lowering, `mir-lower` recognizes complete straight-line regions and a narrow
//! canonical counted K-loop shape. Straight-line pointer-form sequences may use
//! the deferred group, which keeps all 32 per-thread accumulator values in
//! one inline-PTX scope until `wait_group<0>` completes.
//!
//! The internal value-form group represents those same 32 accumulator values
//! explicitly as SSA operands/results. A counted-loop variant additionally owns
//! descriptor recurrences and loop control. A pipelined variant carries multiple
//! independent 32-value accumulator slots so statically known `wait_group<N>`
//! depths can keep committed groups in flight without exposing an in-flight
//! accumulator to LLVM. The pointer-form group remains the deferred fallback.
//!
//! F16 uses the same 32-value accumulator model for canonical linear
//! full-drain regions and the canonical counted K-loop. TF32 uses the same
//! carrier only for canonical linear full-drain regions, with the hardware
//! `m64n64k8` shape. Neither F16 nor TF32 has a deferred pointer-form group or
//! partial-wait pipeline carrier, and TF32 has no counted-loop carrier.

use dialect_mir::types::{MirPtrType, address_space};
use pliron::{
    attribute::Attribute,
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface},
        types::{FP32Type, IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    value::Value,
    verify_err, verify_err_noloc,
};
use pliron_derive::{pliron_attr, pliron_op};

const WGMMA_M64N64_F32_ACCUMULATOR_COUNT: usize = 32;
const WGMMA_M64N128_F32_ACCUMULATOR_COUNT: usize = 64;
const WGMMA_COUNTED_LOOP_CONTROL_COUNT: usize = 5;
const WGMMA_MAX_PENDING_GROUPS: u8 = 7;
const WGMMA_COUNTED_PIPELINE_MAX_PENDING_GROUPS: u8 = 2;

// =============================================================================
// Descriptor Operations
// =============================================================================

/// Create a shared memory descriptor for WGMMA.
#[pliron_op(
    name = "nvvm.wgmma_make_smem_desc",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct WgmmaMakeSmemDescOp;

impl WgmmaMakeSmemDescOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        WgmmaMakeSmemDescOp { op }
    }
}

fn is_u64(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {
    ty.deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| {
            integer.width() == 64 && integer.signedness() == Signedness::Unsigned
        })
}

fn is_f32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {
    ty.deref(ctx).downcast_ref::<FP32Type>().is_some()
}

fn is_supported_wgmma_accumulator(ctx: &Context, value: Value) -> bool {
    let value_type = value.get_type(ctx);
    let value_type_ref = value_type.deref(ctx);
    let Some(pointer_type) = value_type_ref.downcast_ref::<MirPtrType>() else {
        return false;
    };

    pointer_type.is_mutable() && pointer_type.address_space() == address_space::GENERIC
}

impl Verify for WgmmaMakeSmemDescOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_make_smem_desc requires one operand and one result"
            );
        }
        let pointer_ty = op.get_operand(0).get_type(ctx);
        let pointer_ty_obj = pointer_ty.deref(ctx);
        let Some(pointer_ty) = pointer_ty_obj.downcast_ref::<MirPtrType>() else {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_make_smem_desc operand must be a MIR pointer"
            );
        };
        if !matches!(
            pointer_ty.address_space,
            address_space::GENERIC | address_space::SHARED
        ) {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_make_smem_desc operand must point to generic or shared memory"
            );
        }
        if !is_u64(ctx, op.get_result(0).get_type(ctx)) {
            return verify_err!(op.loc(), "nvvm.wgmma_make_smem_desc result must be u64");
        }
        Ok(())
    }
}

// =============================================================================
// Matrix Multiply-Accumulate Operations
// =============================================================================

/// Pointer-form BF16 WGMMA operation emitted by `mir-importer`.
///
/// This operation is not legal at final lowering. It must be consumed by the
/// deferred-accumulator fusion pass together with its fence, commit, and
/// `wait_group<0>` operations.
#[pliron_op(
    name = "nvvm.wgmma_mma_m64n64k16_f32_bf16",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct WgmmaMmaM64N64K16F32Bf16Op;

impl WgmmaMmaM64N64K16F32Bf16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        WgmmaMmaM64N64K16F32Bf16Op { op }
    }
}

impl Verify for WgmmaMmaM64N64K16F32Bf16Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 3 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_m64n64k16_f32_bf16 requires three operands and no results"
            );
        }
        if !is_supported_wgmma_accumulator(ctx, op.get_operand(0)) {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_m64n64k16_f32_bf16 accumulator must be a mutable generic MIR pointer"
            );
        }
        if !is_u64(ctx, op.get_operand(1).get_type(ctx))
            || !is_u64(ctx, op.get_operand(2).get_type(ctx))
        {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_m64n64k16_f32_bf16 descriptors must be u64"
            );
        }
        Ok(())
    }
}

/// Pointer-form BF16 m64n128k16 WGMMA operation emitted by `mir-importer`.
///
/// This operation is not legal at final lowering. The deferred-accumulator
/// fusion pass accepts it only in a canonical linear full-drain region and
/// rewrites it to the 64-value form.
#[pliron_op(
    name = "nvvm.wgmma_mma_m64n128k16_f32_bf16",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct WgmmaMmaM64N128K16F32Bf16Op;

impl WgmmaMmaM64N128K16F32Bf16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for WgmmaMmaM64N128K16F32Bf16Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 3 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_m64n128k16_f32_bf16 requires three operands and no results"
            );
        }
        if !is_supported_wgmma_accumulator(ctx, op.get_operand(0)) {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_m64n128k16_f32_bf16 accumulator must be a mutable generic MIR pointer"
            );
        }
        if !is_u64(ctx, op.get_operand(1).get_type(ctx))
            || !is_u64(ctx, op.get_operand(2).get_type(ctx))
        {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_m64n128k16_f32_bf16 descriptors must be u64"
            );
        }
        Ok(())
    }
}

/// Pointer-form F16 WGMMA operation emitted by `mir-importer`.
///
/// This operation is not legal at final lowering. The deferred-accumulator
/// fusion pass accepts it only in a canonical linear full-drain region and
/// rewrites it to the F16 value-form group.
#[pliron_op(
    name = "nvvm.wgmma_mma_m64n64k16_f32_f16",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct WgmmaMmaM64N64K16F32F16Op;

impl WgmmaMmaM64N64K16F32F16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        WgmmaMmaM64N64K16F32F16Op { op }
    }
}

impl Verify for WgmmaMmaM64N64K16F32F16Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 3 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_m64n64k16_f32_f16 requires three operands and no results"
            );
        }
        if !is_supported_wgmma_accumulator(ctx, op.get_operand(0)) {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_m64n64k16_f32_f16 accumulator must be a mutable generic MIR pointer"
            );
        }
        if !is_u64(ctx, op.get_operand(1).get_type(ctx))
            || !is_u64(ctx, op.get_operand(2).get_type(ctx))
        {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_m64n64k16_f32_f16 descriptors must be u64"
            );
        }
        Ok(())
    }
}

/// Pointer-form TF32 WGMMA operation emitted by `mir-importer`.
///
/// This operation is not legal at final lowering. The deferred-accumulator
/// fusion pass accepts it only in a canonical linear full-drain region and
/// rewrites it to the TF32 K=8 value-form group.
#[pliron_op(
    name = "nvvm.wgmma_mma_m64n64k8_f32_tf32",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct WgmmaMmaM64N64K8F32Tf32Op;

impl WgmmaMmaM64N64K8F32Tf32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        WgmmaMmaM64N64K8F32Tf32Op { op }
    }
}

impl Verify for WgmmaMmaM64N64K8F32Tf32Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 3 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_m64n64k8_f32_tf32 requires three operands and no results"
            );
        }
        if !is_supported_wgmma_accumulator(ctx, op.get_operand(0)) {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_m64n64k8_f32_tf32 accumulator must be a mutable generic MIR pointer"
            );
        }
        if !is_u64(ctx, op.get_operand(1).get_type(ctx))
            || !is_u64(ctx, op.get_operand(2).get_type(ctx))
        {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_m64n64k8_f32_tf32 descriptors must be u64"
            );
        }
        Ok(())
    }
}

/// Deferred BF16 WGMMA group with one accumulator and one or more descriptor pairs.
///
/// Operand layout:
///
/// ```text
/// [acc_ptr, desc_a_0, desc_b_0, ..., desc_a_n, desc_b_n]
/// ```
///
/// The operation represents a complete sequence containing an implicit fence,
/// all MMA instructions, one commit, and `wait_group<0>`. It has no results
/// because the accumulator is written back through `acc_ptr` after the wait.
#[pliron_op(name = "nvvm.wgmma_mma_group_m64n64k16_f32_bf16", format)]
pub struct WgmmaMmaGroupM64N64K16F32Bf16Op;

impl WgmmaMmaGroupM64N64K16F32Bf16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Build a deferred group from one accumulator and descriptor pairs.
    pub fn build(ctx: &mut Context, accumulator: Value, descriptors: Vec<Value>) -> Ptr<Operation> {
        let mut operands = Vec::with_capacity(1 + descriptors.len());
        operands.push(accumulator);
        operands.extend(descriptors);
        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            operands,
            vec![],
            0,
        )
    }
}

impl Verify for WgmmaMmaGroupM64N64K16F32Bf16Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let operand_count = op.get_num_operands();
        if operand_count < 3 || operand_count.is_multiple_of(2) || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_group_m64n64k16_f32_bf16 requires one accumulator, one or more descriptor pairs, and no results"
            );
        }
        if !is_supported_wgmma_accumulator(ctx, op.get_operand(0)) {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_group_m64n64k16_f32_bf16 accumulator must be a mutable generic MIR pointer"
            );
        }
        for descriptor_index in 1..operand_count {
            if !is_u64(ctx, op.get_operand(descriptor_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_group_m64n64k16_f32_bf16 descriptors must be u64"
                );
            }
        }
        Ok(())
    }
}

/// Value-form BF16 WGMMA group with 32 SSA accumulator values.
///
/// Operand layout:
///
/// ```text
/// [acc_0, ..., acc_31, desc_a_0, desc_b_0, ..., desc_a_n, desc_b_n]
/// ```
///
/// Result layout:
///
/// ```text
/// [acc_0', ..., acc_31']
/// ```
///
/// The operation represents one complete register lifetime scope containing an
/// implicit fence, one or more MMA instructions, one commit, and
/// `wait_group<0>`. Unlike the deferred pointer-form group, it does not
/// materialize the accumulator through memory at either boundary.
#[pliron_op(name = "nvvm.wgmma_mma_group_values_m64n64k16_f32_bf16", format)]
pub struct WgmmaMmaGroupValuesM64N64K16F32Bf16Op;

impl WgmmaMmaGroupValuesM64N64K16F32Bf16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Build a value-form group from 32 accumulator values and descriptor pairs.
    pub fn build(
        ctx: &mut Context,
        accumulators: Vec<Value>,
        descriptors: Vec<Value>,
    ) -> Ptr<Operation> {
        let f32_ty = FP32Type::get(ctx);
        let mut operands = Vec::with_capacity(accumulators.len() + descriptors.len());
        operands.extend(accumulators);
        operands.extend(descriptors);

        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![f32_ty.into(); WGMMA_M64N64_F32_ACCUMULATOR_COUNT],
            operands,
            vec![],
            0,
        )
    }
}

impl Verify for WgmmaMmaGroupValuesM64N64K16F32Bf16Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let operand_count = op.get_num_operands();
        let result_count = op.get_num_results();

        if result_count != WGMMA_M64N64_F32_ACCUMULATOR_COUNT {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_group_values_m64n64k16_f32_bf16 requires exactly 32 f32 results"
            );
        }

        if operand_count < WGMMA_M64N64_F32_ACCUMULATOR_COUNT + 2 {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_group_values_m64n64k16_f32_bf16 requires 32 f32 accumulators and one or more descriptor pairs"
            );
        }

        let descriptor_count = operand_count - WGMMA_M64N64_F32_ACCUMULATOR_COUNT;
        if !descriptor_count.is_multiple_of(2) {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_group_values_m64n64k16_f32_bf16 descriptors must form pairs"
            );
        }

        for accumulator_index in 0..WGMMA_M64N64_F32_ACCUMULATOR_COUNT {
            if !is_f32(ctx, op.get_operand(accumulator_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_group_values_m64n64k16_f32_bf16 accumulator operands must be f32"
                );
            }

            if !is_f32(ctx, op.get_result(accumulator_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_group_values_m64n64k16_f32_bf16 results must be f32"
                );
            }
        }

        for descriptor_index in WGMMA_M64N64_F32_ACCUMULATOR_COUNT..operand_count {
            if !is_u64(ctx, op.get_operand(descriptor_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_group_values_m64n64k16_f32_bf16 descriptors must be u64"
                );
            }
        }

        Ok(())
    }
}

/// Value-form BF16 m64n128 WGMMA group with 64 SSA accumulator values.
///
/// Operand layout:
///
/// ```text
/// [acc_0, ..., acc_63, desc_a_0, desc_b_0, ..., desc_a_n, desc_b_n]
/// ```
///
/// Result layout:
///
/// ```text
/// [acc_0', ..., acc_63']
/// ```
///
/// The operation represents one canonical linear full-drain lifetime with an
/// implicit fence, one or more m64n128 BF16 MMA instructions, one commit, and
/// `wait_group<0>`. It does not model pointer fallback, counted loops, or
/// partial waits.
#[pliron_op(name = "nvvm.wgmma_mma_group_values_m64n128k16_f32_bf16", format)]
pub struct WgmmaMmaGroupValuesM64N128K16F32Bf16Op;

impl WgmmaMmaGroupValuesM64N128K16F32Bf16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Build a value-form group from 64 accumulator values and descriptor pairs.
    pub fn build(
        ctx: &mut Context,
        accumulators: Vec<Value>,
        descriptors: Vec<Value>,
    ) -> Ptr<Operation> {
        let f32_ty = FP32Type::get(ctx);
        let mut operands = Vec::with_capacity(accumulators.len() + descriptors.len());
        operands.extend(accumulators);
        operands.extend(descriptors);

        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![f32_ty.into(); WGMMA_M64N128_F32_ACCUMULATOR_COUNT],
            operands,
            vec![],
            0,
        )
    }
}

impl Verify for WgmmaMmaGroupValuesM64N128K16F32Bf16Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let operand_count = op.get_num_operands();
        let result_count = op.get_num_results();

        if result_count != WGMMA_M64N128_F32_ACCUMULATOR_COUNT {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_group_values_m64n128k16_f32_bf16 requires exactly 64 f32 results"
            );
        }
        if operand_count < WGMMA_M64N128_F32_ACCUMULATOR_COUNT + 2 {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_group_values_m64n128k16_f32_bf16 requires 64 f32 accumulators and one or more descriptor pairs"
            );
        }

        let descriptor_count = operand_count - WGMMA_M64N128_F32_ACCUMULATOR_COUNT;
        if !descriptor_count.is_multiple_of(2) {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_group_values_m64n128k16_f32_bf16 descriptors must form pairs"
            );
        }

        for accumulator_index in 0..WGMMA_M64N128_F32_ACCUMULATOR_COUNT {
            if !is_f32(ctx, op.get_operand(accumulator_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_group_values_m64n128k16_f32_bf16 accumulator operands must be f32"
                );
            }
            if !is_f32(ctx, op.get_result(accumulator_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_group_values_m64n128k16_f32_bf16 results must be f32"
                );
            }
        }

        for descriptor_index in WGMMA_M64N128_F32_ACCUMULATOR_COUNT..operand_count {
            if !is_u64(ctx, op.get_operand(descriptor_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_group_values_m64n128k16_f32_bf16 descriptors must be u64"
                );
            }
        }

        Ok(())
    }
}

/// Value-form F16 WGMMA group with 32 SSA accumulator values.
///
/// Operand layout:
///
/// ```text
/// [acc_0, ..., acc_31, desc_a_0, desc_b_0, ..., desc_a_n, desc_b_n]
/// ```
///
/// Result layout:
///
/// ```text
/// [acc_0', ..., acc_31']
/// ```
///
/// The operation represents one canonical linear full-drain lifetime with an
/// implicit fence, one or more F16 MMA instructions, one commit, and
/// `wait_group<0>`. It does not model counted loops or partial waits.
#[pliron_op(name = "nvvm.wgmma_mma_group_values_m64n64k16_f32_f16", format)]
pub struct WgmmaMmaGroupValuesM64N64K16F32F16Op;

impl WgmmaMmaGroupValuesM64N64K16F32F16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Build a value-form group from 32 accumulator values and descriptor pairs.
    pub fn build(
        ctx: &mut Context,
        accumulators: Vec<Value>,
        descriptors: Vec<Value>,
    ) -> Ptr<Operation> {
        let f32_ty = FP32Type::get(ctx);
        let mut operands = Vec::with_capacity(accumulators.len() + descriptors.len());
        operands.extend(accumulators);
        operands.extend(descriptors);

        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![f32_ty.into(); WGMMA_M64N64_F32_ACCUMULATOR_COUNT],
            operands,
            vec![],
            0,
        )
    }
}

impl Verify for WgmmaMmaGroupValuesM64N64K16F32F16Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let operand_count = op.get_num_operands();
        let result_count = op.get_num_results();

        if result_count != WGMMA_M64N64_F32_ACCUMULATOR_COUNT {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_group_values_m64n64k16_f32_f16 requires exactly 32 f32 results"
            );
        }

        if operand_count < WGMMA_M64N64_F32_ACCUMULATOR_COUNT + 2 {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_group_values_m64n64k16_f32_f16 requires 32 f32 accumulators and one or more descriptor pairs"
            );
        }

        let descriptor_count = operand_count - WGMMA_M64N64_F32_ACCUMULATOR_COUNT;
        if !descriptor_count.is_multiple_of(2) {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_group_values_m64n64k16_f32_f16 descriptors must form pairs"
            );
        }

        for accumulator_index in 0..WGMMA_M64N64_F32_ACCUMULATOR_COUNT {
            if !is_f32(ctx, op.get_operand(accumulator_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_group_values_m64n64k16_f32_f16 accumulator operands must be f32"
                );
            }

            if !is_f32(ctx, op.get_result(accumulator_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_group_values_m64n64k16_f32_f16 results must be f32"
                );
            }
        }

        for descriptor_index in WGMMA_M64N64_F32_ACCUMULATOR_COUNT..operand_count {
            if !is_u64(ctx, op.get_operand(descriptor_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_group_values_m64n64k16_f32_f16 descriptors must be u64"
                );
            }
        }

        Ok(())
    }
}

/// Value-form TF32 WGMMA group with 32 SSA accumulator values.
///
/// Operand layout:
///
/// ```text
/// [acc_0, ..., acc_31, desc_a_0, desc_b_0, ..., desc_a_n, desc_b_n]
/// ```
///
/// Result layout:
///
/// ```text
/// [acc_0', ..., acc_31']
/// ```
///
/// The operation represents one canonical linear full-drain lifetime using the
/// hardware K=8 shape with an implicit fence, one or more TF32 MMA instructions,
/// one commit, and
/// `wait_group<0>`. It does not model counted loops or partial waits.
#[pliron_op(name = "nvvm.wgmma_mma_group_values_m64n64k8_f32_tf32", format)]
pub struct WgmmaMmaGroupValuesM64N64K8F32Tf32Op;

impl WgmmaMmaGroupValuesM64N64K8F32Tf32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Build a value-form group from 32 accumulator values and descriptor pairs.
    pub fn build(
        ctx: &mut Context,
        accumulators: Vec<Value>,
        descriptors: Vec<Value>,
    ) -> Ptr<Operation> {
        let f32_ty = FP32Type::get(ctx);
        let mut operands = Vec::with_capacity(accumulators.len() + descriptors.len());
        operands.extend(accumulators);
        operands.extend(descriptors);

        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![f32_ty.into(); WGMMA_M64N64_F32_ACCUMULATOR_COUNT],
            operands,
            vec![],
            0,
        )
    }
}

impl Verify for WgmmaMmaGroupValuesM64N64K8F32Tf32Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let operand_count = op.get_num_operands();
        let result_count = op.get_num_results();

        if result_count != WGMMA_M64N64_F32_ACCUMULATOR_COUNT {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_group_values_m64n64k8_f32_tf32 requires exactly 32 f32 results"
            );
        }

        if operand_count < WGMMA_M64N64_F32_ACCUMULATOR_COUNT + 2 {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_group_values_m64n64k8_f32_tf32 requires 32 f32 accumulators and one or more descriptor pairs"
            );
        }

        let descriptor_count = operand_count - WGMMA_M64N64_F32_ACCUMULATOR_COUNT;
        if !descriptor_count.is_multiple_of(2) {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_group_values_m64n64k8_f32_tf32 descriptors must form pairs"
            );
        }

        for accumulator_index in 0..WGMMA_M64N64_F32_ACCUMULATOR_COUNT {
            if !is_f32(ctx, op.get_operand(accumulator_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_group_values_m64n64k8_f32_tf32 accumulator operands must be f32"
                );
            }

            if !is_f32(ctx, op.get_result(accumulator_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_group_values_m64n64k8_f32_tf32 results must be f32"
                );
            }
        }

        for descriptor_index in WGMMA_M64N64_F32_ACCUMULATOR_COUNT..operand_count {
            if !is_u64(ctx, op.get_operand(descriptor_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_group_values_m64n64k8_f32_tf32 descriptors must be u64"
                );
            }
        }

        Ok(())
    }
}

/// Value-form BF16 WGMMA counted loop with 32 SSA accumulator values.
///
/// Operand layout:
///
/// ```text
/// [
///   acc_0, ..., acc_31,
///   desc_a_base, desc_b_base,
///   desc_a_step, desc_b_step,
///   trip_count,
/// ]
/// ```
///
/// Result layout:
///
/// ```text
/// [acc_0', ..., acc_31']
/// ```
///
/// The operation owns one complete asynchronous WGMMA lifetime. It fences the
/// accumulator registers, executes one MMA per counted-loop iteration while
/// advancing both descriptors by their supplied descriptor deltas,
/// commits the resulting group, and performs a final `wait_group<0>` before the
/// 32 accumulator values become visible to LLVM again.
#[pliron_op(name = "nvvm.wgmma_mma_loop_values_m64n64k16_f32_bf16", format)]
pub struct WgmmaMmaLoopValuesM64N64K16F32Bf16Op;

impl WgmmaMmaLoopValuesM64N64K16F32Bf16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Build a counted-loop group from 32 accumulators and loop-control values.
    pub fn build(
        ctx: &mut Context,
        accumulators: Vec<Value>,
        desc_a_base: Value,
        desc_b_base: Value,
        desc_a_step: Value,
        desc_b_step: Value,
        trip_count: Value,
    ) -> Ptr<Operation> {
        let f32_ty = FP32Type::get(ctx);
        let mut operands =
            Vec::with_capacity(accumulators.len() + WGMMA_COUNTED_LOOP_CONTROL_COUNT);
        operands.extend(accumulators);
        operands.extend([
            desc_a_base,
            desc_b_base,
            desc_a_step,
            desc_b_step,
            trip_count,
        ]);

        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![f32_ty.into(); WGMMA_M64N64_F32_ACCUMULATOR_COUNT],
            operands,
            vec![],
            0,
        )
    }
}

impl Verify for WgmmaMmaLoopValuesM64N64K16F32Bf16Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let expected_operands =
            WGMMA_M64N64_F32_ACCUMULATOR_COUNT + WGMMA_COUNTED_LOOP_CONTROL_COUNT;

        if op.get_num_operands() != expected_operands {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_loop_values_m64n64k16_f32_bf16 requires 32 f32 accumulators and exactly five u64 loop-control operands"
            );
        }

        if op.get_num_results() != WGMMA_M64N64_F32_ACCUMULATOR_COUNT {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_loop_values_m64n64k16_f32_bf16 requires exactly 32 f32 results"
            );
        }

        for accumulator_index in 0..WGMMA_M64N64_F32_ACCUMULATOR_COUNT {
            if !is_f32(ctx, op.get_operand(accumulator_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_loop_values_m64n64k16_f32_bf16 accumulator operands must be f32"
                );
            }
            if !is_f32(ctx, op.get_result(accumulator_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_loop_values_m64n64k16_f32_bf16 results must be f32"
                );
            }
        }

        for control_index in WGMMA_M64N64_F32_ACCUMULATOR_COUNT..expected_operands {
            if !is_u64(ctx, op.get_operand(control_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_loop_values_m64n64k16_f32_bf16 descriptor bases, descriptor steps, and trip count must be u64"
                );
            }
        }

        Ok(())
    }
}

/// Value-form F16 WGMMA counted loop with 32 SSA accumulator values.
///
/// Operand layout:
///
/// ```text
/// [
///   acc_0, ..., acc_31,
///   desc_a_base, desc_b_base,
///   desc_a_step, desc_b_step,
///   trip_count,
/// ]
/// ```
///
/// Result layout:
///
/// ```text
/// [acc_0', ..., acc_31']
/// ```
///
/// The operation owns one complete asynchronous WGMMA lifetime. It fences the
/// accumulator registers, executes one MMA per counted-loop iteration while
/// advancing both descriptors by their supplied descriptor deltas,
/// commits the resulting group, and performs a final `wait_group<0>` before the
/// 32 accumulator values become visible to LLVM again.
#[pliron_op(name = "nvvm.wgmma_mma_loop_values_m64n64k16_f32_f16", format)]
pub struct WgmmaMmaLoopValuesM64N64K16F32F16Op;

impl WgmmaMmaLoopValuesM64N64K16F32F16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Build a counted-loop group from 32 accumulators and loop-control values.
    pub fn build(
        ctx: &mut Context,
        accumulators: Vec<Value>,
        desc_a_base: Value,
        desc_b_base: Value,
        desc_a_step: Value,
        desc_b_step: Value,
        trip_count: Value,
    ) -> Ptr<Operation> {
        let f32_ty = FP32Type::get(ctx);
        let mut operands =
            Vec::with_capacity(accumulators.len() + WGMMA_COUNTED_LOOP_CONTROL_COUNT);
        operands.extend(accumulators);
        operands.extend([
            desc_a_base,
            desc_b_base,
            desc_a_step,
            desc_b_step,
            trip_count,
        ]);

        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![f32_ty.into(); WGMMA_M64N64_F32_ACCUMULATOR_COUNT],
            operands,
            vec![],
            0,
        )
    }
}

impl Verify for WgmmaMmaLoopValuesM64N64K16F32F16Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let expected_operands =
            WGMMA_M64N64_F32_ACCUMULATOR_COUNT + WGMMA_COUNTED_LOOP_CONTROL_COUNT;

        if op.get_num_operands() != expected_operands {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_loop_values_m64n64k16_f32_f16 requires 32 f32 accumulators and exactly five u64 loop-control operands"
            );
        }

        if op.get_num_results() != WGMMA_M64N64_F32_ACCUMULATOR_COUNT {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_loop_values_m64n64k16_f32_f16 requires exactly 32 f32 results"
            );
        }

        for accumulator_index in 0..WGMMA_M64N64_F32_ACCUMULATOR_COUNT {
            if !is_f32(ctx, op.get_operand(accumulator_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_loop_values_m64n64k16_f32_f16 accumulator operands must be f32"
                );
            }
            if !is_f32(ctx, op.get_result(accumulator_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_loop_values_m64n64k16_f32_f16 results must be f32"
                );
            }
        }

        for control_index in WGMMA_M64N64_F32_ACCUMULATOR_COUNT..expected_operands {
            if !is_u64(ctx, op.get_operand(control_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_loop_values_m64n64k16_f32_f16 descriptor bases, descriptor steps, and trip count must be u64"
                );
            }
        }

        Ok(())
    }
}

/// Value-form BF16 WGMMA counted loop with independent accumulator slots.
///
/// This deliberately narrow carrier supports the production counted-loop
/// pipeline depths validated for Hopper:
///
/// ```text
/// wait_group<1> -> 2 accumulator slots
/// wait_group<2> -> 3 accumulator slots
/// ```
///
/// For `max_pending_groups = N`, the operation carries `N + 1` independent
/// 32-value accumulator slots. Each slot owns one descriptor pair and one affine
/// descriptor-step pair. The loop issues one committed group per slot, throttles
/// each stage with `wait_group<N>`, advances all descriptor recurrences, and
/// performs a final `wait_group<0>` before any accumulator becomes visible to
/// LLVM.
///
/// Operand layout:
///
/// ```text
/// [
///   slot0_acc_0, ..., slot0_acc_31,
///   ...
///   slotN_acc_0, ..., slotN_acc_31,
///   desc_a0_base, desc_b0_base, ..., desc_aN_base, desc_bN_base,
///   desc_a0_step, desc_b0_step, ..., desc_aN_step, desc_bN_step,
///   trip_count,
/// ]
/// ```
///
/// Result layout mirrors the flattened accumulator inputs.
#[pliron_op(
    name = "nvvm.wgmma_mma_loop_pipeline_values_m64n64k16_f32_bf16",
    format,
    attributes = (counted_max_pending_groups: WgmmaMaxPendingAttr)
)]
pub struct WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op;

impl WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Build a counted-loop WGMMA pipeline for a validated partial-wait depth.
    pub fn build(
        ctx: &mut Context,
        accumulators: Vec<Value>,
        desc_bases: Vec<Value>,
        desc_steps: Vec<Value>,
        trip_count: Value,
        max_pending_groups: u8,
    ) -> Ptr<Operation> {
        let f32_ty = FP32Type::get(ctx);
        let result_count = accumulators.len();
        let mut operands =
            Vec::with_capacity(result_count + desc_bases.len() + desc_steps.len() + 1);
        operands.extend(accumulators);
        operands.extend(desc_bases);
        operands.extend(desc_steps);
        operands.push(trip_count);

        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![f32_ty.into(); result_count],
            operands,
            vec![],
            0,
        );
        Self::new(op)
            .set_attr_counted_max_pending_groups(ctx, WgmmaMaxPendingAttr(max_pending_groups));
        op
    }

    /// Return the statically known `wait_group<N>` bound.
    pub fn max_pending_groups(&self, ctx: &Context) -> Option<u8> {
        self.get_attr_counted_max_pending_groups(ctx)
            .map(|attribute| attribute.0)
    }
}

impl Verify for WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let Some(max_pending_attr) = self.get_attr_counted_max_pending_groups(ctx) else {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_loop_pipeline_values_m64n64k16_f32_bf16 requires an nvvm.wgmma_max_pending_groups attribute"
            );
        };
        max_pending_attr.verify(ctx)?;
        let max_pending_groups = max_pending_attr.0;
        drop(max_pending_attr);

        if max_pending_groups > WGMMA_COUNTED_PIPELINE_MAX_PENDING_GROUPS {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_loop_pipeline_values_m64n64k16_f32_bf16 supports only max_pending_groups 1 or 2"
            );
        }

        let slot_count = usize::from(max_pending_groups) + 1;
        let expected_results = slot_count * WGMMA_M64N64_F32_ACCUMULATOR_COUNT;
        let expected_controls = slot_count * 4 + 1;
        let expected_operands = expected_results + expected_controls;

        if op.get_num_operands() != expected_operands {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_loop_pipeline_values_m64n64k16_f32_bf16 requires {} f32 accumulators and exactly {} u64 loop-control operands",
                expected_results,
                expected_controls
            );
        }
        if op.get_num_results() != expected_results {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_loop_pipeline_values_m64n64k16_f32_bf16 requires exactly {} f32 results",
                expected_results
            );
        }

        for accumulator_index in 0..expected_results {
            if !is_f32(ctx, op.get_operand(accumulator_index).get_type(ctx))
                || !is_f32(ctx, op.get_result(accumulator_index).get_type(ctx))
            {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_loop_pipeline_values_m64n64k16_f32_bf16 accumulator operands and results must be f32"
                );
            }
        }

        for control_index in expected_results..expected_operands {
            if !is_u64(ctx, op.get_operand(control_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_loop_pipeline_values_m64n64k16_f32_bf16 descriptor bases, descriptor steps, and trip count must be u64"
                );
            }
        }

        Ok(())
    }
}

/// The statically known `wait_group<N>` bound carried by value-form WGMMA
/// pipeline operations.
///
/// PTX `wgmma.wait_group.sync.aligned N;` throttles at most `N` pending
/// groups, with `N` in `1..=7` for a partial wait. `0` is the full drain the
/// non-pipelined group ops already model, and depths above 7 exceed the
/// hardware's pending-group bound, so both are rejected here.
#[pliron_attr(name = "nvvm.wgmma_max_pending_groups", format = "$0")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct WgmmaMaxPendingAttr(pub u8);

impl Verify for WgmmaMaxPendingAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if !(1..=WGMMA_MAX_PENDING_GROUPS).contains(&self.0) {
            return verify_err_noloc!(
                "nvvm.wgmma_max_pending_groups must be in 1..=7, got {}",
                self.0
            );
        }
        Ok(())
    }
}

/// Value-form BF16 WGMMA pipeline with multiple independent accumulator slots.
///
/// The operation carries `N + 1` independent 32-value accumulator slots when
/// `max_pending_groups = N`. Groups are issued round-robin across those slots,
/// committed individually, and throttled with `wait_group<N>` before a slot is
/// reused. A final `wait_group<0>` completes every pending group before any
/// accumulator result becomes visible to LLVM.
///
/// Operand layout:
///
/// ```text
/// [
///   slot0_acc_0, ..., slot0_acc_31,
///   ...
///   slotN_acc_0, ..., slotN_acc_31,
///   desc_a_0, desc_b_0, ..., desc_a_g, desc_b_g,
/// ]
/// ```
///
/// Result layout mirrors the flattened accumulator inputs.
#[pliron_op(
    name = "nvvm.wgmma_mma_pipeline_values_m64n64k16_f32_bf16",
    format,
    attributes = (max_pending_groups: WgmmaMaxPendingAttr)
)]
pub struct WgmmaMmaPipelineValuesM64N64K16F32Bf16Op;

impl WgmmaMmaPipelineValuesM64N64K16F32Bf16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Build a value-form WGMMA pipeline.
    pub fn build(
        ctx: &mut Context,
        accumulators: Vec<Value>,
        descriptors: Vec<Value>,
        max_pending_groups: u8,
    ) -> Ptr<Operation> {
        let f32_ty = FP32Type::get(ctx);
        let mut operands = Vec::with_capacity(accumulators.len() + descriptors.len());
        let result_count = accumulators.len();
        operands.extend(accumulators);
        operands.extend(descriptors);

        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![f32_ty.into(); result_count],
            operands,
            vec![],
            0,
        );
        Self::new(op).set_attr_max_pending_groups(ctx, WgmmaMaxPendingAttr(max_pending_groups));
        op
    }

    /// Return the statically known `wait_group<N>` bound.
    pub fn max_pending_groups(&self, ctx: &Context) -> Option<u8> {
        self.get_attr_max_pending_groups(ctx)
            .map(|attribute| attribute.0)
    }
}

impl Verify for WgmmaMmaPipelineValuesM64N64K16F32Bf16Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let Some(max_pending_attr) = self.get_attr_max_pending_groups(ctx) else {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_pipeline_values_m64n64k16_f32_bf16 requires an nvvm.wgmma_max_pending_groups attribute"
            );
        };
        max_pending_attr.verify(ctx)?;
        let max_pending_groups = max_pending_attr.0;
        drop(max_pending_attr);

        let result_count = op.get_num_results();
        if result_count == 0 || !result_count.is_multiple_of(WGMMA_M64N64_F32_ACCUMULATOR_COUNT) {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_pipeline_values_m64n64k16_f32_bf16 results must contain whole 32-value accumulator slots"
            );
        }
        let slot_count = result_count / WGMMA_M64N64_F32_ACCUMULATOR_COUNT;
        if slot_count != usize::from(max_pending_groups) + 1 {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_pipeline_values_m64n64k16_f32_bf16 requires exactly max_pending_groups + 1 accumulator slots"
            );
        }

        let operand_count = op.get_num_operands();
        if operand_count < result_count + slot_count * 2 {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_pipeline_values_m64n64k16_f32_bf16 requires at least one committed group per accumulator slot"
            );
        }
        let descriptor_count = operand_count - result_count;
        if !descriptor_count.is_multiple_of(2) {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_pipeline_values_m64n64k16_f32_bf16 descriptors must form pairs"
            );
        }
        let group_count = descriptor_count / 2;
        if group_count < slot_count {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_pipeline_values_m64n64k16_f32_bf16 requires at least max_pending_groups + 1 committed groups"
            );
        }

        for accumulator_index in 0..result_count {
            if !is_f32(ctx, op.get_operand(accumulator_index).get_type(ctx))
                || !is_f32(ctx, op.get_result(accumulator_index).get_type(ctx))
            {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_pipeline_values_m64n64k16_f32_bf16 accumulator operands and results must be f32"
                );
            }
        }
        for descriptor_index in result_count..operand_count {
            if !is_u64(ctx, op.get_operand(descriptor_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_pipeline_values_m64n64k16_f32_bf16 descriptors must be u64"
                );
            }
        }

        Ok(())
    }
}

/// Register WGMMA operations with the context.
pub(super) fn register(ctx: &mut Context) {
    WgmmaMaxPendingAttr::register(ctx);

    WgmmaMakeSmemDescOp::register(ctx);
    WgmmaMmaM64N64K16F32Bf16Op::register(ctx);
    WgmmaMmaM64N128K16F32Bf16Op::register(ctx);
    WgmmaMmaM64N64K16F32F16Op::register(ctx);
    WgmmaMmaM64N64K8F32Tf32Op::register(ctx);
    WgmmaMmaGroupM64N64K16F32Bf16Op::register(ctx);
    WgmmaMmaGroupValuesM64N64K16F32Bf16Op::register(ctx);
    WgmmaMmaGroupValuesM64N128K16F32Bf16Op::register(ctx);
    WgmmaMmaGroupValuesM64N64K16F32F16Op::register(ctx);
    WgmmaMmaGroupValuesM64N64K8F32Tf32Op::register(ctx);
    WgmmaMmaLoopValuesM64N64K16F32Bf16Op::register(ctx);
    WgmmaMmaLoopValuesM64N64K16F32F16Op::register(ctx);
    WgmmaMmaLoopPipelineValuesM64N64K16F32Bf16Op::register(ctx);
    WgmmaMmaPipelineValuesM64N64K16F32Bf16Op::register(ctx);
}
