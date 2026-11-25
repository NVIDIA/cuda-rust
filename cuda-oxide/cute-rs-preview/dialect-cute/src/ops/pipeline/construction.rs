/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Load-pipeline handles and barrier-ring initialization.

use super::*;

/// Bind one shared-memory barrier allocation to a TMA load-pipeline config.
///
/// The base points at this exact physical order:
///
/// ```text
/// [full 0 ... full S-1][empty 0 ... empty S-1]
/// ```
///
/// MIR represents CUDA's 8-byte `Barrier` storage as one unsigned `u64`.
#[pliron_op(
    name = "cute.tma_load_pipeline_make",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>, OneResultInterface],
    attributes = (pipeline_base_alignment_bytes: CuteAlignmentAttr)
)]
pub struct CuteTmaLoadPipelineMakeOp;

impl CuteTmaLoadPipelineMakeOp {
    /// Create a compiler-only view over an existing shared barrier ring.
    pub fn new(
        ctx: &mut Context,
        base: Value,
        stages: u64,
        consumer_warps: u32,
        transaction_bytes: u32,
        base_alignment_bytes: u64,
    ) -> Self {
        let pipeline: TypeHandle =
            CuteTmaLoadPipelineType::get(ctx, stages, consumer_warps, transaction_bytes).into();
        let operation = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![pipeline],
                vec![base],
                vec![],
                0,
            ),
        };
        operation
            .set_attr_pipeline_base_alignment_bytes(ctx, CuteAlignmentAttr(base_alignment_bytes));
        operation
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Shared base of the first full barrier.
    #[must_use]
    pub fn base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Compiler-only pipeline handle.
    #[must_use]
    pub fn pipeline(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }

    /// Alignment promised for the start of the whole barrier ring.
    #[must_use]
    pub fn promised_base_alignment(&self, ctx: &Context) -> Option<u64> {
        self.get_attr_pipeline_base_alignment_bytes(ctx)
            .map(|alignment| alignment.0)
    }

    /// Static pipeline configuration carried by the result type.
    #[must_use]
    pub fn pipeline_type(&self, ctx: &Context) -> Option<CuteTmaLoadPipelineType> {
        pipeline_type_of(ctx, self.pipeline(ctx))
    }
}

impl Verify for CuteTmaLoadPipelineMakeOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "cute.tma_load_pipeline_make needs 1 operand and 1 result"
            );
        }
        if !valid_shared_barrier_pointer(ctx, self.base(ctx)) {
            return verify_err!(
                op.loc(),
                "cute.tma_load_pipeline_make base must be a mutable shared pointer to the canonical unsigned u64 Barrier storage"
            );
        }
        let Some(pipeline) = self.pipeline_type(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.tma_load_pipeline_make result must be a TMA load-pipeline handle"
            );
        };
        pipeline.verify(ctx)?;
        let Some(alignment) = self.promised_base_alignment(ctx) else {
            return verify_err!(
                op.loc(),
                "cute.tma_load_pipeline_make must carry a base-alignment promise"
            );
        };
        CuteAlignmentAttr(alignment).verify(ctx)?;
        if alignment < BARRIER_BYTES {
            return verify_err!(
                op.loc(),
                "cute.tma_load_pipeline_make base alignment must be at least 8 bytes"
            );
        }
        if self
            .pipeline(ctx)
            .uses(ctx)
            .into_iter()
            .any(|r#use| !allowed_pipeline_user(ctx, r#use.user_op()))
        {
            return verify_err!(
                op.loc(),
                "cute.tma_load_pipeline handle may only feed load-pipeline lifecycle operations"
            );
        }
        Ok(())
    }
}

/// Initialize all full/empty barriers once, then publish them to the CTA.
#[pliron_op(
    name = "cute.tma_load_pipeline_init",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>]
)]
pub struct CuteTmaLoadPipelineInitOp;

impl CuteTmaLoadPipelineInitOp {
    /// Build a detached initialization/publication operation for both barrier rings.
    pub fn new(ctx: &mut Context, pipeline: Value, init_thread: Value) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![pipeline, init_thread],
                vec![],
                0,
            ),
        }
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Return the direct compiler-only load-pipeline handle operand.
    #[must_use]
    pub fn pipeline(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Thread index that initializes every barrier. All threads synchronize.
    #[must_use]
    pub fn init_thread(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

impl Verify for CuteTmaLoadPipelineInitOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.tma_load_pipeline_init needs 2 operands and 0 results"
            );
        }
        if let Err(message) = checked_pipeline(ctx, self.pipeline(ctx)) {
            return verify_err!(op.loc(), "cute.tma_load_pipeline_init {message}");
        }
        if !is_u32(ctx, self.init_thread(ctx)) {
            return verify_err!(
                op.loc(),
                "cute.tma_load_pipeline_init thread index must be an unsigned 32-bit integer"
            );
        }
        Ok(())
    }
}
