/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Asynchronous TMA output stores and their hardware queue lifecycle.

use super::*;

/// Require a positive store-ring stage count on a zero-result, zero-operand operation.
fn check_store_pipeline(
    pipeline: Option<CuteTmaStorePipelineAttr>,
    ctx: &Context,
) -> Result<(), String> {
    let pipeline = pipeline.ok_or_else(|| "must carry a store-pipeline config".to_owned())?;
    pipeline
        .verify(ctx)
        .map_err(|error| format!("has an invalid store-pipeline config: {error}"))?;
    if pipeline.stages != 1 {
        return Err("v0 supports the one-buffer store pipeline".to_owned());
    }
    Ok(())
}

/// Wait until the shared result buffer may be overwritten.
#[pliron_op(
    name = "cute.tma_store_acquire",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (tma_store_acquire_config: CuteTmaStorePipelineAttr)
)]
pub struct CuteTmaStoreAcquireOp;

impl CuteTmaStoreAcquireOp {
    /// Build a detached wait operation using the output ring's maximum pending group count.
    pub fn new(ctx: &mut Context, pipeline: CuteTmaStorePipelineAttr) -> Self {
        let operation = Self {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0),
        };
        operation.set_attr_tma_store_acquire_config(ctx, pipeline);
        operation
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Clone the static number of reusable output buffers, if present.
    #[must_use]
    pub fn pipeline(&self, ctx: &Context) -> Option<CuteTmaStorePipelineAttr> {
        self.get_attr_tma_store_acquire_config(ctx)
            .map(|pipeline| *pipeline)
    }
}

impl Verify for CuteTmaStoreAcquireOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 0 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.tma_store_acquire needs 0 operands and 0 results"
            );
        }
        if let Err(message) = check_store_pipeline(self.pipeline(ctx), ctx) {
            return verify_err!(op.loc(), "cute.tma_store_acquire {message}");
        }
        Ok(())
    }
}

/// Close this thread's pending TMA stores as one hardware copy group.
#[pliron_op(
    name = "cute.tma_store_commit",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (tma_store_commit_config: CuteTmaStorePipelineAttr)
)]
pub struct CuteTmaStoreCommitOp;

impl CuteTmaStoreCommitOp {
    /// Build a detached operation committing the issuer's pending TMA stores as one group.
    pub fn new(ctx: &mut Context, pipeline: CuteTmaStorePipelineAttr) -> Self {
        let operation = Self {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0),
        };
        operation.set_attr_tma_store_commit_config(ctx, pipeline);
        operation
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Clone the static number of reusable output buffers, if present.
    #[must_use]
    pub fn pipeline(&self, ctx: &Context) -> Option<CuteTmaStorePipelineAttr> {
        self.get_attr_tma_store_commit_config(ctx)
            .map(|pipeline| *pipeline)
    }
}

impl Verify for CuteTmaStoreCommitOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 0 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.tma_store_commit needs 0 operands and 0 results"
            );
        }
        if let Err(message) = check_store_pipeline(self.pipeline(ctx), ctx) {
            return verify_err!(op.loc(), "cute.tma_store_commit {message}");
        }
        Ok(())
    }
}

/// Before exit, wait until no TMA store still reads shared memory.
#[pliron_op(
    name = "cute.tma_store_tail",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (tma_store_tail_config: CuteTmaStorePipelineAttr)
)]
pub struct CuteTmaStoreTailOp;

impl CuteTmaStoreTailOp {
    /// Build a detached operation waiting for all committed stores to stop reading shared storage.
    pub fn new(ctx: &mut Context, pipeline: CuteTmaStorePipelineAttr) -> Self {
        let operation = Self {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0),
        };
        operation.set_attr_tma_store_tail_config(ctx, pipeline);
        operation
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Clone the static number of reusable output buffers, if present.
    #[must_use]
    pub fn pipeline(&self, ctx: &Context) -> Option<CuteTmaStorePipelineAttr> {
        self.get_attr_tma_store_tail_config(ctx)
            .map(|pipeline| *pipeline)
    }
}

impl Verify for CuteTmaStoreTailOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 0 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "cute.tma_store_tail needs 0 operands and 0 results"
            );
        }
        if let Err(message) = check_store_pipeline(self.pipeline(ctx), ctx) {
            return verify_err!(op.loc(), "cute.tma_store_tail {message}");
        }
        Ok(())
    }
}

/// Copy one typed shared tile to a descriptor-backed global destination.
///
/// Both operands are short-lived direct TMA-view ghosts. The source view owns
/// the ordinary shared pointer/capacity pair; the destination view owns the
/// immutable descriptor pointer.
#[pliron_op(
    name = "cute.tma_store_2d",
    format,
    interfaces = [NOpdsInterface<4>, NResultsInterface<0>]
)]
pub struct CuteTmaStore2dSemanticOp;

impl CuteTmaStore2dSemanticOp {
    /// Build a detached copy from a typed shared half to its descriptor-backed global tile.
    pub fn new(
        ctx: &mut Context,
        source: Value,
        destination: Value,
        tile_row: Value,
        tile_column: Value,
    ) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![source, destination, tile_row, tile_column],
                vec![],
                0,
            ),
        }
    }

    /// View an existing operation through this wrapper without allocating or verifying it.
    pub fn wrap(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Return the source view operand of the asynchronous transfer.
    #[must_use]
    pub fn source(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Return the destination view operand of the asynchronous transfer.
    #[must_use]
    pub fn destination(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    /// Return the runtime row index in units of complete TMA tiles.
    #[must_use]
    pub fn tile_row(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }

    /// Return the runtime column index in units of complete TMA tiles.
    #[must_use]
    pub fn tile_column(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(3)
    }
}

impl Verify for CuteTmaStore2dSemanticOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 4 || op.get_num_results() != 0 {
            return verify_err!(op.loc(), "cute.tma_store_2d needs 4 operands and 0 results");
        }
        let Some(source) = tma_view_of(ctx, self.source(ctx)) else {
            return verify_err!(op.loc(), "cute.tma_store_2d source must be a TMA view");
        };
        let Some(destination) = tma_view_of(ctx, self.destination(ctx)) else {
            return verify_err!(op.loc(), "cute.tma_store_2d destination must be a TMA view");
        };
        source.verify(ctx)?;
        destination.verify(ctx)?;
        let source_tensor = source.tensor_view(ctx).expect("verified source tensor");
        let destination_tensor = destination
            .tensor_view(ctx)
            .expect("verified destination tensor");
        if source_tensor.space != CuteTensorAddressSpaceAttr::Smem
            || source_tensor.access != CuteTensorAccessAttr::ReadWrite
            || destination_tensor.space != CuteTensorAddressSpaceAttr::Gmem
            || destination_tensor.access != CuteTensorAccessAttr::ReadWrite
        {
            return verify_err!(
                op.loc(),
                "cute.tma_store_2d direction must be writable Smem to writable Gmem"
            );
        }
        if source_tensor.storage != destination_tensor.storage
            || source.smem_layout != destination.smem_layout
        {
            return verify_err!(
                op.loc(),
                "cute.tma_store_2d source and destination need the same carrier and tile layout"
            );
        }
        if source_tensor
            .storage
            .deref(ctx)
            .downcast_ref::<MirFP16Type>()
            .is_none()
            || source.smem_layout.0 != current_half_layout()
            || source.tile_elements() != Some(8192)
            || source.tile_bytes(ctx) != Some(16 * 1024)
            || source_tensor.alignment.0 != 1024
            || destination_tensor.alignment.0 != 2
        {
            return verify_err!(
                op.loc(),
                "cute.tma_store_2d v0 needs one 1024-byte-aligned B128 128x64 f16 half"
            );
        }
        if !is_u64(ctx, self.tile_row(ctx)) || !is_u64(ctx, self.tile_column(ctx)) {
            return verify_err!(
                op.loc(),
                "cute.tma_store_2d row and column must be unsigned 64-bit integers"
            );
        }
        Ok(())
    }
}
