/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Checked composition using existing buffer ownership and typed weight roles.
use crate::kernel::kernels;
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaStream, DeviceBuffer};
use cute_rs::{
    BFloat16,
    ffn::{FfnShape, FfnWeights},
};
use std::sync::Arc;

/// Host result carrying CUDA failures and rejected shape or context contracts.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
/// BF16 quiet-NaN bit pattern used to detect writes outside logical tensors.
pub const GUARD: u16 = 0x7fc1;
/// Number of extra BF16 sentinel elements appended to each workspace buffer.
pub const GUARD_LEN: usize = 8;

/// Reusable owned intermediate buffers; const widths prevent cross-model reuse.
pub struct Workspace<const H: usize, const I: usize> {
    /// Batch row count that a runner must match when reusing this workspace.
    rows: usize,
    /// Dense `[rows, I]` gate projection, rounded to BF16, plus trailing guards.
    pub gate: DeviceBuffer<BFloat16>,
    /// Dense `[rows, I]` up projection, rounded to BF16, plus trailing guards.
    pub up: DeviceBuffer<BFloat16>,
    /// Dense `[rows, I]` BF16 SiLU values before multiplication, plus guards.
    pub silu: DeviceBuffer<BFloat16>,
    /// Dense `[rows, I]` BF16 SiLU/up products used by the down projection, plus guards.
    pub product: DeviceBuffer<BFloat16>,
    /// Dense `[rows, H]` BF16 down-projection result, plus trailing guards.
    pub output: DeviceBuffer<BFloat16>,
}
impl<const H: usize, const I: usize> Workspace<H, I> {
    /// Allocate separate sentinel-filled buffers for this batch and the const widths.
    pub fn new(stream: &CudaStream, shape: FfnShape<H, I>) -> Result<Self> {
        let alloc = |len: usize| -> Result<DeviceBuffer<BFloat16>> {
            let total = len.checked_add(GUARD_LEN).ok_or("guard extent overflow")?;
            Ok(DeviceBuffer::from_host(
                stream,
                &vec![BFloat16::from_bits(GUARD); total],
            )?)
        };
        Ok(Self {
            rows: shape.rows(),
            gate: alloc(shape.intermediate().required_len())?,
            up: alloc(shape.intermediate().required_len())?,
            silu: alloc(shape.intermediate().required_len())?,
            product: alloc(shape.intermediate().required_len())?,
            output: alloc(shape.hidden().required_len())?,
        })
    }
}

/// Return a 128-thread grid with four warps, each owning one 16×8 output tile.
///
/// Dimensions are element counts. Zero rows return `None`; zero output width,
/// tile-count overflow, and grids beyond CUDA's signed grid-x limit are errors.
/// This check does not validate matrix allocation lengths or strides.
pub fn linear_grid(rows: usize, outputs: usize) -> Result<Option<LaunchConfig>> {
    if outputs == 0 {
        return Err("linear output width must be positive".into());
    }
    if rows == 0 {
        return Ok(None);
    }
    let tiles = rows
        .div_ceil(16)
        .checked_mul(outputs.div_ceil(8))
        .ok_or("tile count overflow")?;
    let blocks = tiles.div_ceil(4);
    if blocks > i32::MAX as usize {
        return Err("linear grid exceeds CUDA grid-x range".into());
    }
    Ok(Some(LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }))
}

/// The module and stream are bound to the same context at construction.
pub struct Runner {
    /// Kernel bundle loaded in the same CUDA context as the execution stream.
    module: kernels::LoadedModule,
    /// Stream ordering all projections and activation; each run waits for completion.
    stream: Arc<CudaStream>,
}
impl Runner {
    /// Load this example's kernels and retain the stream used for synchronous execution.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        Ok(Self {
            // SAFETY: the loaded artifact is generated from this example's
            // kernel source, including the checked one-warp decode entries.
            module: unsafe { kernels::load(stream.context())? },
            stream,
        })
    }
    /// Execute synchronously, ordering prior work in the owning context and
    /// retaining all buffer borrows through completion, including enqueue errors.
    ///
    /// Input uses the shape's row stride; weights retain `[output, input]`
    /// orientation. Intermediates and output are dense, with BF16 rounding after
    /// each projection, after SiLU, and after the SiLU/up product. Shape, buffer
    /// capacity, context, and launch errors are rejected before execution.
    /// A validated zero-row batch launches nothing.
    pub fn run<const H: usize, const I: usize>(
        &self,
        shape: FfnShape<H, I>,
        input: &DeviceBuffer<BFloat16>,
        weights: &FfnWeights<'_, H, I>,
        workspace: &mut Workspace<H, I>,
    ) -> Result<()> {
        shape.input().validate_len(input.len())?;
        if workspace.rows != shape.rows() {
            return Err("workspace batch does not match operation".into());
        }
        for buffer in [
            &workspace.gate,
            &workspace.up,
            &workspace.silu,
            &workspace.product,
        ] {
            shape.intermediate().validate_len(buffer.len())?;
        }
        shape.hidden().validate_len(workspace.output.len())?;
        for buffer in [
            input,
            weights.gate().buffer(),
            weights.up().buffer(),
            weights.down().buffer(),
            &workspace.gate,
            &workspace.up,
            &workspace.silu,
            &workspace.product,
            &workspace.output,
        ] {
            if !Arc::ptr_eq(buffer.context(), self.stream.context()) {
                return Err("FFN buffers must share the runner context".into());
            }
        }
        let Some(gate_grid) = linear_grid(shape.rows(), I)? else {
            return Ok(());
        };
        let down_grid = linear_grid(shape.rows(), H)?.ok_or("missing down grid")?;
        let length = shape.intermediate().required_len();
        let activation_tiles = length.div_ceil(8);
        let activation_blocks = activation_tiles.div_ceil(128);
        if activation_blocks > i32::MAX as usize {
            return Err("activation grid exceeds CUDA grid-x range".into());
        }
        let activation_grid = LaunchConfig {
            grid_dim: (activation_blocks as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // A buffer can have pending safe asynchronous writes on another stream.
        // This correctness-first runner has no readiness tracking, so establish
        // readiness for every borrowed buffer before launching any readers.
        self.stream.context().synchronize()?;
        // SAFETY: checked extents, typed weight roles, distinct owned workspace
        // allocations and complete warps. All launches use one stream/context;
        // the unconditional synchronization below retains borrows even on error.
        let result = (|| -> Result<()> {
            unsafe {
                self.module.linear_bf16(
                    &self.stream,
                    gate_grid,
                    input,
                    weights.gate().buffer(),
                    &mut workspace.gate,
                    shape.rows(),
                    H,
                    I,
                    shape.input().row_stride(),
                    weights.gate().shape().row_stride(),
                )?;
                self.module.linear_bf16(
                    &self.stream,
                    gate_grid,
                    input,
                    weights.up().buffer(),
                    &mut workspace.up,
                    shape.rows(),
                    H,
                    I,
                    shape.input().row_stride(),
                    weights.up().shape().row_stride(),
                )?;
                self.module.activate_bf16(
                    &self.stream,
                    activation_grid,
                    &workspace.gate,
                    &workspace.up,
                    &mut workspace.silu,
                    &mut workspace.product,
                    length,
                )?;
                self.module.linear_bf16(
                    &self.stream,
                    down_grid,
                    &workspace.product,
                    weights.down().buffer(),
                    &mut workspace.output,
                    shape.rows(),
                    I,
                    H,
                    I,
                    weights.down().shape().row_stride(),
                )?;
            }
            Ok(())
        })();
        let completion = self.stream.synchronize();
        result?;
        completion?;
        Ok(())
    }
    /// Exercise the exact SiLU rounding discriminator through the real kernel.
    pub fn check_rounding(&self) -> Result<()> {
        let gate = DeviceBuffer::from_host(&self.stream, &[BFloat16::from_bits(0x3f80)])?;
        let up = DeviceBuffer::from_host(&self.stream, &[BFloat16::from_bits(0x4001)])?;
        let mut silu = DeviceBuffer::from_host(&self.stream, &[BFloat16::from_bits(GUARD)])?;
        let mut product = DeviceBuffer::from_host(&self.stream, &[BFloat16::from_bits(GUARD)])?;
        unsafe {
            self.module.activate_bf16(
                &self.stream,
                LaunchConfig::for_num_elems(1),
                &gate,
                &up,
                &mut silu,
                &mut product,
                1,
            )?;
        }
        let mut actual = [BFloat16::from_bits(0)];
        silu.copy_to_host(&self.stream, &mut actual)?;
        assert_eq!(actual[0].to_bits(), 0x3f3b);
        product.copy_to_host(&self.stream, &mut actual)?;
        assert_eq!(
            actual[0].to_bits(),
            0x3fbc,
            "cast-once would produce 0x3fbd"
        );
        Ok(())
    }
}

/// CPU checks of projection tile counts and CUDA grid overflow rejection.
#[cfg(test)]
mod tests {
    use super::*;
    /// Check partial 16×8 tiles, empty batches, invalid widths, and overflow rejection.
    #[test]
    fn launch_geometry_handles_tails_and_rejects_overflow() {
        assert!(linear_grid(0, 1).unwrap().is_none());
        assert!(linear_grid(1, 0).is_err());
        assert_eq!(linear_grid(17, 9).unwrap().unwrap().grid_dim, (1, 1, 1));
        assert_eq!(linear_grid(33, 17).unwrap().unwrap().grid_dim, (3, 1, 1));
        assert!(linear_grid(usize::MAX, usize::MAX).is_err());
    }
}
