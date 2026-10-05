/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Demonstrates integrating a custom CUDA kernel with cuTile tile kernels.
//!
//! Pipeline:
//!   1. Tile kernel:   z = x + y       (element-wise add via cuTile)
//!   2. Custom kernel: w = scale * z   (element-wise scale via hand-written PTX)
//!
//! Shows:
//!   - Loading a kernel from inline PTX via `load_module_from_ptx`
//!   - Launching with `AsyncKernelLaunch` (`push_arg` / `push_device_ptr`)
//!   - Wrapping a custom kernel in a `DeviceOp` struct for a safe call-site
//!   - Chaining tile and custom kernels on the same stream with `and_then`

use cuda_async::device_context::{load_module_from_ptx, with_default_device_policy};
use cuda_async::device_future::DeviceFuture;
use cuda_async::device_operation::DeviceOp;
use cuda_async::device_operation::ExecutionContext;
use cuda_async::error::DeviceError;
use cuda_async::launch::AsyncKernelLaunch;
use cuda_core::{Function, LaunchConfig};
use cutile::api::{arange, zeros};
use cutile::tensor::{IntoPartition, Tensor, ToHostVec};
use std::future::IntoFuture;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Tile kernel: element-wise add — z = x + y
// ---------------------------------------------------------------------------

#[cutile::module]
mod tile_add {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = x.load_like(z);
        let tile_y = y.load_like(z);
        z.store(tile_x + tile_y);
    }
}

// ---------------------------------------------------------------------------
// Custom CUDA kernel: element-wise scale — out[i] = scale * in[i]
//
// Hand-written PTX targeting sm_52+ (JIT-compiled for the actual GPU at load
// time). Each thread handles one element; the grid is sized to cover all
// elements.
// ---------------------------------------------------------------------------

const SCALE_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry scale_f32(
    .param .u32 n,
    .param .f32 scale,
    .param .u64 in_ptr,
    .param .u64 out_ptr
)
{
    .reg .u32 %r<5>;
    .reg .u64 %rd<4>;
    .reg .f32 %f<3>;
    .reg .pred %p;

    ld.param.u32 %r0, [n];
    ld.param.f32 %f0, [scale];
    ld.param.u64 %rd0, [in_ptr];
    ld.param.u64 %rd1, [out_ptr];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.u32 %r4, %r2, %r3, %r1;

    setp.ge.u32 %p, %r4, %r0;
    @%p bra $done;

    cvt.u64.u32 %rd2, %r4;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd0, %rd0, %rd2;
    add.u64 %rd1, %rd1, %rd2;

    ld.global.f32 %f1, [%rd0];
    mul.f32 %f2, %f1, %f0;
    st.global.f32 [%rd1], %f2;

$done:
    ret;
}
";

// ---------------------------------------------------------------------------
// Safe DeviceOp wrapper for the custom scale kernel.
//
// The struct's typed fields enforce the correct argument signature.  `unsafe`
// is confined to the `execute` implementation where device pointers are
// marshalled — callers construct and `.await` this type without `unsafe`.
// ---------------------------------------------------------------------------

struct ScaleKernel {
    function: Arc<Function>,
    n: u32,
    scale: f32,
    input: Arc<Tensor<f32>>,
    output: Tensor<f32>,
}

impl DeviceOp for ScaleKernel {
    type Output = (Arc<Tensor<f32>>, Tensor<f32>);

    // Safety: execute is unsafe because it enqueues work on a CUDA stream
    // without synchronizing. The returned tensors may still be in-flight on
    // the GPU. Callers must synchronize the stream (e.g. via DeviceFuture)
    // before accessing the output.
    unsafe fn execute(
        self,
        ctx: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let mut launcher = AsyncKernelLaunch::new(self.function);
        launcher.push_arg(self.n);
        launcher.push_arg(self.scale);
        // Safety:
        // Input okay since it's behind an Arc.
        // Output okay since this struct owns the tensor we will be writing to.
        // Pointer validity / sizing — both cu_deviceptr() values come from Tensor allocations that are at least self.n f32 elements.
        // The kernel accesses elements 0..n, each at byte offset gid * 4, so both allocations are large enough.
        unsafe {
            launcher
                .push_device_ptr(self.input.device_pointer().cu_deviceptr())
                .push_device_ptr(self.output.device_pointer().cu_deviceptr());
        }
        launcher.set_launch_config(LaunchConfig {
            grid_dim: (self.n.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        });
        // Safety:
        // Kernel impl needs to guarantee freedom from UB.
        unsafe { launcher.execute(ctx)? };
        Ok((self.input, self.output))
    }
}

impl IntoFuture for ScaleKernel {
    type Output = Result<(Arc<Tensor<f32>>, Tensor<f32>), DeviceError>;
    type IntoFuture = DeviceFuture<(Arc<Tensor<f32>>, Tensor<f32>), ScaleKernel>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            Ok(DeviceFuture::scheduled(self, ExecutionContext::new(stream)))
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

// ---------------------------------------------------------------------------
// Main: run the tile add kernel, then the custom scale kernel, and verify.
// ---------------------------------------------------------------------------

#[tokio::main()]
async fn main() -> Result<(), cutile::error::Error> {
    if !cutile_examples::requirements::BASELINE.check("interop", 0)? {
        return Ok(());
    }

    let num_elements = 2usize.pow(5);
    let tile_size = 4usize;
    let scale = 3.0f32;
    let device_id = 0;

    // Step 1: Load the custom scale kernel from PTX.
    let module = load_module_from_ptx(SCALE_PTX, device_id)?;
    let scale_function = Arc::new(module.load_function("scale_f32")?);

    // Step 2: Allocate input tensors.
    let x: Arc<Tensor<f32>> = arange(num_elements).await?.into();
    let y: Arc<Tensor<f32>> = arange(num_elements).await?.into();
    let z: Tensor<f32> = zeros::<f32>(&[num_elements]).await?;

    // Step 3: Run tile add kernel — z = x + y.
    let (z_part, _x, _y) = tile_add::add(z.partition([tile_size]), x.clone(), y.clone()).await?;
    let z: Tensor<f32> = z_part.unpartition();

    // Step 4: Run custom scale kernel — w = scale * z.
    let w: Tensor<f32> = zeros::<f32>(&[num_elements]).await?;
    let (_z, w) = ScaleKernel {
        function: scale_function,
        n: num_elements as u32,
        scale,
        input: Arc::new(z),
        output: w,
    }
    .await?;

    // Step 5: Verify w[i] == scale * (x[i] + y[i]).
    let x_host = x.to_host_vec().await?;
    let y_host = y.to_host_vec().await?;
    let w_host = w.to_host_vec().await?;
    for i in 0..num_elements {
        let expected = scale * (x_host[i] + y_host[i]);
        println!(
            "{} * ({} + {}) = {} (got {})",
            scale, x_host[i], y_host[i], expected, w_host[i]
        );
        assert_eq!(expected, w_host[i]);
    }
    println!("Interop example passed.");
    Ok(())
}
