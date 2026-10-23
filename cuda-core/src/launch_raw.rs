/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Context-aware raw CUDA launch helpers.

use crate::{launch_kernel, DriverError, Function, IntoResult, Stream};

/// Launches a CUDA kernel on a specific [`Stream`], binding its context first.
///
/// This is the usual host-side helper for `cuda_launch!` and async launches.
/// It ensures the stream's owning context is current before calling the raw
/// [`launch_kernel`] entry point, so callers do not need to manually call
/// [`crate::Context::bind_to_thread`] before every launch.
///
/// Unlike [`launch_kernel`], this helper works with typed wrappers rather than
/// raw driver handles. It is therefore the preferred API whenever you already
/// have a [`Function`] and [`Stream`].
///
/// # Safety
///
/// - `func` must refer to a kernel loaded from the same CUDA context that owns
///   `stream`.
/// - Each element of `kernel_params` must point to a region of memory that
///   matches the corresponding kernel parameter in size and alignment.
/// - The pointed-to argument values must remain valid until this function
///   returns.
/// - The grid and block dimensions must not exceed device limits.
/// - The grid, block, dynamic-shared-memory size, and launch mode must satisfy
///   every semantic invariant assumed by the kernel body, including index-space
///   uniqueness and synchronization requirements.
///
/// # Errors
///
/// Returns an error if binding `stream.context()` fails or if the underlying
/// `cuLaunchKernel` call rejects the launch.
#[inline]
pub unsafe fn launch_kernel_on_stream(
    func: &Function,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    stream: &Stream,
    kernel_params: &mut [*mut std::ffi::c_void],
) -> Result<(), DriverError> {
    stream.context().bind_to_thread()?;
    unsafe {
        launch_kernel(
            func.cu_function(),
            grid_dim,
            block_dim,
            shared_mem_bytes,
            stream.cu_stream(),
            kernel_params,
        )
    }
}

/// Low-level wrapper around `cuLaunchKernelEx`.
///
/// This is the cluster-aware launch path. It builds a `CUlaunchConfig` with a
/// `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION` attribute set to `cluster_dim`.
/// Required for thread-block cluster launches on sm_90+ (Hopper / Blackwell).
///
/// This helper performs **no context binding**. Prefer
/// [`launch_kernel_ex_on_stream`] in normal host-side code so the correct
/// stream context is made current automatically.
///
/// # Safety
///
/// Same preconditions as [`launch_kernel`], plus:
/// - Each component of `cluster_dim` must divide the corresponding `grid_dim`
///   component.
/// - The total cluster size must not exceed the device maximum.
/// - The device must support compute capability 9.0 or higher.
///
/// # Errors
///
/// Returns the CUDA driver error produced by `cuLaunchKernelEx` if launch
/// submission fails.
#[inline]
pub unsafe fn launch_kernel_ex(
    func: cuda_bindings::CUfunction,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    cluster_dim: (u32, u32, u32),
    stream: cuda_bindings::CUstream,
    kernel_params: &mut [*mut std::ffi::c_void],
) -> Result<(), DriverError> {
    // CUlaunchAttribute_st is opaque (see cuda-bindings/build.rs) for CUDA 13.2+
    // compatibility. C layout: { id: u32 @ 0, pad: [u8;4] @ 4, value: union @ 8 }.
    // clusterDim is three u32 fields (x, y, z) at offset 0 within the value union.
    let mut cluster_attr: cuda_bindings::CUlaunchAttribute_st = unsafe { std::mem::zeroed() };
    unsafe {
        let base = &mut cluster_attr as *mut _ as *mut u8;
        // id at offset 0
        (base as *mut u32)
            .write(cuda_bindings::CUlaunchAttributeID_enum_CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION);
        // clusterDim.x/y/z at offsets 8, 12, 16
        let dim_ptr = base.add(8) as *mut u32;
        dim_ptr.write(cluster_dim.0);
        dim_ptr.add(1).write(cluster_dim.1);
        dim_ptr.add(2).write(cluster_dim.2);
    }

    let config = cuda_bindings::CUlaunchConfig_st {
        gridDimX: grid_dim.0,
        gridDimY: grid_dim.1,
        gridDimZ: grid_dim.2,
        blockDimX: block_dim.0,
        blockDimY: block_dim.1,
        blockDimZ: block_dim.2,
        sharedMemBytes: shared_mem_bytes,
        hStream: stream,
        attrs: &mut cluster_attr,
        numAttrs: 1,
    };

    unsafe {
        cuda_bindings::cuLaunchKernelEx(
            &config,
            func,
            kernel_params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    }
    .result()
}

/// Launches a CUDA kernel with extended configuration on a specific stream,
/// binding the stream's owning context first.
///
/// This is the cluster-aware counterpart to [`launch_kernel_on_stream`]. It
/// binds `stream.context()` to the calling thread, then forwards to the raw
/// [`launch_kernel_ex`] helper.
///
/// # Safety
///
/// - `func` must refer to a kernel loaded from the same CUDA context that owns
///   `stream`.
/// - Each element of `kernel_params` must point to a region of memory that
///   matches the corresponding kernel parameter in size and alignment.
/// - The pointed-to argument values must remain valid until this function
///   returns.
/// - The grid, block, and cluster dimensions must satisfy the device limits and
///   CUDA cluster-launch requirements.
/// - The launch geometry, dynamic-shared-memory size, and cluster mode must
///   satisfy every semantic invariant assumed by the kernel body.
///
/// # Errors
///
/// Returns an error if binding `stream.context()` fails or if the underlying
/// `cuLaunchKernelEx` call rejects the launch.
#[inline]
pub unsafe fn launch_kernel_ex_on_stream(
    func: &Function,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    cluster_dim: (u32, u32, u32),
    stream: &Stream,
    kernel_params: &mut [*mut std::ffi::c_void],
) -> Result<(), DriverError> {
    stream.context().bind_to_thread()?;
    unsafe {
        launch_kernel_ex(
            func.cu_function(),
            grid_dim,
            block_dim,
            shared_mem_bytes,
            cluster_dim,
            stream.cu_stream(),
            kernel_params,
        )
    }
}

/// Low-level wrapper around `cuLaunchKernelEx` with the
/// `CU_LAUNCH_ATTRIBUTE_COOPERATIVE` flag set.
///
/// A *cooperative* launch guarantees that every block in the grid is
/// co-resident on the device, which is the precondition for grid-wide
/// barriers like `cuda_device::grid::sync()`. The CUDA driver also
/// populates PTX environment registers `%envreg1` / `%envreg2` with the
/// pointer to the per-launch grid workspace; the device-side barrier
/// implementation reads those registers to find the shared counter.
///
/// This helper performs **no context binding**. Prefer
/// [`launch_kernel_cooperative_on_stream`] in normal host-side code so the
/// correct stream context is made current automatically.
///
/// # Safety
///
/// Same preconditions as [`launch_kernel`], plus:
/// - The device must support cooperative launch (`CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH`).
/// - The grid must fit in the maximum number of resident blocks for this
///   kernel as reported by `cuOccupancyMaxActiveBlocksPerMultiprocessor`
///   (otherwise `cuLaunchKernelEx` returns
///   `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`).
///
/// # Errors
///
/// Returns the CUDA driver error produced by `cuLaunchKernelEx` if launch
/// submission fails.
#[inline]
pub unsafe fn launch_kernel_cooperative(
    func: cuda_bindings::CUfunction,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    stream: cuda_bindings::CUstream,
    kernel_params: &mut [*mut std::ffi::c_void],
) -> Result<(), DriverError> {
    // CUlaunchAttribute_st is opaque (see cuda-bindings/build.rs) for CUDA 13.2+
    // compatibility. C layout: { id: u32 @ 0, pad: [u8;4] @ 4, value: union @ 8 }.
    // For the COOPERATIVE attribute the value union holds a single `int cooperative`
    // at offset 0 — set to 1 to enable, 0 to disable.
    let mut coop_attr: cuda_bindings::CUlaunchAttribute_st = unsafe { std::mem::zeroed() };
    unsafe {
        let base = &mut coop_attr as *mut _ as *mut u8;
        (base as *mut u32)
            .write(cuda_bindings::CUlaunchAttributeID_enum_CU_LAUNCH_ATTRIBUTE_COOPERATIVE);
        let val_ptr = base.add(8) as *mut i32;
        val_ptr.write(1);
    }

    let config = cuda_bindings::CUlaunchConfig_st {
        gridDimX: grid_dim.0,
        gridDimY: grid_dim.1,
        gridDimZ: grid_dim.2,
        blockDimX: block_dim.0,
        blockDimY: block_dim.1,
        blockDimZ: block_dim.2,
        sharedMemBytes: shared_mem_bytes,
        hStream: stream,
        attrs: &mut coop_attr,
        numAttrs: 1,
    };

    unsafe {
        cuda_bindings::cuLaunchKernelEx(
            &config,
            func,
            kernel_params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    }
    .result()
}

/// Launches a cooperative CUDA kernel on a specific stream, binding the
/// stream's owning context first.
///
/// This is the cooperative-launch counterpart to [`launch_kernel_on_stream`].
/// It binds `stream.context()` to the calling thread, then forwards to the
/// raw [`launch_kernel_cooperative`] helper.
///
/// # Safety
///
/// Same preconditions as [`launch_kernel_cooperative`].
///
/// # Errors
///
/// Returns an error if binding `stream.context()` fails or if the underlying
/// `cuLaunchKernelEx` call rejects the launch.
#[inline]
pub unsafe fn launch_kernel_cooperative_on_stream(
    func: &Function,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    stream: &Stream,
    kernel_params: &mut [*mut std::ffi::c_void],
) -> Result<(), DriverError> {
    stream.context().bind_to_thread()?;
    unsafe {
        launch_kernel_cooperative(
            func.cu_function(),
            grid_dim,
            block_dim,
            shared_mem_bytes,
            stream.cu_stream(),
            kernel_params,
        )
    }
}

/// Low-level wrapper around `cuLaunchKernelEx` with **both** the
/// `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION` and
/// `CU_LAUNCH_ATTRIBUTE_COOPERATIVE` attributes set.
///
/// `cuLaunchKernelEx` takes an array of launch attributes, so a single call
/// can request thread-block clusters ([`launch_kernel_ex`]) and a
/// cooperative launch ([`launch_kernel_cooperative`]) at the same time.
/// This is the path used when a `#[cuda_module]` kernel carries both
/// `#[cluster_launch(...)]` and `#[cooperative_launch]`.
///
/// This helper performs **no context binding**. Prefer
/// [`launch_kernel_ex_cooperative_on_stream`] in normal host-side code so
/// the correct stream context is made current automatically.
///
/// # Safety
///
/// The combined preconditions of [`launch_kernel_ex`] (cluster dimensions
/// must divide the grid, sm_90+) and [`launch_kernel_cooperative`] (the
/// device must support cooperative launch and the whole grid must be
/// co-resident).
///
/// # Errors
///
/// Returns the CUDA driver error produced by `cuLaunchKernelEx` if launch
/// submission fails.
#[inline]
pub unsafe fn launch_kernel_ex_cooperative(
    func: cuda_bindings::CUfunction,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    cluster_dim: (u32, u32, u32),
    stream: cuda_bindings::CUstream,
    kernel_params: &mut [*mut std::ffi::c_void],
) -> Result<(), DriverError> {
    // CUlaunchAttribute_st is opaque (see cuda-bindings/build.rs) for CUDA 13.2+
    // compatibility. C layout: { id: u32 @ 0, pad: [u8;4] @ 4, value: union @ 8 }.
    // attrs[0]: clusterDim — three u32 fields (x, y, z) at offset 0 of the union.
    // attrs[1]: cooperative — a single `int` at offset 0 of the union; 1 = enabled.
    let mut attrs: [cuda_bindings::CUlaunchAttribute_st; 2] = unsafe { std::mem::zeroed() };
    unsafe {
        let base = &mut attrs[0] as *mut _ as *mut u8;
        (base as *mut u32)
            .write(cuda_bindings::CUlaunchAttributeID_enum_CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION);
        let dim_ptr = base.add(8) as *mut u32;
        dim_ptr.write(cluster_dim.0);
        dim_ptr.add(1).write(cluster_dim.1);
        dim_ptr.add(2).write(cluster_dim.2);

        let base = &mut attrs[1] as *mut _ as *mut u8;
        (base as *mut u32)
            .write(cuda_bindings::CUlaunchAttributeID_enum_CU_LAUNCH_ATTRIBUTE_COOPERATIVE);
        let val_ptr = base.add(8) as *mut i32;
        val_ptr.write(1);
    }

    let config = cuda_bindings::CUlaunchConfig_st {
        gridDimX: grid_dim.0,
        gridDimY: grid_dim.1,
        gridDimZ: grid_dim.2,
        blockDimX: block_dim.0,
        blockDimY: block_dim.1,
        blockDimZ: block_dim.2,
        sharedMemBytes: shared_mem_bytes,
        hStream: stream,
        attrs: attrs.as_mut_ptr(),
        numAttrs: 2,
    };

    unsafe {
        cuda_bindings::cuLaunchKernelEx(
            &config,
            func,
            kernel_params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    }
    .result()
}

/// Launches a cooperative CUDA kernel with cluster dimensions on a specific
/// stream, binding the stream's owning context first.
///
/// This is the cluster-plus-cooperative counterpart to
/// [`launch_kernel_on_stream`]. It binds `stream.context()` to the calling
/// thread, then forwards to the raw [`launch_kernel_ex_cooperative`] helper.
///
/// # Safety
///
/// Same preconditions as [`launch_kernel_ex_cooperative`].
///
/// # Errors
///
/// Returns an error if binding `stream.context()` fails or if the underlying
/// `cuLaunchKernelEx` call rejects the launch.
#[inline]
pub unsafe fn launch_kernel_ex_cooperative_on_stream(
    func: &Function,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    cluster_dim: (u32, u32, u32),
    stream: &Stream,
    kernel_params: &mut [*mut std::ffi::c_void],
) -> Result<(), DriverError> {
    stream.context().bind_to_thread()?;
    unsafe {
        launch_kernel_ex_cooperative(
            func.cu_function(),
            grid_dim,
            block_dim,
            shared_mem_bytes,
            cluster_dim,
            stream.cu_stream(),
            kernel_params,
        )
    }
}
