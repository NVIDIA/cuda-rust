/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA peer-to-peer (P2P) access management.
//!
//! Enables direct memory access between GPUs over NVLink or PCIe. Once enabled,
//! kernels on one device can read/write memory allocated on the peer device, and
//! `cuMemcpy` between devices avoids staging through host memory.
//!
//! P2P access depends on hardware topology. Use `can_access_peer` to query
//! support before calling `enable_peer_access`.

use crate::error::{DriverError, IntoResult};
use crate::simt::context::CudaContext;
use std::mem::MaybeUninit;
use std::sync::Arc;

/// Checks whether `from` can directly access memory on `to`.
///
/// Returns `true` if the two devices are connected via NVLink or PCIe with P2P
/// support, and the system topology allows direct access. Returns `false` if
/// P2P is not possible (e.g., different PCIe root complexes without a switch).
///
/// This is a query only -- it does not enable access. Call `enable_peer_access`
/// to actually enable it.
pub fn can_access_peer(from: &CudaContext, to: &CudaContext) -> Result<bool, DriverError> {
    let mut can_access = MaybeUninit::uninit();
    unsafe {
        cuda_bindings::cuDeviceCanAccessPeer(
            can_access.as_mut_ptr(),
            from.cu_device(),
            to.cu_device(),
        )
        .result()?;
        Ok(can_access.assume_init() != 0)
    }
}

/// Enables P2P access from `from`'s context to `to`'s device memory.
///
/// After this call, kernels running in `from`'s context can directly read/write
/// memory allocated in `to`'s context, and memory copies between the two devices
/// use the direct P2P path (NVLink or PCIe) instead of staging through host memory.
///
/// This is a one-directional operation. To enable bidirectional access, call this
/// function twice with swapped arguments.
///
/// Returns `Ok(())` if access was enabled or was already enabled. Returns an error
/// if the devices do not support P2P (check with `can_access_peer` first).
pub fn enable_peer_access(
    from: &Arc<CudaContext>,
    to: &Arc<CudaContext>,
) -> Result<(), DriverError> {
    from.bind_to_thread()?;
    let result = unsafe { cuda_bindings::cuCtxEnablePeerAccess(to.cu_ctx(), 0) };
    match result {
        cuda_bindings::cudaError_enum_CUDA_SUCCESS => Ok(()),
        cuda_bindings::cudaError_enum_CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => Ok(()),
        _ => result.result(),
    }
}

/// Disables P2P access from `from`'s context to `to`'s device memory.
///
/// After this call, direct memory access from `from` to `to` is no longer
/// possible. Any in-flight operations accessing peer memory must have completed
/// before calling this.
pub fn disable_peer_access(
    from: &Arc<CudaContext>,
    to: &Arc<CudaContext>,
) -> Result<(), DriverError> {
    from.bind_to_thread()?;
    let result = unsafe { cuda_bindings::cuCtxDisablePeerAccess(to.cu_ctx()) };
    match result {
        cuda_bindings::cudaError_enum_CUDA_SUCCESS => Ok(()),
        cuda_bindings::cudaError_enum_CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => Ok(()),
        _ => result.result(),
    }
}
