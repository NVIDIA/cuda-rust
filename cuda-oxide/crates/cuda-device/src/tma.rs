/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tensor Memory Accelerator (TMA) for async bulk tensor copies.
//!
//! TMA is a hardware unit on Hopper+ (sm_90+) that performs asynchronous
//! bulk memory copies without using thread resources. Unlike manual memory
//! copies, TMA operates as a DMA engine that frees threads for computation.
//!
//! # Architecture
//!
//! ```text
//! Traditional Copy (threads do work):
//! ┌─────────────┐    threads     ┌──────────────┐
//! │   Global    │ ──────────────►│    Shared    │
//! │   Memory    │   (expensive)  │    Memory    │
//! └─────────────┘                └──────────────┘
//!
//! TMA Copy (hardware DMA):
//! ┌─────────────┐      TMA       ┌──────────────┐
//! │   Global    │ ══════════════►│    Shared    │
//! │   Memory    │   (async DMA)  │    Memory    │
//! └─────────────┘                └──────────────┘
//!      │                              │
//!      └── Threads free to compute! ──┘
//! ```
//!
//! # Key Concepts
//!
//! 1. **TmaDescriptor**: A 128-byte descriptor created on the host that describes
//!    the tensor layout in global memory. Passed to kernels as a parameter.
//!
//! 2. **Async Copy**: `cp.async.bulk.tensor.*` instructions copy tiles from
//!    global memory to shared memory without blocking threads.
//!
//! 3. **Barrier Integration**: TMA completion is tracked via `mbarrier` - the
//!    hardware automatically signals the barrier when transfer completes.
//!
//! # Usage Pattern
//!
//! ```rust,ignore
//! use cuda_device::{kernel, thread, SharedArray};
//! use cuda_device::tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s};
//! use cuda_device::barrier::{Barrier, mbarrier_init, mbarrier_arrive, mbarrier_wait};
//!
//! #[kernel]
//! pub fn tma_copy_kernel(
//!     desc: *const TmaDescriptor,  // Host-created descriptor
//!     // ...
//! ) {
//!     static mut TILE: SharedArray<f32, 4096> = SharedArray::UNINIT;
//!     static mut BAR: Barrier = Barrier::UNINIT;
//!
//!     // Initialize barrier (thread 0 only)
//!     if thread::threadIdx_x() == 0 {
//!         unsafe { mbarrier_init(&raw mut BAR, 1); }
//!     }
//!     thread::sync_threads();
//!
//!     // Thread 0 initiates TMA copy
//!     if thread::threadIdx_x() == 0 {
//!         unsafe {
//!             cp_async_bulk_tensor_2d_g2s(
//!                 &raw mut TILE as *mut u8,  // Shared memory destination
//!                 desc,                       // TMA descriptor
//!                 tile_x, tile_y,            // Tile coordinates
//!                 &raw mut BAR,              // Barrier for completion
//!             );
//!         }
//!     }
//!
//!     // All threads wait for TMA completion
//!     let token = unsafe { mbarrier_arrive(&raw const BAR) };
//!     unsafe { mbarrier_wait(&raw const BAR, token); }
//!
//!     // Now shared memory contains the tile data
//! }
//! ```
//!
//! # Host-Side Descriptor Creation
//!
//! TMA descriptors are created on the host using the CUDA driver API:
//!
//! ```rust,ignore
//! use cuda_core::sys::*;
//!
//! // Create descriptor for 2D tensor
//! let mut desc = std::mem::MaybeUninit::<CUtensorMap>::uninit();
//! unsafe {
//!     cuTensorMapEncodeTiled(
//!         desc.as_mut_ptr(),
//!         CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
//!         2,  // 2D tensor
//!         device_ptr as *mut _,
//!         dims.as_ptr(),
//!         strides.as_ptr(),
//!         box_dims.as_ptr(),
//!         element_strides.as_ptr(),
//!         CU_TENSOR_MAP_INTERLEAVE_NONE,
//!         CU_TENSOR_MAP_SWIZZLE_NONE,
//!         CU_TENSOR_MAP_L2_PROMOTION_NONE,
//!         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
//!     );
//! }
//! ```
//!
//! # Hardware Support
//!
//! - **sm_90+ (Hopper)**: Full TMA support with 1D-5D tensors
//! - **sm_100+ (Blackwell)**: Enhanced TMA with additional features
//! - **sm_120 (Blackwell)**: Latest TMA capabilities

use crate::barrier::Barrier;

// =============================================================================
// TMA Descriptor Type
// =============================================================================

/// Opaque TMA descriptor (created on host, passed to kernel).
///
/// This is a 128-byte structure that describes the tensor layout in global
/// memory. The descriptor is created on the host using `cuTensorMapEncodeTiled`
/// and passed to the kernel as a parameter.
///
/// # Size
///
/// - CUDA 12.0-12.x: 128 bytes, 64-byte aligned
/// - CUDA 13.0+: 128 bytes, 128-byte aligned
///
/// # Safety
///
/// - Must be created on host via CUDA driver API
/// - Must remain valid for the duration of the kernel execution
/// - Contents are opaque - do not modify
#[repr(C, align(64))]
#[derive(Copy, Clone)]
pub struct TmaDescriptor {
    /// Opaque 128-byte descriptor data (16 x u64)
    _opaque: [u64; 16],
}

impl TmaDescriptor {
    /// Create an uninitialized descriptor.
    ///
    /// # Safety
    ///
    /// This creates invalid descriptor data. Only use for memory allocation;
    /// the descriptor must be properly initialized via `cuTensorMapEncodeTiled`
    /// on the host before use.
    pub const UNINIT: Self = Self { _opaque: [0; 16] };
}

include!("generated/tma.rs");
