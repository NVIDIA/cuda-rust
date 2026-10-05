/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(non_snake_case)]
//! Thread Block Cluster intrinsics for Hopper (sm_90+).
//!
//! Thread Block Clusters are a new hierarchy level introduced in NVIDIA Hopper
//! that groups multiple thread blocks together. Blocks within a cluster can:
//!
//! 1. **Directly access each other's shared memory** (Distributed Shared Memory)
//! 2. **Synchronize at cluster granularity** (faster than grid-wide sync)
//! 3. **Coordinate TMA operations** across blocks
//!
//! # Hierarchy
//!
//! ```text
//! Grid
//! └── Thread Block Cluster (NEW - sm_90+)
//!     └── Thread Block
//!         └── Warp
//!             └── Thread
//! ```
//!
//! # Hardware Requirements
//!
//! - Minimum: sm_90 (Hopper: H100, H200)
//! - Blackwell (sm_100/sm_120) extends cluster capabilities
//!
//! # Cluster Dimensions
//!
//! Clusters are defined as a 3D arrangement of thread blocks:
//! - Maximum total cluster size: 8 blocks
//! - `clusterDimX × clusterDimY × clusterDimZ ≤ 8`
//!
//! # Example
//!
//! ```rust,ignore
//! use cuda_device::{kernel, thread, cluster, SharedArray, DisjointSlice};
//!
//! #[kernel]
//! pub fn cluster_example(mut output: DisjointSlice<u32>) {
//!     static mut SHMEM: SharedArray<u32, 1> = SharedArray::UNINIT;
//!
//!     let tid = thread::threadIdx_x();
//!     let my_rank = cluster::block_rank();
//!     let cluster_size = cluster::cluster_size();
//!
//!     // Each block writes to shared memory
//!     if tid == 0 {
//!         unsafe { SHMEM.as_mut_ptr().write(my_rank * 100) };
//!     }
//!     thread::sync_threads();
//!
//!     // Sync entire cluster
//!     cluster::cluster_sync();
//!
//!     // Read neighbor's shared memory
//!     let neighbor = (my_rank + 1) % cluster_size;
//!     let neighbor_ptr = unsafe { cluster::map_shared_rank(SHMEM.as_ptr(), neighbor) };
//!     let value = unsafe { *neighbor_ptr };
//! }
//! ```

// =============================================================================
// Cluster Position Intrinsics (Block's position within cluster)
// =============================================================================

/// Get block's X position within cluster.
///
/// Returns a value in range `[0, cluster_nctaidX)`.
///
/// This is the block's coordinate within the cluster, analogous to how
/// `threadIdx.x` is the thread's coordinate within a block.
///
/// # PTX
///
/// Lowers to: `mov.u32 %r, %cluster_ctaid.x`
#[inline(never)]
pub fn cluster_ctaidX() -> u32 {
    unreachable!("cluster_ctaidX called outside CUDA kernel context")
}

/// Get block's Y position within cluster.
///
/// Returns a value in range `[0, cluster_nctaidY)`.
///
/// # PTX
///
/// Lowers to: `mov.u32 %r, %cluster_ctaid.y`
#[inline(never)]
pub fn cluster_ctaidY() -> u32 {
    unreachable!("cluster_ctaidY called outside CUDA kernel context")
}

/// Get block's Z position within cluster.
///
/// Returns a value in range `[0, cluster_nctaidZ)`.
///
/// # PTX
///
/// Lowers to: `mov.u32 %r, %cluster_ctaid.z`
#[inline(never)]
pub fn cluster_ctaidZ() -> u32 {
    unreachable!("cluster_ctaidZ called outside CUDA kernel context")
}

// =============================================================================
// Cluster Dimension Intrinsics (Size of the cluster)
// =============================================================================

/// Get cluster X dimension (number of blocks per cluster in X).
///
/// This is the cluster's size in the X dimension, analogous to how
/// `blockDim.x` is the block's size in threads.
///
/// # PTX
///
/// Lowers to: `mov.u32 %r, %cluster_nctaid.x`
#[inline(never)]
pub fn cluster_nctaidX() -> u32 {
    unreachable!("cluster_nctaidX called outside CUDA kernel context")
}

/// Get cluster Y dimension (number of blocks per cluster in Y).
///
/// # PTX
///
/// Lowers to: `mov.u32 %r, %cluster_nctaid.y`
#[inline(never)]
pub fn cluster_nctaidY() -> u32 {
    unreachable!("cluster_nctaidY called outside CUDA kernel context")
}

/// Get cluster Z dimension (number of blocks per cluster in Z).
///
/// # PTX
///
/// Lowers to: `mov.u32 %r, %cluster_nctaid.z`
#[inline(never)]
pub fn cluster_nctaidZ() -> u32 {
    unreachable!("cluster_nctaidZ called outside CUDA kernel context")
}

include!("generated/cluster_sreg.rs");

// =============================================================================
// Cluster Index Intrinsics (Cluster's position within grid)
// =============================================================================

/// Get cluster's linear index within the grid.
///
/// This identifies which cluster this block belongs to within the grid,
/// analogous to `blockIdx.x` for blocks within a grid.
///
/// # PTX
///
/// Lowers to documented `%clusterid.{x,y,z}` / `%nclusterid.{x,y}` reads and
/// computes `x + y * nx + z * nx * ny`.
#[inline(always)]
pub fn cluster_idx() -> u32 {
    let x = __cluster_idxX();
    let y = __cluster_idxY();
    let z = __cluster_idxZ();
    let nx = __cluster_grid_dimX();
    let ny = __cluster_grid_dimY();
    x + y * nx + z * nx * ny
}

/// Get total number of clusters in the grid.
///
/// # PTX
///
/// Lowers to documented `%nclusterid.{x,y,z}` reads and computes `nx * ny * nz`.
#[inline(always)]
pub fn num_clusters() -> u32 {
    __cluster_grid_dimX() * __cluster_grid_dimY() * __cluster_grid_dimZ()
}

// =============================================================================
// Derived Helpers
// =============================================================================

/// Get block's linear rank within cluster.
///
/// Computes: `cluster_ctaidX + cluster_ctaidY * cluster_nctaidX + cluster_ctaidZ * cluster_nctaidX * cluster_nctaidY`
///
/// This is useful for addressing blocks within a cluster, especially for
/// distributed shared memory operations.
///
/// # Returns
///
/// A value in range `[0, cluster_size)`.
#[inline(always)]
pub fn block_rank() -> u32 {
    let x = cluster_ctaidX();
    let y = cluster_ctaidY();
    let z = cluster_ctaidZ();
    let nx = cluster_nctaidX();
    let ny = cluster_nctaidY();
    x + y * nx + z * nx * ny
}

/// Get total number of blocks in the cluster.
///
/// Computes: `cluster_nctaidX * cluster_nctaidY * cluster_nctaidZ`
#[inline(always)]
pub fn cluster_size() -> u32 {
    cluster_nctaidX() * cluster_nctaidY() * cluster_nctaidZ()
}

// =============================================================================
// Cluster Synchronization
// =============================================================================

include!("generated/cluster_barrier.rs");

/// Synchronize all blocks in the cluster.
///
/// All threads in all blocks of the cluster must reach this barrier before
/// any thread can proceed. This is a cluster-wide barrier, similar to how
/// `sync_threads()` is a block-wide barrier.
///
/// # Usage
///
/// ```rust,ignore
/// // Each block writes to its shared memory
/// if thread::threadIdx_x() == 0 {
///     unsafe { SHMEM.as_mut_ptr().write(value) };
/// }
/// thread::sync_threads();  // Block-local sync first
///
/// // Cluster-wide sync - all blocks have written
/// cluster::cluster_sync();
///
/// // Now safe to read other blocks' shared memory via DSMEM
/// ```
///
/// # Safety
///
/// - All threads in all blocks of the cluster must reach the same barrier
/// - Placing `cluster_sync()` inside a conditional where not all threads enter
///   will cause deadlock
///
/// # PTX
///
/// Lowers to an aligned cluster-barrier arrive followed by wait.
#[inline(never)]
pub fn cluster_sync() {
    // Keep the safe helper while routing both instructions through generated intrinsics.
    unsafe {
        barrier_cluster_arrive_aligned();
        barrier_cluster_wait_aligned();
    }
}

// =============================================================================
// Distributed Shared Memory
// =============================================================================

include!("generated/cluster_memory.rs");

// =============================================================================
// Compile-Time Cluster Configuration
// =============================================================================

/// Marker function for compile-time cluster configuration.
///
/// This is a compile-time configuration marker that tells the compiler to emit
/// `.reqnctapercluster` PTX directive for this kernel. It does NOT generate any
/// runtime code - it only configures the kernel's cluster dimensions at compile time.
///
/// # Usage
///
/// This function should NOT be called directly. Use the `#[cluster_launch(x, y, z)]`
/// attribute macro instead, which injects this marker:
///
/// ```rust,ignore
/// #[kernel]
/// #[cluster_launch(4, 1, 1)]
/// pub fn my_cluster_kernel(output: DisjointSlice<u32>) {
///     // Cluster of 4 blocks in X dimension
/// }
/// ```
///
/// # How It Works
///
/// 1. The `#[cluster_launch]` macro injects `__cluster_config::<X, Y, Z>()` at kernel start
/// 2. MIR importer detects this call and extracts the const generic parameters
/// 3. The marker call is NOT compiled - it's removed during compilation
/// 4. LLVM export emits `!nvvm.annotations` with `cluster_dim_x/y/z` metadata
/// 5. LLVM NVPTX backend emits `.reqnctapercluster X, Y, Z` in PTX
///
/// # PTX Output
///
/// ```ptx
/// .entry my_cluster_kernel .reqnctapercluster 4, 1, 1 { ... }
/// ```
#[inline(never)]
pub fn __cluster_config<const X: u32, const Y: u32, const Z: u32>() {
    // This function is detected at compile time and removed.
    // The const generics X, Y, Z are extracted to set cluster dimensions.
    // No runtime code is generated.
}
