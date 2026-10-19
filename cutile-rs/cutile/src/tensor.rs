/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! GPU tensor types and partitioning primitives.
//!
//! This module provides the core [`Tensor`] type for GPU memory management and the [`Partition`]
//! type for dividing tensors into tiles that map to CUDA thread blocks.
//!
//! ## Overview
//!
//! This module is the foundation for GPU memory management in cuTile Rust. It provides:
//!
//! - **[`Tensor`]** - Smart pointer to GPU memory with shape and stride information
//! - **[`Partition`]** - View of a tensor divided into tiles for parallel processing
//! - **Traits** - For converting between tensors, partitions, and device operations
//!
//! ## Core Types
//!
//! ### Tensor
//!
//! A [`Tensor<T>`] represents a multi-dimensional array stored in GPU memory. Key features:
//!
//! - **Automatic memory management**: Uses RAII via [`DeviceBuffer`]
//! - **Shape tracking**: Maintains shape and stride information
//! - **Zero-copy operations**: Reshape and view operations don't copy data
//! - **Safe concurrency**: `Send + Sync` for safe sharing across async tasks
//!
//! ### Partition
//!
//! A [`Partition<Tensor<T>>`] divides a tensor into tiles (blocks) for GPU kernels. Each tile
//! maps to one CUDA thread block, enabling efficient parallel processing.
//!
//! Key features:
//! - **Grid inference**: Automatically calculates launch grid from partition shape
//! - **Partial tiles**: The grid is the ceiling division `shape / partition_shape` per
//!   axis, so an extent that is not a multiple of the partition shape gets a partial
//!   edge tile (bounds-checked or padded in the kernel) rather than being rejected
//! - **Zero-cost abstraction**: No runtime overhead, just metadata
//!
//! ## Traits
//!
//! ### IntoPartition
//!
//! The [`IntoPartition`] trait enables partitioning tensors:
//!
//! ```rust,ignore
//! use cutile::api;
//! use cutile::tensor::IntoPartition;
//!
//! let tensor = api::zeros(&[256]).await;
//! let partitioned = tensor.partition([64]);  // 4 tiles
//! assert_eq!(partitioned.grid(), (4, 1, 1));
//! ```
//!
//! ### Unpartition
//!
//! The [`Unpartition`] trait removes partition structure, returning the underlying tensor:
//!
//! ```rust,ignore
//! let tensor = partitioned.unpartition();
//! ```
//!
//! ### ToHostVec
//!
//! The [`ToHostVec`] trait provides convenient GPU → CPU data transfer:
//!
//! ```rust,ignore
//! use cutile::tensor::ToHostVec;
//!
//! let tensor = api::ones(&[1024]).await;
//! let host_vec: Vec<f32> = tensor.to_host_vec().await;
//! ```
//!
//! ## Memory Layout
//!
//! Tensors use row-major (C-style) memory layout by default:
//!
//! ```text
//! 2D Tensor [3, 4]:
//! Memory: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
//! Shape:  +-------------+
//!         | 0  1  2  3  |  Row 0
//!         | 4  5  6  7  |  Row 1
//!         | 8  9 10 11  |  Row 2
//!         +-------------+
//! Strides: [4, 1]  (4 elements between rows, 1 between columns)
//! ```
//!
//! ## Partitioning Example
//!
//! ```text
//! Tensor [256] partitioned into [64]:
//!
//! +---------+---------+---------+---------+
//! | Tile 0  | Tile 1  | Tile 2  | Tile 3  |
//! | [0:64)  | [64:128)| [128:192| [192:256|
//! +---------+---------+---------+---------+
//!
//! Launch grid: (4, 1, 1)
//! Each CUDA block processes one tile (64 elements)
//! ```
//!
//! ```text
//! Tensor [128, 128] partitioned into [32, 32]:
//!
//! +------+------+------+------+
//! | 0,0  | 0,1  | 0,2  | 0,3  |  4x4 grid of tiles
//! +------+------+------+------+  Each tile: 32x32 elements
//! | 1,0  | 1,1  | 1,2  | 1,3  |  Grid: (4, 4, 1)
//! +------+------+------+------+  Total: 16 thread blocks
//! | 2,0  | 2,1  | 2,2  | 2,3  |
//! +------+------+------+------+
//! | 3,0  | 3,1  | 3,2  | 3,3  |
//! +------+------+------+------+
//! ```
//!
//! ## Examples
//!
//! ### Basic Tensor Operations
//!
//! ```rust,ignore
//! use cutile::api;
//!
//! // Create tensor
//! let tensor = api::zeros::<f32>(&[1024]).await;
//!
//! // Access properties
//! println!("Shape: {:?}", tensor.shape());
//! println!("Size: {}", tensor.size());
//! println!("Bytes: {}", tensor.num_bytes());
//! ```
//!
//! ### Partitioning for Kernels
//!
//! ```rust,ignore
//! use cutile::api;
//! use cutile::tensor::IntoPartition;
//!
//! let tensor = api::zeros(&[256]).await;
//! let partitioned = tensor.partition([64]);
//!
//! // Use in kernel launch
//! // Each of 4 thread blocks processes 64 elements
//! ```
//!
//! ### Copying to Host
//!
//! ```rust,ignore
//! use cutile::api;
//! use cutile::tensor::ToHostVec;
//!
//! let gpu_tensor = api::ones(&[1024]).await;
//! let cpu_vec: Vec<f32> = gpu_tensor.to_host_vec().await;
//! assert_eq!(cpu_vec.len(), 1024);
//! ```
//!
//! ### Working with Arc
//!
//! ```rust,ignore
//! use cutile::api;
//! use cutile::tensor::IntoPartitionArc;
//! use std::sync::Arc;
//!
//! let tensor = Arc::new(api::zeros(&[256]).await);
//!
//! // Can partition Arc<Tensor> directly
//! let partitioned = tensor.partition_arc([64]);
//! ```
//!
//! ## Safety and Concurrency
//!
//! ### Thread Safety
//!
//! - `Tensor<T>` is `Send + Sync` - safe to share across threads
//! - `Partition<Tensor<T>>` is `Send + Sync` but not `Clone`
//! - GPU memory is freed automatically when the last reference is dropped
//!
//! ### Memory Safety
//!
//! Partitioning ensures that each thread block accesses disjoint memory regions:
//!
//! ```rust,ignore
//! // Safe: Each block writes to non-overlapping tiles
//! let z = api::zeros(&[256]).partition([64]);
//! // Block 0: writes to [0:64)
//! // Block 1: writes to [64:128)
//! // Block 2: writes to [128:192)
//! // Block 3: writes to [192:256)
//! ```
//!
//! ## Performance Considerations
//!
//! - **Partitioning**: Zero-cost abstraction (just metadata)
//! - **Reshaping**: Zero-cost (updates strides, no data copy)
//! - **Copying**: Expensive (requires GPU memory bandwidth)
//! - **Host transfers**: Very expensive (PCIe bandwidth-limited)
//!
//! ## See Also
//!
//! - [`api`](crate::api) - High-level tensor creation functions
//! - [`tile_kernel`](crate::tile_kernel) - Async execution infrastructure
//! - [`core`](crate::core) - GPU kernel DSL types

use crate::api::{copy_device_to_host_vec, copy_host_vec_to_device};
use crate::error::{tensor_error_result, Error};
use crate::tile_kernel::UnwrapPartition;
use anyhow::Result;
use cuda_async::device_buffer::{DeviceAllocation, DeviceBuffer, DevicePointer};
use cuda_async::device_operation::{
    value, AccessLease, AccessTracked, DeviceOp, ExecutionContext, IntoDeviceOp, ReplayResource,
    Value,
};
use cuda_async::error::DeviceError;
use cuda_core::sys::CUdeviceptr;
use cuda_core::{DType, DTypeId};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::{align_of, size_of, MaybeUninit};
use std::ops::Index;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};

/// A partitioned view of a tensor that divides it into tiles for GPU kernel processing.
///
/// `Partition` wraps a tensor and adds partition shape and stride information, enabling
/// tile-based GPU kernels to process the data in blocks that map to CUDA thread blocks.
/// Each thread block processes one partition (tile) of the tensor.
///
/// ## Memory Safety
///
/// This type is `Send + Sync` but not `Clone` or `Copy`. It provides tile kernels with
/// mutable access to disjoint regions of memory, making parallel access safe. When wrapped
/// in an `Arc`, the Arc prevents mutable access, maintaining safety.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::api;
///
/// // Create a tensor and partition it into 64-element tiles
/// let tensor = api::ones(&[256]).await;
/// let partitioned = tensor.partition([64]);
///
/// // The partition has 4 tiles: (256 / 64 = 4)
/// // Grid will be (4, 1, 1)
/// assert_eq!(partitioned.grid(), (4, 1, 1));
/// ```
///
/// ## Grid Inference
///
/// Partitions automatically calculate the launch grid for kernels:
///
/// ```rust,ignore
/// let x = api::zeros(&[128, 128]).partition([32, 32]);
/// assert_eq!(x.grid(), (4, 4, 1)); // 128/32 = 4 in each dimension
/// ```
pub struct Partition<T> {
    pub(crate) object: T,
    pub partition_shape: Vec<usize>,
    pub partition_strides: Vec<usize>,
    /// `true` iff this binding opted into partial coverage
    /// ([`IntoPartition::partition_prefix`]): the launch grid may be a
    /// per-axis PREFIX of this partition's block grid instead of equal to
    /// it. Launched blocks embed identically; blocks beyond the launch grid
    /// are simply never visited (they keep their prior contents).
    pub(crate) prefix_coverage: bool,
}

impl<T> Partition<T> {
    /// Opts this binding into partial coverage: the launch grid may be a
    /// per-axis PREFIX of the inferred block grid (`launch <= inferred` on
    /// every axis) instead of equal to it. Launched blocks map to exactly
    /// the blocks they would under full coverage — a per-axis prefix is the
    /// identity embedding into the block grid, so exclusivity and
    /// disjointness are unchanged — and blocks beyond the launch grid are
    /// never visited, keeping their prior contents. Exceeding the inferred
    /// grid on ANY axis remains a hard launch error: that direction is
    /// genuine out-of-bounds. Prefer the [`IntoPartition::partition_prefix`]
    /// spelling at call sites.
    pub fn prefix(mut self) -> Self {
        self.prefix_coverage = true;
        self
    }

    /// Unwraps the partition to retrieve the underlying object.
    ///
    /// This consumes the partition and returns the original tensor or value.
    pub fn unpartition(self) -> T {
        self.object
    }

    /// Attach a host-side partition map to this partition.
    ///
    /// Mapped partitions separate the logical partition grid from the CUDA
    /// tile-block grid. The map validates that the requested physical
    /// tile-block count can safely traverse the logical partition grid before
    /// the kernel launches. Launch-grid inference uses the mapped partition's
    /// physical tile-block grid, `(num_tile_blocks, 1, 1)`, rather than the
    /// logical partition grid. An explicit `.grid(...)` or `.const_grid(...)`
    /// may still be used as a checked override.
    ///
    /// ```rust,ignore
    /// let z = api::zeros::<f32>(&[128, 256]).await;
    ///
    /// // Launch 8 tile blocks to traverse the mapped output tiles.
    /// let output = z.partition([32, 64]).map([4, 1], 8);
    /// ```
    ///
    /// A map dimension of [`OWNED`] marks an owned axis: the axis is not
    /// traversed by the work-item stream; each stream item owns the axis's
    /// full extent (subtensor-per-CTA exclusivity). `iter_indices()` then
    /// yields one item per streamed-axes coordinate, and the owned axes are
    /// traversed by in-kernel loops:
    ///
    /// ```rust,ignore
    /// // One stream item per axis-0 tile; each item owns all of axis 1.
    /// let output = z.partition([1, 2048]).map([1, OWNED], rows);
    /// ```
    pub fn map<const RANK: usize>(
        self,
        map_shape: [usize; RANK],
        num_tile_blocks: u32,
    ) -> MappedLaunchPartition<Self> {
        assert!(
            !self.prefix_coverage,
            "a partial-coverage (prefix) partition cannot be mapped: a mapped \
             schedule covers its full index space by construction"
        );
        MappedLaunchPartition {
            partition: self,
            map_shape: map_shape.to_vec(),
            num_tile_blocks,
        }
    }
}

/// Map-shape sentinel marking an owned axis.
///
/// An owned axis is not traversed by the mapped work-item stream: each stream
/// item owns the axis's full extent, so in-kernel loops traverse it with
/// plain (`Dim`-bounded) indices while the minted stream index proves
/// disjointness on the streamed axes only.
pub const OWNED: usize = 0;

/// Host-side mapped partition launch argument.
///
/// This wrapper is intentionally separate from the device-side
/// `MappedPartitionMut<T, D, M>` DSL type. The host wrapper owns validation and
/// launch-policy inference; the device type should expose map-specific
/// partition indices inside a kernel.
pub struct MappedLaunchPartition<P> {
    pub(crate) partition: P,
    pub(crate) map_shape: Vec<usize>,
    pub(crate) num_tile_blocks: u32,
}

impl<P> MappedLaunchPartition<P> {
    fn validate(
        &self,
        partition_grid: (u32, u32, u32),
        num_tile_blocks: u32,
    ) -> Result<(u32, u32, u32), Error> {
        let map_rank = self.map_shape.len();
        if map_rank == 0 || map_rank > 3 {
            return tensor_error_result(
                "mapped partitions require a rank-1 through rank-3 map shape.",
            );
        }
        // A map dimension of OWNED (0) marks an owned axis: not traversed by
        // the stream, so it contributes nothing to the streamed tile count.
        if self.map_shape.iter().all(|&dim| dim == OWNED) {
            return tensor_error_result(
                "mapped partition requires at least one streamed (non-OWNED) map axis.",
            );
        }
        let grid_axes = [partition_grid.0, partition_grid.1, partition_grid.2];
        if grid_axes.iter().take(map_rank).any(|&axis| axis == 0) {
            return tensor_error_result(
                "mapped partition requires a non-empty logical partition grid.",
            );
        }
        if grid_axes.iter().skip(map_rank).any(|&axis| axis != 1) {
            return tensor_error_result(
                "mapped partition map rank must match the logical partition grid rank.",
            );
        }
        let streamed_tiles = grid_axes
            .iter()
            .zip(self.map_shape.iter())
            .filter(|&(_, &map_dim)| map_dim != OWNED)
            .map(|(&axis, _)| axis)
            .try_fold(1u32, |total, axis| total.checked_mul(axis))
            .ok_or_else(|| {
                crate::error::tensor_error("mapped partition logical grid is too large")
            })?;
        if num_tile_blocks == 0 {
            return tensor_error_result("mapped partition requires num_tile_blocks > 0.");
        }
        if num_tile_blocks > streamed_tiles {
            return tensor_error_result(
                "mapped partition num_tile_blocks cannot exceed the streamed logical tile count.",
            );
        }
        Ok((num_tile_blocks, 1, 1))
    }
}

impl<T: DType> Partition<Tensor<T>> {
    /// Returns the total size of the tensor in bytes.
    pub fn num_bytes(&self) -> usize {
        self.object.size() * size_of::<T>()
    }

    /// Returns the size of the tensor in megabytes (base 10).
    pub fn num_mb(&self) -> usize {
        self.num_bytes() / 10usize.pow(6)
    }

    /// Returns the size of the tensor in gigabytes (base 10).
    pub fn num_gb(&self) -> usize {
        self.num_bytes() / 10usize.pow(9)
    }

    /// Returns the data type of the tensor elements.
    pub fn dtype(&self) -> DTypeId {
        T::DTYPE
    }

    /// Returns the data type name as a string.
    pub fn dtype_str(&self) -> &'static str {
        T::DTYPE.as_str()
    }

    /// Calculates the CUDA launch grid dimensions based on the partition.
    ///
    /// The grid is the ceiling division `tensor_shape / partition_shape` per
    /// dimension, so a tensor whose extent is not a multiple of the partition
    /// shape gets a partial edge tile rather than being rejected. Supports 1D,
    /// 2D, and 3D tensors.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x = api::zeros(&[256]).partition([64]);
    /// assert_eq!(x.grid()?, (4, 1, 1));
    ///
    /// let y = api::zeros(&[100, 256]).partition([32, 64]);
    /// assert_eq!(y.grid()?, (4, 4, 1)); // 100 / 32 rounds up to 4
    /// ```
    ///
    /// ## Errors
    ///
    /// Returns `Err` if the tensor rank is greater than 3, a tensor dimension
    /// is not positive, a partition dimension is zero, or the partition rank
    /// differs from the tensor rank.
    pub fn grid(&self) -> Result<(u32, u32, u32), Error> {
        partition_launch_grid(&self.object.shape, &self.partition_shape)
    }
}

impl<T> From<Partition<T>> for Arc<T> {
    fn from(val: Partition<T>) -> Self {
        Arc::new(val.unpartition())
    }
}

/// Enables partitioning a value into tiles.
///
/// This trait allows values to be divided into partitions for tile-based processing.
/// The partition shape determines how the value is divided across thread blocks.
pub trait IntoPartition {
    /// Partitions this value with the specified partition shape.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let tensor = api::zeros(&[1024]).await;
    /// let partitioned = tensor.partition([128]); // 8 partitions
    /// ```
    fn partition<const RANK: usize>(self, partition_shape: [usize; RANK]) -> Partition<Self>
    where
        Self: Sized;

    /// Partitions with PARTIAL coverage: the launch grid may be a per-axis
    /// prefix of the block grid instead of equal to it (see
    /// [`Partition::prefix`] for the contract). The strict-equality
    /// diagnostic of [`Self::partition`] remains the default; this spelling
    /// is the explicit opt-in for kernels that deliberately cover only an
    /// aligned prefix and leave the remainder to another kernel.
    fn partition_prefix<const RANK: usize>(self, partition_shape: [usize; RANK]) -> Partition<Self>
    where
        Self: Sized,
    {
        self.partition(partition_shape).prefix()
    }
}

/// Enables partitioning an `Arc`-wrapped value into tiles.
///
/// This trait is similar to [`IntoPartition`] but works with `Arc`-wrapped values,
/// consuming the `Arc` to create a partition. This is commonly used with async operations.
pub trait IntoPartitionArc {
    /// Partitions this Arc-wrapped value with the specified partition shape.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let tensor = Arc::new(api::zeros(&[1024]).await);
    /// let partitioned = tensor.partition([128]);
    /// ```
    fn partition<const RANK: usize>(
        self: Arc<Self>,
        partition_shape: [usize; RANK],
    ) -> Partition<Self>
    where
        Self: Sized;
}

pub use cutile_compiler::specialization::{compute_spec, SpecializationBits};

/// Backing storage for a [`Tensor`].
///
/// A tensor is normally backed by a real GPU allocation (`Device`). The `Meta`
/// variant carries only the byte length and device id — no device memory — and
/// is produced by [`crate::api::meta`] for kernel warmup: it lets `.compile()`
/// build the cache key from shape/stride/spec metadata without allocating.
///
/// Reading the device pointer of a `Meta` tensor panics. `cu_deviceptr` is only
/// reached on real execution paths (kernel launch arg push, host copies), so a
/// meta tensor is accepted by `.compile()` (metadata only) yet rejected the
/// moment anything tries to actually run it. Reshape/view/slice recompute spec
/// through `spec_ptr` instead, which never dereferences.
#[derive(Debug)]
pub(crate) struct Storage {
    allocation: Allocation,
    accesses: Mutex<Vec<StorageAccess>>,
}

#[derive(Debug)]
enum Allocation {
    /// A real, owned GPU allocation.
    Device(DeviceBuffer),
    /// Metadata-only placeholder: valid shape/stride/spec, no device memory.
    Meta { len_bytes: usize, device_id: usize },
}

impl Storage {
    fn new(allocation: Allocation) -> Self {
        Self {
            allocation,
            accesses: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn retain(
        self: &Arc<Self>,
        ctx: &ExecutionContext,
        write: bool,
    ) -> Result<(), DeviceError> {
        if self.device_id() != ctx.get_device_id() {
            return Err(DeviceError::Internal(
                "tensor and execution stream are on different devices".into(),
            ));
        }
        let id = self.register_access(
            ctx.get_cuda_stream().cu_stream() as usize,
            write,
            ctx.is_recording(),
        )?;
        if ctx.is_recording() {
            // Owned by the graph; reacquired (as a plain lease) on each replay.
            ctx.record_resource(Arc::new(StorageLease {
                storage: self.clone(),
                id,
                write,
            }));
            Ok(())
        } else {
            // Hot path: a refcount and an inline push, no allocation.
            ctx.retain_lease(AccessLease::new(self.clone(), id))
        }
    }

    /// Test spelling of [`register_access`](Self::register_access) that hands
    /// back a lease, so an access is released when the lease drops.
    #[cfg(test)]
    fn acquire(
        self: &Arc<Self>,
        stream: usize,
        write: bool,
        recording: bool,
    ) -> Result<StorageLease, DeviceError> {
        let id = self.register_access(stream, write, recording)?;
        Ok(StorageLease {
            storage: self.clone(),
            id,
            write,
        })
    }

    /// Registers an in-flight access and returns its id; released through
    /// [`AccessTracked::release_access`] when the holding submission completes.
    fn register_access(
        self: &Arc<Self>,
        stream: usize,
        write: bool,
        recording: bool,
    ) -> Result<u64, DeviceError> {
        let mut accesses = self.accesses.lock().unwrap_or_else(|e| e.into_inner());
        if !recording
            && accesses.iter().any(|access| {
                !access.recording && access.stream != stream && (write || access.write)
            })
        {
            return Err(DeviceError::Internal(
                "tensor has a conflicting in-flight access on another stream; await its operation before reusing it".into(),
            ));
        }
        let id = NEXT_ACCESS_ID.fetch_add(1, AtomicOrdering::Relaxed);
        accesses.push(StorageAccess {
            id,
            stream,
            write,
            recording,
        });
        Ok(id)
    }

    fn is_unique(self: &Arc<Self>) -> bool {
        let accesses = self.accesses.lock().unwrap_or_else(|e| e.into_inner());
        // Each lease has exactly one internal Arc. Only user-visible aliases
        // prevent partitioning; same-stream leases are ordered on submission.
        Arc::strong_count(self) == 1 + accesses.len()
    }

    /// 16-aligned sentinel address fed to [`compute_spec`] for meta tensors.
    ///
    /// `DivHint::from_ptr` clamps pointer alignment to 16, and every
    /// cutile-owned device allocation is >=16-byte aligned, so this yields
    /// the exact same `base_ptr_div` an owned tensor would — keeping the
    /// warmup cache key byte-identical to the launch key. Foreign/borrowed
    /// pointers (`from_foreign`, `borrow_raw_parts`) may be less aligned;
    /// they get their own measured `base_ptr_div` and therefore their own
    /// specialization — a warmup-key mismatch at worst, never wrong code —
    /// so the sentinel deliberately reflects only the owned common case.
    const META_SPEC_PTR: u64 = 16;

    fn len_bytes(&self) -> usize {
        match &self.allocation {
            Allocation::Device(b) => b.len_bytes(),
            Allocation::Meta { len_bytes, .. } => *len_bytes,
        }
    }

    fn device_id(&self) -> usize {
        match &self.allocation {
            Allocation::Device(b) => b.device_id(),
            Allocation::Meta { device_id, .. } => *device_id,
        }
    }

    fn cu_deviceptr(&self) -> CUdeviceptr {
        match &self.allocation {
            Allocation::Device(b) => b.cu_deviceptr(),
            Allocation::Meta { .. } => panic!(
                "cutile: read of a meta tensor's device pointer. Meta tensors \
                 (api::meta) carry only shape/stride metadata for warmup via \
                 `.compile()`; they have no device memory and cannot be launched \
                 or copied. Use a real tensor (api::zeros/ones/...) on `.sync()` \
                 / `.await` paths."
            ),
        }
    }

    /// Pointer for recomputing [`compute_spec`] on reshape/view/slice — never
    /// dereferenced, so unlike [`cu_deviceptr`](Self::cu_deviceptr) it doesn't
    /// panic on `Meta`. Meta returns the sentinel, keeping warmup of reshaping
    /// kernels on the meta path with a `base_ptr_div` that matches a real tensor's.
    fn spec_ptr(&self) -> CUdeviceptr {
        match &self.allocation {
            Allocation::Device(b) => b.cu_deviceptr(),
            Allocation::Meta { .. } => Storage::META_SPEC_PTR,
        }
    }
}

/// Process-wide access ids; never reused, so a stale release cannot match.
static NEXT_ACCESS_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
struct StorageAccess {
    id: u64,
    stream: usize,
    write: bool,
    recording: bool,
}

impl AccessTracked for Storage {
    fn release_access(&self, id: u64) {
        // Remove the entry before the lease drops its Arc, so uniqueness
        // checks can be conservative during release but can never overlook
        // a real alias.
        let mut accesses = self.accesses.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pos) = accesses.iter().position(|access| access.id == id) {
            accesses.swap_remove(pos);
        }
    }
}

/// A recorded (graph-owned) access: alive for as long as the graph can
/// replay, and reacquired as a plain lease on each replay.
struct StorageLease {
    storage: Arc<Storage>,
    id: u64,
    write: bool,
}

impl ReplayResource for StorageLease {
    fn retain_for_launch(&self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        self.storage.retain(ctx, self.write)
    }
    fn replay_identity(&self) -> (usize, bool) {
        (Arc::as_ptr(&self.storage) as *const () as usize, self.write)
    }
}

impl Drop for StorageLease {
    fn drop(&mut self) {
        self.storage.release_access(self.id);
    }
}

/// A multi-dimensional array stored in GPU memory.
///
/// `Tensor` is the primary type for working with GPU data in cuTile Rust. It pairs a
/// shared backing storage (a [`DeviceBuffer`], or metadata only for [`crate::api::meta`])
/// with shape, stride, and specialization metadata, providing a typed, multi-dimensional
/// view of GPU memory.
///
/// ## Memory Management
///
/// The backing storage is reference counted: it is freed when the last tensor that
/// shares it is dropped. Zero-copy views over the same storage are created through
/// `Arc<Tensor<T>>` ([`Reshape`] for `&Arc<Tensor<T>>`, [`Tensor::reinterpret`]) or
/// by borrowing ([`Tensor::view`], [`Tensor::slice`]). Mutable partitioning requires
/// the storage to be unshared.
///
/// ## Examples
///
/// ### Creating tensors
///
/// ```rust,ignore
/// use cutile::api;
///
/// // Create tensors using the API
/// let x = api::zeros::<f32>(&[1024]).await;
/// let y = api::ones::<f32>(&[512, 512]).await;
/// let z = api::arange::<i32>(256).await;
/// ```
///
/// ### Copying and reshaping
///
/// ```rust,ignore
/// let x: Tensor<f32> = api::zeros(&[1024]).await;
///
/// // Duplicate to create a new tensor with the same data
/// let y: Tensor<f32> = x.dup().await;
///
/// // Reshape (must preserve total size and be contiguous)
/// let reshaped = y.reshape(&[32, 32])?; // 1024 = 32 * 32
/// ```
///
/// ### Transferring to host
///
/// ```rust,ignore
/// use cutile::tensor::ToHostVec;
///
/// let gpu_tensor = api::arange::<f32>(100).await;
/// let cpu_vec: Vec<f32> = gpu_tensor.to_host_vec().await;
/// ```
#[derive(Debug)]
pub struct Tensor<T: DType> {
    pub(crate) storage: Arc<Storage>,
    pub(crate) shape: Vec<i32>,
    pub(crate) strides: Vec<i32>,
    pub(crate) spec: SpecializationBits,
    _dtype: PhantomData<T>,
}

// Computes row-major contiguous strides for a given shape.
fn contiguous_strides(shape: &[i32]) -> Vec<i32> {
    let mut stride = 1;
    let mut strides = Vec::with_capacity(shape.len());
    for dim in shape.iter().rev() {
        strides.push(stride);
        stride *= *dim;
    }
    strides.reverse();
    strides
}

// Multiplies shape dimensions with overflow checks to recover the logical element count.
fn checked_num_elements(shape: &[usize]) -> Result<usize, Error> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| crate::error::tensor_error("Tensor shape overflowed usize."))
    })
}

// Computes the logical byte size for a typed shape while guarding against overflow.
fn checked_num_bytes<T>(shape: &[usize]) -> Result<usize, Error> {
    checked_num_elements(shape)?
        .checked_mul(size_of::<T>())
        .ok_or_else(|| crate::error::tensor_error("Tensor byte size overflowed usize."))
}

// Variant of checked_num_elements for i32-backed metadata, rejecting negative dimensions.
fn checked_num_elements_i32(shape: &[i32]) -> Result<usize, Error> {
    shape.iter().try_fold(1usize, |acc, dim| {
        let dim = usize::try_from(*dim)
            .map_err(|_| crate::error::tensor_error("Tensor shape contains negative dimension."))?;
        acc.checked_mul(dim)
            .ok_or_else(|| crate::error::tensor_error("Tensor shape overflowed usize."))
    })
}

// Computes the logical byte size for i32-backed tensor metadata.
fn checked_num_bytes_i32<T>(shape: &[i32]) -> Result<usize, Error> {
    checked_num_elements_i32(shape)?
        .checked_mul(size_of::<T>())
        .ok_or_else(|| crate::error::tensor_error("Tensor byte size overflowed usize."))
}

// Launch grid for a partition binding: the ceiling division of every tensor axis by the
// matching partition axis, so a partial edge tile still gets a block. Both shapes are
// caller-supplied, so every step is checked: a zero partition axis would divide by zero,
// a rank mismatch would index past the shorter vector, and a non-positive tensor axis
// has no blocks to launch.
fn partition_launch_grid(
    shape: &[i32],
    partition_shape: &[usize],
) -> Result<(u32, u32, u32), Error> {
    if shape.iter().any(|&d| d <= 0) {
        return tensor_error_result(&format!(
            "Shape dimensions must be positive, got {shape:?}."
        ));
    }
    if partition_shape.len() != shape.len() {
        return tensor_error_result(&format!(
            "Partition rank {} does not match tensor rank {} (partition shape {partition_shape:?}, tensor shape {shape:?}).",
            partition_shape.len(),
            shape.len(),
        ));
    }
    if partition_shape.contains(&0) {
        return tensor_error_result(&format!(
            "Partition dimensions must be positive, got {partition_shape:?}."
        ));
    }
    let axis = |i: usize| -> Result<u32, Error> {
        // `shape[i] > 0` was checked above, so the cast is lossless.
        let extent = shape[i] as u32;
        let tile = u32::try_from(partition_shape[i]).map_err(|_| {
            crate::error::tensor_error(&format!(
                "Partition dimension {} exceeds u32::MAX.",
                partition_shape[i]
            ))
        })?;
        Ok(extent.div_ceil(tile))
    };
    match shape.len() {
        1 => Ok((axis(0)?, 1, 1)),
        2 => Ok((axis(0)?, axis(1)?, 1)),
        3 => Ok((axis(0)?, axis(1)?, axis(2)?)),
        _ => tensor_error_result("Mutable tensor must be at most rank 3."),
    }
}

// Validates that `target` is a legal zero-copy reinterpretation of a tensor or view whose
// metadata is `(shape, strides)`. The source must be contiguous row-major — a view assumes
// contiguous strides for `target`, which address different elements over any other
// layout — and both sides must describe the same byte size. The comparison uses
// overflow-checked `usize` arithmetic: a caller-supplied shape can make a wrapping `i32`
// product (or an `as i32` truncation) agree with the current element count while
// describing billions of elements beyond the backing storage.
fn validate_view_shape_for<T>(
    shape: &[i32],
    strides: &[i32],
    target: &[usize],
) -> Result<(), Error> {
    if strides != contiguous_strides(shape) {
        return tensor_error_result(&format!(
            "Zero-copy tensor views require contiguous storage (shape {shape:?}, strides {strides:?})."
        ));
    }
    let target_num_bytes = checked_num_bytes::<T>(target)?;
    let current_num_bytes = checked_num_bytes_i32::<T>(shape)?;
    if target_num_bytes != current_num_bytes {
        return tensor_error_result(&format!(
            "View shape must preserve tensor size: {target:?} does not have the same number of elements as {shape:?}."
        ));
    }
    Ok(())
}

// Largest byte extent any element of `(shape, strides)` addresses from the base
// pointer: `(1 + Σ (shape[i]-1) * strides[i]) * size_of::<T>()`. Strides can
// push this beyond `product(shape) * size_of::<T>()`, so it — not the logical
// size — is what a backing allocation must cover. Assumes non-negative strides
// (a negative stride would place addresses below the base, which the safe
// `from_foreign` cannot bound; empty shapes address nothing).
fn addressable_bytes<T>(shape: &[i32], strides: &[i32]) -> usize {
    assert_eq!(
        shape.len(),
        strides.len(),
        "Tensor shape/stride rank mismatch."
    );
    if shape.iter().any(|&d| d <= 0) {
        return 0;
    }
    // Checked throughout: shape/strides are caller-supplied metadata, and a
    // wrap here would UNDERSTATE the extent — admitting an out-of-bounds
    // borrow through the safe validation this function exists to provide.
    let mut max_elem_offset: usize = 0;
    for (&dim, &stride) in shape.iter().zip(strides) {
        assert!(
            stride >= 0,
            "from_foreign requires non-negative strides; use borrow_raw_parts for negative-stride views."
        );
        max_elem_offset = (dim as usize - 1)
            .checked_mul(stride as usize)
            .and_then(|off| max_elem_offset.checked_add(off))
            .expect("Tensor addressable extent overflowed usize.");
    }
    max_elem_offset
        .checked_add(1)
        .and_then(|elems| elems.checked_mul(size_of::<T>()))
        .expect("Tensor addressable extent overflowed usize.")
}

impl<T: DType> Tensor<T> {
    // Enforces the core tensor invariant: shape/stride ranks must agree and the logical
    // typed byte size must exactly match the backing storage byte length.
    fn assert_valid_metadata(shape: &[i32], strides: &[i32], storage_num_bytes: usize) {
        assert_eq!(
            shape.len(),
            strides.len(),
            "Tensor shape/stride rank mismatch."
        );

        let num_bytes = checked_num_bytes_i32::<T>(shape)
            .expect("Tensor shape contains invalid dimensions or overflows.");
        assert_eq!(
            num_bytes, storage_num_bytes,
            "Tensor logical byte size must match storage byte size."
        );
    }

    /// Wraps a byte allocation (owned, foreign, or raw-borrowed) as a tensor after
    /// validating that the supplied shape/stride metadata is consistent with the
    /// allocation's logical size.
    pub(crate) fn from_device_buffer(
        device_buffer: DeviceBuffer,
        shape: Vec<i32>,
        strides: Vec<i32>,
    ) -> Self {
        Self::assert_valid_metadata(&shape, &strides, device_buffer.len_bytes());
        let storage = Arc::new(Storage::new(Allocation::Device(device_buffer)));
        let spec = compute_spec(
            storage.cu_deviceptr(),
            &shape,
            &strides,
            size_of::<T>() as i32,
        );
        Self {
            storage,
            shape,
            strides,
            spec,
            _dtype: PhantomData,
        }
    }

    /// Rebuilds a tensor from raw device allocation parts and validates the metadata
    /// against the provided byte length before taking ownership of the pointer.
    pub unsafe fn from_raw_parts(
        dptr: CUdeviceptr,
        len_bytes: usize,
        device_id: usize,
        shape: Vec<i32>,
        strides: Vec<i32>,
    ) -> Self {
        Self::assert_valid_metadata(&shape, &strides, len_bytes);
        Self::from_device_buffer(
            DeviceBuffer::from_raw_parts(dptr, len_bytes, device_id),
            shape,
            strides,
        )
    }

    /// Low-level, `unsafe` borrow of a bare device pointer as a tensor, without
    /// taking ownership: dropping the returned tensor — or any tensor sharing
    /// its storage — never frees `dptr`. The byte length is derived from `shape`
    /// and `T`.
    ///
    /// **Prefer [`from_foreign`](Self::from_foreign)** when the memory has an
    /// owner object (a cudarc buffer, a torch storage, ...): it holds that
    /// owner alive, so validity and liveness are verified at construction and
    /// only the aliasing obligation remains. Reach for `borrow_raw_parts` only
    /// when you hold a bare pointer with no owner to hand over, and can uphold
    /// every obligation below yourself.
    ///
    /// # Safety
    /// Unlike `Vec::from_raw_parts`, this does **not** transfer ownership of
    /// `dptr` to cutile — it is a borrow. The caller keeps ownership and must
    /// uphold all of the following for the entire lifetime of the returned
    /// tensor *and everything derived from it* (clones, partitions,
    /// `TensorView`s, and any `DeviceOp` built from them):
    ///
    /// - **Validity.** `dptr` points to device memory on `device_id` valid for
    ///   every byte the view can address. With non-negative strides that is
    ///   `(1 + Σ (shape[i]-1)·strides[i]) · size_of::<T>()` bytes from `dptr`,
    ///   which strides can push well past `product(shape) · size_of::<T>()`.
    ///   A negative stride additionally addresses BELOW `dptr`; bounding that
    ///   region is entirely the caller's problem (the safe `from_foreign`
    ///   refuses negative strides for exactly this reason).
    /// - **Liveness.** The allocation stays mapped at `dptr` — never freed,
    ///   reallocated, or resized by its owner — until *every* cutile operation
    ///   referencing it has completed. Launches are asynchronous, so
    ///   "completed" means past the final stream synchronize / `.await`, not
    ///   the return of the builder call.
    /// - **Aliasing.** While this tensor (or a partition of it) is used as a
    ///   mutable output, the owner must not read or write the same memory
    ///   through any other path — its own kernels, copies, or host access —
    ///   until that work completes. An immutable (`&Tensor`) use only requires
    ///   that the owner not mutate the memory concurrently. cutile cannot see
    ///   the foreign owner, so this aliasing-XOR-mutability obligation is the
    ///   caller's to enforce.
    /// - **Freeing.** cutile never frees a borrowed allocation; the owner frees
    ///   it exactly once, after all cutile work above has completed.
    /// - **Bit validity.** Every element the memory holds — now and after any
    ///   write the owner makes — must be a valid `T`, as the `# Safety`
    ///   section of [`DType`](cuda_core::DType) requires. This is automatic
    ///   for integer and floating-point `T`; for `T = bool` the owner must
    ///   store only `0` or `1`, since host code reads the bytes back as `bool`
    ///   without validation.
    ///
    /// [`Device::borrow_raw`]: cuda_core::Device::borrow_raw
    /// [`Stream::borrow_raw`]: cuda_core::Stream::borrow_raw
    pub unsafe fn borrow_raw_parts(
        dptr: CUdeviceptr,
        device_id: usize,
        shape: Vec<i32>,
        strides: Vec<i32>,
    ) -> Self {
        let len_bytes = checked_num_bytes_i32::<T>(&shape)
            .expect("Tensor shape contains invalid dimensions or overflows.");
        Self::from_device_buffer(
            DeviceBuffer::borrowed_from_raw_parts(dptr, len_bytes, device_id),
            shape,
            strides,
        )
    }

    /// Wraps device memory owned by an external framework (cudarc, torch, a VMM
    /// allocation, ...) as a tensor, holding the owner alive so the memory
    /// provably outlives every use — **no copy, no ownership transfer**.
    ///
    /// This is the preferred interop entry point: compared to
    /// [`borrow_raw_parts`](Self::borrow_raw_parts) (a bare pointer, every
    /// obligation on the caller), almost everything here is verified at
    /// construction. `owner` implements [`DeviceAllocation`] — the one-time
    /// pointer-validity assertion — and holding it keeps the allocation mapped
    /// for the whole life of this tensor and everything derived from it
    /// (clones, partitions, `DeviceOp`s). Liveness is therefore a refcount
    /// fact established at construction, not an ongoing obligation; only the
    /// aliasing clause below remains, which is why the function is `unsafe`.
    ///
    /// # Safety
    /// Everything positional is verified at construction: the pointer, length,
    /// and device ordinal come from `owner`'s one-time
    /// [`DeviceAllocation`] assertion; liveness is the held refcount; and the
    /// shape/stride addressable extent is checked (overflow-checked) against
    /// `owner.len_bytes()` right here. What cannot be verified at any call
    /// boundary is the TEMPORAL obligation, and it is why this function is
    /// `unsafe` — the same line `std` draws for `slice::from_raw_parts`, which
    /// stays `unsafe` even with a known-good pointer because
    /// aliasing-for-the-duration is not checkable at construction:
    ///
    /// - While any cutile work launched from this tensor (or anything derived
    ///   from it) **writes** the borrowed bytes, no other party — the owner,
    ///   another `from_foreign` tensor over the same memory, host copies —
    ///   may read or write them; while cutile work **reads** them, no other
    ///   party may write them. "Until the work completes" means past the
    ///   final stream synchronize / `.await`, not the builder-call return.
    /// - The memory must hold valid `T` values, as the `# Safety` section of
    ///   [`DType`](cuda_core::DType) requires: automatic for integer and
    ///   floating-point `T`, but a `Tensor<bool>` over foreign memory requires
    ///   the owner to store only `0` or `1`, because host code reads the bytes
    ///   back as `bool` without validation.
    ///
    /// Constructing two tensors over the same allocation and using either as
    /// a mutable output is a data race the type system cannot see: distinct
    /// storages defeat the unique-storage check that protects owned tensors.
    pub unsafe fn from_foreign(
        owner: Arc<dyn DeviceAllocation>,
        shape: Vec<i32>,
        strides: Vec<i32>,
    ) -> Self {
        let len_bytes = checked_num_bytes_i32::<T>(&shape)
            .expect("Tensor shape contains invalid dimensions or overflows.");
        // Validate the *addressable* extent (which strides can grow beyond the
        // shape product), not just the logical size — otherwise a strided borrow
        // could address past `owner`'s allocation from safe code.
        let extent_bytes = addressable_bytes::<T>(&shape, &strides);
        assert!(
            owner.len_bytes() >= extent_bytes,
            "foreign allocation ({} bytes) is smaller than the tensor's addressable extent \
             ({extent_bytes} bytes for the given shape/strides)",
            owner.len_bytes(),
        );
        Self::from_device_buffer(DeviceBuffer::foreign(owner, len_bytes), shape, strides)
    }

    /// Builds a metadata-only tensor (no GPU allocation) with contiguous strides.
    ///
    /// Backed by metadata-only storage. The spec is computed against a fixed
    /// 16-aligned sentinel pointer so the resulting cache key matches a real
    /// allocation's. Only usable for warmup via `.compile()`; any launch or copy
    /// panics when it reads the (absent) device pointer.
    pub(crate) fn from_meta(shape: Vec<i32>, device_id: usize) -> Self {
        let strides = contiguous_strides(&shape);
        let len_bytes = checked_num_bytes_i32::<T>(&shape)
            .expect("Tensor shape contains invalid dimensions or overflows.");
        Self::assert_valid_metadata(&shape, &strides, len_bytes);
        let spec = compute_spec(
            Storage::META_SPEC_PTR,
            &shape,
            &strides,
            size_of::<T>() as i32,
        );
        Self {
            storage: Arc::new(Storage::new(Allocation::Meta {
                len_bytes,
                device_id,
            })),
            shape,
            strides,
            spec,
            _dtype: PhantomData,
        }
    }

    // Returns the physical byte length of the shared backing allocation.
    fn storage_num_bytes(&self) -> usize {
        self.storage.len_bytes()
    }

    // Returns the logical element count described by the tensor's shape metadata.
    fn num_elements(&self) -> usize {
        checked_num_elements_i32(&self.shape)
            .expect("Tensor shape contains invalid dimensions or overflows.")
    }

    // Returns the byte size implied by shape metadata and dtype T.
    fn typed_num_bytes(&self) -> usize {
        checked_num_bytes_i32::<T>(&self.shape)
            .expect("Tensor shape contains invalid dimensions or overflows.")
    }

    // Validates that a zero-copy view keeps the same logical byte size and starts from
    // a layout that this implementation can safely reinterpret as contiguous.
    fn validate_view_shape(&self, shape: &[usize]) -> Result<(), Error> {
        validate_view_shape_for::<T>(&self.shape, &self.strides, shape)
    }

    // Validates zero-copy reinterpret by checking total byte size and target-type
    // alignment on top of the same contiguous-layout requirement as views.
    fn validate_reinterpret_shape<U: DType>(&self, shape: &[usize]) -> Result<(), Error> {
        if !self.is_contiguous() {
            return tensor_error_result("Zero-copy reinterpret requires contiguous storage.");
        }
        let target_num_bytes = checked_num_bytes::<U>(shape)?;
        if target_num_bytes != self.typed_num_bytes() {
            return tensor_error_result("Reinterpret shape must preserve total byte size.");
        }
        // spec_ptr, not cu_deviceptr: the sentinel satisfies every DType's
        // alignment (like a real >=256-aligned allocation) without panicking on meta.
        let alignment = align_of::<U>() as u64;
        if alignment > 1 && !self.storage.spec_ptr().is_multiple_of(alignment) {
            return tensor_error_result(
                "Tensor storage alignment is incompatible with reinterpret target type.",
            );
        }
        Ok(())
    }

    // Mutable partitioning is only sound when no other tensor/view aliases the backing
    // storage: a second `Arc<Tensor>` over the same allocation (reshape_shared,
    // reinterpret, into_shared_alias) could be read by a concurrent launch.
    fn has_unique_storage(&self) -> bool {
        self.storage.is_unique()
    }

    pub(crate) fn assert_unique_storage(&self) {
        assert!(
            self.has_unique_storage(),
            "Cannot create mutable partition from shared tensor storage."
        );
    }

    /// Allocates uninitialized GPU memory for a 1D tensor.
    ///
    /// This is a low-level function that allocates memory asynchronously but does not
    /// initialize it. The returned value must be initialized before use with `assume_init()`.
    ///
    /// ## Safety
    ///
    /// The returned tensor is wrapped in `MaybeUninit`. It must be initialized by a kernel
    /// or other operation before calling `assume_init()` on it.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// use cutile::tensor::Tensor;
    ///
    /// let uninit = Tensor::<f32>::uninitialized(1024).await;
    /// // Must initialize before use
    /// let tensor = unsafe { uninit.assume_init() };
    /// ```
    pub fn uninitialized(len: usize) -> impl DeviceOp<Output = MaybeUninit<Self>> {
        assert!(len > 0, "Non-zero length required.");
        // A failed device allocation is the operation's error, not a panic;
        // see `api::AllocUninitialized`.
        crate::api::alloc_uninitialized::<T>(len)
    }

    pub fn dtype(&self) -> DTypeId {
        T::DTYPE
    }

    pub(crate) fn cu_deviceptr(&self) -> CUdeviceptr {
        self.storage.cu_deviceptr()
    }

    pub fn device_id(&self) -> usize {
        self.storage.device_id()
    }

    /// Returns a typed device pointer.
    pub fn device_pointer(&self) -> DevicePointer<T> {
        unsafe { DevicePointer::from_cu_deviceptr(self.cu_deviceptr()) }
    }

    /// Returns the tensor's shape.
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }

    /// Returns the tensor's strides.
    pub fn strides(&self) -> &[i32] {
        &self.strides
    }

    /// Returns the tensor's specialization bits.
    pub fn spec(&self) -> &SpecializationBits {
        &self.spec
    }

    /// Returns the total number of elements in the tensor.
    pub fn size(&self) -> usize {
        debug_assert_eq!(self.typed_num_bytes(), self.storage_num_bytes());
        self.num_elements()
    }

    /// Creates an independent copy of this tensor's GPU data.
    ///
    /// Returns a device operation that, when executed, will allocate new GPU memory
    /// and copy the tensor's data.
    pub fn dup(&self) -> impl DeviceOp<Output = Self> {
        crate::api::dup(self)
    }

    /// Returns the total size of the tensor in bytes.
    pub fn num_bytes(&self) -> usize {
        self.typed_num_bytes()
    }

    /// Returns `true` if the tensor metadata describes a contiguous row-major layout.
    pub fn is_contiguous(&self) -> bool {
        self.strides == contiguous_strides(&self.shape)
    }

    /// Create an `Arc<Tensor<T>>` that shares this tensor's device memory.
    ///
    /// # Safety
    ///
    /// Two tensors sharing storage can cause mutable aliasing if both are
    /// passed to kernels. The caller must ensure only one is written at a
    /// time. Prefer `tensor.view()` (returns `TensorView`) for safe
    /// borrow-based sharing.
    pub unsafe fn into_shared_alias(&self) -> Arc<Self> {
        Arc::new(Self {
            storage: self.storage.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            spec: self.spec.clone(),
            _dtype: PhantomData,
        })
    }

    // Internal: reshape without validation. Caller must ensure element count matches.
    pub(crate) fn reshape_unchecked(mut self, shape: &[usize]) -> Self {
        let shape: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        self.strides = contiguous_strides(&shape);
        self.spec = compute_spec(
            self.storage.spec_ptr(),
            &shape,
            &self.strides,
            size_of::<T>() as i32,
        );
        self.shape = shape;
        self
    }

    // Internal: create a new Arc sharing storage with different shape.
    // Used by ReshapeOp and the Reshape trait impl for &Arc<Tensor<T>>.
    pub(crate) fn reshape_shared(self: &Arc<Self>, shape: &[usize]) -> Result<Arc<Self>, Error> {
        self.validate_view_shape(shape)?;
        let new_shape: Vec<i32> = shape.iter().map(|x| *x as i32).collect();
        let new_strides = contiguous_strides(&new_shape);
        let spec = compute_spec(
            self.storage.spec_ptr(),
            &new_shape,
            &new_strides,
            size_of::<T>() as i32,
        );
        Ok(Arc::new(Self {
            storage: self.storage.clone(),
            strides: new_strides,
            shape: new_shape,
            spec,
            _dtype: PhantomData,
        }))
    }

    /// Reinterprets the tensor's bytes as a different type with a new shape.
    ///
    /// Zero-copy. Returns `Err` if the tensor is not contiguous, the total
    /// byte size doesn't match, or the pointer alignment is incompatible.
    pub fn reinterpret<U: DType>(
        self: &Arc<Self>,
        shape: &[usize],
    ) -> Result<Arc<Tensor<U>>, Error> {
        self.validate_reinterpret_shape::<U>(shape)?;
        let new_shape: Vec<i32> = shape.iter().map(|x| *x as i32).collect();
        let new_strides = contiguous_strides(&new_shape);
        let spec = compute_spec(
            self.storage.spec_ptr(),
            &new_shape,
            &new_strides,
            size_of::<U>() as i32,
        );
        Ok(Arc::new(Tensor::<U> {
            storage: self.storage.clone(),
            strides: new_strides,
            shape: new_shape,
            spec,
            _dtype: PhantomData,
        }))
    }
}

/// Converts a GPU tensor to a host-side vector.
///
/// This trait provides a method to asynchronously copy tensor data from GPU to CPU memory
/// as a `Vec<T>`. Implemented for both owned tensors and `Arc<Tensor<T>>`.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::tensor::ToHostVec;
///
/// let gpu_tensor = api::arange::<f32>(100).await;
/// let cpu_data: Vec<f32> = gpu_tensor.to_host_vec().await;
/// assert_eq!(cpu_data.len(), 100);
/// ```
pub trait ToHostVec<T: Send> {
    /// Copies the tensor data from GPU to host memory, returning a `Vec<T>`.
    fn to_host_vec(self) -> impl DeviceOp<Output = Vec<T>>;
}

impl<T: DType> ToHostVec<T> for Tensor<T> {
    fn to_host_vec(self) -> impl DeviceOp<Output = Vec<T>> {
        let arc_self = Arc::new(self);
        copy_device_to_host_vec(&arc_self)
    }
}

impl<T: DType> ToHostVec<T> for Arc<Tensor<T>> {
    fn to_host_vec(self) -> impl DeviceOp<Output = Vec<T>> {
        copy_device_to_host_vec(&self)
    }
}

impl<T: DType> ToHostVec<T> for &Arc<Tensor<T>> {
    fn to_host_vec(self) -> impl DeviceOp<Output = Vec<T>> {
        copy_device_to_host_vec(self)
    }
}

// ── Reshape trait ────────────────────────────────────────────────────────────

/// Reshape a tensor or `Arc<Tensor>` to a new shape.
///
/// - On `Tensor<T>`: consumes and returns a reshaped `Tensor<T>`.
/// - On `&Arc<Tensor<T>>`: creates a new `Arc` sharing device memory.
pub trait Reshape {
    type Output;
    fn reshape(self, shape: &[usize]) -> Result<Self::Output, Error>;
}

impl<T: DType> Reshape for Tensor<T> {
    type Output = Tensor<T>;
    fn reshape(self, shape: &[usize]) -> Result<Tensor<T>, Error> {
        self.validate_view_shape(shape)?;
        Ok(self.reshape_unchecked(shape))
    }
}

impl<T: DType> Reshape for &Arc<Tensor<T>> {
    type Output = Arc<Tensor<T>>;
    fn reshape(self, shape: &[usize]) -> Result<Arc<Tensor<T>>, Error> {
        self.reshape_shared(shape)
    }
}

// ── TensorView ──────────────────────────────────────────────────────────────

/// A borrowed, reshaped view of a tensor.
///
/// Created by [`Tensor::view`]. The view borrows the base tensor's device
/// memory with different shape/strides metadata. The borrow checker ensures
/// the base tensor can't be mutated while the view exists.
///
/// Kernel `&Tensor` params accept `&TensorView<T>` via `KernelInput`.
///
/// ```rust,ignore
/// let tensor = api::ones::<f32>(&[1024]).sync()?;
/// let view = tensor.view(&[32, 32])?;    // borrows tensor
/// kernel(out, &view).sync()?;             // view accepted as &Tensor param
/// // view dropped — tensor can be mutated again
/// ```
pub struct TensorView<'a, T: DType> {
    base: &'a Tensor<T>,
    offset_bytes: usize,
    shape: Vec<i32>,
    strides: Vec<i32>,
    spec: SpecializationBits,
}

impl<'a, T: DType> TensorView<'a, T> {
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }
    pub fn strides(&self) -> &[i32] {
        &self.strides
    }
    pub fn spec(&self) -> &SpecializationBits {
        &self.spec
    }
    pub fn size(&self) -> usize {
        self.shape.iter().map(|&x| x as usize).product()
    }
    /// Re-view with a different shape.
    pub fn view(&self, shape: &[usize]) -> Result<TensorView<'_, T>, Error> {
        validate_view_shape_for::<T>(&self.shape, &self.strides, shape)?;
        let new_shape: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        let new_strides = contiguous_strides(&new_shape);
        let spec = compute_spec(
            self.base.storage.spec_ptr(),
            &new_shape,
            &new_strides,
            size_of::<T>() as i32,
        );
        Ok(TensorView {
            base: self.base,
            offset_bytes: self.offset_bytes,
            shape: new_shape,
            strides: new_strides,
            spec,
        })
    }
    /// Slice this view along one or more axes, numpy-style.
    ///
    /// Each range corresponds to an axis. Fewer ranges than axes is
    /// allowed: trailing axes are left unsliced. The strides are
    /// preserved (slicing never changes strides).
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Result<TensorView<'_, T>, Error> {
        if ranges.len() > self.shape.len() {
            return tensor_error_result("slice: more ranges than axes.");
        }
        let mut offset_elems: usize = 0;
        let mut new_shape = self.shape.clone();
        for (axis, range) in ranges.iter().enumerate() {
            let dim = self.shape[axis] as usize;
            if range.start > range.end || range.end > dim {
                return tensor_error_result("slice: range out of bounds.");
            }
            offset_elems += range.start * self.strides[axis] as usize;
            new_shape[axis] = (range.end - range.start) as i32;
        }
        let new_strides = self.strides.clone();
        let spec = compute_spec(
            self.base.storage.spec_ptr()
                + (self.offset_bytes + offset_elems * size_of::<T>()) as u64,
            &new_shape,
            &new_strides,
            size_of::<T>() as i32,
        );
        Ok(TensorView {
            base: self.base,
            offset_bytes: self.offset_bytes + offset_elems * size_of::<T>(),
            shape: new_shape,
            strides: new_strides,
            spec,
        })
    }
}

impl<T: DType> Tensor<T> {
    /// Create a borrowed view with a different shape.
    ///
    /// The view borrows `self` — the tensor can't be mutated while the
    /// view exists. No allocation or copy.
    pub fn view(&self, shape: &[usize]) -> Result<TensorView<'_, T>, Error> {
        self.validate_view_shape(shape)?;
        let new_shape: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        let new_strides = contiguous_strides(&new_shape);
        let spec = compute_spec(
            self.storage.spec_ptr(),
            &new_shape,
            &new_strides,
            size_of::<T>() as i32,
        );
        Ok(TensorView {
            base: self,
            offset_bytes: 0,
            shape: new_shape,
            strides: new_strides,
            spec,
        })
    }

    /// Slice this tensor along one or more axes, numpy-style.
    ///
    /// Returns a `TensorView` with adjusted offset and shape. Strides
    /// are preserved from the original tensor. No copy.
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Result<TensorView<'_, T>, Error> {
        if ranges.len() > self.shape.len() {
            return tensor_error_result("slice: more ranges than axes.");
        }
        let mut offset_elems: usize = 0;
        let mut new_shape = self.shape.clone();
        for (axis, range) in ranges.iter().enumerate() {
            let dim = self.shape[axis] as usize;
            if range.start > range.end || range.end > dim {
                return tensor_error_result("slice: range out of bounds.");
            }
            offset_elems += range.start * self.strides[axis] as usize;
            new_shape[axis] = (range.end - range.start) as i32;
        }
        let new_strides = self.strides.clone();
        let spec = compute_spec(
            self.storage.spec_ptr() + (offset_elems * size_of::<T>()) as u64,
            &new_shape,
            &new_strides,
            size_of::<T>() as i32,
        );
        Ok(TensorView {
            base: self,
            offset_bytes: offset_elems * size_of::<T>(),
            shape: new_shape,
            strides: new_strides,
            spec,
        })
    }
}

impl<T: DType> IntoPartitionArc for Tensor<T> {
    fn partition<const RANK: usize>(
        self: Arc<Tensor<T>>,
        partition_shape: [usize; RANK],
    ) -> Partition<Tensor<T>> {
        let partition_shape = partition_shape.to_vec();
        let partition_strides: Vec<usize> = self.strides.iter().map(|&s| s as usize).collect();
        let tensor = Arc::try_unwrap(self).expect("Failed to convert Arc to Partition.");
        tensor.assert_unique_storage();
        Partition::<Tensor<T>> {
            object: tensor,
            partition_shape,
            partition_strides,
            prefix_coverage: false,
        }
    }
}

impl<T: DType> IntoPartition for Tensor<T> {
    fn partition<const RANK: usize>(self, partition_shape: [usize; RANK]) -> Partition<Tensor<T>> {
        let partition_shape = partition_shape.to_vec();
        let partition_strides: Vec<usize> = self.strides.iter().map(|&s| s as usize).collect();
        self.assert_unique_storage();
        Partition::<Tensor<T>> {
            object: self,
            partition_shape,
            partition_strides,
            prefix_coverage: false,
        }
    }
}

// ── Partition<&'a mut Tensor<T>> ─────────────────────────────────────────────

/// Partition a mutably borrowed tensor. The partition borrows the tensor,
/// so no `unpartition()` is needed — the tensor already has the kernel's output.
pub trait PartitionMut<'a, T: DType> {
    fn partition<const RANK: usize>(
        self,
        partition_shape: [usize; RANK],
    ) -> Partition<&'a mut Tensor<T>>;

    /// Partial-coverage variant of [`Self::partition`]; see
    /// [`IntoPartition::partition_prefix`].
    fn partition_prefix<const RANK: usize>(
        self,
        partition_shape: [usize; RANK],
    ) -> Partition<&'a mut Tensor<T>>
    where
        Self: Sized,
    {
        self.partition(partition_shape).prefix()
    }
}

impl<'a, T: DType> PartitionMut<'a, T> for &'a mut Tensor<T> {
    fn partition<const RANK: usize>(
        self,
        partition_shape: [usize; RANK],
    ) -> Partition<&'a mut Tensor<T>> {
        let partition_shape = partition_shape.to_vec();
        let partition_strides: Vec<usize> = self.strides.iter().map(|&s| s as usize).collect();
        // Same invariant as the owned `IntoPartition` path: `&mut` proves
        // exclusivity of this handle, not of the storage — an `Arc<Tensor>`
        // alias produced by reshape_shared/reinterpret/into_shared_alias can
        // still be a live kernel input.
        self.assert_unique_storage();
        Partition {
            object: self,
            partition_shape,
            partition_strides,
            prefix_coverage: false,
        }
    }
}

impl<T: DType> Partition<&mut Tensor<T>> {
    pub fn dtype_str(&self) -> &'static str {
        T::DTYPE.as_str()
    }

    pub fn grid(&self) -> Result<(u32, u32, u32), Error> {
        partition_launch_grid(&self.object.shape, &self.partition_shape)
    }
}

impl<'a, T: DType + Sync> IntoDeviceOp<Partition<&'a mut Tensor<T>>>
    for Partition<&'a mut Tensor<T>>
{
    type Op = Value<Partition<&'a mut Tensor<T>>>;
    fn into_op(self) -> Value<Partition<&'a mut Tensor<T>>> {
        value(self)
    }
}

/// Extension trait for partitioning an `Arc<Tensor<T>>` by consuming sole ownership.
pub trait TryPartition<T: DType> {
    /// Consumes the Arc and partitions the tensor.
    ///
    /// Returns `Err` if the Arc has other owners (refcount > 1) or the
    /// underlying storage is shared with views.
    fn try_partition<const RANK: usize>(
        self,
        partition_shape: [usize; RANK],
    ) -> Result<Partition<Tensor<T>>, Error>;
}

impl<T: DType> TryPartition<T> for Arc<Tensor<T>> {
    fn try_partition<const RANK: usize>(
        self,
        partition_shape: [usize; RANK],
    ) -> Result<Partition<Tensor<T>>, Error> {
        let tensor = Arc::try_unwrap(self).map_err(|_| {
            crate::error::tensor_error("try_partition: Arc<Tensor> has multiple owners")
        })?;
        // The documented contract is `Err`, not the panic `partition` raises,
        // when another tensor or view still shares the backing storage.
        if !tensor.has_unique_storage() {
            return tensor_error_result(
                "try_partition: tensor storage is shared with other tensors or views",
            );
        }
        Ok(tensor.partition(partition_shape))
    }
}

pub trait Unpartition<T: DType> {
    /// Unwraps the partition to produce the underlying value.
    fn unpartition(self) -> impl DeviceOp<Output = Tensor<T>>;
}

impl<T: DType, DI: DeviceOp<Output = Partition<Tensor<T>>>> Unpartition<T> for DI {
    fn unpartition(self) -> impl DeviceOp<Output = Tensor<T>> {
        UnwrapPartition { op: self }
    }
}

// Preliminary support for vectors of tensors is done by providing an unsafe interior mutability pattern.
#[derive(Clone, Debug)]
pub struct DeviceVec<T> {
    _ty: PhantomData<T>,
    host_vec: Vec<Arc<T>>,
    device_vec: Arc<Tensor<i64>>,
}

impl<T: DType> DeviceVec<Tensor<T>> {
    pub fn from(v: Vec<Tensor<T>>) -> DeviceVec<Tensor<T>> {
        let i64vec: Arc<Vec<i64>> = v
            .iter()
            .map(|x| x.cu_deviceptr() as i64)
            .collect::<Vec<_>>()
            .into();
        let device_vec: Arc<Tensor<i64>> = copy_host_vec_to_device(&i64vec)
            .sync()
            .expect("Failed to execute device operation.")
            .reshape_unchecked(&[v.len()])
            .into();
        let host_vec: Vec<Arc<Tensor<T>>> = v.into_iter().map(Arc::new).collect::<Vec<_>>();
        DeviceVec {
            _ty: PhantomData,
            host_vec,
            device_vec,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.host_vec.len() == 0
    }
    pub fn len(&self) -> usize {
        self.host_vec.len()
    }
    pub unsafe fn inner(&self) -> &Arc<Tensor<i64>> {
        &self.device_vec
    }
}

impl<T: DType> From<Vec<Tensor<T>>> for DeviceVec<Tensor<T>> {
    fn from(v: Vec<Tensor<T>>) -> Self {
        DeviceVec::from(v)
    }
}

impl<T: DType> Index<usize> for DeviceVec<Tensor<T>> {
    type Output = Arc<Tensor<T>>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.host_vec[index]
    }
}

pub struct DeviceVecIntoIter<Item> {
    items: DeviceVec<Item>,
}

impl<T: DType> Iterator for DeviceVecIntoIter<Tensor<T>> {
    type Item = Tensor<T>;
    fn next(&mut self) -> Option<Self::Item> {
        if !self.items.is_empty() {
            let x = self.items.host_vec.remove(0);
            let x = Arc::try_unwrap(x).expect("Unable to perform into_iter from non-unique Arc.");
            Some(x)
        } else {
            None
        }
    }
}

impl<T: DType> IntoIterator for DeviceVec<Tensor<T>> {
    type Item = Tensor<T>;
    type IntoIter = DeviceVecIntoIter<Tensor<T>>;
    fn into_iter(self) -> Self::IntoIter {
        DeviceVecIntoIter { items: self }
    }
}

// IntoDeviceOp impls for Tensor types

impl<T: DType> IntoDeviceOp<Partition<Tensor<T>>> for Partition<Tensor<T>> {
    type Op = Value<Partition<Tensor<T>>>;
    fn into_op(self) -> Value<Partition<Tensor<T>>> {
        value(self)
    }
}

impl<T: DType> IntoDeviceOp<MappedLaunchPartition<Partition<Tensor<T>>>>
    for MappedLaunchPartition<Partition<Tensor<T>>>
{
    type Op = Value<MappedLaunchPartition<Partition<Tensor<T>>>>;
    fn into_op(self) -> Value<MappedLaunchPartition<Partition<Tensor<T>>>> {
        value(self)
    }
}

impl<'a, T: DType + Sync> IntoDeviceOp<MappedLaunchPartition<Partition<&'a mut Tensor<T>>>>
    for MappedLaunchPartition<Partition<&'a mut Tensor<T>>>
{
    type Op = Value<MappedLaunchPartition<Partition<&'a mut Tensor<T>>>>;
    fn into_op(self) -> Value<MappedLaunchPartition<Partition<&'a mut Tensor<T>>>> {
        value(self)
    }
}

impl<T: DType> IntoDeviceOp<Tensor<T>> for Tensor<T> {
    type Op = Value<Tensor<T>>;
    fn into_op(self) -> Value<Tensor<T>> {
        value(self)
    }
}

impl<'a, T: DType + Sync> IntoDeviceOp<&'a Tensor<T>> for &'a Tensor<T> {
    type Op = Value<&'a Tensor<T>>;
    fn into_op(self) -> Value<&'a Tensor<T>> {
        value(self)
    }
}

// KernelInput impls — how &Tensor kernel params are held and recovered.

use cuda_async::launch::AsyncKernelLaunch;

// ── KernelOutput trait ──────────────────────────────────────────────────────
//
// Abstracts over Partition<Tensor<T>> and Partition<&mut Tensor<T>> so the
// macro-generated launcher accepts both for &mut Tensor params.

/// A partition binding's constraint on the launch grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridBound {
    /// The launch grid must EQUAL this per axis — full coverage, the
    /// default, and the diagnostic that catches accidental mismatches.
    Exact((u32, u32, u32)),
    /// The launch grid may be a per-axis PREFIX of this (opt-in via
    /// `partition_prefix`). `launch > bound` on any axis is a launch error:
    /// that direction is genuine out-of-bounds.
    AtMost((u32, u32, u32)),
}

pub trait KernelOutputStored<T: DType>: Send {
    /// Retain storage and acquire exclusive device access before enqueueing.
    fn retain(&self, ctx: &ExecutionContext) -> Result<(), DeviceError>;
    fn push_kernel_args(&self, launcher: &mut AsyncKernelLaunch);
    fn grid(&self) -> Result<(u32, u32, u32), Error>;
    /// This binding's launch-grid constraint. Defaults to exact coverage;
    /// only bindings that explicitly opted into partial coverage return
    /// [`GridBound::AtMost`].
    fn grid_bound(&self) -> Result<GridBound, Error> {
        Ok(GridBound::Exact(self.grid()?))
    }
    fn map_shape_as_i32(&self) -> Option<Vec<i32>> {
        None
    }
    fn dtype_str(&self) -> &'static str;
    fn partition_shape_as_i32(&self) -> Vec<i32>;
    /// The partition shape as bound, borrowed: what launch validation reads.
    fn partition_shape(&self) -> &[usize];
    fn strides_hint(&self) -> Vec<i32>;
    fn spec(&self) -> &SpecializationBits;
    fn shape_as_i32(&self) -> Vec<i32>;
}

/// How a `&mut Tensor` kernel param is stored during execution and recovered.
///
/// | Input | Stored | Returned |
/// |---|---|---|
/// | `Partition<Tensor<T>>` | `Partition<Tensor<T>>` | `Partition<Tensor<T>>` |
/// | `Partition<&'a mut Tensor<T>>` | `Partition<&'a mut Tensor<T>>` | `Partition<&'a mut Tensor<T>>` |
/// | `MappedLaunchPartition<Partition<..>>` | `MappedLaunchPartition<Partition<..>>` | `Partition<..>` |
pub trait KernelOutput<T: DType>: Send + Sized {
    type Stored: KernelOutputStored<T>;
    type Returned: Send;
    fn prepare(self) -> Self::Stored;
    fn recover(stored: Self::Stored) -> Self::Returned;
}

impl<T: DType> KernelOutputStored<T> for Partition<Tensor<T>> {
    fn retain(&self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        self.object.storage.retain(ctx, true)
    }
    fn grid_bound(&self) -> Result<GridBound, Error> {
        let grid = KernelOutputStored::grid(self)?;
        Ok(if self.prefix_coverage {
            GridBound::AtMost(grid)
        } else {
            GridBound::Exact(grid)
        })
    }

    fn push_kernel_args(&self, launcher: &mut AsyncKernelLaunch) {
        unsafe {
            launcher.push_device_ptr(self.object.cu_deviceptr());
        }
        for dim in self.object.shape.iter() {
            launcher.push_arg(*dim);
        }
        for stride in self.object.strides.iter() {
            launcher.push_arg(*stride);
        }
        for dim in self.partition_shape.iter() {
            launcher.push_arg(*dim as i32);
        }
        for stride in self.partition_strides.iter() {
            launcher.push_arg(*stride as i32);
        }
    }
    fn grid(&self) -> Result<(u32, u32, u32), Error> {
        partition_launch_grid(&self.object.shape, &self.partition_shape)
    }
    fn dtype_str(&self) -> &'static str {
        T::DTYPE.as_str()
    }
    fn partition_shape_as_i32(&self) -> Vec<i32> {
        self.partition_shape.iter().map(|&x| x as i32).collect()
    }
    fn partition_shape(&self) -> &[usize] {
        &self.partition_shape
    }
    fn strides_hint(&self) -> Vec<i32> {
        self.object
            .spec
            .stride_one
            .iter()
            .map(|&is_one| if is_one { 1 } else { -1 })
            .collect()
    }
    fn spec(&self) -> &SpecializationBits {
        &self.object.spec
    }
    fn shape_as_i32(&self) -> Vec<i32> {
        self.object.shape.clone()
    }
}

impl<T: DType> KernelOutputStored<T> for Partition<&mut Tensor<T>> {
    fn retain(&self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        self.object.storage.retain(ctx, true)
    }
    fn grid_bound(&self) -> Result<GridBound, Error> {
        let grid = KernelOutputStored::grid(self)?;
        Ok(if self.prefix_coverage {
            GridBound::AtMost(grid)
        } else {
            GridBound::Exact(grid)
        })
    }

    fn push_kernel_args(&self, launcher: &mut AsyncKernelLaunch) {
        unsafe {
            launcher.push_device_ptr(self.object.cu_deviceptr());
        }
        for dim in self.object.shape.iter() {
            launcher.push_arg(*dim);
        }
        for stride in self.object.strides.iter() {
            launcher.push_arg(*stride);
        }
        for dim in self.partition_shape.iter() {
            launcher.push_arg(*dim as i32);
        }
        for stride in self.partition_strides.iter() {
            launcher.push_arg(*stride as i32);
        }
    }
    fn grid(&self) -> Result<(u32, u32, u32), Error> {
        partition_launch_grid(&self.object.shape, &self.partition_shape)
    }
    fn dtype_str(&self) -> &'static str {
        T::DTYPE.as_str()
    }
    fn partition_shape_as_i32(&self) -> Vec<i32> {
        self.partition_shape.iter().map(|&x| x as i32).collect()
    }
    fn partition_shape(&self) -> &[usize] {
        &self.partition_shape
    }
    fn strides_hint(&self) -> Vec<i32> {
        self.object
            .spec
            .stride_one
            .iter()
            .map(|&is_one| if is_one { 1 } else { -1 })
            .collect()
    }
    fn spec(&self) -> &SpecializationBits {
        &self.object.spec
    }
    fn shape_as_i32(&self) -> Vec<i32> {
        self.object.shape.clone()
    }
}

impl<T: DType> KernelOutputStored<T> for MappedLaunchPartition<Partition<Tensor<T>>> {
    fn retain(&self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        self.partition.retain(ctx)
    }
    fn push_kernel_args(&self, launcher: &mut AsyncKernelLaunch) {
        self.partition.push_kernel_args(launcher);
    }

    fn grid(&self) -> Result<(u32, u32, u32), Error> {
        self.validate(self.partition.grid()?, self.num_tile_blocks)
    }

    fn map_shape_as_i32(&self) -> Option<Vec<i32>> {
        Some(self.map_shape.iter().map(|&dim| dim as i32).collect())
    }

    fn dtype_str(&self) -> &'static str {
        self.partition.dtype_str()
    }

    fn partition_shape_as_i32(&self) -> Vec<i32> {
        self.partition.partition_shape_as_i32()
    }
    fn partition_shape(&self) -> &[usize] {
        KernelOutputStored::partition_shape(&self.partition)
    }

    fn strides_hint(&self) -> Vec<i32> {
        self.partition.strides_hint()
    }

    fn spec(&self) -> &SpecializationBits {
        self.partition.spec()
    }

    fn shape_as_i32(&self) -> Vec<i32> {
        self.partition.shape_as_i32()
    }
}

impl<T: DType> KernelOutputStored<T> for MappedLaunchPartition<Partition<&mut Tensor<T>>> {
    fn retain(&self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        self.partition.retain(ctx)
    }
    fn push_kernel_args(&self, launcher: &mut AsyncKernelLaunch) {
        self.partition.push_kernel_args(launcher);
    }

    fn grid(&self) -> Result<(u32, u32, u32), Error> {
        self.validate(self.partition.grid()?, self.num_tile_blocks)
    }

    fn map_shape_as_i32(&self) -> Option<Vec<i32>> {
        Some(self.map_shape.iter().map(|&dim| dim as i32).collect())
    }

    fn dtype_str(&self) -> &'static str {
        self.partition.dtype_str()
    }

    fn partition_shape_as_i32(&self) -> Vec<i32> {
        self.partition.partition_shape_as_i32()
    }
    fn partition_shape(&self) -> &[usize] {
        KernelOutputStored::partition_shape(&self.partition)
    }

    fn strides_hint(&self) -> Vec<i32> {
        self.partition.strides_hint()
    }

    fn spec(&self) -> &SpecializationBits {
        self.partition.spec()
    }

    fn shape_as_i32(&self) -> Vec<i32> {
        self.partition.shape_as_i32()
    }
}

impl<T: DType> KernelOutput<T> for Partition<Tensor<T>> {
    type Stored = Partition<Tensor<T>>;
    type Returned = Partition<Tensor<T>>;
    fn prepare(self) -> Self::Stored {
        self
    }
    fn recover(stored: Self::Stored) -> Self::Returned {
        stored
    }
}

impl<'a, T: DType> KernelOutput<T> for Partition<&'a mut Tensor<T>> {
    type Stored = Partition<&'a mut Tensor<T>>;
    type Returned = Partition<&'a mut Tensor<T>>;
    fn prepare(self) -> Self::Stored {
        self
    }
    fn recover(stored: Self::Stored) -> Self::Returned {
        stored
    }
}

impl<T: DType> KernelOutput<T> for MappedLaunchPartition<Partition<Tensor<T>>> {
    type Stored = MappedLaunchPartition<Partition<Tensor<T>>>;
    type Returned = Partition<Tensor<T>>;

    fn prepare(self) -> Self::Stored {
        self
    }

    fn recover(stored: Self::Stored) -> Self::Returned {
        stored.partition
    }
}

impl<'a, T: DType> KernelOutput<T> for MappedLaunchPartition<Partition<&'a mut Tensor<T>>> {
    type Stored = MappedLaunchPartition<Partition<&'a mut Tensor<T>>>;
    type Returned = Partition<&'a mut Tensor<T>>;

    fn prepare(self) -> Self::Stored {
        self
    }

    fn recover(stored: Self::Stored) -> Self::Returned {
        stored.partition
    }
}

// ── KernelInput traits ──────────────────────────────────────────────────────
//
// Defined here (not in cuda-async) so that impls on Arc<Tensor<T>> satisfy
// the orphan rule — both trait and Tensor<T> are in the same crate.

/// How a stored kernel input pushes its arguments to the launcher.
///
/// Implemented for `Arc<Tensor<T>>` and `&Tensor<T>`. Both push the same
/// data: device pointer, shape, and strides.
pub trait KernelInputStored: Send {
    /// Retain storage and acquire shared device access before enqueueing.
    fn retain(&self, ctx: &ExecutionContext) -> Result<(), DeviceError>;
    fn push_kernel_args(&self, launcher: &mut AsyncKernelLaunch);
    fn shape(&self) -> &[i32];
    fn strides(&self) -> &[i32];
    fn spec(&self) -> &SpecializationBits;
    fn dtype_str(&self) -> &'static str;
}

/// Converts a user-provided kernel input into a stored form for execution,
/// and recovers the caller's original type afterward.
///
/// | Input | Stored | Returned | `'static`? |
/// |---|---|---|---|
/// | `Tensor<T>` | `Arc<Tensor<T>>` | `Tensor<T>` | Yes |
/// | `Arc<Tensor<T>>` | `Arc<Tensor<T>>` | `Arc<Tensor<T>>` | Yes |
/// | `&'a Tensor<T>` | `&'a Tensor<T>` | `&'a Tensor<T>` | No |
pub trait KernelInput<T: DType>: Send + Sized {
    type Stored: KernelInputStored;
    type Returned: Send;
    fn prepare(self) -> Self::Stored;
    fn recover(stored: Self::Stored) -> Self::Returned;
}

// ── KernelInputStored impls ─────────────────────────────────────────────────

impl<T: DType> KernelInputStored for Arc<Tensor<T>> {
    fn retain(&self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        self.storage.retain(ctx, false)
    }
    fn push_kernel_args(&self, launcher: &mut AsyncKernelLaunch) {
        unsafe {
            launcher.push_device_ptr(self.cu_deviceptr());
        }
        for dim in self.shape.iter() {
            launcher.push_arg(*dim);
        }
        for stride in self.strides.iter() {
            launcher.push_arg(*stride);
        }
    }
    fn shape(&self) -> &[i32] {
        Tensor::shape(self)
    }
    fn strides(&self) -> &[i32] {
        Tensor::strides(self)
    }
    fn spec(&self) -> &SpecializationBits {
        Tensor::spec(self)
    }
    fn dtype_str(&self) -> &'static str {
        T::DTYPE.as_str()
    }
}

impl<T: DType + Sync> KernelInputStored for &Tensor<T> {
    fn retain(&self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        self.storage.retain(ctx, false)
    }
    fn push_kernel_args(&self, launcher: &mut AsyncKernelLaunch) {
        unsafe {
            launcher.push_device_ptr(self.cu_deviceptr());
        }
        for dim in self.shape.iter() {
            launcher.push_arg(*dim);
        }
        for stride in self.strides.iter() {
            launcher.push_arg(*stride);
        }
    }
    fn shape(&self) -> &[i32] {
        Tensor::shape(self)
    }
    fn strides(&self) -> &[i32] {
        Tensor::strides(self)
    }
    fn spec(&self) -> &SpecializationBits {
        Tensor::spec(self)
    }
    fn dtype_str(&self) -> &'static str {
        T::DTYPE.as_str()
    }
}

// ── KernelInput impls ───────────────────────────────────────────────────────

impl<T: DType> KernelInput<T> for Tensor<T> {
    type Stored = Arc<Tensor<T>>;
    type Returned = Tensor<T>;
    fn prepare(self) -> Arc<Tensor<T>> {
        Arc::new(self)
    }
    fn recover(stored: Arc<Tensor<T>>) -> Tensor<T> {
        Arc::try_unwrap(stored).expect("KernelInput::recover: Arc has multiple owners")
    }
}

impl<T: DType> KernelInput<T> for Arc<Tensor<T>> {
    type Stored = Arc<Tensor<T>>;
    type Returned = Arc<Tensor<T>>;
    fn prepare(self) -> Arc<Tensor<T>> {
        self
    }
    fn recover(stored: Arc<Tensor<T>>) -> Arc<Tensor<T>> {
        stored
    }
}

impl<'a, T: DType + Sync> KernelInput<T> for &'a Tensor<T> {
    type Stored = &'a Tensor<T>;
    type Returned = &'a Tensor<T>;
    fn prepare(self) -> &'a Tensor<T> {
        self
    }
    fn recover(stored: &'a Tensor<T>) -> &'a Tensor<T> {
        stored
    }
}

// ── TensorView KernelInput impls ────────────────────────────────────────────

impl<'a, T: DType + Sync> KernelInputStored for &'a TensorView<'a, T> {
    fn retain(&self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        self.base.storage.retain(ctx, false)
    }
    fn push_kernel_args(&self, launcher: &mut AsyncKernelLaunch) {
        // Push the already-offset device pointer. The offset is applied
        // host-side so the kernel sees the correct base address directly.
        unsafe {
            launcher.push_device_ptr(self.base.cu_deviceptr() + self.offset_bytes as u64);
        }
        for dim in self.shape.iter() {
            launcher.push_arg(*dim);
        }
        for stride in self.strides.iter() {
            launcher.push_arg(*stride);
        }
    }
    fn shape(&self) -> &[i32] {
        &self.shape
    }
    fn strides(&self) -> &[i32] {
        &self.strides
    }
    fn spec(&self) -> &SpecializationBits {
        TensorView::spec(self)
    }
    fn dtype_str(&self) -> &'static str {
        T::DTYPE.as_str()
    }
}

impl<'a, T: DType + Sync> KernelInput<T> for &'a TensorView<'a, T> {
    type Stored = &'a TensorView<'a, T>;
    type Returned = &'a TensorView<'a, T>;
    fn prepare(self) -> Self::Stored {
        self
    }
    fn recover(stored: Self::Stored) -> Self::Returned {
        stored
    }
}

impl<'a, T: DType + Sync> IntoDeviceOp<&'a TensorView<'a, T>> for &'a TensorView<'a, T> {
    type Op = Value<&'a TensorView<'a, T>>;
    fn into_op(self) -> Value<&'a TensorView<'a, T>> {
        value(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A fake foreign allocation with a caller-chosen byte length; the pointer is
    // never dereferenced on the metadata-only paths exercised here.
    struct FakeAlloc {
        len_bytes: usize,
    }
    unsafe impl DeviceAllocation for FakeAlloc {
        fn device_ptr(&self) -> CUdeviceptr {
            16 // 16-aligned sentinel; not dereferenced in from_foreign's spec calc
        }
        fn len_bytes(&self) -> usize {
            self.len_bytes
        }
        fn device_id(&self) -> usize {
            0
        }
    }

    #[test]
    fn addressable_bytes_accounts_for_strides() {
        // Contiguous 4-element f32: extent == logical size.
        assert_eq!(addressable_bytes::<f32>(&[4], &[1]), 4 * 4);
        // Strided: last element at offset (4-1)*10 = 30, extent 31 elements.
        assert_eq!(addressable_bytes::<f32>(&[4], &[10]), 31 * 4);
        // Empty shape addresses nothing.
        assert_eq!(addressable_bytes::<f32>(&[0], &[1]), 0);
    }

    #[test]
    fn from_foreign_accepts_allocation_covering_strided_extent() {
        // 4 elements at stride 10 → needs 31 f32 = 124 bytes.
        let owner: Arc<dyn DeviceAllocation> = Arc::new(FakeAlloc { len_bytes: 124 });
        // SAFETY: metadata-only test; the pointer is never dereferenced and
        // nothing else aliases the fake allocation.
        let _t = unsafe { Tensor::<f32>::from_foreign(owner, vec![4], vec![10]) };
    }

    #[test]
    #[should_panic(expected = "addressable extent")]
    fn from_foreign_rejects_strides_overrunning_allocation() {
        // Logical size is 16 bytes, but stride 10 addresses 124 bytes; an owner
        // that only covers the logical size must be rejected at construction.
        let owner: Arc<dyn DeviceAllocation> = Arc::new(FakeAlloc { len_bytes: 16 });
        // SAFETY: as above — construction panics before any use.
        let _t = unsafe { Tensor::<f32>::from_foreign(owner, vec![4], vec![10]) };
    }

    #[test]
    #[should_panic(expected = "addressable extent overflowed")]
    fn from_foreign_rejects_extent_overflow_instead_of_wrapping() {
        // Two i32::MAX-by-i32::MAX axes: the element offset sum (~2^63) fits
        // usize, but ×size_of::<f32>() wraps past 2^64. Unchecked arithmetic
        // would wrap to a tiny extent and ACCEPT this against a 16-byte owner
        // — an out-of-bounds borrow through the constructed-time validation.
        let owner: Arc<dyn DeviceAllocation> = Arc::new(FakeAlloc { len_bytes: 16 });
        // SAFETY: construction panics before any use.
        let _t = unsafe {
            Tensor::<f32>::from_foreign(owner, vec![i32::MAX, i32::MAX], vec![i32::MAX, i32::MAX])
        };
    }

    // Metadata-only tensor for the view/reshape validation tests: `from_meta`
    // allocates nothing and its device pointer is never read on these paths.
    fn meta_f32(shape: &[i32]) -> Tensor<f32> {
        Tensor::<f32>::from_meta(shape.to_vec(), 0)
    }

    #[test]
    fn forgotten_storage_lease_keeps_allocation_and_access_exclusion() {
        let tensor = meta_f32(&[8]);
        let weak = Arc::downgrade(&tensor.storage);
        let lease = tensor.storage.acquire(1, true, false).unwrap();
        std::mem::forget(lease);
        assert!(tensor.storage.acquire(2, false, false).is_err());
        assert!(tensor.storage.acquire(2, true, false).is_err());
        drop(tensor);
        assert!(weak.upgrade().is_some());
    }

    #[test]
    fn storage_leases_allow_reads_and_ordered_same_stream_writes() {
        let tensor = meta_f32(&[8]);
        let read1 = tensor.storage.acquire(1, false, false).unwrap();
        let read2 = tensor.storage.acquire(2, false, false).unwrap();
        assert!(tensor.storage.acquire(1, true, false).is_err());
        drop(read2);
        let write = tensor.storage.acquire(1, true, false).unwrap();
        assert!(tensor.storage.acquire(2, false, false).is_err());
        drop((read1, write));
        assert!(tensor.storage.acquire(2, true, false).is_ok());
    }

    #[test]
    fn internal_storage_leases_do_not_hide_user_aliases() {
        let mut tensor = meta_f32(&[8]);
        let lease = tensor.storage.acquire(1, true, false).unwrap();
        assert!(tensor.has_unique_storage());
        drop((&mut tensor).partition([4]));
        let alias = tensor.storage.clone();
        assert!(!tensor.has_unique_storage());
        drop(lease);
        assert!(!tensor.has_unique_storage());
        drop(alias);
        assert!(tensor.has_unique_storage());
    }

    #[test]
    fn recording_keeps_storage_without_claiming_executed_access() {
        let tensor = meta_f32(&[8]);
        let recorded = tensor.storage.acquire(1, true, true).unwrap();
        assert!(tensor.storage.acquire(2, true, false).is_ok());
        let executing = tensor.storage.acquire(2, true, false).unwrap();
        assert!(recorded.storage.acquire(1, true, false).is_err());
        drop(executing);
        assert!(recorded.storage.acquire(1, true, false).is_ok());
    }

    #[test]
    fn reshape_and_view_accept_same_size_contiguous_shapes() {
        let t = meta_f32(&[8]);
        assert!(t.view(&[2, 4]).is_ok());
        assert!(t.view(&[2, 4]).unwrap().view(&[4, 2]).is_ok());
        let t = t.reshape(&[2, 2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2, 2]);
        assert!(t.is_contiguous());
    }

    #[test]
    fn reshape_and_view_reject_element_count_mismatch() {
        assert!(meta_f32(&[8]).reshape(&[3, 3]).is_err());
        assert!(meta_f32(&[8]).view(&[5]).is_err());
        assert!(meta_f32(&[8]).view(&[2, 4]).unwrap().view(&[3]).is_err());
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn reshape_and_view_reject_shapes_whose_i32_product_wraps() {
        // Every dimension fits an i32, but the i32 product wraps to exactly 8:
        // (2^30 + 1) * 8 = 2^33 + 8 ≡ 8 (mod 2^32). The former wrapping
        // comparison accepted this for an 8-element tensor, producing metadata
        // that addresses ~8.6e9 elements over 32 bytes of storage.
        let wrapping = [(1usize << 30) + 1, 8];
        // A single dimension above i32::MAX truncates to 8 under `as i32`.
        let truncating = [(1usize << 32) + 8];
        let t = meta_f32(&[8]);
        for bad in [&wrapping[..], &truncating[..]] {
            assert!(t.view(bad).is_err(), "view accepted {bad:?}");
            assert!(
                t.view(&[2, 4]).unwrap().view(bad).is_err(),
                "TensorView::view accepted {bad:?}"
            );
        }
        assert!(meta_f32(&[8]).reshape(&wrapping).is_err());
        assert!(meta_f32(&[8]).reshape(&truncating).is_err());
    }

    #[test]
    fn launch_grid_is_the_ceiling_division() {
        assert_eq!(meta_f32(&[256]).partition([64]).grid().unwrap(), (4, 1, 1));
        // Partial edge tiles launch a block: 100 / 32 rounds up to 4.
        assert_eq!(
            meta_f32(&[100, 256]).partition([32, 64]).grid().unwrap(),
            (4, 4, 1)
        );
        let mut t = meta_f32(&[8, 8, 8]);
        assert_eq!(
            KernelOutputStored::grid(&(&mut t).partition([4, 8, 3])).unwrap(),
            (2, 1, 3)
        );
    }

    #[test]
    fn launch_grid_rejects_zero_partition_dimension_and_rank_mismatch() {
        // Zero partition axis: formerly a divide-by-zero panic in div_ceil.
        let err = meta_f32(&[16]).partition([0]).grid().unwrap_err();
        assert!(err.to_string().contains("Partition dimensions"), "{err}");
        let err = meta_f32(&[16, 16]).partition([8, 0]).grid().unwrap_err();
        assert!(err.to_string().contains("Partition dimensions"), "{err}");
        // Partition rank below the tensor rank: formerly an index-out-of-bounds panic.
        let err = meta_f32(&[16, 16]).partition([8]).grid().unwrap_err();
        assert!(err.to_string().contains("rank"), "{err}");
        // The same checks hold on the launcher-facing KernelOutputStored path.
        let mut t = meta_f32(&[16]);
        assert!(KernelOutputStored::grid(&(&mut t).partition([0])).is_err());
        assert!(meta_f32(&[2, 2, 2, 2])
            .partition([1, 1, 1, 1])
            .grid()
            .is_err());
    }

    #[test]
    fn borrowed_mutable_partition_requires_unique_storage() {
        let mut t = meta_f32(&[8]);
        // SAFETY: metadata-only; the alias is never launched or dereferenced.
        let _alias = unsafe { t.into_shared_alias() };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = (&mut t).partition([8]);
        }));
        assert!(
            result.is_err(),
            "a &mut partition over shared storage must be rejected"
        );
    }

    #[test]
    fn try_partition_returns_err_for_shared_storage() {
        let t = meta_f32(&[8]);
        // SAFETY: metadata-only; the alias is never launched or dereferenced.
        let _alias = unsafe { t.into_shared_alias() };
        let err = Arc::new(t)
            .try_partition([8])
            .err()
            .expect("shared storage must be an Err, not a panic");
        assert!(err.to_string().contains("shared"), "{err}");
        // Unique storage partitions fine.
        assert!(Arc::new(meta_f32(&[8])).try_partition([8]).is_ok());
    }

    #[test]
    fn reshape_and_view_reject_non_contiguous_source() {
        // 4 elements at stride 2 address 7 f32 = 28 bytes; the tensor is not
        // contiguous, so a contiguous-stride view would read other elements.
        let owner: Arc<dyn DeviceAllocation> = Arc::new(FakeAlloc { len_bytes: 28 });
        // SAFETY: metadata-only test; the pointer is never dereferenced.
        let strided = unsafe { Tensor::<f32>::from_foreign(owner, vec![4], vec![2]) };
        assert!(!strided.is_contiguous());
        assert!(strided.view(&[2, 2]).is_err());
        assert!(strided.reshape(&[2, 2]).is_err());
    }

    #[test]
    fn swizzle_accepts_tile_block_count() {
        let partition = MappedLaunchPartition {
            partition: (),
            map_shape: vec![4, 1],
            num_tile_blocks: 3,
        };
        let launch_grid = partition.validate((2, 3, 1), 3).unwrap();
        assert_eq!(launch_grid, (3, 1, 1));
    }

    #[test]
    fn swizzle_rejects_tile_block_count_larger_than_logical_grid() {
        let partition = MappedLaunchPartition {
            partition: (),
            map_shape: vec![4, 1],
            num_tile_blocks: 7,
        };
        let err = partition.validate((2, 3, 1), 7).unwrap_err();
        assert!(err
            .to_string()
            .contains("num_tile_blocks cannot exceed the streamed logical tile count"));
    }

    #[test]
    fn owned_axis_excluded_from_streamed_tile_count() {
        // Axis 1 is OWNED: only the 2-tile axis 0 is streamed, so
        // num_tile_blocks is capped at 2 rather than 2*3.
        let partition = MappedLaunchPartition {
            partition: (),
            map_shape: vec![1, OWNED],
            num_tile_blocks: 2,
        };
        let launch_grid = partition.validate((2, 3, 1), 2).unwrap();
        assert_eq!(launch_grid, (2, 1, 1));

        let partition = MappedLaunchPartition {
            partition: (),
            map_shape: vec![1, OWNED],
            num_tile_blocks: 3,
        };
        let err = partition.validate((2, 3, 1), 3).unwrap_err();
        assert!(err
            .to_string()
            .contains("num_tile_blocks cannot exceed the streamed logical tile count"));
    }

    #[test]
    fn owned_axis_rejects_all_owned_map() {
        let partition = MappedLaunchPartition {
            partition: (),
            map_shape: vec![OWNED, OWNED],
            num_tile_blocks: 1,
        };
        let err = partition.validate((2, 3, 1), 1).unwrap_err();
        assert!(err
            .to_string()
            .contains("at least one streamed (non-OWNED) map axis"));
    }

    #[test]
    fn owned_axis_accepts_leading_owned() {
        // Owned axes may sit at any position: axis 0 owned, axis 1 streamed.
        let partition = MappedLaunchPartition {
            partition: (),
            map_shape: vec![OWNED, 1],
            num_tile_blocks: 3,
        };
        let launch_grid = partition.validate((2, 3, 1), 3).unwrap();
        assert_eq!(launch_grid, (3, 1, 1));
    }

    #[test]
    fn swizzle_rejects_zero_tile_blocks() {
        let partition = MappedLaunchPartition {
            partition: (),
            map_shape: vec![4, 1],
            num_tile_blocks: 0,
        };
        let err = partition.validate((2, 3, 1), 0).unwrap_err();
        assert!(err.to_string().contains("num_tile_blocks > 0"));
    }

    #[test]
    fn swizzle_rejects_grid_rank_above_map_rank() {
        let partition = MappedLaunchPartition {
            partition: (),
            map_shape: vec![4, 1],
            num_tile_blocks: 3,
        };
        let err = partition.validate((2, 3, 4), 3).unwrap_err();
        assert!(err
            .to_string()
            .contains("map rank must match the logical partition grid rank"));
    }

    #[test]
    fn swizzle_accepts_rank1_map() {
        let partition = MappedLaunchPartition {
            partition: (),
            map_shape: vec![1],
            num_tile_blocks: 4,
        };
        let launch_grid = partition.validate((8, 1, 1), 4).unwrap();
        assert_eq!(launch_grid, (4, 1, 1));
    }

    #[test]
    fn swizzle_accepts_rank3_map() {
        let partition = MappedLaunchPartition {
            partition: (),
            map_shape: vec![1, 4, 1],
            num_tile_blocks: 6,
        };
        let launch_grid = partition.validate((2, 3, 4), 6).unwrap();
        assert_eq!(launch_grid, (6, 1, 1));
    }

    #[test]
    fn swizzle_rejects_rank1_map_with_2d_grid() {
        let partition = MappedLaunchPartition {
            partition: (),
            map_shape: vec![1],
            num_tile_blocks: 2,
        };
        let err = partition.validate((2, 3, 1), 2).unwrap_err();
        assert!(err
            .to_string()
            .contains("map rank must match the logical partition grid rank"));
    }

    #[test]
    fn swizzle_rejects_map_rank_above_3() {
        let partition = MappedLaunchPartition {
            partition: (),
            map_shape: vec![1, 1, 1, 1],
            num_tile_blocks: 2,
        };
        let err = partition.validate((2, 3, 4), 2).unwrap_err();
        assert!(err.to_string().contains("rank-1 through rank-3 map shape"));
    }
}
