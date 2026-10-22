/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed tile copies with the Tensor Memory Accelerator (TMA).
//!
//! TMA is hardware that moves a whole tile between global memory and shared
//! memory. A thread is one GPU worker. A block, also called a CTA, is a group
//! of threads that can use the same shared memory.
//!
//! Before launch, the host builds a 128-byte *descriptor*. It contains the
//! global address, sizes, element type, and tile layout. The kernel gives that
//! descriptor to TMA. Here, Rust also puts the element and shared layout in
//! the descriptor type:
//!
//! ```text
//! host:    TensorMap<f32, SmemL>  ->  kernel: &TensorMap<f32, SmemL>
//!          descriptor value       #[grid_constant] parameter
//!                     same element and layout types
//! ```
//!
//! If host and kernel layouts differ, the program does not compile. A
//! *swizzle* is a fixed address rearrangement that reduces shared-memory bank
//! conflicts: several threads requesting the same memory bank and having to
//! wait. Both sides read sizes, strides, and the swizzle from [`ReifySmem2D`].
//!
//! MXFP4 is a block-scaled 4-bit floating-point format. It uses ordinary byte
//! and 16-bit transport views:
//!
//! ```text
//! A or B: 128x128 FP4 values -> 128x64 bytes -> 8192 bytes
//!          two FP4 values share each byte; use a 64-byte (B64) swizzle
//!
//! scales: 128x4 UE8M0 bytes -> 1x256 u16 values -> 512 bytes
//!          UE8M0 is an 8-bit power-of-two scale
//!          u16 is only the unit TMA uses to move the same bytes
//! ```
//!
//! The scale layout expects the standard SM120 packing. After the copy, the
//! kernel may read the same 512 bytes as 128 packed `u32` scale words.

use core::marker::PhantomData;

use crate::markers::ReifySmem2D;

/// One 128-byte TMA descriptor typed by element and shared layout.
///
/// Its runtime bytes have the same layout as the CUDA driver's `CUtensorMap`.
/// A `#[grid_constant] desc: &TensorMap<T, SmemL>` kernel parameter takes this
/// value at launch. Every thread borrows the same read-only descriptor bytes.
/// The tensor data stays in device memory and must remain live until the GPU
/// work completes.
#[repr(C, align(64))]
pub struct TensorMap<T, SmemL> {
    /// Descriptor bytes produced by the CUDA driver.
    pub bytes: [u8; 128],
    /// Zero-byte marker that keeps `T` and `SmemL` in the type.
    pub marker: PhantomData<(T, SmemL)>,
}

impl<T, SmemL> TensorMap<T, SmemL> {
    /// Prefetch this descriptor into the device's tensor-map cache.
    ///
    /// This is CuTeDSL's `cpasync.prefetch_descriptor`. Issuer selection
    /// remains with the caller, so a kernel can prefetch its descriptors once
    /// before starting TMA transfers.
    ///
    /// # Safety
    ///
    /// Call only on a device supporting TMA. `self` must contain a valid encoded
    /// CUDA tensor map in device-accessible storage, such as a grid-constant
    /// kernel argument, and remain live through the prefetch.
    #[inline(always)]
    pub unsafe fn prefetch_descriptor(&self) {
        unsafe {
            ctm_rs::tma::prefetch_tensormap(
                core::ptr::from_ref(self).cast::<cuda_device::TmaDescriptor>(),
            )
        };
    }
}

// The type parameters describe the encoded bytes; they are not stored values.
impl<T, SmemL> Copy for TensorMap<T, SmemL> {}

impl<T, SmemL> Clone for TensorMap<T, SmemL> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Shared-memory swizzles supported by TMA.
///
/// TMA supports only these byte-based patterns. Each keeps 16-byte chunks
/// together while rearranging the surrounding addresses:
///
/// ```text
/// mode        byte-based pattern
/// None        no rearrangement
/// B32         S<1,4,3>
/// B64         S<2,4,3>
/// B128        S<3,4,3>
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TmaSwizzleMode {
    /// Keep shared-memory addresses in their original order.
    None,
    /// Rearrange addresses within a 32-byte span.
    B32,
    /// Rearrange addresses within a 64-byte span.
    B64,
    /// Rearrange addresses within a 128-byte span.
    B128,
}

/// Requested level-2 (L2) cache fetch size for a TMA descriptor.
///
/// [`Self::None`] is the default used by `create_tensor_map_tiled`. Select
/// [`Self::B128`] to request a 128-byte fetch for GEMM input tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TmaL2Promotion {
    /// Do not request a larger L2 fetch.
    None,
    /// Fetch 64-byte L2 sectors.
    B64,
    /// Fetch 128-byte L2 sectors.
    B128,
    /// Fetch 256-byte L2 sectors.
    B256,
}

/// Host options for `create_tensor_map_tiled_with_options`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TmaEncodeOptions {
    /// L2 fetch size requested for this descriptor.
    pub l2_promotion: TmaL2Promotion,
}

impl TmaEncodeOptions {
    /// Do not request a larger L2 fetch.
    pub const DEFAULT: Self = Self {
        l2_promotion: TmaL2Promotion::None,
    };

    /// Request 128-byte L2 sectors.
    pub const L2_128B: Self = Self {
        l2_promotion: TmaL2Promotion::B128,
    };
}

impl Default for TmaEncodeOptions {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Convert a byte-based `S<bits,m_bytes,shift>` swizzle to a TMA mode.
///
/// Returns `None` when hardware cannot represent the exact pattern. It never
/// rounds to a nearby mode.
pub const fn tma_swizzle_mode(bits: u32, m_bytes: u32, shift: i32) -> Option<TmaSwizzleMode> {
    if bits == 0 {
        return Some(TmaSwizzleMode::None);
    }
    match (bits, m_bytes, shift) {
        (1, 4, 3) => Some(TmaSwizzleMode::B32),
        (2, 4, 3) => Some(TmaSwizzleMode::B64),
        (3, 4, 3) => Some(TmaSwizzleMode::B128),
        _ => None,
    }
}

/// Convert `SmemL`'s element-based swizzle to a byte-based TMA mode.
///
/// `element_bytes` shifts the address fields from elements to bytes. Returns
/// `None` if the size is zero, is not a power of two, or TMA cannot represent
/// the result exactly. The function never rounds the size.
pub const fn tma_swizzle_mode_for<SmemL: ReifySmem2D>(
    element_bytes: u32,
) -> Option<TmaSwizzleMode> {
    if !element_bytes.is_power_of_two() {
        return None;
    }
    let log2 = element_bytes.trailing_zeros();
    tma_swizzle_mode(SmemL::NUM_BITS, SmemL::NUM_BASE + log2, SmemL::NUM_SHIFT)
}

/// Bytes covered by one repeat of the swizzle, or `None` without a swizzle.
#[cfg(any(feature = "host", test))]
const fn tma_swizzle_span_bytes(mode: TmaSwizzleMode) -> Option<i64> {
    match mode {
        TmaSwizzleMode::None => None,
        TmaSwizzleMode::B32 => Some(32),
        TmaSwizzleMode::B64 => Some(64),
        TmaSwizzleMode::B128 => Some(128),
    }
}

/// Bytes covered by one `SmemL` tile of `T`: the layout's cosize times the
/// element size, as CuTeDSL's `size_in_bytes(dtype, layout)` computes it.
///
/// The cosize is the offset of the tile's last element plus one, taken from
/// the affine part of the layout (its shape and strides). A swizzle only
/// permutes addresses inside that span and the composed offset is not
/// counted, so a padded row stride enlarges the result while a swizzle does
/// not. For the dense row-major tiles TMA encodes, this equals rows times
/// columns times the element size.
///
/// Pass this number to the load pipeline's expected-transaction operation for
/// [`copy_tma_2d`]. In a kernel, compute it as a constant so it creates no
/// runtime function call:
///
/// ```text
/// const N: u32 = size_in_bytes::<T, SmemL>();
/// ```
#[inline(always)]
pub const fn size_in_bytes<T, SmemL: ReifySmem2D>() -> u32 {
    let (rows, cols) = SmemL::SHAPE;
    if rows <= 0 || cols <= 0 {
        return 0;
    }
    let (row_stride, col_stride) = SmemL::STRIDE;
    let cosize = (rows - 1) * row_stride + (cols - 1) * col_stride + 1;
    cosize as u32 * size_of::<T>() as u32
}

/// Start one TMA copy from global memory into a shared tile.
///
/// The compiler recognizes this call. `desc` and the destination must carry
/// the same `SmemL`, which Rust checks at the call. The compiler also checks
/// that TMA supports the layout.
///
/// `coord = (r, c)` counts tiles, not elements. The element origin is:
///
/// ```text
/// row = r * SmemL::SHAPE.0
/// col = c * SmemL::SHAPE.1
/// ```
///
/// The caller must initialize the `mbarrier`, attach [`size_in_bytes`] as its
/// expected byte count, and wait for it.
///
/// # Safety
///
/// Exactly one thread in the block must call this function. `barrier` and
/// `dst_smem_tile` must be valid objects in this block's shared memory. Use
/// the lower-level raw TMA APIs for remote or multicast cluster copies.
///
/// The destination address must meet the alignment required by the layout's
/// swizzle phase. `desc` must have been built by `create_tensor_map_tiled` for the
/// same `SmemL`. The packed-MXFP4 B64 layout described above requires a
/// 512-byte-aligned buffer base; the CuTeDSL comparison kernel uses 1024.
#[inline(never)]
pub unsafe fn copy_tma_2d<T, SmemL>(
    desc: *const TensorMap<T, SmemL>,
    coord: (usize, usize),
    dst_smem_tile: &mut crate::copy::cooperative::SmemTile<T, SmemL>,
    barrier: *mut cuda_device::barrier::Barrier,
) {
    let _ = (desc, coord, dst_smem_tile, barrier);
    unreachable!("cute-rs `copy_tma_2d` executed outside device compilation")
}

/// Start one TMA copy from a shared tile into global memory.
///
/// This is the reverse of [`copy_tma_2d`]. The descriptor and source carry
/// the same element and layout types, so Rust rejects mismatches. `coord`
/// counts tiles; `(r, c)` starts at element
/// `(r * SmemL::SHAPE.0, c * SmemL::SHAPE.1)`.
///
/// TMA stores are tracked in copy groups rather than `mbarrier` objects. TMA
/// observes shared memory through an asynchronous *proxy*, a hardware view of
/// memory separate from ordinary thread accesses. The caller must publish
/// earlier ordinary shared writes to that proxy, issue the stores, commit the
/// group, and wait before reusing the source. [`crate::PipelineTmaStore`]
/// provides the commit and wait steps.
///
/// The small `SmemTile` view is passed by value. This consumes only its base
/// pointer and capacity; it does not move or copy the shared-memory contents.
///
/// # Safety
///
/// Exactly one thread in the block must call this function.
/// `src_smem_tile` must be a live, correctly aligned tile in this block's
/// shared memory and use `SmemL`. `desc` must have been built for the same `T`
/// and `SmemL`. Both must remain valid until the committed copy group has
/// finished reading shared memory. Before this call, a proxy fence must make
/// all earlier ordinary shared-memory writes visible to TMA's async proxy.
#[inline(never)]
pub unsafe fn copy_tma_s2g_2d<T, SmemL>(
    desc: *const TensorMap<T, SmemL>,
    coord: (usize, usize),
    src_smem_tile: crate::copy::cooperative::SmemTile<T, SmemL>,
) {
    let _ = (desc, coord, src_smem_tile);
    unreachable!("cute-rs `copy_tma_s2g_2d` executed outside device compilation")
}

#[cfg(feature = "host")]
mod host;
#[cfg(feature = "host")]
pub use host::{TmaElement, create_tensor_map_tiled, create_tensor_map_tiled_with_options};

/// Regression tests for this module’s layout, representation and rejection contracts.
#[cfg(test)]
mod tests;
