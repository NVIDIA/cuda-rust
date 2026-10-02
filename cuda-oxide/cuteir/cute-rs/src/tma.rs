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
//! host:    TmaDesc<f32, SmemL>  ->  kernel: &TmaDesc<f32, SmemL>
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
/// A `#[grid_constant] desc: &TmaDesc<T, SmemL>` kernel parameter takes this
/// value at launch. Every thread borrows the same read-only descriptor bytes.
/// The tensor data stays in device memory and must remain live until the GPU
/// work completes.
#[repr(C, align(64))]
pub struct TmaDesc<T, SmemL> {
    /// Descriptor bytes produced by the CUDA driver.
    pub bytes: [u8; 128],
    /// Zero-byte marker that keeps `T` and `SmemL` in the type.
    pub marker: PhantomData<(T, SmemL)>,
}

// The type parameters describe the encoded bytes; they are not stored values.
impl<T, SmemL> Copy for TmaDesc<T, SmemL> {}

impl<T, SmemL> Clone for TmaDesc<T, SmemL> {
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
/// [`Self::None`] is the default used by `make_tma_desc_2d`. Select
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

/// Host options for `make_tma_desc_2d_with_options`.
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
    tma_swizzle_mode(SmemL::SWIZZLE_B, SmemL::SWIZZLE_M + log2, SmemL::SWIZZLE_S)
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

/// Number of bytes in one `SmemL` tile of `T`.
///
/// Pass this number to the load pipeline's expected-transaction operation for
/// [`copy_tma_2d`]. In a kernel, compute it as a constant so it creates no
/// runtime function call:
///
/// ```text
/// const N: u32 = tile_bytes::<T, SmemL>();
/// ```
#[inline(always)]
pub const fn tile_bytes<T, SmemL: ReifySmem2D>() -> u32 {
    (SmemL::ROWS * SmemL::COLS) as u32 * size_of::<T>() as u32
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
/// row = r * SmemL::ROWS
/// col = c * SmemL::COLS
/// ```
///
/// The caller must initialize the `mbarrier`, attach [`tile_bytes`] as its
/// expected byte count, and wait for it.
///
/// # Safety
///
/// Exactly one thread in the block must call this function. `barrier` and
/// `dst_smem_tile` must be valid objects in this block's shared memory. Use
/// the lower-level raw TMA APIs for remote or multicast cluster copies.
///
/// The destination address must meet the alignment required by the layout's
/// swizzle phase. `desc` must have been built by `make_tma_desc_2d` for the
/// same `SmemL`. The packed-MXFP4 B64 layout described above requires a
/// 512-byte-aligned buffer base; the CuTeDSL comparison kernel uses 1024.
#[inline(never)]
pub unsafe fn copy_tma_2d<T, SmemL>(
    desc: *const TmaDesc<T, SmemL>,
    coord: (usize, usize),
    dst_smem_tile: &mut crate::SmemTile<T, SmemL>,
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
/// `(r * SmemL::ROWS, c * SmemL::COLS)`.
///
/// TMA stores are tracked in copy groups rather than `mbarrier` objects. TMA
/// observes shared memory through an asynchronous *proxy*, a hardware view of
/// memory separate from ordinary thread accesses. The caller must publish
/// earlier ordinary shared writes to that proxy, issue the stores, commit the
/// group, and wait before reusing the source. [`crate::TmaStorePipeline`]
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
    desc: *const TmaDesc<T, SmemL>,
    coord: (usize, usize),
    src_smem_tile: crate::SmemTile<T, SmemL>,
) {
    let _ = (desc, coord, src_smem_tile);
    unreachable!("cute-rs `copy_tma_s2g_2d` executed outside device compilation")
}

#[cfg(feature = "host")]
mod host {
    use super::{
        ReifySmem2D, TmaDesc, TmaEncodeOptions, TmaL2Promotion, TmaSwizzleMode,
        tma_swizzle_mode_for, tma_swizzle_span_bytes,
    };
    use core::marker::PhantomData;
    use core::mem::MaybeUninit;
    use cuda_core::sys as driver;

    /// An element type TMA can encode in a descriptor.
    ///
    /// `DATA_TYPE` is the CUDA driver value. `BYTES` is one element's size.
    pub trait TmaElement {
        /// CUDA driver code for this element type.
        const DATA_TYPE: driver::CUtensorMapDataType;
        /// Bytes in one element.
        const BYTES: u32;
    }

    /// Raw bytes, including one byte containing two packed FP4 values.
    impl TmaElement for u8 {
        const DATA_TYPE: driver::CUtensorMapDataType =
            driver::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_UINT8;
        const BYTES: u32 = 1;
    }

    impl TmaElement for f32 {
        const DATA_TYPE: driver::CUtensorMapDataType =
            driver::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        const BYTES: u32 = 4;
    }

    impl TmaElement for f16 {
        const DATA_TYPE: driver::CUtensorMapDataType =
            driver::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
        const BYTES: u32 = 2;
    }

    /// Raw 16-bit values, including BF16 bit patterns. TMA only moves bytes.
    impl TmaElement for u16 {
        const DATA_TYPE: driver::CUtensorMapDataType =
            driver::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_UINT16;
        const BYTES: u32 = 2;
    }

    /// Build a descriptor that loads or stores one `SmemL` tile.
    ///
    /// The global matrix is row-major. Types provide the tile size, swizzle,
    /// and element format. Arguments provide the global pointer, matrix size,
    /// and row pitch in elements. Unsupported layouts return an error; they
    /// are never rounded to a nearby TMA mode.
    pub fn make_tma_desc_2d<T: TmaElement, SmemL: ReifySmem2D>(
        global_base: *mut core::ffi::c_void,
        rows: u64,
        cols: u64,
        ld_elements: u64,
    ) -> Result<TmaDesc<T, SmemL>, alloc::string::String> {
        make_tma_desc_2d_with_options(
            global_base,
            rows,
            cols,
            ld_elements,
            TmaEncodeOptions::DEFAULT,
        )
    }

    /// Build a row-major descriptor with an explicit L2 cache policy.
    ///
    /// Tile shape and validation match [`make_tma_desc_2d`]. Use
    /// [`TmaEncodeOptions::L2_128B`] to match the usual CUTLASS/CuTeDSL input
    /// policy. The shorter function keeps the original no-promotion default.
    // The driver stores this address in the descriptor. It does not read from
    // `global_base` while encoding it.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn make_tma_desc_2d_with_options<T: TmaElement, SmemL: ReifySmem2D>(
        global_base: *mut core::ffi::c_void,
        rows: u64,
        cols: u64,
        ld_elements: u64,
        options: TmaEncodeOptions,
    ) -> Result<TmaDesc<T, SmemL>, alloc::string::String> {
        use alloc::format;

        if SmemL::ROWS <= 0 || SmemL::COLS <= 0 {
            return Err(format!(
                "TMA tile extents must be positive, got {} x {}",
                SmemL::ROWS,
                SmemL::COLS
            ));
        }
        if SmemL::ROW_STRIDE != SmemL::COLS || SmemL::COL_STRIDE != 1 {
            return Err(format!(
                "TMA v0 requires a row-major shared tile, got strides ({}, {})",
                SmemL::ROW_STRIDE,
                SmemL::COL_STRIDE
            ));
        }
        if SmemL::OFFSET != 0 {
            return Err(format!(
                "TMA cannot express a composed offset (got {})",
                SmemL::OFFSET
            ));
        }
        let Some(mode) = tma_swizzle_mode_for::<SmemL>(T::BYTES) else {
            return Err(format!(
                "swizzle S<{},{},{}> (elements) is not one of TMA's encodable modes",
                SmemL::SWIZZLE_B,
                SmemL::SWIZZLE_M,
                SmemL::SWIZZLE_S
            ));
        };
        if let Some(span_bytes) = tma_swizzle_span_bytes(mode) {
            let pitch_bytes = SmemL::COLS
                .checked_mul(i64::from(T::BYTES))
                .ok_or_else(|| {
                    alloc::string::String::from("TMA shared-tile pitch overflows i64")
                })?;
            if pitch_bytes != span_bytes {
                return Err(format!(
                    "swizzled TMA tile pitch must equal the {span_bytes}-byte swizzle span, got {pitch_bytes} bytes"
                ));
            }
        }
        let swizzle = match mode {
            TmaSwizzleMode::None => driver::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE,
            TmaSwizzleMode::B32 => driver::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_32B,
            TmaSwizzleMode::B64 => driver::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_64B,
            TmaSwizzleMode::B128 => driver::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
        };
        if ld_elements < cols {
            return Err(format!(
                "pitch {ld_elements} is smaller than the column count {cols}"
            ));
        }

        // TMA lists the innermost dimension first. Dimension 0 is columns,
        // which are contiguous in a row-major matrix.
        let global_dim: [u64; 2] = [cols, rows];
        let global_strides: [u64; 1] = [ld_elements * u64::from(T::BYTES)];
        let box_dim: [u32; 2] = [SmemL::COLS as u32, SmemL::ROWS as u32];
        let element_strides: [u32; 2] = [1, 1];
        let l2_promotion = match options.l2_promotion {
            TmaL2Promotion::None => {
                driver::CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE
            }
            TmaL2Promotion::B64 => {
                driver::CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_L2_64B
            }
            TmaL2Promotion::B128 => {
                driver::CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_L2_128B
            }
            TmaL2Promotion::B256 => {
                driver::CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_L2_256B
            }
        };

        let mut tensor_map = MaybeUninit::<driver::CUtensorMap>::uninit();
        let result = unsafe {
            driver::cuTensorMapEncodeTiled(
                tensor_map.as_mut_ptr(),
                T::DATA_TYPE,
                2,
                global_base,
                global_dim.as_ptr(),
                global_strides.as_ptr(),
                box_dim.as_ptr(),
                element_strides.as_ptr(),
                driver::CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
                swizzle,
                l2_promotion,
                driver::CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            )
        };
        if result != driver::cudaError_enum_CUDA_SUCCESS {
            return Err(format!("cuTensorMapEncodeTiled failed: {result:?}"));
        }
        let map = unsafe { tensor_map.assume_init() };
        const _: () = assert!(core::mem::size_of::<driver::CUtensorMap>() == 128);
        Ok(TmaDesc {
            bytes: unsafe { core::mem::transmute::<driver::CUtensorMap, [u8; 128]>(map) },
            marker: PhantomData,
        })
    }
}

#[cfg(feature = "host")]
pub use host::{TmaElement, make_tma_desc_2d, make_tma_desc_2d_with_options};

#[cfg(test)]
mod tests {
    #[test]
    fn descriptor_value_abi_does_not_require_copy_marker_types() {
        // Layout and element markers need not be Copy: only descriptor bytes move.
        struct Marker;
        fn assert_copy<T: Copy>() {}
        assert_copy::<super::TmaDesc<Marker, Marker>>();
        assert_eq!(core::mem::size_of::<super::TmaDesc<Marker, Marker>>(), 128);
        assert_eq!(core::mem::align_of::<super::TmaDesc<Marker, Marker>>(), 64);
    }

    use super::*;

    #[test]
    fn only_the_hardware_modes_are_encodable() {
        assert_eq!(tma_swizzle_mode(0, 9, 9), Some(TmaSwizzleMode::None));
        assert_eq!(tma_swizzle_mode(1, 4, 3), Some(TmaSwizzleMode::B32));
        assert_eq!(tma_swizzle_mode(2, 4, 3), Some(TmaSwizzleMode::B64));
        assert_eq!(tma_swizzle_mode(3, 4, 3), Some(TmaSwizzleMode::B128));
        // Similar but unsupported patterns return None instead of rounding.
        assert_eq!(tma_swizzle_mode(3, 5, 3), None);
        assert_eq!(tma_swizzle_mode(3, 4, 4), None);
        assert_eq!(tma_swizzle_mode(4, 4, 3), None);
    }

    #[test]
    fn element_recast_reaches_the_byte_modes() {
        use crate::markers::{Composed, ReifySmem2D, RowMajor, Swizzle};
        // For f16, element pattern S<3,3,3> becomes byte pattern S<3,4,3>,
        // which is TMA B128.
        type Smem = Composed<Swizzle<3, 3, 3>, 0, RowMajor<8, 64>>;
        assert_eq!(Smem::SWIZZLE_B, 3);
        assert_eq!(tma_swizzle_mode_for::<Smem>(2), Some(TmaSwizzleMode::B128));
        // For f32, it becomes S<3,5,3>, which TMA cannot encode.
        assert_eq!(tma_swizzle_mode_for::<Smem>(4), None);
        // Invalid element sizes fail instead of being rounded.
        assert_eq!(tma_swizzle_mode_for::<Smem>(0), None);
        assert_eq!(tma_swizzle_mode_for::<Smem>(3), None);
    }

    #[test]
    fn swizzle_modes_name_their_exact_row_spans() {
        assert_eq!(tma_swizzle_span_bytes(TmaSwizzleMode::None), None);
        assert_eq!(tma_swizzle_span_bytes(TmaSwizzleMode::B32), Some(32));
        assert_eq!(tma_swizzle_span_bytes(TmaSwizzleMode::B64), Some(64));
        assert_eq!(tma_swizzle_span_bytes(TmaSwizzleMode::B128), Some(128));
    }

    #[test]
    fn flattened_mxfp4_ab_tile_is_8192_bytes_with_b64_swizzle() {
        use crate::markers::{Composed, RowMajor, Swizzle};

        // Two FP4 values share each byte: 128 K values become 64 bytes.
        // Byte pattern S<2,4,3> is TMA B64.
        type ByteSmem = Composed<Swizzle<2, 4, 3>, 0, RowMajor<128, 64>>;
        assert_eq!(tile_bytes::<u8, ByteSmem>(), 8_192);
        assert_eq!(
            tma_swizzle_mode_for::<ByteSmem>(1),
            Some(TmaSwizzleMode::B64)
        );
    }

    #[test]
    fn canonical_mxfp4_scale_tile_is_512_bytes_without_swizzle() {
        use crate::markers::RowMajor;

        // Standard SM120 packing: 128 rows x 4 UE8M0 bytes = 512 bytes.
        // Viewing those bytes as u16 creates one legal 1x256 TMA tile.
        type ScaleSmem = RowMajor<1, 256>;
        assert_eq!(tile_bytes::<u16, ScaleSmem>(), 512);
        assert_eq!(
            tma_swizzle_mode_for::<ScaleSmem>(2),
            Some(TmaSwizzleMode::None)
        );
    }

    #[test]
    fn descriptor_options_preserve_default_and_offer_cutedsl_policy() {
        assert_eq!(
            TmaEncodeOptions::default().l2_promotion,
            TmaL2Promotion::None
        );
        assert_eq!(TmaEncodeOptions::L2_128B.l2_promotion, TmaL2Promotion::B128);
    }

    #[cfg(feature = "host")]
    #[test]
    fn byte_carrier_has_uint8_tma_format() {
        use cuda_core::sys as driver;

        assert_eq!(<u8 as TmaElement>::BYTES, 1);
        assert_eq!(
            <u8 as TmaElement>::DATA_TYPE,
            driver::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_UINT8
        );
    }
}
