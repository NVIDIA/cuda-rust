/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host-side descriptor encoding and element contracts.

use super::{
    ReifySmem2D, TmaDesc, TmaEncodeOptions, TmaL2Promotion, TmaSwizzleMode, tma_swizzle_mode_for,
    tma_swizzle_span_bytes,
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
            .ok_or_else(|| alloc::string::String::from("TMA shared-tile pitch overflows i64"))?;
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
        TmaL2Promotion::None => driver::CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
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
    /// Check that driver descriptor storage remains exactly 128 bytes before transmuting.
    const _: () = assert!(core::mem::size_of::<driver::CUtensorMap>() == 128);
    Ok(TmaDesc {
        bytes: unsafe { core::mem::transmute::<driver::CUtensorMap, [u8; 128]>(map) },
        marker: PhantomData,
    })
}
