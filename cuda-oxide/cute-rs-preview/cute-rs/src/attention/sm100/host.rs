/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Host construction of the rank-three tensor maps consumed by attention tiles.
use super::SmemLayout;
use crate::copy::tma::TmaDesc;
use alloc::{format, string::String};
use core::{ffi::c_void, marker::PhantomData, mem::MaybeUninit};
use cuda_core::sys as driver;

/// Encode a physical row-major matrix into the selected shared operand layout.
///
/// Large contiguous dimensions are split into swizzle-sized chunks. For an
/// MN-major operand the logical rows/columns are transposed relative to global
/// physical rows/columns. Global coordinates always use the physical ordering.
/// The descriptor does not retain the allocation or track its lifetime; callers
/// keep the allocation live until every asynchronous use completes.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn make_tma_desc<L: SmemLayout>(
    base: *mut c_void,
    rows: u64,
    cols: u64,
    row_stride_elements: u64,
) -> Result<TmaDesc<f16, L>, String> {
    let () = L::VALID;
    let span = L::SPAN;
    let (tile_rows, tile_cols) = if L::MN_MAJOR {
        (L::COLS, L::ROWS)
    } else {
        (L::ROWS, L::COLS)
    };
    if !matches!(span, 32 | 64)
        || tile_rows == 0
        || !tile_rows.is_multiple_of(8)
        || tile_rows > 256
        || tile_cols == 0
        || !tile_cols.is_multiple_of(span)
    {
        return Err("unsupported SM100 shared TMA tile".into());
    }
    if rows == 0 || cols == 0 || !cols.is_multiple_of(u64::from(span)) || row_stride_elements < cols
    {
        return Err("TMA matrix must be nonempty, with contiguous dimension divisible by its swizzle span and pitch at least the column extent".into());
    }
    if base.is_null() || !(base as usize).is_multiple_of(16) {
        return Err("TMA global base must be nonnull and 16-byte aligned".into());
    }
    let pitch = row_stride_elements
        .checked_mul(2)
        .ok_or("TMA byte pitch overflow")?;
    if pitch % 16 != 0 {
        return Err("TMA byte pitch must be a multiple of 16".into());
    }
    if rows > u32::MAX as u64 || cols > u32::MAX as u64 || pitch >= (1u64 << 40) {
        return Err("TMA extent or pitch exceeds the driver encoding".into());
    }
    let dims = [u64::from(span), rows, cols / u64::from(span)];
    let strides = [pitch, u64::from(span) * 2];
    let tile = [span, tile_rows, tile_cols / span];
    let element_strides = [1u32; 3];
    let swizzle = if span == 32 {
        driver::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_64B
    } else {
        driver::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B
    };
    let mut out = MaybeUninit::<driver::CUtensorMap>::uninit();
    let status = unsafe {
        driver::cuTensorMapEncodeTiled(
            out.as_mut_ptr(),
            driver::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            3,
            base,
            dims.as_ptr(),
            strides.as_ptr(),
            tile.as_ptr(),
            element_strides.as_ptr(),
            driver::CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle,
            driver::CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            driver::CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };
    if status != driver::cudaError_enum_CUDA_SUCCESS {
        return Err(format!(
            "SM100 rank-three cuTensorMapEncodeTiled failed: {status:?}"
        ));
    }
    Ok(TmaDesc {
        bytes: unsafe { core::mem::transmute::<driver::CUtensorMap, [u8; 128]>(out.assume_init()) },
        marker: PhantomData,
    })
}
