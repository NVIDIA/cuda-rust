/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host encoding for the explicit Rust primitives tensor-map API.

use crate::{memory::PhysicalFormat, tma::TensorMap};
use core::{ffi::c_void, mem::MaybeUninit};
use cuda_core::sys as driver;
use cuda_device::TmaDescriptor;

/// Restricts driver element codes to formats supported by this encoder.
mod sealed {
    /// Implemented only for the storage elements with reviewed CUDA driver codes.
    pub trait Element {}
}
/// Element types supported by the reviewed tensor-map encoder.
pub trait TmaElement: sealed::Element {
    /// CUDA tensor-map element code used by the host driver.
    const DRIVER_TYPE: driver::CUtensorMapDataType;
}
impl sealed::Element for f16 {}
impl TmaElement for f16 {
    const DRIVER_TYPE: driver::CUtensorMapDataType =
        driver::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
}
impl sealed::Element for u8 {}
impl TmaElement for u8 {
    const DRIVER_TYPE: driver::CUtensorMapDataType =
        driver::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_UINT8;
}
impl sealed::Element for u16 {}
impl TmaElement for u16 {
    const DRIVER_TYPE: driver::CUtensorMapDataType =
        driver::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_UINT16;
}

/// L2 cache-fetch promotion requested when encoding the descriptor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum L2Promotion {
    /// Use the driver’s default fetch behavior without promotion.
    #[default]
    None,
    /// Request a 128-byte L2 fetch.
    B128,
}
/// Optional cache policy; tensor shape and swizzle come from the physical format.
#[derive(Clone, Copy, Debug, Default)]
pub struct EncodeOptions {
    /// L2 promotion passed to cuTensorMapEncodeTiled.
    pub l2_promotion: L2Promotion,
}
/// Invalid tensor-map arguments or a CUDA driver encoding failure.
#[derive(Debug, thiserror::Error)]
pub enum EncodeError {
    /// The base is null or not aligned to 16 bytes.
    #[error("tensor-map base must be non-null and 16-byte aligned")]
    InvalidBase,
    /// A global extent is zero or exceeds 2^32 elements.
    #[error("tensor-map extents must be between 1 and 2^32 inclusive")]
    InvalidExtent,
    /// Row pitch is too short, misaligned, overflowing, or outside the driver range.
    #[error("row stride must cover the row, be 16-byte aligned and be less than 2^40 bytes")]
    InvalidStride,
    /// The selected tile shape and swizzle cannot be encoded by this API.
    #[error("unsupported physical tile format")]
    InvalidFormat,
    /// The CUDA driver rejected descriptor encoding.
    #[error("cuTensorMapEncodeTiled failed with CUDA status {0}")]
    Driver(#[doc = "CUDA driver status returned by the encoder."] driver::CUresult),
}

/// Validated arguments for one rank-two CUDA tensor-map encoding.
struct EncodeParams {
    /// CUDA driver code for one storage element.
    data_type: driver::CUtensorMapDataType,
    /// Global extents in elements, ordered as columns then rows.
    dimensions: [u64; 2],
    /// Byte distance between adjacent global rows.
    strides: [u64; 1],
    /// Shared tile extents in elements, ordered as columns then rows.
    box_dimensions: [u32; 2],
    /// Sampling step within each logical dimension; both are one.
    element_strides: [u32; 2],
    /// Shared-memory byte permutation selected by the physical format.
    swizzle: driver::CUtensorMapSwizzle,
    /// Requested L2 fetch promotion passed to the driver.
    l2: driver::CUtensorMapL2promotion,
}

/// Check address, extent, pitch and physical-format limits before calling the driver.
fn validate_2d<T: TmaElement, F: PhysicalFormat<Element = T>>(
    base: usize,
    rows: u64,
    cols: u64,
    ld_elements: u64,
    options: EncodeOptions,
) -> Result<EncodeParams, EncodeError> {
    if base == 0 || !base.is_multiple_of(16) {
        return Err(EncodeError::InvalidBase);
    }
    if rows == 0 || cols == 0 || rows > 1u64 << 32 || cols > 1u64 << 32 {
        return Err(EncodeError::InvalidExtent);
    }
    let stride = ld_elements
        .checked_mul(size_of::<T>() as u64)
        .ok_or(EncodeError::InvalidStride)?;
    if ld_elements < cols || stride % 16 != 0 || stride >= 1u64 << 40 {
        return Err(EncodeError::InvalidStride);
    }
    if F::ROWS == 0 || F::ROWS > 256 || F::COLS == 0 || F::COLS > 256 {
        return Err(EncodeError::InvalidFormat);
    }
    let row_bytes = F::COLS
        .checked_mul(size_of::<T>())
        .ok_or(EncodeError::InvalidFormat)?;
    let swizzle = match F::SWIZZLE_BYTES {
        0 if row_bytes % 16 == 0 => driver::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE,
        64 if row_bytes == 64 => driver::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_64B,
        128 if row_bytes == 128 => driver::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
        _ => return Err(EncodeError::InvalidFormat),
    };
    Ok(EncodeParams {
        data_type: T::DRIVER_TYPE,
        dimensions: [cols, rows],
        strides: [stride],
        box_dimensions: [F::COLS as u32, F::ROWS as u32],
        element_strides: [1, 1],
        swizzle,
        l2: match options.l2_promotion {
            L2Promotion::None => {
                driver::CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE
            }
            L2Promotion::B128 => {
                driver::CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_L2_128B
            }
        },
    })
}

// Only a successful encoder may initialize and publish these opaque words.
/// Publish opaque descriptor bytes only after the supplied driver encoder succeeds.
fn encode_with<T: TmaElement, F: PhysicalFormat<Element = T>>(
    encode: impl FnOnce(*mut driver::CUtensorMap) -> driver::CUresult,
) -> Result<TensorMap<T, F>, EncodeError> {
    const {
        assert!(size_of::<driver::CUtensorMap>() == 128);
        assert!(align_of::<driver::CUtensorMap>() >= 128);
    }
    let mut output = MaybeUninit::<driver::CUtensorMap>::uninit();
    let result = encode(output.as_mut_ptr());
    if result != driver::cudaError_enum_CUDA_SUCCESS {
        return Err(EncodeError::Driver(result));
    }
    let output = unsafe { output.assume_init() };
    Ok(unsafe { TensorMap::from_encoded(TmaDescriptor::from_raw_words(output.opaque)) })
}

/// Encode a row-major global tensor for a reviewed physical shared format.
///
/// Coordinates use the inner (column) dimension first. Encoding does not
/// retain the allocation: it must cover the specified extents and remain live
/// until every asynchronous device operation using the returned map completes.
// The driver encodes this device address; it does not access the tensor data.
// Actual tensor-map accesses are unsafe device operations with lifetime contracts.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn encode_2d<T: TmaElement, F: PhysicalFormat<Element = T>>(
    global_base: *mut c_void,
    rows: u64,
    cols: u64,
    ld_elements: u64,
    options: EncodeOptions,
) -> Result<TensorMap<T, F>, EncodeError> {
    let p = validate_2d::<T, F>(global_base as usize, rows, cols, ld_elements, options)?;
    encode_with(|output| unsafe {
        driver::cuTensorMapEncodeTiled(
            output,
            p.data_type,
            2,
            global_base,
            p.dimensions.as_ptr(),
            p.strides.as_ptr(),
            p.box_dimensions.as_ptr(),
            p.element_strides.as_ptr(),
            driver::CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            p.swizzle,
            p.l2,
            driver::CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    })
}

/// Regression tests for this module’s layout, representation and rejection contracts.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{F16Tile128, F16TileB80, F16TileB96, F16TileC};

    /// Host encoding must keep packed values, scale words and output elements in their correct driver formats.
    #[test]
    fn blockscaled_tma_arguments_preserve_packed_bytes_scales_and_output() {
        use crate::memory::{Mxf4ScaleTile, Mxf4Tile128};

        let packed =
            validate_2d::<u8, Mxf4Tile128>(0x1000, 256, 128, 128, EncodeOptions::default())
                .unwrap();
        assert_eq!(
            packed.data_type,
            driver::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_UINT8
        );
        assert_eq!(packed.dimensions, [128, 256]);
        assert_eq!(packed.strides, [128]);
        assert_eq!(packed.box_dimensions, [64, 128]);
        assert_eq!(packed.element_strides, [1, 1]);
        assert_eq!(
            packed.swizzle,
            driver::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_64B
        );

        let scales =
            validate_2d::<u16, Mxf4ScaleTile>(0x1000, 2, 512, 512, EncodeOptions::default())
                .unwrap();
        assert_eq!(
            scales.data_type,
            driver::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_UINT16
        );
        assert_eq!(scales.dimensions, [512, 2]);
        assert_eq!(scales.strides, [1024]);
        assert_eq!(scales.box_dimensions, [256, 1]);
        assert_eq!(scales.element_strides, [1, 1]);
        assert_eq!(
            scales.swizzle,
            driver::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_NONE
        );

        let output =
            validate_2d::<f16, F16Tile128>(0x1000, 256, 256, 256, EncodeOptions::default())
                .unwrap();
        assert_eq!(
            output.data_type,
            driver::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16
        );
        assert_eq!(output.dimensions, [256, 256]);
        assert_eq!(output.strides, [512]);
        assert_eq!(output.box_dimensions, [64, 128]);
        assert_eq!(
            output.swizzle,
            driver::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B
        );
    }

    /// Row pitches must be checked after converting packed storage elements to bytes.
    #[test]
    fn blockscaled_tma_strides_are_validated_in_bytes() {
        use crate::memory::{Mxf4ScaleTile, Mxf4Tile128};

        for pitch in [63, 65, 1u64 << 40, u64::MAX] {
            assert!(matches!(
                validate_2d::<u8, Mxf4Tile128>(0x1000, 128, 64, pitch, EncodeOptions::default()),
                Err(EncodeError::InvalidStride)
            ));
        }
        for pitch in [255, 257, 1u64 << 39, u64::MAX] {
            assert!(matches!(
                validate_2d::<u16, Mxf4ScaleTile>(0x1000, 1, 256, pitch, EncodeOptions::default()),
                Err(EncodeError::InvalidStride)
            ));
        }
    }

    /// A failing driver callback must not publish its uninitialized descriptor output.
    #[test]
    fn tma_driver_failure_does_not_adopt_uninitialized_output() {
        let result = encode_with::<f16, F16Tile128>(|output| {
            assert_eq!((output as usize) % 128, 0);
            driver::cudaError_enum_CUDA_ERROR_INVALID_VALUE
        });
        assert!(matches!(
            result,
            Err(EncodeError::Driver(
                driver::cudaError_enum_CUDA_ERROR_INVALID_VALUE
            ))
        ));
    }

    /// Validated encoder parameters must preserve tile extents, swizzle spans and cache options.
    #[test]
    fn tma_arguments_preserve_physical_shapes_and_options() {
        let p =
            validate_2d::<f16, F16Tile128>(0x1000, 256, 64, 64, EncodeOptions::default()).unwrap();
        assert_eq!(p.dimensions, [64, 256]);
        assert_eq!(p.strides, [128]);
        assert_eq!(p.box_dimensions, [64, 128]);
        assert_eq!(p.element_strides, [1, 1]);
        assert_eq!(
            p.swizzle,
            driver::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B
        );
        let b = validate_2d::<f16, F16TileB96>(
            0x1000,
            192,
            64,
            64,
            EncodeOptions {
                l2_promotion: L2Promotion::B128,
            },
        )
        .unwrap();
        assert_eq!(b.box_dimensions, [64, 96]);
        assert_eq!(
            b.l2,
            driver::CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        );
        assert_eq!(
            validate_2d::<f16, F16TileB80>(0x1000, 160, 64, 64, EncodeOptions::default())
                .unwrap()
                .box_dimensions,
            [64, 80]
        );
        let c =
            validate_2d::<f16, F16TileC>(0x1000, 128, 352, 352, EncodeOptions::default()).unwrap();
        assert_eq!(c.box_dimensions, [32, 128]);
        assert_eq!(
            c.swizzle,
            driver::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_64B
        );
    }

    /// Invalid bases, extents and pitches must fail validation before invoking the CUDA driver.
    #[test]
    fn tma_rejects_invalid_dynamic_layout_before_driver_call() {
        for (base, rows, cols, pitch) in [
            (0, 256, 64, 64),
            (0x1001, 256, 64, 64),
            (0x1000, 0, 64, 64),
            (0x1000, 256, 0, 64),
            (0x1000, (1 << 32) + 1, 64, 64),
            (0x1000, 256, (1 << 32) + 1, 1 << 33),
            (0x1000, 256, 64, 63),
            (0x1000, 256, 64, 65),
            (0x1000, 256, 64, 1 << 39),
            (0x1000, 256, 64, u64::MAX),
        ] {
            assert!(
                validate_2d::<f16, F16Tile128>(base, rows, cols, pitch, EncodeOptions::default())
                    .is_err()
            );
        }
        assert!(
            validate_2d::<f16, F16Tile128>(0x1000, 1 << 32, 64, 64, EncodeOptions::default())
                .is_ok()
        );
    }
}
