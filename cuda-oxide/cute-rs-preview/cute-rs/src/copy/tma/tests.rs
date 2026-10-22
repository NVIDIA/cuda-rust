/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![doc = "Checks TMA descriptor ABI, swizzle modes, element units and host encoding options."]
/// TMA descriptors must retain their opaque ABI and Copy behavior even with non-Copy markers.
#[test]
fn descriptor_value_abi_does_not_require_copy_marker_types() {
    // Layout and element markers need not be Copy: only descriptor bytes move.
    /// Non-Copy marker proving descriptor Copy does not require marker trait bounds.
    struct Marker;
    /// Require Copy at compile time without constructing or reading a descriptor.
    fn assert_copy<T: Copy>() {}
    assert_copy::<super::TensorMap<Marker, Marker>>();
    assert_eq!(
        core::mem::size_of::<super::TensorMap<Marker, Marker>>(),
        128
    );
    assert_eq!(
        core::mem::align_of::<super::TensorMap<Marker, Marker>>(),
        64
    );
}

use super::*;

/// Swizzle classification must accept exact hardware modes and reject nearby incompatible patterns.
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

/// Element-unit swizzles must convert to the exact byte-unit TMA encoding.
#[test]
fn element_recast_reaches_the_byte_modes() {
    use crate::markers::{ComposedLayout, ReifySmem2D, RowMajor, Swizzle};
    // For f16, element pattern S<3,3,3> becomes byte pattern S<3,4,3>,
    // which is TMA B128.
    /// FP16 element-unit layout that must recast to TMA’s B128 byte pattern.
    type Smem = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<8, 64>>;
    assert_eq!(Smem::NUM_BITS, 3);
    assert_eq!(tma_swizzle_mode_for::<Smem>(2), Some(TmaSwizzleMode::B128));
    // For f32, it becomes S<3,5,3>, which TMA cannot encode.
    assert_eq!(tma_swizzle_mode_for::<Smem>(4), None);
    // Invalid element sizes fail instead of being rounded.
    assert_eq!(tma_swizzle_mode_for::<Smem>(0), None);
    assert_eq!(tma_swizzle_mode_for::<Smem>(3), None);
}

/// Named TMA modes must report their exact supported row-byte spans.
#[test]
fn swizzle_modes_name_their_exact_row_spans() {
    assert_eq!(tma_swizzle_span_bytes(TmaSwizzleMode::None), None);
    assert_eq!(tma_swizzle_span_bytes(TmaSwizzleMode::B32), Some(32));
    assert_eq!(tma_swizzle_span_bytes(TmaSwizzleMode::B64), Some(64));
    assert_eq!(tma_swizzle_span_bytes(TmaSwizzleMode::B128), Some(128));
}

/// The packed MXFP4 value tile must encode an 8192-byte footprint using B64 swizzling.
#[test]
fn flattened_mxfp4_ab_tile_is_8192_bytes_with_b64_swizzle() {
    use crate::markers::{ComposedLayout, RowMajor, Swizzle};

    // Two FP4 values share each byte: 128 K values become 64 bytes.
    // Byte pattern S<2,4,3> is TMA B64.
    /// Packed-byte value tile using the exact B64 TMA swizzle.
    type ByteSmem = ComposedLayout<Swizzle<2, 4, 3>, 0, RowMajor<128, 64>>;
    assert_eq!(size_in_bytes::<u8, ByteSmem>(), 8_192);
    assert_eq!(
        tma_swizzle_mode_for::<ByteSmem>(1),
        Some(TmaSwizzleMode::B64)
    );
}

/// Canonical scale transport must encode 512 bytes with no shared swizzle.
#[test]
fn canonical_mxfp4_scale_tile_is_512_bytes_without_swizzle() {
    use crate::markers::RowMajor;

    // Standard SM120 packing: 128 rows x 4 UE8M0 bytes = 512 bytes.
    // Viewing those bytes as u16 creates one legal 1x256 TMA tile.
    /// One unswizzled row of 256 u16 scale carriers.
    type ScaleSmem = RowMajor<1, 256>;
    assert_eq!(size_in_bytes::<u16, ScaleSmem>(), 512);
    assert_eq!(
        tma_swizzle_mode_for::<ScaleSmem>(2),
        Some(TmaSwizzleMode::None)
    );
}

/// `size_in_bytes` follows CuTeDSL: the cosize of the affine layout times the element size.
#[test]
fn size_in_bytes_is_cosize_times_element_size() {
    use crate::markers::{C, ComposedLayout, Layout, RowMajor, Swizzle};

    // A padded row stride enlarges the cosize beyond rows x columns.
    /// Eight 64-element rows with a 72-element pitch.
    type Padded = Layout<(C<8>, C<64>), (C<72>, C<1>)>;
    assert_eq!(size_in_bytes::<u16, Padded>(), (7 * 72 + 63 + 1) * 2);
    // A swizzle permutes addresses inside the span and adds no bytes.
    /// B128-swizzled 8x64 FP16 tile.
    type Swizzled = ComposedLayout<Swizzle<3, 3, 3>, 0, RowMajor<8, 64>>;
    assert_eq!(size_in_bytes::<u16, Swizzled>(), 8 * 64 * 2);
}

/// Default descriptor options must retain no promotion while exposing the CuTeDSL L2 policy.
#[test]
fn descriptor_options_preserve_default_and_offer_cutedsl_policy() {
    assert_eq!(
        TmaEncodeOptions::default().l2_promotion,
        TmaL2Promotion::None
    );
    assert_eq!(TmaEncodeOptions::L2_128B.l2_promotion, TmaL2Promotion::B128);
}

/// The packed-byte host carrier must use the driver’s UINT8 tensor-map element code.
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
