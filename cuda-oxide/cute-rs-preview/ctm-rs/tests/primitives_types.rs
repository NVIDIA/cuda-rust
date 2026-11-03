/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#![doc = "Checks instruction-facing storage formats, opaque descriptor ABI and typed address domains."]
#![feature(f16)]
use ctm_rs::{memory::*, tma::TensorMap};

/// Physical formats and opaque tensor maps must preserve expected footprints, alignment and Copy behavior.
#[test]
fn physical_formats_and_tensor_map_preserve_kernel_abi() {
    assert_eq!(size_of::<TensorMap<f16, F16Tile128>>(), 128);
    assert_eq!(align_of::<TensorMap<f16, F16Tile128>>(), 64);
    assert_eq!(F16Tile128::FOOTPRINT_BYTES, 16384);
    assert_eq!(F16TileB96::FOOTPRINT_BYTES, 12288);
    assert_eq!(F16TileB80::FOOTPRINT_BYTES, 10240);
    assert_eq!(F16TileC::FOOTPRINT_BYTES, 8192);
    assert_eq!(F16Tile128::BASE_ALIGNMENT, 1024);
    assert_eq!(F16TileC::BASE_ALIGNMENT, 512);
    /// Require the tensor-map type to implement Copy without adding marker bounds.
    fn copy<T: Copy>() {}
    copy::<TensorMap<f16, F16Tile128>>();
}

/// Shared-region checks must reject null, short or misaligned storage before exposing an address.
#[test]
fn shared_region_rejects_invalid_storage_before_use() {
    // These invalid layouts return Err before any address can be used.
    assert!(
        unsafe { SharedRegion::<F16Tile128>::from_raw_parts(core::ptr::null_mut(), 16384) }
            .is_err()
    );
    assert!(
        unsafe {
            SharedRegion::<F16Tile128>::from_raw_parts(
                core::ptr::without_provenance_mut(0x1002),
                16384,
            )
        }
        .is_err()
    );
    assert!(
        unsafe {
            SharedRegion::<F16Tile128>::from_raw_parts(
                core::ptr::without_provenance_mut(0x1000),
                16383,
            )
        }
        .is_err()
    );
}

/// Masks must remain nonempty and signed tensor coordinates must retain negative values.
#[test]
fn multicast_masks_and_signed_coordinates_preserve_instruction_domains() {
    use ctm_rs::{cluster::CtaMask, groups::CtaGroup2, tma::Coordinates2D};
    assert!(CtaMask::<CtaGroup2>::try_from_bits(0).is_err());
    assert_eq!(
        CtaMask::<CtaGroup2>::try_from_bits(0b1111).unwrap().bits(),
        0b1111
    );
    assert_eq!(Coordinates2D::new(-32, 4096).inner(), -32);
    assert_eq!(Coordinates2D::new(-32, 4096).outer(), 4096);
}

/// The relaxed remote-arrival signature must retain the cluster-shared address domain.
#[test]
fn relaxed_remote_arrival_requires_a_cluster_barrier_address() {
    use ctm_rs::cluster::arrive_remote_relaxed;
    use cuda_device::barrier::Barrier;
    let _: for<'a> unsafe fn(ClusterSharedAddress<'a, Barrier>) = arrive_remote_relaxed;
}

/// BF16 operands retain packed bits and distinct roles without changing the FP32 result carrier.
#[test]
fn bf16_mma_carriers_preserve_bits_roles_and_fp32_accumulation() {
    use ctm_rs::mma::{Accumulator16x8, Bf16A, Bf16B, mma_bf16};
    let a = [0x8000_0000, 0x7fc1_3f80, 0xffff_7f80, 0x0001_ff80];
    let b = [0xbf80_3f80, 0x7f81_0080];
    assert_eq!(Bf16A::from_registers(a).registers(), a);
    assert_eq!(Bf16B::from_registers(b).registers(), b);
    assert_eq!(size_of::<Bf16A>(), size_of::<[u32; 4]>());
    assert_eq!(align_of::<Bf16A>(), align_of::<[u32; 4]>());
    assert_eq!(size_of::<Bf16B>(), size_of::<[u32; 2]>());
    assert_eq!(align_of::<Bf16B>(), align_of::<[u32; 2]>());
    let _: unsafe fn(Accumulator16x8, Bf16A, Bf16B) -> Accumulator16x8 = mma_bf16;
}
