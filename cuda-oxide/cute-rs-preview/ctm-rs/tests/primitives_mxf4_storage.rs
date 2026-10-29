/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#![doc = "Checks MXFP4 physical storage validation and typed local TMA signatures."]
use ctm_rs::{
    memory::{
        F16Tile128, MemoryError, Mxf4ScaleTile, Mxf4Tile128, PhysicalFormat, SharedAddress,
        SharedRegion,
    },
    tma::{Coordinates2D, TensorMap, cp_async_bulk_tensor_shared_cta_global},
};
use cuda_device::barrier::Barrier;

/// MXFP4 value and scale regions must report exact instruction footprints and element types.
#[test]
fn blockscaled_stage_storage_matches_instruction_footprints() {
    assert_eq!(Mxf4Tile128::FOOTPRINT_BYTES, 8192);
    assert_eq!(Mxf4Tile128::BASE_ALIGNMENT, 512);
    assert_eq!(Mxf4Tile128::SWIZZLE_BYTES, 64);
    assert_eq!(Mxf4ScaleTile::FOOTPRINT_BYTES, 512);
    assert_eq!(Mxf4ScaleTile::BASE_ALIGNMENT, 128);
    assert_eq!(Mxf4ScaleTile::SWIZZLE_BYTES, 0);
    assert_eq!(F16Tile128::FOOTPRINT_BYTES, 16384);
    assert_eq!(size_of::<TensorMap<u8, Mxf4Tile128>>(), 128);
    assert_eq!(size_of::<TensorMap<u16, Mxf4ScaleTile>>(), 128);
    assert_eq!(align_of::<TensorMap<u8, Mxf4Tile128>>(), 64);
    assert_eq!(align_of::<TensorMap<u16, Mxf4ScaleTile>>(), 64);
}

/// Region construction must reject insufficient capacity and misaligned shared bases.
#[test]
fn blockscaled_storage_rejects_short_or_misaligned_stages() {
    unsafe {
        assert!(
            SharedRegion::<Mxf4ScaleTile>::from_raw_parts(
                core::ptr::without_provenance_mut(0x1080),
                512
            )
            .is_ok()
        );
        assert!(matches!(
            SharedRegion::<Mxf4Tile128>::from_raw_parts(
                core::ptr::without_provenance_mut(0x1000),
                8191
            ),
            Err(MemoryError::InsufficientCapacity)
        ));
        assert!(matches!(
            SharedRegion::<Mxf4Tile128>::from_raw_parts(
                core::ptr::without_provenance_mut(0x1010),
                8192
            ),
            Err(MemoryError::Misaligned)
        ));
        assert!(matches!(
            SharedRegion::<Mxf4ScaleTile>::from_raw_parts(
                core::ptr::without_provenance_mut(0x1080),
                511
            ),
            Err(MemoryError::InsufficientCapacity)
        ));
        assert!(matches!(
            SharedRegion::<Mxf4ScaleTile>::from_raw_parts(
                core::ptr::without_provenance_mut(0x1010),
                512
            ),
            Err(MemoryError::Misaligned)
        ));
    }
}

/// Both value and scale load signatures must use a local shared completion barrier.
#[test]
// The explicit instruction signatures are the contract checked by this test.
#[allow(clippy::type_complexity)]
fn ordinary_tma_load_uses_cta_local_completion_for_both_stage_types() {
    let _: unsafe fn(
        &SharedRegion<'_, Mxf4Tile128>,
        &TensorMap<u8, Mxf4Tile128>,
        Coordinates2D,
        SharedAddress<'_, Barrier>,
    ) = cp_async_bulk_tensor_shared_cta_global::<u8, Mxf4Tile128>;
    let _: unsafe fn(
        &SharedRegion<'_, Mxf4ScaleTile>,
        &TensorMap<u16, Mxf4ScaleTile>,
        Coordinates2D,
        SharedAddress<'_, Barrier>,
    ) = cp_async_bulk_tensor_shared_cta_global::<u16, Mxf4ScaleTile>;
}

/// Successful trapping constructors must retain the same validated addresses as checked constructors.
#[test]
fn trapping_regions_preserve_checked_constructor_validation_and_address() {
    /// Aligned host buffer used to test numeric region validation without GPU accesses.
    #[repr(align(1024))]
    struct Storage(
        #[doc = "Backing bytes with enough room for every tested physical format."] [u8; 32768],
    );

    /// Compare checked and trapping region constructors for one supported physical format.
    fn check<F: PhysicalFormat>()
    where
        F::Element: std::panic::RefUnwindSafe,
    {
        let mut storage = Storage([0; 32768]);
        let base = storage.0.as_mut_ptr().cast::<F::Element>();
        // Host-side representation check only: neither constructor dereferences
        // the pointer. Device execution must supply actual shared storage.
        for offset in [0, F::BASE_ALIGNMENT] {
            let pointer = base.cast::<u8>().wrapping_add(offset).cast();
            for capacity in [F::FOOTPRINT_BYTES, F::FOOTPRINT_BYTES + F::BASE_ALIGNMENT] {
                let region =
                    unsafe { SharedRegion::<F>::from_raw_parts_or_trap(pointer, capacity) };
                assert_eq!(region.address().as_raw(), pointer);
            }
        }

        for (pointer, bytes, error) in [
            (
                core::ptr::null_mut(),
                F::FOOTPRINT_BYTES,
                MemoryError::NullAddress,
            ),
            (
                base.cast::<u8>().wrapping_add(1).cast(),
                F::FOOTPRINT_BYTES,
                MemoryError::Misaligned,
            ),
            (
                base,
                F::FOOTPRINT_BYTES - 1,
                MemoryError::InsufficientCapacity,
            ),
        ] {
            assert!(matches!(
                unsafe { SharedRegion::<F>::from_raw_parts(pointer, bytes) },
                Err(actual) if actual == error
            ));
            let failure = std::panic::catch_unwind(|| unsafe {
                SharedRegion::<F>::from_raw_parts_or_trap(pointer, bytes);
            })
            .expect_err("invalid storage must reach the device trap");
            let message = failure
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| failure.downcast_ref::<String>().map(String::as_str));
            assert!(
                message
                    .unwrap_or_default()
                    .contains("trap called outside CUDA kernel context")
            );
        }
    }
    check::<Mxf4Tile128>();
    check::<Mxf4ScaleTile>();
    check::<F16Tile128>();
}
