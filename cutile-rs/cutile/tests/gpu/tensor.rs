/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU-dependent tensor invariant tests.

use cutile::api;
use cutile::tensor::Tensor;
use cutile::tile_kernel::DeviceOp;
use std::mem::forget;
use std::sync::Arc;

#[test]
fn reinterpret_rejects_misaligned_storage() {
    let base = Arc::new(api::zeros::<u8>(&[8]).sync().expect("Failed."));
    // Shift the pointer by one byte so the storage is no longer aligned for u32.
    let misaligned = Arc::new(unsafe {
        Tensor::<u8>::from_raw_parts(
            base.device_pointer().cu_deviceptr() + 1,
            4,
            base.device_id(),
            vec![4],
            vec![1],
        )
    });

    assert!(misaligned.reinterpret::<u32>(&[1]).is_err());

    // The misaligned tensor is a borrowed view onto `base`'s allocation and must not free it.
    forget(misaligned);
}

#[test]
#[should_panic(expected = "Tensor logical byte size must match storage byte size.")]
fn from_raw_parts_rejects_shape_storage_mismatch() {
    let base = Arc::new(api::zeros::<u8>(&[4]).sync().expect("Failed."));

    // Four bytes of storage cannot describe a Tensor<u32> with shape [2], which would
    // logically require eight bytes.
    let _ = unsafe {
        Tensor::<u32>::from_raw_parts(
            base.device_pointer().cu_deviceptr(),
            4,
            base.device_id(),
            vec![2],
            vec![1],
        )
    };
}
