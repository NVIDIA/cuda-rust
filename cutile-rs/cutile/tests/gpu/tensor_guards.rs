/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Guards on tensor metadata that safe code reaches through `api`: a tensor's
//! shape must never describe more memory than its storage holds, and a
//! mutable partition must never alias live shared storage.

use std::panic::{self, AssertUnwindSafe};

use cutile::api;
use cutile::prelude::*;

/// Mirrors `shared_storage_blocks_mutable_partition` (tensor_views.rs) for the
/// borrowed path: `PartitionMut for &mut Tensor` skipped the unique-storage
/// check the owned path has, so a `&mut` partition could alias a live
/// `Arc<Tensor>` input (device data race, reproduced).
#[test]
fn shared_storage_blocks_borrowed_mutable_partition() {
    let base = Arc::new(api::arange::<f32>(8).sync().expect("Failed."));
    let _view = base.reshape(&[2, 4]).unwrap();
    let mut owned = Arc::try_unwrap(base).expect("Expected unique outer Arc.");
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = (&mut owned).partition([8]);
    }));
    assert!(
        result.is_err(),
        "Expected the borrowed mutable partition to be rejected"
    );
}

/// `try_partition` documents `Err` for storage shared with other tensors or
/// views; it used to panic through `partition` instead.
#[test]
fn try_partition_rejects_shared_storage_with_err() {
    let base = Arc::new(api::arange::<f32>(8).sync().expect("Failed."));
    let _view = base.reshape(&[2, 4]).unwrap();
    let err = base
        .try_partition([8])
        .err()
        .expect("shared storage must be an Err");
    assert!(format!("{err}").contains("shared"), "{err}");
}

/// `eye_rect` multiplied `rows * cols` unchecked (a wrapped product sizes the
/// allocation below the shape) and forwarded a zero count to a panicking
/// allocator; both now surface as `Err` on the execution path.
#[test]
fn eye_rect_reports_invalid_shapes_as_errors() {
    assert!(api::eye_rect(usize::MAX, 2).sync().is_err());
    assert!(api::eye_rect(0, 4).sync().is_err());
    let eye = api::eye_rect(3, 5).sync().expect("valid eye_rect");
    assert_eq!(eye.shape(), &[3, 5]);
    let host: Vec<f32> = eye.to_host_vec().sync().expect("copy");
    for r in 0..3 {
        for c in 0..5 {
            let expected = if r == c { 1.0 } else { 0.0 };
            assert_eq!(host[r * 5 + c], expected, "eye_rect[{r}][{c}]");
        }
    }
}

#[test]
fn reshape_op_rejects_shape_that_exceeds_storage() {
    // 16 f32 of storage. The former `ReshapeOp` called `reshape_unchecked`, so
    // this returned a [32] tensor whose every later launch read 64 bytes past
    // the allocation (reproduced under compute-sanitizer).
    let result = api::ones::<f32>(&[16]).reshape(&[32]).sync();
    let err = result.expect_err("reshape to a larger element count must fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("preserve tensor size"),
        "error must name the size mismatch, got: {msg}"
    );

    // A same-size reshape still goes through.
    let t = api::ones::<f32>(&[16])
        .reshape(&[4, 4])
        .sync()
        .expect("same-size reshape");
    assert_eq!(t.shape(), &[4, 4]);
}

/// A failed device allocation is an error the caller can act on (free and
/// retry, back off, fall back to a smaller request), not a panic:
/// `ExecutionContext::alloc_async` returns `Result`, and an out-of-memory
/// failure is not sticky, so the context stays usable afterwards.
#[test]
fn oversized_allocation_is_an_error_not_a_panic() {
    // 2^44 f32 elements = 64 TiB, beyond any device.
    let err = Tensor::<f32>::uninitialized(1usize << 44)
        .sync()
        .expect_err("oversized allocation must fail");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("Driver"),
        "expected the driver's allocation error, got: {msg}"
    );

    // The context is still usable after the failed allocation.
    let t = api::ones::<f32>(&[16])
        .sync()
        .expect("allocation after OOM");
    assert_eq!(t.shape(), &[16]);
}
