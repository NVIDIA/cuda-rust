/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `CudaContext::{set_sync_policy, sync_policy}` round-trip.
//!
//! Its own test binary (not merged into `device_buffer.rs`'s): the context's
//! scheduling flags are shared, process-wide state on the primary context
//! (`cuDevicePrimaryCtxSetFlags_v2`), so setting them from a test that runs
//! concurrently with others in the same binary would race. A single test
//! function exercising every policy in sequence keeps the whole check inside
//! one thread.

use cuda_core::{CudaContext, SyncPolicy};

#[test]
fn sync_policy_round_trips_through_every_named_value() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");

    for policy in [
        SyncPolicy::BlockingSync,
        SyncPolicy::Spin,
        SyncPolicy::Yield,
        SyncPolicy::Auto,
        // Land on BlockingSync last: it is the one #705 asks for, and this
        // proves the round trip through every other value first didn't
        // leave some other bit set that corrupts it.
        SyncPolicy::BlockingSync,
    ] {
        ctx.set_sync_policy(policy)
            .unwrap_or_else(|e| panic!("set_sync_policy({policy:?}) failed: {e:?}"));
        let got = ctx
            .sync_policy()
            .unwrap_or_else(|e| panic!("sync_policy() after set({policy:?}) failed: {e:?}"));
        assert_eq!(
            got,
            Some(policy),
            "sync_policy() must read back exactly what set_sync_policy({policy:?}) wrote"
        );
    }
}
