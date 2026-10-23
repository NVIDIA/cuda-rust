/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `StreamPoolRoundRobin` refuses an empty pool at construction instead of
//! dividing by zero on the first `next_stream`.

use cuda_async::error::DeviceError;
use cuda_async::scheduling_policies::{SchedulingPolicy, StreamPoolRoundRobin};
use cuda_core::Device;

#[test]
fn round_robin_rejects_zero_streams() {
    let device = Device::new(0).unwrap();
    let err = StreamPoolRoundRobin::new(&device, 0)
        .err()
        .expect("new(0) must fail");
    assert!(matches!(err, DeviceError::Scheduling(_)), "{err}");
}

#[test]
fn round_robin_serves_a_one_stream_pool() {
    let device = Device::new(0).unwrap();
    let policy = StreamPoolRoundRobin::new(&device, 1).unwrap();
    policy.next_stream().unwrap();
    policy.next_stream().unwrap();
}
