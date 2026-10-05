/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Module-level atomic globals: scalar load/store and concurrent RMW.
//!
//! A load followed by a store is not an atomic increment; the separate RMW
//! regression below checks concurrent-counter semantics.

use cutile::prelude::*;

use crate::common;

#[cutile::module]
mod global_kernels {
    use cutile::core::*;

    static COUNTER: Global<AtomicI32, { [] }> = Global::new(0i32);

    #[cutile::entry()]
    fn update_counter_ordered(out: &mut Tensor<i32, { [1] }>) {
        let (old_value, _load_token) = COUNTER.load(ordering::Acquire, scope::Device);
        let next_value = old_value + constant(1i32, shape![]);
        let _store_token = COUNTER.store(next_value, ordering::Release, scope::Device);
        out.store(old_value.reshape(shape![1]));
    }
}

#[cutile::module]
mod atomic_counter_kernels {
    use cutile::core::*;

    static COUNTER: Global<AtomicI32, { [] }> = Global::new(0i32);

    #[cutile::entry()]
    fn increment(out: &mut Tensor<i32, { [1] }>) {
        let one: Tile<i32, { [] }> = constant(1i32, shape![]);
        let (old, _) = COUNTER.atomic_add(one, ordering::Relaxed, scope::Device);
        out.store(old.reshape(shape![1]));
    }
}

#[test]
fn atomic_global_increments_are_unique_across_tile_programs() {
    common::with_test_stack(|| {
        const N: usize = 1024;
        let device = cuda_core::Device::new(0).expect("device");
        let stream = device.new_stream().expect("stream");
        let mut out = api::zeros::<i32>(&[N]).sync_on(&stream).expect("zeros");
        // Same specialization twice: test uniqueness under contention and
        // persistence across launches, without relying on program scheduling.
        for start in [0, N as i32] {
            atomic_counter_kernels::increment((&mut out).partition([1]))
                .sync_on(&stream)
                .expect("atomic increment");
            let mut actual = out.dup().to_host_vec().sync_on(&stream).expect("readback");
            actual.sort_unstable();
            assert_eq!(actual, (start..start + N as i32).collect::<Vec<_>>());
        }
    });
}

use global_kernels::update_counter_ordered;

#[test]
fn smoke_global_counter_ordered() {
    common::with_test_stack(|| {
        let device = cuda_core::Device::new(0).expect("device");
        let stream = device.new_stream().expect("stream");

        let mut first = api::zeros::<i32>(&[1]).sync_on(&stream).expect("zeros");
        update_counter_ordered((&mut first).partition([1]))
            .grid((1, 1, 1))
            .sync_on(&stream)
            .expect("first launch");
        let first_host: Vec<i32> = first
            .dup()
            .to_host_vec()
            .sync_on(&stream)
            .expect("first to_host");
        assert_eq!(first_host, vec![0]);

        let mut second = api::zeros::<i32>(&[1]).sync_on(&stream).expect("zeros");
        update_counter_ordered((&mut second).partition([1]))
            .grid((1, 1, 1))
            .sync_on(&stream)
            .expect("second launch");
        let second_host: Vec<i32> = second
            .dup()
            .to_host_vec()
            .sync_on(&stream)
            .expect("second to_host");
        assert_eq!(second_host, vec![1]);
    });
}
