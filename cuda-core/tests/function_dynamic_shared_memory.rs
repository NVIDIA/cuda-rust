/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `Function::set_max_dynamic_shared_memory_bytes` on a function loaded from
//! PTX, the untyped case that cannot rely on typed launch preparation: the
//! value round-trips through `max_dynamic_shared_memory_bytes`, and a request
//! past the device's opt-in limit is refused by the driver.

use cuda_core::Context;

const NOOP_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry noop()
{
    ret;
}
";

const INVALID_VALUE: cuda_core::sys::CUresult =
    cuda_core::sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE;

fn has_gpu() -> bool {
    Context::device_count().map(|n| n > 0).unwrap_or(false)
}

#[test]
fn dynamic_shared_memory_opt_in_round_trips_and_respects_the_device_limit() {
    if !has_gpu() {
        return;
    }
    let ctx = Context::new(0).expect("failed to create CUDA context");
    let module = ctx
        .load_module_from_ptx_src(NOOP_PTX)
        .expect("failed to load the PTX module");
    let function = module.load_function("noop").expect("missing `noop`");

    let opt_in = ctx
        .max_opt_in_shared_memory_per_block()
        .expect("opt-in shared-memory limit");
    let static_bytes = function
        .static_shared_memory_bytes()
        .expect("static shared memory");
    let allowed = opt_in - static_bytes;

    // The whole opt-in budget is accepted and observable.
    function
        .set_max_dynamic_shared_memory_bytes(allowed)
        .expect("setting the opt-in maximum must succeed");
    assert_eq!(function.max_dynamic_shared_memory_bytes().unwrap(), allowed);

    // Lowering it again is also a plain round trip.
    function.set_max_dynamic_shared_memory_bytes(0).unwrap();
    assert_eq!(function.max_dynamic_shared_memory_bytes().unwrap(), 0);

    // One byte past the budget is the driver's to refuse.
    let err = function
        .set_max_dynamic_shared_memory_bytes(allowed + 1)
        .expect_err("a request past the opt-in limit must fail");
    assert_eq!(err.0, INVALID_VALUE);

    // Values that do not fit the driver's `int` are rejected before the call.
    let err = function
        .set_max_dynamic_shared_memory_bytes(u32::MAX)
        .expect_err("u32::MAX does not fit in an i32");
    assert_eq!(err.0, INVALID_VALUE);
}
