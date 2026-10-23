// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_macros::kernel;

unsafe extern "C" {
    // Deliberately undefined on the host. Looking up a specialization's name
    // must not retain the kernel body or any of its device-only callees.
    fn device_only_retention_probe();
}

#[kernel]
pub fn guarded<const ITEMS: usize>() {
    // SAFETY: this body is a device-only link probe and is never called.
    unsafe { device_only_retention_probe() };
}

#[kernel]
pub fn guarded_closure<F: Fn() -> i32 + Copy>(f: F) {
    let _ = f();
    // SAFETY: this body is a device-only link probe and is never called.
    unsafe { device_only_retention_probe() };
}

#[test]
fn names_do_not_require_host_kernel_bodies() {
    let first = guarded_ptx_name::<1>();
    let fourth = guarded_ptx_name::<4>();
    assert!(first.starts_with("guarded_TID_"));
    assert_ne!(first, fourth);
    assert_eq!(first, guarded_ptx_name::<1>());
    assert_eq!(
        first,
        <__guarded_CudaKernel<1> as cuda_host::GenericCudaKernel>::ptx_name()
    );
}

#[test]
fn naming_accepts_a_borrowed_closure() {
    fn name<F: Fn() -> i32 + Copy>(_: &F) -> &'static str {
        guarded_closure_ptx_name::<F>()
    }

    let value = 7;
    let borrowed = || value;
    assert!(name(&borrowed).starts_with("guarded_closure_TID_"));
}
