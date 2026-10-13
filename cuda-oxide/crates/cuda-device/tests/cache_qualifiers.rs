/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![feature(f16)]

use cuda_device::cache::{
    CacheLoad, CacheStore, load_cached, load_l2_only, load_last_use, load_streaming, load_volatile,
    store_l2_only, store_streaming, store_write_back, store_write_through,
};
use cuda_device::cusimd::{Double2, Float2, Float4, Int2, Int4, Uint2, Uint4};
use cuda_device::vector::{F32x2, F32x4, F64x2, U32x2, U32x4};

fn assert_cache_load_store<T: CacheLoad + CacheStore>() {}

#[test]
fn test_cache_qualifier_trait_implementations() {
    // Integer scalars
    assert_cache_load_store::<u8>();
    assert_cache_load_store::<i8>();
    assert_cache_load_store::<u16>();
    assert_cache_load_store::<i16>();
    assert_cache_load_store::<u32>();
    assert_cache_load_store::<i32>();
    assert_cache_load_store::<u64>();
    assert_cache_load_store::<i64>();
    assert_cache_load_store::<usize>();
    assert_cache_load_store::<isize>();

    // Float scalars
    assert_cache_load_store::<f16>();
    assert_cache_load_store::<f32>();
    assert_cache_load_store::<f64>();

    // CuSimd register vector types
    assert_cache_load_store::<Float2>();
    assert_cache_load_store::<Float4>();
    assert_cache_load_store::<Double2>();
    assert_cache_load_store::<Uint2>();
    assert_cache_load_store::<Uint4>();
    assert_cache_load_store::<Int2>();
    assert_cache_load_store::<Int4>();

    // Over-aligned memory vector elements
    assert_cache_load_store::<F32x2>();
    assert_cache_load_store::<F32x4>();
    assert_cache_load_store::<F64x2>();
    assert_cache_load_store::<U32x2>();
    assert_cache_load_store::<U32x4>();
}

#[test]
fn test_standalone_functions_type_inference_all_types() {
    unsafe fn check_signatures<T: CacheLoad + CacheStore>(ptr: *mut T, val: T) {
        unsafe {
            let _ = load_streaming(ptr as *const T);
            let _ = load_l2_only(ptr as *const T);
            let _ = load_cached(ptr as *const T);
            let _ = load_last_use(ptr as *const T);
            let _ = load_volatile(ptr as *const T);

            store_streaming(ptr, val);
            store_l2_only(ptr, val);
            store_write_back(ptr, val);
            store_write_through(ptr, val);
        }
    }

    // Integers
    let _ = check_signatures::<u8> as unsafe fn(*mut u8, u8);
    let _ = check_signatures::<i8> as unsafe fn(*mut i8, i8);
    let _ = check_signatures::<u16> as unsafe fn(*mut u16, u16);
    let _ = check_signatures::<i16> as unsafe fn(*mut i16, i16);
    let _ = check_signatures::<u32> as unsafe fn(*mut u32, u32);
    let _ = check_signatures::<i32> as unsafe fn(*mut i32, i32);
    let _ = check_signatures::<u64> as unsafe fn(*mut u64, u64);
    let _ = check_signatures::<i64> as unsafe fn(*mut i64, i64);
    let _ = check_signatures::<usize> as unsafe fn(*mut usize, usize);
    let _ = check_signatures::<isize> as unsafe fn(*mut isize, isize);

    // Floats
    let _ = check_signatures::<f16> as unsafe fn(*mut f16, f16);
    let _ = check_signatures::<f32> as unsafe fn(*mut f32, f32);
    let _ = check_signatures::<f64> as unsafe fn(*mut f64, f64);

    // Register vectors
    let _ = check_signatures::<Float2> as unsafe fn(*mut Float2, Float2);
    let _ = check_signatures::<Float4> as unsafe fn(*mut Float4, Float4);
    let _ = check_signatures::<Double2> as unsafe fn(*mut Double2, Double2);
    let _ = check_signatures::<Uint2> as unsafe fn(*mut Uint2, Uint2);
    let _ = check_signatures::<Uint4> as unsafe fn(*mut Uint4, Uint4);
    let _ = check_signatures::<Int2> as unsafe fn(*mut Int2, Int2);
    let _ = check_signatures::<Int4> as unsafe fn(*mut Int4, Int4);

    // Memory vectors
    let _ = check_signatures::<F32x2> as unsafe fn(*mut F32x2, F32x2);
    let _ = check_signatures::<F32x4> as unsafe fn(*mut F32x4, F32x4);
    let _ = check_signatures::<F64x2> as unsafe fn(*mut F64x2, F64x2);
    let _ = check_signatures::<U32x2> as unsafe fn(*mut U32x2, U32x2);
    let _ = check_signatures::<U32x4> as unsafe fn(*mut U32x4, U32x4);
}
