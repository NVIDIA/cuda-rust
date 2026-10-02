/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! One-thread vector loads and stores.
//!
//! These functions move one short, adjacent tile in a single transaction:
//!
//! ```text
//! load_tile::<f32, 4>(src, 8)
//!                         │
//!                         ▼
//! src: [... | 8 | 9 | 10 | 11 | ...]
//!             └──── one 16-byte load ────┘
//! ```
//!
//! During device compilation, the compiler replaces each function call with
//! a CuTe vector-copy operation. The Rust bodies are never run on the GPU and
//! intentionally panic if called on the host.
//!
//! # Size and alignment
//!
//! The first version supports a total transfer size that is a power of two
//! from 4 through 16 bytes. The caller must guarantee:
//!
//! ```text
//! bytes = N * size_of::<T>()
//! idx % N == 0
//! buffer address % bytes == 0
//! idx + N <= buffer length
//! ```
//!
//! GPU allocations used for kernel slices are 256-byte aligned. A tile index
//! built as `tile_number * N` therefore meets the remaining alignment rule.
//! A short final tile must use checked scalar loads or stores instead.

/// Load `src[idx..idx + N]` as one vector transaction.
///
/// # Safety
///
/// - The full range must be in bounds.
/// - `idx` must be divisible by `N`.
/// - The first byte must be aligned to `N * size_of::<T>()` bytes.
/// - The total byte size must be a supported power of two from 4 through 16.
#[inline(never)]
pub unsafe fn load_tile<T: Copy, const N: usize>(src: &[T], idx: usize) -> [T; N] {
    let _ = (src, idx);
    unreachable!("cute-rs `load_tile` executed outside device compilation")
}

/// Store `vals` at `dst[idx..idx + N]` as one vector transaction.
///
/// `dst` is a raw device pointer, such as one returned by
/// `DisjointSlice::as_mut_ptr`.
///
/// # Safety
///
/// - The full destination range must be live, writable, and in bounds.
/// - No other active access may conflict with that range.
/// - `idx` must be divisible by `N`.
/// - The first byte must be aligned to `N * size_of::<T>()` bytes.
/// - The total byte size must be a supported power of two from 4 through 16.
#[inline(never)]
pub unsafe fn store_tile<T: Copy, const N: usize>(dst: *mut T, idx: usize, vals: &[T; N]) {
    let _ = (dst, idx, vals);
    unreachable!("cute-rs `store_tile` executed outside device compilation")
}

/// Return `x` while promising the compiler that it is divisible by `D`.
///
/// ```text
/// x = 3 * 4 = 12
/// assume_div::<4>(x) ──► 12, with the promise 12 % 4 == 0
/// ```
///
/// Pass the returned value directly to the operation that needs the promise.
/// The compiler does not search through stored structs to find it. Struct
/// fields such as matrix row pitch use `LeadingDim<D>` to keep the same fact
/// in their type.
///
/// Prefer building the value so the promise is obvious; `thread * D` is
/// always divisible by `D`.
///
/// # Safety
///
/// `D` must be greater than zero and `x % D` must equal zero. A false promise
/// may make a later memory instruction misaligned, which is undefined
/// behavior.
#[inline(never)]
pub unsafe fn assume_div<const D: usize>(x: usize) -> usize {
    let _ = x;
    unreachable!("cute-rs `assume_div` executed outside device compilation")
}

// Short final tiles are handled by ordinary checked Rust code:
//
//     if base + N <= len { /* vectorized tile calls */ }
//     else { /* scalar loop over the tail */ }
//
// Only boundary threads use the scalar path. A block-wide `copy_g2s` follows
// a different rule because every thread must reach nearby barriers:
//
//     every thread calls copy_g2s -> its assigned part copies 16 / fewer / 0 bytes
