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

/// Restricts cache-aware stores to supported scalar carrier types.
mod global_store_element {
    /// Implemented only for f16, f32 and packed u32 store elements.
    pub trait Sealed {}
    impl Sealed for f16 {}
    impl Sealed for f32 {}
    impl Sealed for u32 {}
}

/// Scalar storage supported by a cache-aware register-to-global copy.
///
/// Implemented for `f16`, `f32`, and `u32` (including packed FP16 pairs).
pub trait GlobalStoreElement: Copy + global_store_element::Sealed {}
impl GlobalStoreElement for f16 {}
impl GlobalStoreElement for f32 {}
impl GlobalStoreElement for u32 {}

/// A naturally owned, 32-byte-aligned carrier for a global register-tile store.
///
/// The alignment is part of the Rust type, so passing a reference to this
/// carrier never strengthens the source pointer's alignment by assumption.
/// Construct it directly as `GlobalStoreTile([a, b, c, d, e, f, g, h])`.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalStoreTile<T: GlobalStoreElement, const N: usize>(
    #[doc = "Ordered scalar carriers to write in one aligned global transaction."] pub [T; N],
);

/// Store one adjacent register tile without allocating its data in L1.
///
/// This is an ordinary weak, non-atomic store with a cache eviction hint; it
/// supplies no synchronization. The copy width is `N * size_of::<T>()` bytes.
/// Supported widths are 4, 8, 16, and 32 bytes. The 32-byte form requires SM100
/// or newer; the selected backend checks target support. Existing memory
/// contents are replaced without conversion, including packed `u32` values.
///
/// # Safety
///
/// `dst` must point to a live, writable global-memory range of `N` elements,
/// aligned to the full copy width. No conflicting access may race this store.
/// The aligned source carrier must remain live during the copy. The transfer
/// width must be supported on the selected target. This function is a device compiler
/// boundary and panics if executed on the host.
#[inline(never)]
pub unsafe fn store_global_no_allocate<T: GlobalStoreElement, const N: usize>(
    dst: *mut T,
    values: &GlobalStoreTile<T, N>,
) {
    const {
        assert!(matches!(
            N.checked_mul(core::mem::size_of::<T>()),
            Some(4 | 8 | 16 | 32)
        ));
    }
    let _ = (dst, values);
    unreachable!("cute-rs `store_global_no_allocate` executed outside device compilation")
}
