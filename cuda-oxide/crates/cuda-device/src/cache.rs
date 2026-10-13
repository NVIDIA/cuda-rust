/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cache-control streaming and eviction qualifiers for global memory operations.
//!
//! Standard global memory accesses in CUDA default to caching at all levels
//! (`.ca` for loads, `.wb` for stores). In bandwidth-bound kernels (attention KV
//! caching, streaming reductions, or large batched scans), single-pass data
//! lacks temporal reuse and evicts hot resident tiles or lookup tables from L1
//! and L2 caches.
//!
//! This module exposes PTX cache management qualifiers as zero-overhead,
//! type-safe primitives supporting both scalar and vectorized (`v2`, `v4`)
//! transactions.
//!
//! # Qualifiers
//!
//! ### Loads
//! - **Streaming (`.cs`) / [`load_streaming`]:** Allocates cache lines with
//!   evict-first priority so single-pass sweeps do not displace hot tiles.
//! - **L2-Only / Bypass L1 (`.cg`) / [`load_l2_only`]:** Caches at global (L2)
//!   level only, preserving L1 capacity for register spills and shared memory.
//! - **Cache All (`.ca`) / [`load_cached`]:** Default caching at all levels.
//! - **Last Use (`.lu`) / [`load_last_use`]:** Demotes line priority to LRU /
//!   evict-first upon read completion.
//! - **Volatile (`.cv`) / [`load_volatile`]:** Bypasses all caches.
//!
//! ### Stores
//! - **Streaming (`.cs`) / [`store_streaming`]:** Evict-first priority on write.
//! - **L2-Only / Bypass L1 (`.cg`) / [`store_l2_only`]:** Writes to L2, bypassing L1.
//! - **Write-Back (`.wb`) / [`store_write_back`]:** Standard write-back caching.
//! - **Write-Through (`.wt`) / [`store_write_through`]:** Writes through to L2 and
//!   invalidates the line in L1.

use crate::cusimd::{Double2, Float2, Float4, Int2, Int4, Uint2, Uint4};
use crate::ptx_asm;
use crate::vector::{F32x2, F32x4, F64x2, U32x2, U32x4};

/// Types loaded from global memory with explicit PTX cache qualifiers.
pub trait CacheLoad: Copy + Sized {
    /// Load with `.cs` (cache streaming / evict-first).
    ///
    /// # Safety
    /// `ptr` must be non-null, aligned for `Self`, and point to valid global memory.
    unsafe fn load_cs(ptr: *const Self) -> Self;

    /// Load with `.cg` (L2 only, bypass L1).
    ///
    /// # Safety
    /// `ptr` must be non-null, aligned for `Self`, and point to valid global memory.
    unsafe fn load_cg(ptr: *const Self) -> Self;

    /// Load with `.ca` (cache at all levels).
    ///
    /// # Safety
    /// `ptr` must be non-null, aligned for `Self`, and point to valid global memory.
    unsafe fn load_ca(ptr: *const Self) -> Self;

    /// Load with `.lu` (last-use / demote priority).
    ///
    /// # Safety
    /// `ptr` must be non-null, aligned for `Self`, and point to valid global memory.
    unsafe fn load_lu(ptr: *const Self) -> Self;

    /// Load with `.cv` (volatile).
    ///
    /// # Safety
    /// `ptr` must be non-null, aligned for `Self`, and point to valid global memory.
    unsafe fn load_cv(ptr: *const Self) -> Self;
}

/// Types stored to global memory with explicit PTX cache qualifiers.
pub trait CacheStore: Copy + Sized {
    /// Store with `.cs` (cache streaming / evict-first).
    ///
    /// # Safety
    /// `ptr` must be non-null, aligned for `Self`, and point to valid writable global memory.
    unsafe fn store_cs(ptr: *mut Self, val: Self);

    /// Store with `.cg` (L2 only, bypass L1).
    ///
    /// # Safety
    /// `ptr` must be non-null, aligned for `Self`, and point to valid writable global memory.
    unsafe fn store_cg(ptr: *mut Self, val: Self);

    /// Store with `.wb` (write-back caching).
    ///
    /// # Safety
    /// `ptr` must be non-null, aligned for `Self`, and point to valid writable global memory.
    unsafe fn store_wb(ptr: *mut Self, val: Self);

    /// Store with `.wt` (write-through to L2, invalidate L1).
    ///
    /// # Safety
    /// `ptr` must be non-null, aligned for `Self`, and point to valid writable global memory.
    unsafe fn store_wt(ptr: *mut Self, val: Self);
}

// =============================================================================
// Standalone Free Functions
// =============================================================================

/// Load with streaming eviction (`.cs`, evict-first).
///
/// # Safety
/// `ptr` must be non-null, aligned for `T`, and point to valid global memory.
#[inline(always)]
pub unsafe fn load_streaming<T: CacheLoad>(ptr: *const T) -> T {
    // SAFETY: caller guarantees pointer validity and alignment.
    unsafe { T::load_cs(ptr) }
}

/// Load caching at L2 only and bypassing L1 (`.cg`).
///
/// # Safety
/// `ptr` must be non-null, aligned for `T`, and point to valid global memory.
#[inline(always)]
pub unsafe fn load_l2_only<T: CacheLoad>(ptr: *const T) -> T {
    // SAFETY: caller guarantees pointer validity and alignment.
    unsafe { T::load_cg(ptr) }
}

/// Load caching at all levels (`.ca`).
///
/// # Safety
/// `ptr` must be non-null, aligned for `T`, and point to valid global memory.
#[inline(always)]
pub unsafe fn load_cached<T: CacheLoad>(ptr: *const T) -> T {
    // SAFETY: caller guarantees pointer validity and alignment.
    unsafe { T::load_ca(ptr) }
}

/// Load marking the cache line as last-use (`.lu`).
///
/// # Safety
/// `ptr` must be non-null, aligned for `T`, and point to valid global memory.
#[inline(always)]
pub unsafe fn load_last_use<T: CacheLoad>(ptr: *const T) -> T {
    // SAFETY: caller guarantees pointer validity and alignment.
    unsafe { T::load_lu(ptr) }
}

/// Load bypassing all caches (`.cv`, volatile).
///
/// # Safety
/// `ptr` must be non-null, aligned for `T`, and point to valid global memory.
#[inline(always)]
pub unsafe fn load_volatile<T: CacheLoad>(ptr: *const T) -> T {
    // SAFETY: caller guarantees pointer validity and alignment.
    unsafe { T::load_cv(ptr) }
}

/// Store with streaming eviction (`.cs`, evict-first).
///
/// # Safety
/// `ptr` must be non-null, aligned for `T`, and point to valid writable global memory.
#[inline(always)]
pub unsafe fn store_streaming<T: CacheStore>(ptr: *mut T, val: T) {
    // SAFETY: caller guarantees pointer validity and alignment.
    unsafe { T::store_cs(ptr, val) }
}

/// Store caching at L2 only and bypassing L1 (`.cg`).
///
/// # Safety
/// `ptr` must be non-null, aligned for `T`, and point to valid writable global memory.
#[inline(always)]
pub unsafe fn store_l2_only<T: CacheStore>(ptr: *mut T, val: T) {
    // SAFETY: caller guarantees pointer validity and alignment.
    unsafe { T::store_cg(ptr, val) }
}

/// Store with write-back caching at all levels (`.wb`).
///
/// # Safety
/// `ptr` must be non-null, aligned for `T`, and point to valid writable global memory.
#[inline(always)]
pub unsafe fn store_write_back<T: CacheStore>(ptr: *mut T, val: T) {
    // SAFETY: caller guarantees pointer validity and alignment.
    unsafe { T::store_wb(ptr, val) }
}

/// Store with write-through caching (`.wt`).
///
/// # Safety
/// `ptr` must be non-null, aligned for `T`, and point to valid writable global memory.
#[inline(always)]
pub unsafe fn store_write_through<T: CacheStore>(ptr: *mut T, val: T) {
    // SAFETY: caller guarantees pointer validity and alignment.
    unsafe { T::store_wt(ptr, val) }
}

// =============================================================================
// Scalar Implementations
// =============================================================================

macro_rules! impl_cache_scalar {
    (
        $ty:ty,
        $ld_cs:literal,
        $ld_cg:literal,
        $ld_ca:literal,
        $ld_lu:literal,
        $ld_cv:literal,
        $st_cs:literal,
        $st_cg:literal,
        $st_wb:literal,
        $st_wt:literal,
        $reg:literal,
        $out_reg:literal
    ) => {
        impl CacheLoad for $ty {
            #[inline(always)]
            unsafe fn load_cs(ptr: *const Self) -> Self {
                let res: Self;
                unsafe {
                    ptx_asm!($ld_cs, out($out_reg) res, in("l") ptr);
                }
                res
            }

            #[inline(always)]
            unsafe fn load_cg(ptr: *const Self) -> Self {
                let res: Self;
                unsafe {
                    ptx_asm!($ld_cg, out($out_reg) res, in("l") ptr);
                }
                res
            }

            #[inline(always)]
            unsafe fn load_ca(ptr: *const Self) -> Self {
                let res: Self;
                unsafe {
                    ptx_asm!($ld_ca, out($out_reg) res, in("l") ptr);
                }
                res
            }

            #[inline(always)]
            unsafe fn load_lu(ptr: *const Self) -> Self {
                let res: Self;
                unsafe {
                    ptx_asm!($ld_lu, out($out_reg) res, in("l") ptr);
                }
                res
            }

            #[inline(always)]
            unsafe fn load_cv(ptr: *const Self) -> Self {
                let res: Self;
                unsafe {
                    ptx_asm!($ld_cv, out($out_reg) res, in("l") ptr);
                }
                res
            }
        }

        impl CacheStore for $ty {
            #[inline(always)]
            unsafe fn store_cs(ptr: *mut Self, val: Self) {
                unsafe {
                    ptx_asm!($st_cs, in("l") ptr, in($reg) val);
                }
            }

            #[inline(always)]
            unsafe fn store_cg(ptr: *mut Self, val: Self) {
                unsafe {
                    ptx_asm!($st_cg, in("l") ptr, in($reg) val);
                }
            }

            #[inline(always)]
            unsafe fn store_wb(ptr: *mut Self, val: Self) {
                unsafe {
                    ptx_asm!($st_wb, in("l") ptr, in($reg) val);
                }
            }

            #[inline(always)]
            unsafe fn store_wt(ptr: *mut Self, val: Self) {
                unsafe {
                    ptx_asm!($st_wt, in("l") ptr, in($reg) val);
                }
            }
        }
    };
}

// 8-bit integers widened to 32-bit registers at PTX boundary
impl CacheLoad for u8 {
    #[inline(always)]
    unsafe fn load_cs(ptr: *const Self) -> Self {
        let res: u32;
        unsafe {
            ptx_asm!("ld.global.cs.u8 %0, [%1];", out("=r") res, in("l") ptr);
        }
        res as u8
    }

    #[inline(always)]
    unsafe fn load_cg(ptr: *const Self) -> Self {
        let res: u32;
        unsafe {
            ptx_asm!("ld.global.cg.u8 %0, [%1];", out("=r") res, in("l") ptr);
        }
        res as u8
    }

    #[inline(always)]
    unsafe fn load_ca(ptr: *const Self) -> Self {
        let res: u32;
        unsafe {
            ptx_asm!("ld.global.ca.u8 %0, [%1];", out("=r") res, in("l") ptr);
        }
        res as u8
    }

    #[inline(always)]
    unsafe fn load_lu(ptr: *const Self) -> Self {
        let res: u32;
        unsafe {
            ptx_asm!("ld.global.lu.u8 %0, [%1];", out("=r") res, in("l") ptr);
        }
        res as u8
    }

    #[inline(always)]
    unsafe fn load_cv(ptr: *const Self) -> Self {
        let res: u32;
        unsafe {
            ptx_asm!("ld.global.cv.u8 %0, [%1];", out("=r") res, in("l") ptr);
        }
        res as u8
    }
}

impl CacheStore for u8 {
    #[inline(always)]
    unsafe fn store_cs(ptr: *mut Self, val: Self) {
        unsafe {
            ptx_asm!(
                "st.global.cs.u8 [%0], %1;",
                in("l") ptr,
                in("r") val as u32
            );
        }
    }

    #[inline(always)]
    unsafe fn store_cg(ptr: *mut Self, val: Self) {
        unsafe {
            ptx_asm!(
                "st.global.cg.u8 [%0], %1;",
                in("l") ptr,
                in("r") val as u32
            );
        }
    }

    #[inline(always)]
    unsafe fn store_wb(ptr: *mut Self, val: Self) {
        unsafe {
            ptx_asm!(
                "st.global.wb.u8 [%0], %1;",
                in("l") ptr,
                in("r") val as u32
            );
        }
    }

    #[inline(always)]
    unsafe fn store_wt(ptr: *mut Self, val: Self) {
        unsafe {
            ptx_asm!(
                "st.global.wt.u8 [%0], %1;",
                in("l") ptr,
                in("r") val as u32
            );
        }
    }
}

impl CacheLoad for i8 {
    #[inline(always)]
    unsafe fn load_cs(ptr: *const Self) -> Self {
        let res: i32;
        unsafe {
            ptx_asm!("ld.global.cs.s8 %0, [%1];", out("=r") res, in("l") ptr);
        }
        res as i8
    }

    #[inline(always)]
    unsafe fn load_cg(ptr: *const Self) -> Self {
        let res: i32;
        unsafe {
            ptx_asm!("ld.global.cg.s8 %0, [%1];", out("=r") res, in("l") ptr);
        }
        res as i8
    }

    #[inline(always)]
    unsafe fn load_ca(ptr: *const Self) -> Self {
        let res: i32;
        unsafe {
            ptx_asm!("ld.global.ca.s8 %0, [%1];", out("=r") res, in("l") ptr);
        }
        res as i8
    }

    #[inline(always)]
    unsafe fn load_lu(ptr: *const Self) -> Self {
        let res: i32;
        unsafe {
            ptx_asm!("ld.global.lu.s8 %0, [%1];", out("=r") res, in("l") ptr);
        }
        res as i8
    }

    #[inline(always)]
    unsafe fn load_cv(ptr: *const Self) -> Self {
        let res: i32;
        unsafe {
            ptx_asm!("ld.global.cv.s8 %0, [%1];", out("=r") res, in("l") ptr);
        }
        res as i8
    }
}

impl CacheStore for i8 {
    #[inline(always)]
    unsafe fn store_cs(ptr: *mut Self, val: Self) {
        unsafe {
            ptx_asm!(
                "st.global.cs.s8 [%0], %1;",
                in("l") ptr,
                in("r") val as i32
            );
        }
    }

    #[inline(always)]
    unsafe fn store_cg(ptr: *mut Self, val: Self) {
        unsafe {
            ptx_asm!(
                "st.global.cg.s8 [%0], %1;",
                in("l") ptr,
                in("r") val as i32
            );
        }
    }

    #[inline(always)]
    unsafe fn store_wb(ptr: *mut Self, val: Self) {
        unsafe {
            ptx_asm!(
                "st.global.wb.s8 [%0], %1;",
                in("l") ptr,
                in("r") val as i32
            );
        }
    }

    #[inline(always)]
    unsafe fn store_wt(ptr: *mut Self, val: Self) {
        unsafe {
            ptx_asm!(
                "st.global.wt.s8 [%0], %1;",
                in("l") ptr,
                in("r") val as i32
            );
        }
    }
}

// 16-bit, 32-bit, and 64-bit integer scalars
impl_cache_scalar!(
    u16,
    "ld.global.cs.u16 %0, [%1];",
    "ld.global.cg.u16 %0, [%1];",
    "ld.global.ca.u16 %0, [%1];",
    "ld.global.lu.u16 %0, [%1];",
    "ld.global.cv.u16 %0, [%1];",
    "st.global.cs.u16 [%0], %1;",
    "st.global.cg.u16 [%0], %1;",
    "st.global.wb.u16 [%0], %1;",
    "st.global.wt.u16 [%0], %1;",
    "h",
    "=h"
);

impl_cache_scalar!(
    i16,
    "ld.global.cs.s16 %0, [%1];",
    "ld.global.cg.s16 %0, [%1];",
    "ld.global.ca.s16 %0, [%1];",
    "ld.global.lu.s16 %0, [%1];",
    "ld.global.cv.s16 %0, [%1];",
    "st.global.cs.s16 [%0], %1;",
    "st.global.cg.s16 [%0], %1;",
    "st.global.wb.s16 [%0], %1;",
    "st.global.wt.s16 [%0], %1;",
    "h",
    "=h"
);

impl_cache_scalar!(
    u32,
    "ld.global.cs.u32 %0, [%1];",
    "ld.global.cg.u32 %0, [%1];",
    "ld.global.ca.u32 %0, [%1];",
    "ld.global.lu.u32 %0, [%1];",
    "ld.global.cv.u32 %0, [%1];",
    "st.global.cs.u32 [%0], %1;",
    "st.global.cg.u32 [%0], %1;",
    "st.global.wb.u32 [%0], %1;",
    "st.global.wt.u32 [%0], %1;",
    "r",
    "=r"
);

impl_cache_scalar!(
    i32,
    "ld.global.cs.s32 %0, [%1];",
    "ld.global.cg.s32 %0, [%1];",
    "ld.global.ca.s32 %0, [%1];",
    "ld.global.lu.s32 %0, [%1];",
    "ld.global.cv.s32 %0, [%1];",
    "st.global.cs.s32 [%0], %1;",
    "st.global.cg.s32 [%0], %1;",
    "st.global.wb.s32 [%0], %1;",
    "st.global.wt.s32 [%0], %1;",
    "r",
    "=r"
);

impl_cache_scalar!(
    u64,
    "ld.global.cs.u64 %0, [%1];",
    "ld.global.cg.u64 %0, [%1];",
    "ld.global.ca.u64 %0, [%1];",
    "ld.global.lu.u64 %0, [%1];",
    "ld.global.cv.u64 %0, [%1];",
    "st.global.cs.u64 [%0], %1;",
    "st.global.cg.u64 [%0], %1;",
    "st.global.wb.u64 [%0], %1;",
    "st.global.wt.u64 [%0], %1;",
    "l",
    "=l"
);

impl_cache_scalar!(
    i64,
    "ld.global.cs.s64 %0, [%1];",
    "ld.global.cg.s64 %0, [%1];",
    "ld.global.ca.s64 %0, [%1];",
    "ld.global.lu.s64 %0, [%1];",
    "ld.global.cv.s64 %0, [%1];",
    "st.global.cs.s64 [%0], %1;",
    "st.global.cg.s64 [%0], %1;",
    "st.global.wb.s64 [%0], %1;",
    "st.global.wt.s64 [%0], %1;",
    "l",
    "=l"
);

impl_cache_scalar!(
    f32,
    "ld.global.cs.f32 %0, [%1];",
    "ld.global.cg.f32 %0, [%1];",
    "ld.global.ca.f32 %0, [%1];",
    "ld.global.lu.f32 %0, [%1];",
    "ld.global.cv.f32 %0, [%1];",
    "st.global.cs.f32 [%0], %1;",
    "st.global.cg.f32 [%0], %1;",
    "st.global.wb.f32 [%0], %1;",
    "st.global.wt.f32 [%0], %1;",
    "f",
    "=f"
);

impl_cache_scalar!(
    f64,
    "ld.global.cs.f64 %0, [%1];",
    "ld.global.cg.f64 %0, [%1];",
    "ld.global.ca.f64 %0, [%1];",
    "ld.global.lu.f64 %0, [%1];",
    "ld.global.cv.f64 %0, [%1];",
    "st.global.cs.f64 [%0], %1;",
    "st.global.cg.f64 [%0], %1;",
    "st.global.wb.f64 [%0], %1;",
    "st.global.wt.f64 [%0], %1;",
    "d",
    "=d"
);

// usize and isize on 64-bit NVPTX
impl CacheLoad for usize {
    #[inline(always)]
    unsafe fn load_cs(ptr: *const Self) -> Self {
        unsafe { u64::load_cs(ptr as *const u64) as usize }
    }

    #[inline(always)]
    unsafe fn load_cg(ptr: *const Self) -> Self {
        unsafe { u64::load_cg(ptr as *const u64) as usize }
    }

    #[inline(always)]
    unsafe fn load_ca(ptr: *const Self) -> Self {
        unsafe { u64::load_ca(ptr as *const u64) as usize }
    }

    #[inline(always)]
    unsafe fn load_lu(ptr: *const Self) -> Self {
        unsafe { u64::load_lu(ptr as *const u64) as usize }
    }

    #[inline(always)]
    unsafe fn load_cv(ptr: *const Self) -> Self {
        unsafe { u64::load_cv(ptr as *const u64) as usize }
    }
}

impl CacheStore for usize {
    #[inline(always)]
    unsafe fn store_cs(ptr: *mut Self, val: Self) {
        unsafe { u64::store_cs(ptr as *mut u64, val as u64) }
    }

    #[inline(always)]
    unsafe fn store_cg(ptr: *mut Self, val: Self) {
        unsafe { u64::store_cg(ptr as *mut u64, val as u64) }
    }

    #[inline(always)]
    unsafe fn store_wb(ptr: *mut Self, val: Self) {
        unsafe { u64::store_wb(ptr as *mut u64, val as u64) }
    }

    #[inline(always)]
    unsafe fn store_wt(ptr: *mut Self, val: Self) {
        unsafe { u64::store_wt(ptr as *mut u64, val as u64) }
    }
}

impl CacheLoad for isize {
    #[inline(always)]
    unsafe fn load_cs(ptr: *const Self) -> Self {
        unsafe { i64::load_cs(ptr as *const i64) as isize }
    }

    #[inline(always)]
    unsafe fn load_cg(ptr: *const Self) -> Self {
        unsafe { i64::load_cg(ptr as *const i64) as isize }
    }

    #[inline(always)]
    unsafe fn load_ca(ptr: *const Self) -> Self {
        unsafe { i64::load_ca(ptr as *const i64) as isize }
    }

    #[inline(always)]
    unsafe fn load_lu(ptr: *const Self) -> Self {
        unsafe { i64::load_lu(ptr as *const i64) as isize }
    }

    #[inline(always)]
    unsafe fn load_cv(ptr: *const Self) -> Self {
        unsafe { i64::load_cv(ptr as *const i64) as isize }
    }
}

impl CacheStore for isize {
    #[inline(always)]
    unsafe fn store_cs(ptr: *mut Self, val: Self) {
        unsafe { i64::store_cs(ptr as *mut i64, val as i64) }
    }

    #[inline(always)]
    unsafe fn store_cg(ptr: *mut Self, val: Self) {
        unsafe { i64::store_cg(ptr as *mut i64, val as i64) }
    }

    #[inline(always)]
    unsafe fn store_wb(ptr: *mut Self, val: Self) {
        unsafe { i64::store_wb(ptr as *mut i64, val as i64) }
    }

    #[inline(always)]
    unsafe fn store_wt(ptr: *mut Self, val: Self) {
        unsafe { i64::store_wt(ptr as *mut i64, val as i64) }
    }
}

impl CacheLoad for f16 {
    #[inline(always)]
    unsafe fn load_cs(ptr: *const Self) -> Self {
        let bits: u16;
        unsafe {
            ptx_asm!("ld.global.cs.b16 %0, [%1];", out("=h") bits, in("l") ptr);
        }
        Self::from_bits(bits)
    }

    #[inline(always)]
    unsafe fn load_cg(ptr: *const Self) -> Self {
        let bits: u16;
        unsafe {
            ptx_asm!("ld.global.cg.b16 %0, [%1];", out("=h") bits, in("l") ptr);
        }
        Self::from_bits(bits)
    }

    #[inline(always)]
    unsafe fn load_ca(ptr: *const Self) -> Self {
        let bits: u16;
        unsafe {
            ptx_asm!("ld.global.ca.b16 %0, [%1];", out("=h") bits, in("l") ptr);
        }
        Self::from_bits(bits)
    }

    #[inline(always)]
    unsafe fn load_lu(ptr: *const Self) -> Self {
        let bits: u16;
        unsafe {
            ptx_asm!("ld.global.lu.b16 %0, [%1];", out("=h") bits, in("l") ptr);
        }
        Self::from_bits(bits)
    }

    #[inline(always)]
    unsafe fn load_cv(ptr: *const Self) -> Self {
        let bits: u16;
        unsafe {
            ptx_asm!("ld.global.cv.b16 %0, [%1];", out("=h") bits, in("l") ptr);
        }
        Self::from_bits(bits)
    }
}

impl CacheStore for f16 {
    #[inline(always)]
    unsafe fn store_cs(ptr: *mut Self, val: Self) {
        let bits = val.to_bits();
        unsafe {
            ptx_asm!("st.global.cs.b16 [%0], %1;", in("l") ptr, in("h") bits);
        }
    }

    #[inline(always)]
    unsafe fn store_cg(ptr: *mut Self, val: Self) {
        let bits = val.to_bits();
        unsafe {
            ptx_asm!("st.global.cg.b16 [%0], %1;", in("l") ptr, in("h") bits);
        }
    }

    #[inline(always)]
    unsafe fn store_wb(ptr: *mut Self, val: Self) {
        let bits = val.to_bits();
        unsafe {
            ptx_asm!("st.global.wb.b16 [%0], %1;", in("l") ptr, in("h") bits);
        }
    }

    #[inline(always)]
    unsafe fn store_wt(ptr: *mut Self, val: Self) {
        let bits = val.to_bits();
        unsafe {
            ptx_asm!("st.global.wt.b16 [%0], %1;", in("l") ptr, in("h") bits);
        }
    }
}

// =============================================================================
// Vector Implementations (Float2, Float4, Double2, Uint2, Uint4, Int2, Int4)
// =============================================================================

macro_rules! impl_cache_vector_v2 {
    (
        $vec_ty:ident,
        $elem_ty:ident,
        $ld_cs:literal,
        $ld_cg:literal,
        $ld_ca:literal,
        $ld_lu:literal,
        $ld_cv:literal,
        $st_cs:literal,
        $st_cg:literal,
        $st_wb:literal,
        $st_wt:literal,
        $reg:literal,
        $out_reg:literal
    ) => {
        impl CacheLoad for $vec_ty {
            #[inline(always)]
            unsafe fn load_cs(ptr: *const Self) -> Self {
                let (x, y): ($elem_ty, $elem_ty);
                unsafe {
                    ptx_asm!($ld_cs, out($out_reg) x, out($out_reg) y, in("l") ptr);
                }
                $vec_ty::new([x, y])
            }

            #[inline(always)]
            unsafe fn load_cg(ptr: *const Self) -> Self {
                let (x, y): ($elem_ty, $elem_ty);
                unsafe {
                    ptx_asm!($ld_cg, out($out_reg) x, out($out_reg) y, in("l") ptr);
                }
                $vec_ty::new([x, y])
            }

            #[inline(always)]
            unsafe fn load_ca(ptr: *const Self) -> Self {
                let (x, y): ($elem_ty, $elem_ty);
                unsafe {
                    ptx_asm!($ld_ca, out($out_reg) x, out($out_reg) y, in("l") ptr);
                }
                $vec_ty::new([x, y])
            }

            #[inline(always)]
            unsafe fn load_lu(ptr: *const Self) -> Self {
                let (x, y): ($elem_ty, $elem_ty);
                unsafe {
                    ptx_asm!($ld_lu, out($out_reg) x, out($out_reg) y, in("l") ptr);
                }
                $vec_ty::new([x, y])
            }

            #[inline(always)]
            unsafe fn load_cv(ptr: *const Self) -> Self {
                let (x, y): ($elem_ty, $elem_ty);
                unsafe {
                    ptx_asm!($ld_cv, out($out_reg) x, out($out_reg) y, in("l") ptr);
                }
                $vec_ty::new([x, y])
            }
        }

        impl CacheStore for $vec_ty {
            #[inline(always)]
            unsafe fn store_cs(ptr: *mut Self, val: Self) {
                unsafe {
                    ptx_asm!($st_cs, in("l") ptr, in($reg) val.x(), in($reg) val.y());
                }
            }

            #[inline(always)]
            unsafe fn store_cg(ptr: *mut Self, val: Self) {
                unsafe {
                    ptx_asm!($st_cg, in("l") ptr, in($reg) val.x(), in($reg) val.y());
                }
            }

            #[inline(always)]
            unsafe fn store_wb(ptr: *mut Self, val: Self) {
                unsafe {
                    ptx_asm!($st_wb, in("l") ptr, in($reg) val.x(), in($reg) val.y());
                }
            }

            #[inline(always)]
            unsafe fn store_wt(ptr: *mut Self, val: Self) {
                unsafe {
                    ptx_asm!($st_wt, in("l") ptr, in($reg) val.x(), in($reg) val.y());
                }
            }
        }
    };
}

macro_rules! impl_cache_vector_v4 {
    (
        $vec_ty:ident,
        $elem_ty:ident,
        $ld_cs:literal,
        $ld_cg:literal,
        $ld_ca:literal,
        $ld_lu:literal,
        $ld_cv:literal,
        $st_cs:literal,
        $st_cg:literal,
        $st_wb:literal,
        $st_wt:literal,
        $reg:literal,
        $out_reg:literal
    ) => {
        impl CacheLoad for $vec_ty {
            #[inline(always)]
            unsafe fn load_cs(ptr: *const Self) -> Self {
                let (x, y, z, w): ($elem_ty, $elem_ty, $elem_ty, $elem_ty);
                unsafe {
                    ptx_asm!(
                        $ld_cs,
                        out($out_reg) x,
                        out($out_reg) y,
                        out($out_reg) z,
                        out($out_reg) w,
                        in("l") ptr,
                    );
                }
                $vec_ty::new([x, y, z, w])
            }

            #[inline(always)]
            unsafe fn load_cg(ptr: *const Self) -> Self {
                let (x, y, z, w): ($elem_ty, $elem_ty, $elem_ty, $elem_ty);
                unsafe {
                    ptx_asm!(
                        $ld_cg,
                        out($out_reg) x,
                        out($out_reg) y,
                        out($out_reg) z,
                        out($out_reg) w,
                        in("l") ptr,
                    );
                }
                $vec_ty::new([x, y, z, w])
            }

            #[inline(always)]
            unsafe fn load_ca(ptr: *const Self) -> Self {
                let (x, y, z, w): ($elem_ty, $elem_ty, $elem_ty, $elem_ty);
                unsafe {
                    ptx_asm!(
                        $ld_ca,
                        out($out_reg) x,
                        out($out_reg) y,
                        out($out_reg) z,
                        out($out_reg) w,
                        in("l") ptr,
                    );
                }
                $vec_ty::new([x, y, z, w])
            }

            #[inline(always)]
            unsafe fn load_lu(ptr: *const Self) -> Self {
                let (x, y, z, w): ($elem_ty, $elem_ty, $elem_ty, $elem_ty);
                unsafe {
                    ptx_asm!(
                        $ld_lu,
                        out($out_reg) x,
                        out($out_reg) y,
                        out($out_reg) z,
                        out($out_reg) w,
                        in("l") ptr,
                    );
                }
                $vec_ty::new([x, y, z, w])
            }

            #[inline(always)]
            unsafe fn load_cv(ptr: *const Self) -> Self {
                let (x, y, z, w): ($elem_ty, $elem_ty, $elem_ty, $elem_ty);
                unsafe {
                    ptx_asm!(
                        $ld_cv,
                        out($out_reg) x,
                        out($out_reg) y,
                        out($out_reg) z,
                        out($out_reg) w,
                        in("l") ptr,
                    );
                }
                $vec_ty::new([x, y, z, w])
            }
        }

        impl CacheStore for $vec_ty {
            #[inline(always)]
            unsafe fn store_cs(ptr: *mut Self, val: Self) {
                unsafe {
                    ptx_asm!(
                        $st_cs,
                        in("l") ptr,
                        in($reg) val.x(),
                        in($reg) val.y(),
                        in($reg) val.z(),
                        in($reg) val.w(),
                    );
                }
            }

            #[inline(always)]
            unsafe fn store_cg(ptr: *mut Self, val: Self) {
                unsafe {
                    ptx_asm!(
                        $st_cg,
                        in("l") ptr,
                        in($reg) val.x(),
                        in($reg) val.y(),
                        in($reg) val.z(),
                        in($reg) val.w(),
                    );
                }
            }

            #[inline(always)]
            unsafe fn store_wb(ptr: *mut Self, val: Self) {
                unsafe {
                    ptx_asm!(
                        $st_wb,
                        in("l") ptr,
                        in($reg) val.x(),
                        in($reg) val.y(),
                        in($reg) val.z(),
                        in($reg) val.w(),
                    );
                }
            }

            #[inline(always)]
            unsafe fn store_wt(ptr: *mut Self, val: Self) {
                unsafe {
                    ptx_asm!(
                        $st_wt,
                        in("l") ptr,
                        in($reg) val.x(),
                        in($reg) val.y(),
                        in($reg) val.z(),
                        in($reg) val.w(),
                    );
                }
            }
        }
    };
}

impl_cache_vector_v2!(
    Float2,
    f32,
    "ld.global.cs.v2.f32 {%0, %1}, [%2];",
    "ld.global.cg.v2.f32 {%0, %1}, [%2];",
    "ld.global.ca.v2.f32 {%0, %1}, [%2];",
    "ld.global.lu.v2.f32 {%0, %1}, [%2];",
    "ld.global.cv.v2.f32 {%0, %1}, [%2];",
    "st.global.cs.v2.f32 [%0], {%1, %2};",
    "st.global.cg.v2.f32 [%0], {%1, %2};",
    "st.global.wb.v2.f32 [%0], {%1, %2};",
    "st.global.wt.v2.f32 [%0], {%1, %2};",
    "f",
    "=f"
);

impl_cache_vector_v4!(
    Float4,
    f32,
    "ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];",
    "ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];",
    "ld.global.ca.v4.f32 {%0, %1, %2, %3}, [%4];",
    "ld.global.lu.v4.f32 {%0, %1, %2, %3}, [%4];",
    "ld.global.cv.v4.f32 {%0, %1, %2, %3}, [%4];",
    "st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};",
    "st.global.cg.v4.f32 [%0], {%1, %2, %3, %4};",
    "st.global.wb.v4.f32 [%0], {%1, %2, %3, %4};",
    "st.global.wt.v4.f32 [%0], {%1, %2, %3, %4};",
    "f",
    "=f"
);

impl_cache_vector_v2!(
    Double2,
    f64,
    "ld.global.cs.v2.f64 {%0, %1}, [%2];",
    "ld.global.cg.v2.f64 {%0, %1}, [%2];",
    "ld.global.ca.v2.f64 {%0, %1}, [%2];",
    "ld.global.lu.v2.f64 {%0, %1}, [%2];",
    "ld.global.cv.v2.f64 {%0, %1}, [%2];",
    "st.global.cs.v2.f64 [%0], {%1, %2};",
    "st.global.cg.v2.f64 [%0], {%1, %2};",
    "st.global.wb.v2.f64 [%0], {%1, %2};",
    "st.global.wt.v2.f64 [%0], {%1, %2};",
    "d",
    "=d"
);

impl_cache_vector_v2!(
    Uint2,
    u32,
    "ld.global.cs.v2.u32 {%0, %1}, [%2];",
    "ld.global.cg.v2.u32 {%0, %1}, [%2];",
    "ld.global.ca.v2.u32 {%0, %1}, [%2];",
    "ld.global.lu.v2.u32 {%0, %1}, [%2];",
    "ld.global.cv.v2.u32 {%0, %1}, [%2];",
    "st.global.cs.v2.u32 [%0], {%1, %2};",
    "st.global.cg.v2.u32 [%0], {%1, %2};",
    "st.global.wb.v2.u32 [%0], {%1, %2};",
    "st.global.wt.v2.u32 [%0], {%1, %2};",
    "r",
    "=r"
);

impl_cache_vector_v4!(
    Uint4,
    u32,
    "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];",
    "ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];",
    "ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];",
    "ld.global.lu.v4.u32 {%0, %1, %2, %3}, [%4];",
    "ld.global.cv.v4.u32 {%0, %1, %2, %3}, [%4];",
    "st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};",
    "st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};",
    "st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};",
    "st.global.wt.v4.u32 [%0], {%1, %2, %3, %4};",
    "r",
    "=r"
);

impl_cache_vector_v2!(
    Int2,
    i32,
    "ld.global.cs.v2.s32 {%0, %1}, [%2];",
    "ld.global.cg.v2.s32 {%0, %1}, [%2];",
    "ld.global.ca.v2.s32 {%0, %1}, [%2];",
    "ld.global.lu.v2.s32 {%0, %1}, [%2];",
    "ld.global.cv.v2.s32 {%0, %1}, [%2];",
    "st.global.cs.v2.s32 [%0], {%1, %2};",
    "st.global.cg.v2.s32 [%0], {%1, %2};",
    "st.global.wb.v2.s32 [%0], {%1, %2};",
    "st.global.wt.v2.s32 [%0], {%1, %2};",
    "r",
    "=r"
);

impl_cache_vector_v4!(
    Int4,
    i32,
    "ld.global.cs.v4.s32 {%0, %1, %2, %3}, [%4];",
    "ld.global.cg.v4.s32 {%0, %1, %2, %3}, [%4];",
    "ld.global.ca.v4.s32 {%0, %1, %2, %3}, [%4];",
    "ld.global.lu.v4.s32 {%0, %1, %2, %3}, [%4];",
    "ld.global.cv.v4.s32 {%0, %1, %2, %3}, [%4];",
    "st.global.cs.v4.s32 [%0], {%1, %2, %3, %4};",
    "st.global.cg.v4.s32 [%0], {%1, %2, %3, %4};",
    "st.global.wb.v4.s32 [%0], {%1, %2, %3, %4};",
    "st.global.wt.v4.s32 [%0], {%1, %2, %3, %4};",
    "r",
    "=r"
);

// Over-aligned memory vector types (crate::vector)
macro_rules! impl_cache_vector_struct {
    ($vec_ty:ident, $carrier:ident) => {
        impl CacheLoad for $vec_ty {
            #[inline(always)]
            unsafe fn load_cs(ptr: *const Self) -> Self {
                let val = unsafe { $carrier::load_cs(ptr as *const $carrier) };
                Self(val.to_array())
            }

            #[inline(always)]
            unsafe fn load_cg(ptr: *const Self) -> Self {
                let val = unsafe { $carrier::load_cg(ptr as *const $carrier) };
                Self(val.to_array())
            }

            #[inline(always)]
            unsafe fn load_ca(ptr: *const Self) -> Self {
                let val = unsafe { $carrier::load_ca(ptr as *const $carrier) };
                Self(val.to_array())
            }

            #[inline(always)]
            unsafe fn load_lu(ptr: *const Self) -> Self {
                let val = unsafe { $carrier::load_lu(ptr as *const $carrier) };
                Self(val.to_array())
            }

            #[inline(always)]
            unsafe fn load_cv(ptr: *const Self) -> Self {
                let val = unsafe { $carrier::load_cv(ptr as *const $carrier) };
                Self(val.to_array())
            }
        }

        impl CacheStore for $vec_ty {
            #[inline(always)]
            unsafe fn store_cs(ptr: *mut Self, val: Self) {
                unsafe { $carrier::store_cs(ptr as *mut $carrier, $carrier::new(val.0)) }
            }

            #[inline(always)]
            unsafe fn store_cg(ptr: *mut Self, val: Self) {
                unsafe { $carrier::store_cg(ptr as *mut $carrier, $carrier::new(val.0)) }
            }

            #[inline(always)]
            unsafe fn store_wb(ptr: *mut Self, val: Self) {
                unsafe { $carrier::store_wb(ptr as *mut $carrier, $carrier::new(val.0)) }
            }

            #[inline(always)]
            unsafe fn store_wt(ptr: *mut Self, val: Self) {
                unsafe { $carrier::store_wt(ptr as *mut $carrier, $carrier::new(val.0)) }
            }
        }
    };
}

impl_cache_vector_struct!(F32x2, Float2);
impl_cache_vector_struct!(F32x4, Float4);
impl_cache_vector_struct!(F64x2, Double2);
impl_cache_vector_struct!(U32x2, Uint2);
impl_cache_vector_struct!(U32x4, Uint4);
