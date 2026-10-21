/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `SharedPtr` across every boundary a helper call can put it behind.
//!
//! `SharedPtr<T>` is a raw pointer into the issuing block's shared memory
//! whose address space is part of its type. Each kernel here moves one
//! through a boundary where a generic `*mut T` would lose that fact, or where
//! the lowering has to rebuild the value from parts:
//!
//! - a `from_raw` round trip and a negative `offset` on a static array,
//! - a closure that captured one, called through a generic helper,
//! - `addr_u32` deltas against `cvta_generic_to_shared_u32(get_raw())`,
//! - `&Region<T>` and `&mut SharedPtr<T>` through non-inlined helpers,
//! - a non-inlined helper that returns one,
//! - `[SharedPtr<T>; 2]` passed by value,
//! - a struct pointee and a `cast` to its first field,
//! - `f64` and `u16` views of a dynamic byte pool,
//! - `Option<SharedPtr<u32>>` returned and passed,
//! - `SharedPtr` values stored in shared memory itself,
//! - `transmute` to `usize` and back.
//!
//! The host checks every value and prints one SUCCESS line. Structs that mix
//! a `SharedPtr` with a narrower field stay out of this example: on the
//! libNVVM route a shared pointer is 4 bytes wide, and such a struct is not
//! yet supported in arrays or in shared memory.
//!
//! Usage:
//!   cargo oxide run shared_ptr_api

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::{DynamicSharedArray, SharedArray, SharedPtr, cuda_module, kernel, thread};

const THREADS: usize = 64;

/// A pointee with two fields; a `cast` to `f32` reaches the first.
#[derive(Clone, Copy)]
#[repr(C)]
struct Pair {
    a: f32,
    b: f32,
}

/// A typed shared pointer next to an element offset, read through `&Region<T>`.
#[derive(Clone, Copy)]
struct Region<T> {
    base: SharedPtr<T>,
    offset: usize,
}

static mut TILE_F32: SharedArray<f32, THREADS> = SharedArray::UNINIT;
static mut TILE_U32: SharedArray<u32, THREADS> = SharedArray::UNINIT;
static mut PAIRS: SharedArray<Pair, THREADS> = SharedArray::UNINIT;
static mut SLOTS: SharedArray<SharedPtr<u32>, THREADS> = SharedArray::UNINIT;

/// Stores through a typed shared pointer inside a non-inlined generic helper.
///
/// # Safety
/// `p.add(i)` must be in bounds of live shared memory.
#[inline(never)]
unsafe fn smem_fill<T>(p: SharedPtr<T>, i: usize, v: T) {
    unsafe { p.add(i).as_ptr().write(v) }
}

/// Loads through a typed shared pointer inside a non-inlined generic helper.
///
/// # Safety
/// `p.add(i)` must be in bounds of initialized shared memory.
#[inline(never)]
unsafe fn smem_load<T: Copy>(p: SharedPtr<T>, i: usize) -> T {
    unsafe { p.add(i).as_ptr().read() }
}

/// Returns one of two typed shared pointers. The return type lowers to
/// `ptr addrspace(3)`.
#[inline(never)]
fn pick<T>(a: SharedPtr<T>, b: SharedPtr<T>, second: bool) -> SharedPtr<T> {
    if second { b } else { a }
}

/// Calls a closure that captured a `SharedPtr` by value.
#[inline(never)]
fn apply<F: Fn(usize)>(f: F, i: usize) {
    f(i)
}

/// Reads through `&Region<T>`: a deref, a field of `Region`, then the pointer.
///
/// # Safety
/// `r.base.add(r.offset + i)` must be in bounds of initialized shared memory.
#[inline(never)]
unsafe fn read_region<T: Copy>(r: &Region<T>, i: usize) -> T {
    unsafe { r.base.add(r.offset + i).as_ptr().read() }
}

/// Advances a `SharedPtr` in place through `&mut SharedPtr<T>`.
///
/// # Safety
/// The advanced pointer must stay inside the same allocation.
#[inline(never)]
unsafe fn bump<T>(slot: &mut SharedPtr<T>, by: usize) {
    unsafe { *slot = slot.add(by) }
}

/// Stores through one of two `SharedPtr`s passed in an array by value.
///
/// # Safety
/// `ptrs[which].add(i)` must be in bounds of live shared memory.
#[inline(never)]
unsafe fn store_via_array<T>(ptrs: [SharedPtr<T>; 2], which: usize, i: usize, v: T) {
    unsafe { ptrs[which].add(i).as_ptr().write(v) }
}

/// Stores through an `Option<SharedPtr<u32>>` and reports whether it did.
///
/// # Safety
/// A `Some` pointer must be in bounds of live shared memory at `i`.
#[inline(never)]
unsafe fn store_opt(p: Option<SharedPtr<u32>>, i: usize, v: u32) -> u32 {
    match p {
        Some(p) => {
            unsafe { p.add(i).as_ptr().write(v) };
            1
        }
        None => 0,
    }
}

/// Wraps a `SharedPtr` in `Some` or drops it, in a non-inlined helper.
#[inline(never)]
fn maybe(p: SharedPtr<u32>, take: bool) -> Option<SharedPtr<u32>> {
    if take { Some(p) } else { None }
}

/// A `SharedPtr<u32>` as its address bits.
#[inline(never)]
fn to_bits(p: SharedPtr<u32>) -> usize {
    // SAFETY: `SharedPtr<T>` is `repr(transparent)` over a thin pointer.
    unsafe { core::mem::transmute::<SharedPtr<u32>, usize>(p) }
}

/// Address bits back to a `SharedPtr<u32>`.
///
/// # Safety
/// `bits` must come from `to_bits` on a live `SharedPtr<u32>`.
#[inline(never)]
unsafe fn from_bits(bits: usize) -> SharedPtr<u32> {
    unsafe { core::mem::transmute::<usize, SharedPtr<u32>>(bits) }
}

#[cuda_module]
mod kernels {
    use super::*;

    /// A static array through `shared_ptr`, a `from_raw` round trip, and a
    /// negative `offset` back to this thread's slot.
    #[kernel]
    pub fn static_round_trip(out: *mut f32) {
        let tid = thread::threadIdx_x() as usize;
        // SAFETY: TILE_F32 is a `static mut SharedArray` of this kernel; each
        // thread writes and reads its own slot, with a barrier between.
        unsafe {
            let tile = SharedArray::shared_ptr(&raw mut TILE_F32);
            smem_fill(tile, tid, tid as f32 + 0.5);
            thread::sync_threads();
            let again = SharedPtr::from_raw(tile.as_ptr());
            let last = again.add(THREADS - 1);
            let mine = last.offset(-((THREADS - 1 - tid) as isize));
            *out.add(tid) = smem_load(mine, 0);
        }
    }

    /// A closure that captured a `SharedPtr`, called through a generic
    /// non-inlined helper.
    #[kernel]
    pub fn closure_capture(out: *mut u32) {
        let tid = thread::threadIdx_x() as usize;
        // SAFETY: TILE_U32 is a `static mut SharedArray` of this kernel.
        let tile = unsafe { SharedArray::shared_ptr(&raw mut TILE_U32) };
        // SAFETY: each thread writes its own slot.
        let store = move |i: usize| unsafe { tile.add(i).as_ptr().write(i as u32 * 3) };
        apply(store, tid);
        thread::sync_threads();
        // SAFETY: the slot was written before the barrier.
        unsafe { *out.add(tid) = tile.add(tid).as_ptr().read() };
    }

    /// `addr_u32` deltas within the dynamic pool, and equality with
    /// `cvta_generic_to_shared_u32` on the pool's raw pointer.
    #[kernel]
    pub fn addr_u32_deltas(out: *mut u32) {
        let tid = thread::threadIdx_x() as usize;
        let base = DynamicSharedArray::<u32, 16>::shared_ptr();
        let at64 = DynamicSharedArray::<u32, 16>::shared_ptr_at(64);
        // SAFETY: `get_raw` points into the dynamic pool; thread 0 writes the
        // four output slots.
        unsafe {
            let raw = cvta_generic_to_shared_u32(DynamicSharedArray::<u32, 16>::get_raw());
            if tid == 0 {
                *out.add(0) = at64.addr_u32() - base.addr_u32();
                *out.add(1) = base.add(3).addr_u32() - base.addr_u32();
                *out.add(2) = (base.addr_u32() == raw) as u32;
                *out.add(3) = base.cast::<u8>().byte_add(7).addr_u32() - base.addr_u32();
            }
        }
    }

    /// `SharedPtr` behind references: `&Region<T>` and `&mut SharedPtr<T>`.
    #[kernel]
    pub fn behind_references(out: *mut u32) {
        let tid = thread::threadIdx_x() as usize;
        let pool = DynamicSharedArray::<u32, 16>::shared_ptr();
        // SAFETY: the launch supplies 1024 dynamic shared bytes, enough for
        // the 64 `u32` slots; every slot is written before the barrier.
        unsafe {
            pool.add(tid).as_ptr().write(100 + tid as u32);
            thread::sync_threads();
            let region = Region {
                base: pool,
                offset: 0,
            };
            let a = read_region(&region, tid);
            let mut cursor = pool;
            bump(&mut cursor, tid);
            let b = cursor.as_ptr().read();
            let shifted = Region {
                base: pool,
                offset: THREADS / 2,
            };
            let c = read_region(&shifted, tid % (THREADS / 2));
            *out.add(tid) = a + b + c;
        }
    }

    /// A non-inlined helper that returns a `SharedPtr` picks between a static
    /// array and the dynamic pool.
    #[kernel]
    pub fn returned_pointer(out: *mut f32) {
        let tid = thread::threadIdx_x() as usize;
        let dynp = DynamicSharedArray::<f32, 16>::shared_ptr();
        // SAFETY: TILE_F32 is a `static mut SharedArray` of this kernel and
        // the pool holds 64 `f32`; each thread writes its own slot in both.
        unsafe {
            let tile = SharedArray::shared_ptr(&raw mut TILE_F32);
            smem_fill(tile, tid, 1000.0 + tid as f32);
            smem_fill(dynp, tid, 2000.0 + tid as f32);
            thread::sync_threads();
            let chosen = pick(tile, dynp, tid % 2 == 1);
            *out.add(tid) = smem_load(chosen, tid);
        }
    }

    /// `[SharedPtr<T>; 2]` passed by value to a non-inlined helper.
    #[kernel]
    pub fn array_by_value(out: *mut u32) {
        let tid = thread::threadIdx_x() as usize;
        let pool = DynamicSharedArray::<u32, 16>::shared_ptr();
        // SAFETY: TILE_U32 is a `static mut SharedArray` of this kernel and
        // the pool holds 64 `u32`; each thread writes its own slot in one.
        unsafe {
            let tile = SharedArray::shared_ptr(&raw mut TILE_U32);
            store_via_array([tile, pool], tid % 2, tid, 7 + tid as u32);
            thread::sync_threads();
            let v = if tid.is_multiple_of(2) {
                tile.add(tid).as_ptr().read()
            } else {
                pool.add(tid).as_ptr().read()
            };
            *out.add(tid) = v;
        }
    }

    /// A struct pointee, and a `cast` to its first field.
    #[kernel]
    pub fn struct_pointee(out: *mut f32) {
        let tid = thread::threadIdx_x() as usize;
        // SAFETY: PAIRS is a `static mut SharedArray` of this kernel; each
        // thread writes and reads its own element.
        unsafe {
            let pairs = SharedArray::shared_ptr(&raw mut PAIRS);
            smem_fill(
                pairs,
                tid,
                Pair {
                    a: tid as f32,
                    b: 2.0 * tid as f32,
                },
            );
            thread::sync_threads();
            let p = smem_load(pairs, tid);
            let first: f32 = smem_load(pairs.add(tid).cast::<f32>(), 0);
            *out.add(tid) = p.a + p.b + first;
        }
    }

    /// `byte_add` into a dynamic byte pool, viewed as `f64` and as `u16`.
    #[kernel]
    pub fn byte_views(out: *mut f64) {
        let tid = thread::threadIdx_x() as usize;
        let pool = DynamicSharedArray::<u8, 16>::shared_ptr();
        // SAFETY: the pool holds 1024 bytes; the `f64` view uses the first
        // 512 and the `u16` view the next 128, each thread its own slot.
        unsafe {
            let d = pool.byte_add(8 * tid).cast::<f64>();
            smem_fill(d, 0, tid as f64 * 0.25);
            let h = pool.byte_add(8 * THREADS + 2 * tid).cast::<u16>();
            smem_fill(h, 0, tid as u16 + 7);
            thread::sync_threads();
            let dv: f64 = smem_load(pool.cast::<f64>(), tid);
            let hv: u16 = smem_load(pool.byte_add(8 * THREADS).cast::<u16>(), tid);
            *out.add(tid) = dv + hv as f64;
        }
    }

    /// `Option<SharedPtr<u32>>` returned from and passed into non-inlined
    /// helpers.
    #[kernel]
    pub fn option_pointer(out: *mut u32) {
        let tid = thread::threadIdx_x() as usize;
        // SAFETY: TILE_U32 is a `static mut SharedArray` of this kernel; each
        // thread writes and reads its own slot.
        unsafe {
            let tile = SharedArray::shared_ptr(&raw mut TILE_U32);
            let took = store_opt(maybe(tile, true), tid, 5 + tid as u32);
            let skipped = store_opt(maybe(tile, false), tid, 0);
            thread::sync_threads();
            *out.add(tid) = tile.add(tid).as_ptr().read() + 10 * took + 100 * skipped;
        }
    }

    /// `SharedPtr` values stored in shared memory: a `SharedArray` of them.
    #[kernel]
    pub fn stored_in_shared(out: *mut u32) {
        let tid = thread::threadIdx_x() as usize;
        // SAFETY: TILE_U32 and SLOTS are `static mut SharedArray`s of this
        // kernel; each thread writes its own slot in both and reads one slot
        // after the barrier.
        unsafe {
            let tile = SharedArray::shared_ptr(&raw mut TILE_U32);
            let slots = SharedArray::shared_ptr(&raw mut SLOTS);
            tile.add(tid).as_ptr().write(tid as u32 * 11);
            slots.add(tid).as_ptr().write(tile.add(THREADS - 1 - tid));
            thread::sync_threads();
            let p: SharedPtr<u32> = slots.add(tid).as_ptr().read();
            *out.add(tid) = p.as_ptr().read();
        }
    }

    /// `transmute` to `usize` and back, in non-inlined helpers. The high word
    /// records whether the bits of this thread's slot sit `4 * tid` bytes
    /// past the bits of the array base.
    #[kernel]
    pub fn transmuted(out: *mut u64) {
        let tid = thread::threadIdx_x() as usize;
        // SAFETY: TILE_U32 is a `static mut SharedArray` of this kernel; each
        // thread writes and reads its own slot.
        unsafe {
            let tile = SharedArray::shared_ptr(&raw mut TILE_U32);
            let bits = to_bits(tile.add(tid));
            let back = from_bits(bits);
            back.as_ptr().write(tid as u32 * 13);
            thread::sync_threads();
            let v = tile.add(tid).as_ptr().read();
            let strided = bits == to_bits(tile) + 4 * tid;
            *out.add(tid) = v as u64 | ((strided as u64) << 32);
        }
    }
}

/// Counts the mismatches of one kernel and prints its line.
fn check(name: &str, bad: usize, mismatches: &mut usize, kernels: &mut usize) {
    if bad == 0 {
        println!("  ok    {name}");
    } else {
        println!("  BAD   {name}: {bad} mismatches");
    }
    *mismatches += bad;
    *kernels += 1;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: 1024,
    };
    let mut mismatches = 0usize;
    let mut kernels = 0usize;
    println!("=== SharedPtr API ({THREADS} threads) ===");

    let out = DeviceBuffer::<f32>::zeroed(&stream, THREADS)?;
    // SAFETY: one block of THREADS threads; every output buffer holds one
    // slot per thread (four slots for addr_u32_deltas, written by thread 0).
    unsafe { module.static_round_trip(&stream, config, out.cu_deviceptr() as *mut f32)? };
    let got = out.to_host_vec(&stream)?;
    check(
        "static_round_trip",
        (0..THREADS).filter(|&i| got[i] != i as f32 + 0.5).count(),
        &mut mismatches,
        &mut kernels,
    );

    let out = DeviceBuffer::<u32>::zeroed(&stream, THREADS)?;
    unsafe { module.closure_capture(&stream, config, out.cu_deviceptr() as *mut u32)? };
    let got = out.to_host_vec(&stream)?;
    check(
        "closure_capture",
        (0..THREADS).filter(|&i| got[i] != i as u32 * 3).count(),
        &mut mismatches,
        &mut kernels,
    );

    let out = DeviceBuffer::<u32>::zeroed(&stream, 4)?;
    unsafe { module.addr_u32_deltas(&stream, config, out.cu_deviceptr() as *mut u32)? };
    let got = out.to_host_vec(&stream)?;
    let want = [64u32, 12, 1, 7];
    check(
        "addr_u32_deltas",
        (0..4).filter(|&i| got[i] != want[i]).count(),
        &mut mismatches,
        &mut kernels,
    );

    let out = DeviceBuffer::<u32>::zeroed(&stream, THREADS)?;
    unsafe { module.behind_references(&stream, config, out.cu_deviceptr() as *mut u32)? };
    let got = out.to_host_vec(&stream)?;
    check(
        "behind_references",
        (0..THREADS)
            .filter(|&i| {
                let t = i as u32;
                let half = (THREADS / 2) as u32;
                got[i] != (100 + t) + (100 + t) + (100 + half + t % half)
            })
            .count(),
        &mut mismatches,
        &mut kernels,
    );

    let out = DeviceBuffer::<f32>::zeroed(&stream, THREADS)?;
    unsafe { module.returned_pointer(&stream, config, out.cu_deviceptr() as *mut f32)? };
    let got = out.to_host_vec(&stream)?;
    check(
        "returned_pointer",
        (0..THREADS)
            .filter(|&i| {
                let base = if i % 2 == 1 { 2000.0 } else { 1000.0 };
                got[i] != base + i as f32
            })
            .count(),
        &mut mismatches,
        &mut kernels,
    );

    let out = DeviceBuffer::<u32>::zeroed(&stream, THREADS)?;
    unsafe { module.array_by_value(&stream, config, out.cu_deviceptr() as *mut u32)? };
    let got = out.to_host_vec(&stream)?;
    check(
        "array_by_value",
        (0..THREADS).filter(|&i| got[i] != 7 + i as u32).count(),
        &mut mismatches,
        &mut kernels,
    );

    let out = DeviceBuffer::<f32>::zeroed(&stream, THREADS)?;
    unsafe { module.struct_pointee(&stream, config, out.cu_deviceptr() as *mut f32)? };
    let got = out.to_host_vec(&stream)?;
    check(
        "struct_pointee",
        (0..THREADS).filter(|&i| got[i] != 4.0 * i as f32).count(),
        &mut mismatches,
        &mut kernels,
    );

    let out = DeviceBuffer::<f64>::zeroed(&stream, THREADS)?;
    unsafe { module.byte_views(&stream, config, out.cu_deviceptr() as *mut f64)? };
    let got = out.to_host_vec(&stream)?;
    check(
        "byte_views",
        (0..THREADS)
            .filter(|&i| got[i] != i as f64 * 0.25 + (i as f64 + 7.0))
            .count(),
        &mut mismatches,
        &mut kernels,
    );

    let out = DeviceBuffer::<u32>::zeroed(&stream, THREADS)?;
    unsafe { module.option_pointer(&stream, config, out.cu_deviceptr() as *mut u32)? };
    let got = out.to_host_vec(&stream)?;
    check(
        "option_pointer",
        (0..THREADS).filter(|&i| got[i] != 15 + i as u32).count(),
        &mut mismatches,
        &mut kernels,
    );

    let out = DeviceBuffer::<u32>::zeroed(&stream, THREADS)?;
    unsafe { module.stored_in_shared(&stream, config, out.cu_deviceptr() as *mut u32)? };
    let got = out.to_host_vec(&stream)?;
    check(
        "stored_in_shared",
        (0..THREADS)
            .filter(|&i| got[i] != (THREADS - 1 - i) as u32 * 11)
            .count(),
        &mut mismatches,
        &mut kernels,
    );

    let out = DeviceBuffer::<u64>::zeroed(&stream, THREADS)?;
    unsafe { module.transmuted(&stream, config, out.cu_deviceptr() as *mut u64)? };
    let got = out.to_host_vec(&stream)?;
    check(
        "transmuted",
        (0..THREADS)
            .filter(|&i| got[i] != (i as u64 * 13) | (1u64 << 32))
            .count(),
        &mut mismatches,
        &mut kernels,
    );

    if mismatches == 0 {
        println!("SUCCESS: shared_ptr_api, {kernels} kernels, 0 mismatches");
        Ok(())
    } else {
        println!("FAIL: shared_ptr_api, {mismatches} mismatches across {kernels} kernels");
        std::process::exit(1)
    }
}
