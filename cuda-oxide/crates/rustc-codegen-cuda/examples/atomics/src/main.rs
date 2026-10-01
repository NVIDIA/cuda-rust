/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::approx_constant, internal_features)]
#![feature(core_intrinsics)]

//! Unified Atomics Test Example
//!
//! Comprehensive test suite for sound atomic operations:
//!
//! **Phase 1 (DeviceAtomicU32/I32, load/store/fetch_add/CAS):**
//!  1. `atomic_fetch_add_test` -- DeviceAtomicU32 fetch_add (Relaxed)
//!  2. `atomic_load_store_test` -- DeviceAtomicU32 load/store (Acquire/Release)
//!  3. `atomic_cas_test` -- DeviceAtomicU32 compare_exchange (AcqRel)
//!  4. `atomic_fetch_add_acqrel_test` -- fence-splitting workaround (AcqRel)
//!  5. `atomic_fetch_add_seqcst_test` -- fence.sc pattern (SeqCst)
//!  6. `atomic_i32_test` -- DeviceAtomicI32 fetch_add + compare_exchange
//!  7. `atomic_multiblock_test` -- device-scope atomics across CTAs
//!
//! **Phase 2 (new types + new RMW ops):**
//!  8. `atomic_u64_fetch_add_test` -- DeviceAtomicU64 fetch_add (64-bit)
//!  9. `atomic_i64_test` -- DeviceAtomicI64 fetch_add + compare_exchange
//! 10. `atomic_fetch_sub_test` -- DeviceAtomicU32 fetch_sub
//! 11. `atomic_bitwise_test` -- fetch_and, fetch_or, fetch_xor
//! 12. `atomic_swap_test` -- DeviceAtomicU32 swap (exchange)
//! 13. `atomic_minmax_test` -- DeviceAtomicI32 fetch_min / fetch_max (signed)
//! 14. `atomic_f32_fetch_add_test` -- DeviceAtomicF32 fetch_add (float)
//! 15. `atomic_f64_fetch_add_test` -- DeviceAtomicF64 fetch_add (64-bit float)
//! 16. `atomic_f32_swap_test` -- DeviceAtomicF32 swap (float exchange)
//! 17. `atomic_unsigned_minmax_test` -- DeviceAtomicU32 fetch_min/max (UMin/UMax)
//! 18. `atomic_block_scope_test` -- BlockAtomicU32 fetch_add (.cta scope, Relaxed)
//! 19. `atomic_block_scope_acqrel_test` -- BlockAtomicU32 fetch_add (.cta scope, AcqRel)
//! 20. `core_atomic_fetch_add_test` -- core::sync::atomic::AtomicU32 (system scope)
//! 21. `core_atomic_ptr_test` -- core::sync::atomic::AtomicPtr load/store/swap/CAS
//!
//! 22. `core_atomic_local_test` -- private pointer/integer storage through helpers
//!
//! 23. `core_atomic_ptr_shared_value_test` -- AtomicPtr round-trip of a shared-memory pointer value
//! 24. `core_atomic_ptr_shared_storage_test` -- AtomicPtr storage backed by shared memory
//!
//! `core_atomic_ordering_probe` adds compile-only core intrinsic ordering coverage.
//!
//! Build and run with:
//!   cargo oxide run atomics

use core::sync::atomic::Ordering;
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::atomic::{
    AtomicOrdering, BlockAtomicU32, DeviceAtomicF32, DeviceAtomicF64, DeviceAtomicI32,
    DeviceAtomicI64, DeviceAtomicU32, DeviceAtomicU64,
};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    // Shared atomic storage arrives through raw writable pointers. The unsafe
    // launches supply live, aligned allocations and exclude non-atomic accesses
    // until completion. No immutable slice or disjoint-access promise covers
    // a counter that multiple threads update.

    /// Test 1: Atomic fetch_add -- every thread atomically increments a counter.
    ///
    /// After N threads run, counter[0] should equal N.
    /// This tests the atomicrmw path with the fence-splitting workaround.
    #[kernel]
    pub fn atomic_fetch_add_test(counter: *mut u32, mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();

        // Get an DeviceAtomicU32 reference to counter[0] (shared access via interior mutability)
        let atomic_counter = unsafe { DeviceAtomicU32::from_ptr(counter) };

        // Each thread atomically increments the counter and gets the old value
        let old = atomic_counter.fetch_add(1, AtomicOrdering::Relaxed);

        // Store the old value so we can verify uniqueness on the host
        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = old;
        }
    }

    /// Test 2: Atomic load/store -- thread 0 stores a value, all threads load it.
    ///
    /// Uses Acquire/Release ordering for proper visibility:
    /// - Thread 0: store with Release (makes the write visible)
    /// - Other threads: load with Acquire (sees the Release'd write)
    #[kernel]
    pub fn atomic_load_store_test(flag: *mut u32, mut out: DisjointSlice<u32>) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        // Get an DeviceAtomicU32 reference to flag[0] (shared access via interior mutability)
        let atomic_flag = unsafe { DeviceAtomicU32::from_ptr(flag) };

        // Thread 0 stores a sentinel value
        if tid == 0 {
            atomic_flag.store(42, AtomicOrdering::Release);
        }

        // Barrier ensures all threads see the store
        thread::sync_threads();

        // All threads load the value
        let val = atomic_flag.load(AtomicOrdering::Acquire);
        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = val;
        }
    }

    /// Test 3: Atomic compare_exchange -- only one thread wins the CAS race.
    ///
    /// All threads try to CAS 0 -> their_tid. Exactly one succeeds.
    #[kernel]
    pub fn atomic_cas_test(winner: *mut u32, mut out: DisjointSlice<u32>) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        // Get an DeviceAtomicU32 reference to winner[0] (shared access via interior mutability)
        let atomic_winner = unsafe { DeviceAtomicU32::from_ptr(winner) };

        // Try to be the first thread to swap 0 -> (tid + 1)
        // We use tid+1 so that thread 0's success value (1) differs from the
        // initial value (0).
        let result = atomic_winner.compare_exchange(
            0,
            tid + 1,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Relaxed,
        );

        if let Some(out_elem) = out.get_mut(gid) {
            match result {
                Ok(_old) => {
                    // This thread won the race
                    *out_elem = 1;
                }
                Err(_old) => {
                    // Another thread already swapped
                    *out_elem = 0;
                }
            }
        }
    }

    /// Test 4: Atomic fetch_add with AcqRel ordering -- exercises fence-splitting workaround.
    ///
    /// The LLVM NVPTX backend drops orderings on atomicrmw (fix in LLVM 23).
    /// We work around this by emitting: fence release + atomicrmw monotonic + fence acquire.
    /// This test verifies that path produces correct results.
    #[kernel]
    pub fn atomic_fetch_add_acqrel_test(counter: *mut u32, mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();

        let atomic_counter = unsafe { DeviceAtomicU32::from_ptr(counter) };

        // AcqRel triggers: fence release + atomicrmw monotonic + fence acquire
        let old = atomic_counter.fetch_add(1, AtomicOrdering::AcqRel);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = old;
        }
    }

    /// Test 5: Atomic fetch_add with SeqCst ordering -- exercises fence.sc pattern.
    ///
    /// SeqCst emits: fence seq_cst + atomicrmw monotonic + fence seq_cst.
    #[kernel]
    pub fn atomic_fetch_add_seqcst_test(counter: *mut u32, mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();

        let atomic_counter = unsafe { DeviceAtomicU32::from_ptr(counter) };

        // SeqCst triggers: fence seq_cst + atomicrmw monotonic + fence seq_cst
        let old = atomic_counter.fetch_add(1, AtomicOrdering::SeqCst);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = old;
        }
    }

    /// Test 6: DeviceAtomicI32 -- exercises signed atomic type with fetch_add and compare_exchange.
    ///
    /// Verifies that signed atomics work correctly (i32 vs u32 in LLVM IR).
    #[kernel]
    pub fn atomic_i32_test(counter: *mut i32, cas_target: *mut i32, mut out: DisjointSlice<i32>) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_counter = unsafe { DeviceAtomicI32::from_ptr(counter) };
        let atomic_cas = unsafe { DeviceAtomicI32::from_ptr(cas_target) };

        // All threads increment the counter
        let _old = atomic_counter.fetch_add(1, AtomicOrdering::Relaxed);

        // Every thread reaches the barrier, including the thread doing the CAS.
        let succeeded = if tid == 0 {
            let result = atomic_cas.compare_exchange(
                0,
                -42,
                AtomicOrdering::AcqRel,
                AtomicOrdering::Relaxed,
            );
            result.is_ok() as i32
        } else {
            0
        };
        thread::sync_threads();
        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = if tid == 0 {
                succeeded
            } else {
                atomic_cas.load(AtomicOrdering::Acquire)
            };
        }
    }

    /// Test 7: Multi-block fetch_add -- exercises device-scope atomics across CTAs.
    ///
    /// Uses 4 blocks x 64 threads = 256 threads total. The counter must reach 256,
    /// proving that atomics work across different thread blocks (CTAs).
    #[kernel]
    pub fn atomic_multiblock_test(counter: *mut u32, mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();

        let atomic_counter = unsafe { DeviceAtomicU32::from_ptr(counter) };

        let old = atomic_counter.fetch_add(1, AtomicOrdering::Relaxed);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = old;
        }
    }

    /// Test 8: DeviceAtomicU64 fetch_add -- 64-bit unsigned atomics.
    ///
    /// Same pattern as test 1 but with u64 to verify 64-bit type plumbing.
    #[kernel]
    pub fn atomic_u64_fetch_add_test(counter: *mut u64, mut out: DisjointSlice<u64>) {
        let gid = thread::index_1d();

        let atomic_counter = unsafe { DeviceAtomicU64::from_ptr(counter) };

        let old = atomic_counter.fetch_add(1, AtomicOrdering::Relaxed);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = old;
        }
    }

    /// Test 9: DeviceAtomicI64 fetch_add + compare_exchange -- 64-bit signed atomics.
    ///
    /// Verifies i64 path: fetch_add increments, CAS swaps 0 -> -100.
    #[kernel]
    pub fn atomic_i64_test(counter: *mut i64, cas_target: *mut i64, mut out: DisjointSlice<i64>) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_counter = unsafe { DeviceAtomicI64::from_ptr(counter) };
        let atomic_cas = unsafe { DeviceAtomicI64::from_ptr(cas_target) };

        // All threads increment the counter
        let _old = atomic_counter.fetch_add(1, AtomicOrdering::Relaxed);

        let succeeded = if tid == 0 {
            let result = atomic_cas.compare_exchange(
                0,
                -100,
                AtomicOrdering::AcqRel,
                AtomicOrdering::Relaxed,
            );
            result.is_ok() as i64
        } else {
            0
        };
        thread::sync_threads();
        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = if tid == 0 {
                succeeded
            } else {
                atomic_cas.load(AtomicOrdering::Acquire)
            };
        }
    }

    /// Test 10: DeviceAtomicU32 fetch_sub -- subtraction RMW op.
    ///
    /// Start counter at N, each thread subtracts 1. Result should be 0.
    #[kernel]
    pub fn atomic_fetch_sub_test(counter: *mut u32, mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();

        let atomic_counter = unsafe { DeviceAtomicU32::from_ptr(counter) };

        let old = atomic_counter.fetch_sub(1, AtomicOrdering::Relaxed);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = old;
        }
    }

    /// Test 11: DeviceAtomicU32 bitwise ops -- fetch_and, fetch_or, fetch_xor.
    ///
    /// Three separate counters:
    /// - or_acc: starts at 0, each thread ORs in (1 << (tid % 32))
    /// - and_acc: starts at 0xFFFFFFFF, thread 0 ANDs with 0x0000FFFF
    /// - xor_acc: starts at 0, each thread XORs with 1 (odd/even toggle)
    #[kernel]
    pub fn atomic_bitwise_test(
        or_acc: *mut u32,
        and_acc: *mut u32,
        xor_acc: *mut u32,
        mut out: DisjointSlice<u32>,
    ) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_or = unsafe { DeviceAtomicU32::from_ptr(or_acc) };
        let atomic_and = unsafe { DeviceAtomicU32::from_ptr(and_acc) };
        let atomic_xor = unsafe { DeviceAtomicU32::from_ptr(xor_acc) };

        // Each thread sets its bit in the OR accumulator
        let bit = 1u32 << (tid % 32);
        atomic_or.fetch_or(bit, AtomicOrdering::Relaxed);

        // Thread 0 masks out the upper 16 bits via AND
        if tid == 0 {
            atomic_and.fetch_and(0x0000FFFF, AtomicOrdering::Relaxed);
        }

        // Every thread XORs with 1 (toggles bit 0)
        atomic_xor.fetch_xor(1, AtomicOrdering::Relaxed);

        // Store tid so we can verify the kernel ran
        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = tid;
        }
    }

    /// Test 12: DeviceAtomicU32 swap -- atomic exchange.
    ///
    /// Thread 0 swaps in a sentinel value (0xDEADBEEF), gets back the old value (0).
    #[kernel]
    pub fn atomic_swap_test(target: *mut u32, mut out: DisjointSlice<u32>) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_target = unsafe { DeviceAtomicU32::from_ptr(target) };

        let old = if tid == 0 {
            atomic_target.swap(0xDEADBEEF, AtomicOrdering::AcqRel)
        } else {
            0
        };
        thread::sync_threads();
        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = if tid == 0 {
                old
            } else {
                atomic_target.load(AtomicOrdering::Acquire)
            };
        }
    }

    /// Test 13: DeviceAtomicI32 fetch_min / fetch_max -- signed min/max RMW.
    ///
    /// All threads atomically update min and max accumulators with their
    /// (signed) thread id offset by -128. After 256 threads:
    /// - min should be -128 (thread 0's value)
    /// - max should be 127 (thread 255's value)
    #[kernel]
    pub fn atomic_minmax_test(min_acc: *mut i32, max_acc: *mut i32, mut out: DisjointSlice<i32>) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_min = unsafe { DeviceAtomicI32::from_ptr(min_acc) };
        let atomic_max = unsafe { DeviceAtomicI32::from_ptr(max_acc) };

        // Each thread contributes a signed value: tid - 128
        // Range: -128 to +127 for 256 threads
        let val = tid as i32 - 128;

        atomic_min.fetch_min(val, AtomicOrdering::Relaxed);
        atomic_max.fetch_max(val, AtomicOrdering::Relaxed);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = val;
        }
    }

    /// Test 14: DeviceAtomicF32 fetch_add -- floating-point atomic add.
    ///
    /// Each thread adds 1.0 to a counter. After N threads, should equal N.0.
    /// This tests the FAdd RMW kind path (atomicrmw fadd).
    #[kernel]
    pub fn atomic_f32_fetch_add_test(counter: *mut f32, mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();

        let atomic_counter = unsafe { DeviceAtomicF32::from_ptr(counter) };

        // Each thread adds 1.0
        let _old = atomic_counter.fetch_add(1.0, AtomicOrdering::Relaxed);

        // Store 1 to indicate this thread ran
        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = 1;
        }
    }

    /// Test 15: DeviceAtomicF64 fetch_add -- 64-bit floating-point atomic add.
    ///
    /// Same pattern as test 14 but with f64. Requires sm_60+ (we target sm_80+).
    #[kernel]
    pub fn atomic_f64_fetch_add_test(counter: *mut f64, mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();

        let atomic_counter = unsafe { DeviceAtomicF64::from_ptr(counter) };

        let _old = atomic_counter.fetch_add(1.0, AtomicOrdering::Relaxed);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = 1;
        }
    }

    /// Test 16: DeviceAtomicF32 swap -- float atomic exchange.
    ///
    /// Thread 0 swaps in 3.14, gets back 0.0. Other threads read 3.14 after barrier.
    /// This verifies atomicrmw xchg works on float types.
    #[kernel]
    pub fn atomic_f32_swap_test(target: *mut f32, mut out: DisjointSlice<u32>) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_target = unsafe { DeviceAtomicF32::from_ptr(target) };

        let old = if tid == 0 {
            atomic_target.swap(3.14, AtomicOrdering::AcqRel)
        } else {
            0.0
        };
        thread::sync_threads();
        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = if tid == 0 {
                (old == 0.0) as u32
            } else {
                (atomic_target.load(AtomicOrdering::Acquire) == 3.14) as u32
            };
        }
    }

    /// Test 17: DeviceAtomicU32 fetch_min / fetch_max -- unsigned min/max (UMin/UMax).
    ///
    /// All threads contribute their tid. After 256 threads:
    /// - min should be 0 (thread 0's value)
    /// - max should be 255 (thread 255's value)
    ///
    /// This specifically tests the UMin/UMax path (unsigned), whereas test 13
    /// tests the Min/Max path (signed).
    #[kernel]
    pub fn atomic_unsigned_minmax_test(
        min_acc: *mut u32,
        max_acc: *mut u32,
        mut out: DisjointSlice<u32>,
    ) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_min = unsafe { DeviceAtomicU32::from_ptr(min_acc) };
        let atomic_max = unsafe { DeviceAtomicU32::from_ptr(max_acc) };

        atomic_min.fetch_min(tid, AtomicOrdering::Relaxed);
        atomic_max.fetch_max(tid, AtomicOrdering::Relaxed);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = tid;
        }
    }

    /// Test 18: BlockAtomicU32 fetch_add -- block-scope (.cta) atomics.
    ///
    /// Uses BlockAtomicU32 which emits syncscope("block") → `.cta` in PTX.
    /// Since we launch a single block, block scope is correct here.
    /// The counter should reach N just like device-scope, but with cheaper
    /// coherence (block scope only guarantees visibility within the CTA).
    #[kernel]
    pub fn atomic_block_scope_test(counter: *mut u32, mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();

        let atomic_counter = unsafe { BlockAtomicU32::from_ptr(counter) };

        let old = atomic_counter.fetch_add(1, AtomicOrdering::Relaxed);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = old;
        }
    }

    /// Test 19: BlockAtomicU32 fetch_add with AcqRel -- proves `.cta` scope on fences.
    ///
    /// Same logic as test 18, but uses AcqRel ordering.  Fence-splitting emits:
    ///   fence.acq_rel.cta;  atom.add.u32 ...;  fence.acq_rel.cta;
    /// The `.cta` on the fences confirms the block-scope syncscope is propagated.
    #[kernel]
    pub fn atomic_block_scope_acqrel_test(counter: *mut u32, mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();

        let atomic_counter = unsafe { BlockAtomicU32::from_ptr(counter) };

        let old = atomic_counter.fetch_add(1, AtomicOrdering::AcqRel);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = old;
        }
    }

    /// Test 20: core::sync::atomic::AtomicU32 -- standard library atomics.
    ///
    /// Uses `core::sync::atomic::AtomicU32` with full path (no alias) to avoid
    /// collision with `cuda_device::atomic::DeviceAtomicU32`. Verifies that
    /// std::intrinsics::atomic_xadd is intercepted and lowered to NVVM atomics
    /// with system scope.
    #[kernel]
    pub fn core_atomic_fetch_add_test(counter: *mut u32, mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();

        let atomic_counter = unsafe { core::sync::atomic::AtomicU32::from_ptr(counter) };

        let old = atomic_counter.fetch_add(1, Ordering::Relaxed);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = old;
        }
    }

    /// Compile-only coverage for core intrinsic generic layouts and tuple results.
    #[kernel]
    pub fn core_atomic_ordering_probe(counter: *mut u32, mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();
        let pointer = counter;
        // nightly-2026-08-28 added a `const VOLATILE: bool` generic to the
        // load/store intrinsics; plain atomics pass `false`.
        let current = unsafe {
            core::intrinsics::atomic_load::<u32, { core::intrinsics::AtomicOrdering::Acquire }, false>(
                pointer,
            )
        };
        unsafe {
            core::intrinsics::atomic_store::<
                u32,
                { core::intrinsics::AtomicOrdering::Release },
                false,
            >(pointer, current)
        };
        let swapped = unsafe {
            core::intrinsics::atomic_xchg::<u32, { core::intrinsics::AtomicOrdering::AcqRel }>(
                pointer, current,
            )
        };
        let (observed, succeeded) = unsafe {
            core::intrinsics::atomic_cxchg::<
                u32,
                { core::intrinsics::AtomicOrdering::Release },
                { core::intrinsics::AtomicOrdering::Acquire },
            >(pointer, swapped, swapped + 1)
        };

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = observed + succeeded as u32;
        }
    }

    // The same non-inlined helper receives global and local atomic storage.
    // SAFETY: new_ptr must remain readable and this test alone accesses the cell,
    // initially null, for the duration of the call.
    #[inline(never)]
    unsafe fn pointer_roundtrip(
        atomic_ptr: &core::sync::atomic::AtomicPtr<u16>,
        new_ptr: *mut u16,
        expected_value: u16,
    ) -> bool {
        let loaded = atomic_ptr.load(Ordering::Acquire);
        atomic_ptr.store(new_ptr, Ordering::Release);

        // Store currently contains `new_ptr`; replace it with the original null.
        let swapped = atomic_ptr.swap(loaded, Ordering::AcqRel);

        // Store is null again, so this CAS must succeed and install `new_ptr`.
        let exchanged_ok =
            match atomic_ptr.compare_exchange(loaded, new_ptr, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(old) => old == loaded,
                Err(_) => false,
            };

        let exchanged_err =
            match atomic_ptr.compare_exchange(loaded, loaded, Ordering::SeqCst, Ordering::SeqCst) {
                Err(old) => old == new_ptr,
                Ok(_) => false,
            };
        let final_ptr = atomic_ptr.load(Ordering::SeqCst);
        // SAFETY: the only non-null value stored is this thread's pointer
        // into values, which remains alive and readable for the launch.
        let pointee_ok = final_ptr == new_ptr && unsafe { *final_ptr == expected_value };
        loaded.is_null() && swapped == new_ptr && exchanged_ok && exchanged_err && pointee_ok
    }

    #[inline(never)]
    fn integer_roundtrip(atomic: &core::sync::atomic::AtomicUsize) -> bool {
        let initial = atomic.load(Ordering::Acquire);
        atomic.store(usize::MAX, Ordering::Release);
        let added = atomic.fetch_add(1, Ordering::AcqRel);
        let wrapped = atomic.load(Ordering::SeqCst);
        let subtracted = atomic.fetch_sub(1, Ordering::SeqCst);
        let swapped = atomic.swap(7, Ordering::Relaxed);
        let succeeded = atomic.compare_exchange(7, 9, Ordering::Release, Ordering::Acquire);
        let failed = atomic.compare_exchange(7, 11, Ordering::Relaxed, Ordering::SeqCst);
        initial == 0
            && added == usize::MAX
            && wrapped == 0
            && subtracted == 0
            && swapped == usize::MAX
            && succeeded == Ok(7)
            && failed == Err(9)
            && atomic.load(Ordering::SeqCst) == 9
    }

    /// Test 22: per-thread atomic storage, through the same generic pointer helper.
    #[kernel]
    pub fn core_atomic_local_test(values: &[u16], mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();
        if gid.in_bounds(values.len()) {
            let atomic = core::sync::atomic::AtomicPtr::new(core::ptr::null_mut());
            let integer = core::sync::atomic::AtomicUsize::new(0);
            // SAFETY: the bounds check covers this readable value; the local
            // atomic belongs solely to this thread and starts null.
            let pointer_ok = unsafe {
                pointer_roundtrip(
                    &atomic,
                    values.as_ptr().add(gid.get()) as *mut u16,
                    gid.get() as u16 + 1,
                )
            };
            let integer_ok = integer_roundtrip(&integer);
            if let Some(element) = out.get_mut(gid) {
                *element = (pointer_ok && integer_ok) as u32;
            }
        }
    }

    /// Test 21: core::sync::atomic::AtomicPtr load/store/swap/CAS.
    #[kernel]
    pub fn core_atomic_ptr_test(
        mut storage: DisjointSlice<usize>,
        values: &[u16],
        mut out: DisjointSlice<u32>,
    ) {
        let gid = thread::index_1d();

        if gid.in_bounds(storage.len()) && gid.in_bounds(values.len()) {
            let index = gid.get();

            // SAFETY: the launch gives each thread a distinct index and the
            // bounds check above covers this slot.
            let slot = unsafe { storage.get_unchecked_mut(index) };
            // SAFETY: this thread exclusively owns the slot. On nvptx64 usize
            // and AtomicPtr have the same size/alignment, and zero initializes
            // a null pointer. Access the slot only through the atomic until its
            // borrow ends; no shared reference to ordinary storage is mutated.
            let atomic_ptr = unsafe {
                core::sync::atomic::AtomicPtr::from_ptr((slot as *mut usize).cast::<*mut u16>())
            };
            let new_ptr = unsafe { values.as_ptr().add(index) as *mut u16 };

            // SAFETY: this thread owns the initialized slot and new_ptr points
            // to its live, bounds-checked input element.
            let passed = unsafe { pointer_roundtrip(atomic_ptr, new_ptr, index as u16 + 1) };

            if let Some(out_elem) = out.get_mut(gid) {
                *out_elem = passed as u32;
            }
        }
    }

    /// Test 23: AtomicPtr round-trips a shared-memory pointer value through
    /// load/store/swap/CAS and preserves dereferenceability.
    #[kernel]
    pub fn core_atomic_ptr_shared_value_test(mut storage: DisjointSlice<usize>) {
        static mut SHARED_VALUE: SharedArray<u16, 1> = SharedArray::UNINIT;

        let gid = thread::index_1d();

        if thread::threadIdx_x() == 0 {
            let shared_ptr = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SHARED_VALUE) };
            unsafe { shared_ptr.write(0xBEEF) };
        }

        thread::sync_threads();

        if let Some(slot) = storage.get_mut(gid) {
            let passed = {
                let atomic_ptr = unsafe {
                    core::sync::atomic::AtomicPtr::from_ptr((slot as *mut usize).cast::<*mut u16>())
                };

                let shared_ptr = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SHARED_VALUE) };

                // SAFETY: SHARED_VALUE remains live for the kernel, is initialized
                // before this point, and this thread exclusively owns its atomic slot.
                unsafe { pointer_roundtrip(atomic_ptr, shared_ptr, 0xBEEF) }
            };

            *slot = passed as usize;
        }
    }

    /// Test 24: AtomicPtr storage originates in shared memory.
    #[kernel]
    pub fn core_atomic_ptr_shared_storage_test(values: &[u16], mut out: DisjointSlice<u32>) {
        static mut SHARED_STORAGE: SharedArray<usize, 256> = SharedArray::UNINIT;

        let gid = thread::index_1d();

        if gid.in_bounds(values.len()) && gid.get() < 256 {
            let index = gid.get();

            let storage =
                unsafe { SharedArray::as_raw_mut_ptr(&raw mut SHARED_STORAGE).add(index) };

            // Shared memory is uninitialized. Each thread exclusively owns one
            // pointer-sized slot for the duration of this test.
            unsafe { storage.write(0) };

            let atomic_ptr =
                unsafe { core::sync::atomic::AtomicPtr::from_ptr(storage.cast::<*mut u16>()) };

            let new_ptr = unsafe { values.as_ptr().add(index) as *mut u16 };

            // SAFETY: this thread exclusively owns its shared-memory atomic slot,
            // and new_ptr points to a live bounds-checked input element.
            let passed = unsafe { pointer_roundtrip(atomic_ptr, new_ptr, index as u16 + 1) };

            if let Some(element) = out.get_mut(gid) {
                *element = passed as u32;
            }
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

// Keep the host's writable-buffer authority explicit when passing a device
// address. The caller keeps the allocation alive and synchronizes before any
// host access; this raw pointer must never be dereferenced on the host.
fn atomic_storage<T>(buffer: &mut DeviceBuffer<T>) -> *mut T {
    buffer.cu_deviceptr() as *mut T
}

fn main() {
    println!("=== Unified Atomics Test ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    const N: usize = 256;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (N as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // =========================================================================
    // Test 1: fetch_add
    // =========================================================================
    println!("--- Test 1: atomic_fetch_add_test ---");
    {
        // Allocate a single u32 counter initialized to 0
        let mut counter_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_fetch_add_test(
                (stream).as_ref(),
                cfg,
                atomic_storage(&mut counter_dev),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();

        // The counter should equal N after all threads increment it
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let out_vals = out_dev.to_host_vec(&stream).unwrap();

        if counter_val[0] == N as u32 {
            println!("  Counter final value: {} (expected {})", counter_val[0], N);

            // Verify all old values are unique (each thread got a different value)
            let mut sorted = out_vals.clone();
            sorted.sort();
            sorted.dedup();
            if sorted.len() == N {
                println!("  All {} fetch_add return values are unique", N);
            } else {
                println!(
                    "  FAIL: Only {} unique values (expected {})",
                    sorted.len(),
                    N
                );
                std::process::exit(1);
            }
        } else {
            println!("  FAIL: Counter = {} (expected {})", counter_val[0], N);
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 2: load/store
    // =========================================================================
    println!("\n--- Test 2: atomic_load_store_test ---");
    {
        let mut flag_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_load_store_test(
                (stream).as_ref(),
                cfg,
                atomic_storage(&mut flag_dev),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        let all_42 = result.iter().all(|&x| x == 42);
        if all_42 {
            println!("  All {} threads read 42 after atomic store", N);
        } else {
            let mismatches: Vec<_> = result
                .iter()
                .enumerate()
                .filter(|&(_, &x)| x != 42)
                .take(10)
                .collect();
            println!("  FAIL: {} mismatches: {:?}", mismatches.len(), mismatches);
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 3: compare_exchange
    // =========================================================================
    println!("\n--- Test 3: atomic_cas_test ---");
    {
        let mut winner_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_cas_test(
                (stream).as_ref(),
                cfg,
                atomic_storage(&mut winner_dev),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();

        let winner_val = winner_dev.to_host_vec(&stream).unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        let num_winners: usize = result.iter().filter(|&&x| x == 1).count();
        let winner_tid = winner_val[0];

        if num_winners == 1 && winner_tid >= 1 && winner_tid <= N as u32 {
            println!("  Exactly 1 winner (tid {})", winner_tid - 1);
            println!("  {} threads lost the CAS race", N - 1);
        } else {
            println!(
                "  FAIL: {} winners, winner_val = {} (expected exactly 1 winner)",
                num_winners, winner_tid
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 4: fetch_add with AcqRel (fence-splitting workaround)
    // =========================================================================
    println!("\n--- Test 4: atomic_fetch_add_acqrel_test ---");
    {
        let mut counter_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_fetch_add_acqrel_test(
                (stream).as_ref(),
                cfg,
                atomic_storage(&mut counter_dev),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let out_vals = out_dev.to_host_vec(&stream).unwrap();

        if counter_val[0] == N as u32 {
            let mut sorted = out_vals.clone();
            sorted.sort();
            sorted.dedup();
            if sorted.len() == N {
                println!(
                    "  Counter = {} with AcqRel ordering, all {} values unique",
                    N, N
                );
            } else {
                println!(
                    "  FAIL: Only {} unique values (expected {})",
                    sorted.len(),
                    N
                );
                std::process::exit(1);
            }
        } else {
            println!("  FAIL: Counter = {} (expected {})", counter_val[0], N);
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 5: fetch_add with SeqCst (fence.sc pattern)
    // =========================================================================
    println!("\n--- Test 5: atomic_fetch_add_seqcst_test ---");
    {
        let mut counter_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_fetch_add_seqcst_test(
                (stream).as_ref(),
                cfg,
                atomic_storage(&mut counter_dev),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let out_vals = out_dev.to_host_vec(&stream).unwrap();

        if counter_val[0] == N as u32 {
            let mut sorted = out_vals.clone();
            sorted.sort();
            sorted.dedup();
            if sorted.len() == N {
                println!(
                    "  Counter = {} with SeqCst ordering, all {} values unique",
                    N, N
                );
            } else {
                println!(
                    "  FAIL: Only {} unique values (expected {})",
                    sorted.len(),
                    N
                );
                std::process::exit(1);
            }
        } else {
            println!("  FAIL: Counter = {} (expected {})", counter_val[0], N);
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 6: DeviceAtomicI32 (signed atomics)
    // =========================================================================
    println!("\n--- Test 6: atomic_i32_test ---");
    {
        let mut counter_dev = DeviceBuffer::<i32>::zeroed(&stream, 1).unwrap();
        let mut cas_target_dev = DeviceBuffer::<i32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_i32_test(
                (stream).as_ref(),
                cfg,
                atomic_storage(&mut counter_dev),
                atomic_storage(&mut cas_target_dev),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let cas_val = cas_target_dev.to_host_vec(&stream).unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        let counter_ok = counter_val[0] == N as i32;
        let cas_ok = cas_val[0] == -42;
        assert!(result[1..].iter().all(|&value| value == -42));
        // Thread 0 should have written 1 (CAS succeeded)
        let thread0_ok = result[0] == 1;

        if counter_ok && cas_ok && thread0_ok {
            println!("  i32 counter = {} (expected {})", counter_val[0], N);
            println!("  i32 CAS: 0 -> {} (expected -42)", cas_val[0]);
            println!("  Thread 0 CAS result = {} (1 = success)", result[0]);
        } else {
            println!(
                "  FAIL: counter={} cas={} thread0={}",
                counter_val[0], cas_val[0], result[0]
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 7: Multi-block fetch_add (device-scope across CTAs)
    // =========================================================================
    println!("\n--- Test 7: atomic_multiblock_test ---");
    {
        let multiblock_cfg = LaunchConfig {
            grid_dim: (4, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        let total_threads: usize = 4 * 64;

        let mut counter_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, total_threads).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_multiblock_test(
                (stream).as_ref(),
                multiblock_cfg,
                atomic_storage(&mut counter_dev),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let out_vals = out_dev.to_host_vec(&stream).unwrap();

        if counter_val[0] == total_threads as u32 {
            let mut sorted = out_vals.clone();
            sorted.sort();
            sorted.dedup();
            if sorted.len() == total_threads {
                println!(
                    "  Counter = {} across 4 blocks x 64 threads, all {} values unique",
                    total_threads, total_threads
                );
            } else {
                println!(
                    "  FAIL: Only {} unique values (expected {})",
                    sorted.len(),
                    total_threads
                );
                std::process::exit(1);
            }
        } else {
            println!(
                "  FAIL: Counter = {} (expected {})",
                counter_val[0], total_threads
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 8: DeviceAtomicU64 fetch_add (64-bit unsigned)
    // =========================================================================
    println!("\n--- Test 8: atomic_u64_fetch_add_test ---");
    {
        let mut counter_dev = DeviceBuffer::<u64>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_u64_fetch_add_test(
                (stream).as_ref(),
                cfg,
                atomic_storage(&mut counter_dev),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let out_vals = out_dev.to_host_vec(&stream).unwrap();

        if counter_val[0] == N as u64 {
            let mut sorted = out_vals.clone();
            sorted.sort();
            sorted.dedup();
            if sorted.len() == N {
                println!("  u64 counter = {}, all {} values unique", N, N);
            } else {
                println!(
                    "  FAIL: Only {} unique values (expected {})",
                    sorted.len(),
                    N
                );
                std::process::exit(1);
            }
        } else {
            println!("  FAIL: Counter = {} (expected {})", counter_val[0], N);
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 9: DeviceAtomicI64 fetch_add + compare_exchange (64-bit signed)
    // =========================================================================
    println!("\n--- Test 9: atomic_i64_test ---");
    {
        let mut counter_dev = DeviceBuffer::<i64>::zeroed(&stream, 1).unwrap();
        let mut cas_target_dev = DeviceBuffer::<i64>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<i64>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_i64_test(
                (stream).as_ref(),
                cfg,
                atomic_storage(&mut counter_dev),
                atomic_storage(&mut cas_target_dev),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let cas_val = cas_target_dev.to_host_vec(&stream).unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        let counter_ok = counter_val[0] == N as i64;
        let cas_ok = cas_val[0] == -100;
        assert!(result[1..].iter().all(|&value| value == -100));
        let thread0_ok = result[0] == 1;

        if counter_ok && cas_ok && thread0_ok {
            println!("  i64 counter = {} (expected {})", counter_val[0], N);
            println!("  i64 CAS: 0 -> {} (expected -100)", cas_val[0]);
            println!("  Thread 0 CAS result = {} (1 = success)", result[0]);
        } else {
            println!(
                "  FAIL: counter={} cas={} thread0={}",
                counter_val[0], cas_val[0], result[0]
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 10: fetch_sub
    // =========================================================================
    println!("\n--- Test 10: atomic_fetch_sub_test ---");
    {
        // Start counter at N, each thread subtracts 1 → should reach 0
        let counter_host: Vec<u32> = vec![N as u32];
        let mut counter_dev = DeviceBuffer::from_host(&stream, &counter_host).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_fetch_sub_test(
                (stream).as_ref(),
                cfg,
                atomic_storage(&mut counter_dev),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let out_vals = out_dev.to_host_vec(&stream).unwrap();

        if counter_val[0] == 0 {
            let mut sorted = out_vals.clone();
            sorted.sort();
            sorted.dedup();
            if sorted.len() == N {
                println!("  fetch_sub: {} -> 0, all {} old values unique", N, N);
            } else {
                println!(
                    "  FAIL: Only {} unique values (expected {})",
                    sorted.len(),
                    N
                );
                std::process::exit(1);
            }
        } else {
            println!("  FAIL: Counter = {} (expected 0)", counter_val[0]);
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 11: bitwise ops (fetch_and, fetch_or, fetch_xor)
    // =========================================================================
    println!("\n--- Test 11: atomic_bitwise_test ---");
    {
        // OR accumulator: starts at 0
        let mut or_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        // AND accumulator: starts at 0xFFFFFFFF
        let and_host: Vec<u32> = vec![0xFFFFFFFF];
        let mut and_dev = DeviceBuffer::from_host(&stream, &and_host).unwrap();
        // XOR accumulator: starts at 0
        let mut xor_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();

        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_bitwise_test(
                (stream).as_ref(),
                cfg,
                atomic_storage(&mut or_dev),
                atomic_storage(&mut and_dev),
                atomic_storage(&mut xor_dev),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let or_val = or_dev.to_host_vec(&stream).unwrap()[0];
        let and_val = and_dev.to_host_vec(&stream).unwrap()[0];
        let xor_val = xor_dev.to_host_vec(&stream).unwrap()[0];

        // With N=256 threads and tid % 32, all 32 bits should be set
        let or_ok = or_val == 0xFFFFFFFF;
        // Thread 0 ANDs with 0x0000FFFF, clearing upper 16 bits
        let and_ok = and_val == 0x0000FFFF;
        // 256 threads each XOR with 1: even count → back to 0
        let xor_ok = xor_val == 0;

        if or_ok && and_ok && xor_ok {
            println!("  fetch_or:  0x{:08X} (expected 0xFFFFFFFF)", or_val);
            println!("  fetch_and: 0x{:08X} (expected 0x0000FFFF)", and_val);
            println!("  fetch_xor: 0x{:08X} (expected 0x00000000)", xor_val);
        } else {
            println!(
                "  FAIL: or=0x{:08X} and=0x{:08X} xor=0x{:08X}",
                or_val, and_val, xor_val
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 12: swap
    // =========================================================================
    println!("\n--- Test 12: atomic_swap_test ---");
    {
        let mut target_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_swap_test(
                (stream).as_ref(),
                cfg,
                atomic_storage(&mut target_dev),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let target_val = target_dev.to_host_vec(&stream).unwrap()[0];
        let result = out_dev.to_host_vec(&stream).unwrap();

        // Thread 0 swapped 0 -> 0xDEADBEEF, got back 0
        let swap_ok = result[0] == 0;
        // Target should now be 0xDEADBEEF
        let target_ok = target_val == 0xDEADBEEF;
        assert!(result[1..].iter().all(|&value| value == 0xDEADBEEF));

        if swap_ok && target_ok {
            println!(
                "  swap: old=0x{:08X} (expected 0), target=0x{:08X}",
                result[0], target_val
            );
        } else {
            println!(
                "  FAIL: old=0x{:08X} target=0x{:08X}",
                result[0], target_val
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 13: fetch_min / fetch_max (signed)
    // =========================================================================
    println!("\n--- Test 13: atomic_minmax_test ---");
    {
        // Initialize min to i32::MAX and max to i32::MIN
        let min_host: Vec<i32> = vec![i32::MAX];
        let max_host: Vec<i32> = vec![i32::MIN];
        let mut min_dev = DeviceBuffer::from_host(&stream, &min_host).unwrap();
        let mut max_dev = DeviceBuffer::from_host(&stream, &max_host).unwrap();
        let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_minmax_test(
                (stream).as_ref(),
                cfg,
                atomic_storage(&mut min_dev),
                atomic_storage(&mut max_dev),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let min_val = min_dev.to_host_vec(&stream).unwrap()[0];
        let max_val = max_dev.to_host_vec(&stream).unwrap()[0];

        // With 256 threads, values range from -128 to +127
        let min_ok = min_val == -128;
        let max_ok = max_val == 127;

        if min_ok && max_ok {
            println!("  fetch_min: {} (expected -128)", min_val);
            println!("  fetch_max: {} (expected +127)", max_val);
        } else {
            println!("  FAIL: min={} max={}", min_val, max_val);
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 14: DeviceAtomicF32 fetch_add (float atomic add)
    // =========================================================================
    println!("\n--- Test 14: atomic_f32_fetch_add_test ---");
    {
        let mut counter_dev = DeviceBuffer::<f32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_f32_fetch_add_test(
                (stream).as_ref(),
                cfg,
                atomic_storage(&mut counter_dev),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        let expected = N as f32;
        let diff = (counter_val[0] - expected).abs();
        // f32 atomic adds may have small rounding; allow epsilon
        let counter_ok = diff < 0.01;
        let all_ran = result.iter().all(|&x| x == 1);

        if counter_ok && all_ran {
            println!("  f32 counter = {} (expected {})", counter_val[0], expected);
        } else {
            println!(
                "  FAIL: counter={} (expected ~{}), all_ran={}",
                counter_val[0], expected, all_ran
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 15: DeviceAtomicF64 fetch_add (64-bit float atomic add)
    // =========================================================================
    println!("\n--- Test 15: atomic_f64_fetch_add_test ---");
    {
        let mut counter_dev = DeviceBuffer::<f64>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_f64_fetch_add_test(
                (stream).as_ref(),
                cfg,
                atomic_storage(&mut counter_dev),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        let expected = N as f64;
        let diff = (counter_val[0] - expected).abs();
        let counter_ok = diff < 0.01;
        let all_ran = result.iter().all(|&x| x == 1);

        if counter_ok && all_ran {
            println!("  f64 counter = {} (expected {})", counter_val[0], expected);
        } else {
            println!(
                "  FAIL: counter={} (expected ~{}), all_ran={}",
                counter_val[0], expected, all_ran
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 16: DeviceAtomicF32 swap (float atomic exchange)
    // =========================================================================
    println!("\n--- Test 16: atomic_f32_swap_test ---");
    {
        let mut target_dev = DeviceBuffer::<f32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_f32_swap_test(
                (stream).as_ref(),
                cfg,
                atomic_storage(&mut target_dev),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let target_val = target_dev.to_host_vec(&stream).unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        // target should now hold 3.14 (swapped by thread 0)
        let swap_ok = (target_val[0] - 3.14).abs() < 0.01;
        // Thread 0 must have succeeded (out[0] == 1)
        let t0_ok = result[0] == 1;
        assert!(result[1..].iter().all(|&value| value == 1));

        if swap_ok && t0_ok {
            println!("  target = {} (expected ~3.14), thread 0 ok", target_val[0]);
        } else {
            println!(
                "  FAIL: target={} (expected ~3.14), t0={}",
                target_val[0], result[0]
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 17: DeviceAtomicU32 unsigned fetch_min / fetch_max (UMin/UMax)
    // =========================================================================
    println!("\n--- Test 17: atomic_unsigned_minmax_test ---");
    {
        // Initialize min accumulator to u32::MAX so any tid beats it
        let min_host = vec![u32::MAX; 1];
        let mut min_dev = DeviceBuffer::from_host(&stream, &min_host).unwrap();
        // Initialize max accumulator to 0 so any tid beats it
        let mut max_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_unsigned_minmax_test(
                (stream).as_ref(),
                cfg,
                atomic_storage(&mut min_dev),
                atomic_storage(&mut max_dev),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let min_val = min_dev.to_host_vec(&stream).unwrap();
        let max_val = max_dev.to_host_vec(&stream).unwrap();

        let min_ok = min_val[0] == 0; // thread 0's tid
        let max_ok = max_val[0] == (N as u32 - 1); // thread 255's tid

        if min_ok && max_ok {
            println!(
                "  min = {} (expected 0), max = {} (expected {})",
                min_val[0],
                max_val[0],
                N - 1
            );
        } else {
            println!(
                "  FAIL: min={} (expected 0), max={} (expected {})",
                min_val[0],
                max_val[0],
                N - 1
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 18: BlockAtomicU32 fetch_add (block scope / .cta, Relaxed)
    // =========================================================================
    println!("\n--- Test 18: atomic_block_scope_test ---");
    {
        let mut counter_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_block_scope_test(
                (stream).as_ref(),
                cfg,
                atomic_storage(&mut counter_dev),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        let counter_ok = counter_val[0] == N as u32;
        // Each old value should be unique in [0, N)
        let mut seen = vec![false; N];
        let mut unique_ok = true;
        for &v in &result {
            if (v as usize) < N && !seen[v as usize] {
                seen[v as usize] = true;
            } else {
                unique_ok = false;
                break;
            }
        }

        if counter_ok && unique_ok {
            println!(
                "  counter = {} (expected {}), all old values unique",
                counter_val[0], N
            );
        } else {
            println!(
                "  FAIL: counter={} (expected {}), unique={}",
                counter_val[0], N, unique_ok
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 19: BlockAtomicU32 fetch_add with AcqRel (.cta scope on fences)
    // =========================================================================
    println!("\n--- Test 19: atomic_block_scope_acqrel_test ---");
    {
        let mut counter_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_block_scope_acqrel_test(
                (stream).as_ref(),
                cfg,
                atomic_storage(&mut counter_dev),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        let counter_ok = counter_val[0] == N as u32;
        let mut seen = vec![false; N];
        let mut unique_ok = true;
        for &v in &result {
            if (v as usize) < N && !seen[v as usize] {
                seen[v as usize] = true;
            } else {
                unique_ok = false;
                break;
            }
        }

        if counter_ok && unique_ok {
            println!(
                "  counter = {} (expected {}), all old values unique",
                counter_val[0], N
            );
        } else {
            println!(
                "  FAIL: counter={} (expected {}), unique={}",
                counter_val[0], N, unique_ok
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 20: core::sync::atomic::AtomicU32 fetch_add (system scope)
    // =========================================================================
    println!("\n--- Test 20: core_atomic_fetch_add_test (core::sync::atomic) ---");
    {
        let mut counter_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.core_atomic_fetch_add_test(
                (stream).as_ref(),
                cfg,
                atomic_storage(&mut counter_dev),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        let counter_ok = counter_val[0] == N as u32;
        let mut seen = vec![false; N];
        let mut unique_ok = true;
        for &v in &result {
            if (v as usize) < N && !seen[v as usize] {
                seen[v as usize] = true;
            } else {
                unique_ok = false;
                break;
            }
        }

        if counter_ok && unique_ok {
            println!(
                "  counter = {} (expected {}), all old values unique",
                counter_val[0], N
            );
        } else {
            println!(
                "  FAIL: counter={} (expected {}), unique={}",
                counter_val[0], N, unique_ok
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 21: core::sync::atomic::AtomicPtr load/store/swap/CAS
    // =========================================================================
    println!("\n--- Test 21: core_atomic_ptr_test (core::sync::atomic::AtomicPtr) ---");
    {
        let mut storage_dev = DeviceBuffer::<usize>::zeroed(&stream, N).unwrap();
        let values: Vec<u16> = (0..N).map(|index| index as u16 + 1).collect();
        let values_dev = DeviceBuffer::from_host(&stream, &values).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: every thread uses one pointer-sized storage slot, one value,
        // and one output element covered by the supplied buffers.
        unsafe {
            module.core_atomic_ptr_test(
                (stream).as_ref(),
                cfg,
                &mut storage_dev,
                &values_dev,
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        if let Some(index) = result.iter().position(|&value| value != 1) {
            println!(
                "  FAIL: thread {} reported AtomicPtr result {}",
                index, result[index]
            );
            std::process::exit(1);
        } else {
            println!("  all {} threads passed AtomicPtr load/store/swap/CAS", N);
        }
    }

    println!("\n--- Test 22: core_atomic_local_test ---");
    {
        let values: Vec<u16> = (0..N).map(|index| index as u16 + 1).collect();
        let values_dev = DeviceBuffer::from_host(&stream, &values).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
        // SAFETY: the buffers cover each thread's input/output and out is
        // disjoint. Every atomic object is private to its executing thread.
        unsafe { module.core_atomic_local_test(stream.as_ref(), cfg, &values_dev, &mut out_dev) }
            .expect("Kernel launch failed");
        stream.synchronize().unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();
        assert!(
            result.iter().all(|&value| value == 1),
            "local atomic round-trip failed: {result:?}"
        );
        println!("  all {N} threads passed private pointer and integer atomic operations");
    }

    println!("\n--- Test 23: core_atomic_ptr_shared_value_test ---");
    {
        let mut storage_dev = DeviceBuffer::<usize>::zeroed(&stream, N).unwrap();

        unsafe { module.core_atomic_ptr_shared_value_test(stream.as_ref(), cfg, &mut storage_dev) }
            .expect("Kernel launch failed");

        stream.synchronize().unwrap();

        let result = storage_dev.to_host_vec(&stream).unwrap();

        if let Some(index) = result.iter().position(|&value| value != 1) {
            println!(
                "  FAIL: thread {} reported shared AtomicPtr round-trip result {}",
                index, result[index]
            );
            std::process::exit(1);
        } else {
            println!(
                "  all {} threads passed shared-pointer AtomicPtr load/store/swap/CAS/deref",
                N
            );
        }
    }

    println!("\n--- Test 24: core_atomic_ptr_shared_storage_test ---");
    {
        let values: Vec<u16> = (0..N).map(|index| index as u16 + 1).collect();
        let values_dev = DeviceBuffer::from_host(&stream, &values).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        unsafe {
            module.core_atomic_ptr_shared_storage_test(
                stream.as_ref(),
                cfg,
                &values_dev,
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();

        let result = out_dev.to_host_vec(&stream).unwrap();

        if let Some(index) = result.iter().position(|&value| value != 1) {
            println!(
                "  FAIL: thread {} reported shared-storage AtomicPtr result {}",
                index, result[index]
            );
            std::process::exit(1);
        } else {
            println!(
                "  all {} threads passed shared-storage AtomicPtr load/store/swap/CAS",
                N
            );
        }
    }

    println!("\n=== SUCCESS: All 24 runtime atomic tests passed! ===");
}
