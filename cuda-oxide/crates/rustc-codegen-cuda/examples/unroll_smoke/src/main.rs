/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke test for the `#[unroll]` / `#[unroll(N)]` loop-unroll transform.
//!
//! The attribute goes directly on the loop it should unroll (the `#[kernel]`
//! macro reads it and tags that loop): a function can unroll one loop and leave
//! its neighbours alone.
//!
//! `full_unroll` has a compile-time-constant trip count, so `#[unroll]` should
//! unroll it completely and the per-iteration `i & 3` should fold to literals.
//! `partial_unroll` has a runtime trip count, so `#[unroll(4)]` unrolls the body
//! by 4 and leaves a remainder loop. Both are semantics-preserving; the host
//! checks the sums.
//!
//! Run: cargo oxide run unroll_smoke

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// Full unroll of a constant-trip-count loop. `acc` starts at the thread
    /// index and adds `i & 3` for `i` in `0..8` (= 0+1+2+3+0+1+2+3 = 12), so
    /// `out[tid] == tid + 12`.
    #[kernel]
    pub fn full_unroll(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let base = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = base;
            let mut i: u32 = 0;
            #[unroll]
            while i < 8 {
                acc = acc.wrapping_add(i & 3);
                i += 1;
            }
            *out_elem = acc;
        }
    }

    /// Unrolling makes the shift counts constant while retaining their types.
    /// Folding must handle counts wider and narrower than the shifted value,
    /// and preserve arithmetic versus logical right shifts.
    #[kernel]
    pub fn full_mixed_width_shifts(
        mut left: DisjointSlice<u32>,
        mut arithmetic: DisjointSlice<i32>,
        mut logical: DisjointSlice<u64>,
    ) {
        let (Some(left), Some(arithmetic), Some(logical)) = (
            left.get_mut(thread::index_1d()),
            arithmetic.get_mut(thread::index_1d()),
            logical.get_mut(thread::index_1d()),
        ) else {
            return;
        };
        let mut left_bits = 0u32;
        let mut signed_sum = 0i32;
        let mut i = 0usize;
        #[unroll]
        while i < 4 {
            left_bits |= 1u32 << (2 * i);
            signed_sum += -128i32 >> i;
            i += 1;
        }
        let mut right_bits = 0u64;
        let mut j = 0u8;
        #[unroll]
        while j < 4 {
            right_bits |= (1u64 << 63) >> j;
            j += 1;
        }
        *left = left_bits;
        *arithmetic = signed_sum;
        *logical = right_bits;
    }

    /// Partial unroll (by 4) of a runtime-trip-count loop: `out[tid]` is the
    /// sum `0 + 1 + ... + (n-1) == n*(n-1)/2`.
    #[kernel]
    pub fn partial_unroll(mut out: DisjointSlice<u32>, n: u32) {
        let tid = thread::index_1d();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = 0;
            let mut i: u32 = 0;
            #[unroll(4)]
            while i < n {
                acc = acc.wrapping_add(i);
                i += 1;
            }
            *out_elem = acc;
        }
    }

    /// Partial unroll (by 4) of a runtime loop whose body uses `i & 3` (the
    /// gemm "stage" pattern). After unrolling, the main loop's counter is a
    /// multiple of 4, so `(i+j) & 3` should fold to the constants `0,1,2,3`.
    /// `out[tid]` is the sum of `i & 3` for `i` in `0..n`.
    #[kernel]
    pub fn partial_fold(mut out: DisjointSlice<u32>, n: u32) {
        let tid = thread::index_1d();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = 0;
            let mut i: u32 = 0;
            #[unroll(4)]
            while i < n {
                acc = acc.wrapping_add(i & 3);
                i += 1;
            }
            *out_elem = acc;
        }
    }

    /// Partial unroll (by 4) using `i % 4`: a power-of-two modulo, the same stage
    /// index as `i & 3` but spelled with `%`. After unrolling, the main loop's
    /// counter is a multiple of 4, so `(i + j) % 4` folds to the constants
    /// `0,1,2,3` -- this exercises the `% 2^k` arm of the index fold. `out[tid]`
    /// is the sum of `i % 4` for `i` in `0..n`.
    #[kernel]
    pub fn partial_mod(mut out: DisjointSlice<u32>, n: u32) {
        let tid = thread::index_1d();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = 0;
            let mut i: u32 = 0;
            #[unroll(4)]
            while i < n {
                acc = acc.wrapping_add(i % 4);
                i += 1;
            }
            *out_elem = acc;
        }
    }

    /// Full unroll of a loop whose body has **internal control flow** (an
    /// `if`/`else`), so the body is several basic blocks, not one. This is the
    /// case the earlier single-block unroller could not handle. For `i` in
    /// `0..8`: even `i` adds `i` (0+2+4+6 = 12), odd `i` adds 10 (4 * 10 = 40),
    /// so `out[tid] == tid + 52`.
    #[kernel]
    pub fn full_mb(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let base = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = base;
            let mut i: u32 = 0;
            #[unroll]
            while i < 8 {
                if i & 1 == 0 {
                    acc = acc.wrapping_add(i);
                } else {
                    acc = acc.wrapping_add(10);
                }
                i += 1;
            }
            *out_elem = acc;
        }
    }

    /// Partial unroll (by 4) of a runtime loop whose body has internal control
    /// flow. For `i` in `0..n`: even `i` adds `i`, odd `i` adds 100. With n=10:
    /// even (0+2+4+6+8 = 20) + odd (5 * 100 = 500) = 520.
    #[kernel]
    pub fn partial_mb(mut out: DisjointSlice<u32>, n: u32) {
        let tid = thread::index_1d();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = 0;
            let mut i: u32 = 0;
            #[unroll(4)]
            while i < n {
                if i & 1 == 0 {
                    acc = acc.wrapping_add(i);
                } else {
                    acc = acc.wrapping_add(100);
                }
                i += 1;
            }
            *out_elem = acc;
        }
    }

    /// Full unroll with an early `break`. Only `i = 0..4` reach the addition, so
    /// each thread writes its index plus `0+1+2+3+4 = 10`.
    #[kernel]
    pub fn full_early_break(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let base = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc = base;
            let mut i = 0u32;
            #[unroll]
            while i < 8 {
                if i == 5 {
                    break;
                }
                acc = acc.wrapping_add(i);
                i += 1;
            }
            *out_elem = acc;
        }
    }

    /// Partial unroll with two paths back to the header. Both paths increment
    /// `i` once, while even values add `i` and odd values add `2*i`.
    #[kernel]
    pub fn partial_continue_paths(mut out: DisjointSlice<u32>, n: u32) {
        let tid = thread::index_1d();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc = 0u32;
            let mut i = 0u32;
            #[unroll(4)]
            while i < n {
                let current = i;
                i += 1;
                if current & 1 == 0 {
                    acc = acc.wrapping_add(current);
                    continue;
                }
                acc = acc.wrapping_add(current.wrapping_mul(2));
            }
            *out_elem = acc;
        }
    }

    /// Partial unroll still rejects early exits. This loop must run normally
    /// after the warning and produce the same `0+1+2+3+4 = 10` result.
    #[kernel]
    pub fn partial_early_break_skipped(mut out: DisjointSlice<u32>, n: u32) {
        let tid = thread::index_1d();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc = 0u32;
            let mut i = 0u32;
            #[unroll(4)]
            while i < n {
                if i == 5 {
                    break;
                }
                acc = acc.wrapping_add(i);
                i += 1;
            }
            *out_elem = acc;
        }
    }

    /// Regression guard: the loop bound `hi` is **loop-carried** (it changes each
    /// iteration), so partial unroll's "does a group of 4 still fit" guard would
    /// be unsound. The pass must refuse this loop (a loud warning) and leave it as
    /// an ordinary loop, NOT miscompile or crash. For n=10 it runs 5 iterations
    /// (i=0..4 before i meets the shrinking hi), so `out[tid] == 0+1+2+3+4 = 10`.
    #[kernel]
    pub fn carried_bound(mut out: DisjointSlice<u32>, n: u32) {
        let tid = thread::index_1d();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = 0;
            let mut i: u32 = 0;
            let mut hi: u32 = n;
            #[unroll(4)]
            while i < hi {
                acc = acc.wrapping_add(i);
                i += 1;
                hi = hi.wrapping_sub(1);
            }
            *out_elem = acc;
        }
    }

    /// Nested loops, inner FULLY unrolled. The outer loop (runtime trip `n`) is
    /// left alone; its inner `while j < 4` carries `#[unroll]`. The inner body
    /// reads the OUTER counter `k` by dominance, which the clone must preserve.
    /// Each outer iteration adds `0*k + 1*k + 2*k + 3*k = 6k`; summed over `k` in
    /// `0..n`. For n=10: `6 * (0+1+...+9) = 6 * 45 = 270`.
    #[kernel]
    pub fn nested_full(mut out: DisjointSlice<u32>, n: u32) {
        let tid = thread::index_1d();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = 0;
            let mut k: u32 = 0;
            while k < n {
                let mut j: u32 = 0;
                #[unroll]
                while j < 4 {
                    acc = acc.wrapping_add(j.wrapping_mul(k));
                    j += 1;
                }
                k += 1;
            }
            *out_elem = acc;
        }
    }

    /// Nested loops, inner PARTIALLY unrolled (by 2) with a CONSTANT bound. Same
    /// outer-counter capture; exercises the partial path nested plus constant-bound
    /// re-materialization. Each outer iteration adds `(0+k)+(1+k)+(2+k)+(3+k) =
    /// 6 + 4k`; summed over `k` in `0..n`. For n=10: `10*6 + 4*45 = 240`.
    #[kernel]
    pub fn nested_partial(mut out: DisjointSlice<u32>, n: u32) {
        let tid = thread::index_1d();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = 0;
            let mut k: u32 = 0;
            while k < n {
                let mut j: u32 = 0;
                #[unroll(2)]
                while j < 4 {
                    acc = acc.wrapping_add(j.wrapping_add(k));
                    j += 1;
                }
                k += 1;
            }
            *out_elem = acc;
        }
    }

    /// Nested loops, inner partially unrolled with a NON-constant bound that is the
    /// OUTER counter `k`. `k` is invariant within the inner loop (it changes only
    /// in the outer loop), so partial unroll is sound and `k` dominates the new
    /// main header. Inner runs `j` in `0..k`, summing `j` = `k*(k-1)/2`; summed
    /// over `k` in `0..n`. For n=10 that is `0+0+1+3+6+10+15+21+28+36 = 120`.
    #[kernel]
    pub fn nested_var_bound(mut out: DisjointSlice<u32>, n: u32) {
        let tid = thread::index_1d();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = 0;
            let mut k: u32 = 0;
            while k < n {
                let mut j: u32 = 0;
                #[unroll(2)]
                while j < k {
                    acc = acc.wrapping_add(j);
                    j += 1;
                }
                k += 1;
            }
            *out_elem = acc;
        }
    }

    /// Unroll a loop that *contains* a loop, FULL path. The OUTER loop (constant
    /// trip 4) carries `#[unroll]`; its body holds an inner `while j < 3` that is
    /// left alone and cloned wholesale into each of the 4 copies (it stays a
    /// loop). The outer `i & 3` folds to 0,1,2,3. Each outer iteration adds
    /// `(i & 3) + (0+1+2)`; summed over i in 0..4: (0+1+2+3) + 4*3 = 6 + 12 = 18,
    /// so `out[tid] == tid + 18`.
    #[kernel]
    pub fn outer_full(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let base = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = base;
            let mut i: u32 = 0;
            #[unroll]
            while i < 4 {
                acc = acc.wrapping_add(i & 3);
                let mut j: u32 = 0;
                while j < 3 {
                    acc = acc.wrapping_add(j);
                    j += 1;
                }
                i += 1;
            }
            *out_elem = acc;
        }
    }

    /// Unroll a loop that *contains* a loop, PARTIAL path -- the gemm K-loop
    /// shape. The OUTER loop (runtime trip `n`) carries `#[unroll(4)]`; its body
    /// holds the stage index `i & 3` (which folds to the constants 0,1,2,3 across
    /// the 4 copies because the main counter steps by 4) plus an inner
    /// `while j < 3` that is cloned wholesale into each copy and stays a loop.
    /// Each outer iteration adds `(i & 3) + (0+1+2)`; `out[tid]` is
    /// `sum_{i<n}(i & 3) + n*3`.
    #[kernel]
    pub fn outer_partial(mut out: DisjointSlice<u32>, n: u32) {
        let tid = thread::index_1d();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = 0;
            let mut i: u32 = 0;
            #[unroll(4)]
            while i < n {
                acc = acc.wrapping_add(i & 3);
                let mut j: u32 = 0;
                while j < 3 {
                    acc = acc.wrapping_add(j);
                    j += 1;
                }
                i += 1;
            }
            *out_elem = acc;
        }
    }

    /// Exit tests that compare the counter plus a constant, the way one writes
    /// "stop when a whole tile no longer fits". `while i + 1 <= 4` fully unrolls
    /// like `while i < 4` (`1 + 4 + 16 + 64 = 85`); the tile walk
    /// `while d + 2 <= 8` visits `0 + 2 + 4 + 6 = 12`; and `#[unroll(4)]` on
    /// `while j + 1 <= n` sums `j` over `0..n`, `n*(n-1)/2`.
    #[allow(clippy::int_plus_one)]
    #[kernel]
    pub fn offset_exit_tests(
        mut bits: DisjointSlice<u32>,
        mut walk: DisjointSlice<u32>,
        mut partial: DisjointSlice<u32>,
        n: u32,
    ) {
        let (Some(bits), Some(walk), Some(partial)) = (
            bits.get_mut(thread::index_1d()),
            walk.get_mut(thread::index_1d()),
            partial.get_mut(thread::index_1d()),
        ) else {
            return;
        };
        let mut set = 0u32;
        let mut i = 0usize;
        #[unroll]
        while i + 1 <= 4 {
            set |= 1u32 << (2 * i);
            i += 1;
        }
        const TILE: usize = 2;
        const WIDTH: usize = 8;
        let mut visited = 0u32;
        let mut d = 0usize;
        #[unroll]
        while d + TILE <= WIDTH {
            visited += d as u32;
            d += TILE;
        }
        let mut sum = 0u32;
        let mut j = 0u32;
        #[unroll(4)]
        while j + 1 <= n {
            sum = sum.wrapping_add(j);
            j += 1;
        }
        *bits = set;
        *walk = visited;
        *partial = sum;
    }
}

fn main() {
    println!("=== unroll_smoke ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let stream = ctx.default_stream();

    const BLOCK: u32 = 32;
    const N: usize = BLOCK as usize;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut d_full = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.full_unroll(stream.as_ref(), cfg, &mut d_full) }.expect("launch full_unroll");
    let got_full = d_full.to_host_vec(&stream).unwrap();

    let mut d_shift_left = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    let mut d_shift_arithmetic = DeviceBuffer::<i32>::zeroed(&stream, N).unwrap();
    let mut d_shift_logical = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();
    // SAFETY: each thread writes its own element in three separate buffers.
    unsafe {
        module.full_mixed_width_shifts(
            stream.as_ref(),
            cfg,
            &mut d_shift_left,
            &mut d_shift_arithmetic,
            &mut d_shift_logical,
        )
    }
    .expect("launch full_mixed_width_shifts");
    let got_shift_left = d_shift_left.to_host_vec(&stream).unwrap();
    let got_shift_arithmetic = d_shift_arithmetic.to_host_vec(&stream).unwrap();
    let got_shift_logical = d_shift_logical.to_host_vec(&stream).unwrap();
    assert_eq!(got_shift_left, vec![85; N], "mixed-width left shift");
    assert_eq!(
        got_shift_arithmetic,
        vec![-240; N],
        "arithmetic right shift"
    );
    assert_eq!(
        got_shift_logical,
        vec![0xf000_0000_0000_0000; N],
        "logical right shift"
    );

    let trip: u32 = 10;
    let mut d_part = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.partial_unroll(stream.as_ref(), cfg, &mut d_part, trip) }
        .expect("launch partial_unroll");
    let got_part = d_part.to_host_vec(&stream).unwrap();

    let mut d_fold = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.partial_fold(stream.as_ref(), cfg, &mut d_fold, trip) }
        .expect("launch partial_fold");
    let got_fold = d_fold.to_host_vec(&stream).unwrap();

    let mut d_mod = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.partial_mod(stream.as_ref(), cfg, &mut d_mod, trip) }
        .expect("launch partial_mod");
    let got_mod = d_mod.to_host_vec(&stream).unwrap();

    let mut d_fullmb = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.full_mb(stream.as_ref(), cfg, &mut d_fullmb) }.expect("launch full_mb");
    let got_fullmb = d_fullmb.to_host_vec(&stream).unwrap();

    let mut d_partmb = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.partial_mb(stream.as_ref(), cfg, &mut d_partmb, trip) }
        .expect("launch partial_mb");
    let got_partmb = d_partmb.to_host_vec(&stream).unwrap();

    let mut d_full_break = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.full_early_break(stream.as_ref(), cfg, &mut d_full_break) }
        .expect("launch full_early_break");
    let got_full_break = d_full_break.to_host_vec(&stream).unwrap();

    let mut d_continue = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.partial_continue_paths(stream.as_ref(), cfg, &mut d_continue, trip) }
        .expect("launch partial_continue_paths");
    let got_continue = d_continue.to_host_vec(&stream).unwrap();

    let mut d_partial_break = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.partial_early_break_skipped(stream.as_ref(), cfg, &mut d_partial_break, trip) }
        .expect("launch partial_early_break_skipped");
    let got_partial_break = d_partial_break.to_host_vec(&stream).unwrap();

    let mut d_carried = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.carried_bound(stream.as_ref(), cfg, &mut d_carried, trip) }
        .expect("launch carried_bound");
    let got_carried = d_carried.to_host_vec(&stream).unwrap();

    let mut d_nfull = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.nested_full(stream.as_ref(), cfg, &mut d_nfull, trip) }
        .expect("launch nested_full");
    let got_nfull = d_nfull.to_host_vec(&stream).unwrap();

    let mut d_npart = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.nested_partial(stream.as_ref(), cfg, &mut d_npart, trip) }
        .expect("launch nested_partial");
    let got_npart = d_npart.to_host_vec(&stream).unwrap();

    let mut d_nvar = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.nested_var_bound(stream.as_ref(), cfg, &mut d_nvar, trip) }
        .expect("launch nested_var_bound");
    let got_nvar = d_nvar.to_host_vec(&stream).unwrap();

    let mut d_ofull = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.outer_full(stream.as_ref(), cfg, &mut d_ofull) }.expect("launch outer_full");
    let got_ofull = d_ofull.to_host_vec(&stream).unwrap();

    let mut d_opart = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.outer_partial(stream.as_ref(), cfg, &mut d_opart, trip) }
        .expect("launch outer_partial");
    let got_opart = d_opart.to_host_vec(&stream).unwrap();

    for offset_trip in [7u32, 16] {
        let mut d_bits = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
        let mut d_walk = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
        let mut d_offset_part = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
        // SAFETY: each thread writes its own element in three separate buffers.
        unsafe {
            module.offset_exit_tests(
                stream.as_ref(),
                cfg,
                &mut d_bits,
                &mut d_walk,
                &mut d_offset_part,
                offset_trip,
            )
        }
        .expect("launch offset_exit_tests");
        assert_eq!(
            d_bits.to_host_vec(&stream).unwrap(),
            vec![85; N],
            "offset exit test `i + 1 <= 4`"
        );
        assert_eq!(
            d_walk.to_host_vec(&stream).unwrap(),
            vec![12; N],
            "tile walk `d + 2 <= 8`"
        );
        assert_eq!(
            d_offset_part.to_host_vec(&stream).unwrap(),
            vec![offset_trip * (offset_trip - 1) / 2; N],
            "partial offset exit test `j + 1 <= {offset_trip}`"
        );
    }

    let mut failures = 0usize;
    let want_part = trip * (trip - 1) / 2;
    let want_fold: u32 = (0..trip).map(|i| i & 3).sum();
    let want_mod: u32 = (0..trip).map(|i| i % 4).sum();
    let want_partmb: u32 = (0..trip).map(|i| if i & 1 == 0 { i } else { 100 }).sum();
    let want_continue: u32 = (0..trip).map(|i| if i & 1 == 0 { i } else { 2 * i }).sum();
    let want_partial_break: u32 = (0..trip.min(5)).sum();
    let want_carried: u32 = {
        let (mut a, mut i, mut hi) = (0u32, 0u32, trip);
        while i < hi {
            a = a.wrapping_add(i);
            i += 1;
            hi = hi.wrapping_sub(1);
        }
        a
    };
    let want_nfull: u32 = (0..trip)
        .flat_map(|k| (0..4u32).map(move |j| j.wrapping_mul(k)))
        .fold(0u32, |a, x| a.wrapping_add(x));
    let want_npart: u32 = (0..trip)
        .flat_map(|k| (0..4u32).map(move |j| j.wrapping_add(k)))
        .fold(0u32, |a, x| a.wrapping_add(x));
    let want_nvar: u32 = (0..trip)
        .flat_map(|k| 0..k)
        .fold(0u32, |a, x| a.wrapping_add(x));
    // outer_partial: sum_{i<n}(i & 3) + n*(0+1+2), the same `i & 3` sum as
    // `want_fold` plus the inner-loop contribution.
    let want_opart: u32 = want_fold + trip * 3;
    for tid in 0..N {
        let want_full = tid as u32 + 12;
        let want_fullmb = tid as u32 + 52;
        let want_ofull = tid as u32 + 18;
        if got_full[tid] != want_full {
            println!(
                "FAIL tid={tid}: full_unroll={} expected={want_full}",
                got_full[tid]
            );
            failures += 1;
        }
        if got_part[tid] != want_part {
            println!(
                "FAIL tid={tid}: partial_unroll={} expected={want_part}",
                got_part[tid]
            );
            failures += 1;
        }
        if got_fold[tid] != want_fold {
            println!(
                "FAIL tid={tid}: partial_fold={} expected={want_fold}",
                got_fold[tid]
            );
            failures += 1;
        }
        if got_mod[tid] != want_mod {
            println!(
                "FAIL tid={tid}: partial_mod={} expected={want_mod}",
                got_mod[tid]
            );
            failures += 1;
        }
        if got_fullmb[tid] != want_fullmb {
            println!(
                "FAIL tid={tid}: full_mb={} expected={want_fullmb}",
                got_fullmb[tid]
            );
            failures += 1;
        }
        if got_partmb[tid] != want_partmb {
            println!(
                "FAIL tid={tid}: partial_mb={} expected={want_partmb}",
                got_partmb[tid]
            );
            failures += 1;
        }
        let want_full_break = tid as u32 + 10;
        if got_full_break[tid] != want_full_break {
            println!(
                "FAIL tid={tid}: full_early_break={} expected={want_full_break}",
                got_full_break[tid]
            );
            failures += 1;
        }
        if got_continue[tid] != want_continue {
            println!(
                "FAIL tid={tid}: partial_continue_paths={} expected={want_continue}",
                got_continue[tid]
            );
            failures += 1;
        }
        if got_partial_break[tid] != want_partial_break {
            println!(
                "FAIL tid={tid}: partial_early_break_skipped={} expected={want_partial_break}",
                got_partial_break[tid]
            );
            failures += 1;
        }
        if got_carried[tid] != want_carried {
            println!(
                "FAIL tid={tid}: carried_bound={} expected={want_carried}",
                got_carried[tid]
            );
            failures += 1;
        }
        if got_nfull[tid] != want_nfull {
            println!(
                "FAIL tid={tid}: nested_full={} expected={want_nfull}",
                got_nfull[tid]
            );
            failures += 1;
        }
        if got_npart[tid] != want_npart {
            println!(
                "FAIL tid={tid}: nested_partial={} expected={want_npart}",
                got_npart[tid]
            );
            failures += 1;
        }
        if got_nvar[tid] != want_nvar {
            println!(
                "FAIL tid={tid}: nested_var_bound={} expected={want_nvar}",
                got_nvar[tid]
            );
            failures += 1;
        }
        if got_ofull[tid] != want_ofull {
            println!(
                "FAIL tid={tid}: outer_full={} expected={want_ofull}",
                got_ofull[tid]
            );
            failures += 1;
        }
        if got_opart[tid] != want_opart {
            println!(
                "FAIL tid={tid}: outer_partial={} expected={want_opart}",
                got_opart[tid]
            );
            failures += 1;
        }
    }

    if failures == 0 {
        println!("unroll_smoke: PASS ({N} threads; full + partial unroll correct)");
    } else {
        println!("unroll_smoke: FAIL ({failures} mismatches)");
        std::process::exit(1);
    }
}
