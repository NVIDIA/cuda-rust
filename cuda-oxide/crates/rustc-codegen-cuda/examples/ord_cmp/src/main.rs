// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Regression test for `Ord::cmp` in device code (issue #136) and for
//! enum tags carrying declared discriminant VALUES (issue #146).
//!
//! Four kernels:
//! - `cmp_kernel`: `Ord::cmp` across u32/i32/u64/i64/usize/isize plus the
//!   narrow types i8/u8/i16/u16 and `char` (signedness must pick
//!   `icmp slt/sgt` vs `ult/ugt` per type).
//! - `cmp_cast_kernel`: `Ordering as i8` / `as i32` lanes. `Ordering` is
//!   `repr(i8)` with `Less = -1`, so the tag must SIGN-extend: a
//!   variant-index tag (0) or a zero-extended tag (255) would not produce
//!   -1.
//! - `ordering_match_kernel`: the issue #146 shape - a helper returning
//!   `Ordering` literals, matched in the kernel. With variant-index tags
//!   `Less` (index 0) matched the `Equal` arm (discriminant 0).
//! - `enum_repr_kernel`: default-repr enums whose tag layout rustc picks:
//!   a sparse enum `{ A = 0, B = 1000 }` (u16 tag) and a negative enum
//!   `{ N = -5, Z }` (SIGNED i8 tag; `as i32` must sign-extend to -5/-4).
//!
//! Run: cargo oxide run ord_cmp

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[derive(Clone, Copy, Default, Eq, PartialEq)]
struct Foo<T> {
    pieces: [T; 4],
}

// Hand-written Ord delegating to integer cmp is the test subject: since
// Rust 1.80 it lowers through MIR BinOp::Cmp (the three_way_compare
// intrinsic). PartialOrd delegates to it, the canonical pairing for a
// manual Ord (clippy::derive_ord_xor_partial_ord).
impl<T> Ord for Foo<T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.pieces[0].cmp(&other.pieces[0])
    }
}

impl<T> PartialOrd for Foo<T>
where
    T: Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cuda_module]
mod kernels {
    use super::*;
    use core::cmp::Ordering;

    fn cmp_code<T>(lhs: T, rhs: T) -> i32
    where
        T: Ord + Copy + Default,
    {
        let a = Foo {
            pieces: [lhs, T::default(), T::default(), T::default()],
        };
        let b = Foo {
            pieces: [rhs, T::default(), T::default(), T::default()],
        };
        match a.cmp(&b) {
            Ordering::Less => -1,
            Ordering::Equal => 0,
            Ordering::Greater => 1,
        }
    }

    #[kernel]
    pub fn cmp_kernel(mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        let lane = idx.get();

        let code = match lane {
            0 => cmp_code(0_u32, u32::MAX),
            1 => cmp_code(u32::MAX, u32::MAX),
            2 => cmp_code(u32::MAX, 0_u32),
            3 => cmp_code(i32::MIN, i32::MAX),
            4 => cmp_code(i32::MIN, i32::MIN),
            5 => cmp_code(i32::MAX, i32::MIN),
            6 => cmp_code(0_u64, u64::MAX),
            7 => cmp_code(u64::MAX, u64::MAX),
            8 => cmp_code(u64::MAX, 0_u64),
            9 => cmp_code(i64::MIN, i64::MAX),
            10 => cmp_code(i64::MIN, i64::MIN),
            11 => cmp_code(i64::MAX, i64::MIN),
            12 => cmp_code(0_usize, usize::MAX),
            13 => cmp_code(usize::MAX, usize::MAX),
            14 => cmp_code(usize::MAX, 0_usize),
            15 => cmp_code(isize::MIN, isize::MAX),
            16 => cmp_code(isize::MIN, isize::MIN),
            17 => cmp_code(isize::MAX, isize::MIN),
            // Narrow types: i8::MIN vs i8::MAX is -1 only under SIGNED
            // compare (unsigned would see 0x80 > 0x7F); u8::MAX vs 0 is 1
            // only under UNSIGNED compare (signed would see -1 < 0).
            18 => cmp_code(i8::MIN, i8::MAX),
            19 => cmp_code(7_i8, 7_i8),
            20 => cmp_code(u8::MAX, 0_u8),
            21 => cmp_code(0_u8, u8::MAX),
            22 => cmp_code(i16::MIN, i16::MAX),
            23 => cmp_code(u16::MAX, 0_u16),
            24 => cmp_code('a', 'b'),
            25 => cmp_code('b', 'a'),
            26 => cmp_code('z', 'z'),
            _ => return,
        };

        if let Some(slot) = out.get_mut(idx) {
            *slot = code;
        }
    }

    /// `Ordering` is `repr(i8)`: `Less as i8` is -1 and `Less as i32` is
    /// -1. Both require the stored tag to be the declared discriminant
    /// (255 = -1 as i8) AND the widening to be a sign-extension.
    #[kernel]
    pub fn cmp_cast_kernel(mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        let lane = idx.get();

        // Lane-derived operands so the comparison cannot constant-fold.
        let a = lane as i32;
        let b = a + 1;

        let code = match lane {
            0 => a.cmp(&b) as i8 as i32, // Less  -> -1 via i8
            1 => a.cmp(&b) as i32,       // Less  -> -1 via i32
            2 => b.cmp(&a) as i32,       // Greater -> 1
            3 => a.cmp(&a) as i32,       // Equal -> 0
            _ => return,
        };

        if let Some(slot) = out.get_mut(idx) {
            *slot = code;
        }
    }

    /// Issue #146 shape: a helper that returns `Ordering` LITERALS (enum
    /// constants, not `cmp` results), matched in the kernel. The literal
    /// constants and the match arms must agree on tag semantics.
    fn helper(lhs: [i32; 4], rhs: [i32; 4]) -> Ordering {
        for pos in 0..4 {
            if lhs[pos] < rhs[pos] {
                return Ordering::Less;
            }
            if lhs[pos] > rhs[pos] {
                return Ordering::Greater;
            }
        }
        Ordering::Equal
    }

    #[kernel]
    pub fn ordering_match_kernel(mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();

        if let Some(cell) = out.get_mut(idx) {
            let lhs = [0, 0x169000, 0, 0];
            let rhs = [0, 0xbd1000, 0, 0];

            match helper(lhs, rhs) {
                Ordering::Less => {
                    *cell = -1;
                }
                Ordering::Equal => {
                    *cell = 0;
                }
                Ordering::Greater => {
                    *cell = 1;
                }
            }
        }
    }

    /// Default-repr enum with sparse discriminants: rustc stores a u16
    /// tag holding 0 / 1000 (not variant indices 0 / 1).
    enum Sparse {
        A = 0,
        B = 1000,
    }

    /// Default-repr enum with negative discriminants: rustc stores a
    /// SIGNED i8 tag holding -5 / -4, so `as i32` must sign-extend.
    enum Neg {
        N = -5,
        Z,
    }

    fn pick_sparse(x: u32) -> Sparse {
        if x.is_multiple_of(2) {
            Sparse::A
        } else {
            Sparse::B
        }
    }

    fn pick_neg(x: u32) -> Neg {
        if x.is_multiple_of(2) { Neg::N } else { Neg::Z }
    }

    #[kernel]
    pub fn enum_repr_kernel(mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        let lane = idx.get();

        let code = match lane {
            0 | 1 => match pick_sparse(lane as u32) {
                Sparse::A => 100,
                Sparse::B => 200,
            },
            2 | 3 => pick_neg(lane as u32) as i32,
            _ => return,
        };

        if let Some(slot) = out.get_mut(idx) {
            *slot = code;
        }
    }
}

fn check(name: &str, got: &[i32], expected: &[i32]) -> bool {
    if got == expected {
        println!("{name}: OK {got:?}");
        true
    } else {
        println!("{name}: MISMATCH\n  expected {expected:?}\n  got      {got:?}");
        false
    }
}

fn main() {
    println!("=== Ord cmp regression test ===");

    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let stream = ctx.default_stream();

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut ok = true;

    // cmp_kernel: full-width + narrow integer + char comparisons.
    let mut out = DeviceBuffer::<i32>::zeroed(&stream, 27).expect("failed to allocate output");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.cmp_kernel(stream.as_ref(), config, &mut out) }
        .expect("cmp_kernel launch failed");
    let got = out.to_host_vec(&stream).expect("failed to copy output");
    let expected = [
        -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, // wide types
        -1, 0, 1, -1, -1, 1, -1, 1, 0, // i8/u8/i16/u16/char
    ];
    ok &= check("cmp_kernel", &got, &expected);

    // cmp_cast_kernel: Ordering as i8 / as i32 must sign-extend to -1.
    let mut out = DeviceBuffer::<i32>::zeroed(&stream, 4).expect("failed to allocate output");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.cmp_cast_kernel(stream.as_ref(), config, &mut out) }
        .expect("cmp_cast_kernel launch failed");
    let got = out.to_host_vec(&stream).expect("failed to copy output");
    ok &= check("cmp_cast_kernel", &got, &[-1, -1, 1, 0]);

    // ordering_match_kernel: issue #146 shape; Less must take the Less arm.
    let mut out = DeviceBuffer::<i32>::zeroed(&stream, 1).expect("failed to allocate output");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.ordering_match_kernel(stream.as_ref(), config, &mut out) }
        .expect("ordering_match_kernel launch failed");
    let got = out.to_host_vec(&stream).expect("failed to copy output");
    ok &= check("ordering_match_kernel", &got, &[-1]);

    // enum_repr_kernel: sparse u16 tag + negative signed i8 tag.
    let mut out = DeviceBuffer::<i32>::zeroed(&stream, 4).expect("failed to allocate output");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.enum_repr_kernel(stream.as_ref(), config, &mut out) }
        .expect("enum_repr_kernel launch failed");
    let got = out.to_host_vec(&stream).expect("failed to copy output");
    ok &= check("enum_repr_kernel", &got, &[100, 200, -5, -4]);

    if !ok {
        println!("FAIL");
        std::process::exit(1);
    }
    println!("PASS");
}
