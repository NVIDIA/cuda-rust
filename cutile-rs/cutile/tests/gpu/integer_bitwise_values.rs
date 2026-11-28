/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device value checks for the integer and bitwise tile ops.
//!
//! The per-op suites (`integer_ops.rs`, `bitwise_and_bitcast_ops.rs`) only
//! assert the op appears in the emitted IR; they never run it. A lowering can
//! emit the right op with the wrong signedness, operand order, or shift kind
//! and still pass them. These tests execute each op over a tile of boundary
//! values (type min/max, high-bit-set, negatives) and check every element
//! against a Rust oracle, so a wrong result fails with the op and input named.
//!
//! Batched by op-family shape: one launch per kernel pushes a whole tile of
//! edge values through many ops at once. Host overhead stays flat; coverage is
//! the cross product of {op} x {edge value}.

use cutile::prelude::*;

use crate::audit_common::{host, upload};
use crate::common;

#[allow(clippy::too_many_arguments)]
#[cutile::module]
mod integer_bitwise_module {
    use cutile::core::*;

    /// Signed bitwise, min/max, and multiply-high over one i32 tile pair.
    #[cutile::entry()]
    fn signed_bitwise_minmax<const S: [i32; 1]>(
        and_out: &mut Tensor<i32, S>,
        or_out: &mut Tensor<i32, S>,
        xor_out: &mut Tensor<i32, S>,
        max_out: &mut Tensor<i32, S>,
        min_out: &mut Tensor<i32, S>,
        mulhi_out: &mut Tensor<i32, S>,
        x: &Tensor<i32, { [-1] }>,
        y: &Tensor<i32, { [-1] }>,
    ) {
        let xt: Tile<i32, S> = x.load_tile(shape!(S), [0i32]);
        let yt: Tile<i32, S> = y.load_tile(shape!(S), [0i32]);
        and_out.store(andi(xt, yt));
        or_out.store(ori(xt, yt));
        xor_out.store(xori(xt, yt));
        max_out.store(maxi(xt, yt));
        min_out.store(mini(xt, yt));
        mulhi_out.store(mulhii(xt, yt));
    }

    /// Signed arithmetic right shift and integer unary ops.
    ///
    /// `shli` is deliberately absent: `shli(_, _, overflow::None)` fails
    /// bytecode serialization (#1894 — the writer requires an `overflow`
    /// attribute on ShLI, which `None` omits), so a left-shift value test is
    /// blocked until that is fixed and belongs with the fix as its regression.
    #[cutile::entry()]
    fn signed_shift_unary<const S: [i32; 1]>(
        shr_out: &mut Tensor<i32, S>,
        neg_out: &mut Tensor<i32, S>,
        abs_out: &mut Tensor<i32, S>,
        x: &Tensor<i32, { [-1] }>,
        s: &Tensor<i32, { [-1] }>,
    ) {
        let xt: Tile<i32, S> = x.load_tile(shape!(S), [0i32]);
        let st: Tile<i32, S> = s.load_tile(shape!(S), [0i32]);
        shr_out.store(shri(xt, st));
        neg_out.store(negi(xt, overflow::None));
        abs_out.store(absi(xt));
    }

    /// Unsigned bitwise, min/max, and the logical (zero-fill) right shift;
    /// max/min must compare unsigned. `shli` omitted for the same reason as
    /// the signed kernel above.
    #[cutile::entry()]
    fn unsigned_ops<const S: [i32; 1]>(
        and_out: &mut Tensor<u32, S>,
        or_out: &mut Tensor<u32, S>,
        xor_out: &mut Tensor<u32, S>,
        max_out: &mut Tensor<u32, S>,
        min_out: &mut Tensor<u32, S>,
        shr_out: &mut Tensor<u32, S>,
        x: &Tensor<u32, { [-1] }>,
        y: &Tensor<u32, { [-1] }>,
        s: &Tensor<u32, { [-1] }>,
    ) {
        let xt: Tile<u32, S> = x.load_tile(shape!(S), [0i32]);
        let yt: Tile<u32, S> = y.load_tile(shape!(S), [0i32]);
        let st: Tile<u32, S> = s.load_tile(shape!(S), [0i32]);
        and_out.store(andi(xt, yt));
        or_out.store(ori(xt, yt));
        xor_out.store(xori(xt, yt));
        max_out.store(maxi(xt, yt));
        min_out.store(mini(xt, yt));
        shr_out.store(shri(xt, st));
    }
}

fn zeros_i32(n: usize) -> cutile::tensor::Tensor<i32> {
    api::zeros::<i32>(&[n]).sync().expect("zeros i32")
}
fn zeros_u32(n: usize) -> cutile::tensor::Tensor<u32> {
    api::zeros::<u32>(&[n]).sync().expect("zeros u32")
}

/// Assert element-by-element so a failure names the op, index, and inputs.
fn check_i32(got: &[i32], want: &[i32], op: &str, x: &[i32], y: &[i32]) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g, w,
            "{op} at [{i}]: x={}, y={}: got {g}, want {w}",
            x[i], y[i]
        );
    }
}
fn check_u32(got: &[u32], want: &[u32], op: &str, x: &[u32], y: &[u32]) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g, w,
            "{op} at [{i}]: x={:#x}, y={:#x}: got {g:#x}, want {w:#x}",
            x[i], y[i]
        );
    }
}

#[test]
fn signed_bitwise_minmax_and_mulhi_match_rust() {
    common::with_test_stack(|| {
        const N: usize = 8;
        // Boundary values: zero, +/-1, type min/max, and alternating bit
        // patterns that make wrong operand order or signedness visible.
        let x: Vec<i32> = vec![0, 1, -1, i32::MIN, i32::MAX, -2, 0x5555_5555, -0x5000_0000];
        let y: Vec<i32> = vec![0, -1, i32::MAX, 1, i32::MIN, 0x0F0F_0F0F, -256, 3];

        let (and_o, or_o, xor_o, max_o, min_o, mulhi_o, _x, _y) =
            integer_bitwise_module::signed_bitwise_minmax(
                zeros_i32(N).partition([N]),
                zeros_i32(N).partition([N]),
                zeros_i32(N).partition([N]),
                zeros_i32(N).partition([N]),
                zeros_i32(N).partition([N]),
                zeros_i32(N).partition([N]),
                upload(x.clone()),
                upload(y.clone()),
            )
            .generics(vec![N.to_string()])
            .grid((1, 1, 1))
            .sync()
            .expect("signed_bitwise_minmax");

        let z = |f: &dyn Fn(i32, i32) -> i32| -> Vec<i32> {
            x.iter().zip(&y).map(|(&a, &b)| f(a, b)).collect()
        };
        check_i32(&host(&and_o.unpartition()), &z(&|a, b| a & b), "andi", &x, &y);
        check_i32(&host(&or_o.unpartition()), &z(&|a, b| a | b), "ori", &x, &y);
        check_i32(&host(&xor_o.unpartition()), &z(&|a, b| a ^ b), "xori", &x, &y);
        check_i32(&host(&max_o.unpartition()), &z(&|a, b| a.max(b)), "maxi", &x, &y);
        check_i32(&host(&min_o.unpartition()), &z(&|a, b| a.min(b)), "mini", &x, &y);
        // mulhii carries no signedness attribute: it takes the high 32 bits of
        // the UNSIGNED 64-bit product of the operand bit patterns, regardless
        // of element type. (A signed high-multiply would differ for operands
        // with the top bit set.)
        let want_mulhi = z(&|a, b| {
            (((a as u32 as u64) * (b as u32 as u64)) >> 32) as u32 as i32
        });
        check_i32(&host(&mulhi_o.unpartition()), &want_mulhi, "mulhii", &x, &y);
    });
}

#[test]
fn signed_shifts_are_arithmetic_and_unary_ops_match_rust() {
    common::with_test_stack(|| {
        const N: usize = 8;
        // Negatives included: a logical (zero-fill) right shift would differ
        // from Rust's arithmetic `>>` on i32 in the high bits.
        let x: Vec<i32> = vec![-1, i32::MIN, i32::MAX, -256, 1, -1024, 0x4000_0000, -7];
        let s: Vec<i32> = vec![0, 1, 4, 8, 31, 16, 30, 3];

        let (shr_o, neg_o, abs_o, _x, _s) = integer_bitwise_module::signed_shift_unary(
            zeros_i32(N).partition([N]),
            zeros_i32(N).partition([N]),
            zeros_i32(N).partition([N]),
            upload(x.clone()),
            upload(s.clone()),
        )
        .generics(vec![N.to_string()])
        .grid((1, 1, 1))
        .sync()
        .expect("signed_shift_unary");

        // i32 `>>` is arithmetic: sign bits fill from the left.
        let shr: Vec<i32> = x.iter().zip(&s).map(|(&a, &b)| a >> b).collect();
        // overflow::None is wrapping: neg/abs of i32::MIN stay i32::MIN.
        let neg: Vec<i32> = x.iter().map(|&a| a.wrapping_neg()).collect();
        let abs: Vec<i32> = x.iter().map(|&a| a.wrapping_abs()).collect();
        check_i32(&host(&shr_o.unpartition()), &shr, "shri(arith)", &x, &s);
        check_i32(&host(&neg_o.unpartition()), &neg, "negi", &x, &x);
        check_i32(&host(&abs_o.unpartition()), &abs, "absi", &x, &x);
    });
}

#[test]
fn unsigned_ops_compare_and_shift_without_sign() {
    common::with_test_stack(|| {
        const N: usize = 8;
        // Every value has the top bit set somewhere: a signed max/min or an
        // arithmetic right shift would disagree with the unsigned oracle.
        let x: Vec<u32> = vec![
            0, 1, u32::MAX, 0x8000_0000, 0xFFFF_FFF0, 0xDEAD_BEEF, 0xAAAA_AAAA, 0x7FFF_FFFF,
        ];
        let y: Vec<u32> = vec![
            0xFFFF_FFFF, 0x8000_0000, 2, 0x8000_0001, 0x0F0F_0F0F, 100, 0x5555_5555, 0x8000_0000,
        ];
        let s: Vec<u32> = vec![0, 1, 4, 31, 16, 8, 30, 15];

        let (and_o, or_o, xor_o, max_o, min_o, shr_o, _x, _y, _s) =
            integer_bitwise_module::unsigned_ops(
                zeros_u32(N).partition([N]),
                zeros_u32(N).partition([N]),
                zeros_u32(N).partition([N]),
                zeros_u32(N).partition([N]),
                zeros_u32(N).partition([N]),
                zeros_u32(N).partition([N]),
                upload(x.clone()),
                upload(y.clone()),
                upload(s.clone()),
            )
            .generics(vec![N.to_string()])
            .grid((1, 1, 1))
            .sync()
            .expect("unsigned_ops");

        let zb = |f: &dyn Fn(u32, u32) -> u32| -> Vec<u32> {
            x.iter().zip(&y).map(|(&a, &b)| f(a, b)).collect()
        };
        check_u32(&host(&and_o.unpartition()), &zb(&|a, b| a & b), "andi", &x, &y);
        check_u32(&host(&or_o.unpartition()), &zb(&|a, b| a | b), "ori", &x, &y);
        check_u32(&host(&xor_o.unpartition()), &zb(&|a, b| a ^ b), "xori", &x, &y);
        check_u32(&host(&max_o.unpartition()), &zb(&|a, b| a.max(b)), "maxi(u)", &x, &y);
        check_u32(&host(&min_o.unpartition()), &zb(&|a, b| a.min(b)), "mini(u)", &x, &y);
        // Logical (zero-fill) right shift.
        let shr: Vec<u32> = x.iter().zip(&s).map(|(&a, &b)| a >> b).collect();
        check_u32(&host(&shr_o.unpartition()), &shr, "shri(logical)", &x, &s);
    });
}
