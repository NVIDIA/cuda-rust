/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Narrow unsigned element types (`u8`, `u16`) must be unsigned in every
//! signedness-carrying op. Tile IR stores them as `i8`/`i16`, so the Rust
//! type name is the only place the unsignedness survives; two of the three
//! lowering sites used to list only `u32`/`u64` and compiled `u8` values
//! `>= 128` as negative (2026-08 codegen audit).

use cutile::prelude::*;

use crate::audit_common::{host, upload};
use crate::common;

#[cutile::module]
mod signedness_module {
    use cutile::core::*;

    /// `u8` tiles through every op family: the binary operators
    /// (`compile_binary_op`), the named ops (`compile_cuda_tile_op`), and
    /// `cmpi`.
    #[cutile::entry()]
    fn u8_tile_ops<const S: [i32; 1]>(
        q: &mut Tensor<u8, S>,
        r: &mut Tensor<u8, S>,
        mx: &mut Tensor<u8, S>,
        mn: &mut Tensor<u8, S>,
        gt: &mut Tensor<u8, S>,
        x: &Tensor<u8, { [-1] }>,
        y: &Tensor<u8, { [-1] }>,
    ) {
        let xt: Tile<u8, S> = x.load_tile(shape!(S), [0i32]);
        let yt: Tile<u8, S> = y.load_tile(shape!(S), [0i32]);
        q.store(xt / yt);
        r.store(xt % yt);
        mx.store(maxi(xt, yt));
        mn.store(mini(xt, yt));
        let one: Tile<u8, S> = constant(1u8, shape!(S));
        let zero: Tile<u8, S> = constant(0u8, shape!(S));
        gt.store(select(cmpi(xt, yt, predicate::GreaterThan), one, zero));
    }

    /// Scalar `u8` comparison and division (the scalar `cmpi`/`divi` path).
    #[cutile::entry()]
    fn u8_scalar_ops(out: &mut Tensor<i32, { [1] }>, a: u8, b: u8) {
        let gt: i32 = if a > b { 1i32 } else { 0i32 };
        let big_quotient: i32 = if a / b >= 60u8 { 10i32 } else { 0i32 };
        let t: Tile<i32, { [] }> = scalar_to_tile(gt + big_quotient);
        out.store(t.reshape(shape![1]));
    }
}

#[test]
fn u8_values_above_127_are_unsigned_in_every_op_family() {
    common::with_test_stack(|| {
        let n = 8usize;
        let xs: Vec<u8> = vec![200, 255, 130, 128, 129, 250, 100, 3];
        let ys: Vec<u8> = vec![3, 100, 129, 255, 2, 7, 200, 129];
        let (q, r, mx, mn, gt, _x, _y) = signedness_module::u8_tile_ops(
            api::zeros::<u8>(&[n]).partition([n]),
            api::zeros::<u8>(&[n]).partition([n]),
            api::zeros::<u8>(&[n]).partition([n]),
            api::zeros::<u8>(&[n]).partition([n]),
            api::zeros::<u8>(&[n]).partition([n]),
            upload(xs.clone()),
            upload(ys.clone()),
        )
        .generics(vec![n.to_string()])
        .grid((1, 1, 1))
        .sync()
        .expect("u8_tile_ops");
        let pairs = || xs.iter().copied().zip(ys.iter().copied());
        assert_eq!(
            host(&q.unpartition()),
            pairs().map(|(x, y)| x / y).collect::<Vec<u8>>(),
            "u8 quotients"
        );
        assert_eq!(
            host(&r.unpartition()),
            pairs().map(|(x, y)| x % y).collect::<Vec<u8>>(),
            "u8 remainders"
        );
        assert_eq!(
            host(&mx.unpartition()),
            pairs().map(|(x, y)| x.max(y)).collect::<Vec<u8>>(),
            "u8 maxi"
        );
        assert_eq!(
            host(&mn.unpartition()),
            pairs().map(|(x, y)| x.min(y)).collect::<Vec<u8>>(),
            "u8 mini"
        );
        assert_eq!(
            host(&gt.unpartition()),
            pairs().map(|(x, y)| (x > y) as u8).collect::<Vec<u8>>(),
            "u8 cmpi greater_than"
        );

        // Scalar path: 200 > 3, and 200 / 3 == 66 >= 60 (signed: -56).
        let (out, _a, _b) =
            signedness_module::u8_scalar_ops(api::zeros::<i32>(&[1]).partition([1]), 200u8, 3u8)
                .grid((1, 1, 1))
                .sync()
                .expect("u8_scalar_ops");
        assert_eq!(host(&out.unpartition()), vec![11]);
    });
}
