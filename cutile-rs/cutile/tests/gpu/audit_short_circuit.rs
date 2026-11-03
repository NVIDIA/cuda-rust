/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `&&` / `||` short-circuit: the right operand is compiled into a
//! `cuda_tile.if` region and runs only when the left operand does not decide
//! the result. The former eager `andi`/`ori` lowering executed
//! `idx % n` in `n != 0 && idx % n == 0` for `n == 0` (2026-08 audit).

use cutile::prelude::*;

use crate::audit_common::{self, host};
use crate::common;

#[cutile::module]
mod short_circuit_module {
    use cutile::core::*;

    static HITS: Global<AtomicI32, { [] }> = Global::new(0i32);

    /// Bumps the counter and returns `true`: a right operand with a side
    /// effect, so evaluation is observable.
    fn bump_hits() -> bool {
        let (old, _load_token) = HITS.load(ordering::Acquire, scope::Device);
        let _store_token = HITS.store(
            old + constant(1i32, shape![]),
            ordering::Release,
            scope::Device,
        );
        true
    }

    /// Writes `hits + 100 * taken`. The test drives `flag` with 1 and 3 —
    /// values with the same divisibility hint, so both launches share one
    /// specialization and therefore one copy of `HITS`.
    #[cutile::entry()]
    fn short_circuit_and(out: &mut Tensor<i32, { [1] }>, flag: i32) {
        let taken = flag > 2i32 && bump_hits();
        let t: i32 = if taken { 1i32 } else { 0i32 };
        let (hits, _load_token) = HITS.load(ordering::Acquire, scope::Device);
        let encoded: Tile<i32, { [] }> = hits + scalar_to_tile(t * 100i32);
        out.store(encoded.reshape(shape![1]));
    }

    /// Writes `hits + 100 * taken` (see `short_circuit_and` for the flags).
    #[cutile::entry()]
    fn short_circuit_or(out: &mut Tensor<i32, { [1] }>, flag: i32) {
        let taken = flag < 2i32 || bump_hits();
        let t: i32 = if taken { 1i32 } else { 0i32 };
        let (hits, _load_token) = HITS.load(ordering::Acquire, scope::Device);
        let encoded: Tile<i32, { [] }> = hits + scalar_to_tile(t * 100i32);
        out.store(encoded.reshape(shape![1]));
    }

    /// The motivating shape: the remainder must not be computed for `n == 0`.
    #[cutile::entry()]
    fn guarded_rem(out: &mut Tensor<i32, { [1] }>, idx: i32, n: i32) {
        let divisible = n != 0i32 && idx % n == 0i32;
        let t: i32 = if divisible { 1i32 } else { 0i32 };
        let tile: Tile<i32, { [] }> = scalar_to_tile(t);
        out.store(tile.reshape(shape![1]));
    }
}

use short_circuit_module::__module_ast_self;

#[test]
fn logical_operators_lower_to_conditional_regions() {
    common::with_test_stack(|| {
        let (ir, _) = audit_common::compile(
            __module_ast_self,
            "short_circuit_module",
            "guarded_rem",
            &[],
            &[("out", &[1])],
        )
        .expect("guarded_rem should compile");
        assert!(
            !ir.contains("andi"),
            "`&&` must not lower to an eager `andi`:\n{ir}"
        );
        // The remainder is evaluated only inside the conditional region.
        let if_pos = ir
            .find("if ")
            .unwrap_or_else(|| panic!("expected an if op:\n{ir}"));
        let rem_pos = ir
            .find("remi")
            .unwrap_or_else(|| panic!("expected a remi op:\n{ir}"));
        assert!(
            if_pos < rem_pos,
            "the remainder must sit inside the `if` region:\n{ir}"
        );
    });
}

#[test]
fn logical_operators_short_circuit_on_the_device() {
    common::with_test_stack(|| {
        let and_launch = |flag: i32| {
            let (out, _flag) = short_circuit_module::short_circuit_and(
                api::zeros::<i32>(&[1]).partition([1]),
                flag,
            )
            .grid((1, 1, 1))
            .sync()
            .expect("short_circuit_and");
            host(&out.unpartition())[0]
        };
        let or_launch = |flag: i32| {
            let (out, _flag) = short_circuit_module::short_circuit_or(
                api::zeros::<i32>(&[1]).partition([1]),
                flag,
            )
            .grid((1, 1, 1))
            .sync()
            .expect("short_circuit_or");
            host(&out.unpartition())[0]
        };
        // `HITS` lives in the loaded kernel module, so it persists across
        // launches of one specialization (flags 1 and 3 share one) and only
        // this test touches it; the two kernels are two specializations, each
        // with its own copy. Each result encodes `hits + 100 * taken`.
        assert_eq!(and_launch(1), 0, "`false && bump()` must not bump");
        assert_eq!(and_launch(3), 101, "`true && bump()` bumps once");
        assert_eq!(and_launch(1), 1, "`false && bump()` leaves the count");
        assert_eq!(or_launch(1), 100, "`true || bump()` must not bump");
        assert_eq!(or_launch(3), 101, "`false || bump()` bumps once");
        assert_eq!(or_launch(1), 101, "`true || bump()` leaves the count");

        // The motivating shape: `n == 0` never reaches the remainder.
        for (idx, n, expected) in [(6, 0, 0), (6, 3, 1), (7, 3, 0)] {
            let (out, _idx, _n) =
                short_circuit_module::guarded_rem(api::zeros::<i32>(&[1]).partition([1]), idx, n)
                    .grid((1, 1, 1))
                    .sync()
                    .expect("guarded_rem");
            assert_eq!(host(&out.unpartition()), vec![expected], "idx={idx} n={n}");
        }
    });
}
