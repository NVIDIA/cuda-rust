/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integer `/` lowers with Rust semantics — the quotient truncates toward
//! zero — and the bounds analysis (which always modelled truncation) agrees
//! with the device. The former `negative_inf` (floor) lowering disagreed for
//! negative dividends, dropping or misplacing `index >= 0` guards, and was
//! rejected by the Tile IR verifier for unsigned operands (2026-08 audit).

use cutile::prelude::*;

use crate::audit_common::{self, host, report_outcome, run_in_subprocess, upload, Outcome};
use crate::common;

#[cutile::module]
mod int_division_module {
    use cutile::core::*;

    /// Scalar `/` and `%` on runtime operands.
    #[cutile::entry()]
    fn scalar_div_rem(
        q_out: &mut Tensor<i32, { [1] }>,
        r_out: &mut Tensor<i32, { [1] }>,
        a: i32,
        b: i32,
    ) {
        let q: Tile<i32, { [] }> = scalar_to_tile(a / b);
        let r: Tile<i32, { [] }> = scalar_to_tile(a % b);
        q_out.store(q.reshape(shape![1]));
        r_out.store(r.reshape(shape![1]));
    }

    /// Element-wise `/` and `%` on signed tiles.
    #[cutile::entry()]
    fn tile_div_rem<const S: [i32; 1]>(
        q_out: &mut Tensor<i32, S>,
        r_out: &mut Tensor<i32, S>,
        x: &Tensor<i32, { [-1] }>,
        y: &Tensor<i32, { [-1] }>,
    ) {
        let xt: Tile<i32, S> = x.load_tile(shape!(S), [0i32]);
        let yt: Tile<i32, S> = y.load_tile(shape!(S), [0i32]);
        q_out.store(xt / yt);
        r_out.store(xt % yt);
    }

    /// Element-wise `/` and `%` on unsigned tiles (rejected by the Tile IR
    /// verifier under the former floor rounding).
    #[cutile::entry()]
    fn u32_tile_div_rem<const S: [i32; 1]>(
        q_out: &mut Tensor<u32, S>,
        r_out: &mut Tensor<u32, S>,
        x: &Tensor<u32, { [-1] }>,
        y: &Tensor<u32, { [-1] }>,
    ) {
        let xt: Tile<u32, S> = x.load_tile(shape!(S), [0i32]);
        let yt: Tile<u32, S> = y.load_tile(shape!(S), [0i32]);
        q_out.store(xt / yt);
        r_out.store(xt % yt);
    }

    /// `ceil_div` keeps its round-toward-positive-infinity lowering.
    #[cutile::entry()]
    fn scalar_ceil_div(out: &mut Tensor<i32, { [1] }>, a: i32, b: i32) {
        let q: Tile<i32, { [] }> = scalar_to_tile(ceil_div(a, b));
        out.store(q.reshape(shape![1]));
    }

    /// An index computed with `/` from a runtime scalar: the access must
    /// stay guarded, and the guard must agree with Rust (`-1 / B == 0`).
    #[cutile::entry()]
    fn div_index<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [-1] }>, n: i32) {
        let p = x.partition(shape![B]);
        let idx = n / B;
        z.store(p.load([idx]));
    }

    /// `i / 16` over `-15..32` ranges over `[0, 1]` under truncation, so the
    /// compiler discharges the check statically; the device division must
    /// truncate too, or the loads for `i < 0` read tile `-1` unchecked.
    #[cutile::entry()]
    fn trunc_div_static_discharge(z: &mut Tensor<f32, { [16] }>, x: &Tensor<f32, { [32] }>) {
        let p = x.partition(shape![16]);
        let mut acc: Tile<f32, { [16] }> = constant(0.0, shape![16]);
        for i in -15i32..32i32 {
            acc = acc + p.load([i / 16i32]);
        }
        z.store(acc);
    }
}

use int_division_module::__module_ast_self;

const B: usize = 16;

fn compile(
    function_name: &str,
    generics: &[&str],
    strides: &[(&str, &[i32])],
) -> Result<(String, cutile::compile_api::CheckPlacementCounts), cutile_compiler::error::JITError> {
    audit_common::compile(
        __module_ast_self,
        "int_division_module",
        function_name,
        generics,
        strides,
    )
}

fn compile_ok(function_name: &str, generics: &[&str], strides: &[(&str, &[i32])]) -> String {
    compile(function_name, generics, strides)
        .unwrap_or_else(|err| panic!("{function_name} should compile: {err}"))
        .0
}

/// The first IR line whose op is `op` (a whole token, so that e.g. the module
/// name `int_division_module` does not match `divi`).
fn op_line<'a>(ir: &'a str, op: &str) -> &'a str {
    ir.lines()
        .find(|line| line.split_whitespace().any(|token| token == op))
        .unwrap_or_else(|| panic!("expected a {op} op:\n{ir}"))
}

#[test]
fn integer_division_lowers_with_truncation_and_ceil_div_keeps_positive_inf() {
    common::with_test_stack(|| {
        let ir = compile_ok("scalar_div_rem", &[], &[("q_out", &[1]), ("r_out", &[1])]);
        let divi = op_line(&ir, "divi");
        assert!(
            !divi.contains("negative_inf"),
            "`/` must not round toward negative infinity: {divi}"
        );
        assert!(
            divi.contains("signed"),
            "`/` on i32 must carry the signed attribute: {divi}"
        );
        let ir = compile_ok("scalar_ceil_div", &[], &[("out", &[1])]);
        let divi = op_line(&ir, "divi");
        assert!(
            divi.contains("positive_inf"),
            "`ceil_div` must round toward positive infinity: {divi}"
        );
        let ir = compile_ok(
            "u32_tile_div_rem",
            &["8"],
            &[("q_out", &[1]), ("r_out", &[1]), ("x", &[1]), ("y", &[1])],
        );
        for op in ["divi", "remi"] {
            let line = op_line(&ir, op);
            assert!(
                line.contains("unsigned"),
                "u32 {op} must be unsigned: {line}"
            );
        }
    });
}

#[test]
fn scalar_division_matches_rust_for_negative_operands() {
    common::with_test_stack(|| {
        for (a, b) in [
            (-7i32, 2i32),
            (7, -2),
            (-7, -2),
            (7, 2),
            (-9, 4),
            (i32::MIN + 1, 3),
        ] {
            let (q, r, _a, _b) = int_division_module::scalar_div_rem(
                api::zeros::<i32>(&[1]).partition([1]),
                api::zeros::<i32>(&[1]).partition([1]),
                a,
                b,
            )
            .grid((1, 1, 1))
            .sync()
            .expect("scalar_div_rem");
            assert_eq!(host(&q.unpartition()), vec![a / b], "{a} / {b}");
            assert_eq!(host(&r.unpartition()), vec![a % b], "{a} % {b}");
        }
        let (out, _a, _b) =
            int_division_module::scalar_ceil_div(api::zeros::<i32>(&[1]).partition([1]), 7, 2)
                .grid((1, 1, 1))
                .sync()
                .expect("scalar_ceil_div");
        assert_eq!(host(&out.unpartition()), vec![4]);
    });
}

#[test]
fn tile_division_matches_rust_for_signed_and_unsigned_elements() {
    common::with_test_stack(|| {
        let n = 8usize;
        let xs: Vec<i32> = vec![-7, 7, -7, 7, -9, 9, i32::MIN + 1, -1];
        let ys: Vec<i32> = vec![2, -2, -2, 2, 4, -4, 3, 16];
        let (q, r, _x, _y) = int_division_module::tile_div_rem(
            api::zeros::<i32>(&[n]).partition([n]),
            api::zeros::<i32>(&[n]).partition([n]),
            upload(xs.clone()),
            upload(ys.clone()),
        )
        .generics(vec![n.to_string()])
        .grid((1, 1, 1))
        .sync()
        .expect("tile_div_rem");
        let expected_q: Vec<i32> = xs.iter().zip(&ys).map(|(x, y)| x / y).collect();
        let expected_r: Vec<i32> = xs.iter().zip(&ys).map(|(x, y)| x % y).collect();
        assert_eq!(host(&q.unpartition()), expected_q, "signed quotients");
        assert_eq!(host(&r.unpartition()), expected_r, "signed remainders");

        // Unsigned values with the top bit set: a signed lowering would
        // divide negative numbers.
        let xs: Vec<u32> = vec![
            0xFFFF_FFF0,
            0x8000_0000,
            u32::MAX,
            0xDEAD_BEEF,
            100,
            7,
            0xFFFF_FFFF,
            3,
        ];
        let ys: Vec<u32> = vec![16, 3, 2, 0x1000, 7, 100, 0x8000_0000, 2];
        let (q, r, _x, _y) = int_division_module::u32_tile_div_rem(
            api::zeros::<u32>(&[n]).partition([n]),
            api::zeros::<u32>(&[n]).partition([n]),
            upload(xs.clone()),
            upload(ys.clone()),
        )
        .generics(vec![n.to_string()])
        .grid((1, 1, 1))
        .sync()
        .expect("u32_tile_div_rem");
        let expected_q: Vec<u32> = xs.iter().zip(&ys).map(|(x, y)| x / y).collect();
        let expected_r: Vec<u32> = xs.iter().zip(&ys).map(|(x, y)| x % y).collect();
        assert_eq!(host(&q.unpartition()), expected_q, "unsigned quotients");
        assert_eq!(host(&r.unpartition()), expected_r, "unsigned remainders");
    });
}

#[test]
fn div_computed_index_stays_guarded_and_agrees_with_rust() {
    common::with_test_stack(|| {
        let (ir, counts) = compile("div_index", &["16"], &[("z", &[1]), ("x", &[1])])
            .expect("div_index should compile");
        assert!(
            ir.contains("assert"),
            "a `/`-computed runtime index must keep its device check:\n{ir}"
        );
        assert_eq!(counts.in_place, 1, "the check stays in place: {counts:?}");

        // `-1 / 16 == 0` in Rust: a valid tile, so the guard must pass and
        // the load must read tile 0. (A floor lowering gave `-1`, which
        // the `0 <= index` guard rejects.)
        let x = upload((0..(3 * B) as i32).map(|v| v as f32).collect());
        let (z, _x, _n) =
            int_division_module::div_index(api::zeros::<f32>(&[B]).partition([B]), x, -1)
                .generics(vec![B.to_string()])
                .sync()
                .expect("n = -1 indexes tile 0");
        let expected: Vec<f32> = (0..B).map(|v| v as f32).collect();
        assert_eq!(host(&z.unpartition()), expected);
    });
}

#[test]
fn statically_discharged_truncating_division_reads_the_right_tiles() {
    common::with_test_stack(|| {
        let (ir, counts) = compile(
            "trunc_div_static_discharge",
            &[],
            &[("z", &[1]), ("x", &[1])],
        )
        .expect("trunc_div_static_discharge should compile");
        assert_eq!(
            (counts.discharged, counts.hoisted, counts.in_place),
            (1, 0, 0),
            "the `[0, 1]` range discharges statically: {counts:?}"
        );
        assert!(!ir.contains("assert"), "no device check expected:\n{ir}");

        // 15 loads of tile 0 for `i in -15..0`, 16 more for `i in 0..16`,
        // and 16 loads of tile 1: element j is `31 * j + 16 * (16 + j)`.
        let x = upload((0..32).map(|v| v as f32).collect());
        let (z, _x) = int_division_module::trunc_div_static_discharge(
            api::zeros::<f32>(&[16]).partition([16]),
            x,
        )
        .sync()
        .expect("trunc_div_static_discharge");
        let expected: Vec<f32> = (0..16).map(|j| (47 * j + 256) as f32).collect();
        assert_eq!(host(&z.unpartition()), expected);
    });
}

// ---------------------------------------------------------------------------
// The out-of-range twin must still stop (subprocess: a device trap poisons
// the CUDA context).
// ---------------------------------------------------------------------------

const CASE_ENV: &str = "CUTILE_AUDIT_INT_DIVISION_CASE";

fn execute_trap_case(case: &str) -> Result<(), String> {
    match case {
        // Three tiles; `n / B == 3` is one past the end.
        "div_index_out_of_range" => {
            let x = upload((0..(3 * B) as i32).map(|v| v as f32).collect());
            int_division_module::div_index(
                api::zeros::<f32>(&[B]).partition([B]),
                x,
                (3 * B) as i32,
            )
            .generics(vec![B.to_string()])
            .sync()
            .map_err(|err| err.to_string())?;
            Ok(())
        }
        other => Err(format!("unknown case {other}")),
    }
}

/// Subprocess entry point; see `audit_common::run_in_subprocess`.
#[test]
#[ignore]
fn int_division_case_runner() {
    let case = std::env::var(CASE_ENV).expect("case env var not set");
    common::with_test_stack(move || {
        report_outcome(std::panic::catch_unwind(|| execute_trap_case(&case)));
    });
}

#[test]
fn div_computed_index_out_of_range_still_stops() {
    let case = "div_index_out_of_range";
    match run_in_subprocess(
        "audit_int_division::int_division_case_runner",
        CASE_ENV,
        case,
    ) {
        Outcome::Stop(msg) => eprintln!("{case}: stopped as expected: {msg}"),
        Outcome::Ok => panic!("{case}: an out-of-range access ran to completion"),
    }
}
