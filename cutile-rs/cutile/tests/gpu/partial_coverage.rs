/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Partial-coverage launches: a `partition_prefix` output binding lets the
//! launch grid cover a per-axis PREFIX of the block grid — launched blocks
//! embed identically, uncovered blocks keep their contents — while both
//! guarded directions stay errors: exceeding the block grid on any axis
//! (genuine out-of-bounds), and the default `partition()` binding's strict
//! equality (the intent diagnostic partial coverage opts out of).
//!
//! The motivating shape: a wide-tile kernel covering only the BM-aligned
//! row prefix of a `seq_len % BM != 0` tensor, with a per-row kernel taking
//! the remainder.

use cutile::prelude::*;
use partial_coverage_module::fill_rows;

use crate::common;

#[cutile::module]
mod partial_coverage_module {
    use cutile::core::*;

    /// Writes a constant to this CTA's slab; safe, and provably so.
    #[cutile::entry(deny_in_kernel_checks = true)]
    fn fill_rows<const BM: i32, const D: i32>(out: &mut Tensor<f32, { [BM, D] }>) {
        let t: Tile<f32, { [BM, D] }> = constant(7.0, shape![BM, D]);
        out.store(t);
    }
}

const BM: usize = 16;
const D: usize = 8;
const ROWS: usize = 40; // ceil(40/16) = 3 blocks; the aligned prefix is 2.

fn generics() -> Vec<String> {
    vec![BM.to_string(), D.to_string()]
}

#[test]
fn prefix_binding_writes_the_covered_prefix_and_nothing_else() {
    common::with_test_stack(|| {
        let z = api::zeros::<f32>(&[ROWS, D])
            .sync()
            .expect("alloc")
            .partition_prefix([BM, D]);
        let (out,) = fill_rows(z)
            .generics(generics())
            .grid((2, 1, 1))
            .sync()
            .expect("a per-axis prefix launch must be accepted");
        let out = out.unpartition().to_host_vec().sync().expect("copy back");
        let covered = 2 * BM * D;
        assert!(
            out[..covered].iter().all(|&v| v == 7.0),
            "covered blocks must be written"
        );
        assert!(
            out[covered..].iter().all(|&v| v == 0.0),
            "uncovered blocks must keep their prior contents"
        );
    });
}

#[test]
fn prefix_binding_still_rejects_the_oob_direction_and_cannot_infer() {
    common::with_test_stack(|| {
        // Exceeding the block grid on an axis is genuine out-of-bounds.
        let z = api::zeros::<f32>(&[ROWS, D])
            .sync()
            .expect("alloc")
            .partition_prefix([BM, D]);
        let err = match fill_rows(z).generics(generics()).grid((4, 1, 1)).sync() {
            Ok(_) => panic!("a grid beyond the block grid must be rejected"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("exceeds"),
            "expected the per-axis excess diagnostic: {err}"
        );
        // A prefix binding is a bound, not a grid: inference must refuse.
        let z = api::zeros::<f32>(&[ROWS, D])
            .sync()
            .expect("alloc")
            .partition_prefix([BM, D]);
        let err = match fill_rows(z).generics(generics()).sync() {
            Ok(_) => panic!("a partial-coverage binding must not define the grid"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("partial-coverage"),
            "expected the inference-refusal diagnostic: {err}"
        );
    });
}

#[test]
fn default_binding_keeps_strict_equality() {
    common::with_test_stack(|| {
        let z = api::zeros::<f32>(&[ROWS, D])
            .sync()
            .expect("alloc")
            .partition([BM, D]);
        let err = match fill_rows(z).generics(generics()).grid((2, 1, 1)).sync() {
            Ok(_) => panic!("without the opt-in, a partial grid must still be rejected"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("does not match"),
            "the strict-equality diagnostic must be unchanged: {err}"
        );
    });
}
