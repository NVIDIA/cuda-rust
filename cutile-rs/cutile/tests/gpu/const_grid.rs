/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `.const_grid(..)` with a zero dimension is rejected before anything is
//! compiled, on the launch path and on `specialize()`. Before the check, a
//! `(0, 0, 0)` const grid compiled the kernel for an empty grid and then
//! launched it on the grid inferred from the partition binding.

use const_grid_module::fill;
use cutile::prelude::*;

use crate::common;

#[cutile::module]
mod const_grid_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn fill<const B: i32>(out: &mut Tensor<f32, { [B] }>) {
        let t: Tile<f32, { [B] }> = constant(3.0, shape![B]);
        out.store(t);
    }
}

const B: usize = 8;
const LEN: usize = 32;

fn out() -> Partition<Tensor<f32>> {
    api::zeros::<f32>(&[LEN])
        .sync()
        .expect("alloc")
        .partition([B])
}

fn zero_dimension_error(err: impl std::fmt::Display, grid: (u32, u32, u32)) {
    let msg = err.to_string();
    assert!(
        msg.contains("zero dimension") && msg.contains(&format!("{grid:?}")),
        "unexpected error for const_grid {grid:?}: {msg}"
    );
}

#[test]
fn a_const_grid_with_a_zero_dimension_fails_the_launch() {
    common::with_test_stack(|| {
        // `(0, 0, 0)` used to launch on the grid inferred from `out`'s
        // exact binding, (4, 1, 1), after compiling for an empty grid.
        for grid in [(0, 0, 0), (0, 1, 1), (4, 0, 1), (4, 1, 0)] {
            match fill(out())
                .generics(vec![B.to_string()])
                .const_grid(grid)
                .sync()
            {
                Ok(_) => panic!("const_grid {grid:?} must be rejected"),
                Err(err) => zero_dimension_error(err, grid),
            }
        }
    });
}

#[test]
fn a_const_grid_with_a_zero_dimension_fails_specialize() {
    common::with_test_stack(|| {
        let grid = (0, 0, 0);
        match fill(out())
            .generics(vec![B.to_string()])
            .const_grid(grid)
            .specialize()
        {
            Ok(_) => panic!("specialize() with const_grid {grid:?} must be rejected"),
            Err(err) => zero_dimension_error(err, grid),
        }
    });
}

#[test]
fn a_nonzero_const_grid_still_launches() {
    common::with_test_stack(|| {
        let grid = ((LEN / B) as u32, 1, 1);
        let (out,) = fill(out())
            .generics(vec![B.to_string()])
            .const_grid(grid)
            .sync()
            .expect("a const grid matching the binding launches");
        let host = out.unpartition().to_host_vec().sync().expect("copy back");
        assert!(host.iter().all(|&v| v == 3.0), "every block is written");
    });
}
