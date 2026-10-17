/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Small local arrays consumed through iterator adapters (#399).
//!
//! A four-element array read through `iter().copied().take(n).enumerate()`,
//! with `n` a runtime value, used to leave an `alloca [4 x float]` and
//! pointer-based iterator state behind: `ScalarEvolution` could not bound the
//! trip count, so `loop-unroll` with an `upperbound` never exposed the array to
//! SROA, and the kernel read and wrote `.local`.
//!
//! The picture is mixed: the *iterator* form is register-only today, while the
//! `for k in 0..N { if k >= n { break } }` form people reach for instead is the
//! one that stays in memory. That makes this example a two-kernel test.
//! `iterator_consumed_array` is the fixed shape and carries the regression
//! assertion. `indexed_array_control` computes the same value through the loop
//! form; it is a semantic comparison, and whether it uses local memory is
//! reported rather than required, because a gate on it would fail the day that
//! loop is optimized too.
//!
//! `verify-code-shape.sh` asserts the iterator kernel has no local storage, no
//! local loads and no local stores, and it exercises its own parser against
//! fixtures before it looks at real output, so a green run cannot come from a
//! check that inspected nothing.
//!
//! Both kernels compute the same value; `main` checks every lane against a host
//! oracle for the bounds the adapters can see, including values past the end,
//! and starts each arm from a NaN-filled buffer so a lane the kernel never
//! writes cannot pass.
//!
//! Build and run with:
//!   cargo oxide run iterator_local_array_regression --arch sm_80

use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

/// Elements in the array, and therefore the widest `take` the adapters see.
const LEVELS: usize = 4;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn iterator_consumed_array(bounds: &[u32], mut out: DisjointSlice<f32>) {
        let i = thread::index_1d().get() as usize;
        let coefficients = [4.0_f32, -3.0, 4.0 / 3.0, -0.25];
        let limit = bounds[i] as usize;

        let mut acc = 0.0_f32;
        for (index, coefficient) in coefficients
            .iter()
            .copied()
            .take(limit.min(LEVELS))
            .enumerate()
        {
            acc += coefficient * (index as f32 + 1.0);
        }

        unsafe {
            *out.get_unchecked_mut(i) = acc;
        }
    }

    /// The same array and the same arithmetic behind an index loop with an early
    /// exit. This is the negative control: it is expected to keep the array in
    /// local memory, and `verify-code-shape.sh` fails if it does not.
    #[kernel]
    pub fn indexed_array_control(bounds: &[u32], mut out: DisjointSlice<f32>) {
        let i = thread::index_1d().get() as usize;
        let coefficients = [4.0_f32, -3.0, 4.0 / 3.0, -0.25];
        let limit = bounds[i] as usize;

        let mut acc = 0.0_f32;
        for index in 0..LEVELS {
            if index >= limit {
                break;
            }
            acc += coefficients[index] * (index as f32 + 1.0);
        }

        unsafe {
            *out.get_unchecked_mut(i) = acc;
        }
    }
}

// =============================================================================
// HOST
// =============================================================================

const BLOCK: u32 = 128;

/// The bounds each lane sees, cycled over the block.
///
/// The array has `LEVELS` elements, so the set has to carry the interesting
/// cases explicitly: no iteration, the last element, exactly the array, and
/// values past the end. `take(n.min(LEVELS))` saturates at `LEVELS`, so the
/// past-the-end entries exercise the clamping of the whole runtime-bound
/// expression rather than `Iterator::take` on its own.
const BOUND_CASES: [u32; 8] = [0, 1, 2, 3, 4, 5, 6, u32::MAX];

fn bounds(len: usize) -> Vec<u32> {
    (0..len)
        .map(|i| BOUND_CASES[i % BOUND_CASES.len()])
        .collect()
}

fn expected(limit: u32) -> f32 {
    let coefficients = [4.0_f32, -3.0, 4.0 / 3.0, -0.25];
    let taken = (limit as usize).min(LEVELS);
    coefficients
        .iter()
        .take(taken)
        .enumerate()
        .map(|(index, coefficient)| coefficient * (index as f32 + 1.0))
        .sum()
}

fn main() {
    use cuda_core::simt::LaunchConfig;
    use cuda_core::{CudaContext, DeviceBuffer};

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load module");

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let len = BLOCK as usize;
    let host_bounds = bounds(len);

    // Guard the coverage the description claims: if the generator is ever
    // reshaped, the past-the-end cases must not quietly disappear.
    assert!(
        host_bounds.iter().any(|b| *b == 0)
            && host_bounds.iter().any(|b| *b == LEVELS as u32)
            && host_bounds.iter().any(|b| *b > LEVELS as u32),
        "bounds must cover 0, the full array, and past its end"
    );

    let bounds_dev = DeviceBuffer::from_host(&stream, &host_bounds).unwrap();
    let want: Vec<f32> = host_bounds.iter().map(|b| expected(*b)).collect();

    let mut failed = false;
    for (name, indexed) in [
        ("iterator_consumed_array", false),
        ("indexed_array_control", true),
    ] {
        // A fresh, NaN-filled buffer per arm: a lane the kernel fails to write
        // stays NaN and cannot be mistaken for a value a previous arm left
        // there.
        let mut out_dev = DeviceBuffer::from_host(&stream, &vec![f32::NAN; len]).unwrap();

        // SAFETY: launch shape matches the buffers; every lane stays in range.
        let result = unsafe {
            if indexed {
                module.indexed_array_control(stream.as_ref(), cfg, &bounds_dev, &mut out_dev)
            } else {
                module.iterator_consumed_array(stream.as_ref(), cfg, &bounds_dev, &mut out_dev)
            }
        };
        result.expect("launch failed");

        let got = out_dev.to_host_vec(&stream).unwrap();
        let mut mismatches = 0;
        for lane in 0..len {
            let (actual, expected_value) = (got[lane], want[lane]);
            if actual.is_nan() {
                println!(
                    "✗ {name}: lane {lane} (bound {}) was never written",
                    host_bounds[lane]
                );
                mismatches += 1;
            } else if actual != expected_value {
                println!(
                    "✗ {name}: lane {lane} (bound {}) got {actual} want {expected_value}",
                    host_bounds[lane]
                );
                mismatches += 1;
            }
            if mismatches == 4 {
                println!("  ... further mismatches suppressed");
                break;
            }
        }

        if mismatches == 0 {
            println!("✓ {name}: all {len} lanes match the host oracle");
        } else {
            failed = true;
        }
    }

    if failed {
        std::process::exit(1);
    }
    println!("\nSUCCESS: local arrays through iterator adapters agree with the host");
}
