/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shifting the result of a warp reduction (#1328).
//!
//! A `redux.sync` result used to reach the shift lowering carrying the
//! intrinsic catalog's `ui32` representation while the shift count was
//! signless. `convert_shift` saw "operands differ", compared widths, found
//! them equal and emitted `llvm.trunc i32 -> ui32` — which LLVM rejects,
//! because `trunc` needs a strictly smaller result. The whole device module
//! failed verification, so a plain `maximum << 8` did not compile.
//!
//! The reduction results are translated to signless integers now, but nothing
//! covered the shape: `redux_minmax` writes a reduction straight to memory and
//! `redux_sum` consumes it without an operator in between. This example puts
//! the operators back on and checks the values on the device:
//!
//!   * `<<` / `>>` after `redux.sync.max.u32` — unsigned, so the right shift
//!     must stay logical even when the top bit is set;
//!   * `>>` after `redux.sync.min.s32` — signed, so it must become arithmetic
//!     on negative lanes;
//!   * `wrapping_shl` with counts at and past the bit width, which is the
//!     mask-the-count behaviour `convert_shift` implements;
//!   * a multi-warp block whose warps hold different values, so a warp's
//!     reduction cannot leak into its neighbour's answer unnoticed.
//!
//! Every lane writes its own result: `redux.sync` broadcasts to the lanes named
//! by the member mask, so the shifted value has to agree across all 32.
//!
//! Build and run with:
//!   cargo oxide run redux_shift_regression --arch sm_80

use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;

const FULL_MASK: u32 = 0xffff_ffff;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Lane `i` shifts the unsigned maximum of its own warp left.
    #[kernel]
    pub fn shl_u32(input: &[u32], count: &[u32], mut out: DisjointSlice<u32>) {
        let i = thread::index_1d().get() as usize;
        let maximum = warp::redux_sync_max_u32(FULL_MASK, input[i]);
        let shifted = maximum << (count[i] & 31);
        unsafe {
            *out.get_unchecked_mut(i) = shifted;
        }
    }

    /// The same reduction shifted right: unsigned, so the top bit shifts in as
    /// zero. A signless lowering that picked `ashr` would answer differently on
    /// the patterns that set bit 31.
    #[kernel]
    pub fn shr_u32(input: &[u32], count: &[u32], mut out: DisjointSlice<u32>) {
        let i = thread::index_1d().get() as usize;
        let maximum = warp::redux_sync_max_u32(FULL_MASK, input[i]);
        let shifted = maximum >> (count[i] & 31);
        unsafe {
            *out.get_unchecked_mut(i) = shifted;
        }
    }

    /// A signed reduction shifted right: negative inputs make `ashr` and `lshr`
    /// disagree, which is the control for the case above.
    #[kernel]
    pub fn shr_i32(input: &[i32], count: &[u32], mut out: DisjointSlice<i32>) {
        let i = thread::index_1d().get() as usize;
        let minimum = warp::redux_sync_min_i32(FULL_MASK, input[i]);
        let shifted = minimum >> (count[i] & 31);
        unsafe {
            *out.get_unchecked_mut(i) = shifted;
        }
    }

    /// Counts at and past the bit width: `wrapping_shl` and the lowering both
    /// mask with `bit_width - 1`, so 32 shifts by 0 and 33 by 1.
    #[kernel]
    pub fn wrapping_shl_u32(input: &[u32], count: &[u32], mut out: DisjointSlice<u32>) {
        let i = thread::index_1d().get() as usize;
        let maximum = warp::redux_sync_max_u32(FULL_MASK, input[i]);
        let shifted = maximum.wrapping_shl(count[i]);
        unsafe {
            *out.get_unchecked_mut(i) = shifted;
        }
    }
}

// =============================================================================
// HOST
// =============================================================================

/// Threads per block: four warps, so the multi-warp case is the default one.
const BLOCK: u32 = 128;

/// Fixed-seed LCG, so the "random" pattern is reproducible without a crate.
fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

/// `ramp`: lane 0..31 counted up, offset by the warp, so each warp has a
/// different unsigned max. Warps that read each other's answer cannot produce
/// the expected values.
fn ramp(len: usize) -> Vec<u32> {
    (0..len)
        .map(|i| (i as u32 / 32) * 64 + (i % 32) as u32)
        .collect()
}

/// `extremes`: each warp carries 0, 1, `0x8000_0000` and `0xffff_ffff`, so the
/// unsigned max sets every bit and the signed min is `i32::MIN` on those lanes.
fn extremes(len: usize) -> Vec<u32> {
    (0..len)
        .map(|i| match i % 32 {
            0 => 0,
            1 => 1,
            2 => 0x8000_0000,
            3 => 0xffff_ffff,
            lane => (lane as u32) << 8,
        })
        .collect()
}

/// `random`: fixed-seed values with no structure a constant folder could use.
fn random(len: usize) -> Vec<u32> {
    let mut state = 0x1234_5678;
    (0..len).map(|_| lcg(&mut state)).collect()
}

/// Bitwise inverse of a slice, used as an output poison.
fn invert(values: &[u32]) -> Vec<u32> {
    values.iter().map(|value| !value).collect()
}

fn invert_i32(values: &[i32]) -> Vec<i32> {
    values.iter().map(|value| !value).collect()
}

fn expected_u32(input: &[u32], count: &[u32], shift: impl Fn(u32, u32) -> u32) -> Vec<u32> {
    (0..input.len())
        .map(|i| {
            let base = i & !31;
            let maximum = input[base..base + 32].iter().copied().max().unwrap();
            shift(maximum, count[i] & 31)
        })
        .collect()
}

fn expected_i32(input: &[i32], count: &[u32], shift: impl Fn(i32, u32) -> i32) -> Vec<i32> {
    (0..input.len())
        .map(|i| {
            let base = i & !31;
            let minimum = input[base..base + 32].iter().copied().min().unwrap();
            shift(minimum, count[i] & 31)
        })
        .collect()
}

fn counts(len: usize, values: &[u32]) -> Vec<u32> {
    (0..len).map(|i| values[i % values.len()]).collect()
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
    let mut failed = false;

    let patterns: [(&str, Vec<u32>); 3] = [
        ("ramp", ramp(len)),
        ("extremes", extremes(len)),
        ("random", random(len)),
    ];
    let in_range = counts(len, &[0, 1, 8, 31]);
    let past_width = counts(len, &[32, 33, 63]);

    // ===== Test 1: left shift, and the logical right shift control =====
    println!("--- Test 1: redux.sync.max.u32 << and >> ---");
    for (name, input) in &patterns {
        let input_dev = DeviceBuffer::from_host(&stream, input).unwrap();
        let count_dev = DeviceBuffer::from_host(&stream, &in_range).unwrap();
        let want_shl = expected_u32(input, &in_range, |v, c| v << c);
        let want_shr = expected_u32(input, &in_range, |v, c| v >> c);
        // Initialised to the bitwise inverse of the oracle: a lane the kernel
        // never writes stays at a value no correct answer can equal.
        let mut shl_dev = DeviceBuffer::from_host(&stream, &invert(&want_shl)).unwrap();
        let mut shr_dev = DeviceBuffer::from_host(&stream, &invert(&want_shr)).unwrap();

        // SAFETY: launch shape matches the buffers; every lane stays in range.
        unsafe { module.shl_u32(stream.as_ref(), cfg, &input_dev, &count_dev, &mut shl_dev) }
            .expect("shl_u32 launch failed");
        // SAFETY: as above.
        unsafe { module.shr_u32(stream.as_ref(), cfg, &input_dev, &count_dev, &mut shr_dev) }
            .expect("shr_u32 launch failed");

        let shl = shl_dev.to_host_vec(&stream).unwrap();
        let shr = shr_dev.to_host_vec(&stream).unwrap();

        if shl == want_shl {
            println!("✓ {name}: << matches host for all {len} lanes");
        } else {
            println!("✗ {name}: << mismatch");
            println!("  got  {:?}", &shl[..8]);
            println!("  want {:?}", &want_shl[..8]);
            failed = true;
        }
        if shr == want_shr {
            println!("✓ {name}: >> is logical for all {len} lanes");
        } else {
            println!("✗ {name}: >> mismatch");
            println!("  got  {:?}", &shr[..8]);
            println!("  want {:?}", &want_shr[..8]);
            failed = true;
        }

        // The logical claim is only tested if some answer has bit 31 set.
        if name == &"extremes" && want_shr.iter().all(|v| v >> 31 == 0) {
            panic!("the logical-shift control never set bit 31");
        }
    }

    // ===== Test 2: signed right shift stays arithmetic =====
    println!("\n--- Test 2: redux.sync.min.s32 >> ---");
    {
        let signed: Vec<i32> = extremes(len).iter().map(|v| *v as i32).collect();
        let want = expected_i32(&signed, &in_range, |v, c| v >> c);
        let input_dev = DeviceBuffer::from_host(&stream, &signed).unwrap();
        let count_dev = DeviceBuffer::from_host(&stream, &in_range).unwrap();
        let mut out_dev = DeviceBuffer::from_host(&stream, &invert_i32(&want)).unwrap();

        // SAFETY: launch shape matches the buffers; every lane stays in range.
        unsafe { module.shr_i32(stream.as_ref(), cfg, &input_dev, &count_dev, &mut out_dev) }
            .expect("shr_i32 launch failed");

        let got = out_dev.to_host_vec(&stream).unwrap();
        if got == want {
            println!("✓ signed >> is arithmetic for all {len} lanes");
        } else {
            println!("✗ signed >> mismatch");
            println!("  got  {:?}", &got[..8]);
            println!("  want {:?}", &want[..8]);
            failed = true;
        }
        if want.iter().all(|v| *v >= 0) {
            panic!("the arithmetic-shift control never produced a negative answer");
        }
    }

    // ===== Test 3: counts at and past the bit width =====
    println!("\n--- Test 3: wrapping_shl with counts >= 32 ---");
    {
        let input = extremes(len);
        let want = expected_u32(&input, &past_width, |v, c| v.wrapping_shl(c));
        let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let count_dev = DeviceBuffer::from_host(&stream, &past_width).unwrap();
        let mut out_dev = DeviceBuffer::from_host(&stream, &invert(&want)).unwrap();

        // SAFETY: launch shape matches the buffers; every lane stays in range.
        unsafe {
            module.wrapping_shl_u32(stream.as_ref(), cfg, &input_dev, &count_dev, &mut out_dev)
        }
        .expect("wrapping_shl_u32 launch failed");

        let got = out_dev.to_host_vec(&stream).unwrap();
        if got == want {
            println!("✓ counts 32/33/63 mask to 0/1/31 for all {len} lanes");
        } else {
            println!("✗ wrapping shift mismatch");
            println!("  got  {:?}", &got[..8]);
            println!("  want {:?}", &want[..8]);
            failed = true;
        }
    }

    if failed {
        std::process::exit(1);
    }
    println!("\nSUCCESS: shifted warp reductions match the host on all lanes");
}
