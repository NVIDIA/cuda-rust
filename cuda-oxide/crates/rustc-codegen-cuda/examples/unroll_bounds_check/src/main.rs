/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `#[unroll]` must preserve checked indexing and division-by-zero checks.
//! Run: cargo oxide run unroll_bounds_check

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn control(arr: &mut [u32], mut out: DisjointSlice<u32>) {
        if let Some(slot) = out.get_mut(thread::index_1d()) {
            arr[999] = 7;
            let mut acc = 0u32;
            let mut i = 0u32;
            while i < 4 {
                acc = acc.wrapping_add(i);
                i += 1;
            }
            *slot = acc;
        }
    }

    #[kernel]
    pub fn full_before(arr: &mut [u32], mut out: DisjointSlice<u32>) {
        if let Some(slot) = out.get_mut(thread::index_1d()) {
            // The original report: even an access BEFORE the loop lost its check.
            arr[999] = 7;
            let mut acc = 0u32;
            let mut i = 0u32;
            #[unroll]
            while i < 4 {
                acc = acc.wrapping_add(i);
                i += 1;
            }
            *slot = acc;
        }
    }

    #[kernel]
    pub fn full_inside(arr: &mut [u32], mut out: DisjointSlice<u32>) {
        if let Some(slot) = out.get_mut(thread::index_1d()) {
            let mut acc = 0u32;
            let mut i = 0u32;
            #[unroll]
            while i < 4 {
                arr[i as usize] = i + 1;
                acc = acc.wrapping_add(i);
                i += 1;
            }
            *slot = acc;
        }
    }

    #[kernel]
    pub fn full_after(arr: &mut [u32], mut out: DisjointSlice<u32>) {
        if let Some(slot) = out.get_mut(thread::index_1d()) {
            let mut acc = 0u32;
            let mut i = 0u32;
            #[unroll]
            while i < 4 {
                acc = acc.wrapping_add(i);
                i += 1;
            }
            arr[999] = 7;
            *slot = acc;
        }
    }

    #[kernel]
    pub fn partial_before(arr: &mut [u32], mut out: DisjointSlice<u32>, n: u32) {
        if let Some(slot) = out.get_mut(thread::index_1d()) {
            arr[999] = 7;
            let mut acc = 0u32;
            let mut i = 0u32;
            #[unroll(4)]
            while i < n {
                acc = acc.wrapping_add(i);
                i += 1;
            }
            *slot = acc;
        }
    }

    #[kernel]
    pub fn partial_inside(arr: &mut [u32], mut out: DisjointSlice<u32>, n: u32) {
        if let Some(slot) = out.get_mut(thread::index_1d()) {
            let mut acc = 0u32;
            let mut i = 0u32;
            #[unroll(4)]
            while i < n {
                arr[i as usize] = i + 1;
                acc = acc.wrapping_add(i);
                i += 1;
            }
            *slot = acc;
        }
    }

    #[kernel]
    pub fn partial_after(arr: &mut [u32], mut out: DisjointSlice<u32>, n: u32) {
        if let Some(slot) = out.get_mut(thread::index_1d()) {
            let mut acc = 0u32;
            let mut i = 0u32;
            #[unroll(4)]
            while i < n {
                acc = acc.wrapping_add(i);
                i += 1;
            }
            arr[999] = 7;
            *slot = acc;
        }
    }

    #[kernel]
    pub fn full_division(mut out: DisjointSlice<u32>, divisor: u32) {
        if let Some(slot) = out.get_mut(thread::index_1d()) {
            // A different rustc MIR Assert, with no slice indexing to mask it.
            let mut acc = 84 / divisor;
            let mut i = 0u32;
            #[unroll]
            while i < 4 {
                acc = acc.wrapping_add(i);
                i += 1;
            }
            *slot = acc;
        }
    }
}

const CASES: &[&str] = &[
    "control",
    "full_before",
    "full_inside",
    "full_after",
    "partial_before",
    "partial_inside",
    "partial_after",
    "full_division",
];

fn run_case(name: &str, len: usize, expect_trap: bool) -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let module = kernels::load(&ctx)?;
    let stream = ctx.default_stream();
    let mut arr = DeviceBuffer::from_host(&stream, &vec![0u32; len])?;
    let mut out = DeviceBuffer::from_host(&stream, &[u32::MAX])?;
    stream.synchronize()?;

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: exactly one thread owns the mutable input slice and output.
    // Both allocations remain alive until synchronization completes.
    let launched = unsafe {
        match name {
            "control" => module.control(&stream, config, &mut arr, &mut out),
            "full_before" => module.full_before(&stream, config, &mut arr, &mut out),
            "full_inside" => module.full_inside(&stream, config, &mut arr, &mut out),
            "full_after" => module.full_after(&stream, config, &mut arr, &mut out),
            "partial_before" => module.partial_before(&stream, config, &mut arr, &mut out, 6),
            "partial_inside" => module.partial_inside(&stream, config, &mut arr, &mut out, 6),
            "partial_after" => module.partial_after(&stream, config, &mut arr, &mut out, 6),
            "full_division" => {
                module.full_division(&stream, config, &mut out, if expect_trap { 0 } else { 2 })
            }
            _ => return Err(format!("unknown case: {name}").into()),
        }
    };
    let completed = launched.and_then(|()| stream.synchronize());
    if expect_trap {
        match completed {
            Err(error) if error.0 == cuda_core::sys::cudaError_enum_CUDA_ERROR_LAUNCH_FAILED => {
                println!("{name} (len={len}): PASS (checked operation trapped)");
                return Ok(());
            }
            Err(error) => return Err(format!("{name}: expected trap, got {error}").into()),
            Ok(()) => return Err(format!("{name}: invalid operation did not trap").into()),
        }
    }
    completed?;

    let iterations = if name.starts_with("partial_") { 6 } else { 4 };
    let expected_sum = if name == "full_division" {
        48
    } else {
        iterations * (iterations - 1) / 2
    };
    assert_eq!(out.to_host_vec(&stream)?, [expected_sum], "{name}");
    if name != "full_division" {
        let mut expected = vec![0u32; len];
        if name.ends_with("_inside") {
            for i in 0..iterations {
                expected[i as usize] = i + 1;
            }
        } else {
            expected[999] = 7;
        }
        assert_eq!(arr.to_host_vec(&stream)?, expected, "{name}");
    }
    println!("{name}: PASS (valid values)");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 4 && args[1] == "--expect-trap" {
        return run_case(&args[2], args[3].parse()?, true);
    }
    if args.len() != 1 {
        return Err("usage: unroll_bounds_check [--expect-trap CASE LENGTH]".into());
    }

    for name in CASES {
        run_case(name, 1_000, false)?;
    }
    // A trap poisons its CUDA context. Each negative case gets a new process,
    // so a previous trap cannot make a later missing check appear to pass.
    let executable = std::env::current_exe()?;
    for name in CASES {
        let status = std::process::Command::new(&executable)
            .args(["--expect-trap", name, "1"])
            .status()?;
        if !status.success() {
            return Err(format!("{name}: trap subprocess failed ({status})").into());
        }
    }
    // n=6 has four unrolled iterations plus two remainder iterations. Length
    // one above fails in the unrolled group; length five fails in the tail.
    let status = std::process::Command::new(&executable)
        .args(["--expect-trap", "partial_inside", "5"])
        .status()?;
    if !status.success() {
        return Err(format!("partial_inside remainder: trap subprocess failed ({status})").into());
    }
    println!("SUCCESS: unrolling preserved checked operations");
    Ok(())
}
