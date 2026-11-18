/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host-side launch and allocation-lease scaling repro for issue #1786.
//!
//! Run with `cargo run --release --manifest-path cuda-core/repro/lease_scaling/Cargo.toml -- one 256000`.
//! Modes are `one`, `sync`, `fresh`, and `core`; see the issue description.

use cuda_async::device_operation::ExecutionContext;
use cuda_core::{CudaContext, DeviceBuffer, launch_kernel_on_stream};
use std::time::Instant;

const PTX: &str =
    ".version 8.0\n.target sm_52\n.address_size 64\n.visible .entry noop()\n{\n\tret;\n}\n";
const WINDOW: usize = 1_000;

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "one".into());
    let launches: usize = args
        .next()
        .map_or(100_000, |arg| arg.parse().expect("launch count"));
    assert!(
        matches!(mode.as_str(), "one" | "sync" | "fresh" | "core"),
        "mode must be one, sync, fresh, or core"
    );

    let context = CudaContext::new(0).expect("context");
    let stream = context.new_stream().expect("stream");
    let buffer: DeviceBuffer<f32> = DeviceBuffer::zeroed(&stream, 1 << 16).expect("buffer");
    let module = context.load_module_from_ptx_src(PTX).expect("module");
    let noop = module.load_function("noop").expect("function");

    let mut execution = ExecutionContext::new(stream.clone());
    let mut held = Vec::new();
    let mut start = Instant::now();
    let mut checkpoint = WINDOW;
    for i in 1..=launches {
        if i == checkpoint - WINDOW + 1 {
            start = Instant::now();
        }
        if mode == "fresh" {
            execution = ExecutionContext::new(stream.clone());
        }
        if mode == "core" {
            held.push(
                buffer
                    .storage()
                    .acquire(&stream, true, false)
                    .expect("acquire"),
            );
        } else {
            execution
                .retain_buffer(buffer.storage().clone(), true)
                .expect("retain");
        }

        // SAFETY: `noop` takes no parameters and touches no memory.
        unsafe { launch_kernel_on_stream(&noop, (1, 1, 1), (1, 1, 1), 0, &stream, &mut []) }
            .expect("launch");

        if mode == "sync" && i % 100 == 0 {
            stream.synchronize().expect("sync");
        }
        if i == checkpoint {
            let us = start.elapsed().as_secs_f64() * 1e6 / WINDOW as f64;
            println!(
                "{mode} {i:>7}: {us:7.2} us per launch over launches {}..={i}",
                i - WINDOW + 1
            );
            checkpoint = checkpoint.saturating_mul(4);
        }
    }

    stream.synchronize().expect("sync");
    let count = held.len();
    let before_drop = Instant::now();
    drop(held);
    if count > 0 {
        println!(
            "{mode}: dropping {count} leases took {:.1} ms",
            before_drop.elapsed().as_secs_f64() * 1e3
        );
    }
}
