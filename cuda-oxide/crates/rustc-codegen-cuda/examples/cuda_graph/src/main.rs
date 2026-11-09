/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Capture SIMT kernel launches into a CUDA graph and replay it.
//!
//! One "step" is 16 small launches: eight layers of `axpy` followed by
//! `scale`, the shape of a decode step built from many short kernels. The
//! step runs two ways on the same stream:
//!
//! - eagerly, enqueuing each launch through the ordinary module API, and
//! - as a `CudaGraph` recorded once with `CudaGraph::scope`, then replayed.
//!
//! Both paths run the same kernels in the same order, so their results must
//! match bit for bit, and both are checked against a CPU reference. The time
//! per step is printed for each path; it depends on the GPU and driver.
//!
//! Run: cargo oxide run cuda_graph

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_context::{init_device_contexts, with_cuda_context};
use cuda_async::device_operation::DeviceOp;
use cuda_core::{DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::time::Instant;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn axpy(a: f32, x: &[f32], mut y: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(yi) = y.get_mut(idx) {
            *yi += a * x[i];
        }
    }

    #[kernel]
    pub fn scale(s: f32, mut y: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if let Some(yi) = y.get_mut(idx) {
            *yi *= s;
        }
    }
}

const N: usize = 4096;
const LAYERS: usize = 8;
const STEPS: usize = 200;

fn coefficients(layer: usize) -> (f32, f32) {
    (0.5 / (layer as f32 + 1.0), 0.75 + 0.02 * layer as f32)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CUDA graph replay of SIMT kernels ===\n");

    init_device_contexts(0, 1)?;
    let module = kernels::load_async(0)?;
    // Capture needs a created stream; the legacy default stream cannot capture.
    let stream = with_cuda_context(0, |ctx| ctx.new_stream())??;
    let cfg = LaunchConfig::for_num_elems(N as u32);

    let x_host: Vec<f32> = (0..N).map(|i| ((i % 97) as f32) * 0.01).collect();
    let y_init: Vec<f32> = (0..N).map(|i| 1.0 + ((i % 13) as f32) * 0.1).collect();
    let x = DeviceBuffer::from_host(&stream, &x_host)?;
    let mut y_eager = DeviceBuffer::from_host(&stream, &y_init)?;
    let mut y_graph = DeviceBuffer::from_host(&stream, &y_init)?;

    // ── Eager: 16 launches per step, one synchronize per step ──────────────
    let start = Instant::now();
    for _ in 0..STEPS {
        for layer in 0..LAYERS {
            let (a, s) = coefficients(layer);
            // SAFETY: the 1D config covers N threads and every buffer holds N
            // elements, which is the indexing both kernels assume.
            unsafe {
                module.axpy(stream.as_ref(), cfg, a, &x, &mut y_eager)?;
                module.scale(stream.as_ref(), cfg, s, &mut y_eager)?;
            }
        }
        stream.synchronize()?;
    }
    let eager = start.elapsed();

    // ── Graph: record the same 16 launches once, then replay ───────────────
    // Buffers are allocated before capture: a graph replays into the same
    // device pointers, and allocation cannot be recorded.
    let graph = CudaGraph::scope(&stream, |s| {
        for layer in 0..LAYERS {
            let (a, sc) = coefficients(layer);
            // SAFETY: as for the eager launches above.
            s.record(unsafe { module.axpy_async(cfg, a, &x, &mut y_graph) }?)?;
            s.record(unsafe { module.scale_async(cfg, sc, &mut y_graph) }?)?;
        }
        Ok(())
    })?;
    let start = Instant::now();
    for _ in 0..STEPS {
        graph.launch().sync_on(&stream)?;
    }
    let replay = start.elapsed();

    // ── Check ───────────────────────────────────────────────────────────────
    let eager_out = y_eager.to_host_vec(&stream)?;
    let graph_out = y_graph.to_host_vec(&stream)?;
    let mismatched = eager_out
        .iter()
        .zip(&graph_out)
        .filter(|(e, g)| e.to_bits() != g.to_bits())
        .count();

    let mut reference = y_init.clone();
    for _ in 0..STEPS {
        for layer in 0..LAYERS {
            let (a, s) = coefficients(layer);
            for (y, &xi) in reference.iter_mut().zip(&x_host) {
                *y = (a * xi + *y) * s;
            }
        }
    }
    let worst = reference
        .iter()
        .zip(&graph_out)
        .map(|(r, g)| ((r - g) / r.abs().max(1e-6)).abs())
        .fold(0.0f32, f32::max);

    let per_step = |total: std::time::Duration| total.as_secs_f64() * 1e6 / STEPS as f64;
    println!("{STEPS} steps of {} launches over {N} elements", 2 * LAYERS);
    println!("  eager launches: {:8.1} us/step", per_step(eager));
    println!("  graph replay:   {:8.1} us/step", per_step(replay));
    println!("  eager vs graph: {mismatched} elements differ");
    println!("  graph vs CPU:   max relative error {worst:.2e}");

    if mismatched == 0 && worst < 1e-4 {
        println!("\nPASSED");
        Ok(())
    } else {
        eprintln!("\nFAILED");
        std::process::exit(1);
    }
}
