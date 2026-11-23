/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host verification and 1,000x CUDA Graph replay benchmark for the Rubin (`sm_107a`) and
//! Blackwell (`sm_100a`/`sm_103a`) persistent two-CTA `BFloat16` dense GEMM (`sm107_bf16_gemm_cute`).

#![feature(f16)]
#![deny(missing_docs, clippy::missing_docs_in_private_items)]

mod kernel;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D, sys};
use cute_rs::BFloat16;
use cute_rs::nvgpu::cpasync::sm100_host::create_tensor_map_tiled_typed;
use kernel::{ALayout, BLayout, DYNAMIC_SMEM_BYTES, THREADS, kernels};
use std::{error::Error, sync::Arc};

/// Host result alias carrying CUDA driver and verification errors.
type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Default token/row counts swept across decode-to-prefill problem sizes.
const DEFAULT_M_LIST: [usize; 7] = [128, 256, 512, 1024, 2048, 4096, 8192];

/// Convert a CUDA driver return status into a descriptive `Result`.
fn cuda_status(status: sys::CUresult, op: &str) -> Result<()> {
    if status == sys::cudaError_enum_CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{op}: {}", cuda_core::DriverError(status)).into())
    }
}

/// Convert host `f32` to `BFloat16` using IEEE round-to-nearest-even (`rne`).
fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb);
    (rounded >> 16) as u16
}

/// Widen `BFloat16` bits to exact `f32`.
fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Captured CUDA Graph (`CUgraphExec`) timed across 1,000 back-to-back replays.
struct KernelGraph {
    /// Instantiated CUDA Graph executable handle.
    exec: sys::CUgraphExec,
    /// Stream on which the graph is replayed.
    stream: Arc<CudaStream>,
}

impl KernelGraph {
    /// Capture `launch_fn` between `cuStreamBeginCapture` and `cuStreamEndCapture`.
    fn capture(stream: &Arc<CudaStream>, launch_fn: impl FnOnce() -> Result<()>) -> Result<Self> {
        stream.context().bind_to_thread()?;
        cuda_status(
            unsafe {
                sys::cuStreamBeginCapture_v2(
                    stream.cu_stream(),
                    sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
            },
            "begin graph capture",
        )?;
        let res = launch_fn();
        let mut graph = std::ptr::null_mut();
        let end = unsafe { sys::cuStreamEndCapture(stream.cu_stream(), &mut graph) };
        res?;
        cuda_status(end, "end graph capture")?;
        let mut exec = std::ptr::null_mut();
        let inst = unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        let _ = unsafe { sys::cuGraphDestroy(graph) };
        cuda_status(inst, "instantiate graph")?;
        Ok(Self {
            exec,
            stream: Arc::clone(stream),
        })
    }

    /// Time `replays` back-to-back graph launches in microseconds, returning the minimum sample.
    fn benchmark_us(&self, warmup: usize, replays: usize, samples: usize) -> Result<f64> {
        self.stream.context().bind_to_thread()?;
        for _ in 0..warmup {
            cuda_status(
                unsafe { sys::cuGraphLaunch(self.exec, self.stream.cu_stream()) },
                "warmup graph launch",
            )?;
        }
        self.stream.synchronize()?;
        let mut start = std::ptr::null_mut();
        let mut stop = std::ptr::null_mut();
        cuda_status(unsafe { sys::cuEventCreate(&mut start, 0) }, "create start")?;
        cuda_status(unsafe { sys::cuEventCreate(&mut stop, 0) }, "create stop")?;
        let mut best_us = f64::INFINITY;
        for _ in 0..samples {
            cuda_status(
                unsafe { sys::cuEventRecord(start, self.stream.cu_stream()) },
                "record start",
            )?;
            for _ in 0..replays {
                cuda_status(
                    unsafe { sys::cuGraphLaunch(self.exec, self.stream.cu_stream()) },
                    "replay graph",
                )?;
            }
            cuda_status(
                unsafe { sys::cuEventRecord(stop, self.stream.cu_stream()) },
                "record stop",
            )?;
            cuda_status(unsafe { sys::cuEventSynchronize(stop) }, "sync stop")?;
            let mut ms = 0.0f32;
            cuda_status(
                unsafe { sys::cuEventElapsedTime_v2(&mut ms, start, stop) },
                "elapsed time",
            )?;
            best_us = best_us.min((f64::from(ms) * 1000.0) / (replays as f64));
        }
        unsafe {
            let _ = sys::cuEventDestroy_v2(start);
            let _ = sys::cuEventDestroy_v2(stop);
        }
        Ok(best_us)
    }
}

impl Drop for KernelGraph {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            unsafe {
                let _ = sys::cuGraphExecDestroy(self.exec);
            }
        }
    }
}

/// Entry point running numerical oracle checks and 1,000x CUDA Graph benchmarks.
fn main() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let (cc_major, cc_minor) = ctx.compute_capability()?;
    if cc_major != 10 || !matches!(cc_minor, 0 | 3 | 7) {
        return Err(format!(
            "sm107_bf16_gemm_cute requires SM100/SM103/SM107; found sm_{cc_major}{cc_minor}"
        )
        .into());
    }
    let gpu_name = ctx.name()?;
    let mut sm_count_raw = 0i32;
    cuda_status(
        unsafe {
            sys::cuDeviceGetAttribute(
                &mut sm_count_raw,
                sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                ctx.cu_device(),
            )
        },
        "query SM count",
    )?;
    let sm_count = sm_count_raw as u32;
    let stream = ctx.new_stream()?;
    let module = unsafe { kernels::load(&ctx) }?;
    let probe =
        module.prepare_sm107_bf16_gemm(LaunchConfig1D::new(2, THREADS, DYNAMIC_SMEM_BYTES))?;
    let max_clusters = probe.function().max_active_clusters(
        (sm_count.max(2), 1, 1),
        (THREADS, 1, 1),
        DYNAMIC_SMEM_BYTES,
        (2, 1, 1),
    )?;
    drop(probe);

    let n = 4096usize;
    let k = 4096usize;
    let max_m = *DEFAULT_M_LIST.iter().max().unwrap();

    let a_host: Vec<BFloat16> = (0..max_m * k)
        .map(|i| {
            let val = (((i * 17 + 5) % 29) as f32 - 14.0) * 0.0625;
            BFloat16::from_bits(f32_to_bf16_bits(val))
        })
        .collect();
    let b_host: Vec<BFloat16> = (0..n * k)
        .map(|i| {
            let val = (((i * 13 + 7) % 31) as f32 - 15.0) * 0.0625;
            BFloat16::from_bits(f32_to_bf16_bits(val))
        })
        .collect();

    let d_a = DeviceBuffer::from_host(&stream, &a_host)?;
    let d_b = DeviceBuffer::from_host(&stream, &b_host)?;
    let d_c = DeviceBuffer::from_host(&stream, &vec![BFloat16::from_bits(0); max_m * n])?;

    let b_tma = create_tensor_map_tiled_typed::<BFloat16, BLayout>(
        d_b.cu_deviceptr() as *mut core::ffi::c_void,
        n as u64,
        k as u64,
        k as u64,
    )?;

    println!(
        "\nPersistent Two-CTA BFloat16 Dense GEMM + PDL on {} (sm_{}{}a, {} SMs, Max Active Clusters = {})",
        gpu_name, cc_major, cc_minor, sm_count, max_clusters
    );
    println!(
        "Shape: N = {n}, K = {k}, Tile = 128x256x64 (2-CTA), 8 TMA Stages, 2-Stage TMEM Ping-Pong"
    );
    println!("Timing: 1000x CUDA Graph Replay x 5 Samples (Warmup = 25)\n");
    println!(
        "{:>6} | {:>14} | {:>14} | {:>12} | {:>12}",
        "M", "Standard (us)", "PDL Variant", "TFLOP/s", "Norm Error"
    );
    println!("---------------------------------------------------------------------");

    for &m in &DEFAULT_M_LIST {
        let a_tma = create_tensor_map_tiled_typed::<BFloat16, ALayout>(
            d_a.cu_deviceptr() as *mut core::ffi::c_void,
            m as u64,
            k as u64,
            k as u64,
        )?;
        let total_tiles = ((m / 128) * (n / 256)) as u32;
        let clusters = total_tiles.min(max_clusters);
        let cfg = LaunchConfig1D::new(clusters * 2, THREADS, DYNAMIC_SMEM_BYTES);

        let prep_std = module.prepare_sm107_bf16_gemm(cfg)?;
        let prep_pdl = module.prepare_sm107_bf16_gemm_pdl(cfg)?;

        let mut worst_err = 0.0f32;
        for (label, use_pdl) in [("std", false), ("pdl", true)] {
            if use_pdl {
                unsafe {
                    module.sm107_bf16_gemm_pdl(
                        &stream,
                        &prep_pdl,
                        a_tma,
                        b_tma,
                        d_c.cu_deviceptr() as *mut BFloat16,
                        m as u64,
                        n as u64,
                        k as u64,
                    )?;
                }
            } else {
                unsafe {
                    module.sm107_bf16_gemm(
                        &stream,
                        &prep_std,
                        a_tma,
                        b_tma,
                        d_c.cu_deviceptr() as *mut BFloat16,
                        m as u64,
                        n as u64,
                        k as u64,
                    )?;
                }
            }
            stream.synchronize()?;
            let c_gpu = d_c.to_host_vec(&stream)?;
            let sample_rows = [0usize, m / 2, m - 1];
            let sample_cols = [0usize, 17, 127, 128, 255, n / 2, n - 1];
            let mut max_diff = 0.0f32;
            let mut max_ref = 1e-6f32;
            for &r in &sample_rows {
                for &c in &sample_cols {
                    let mut acc = 0.0f32;
                    for kk in 0..k {
                        acc += bf16_bits_to_f32(a_host[r * k + kk].to_bits())
                            * bf16_bits_to_f32(b_host[c * k + kk].to_bits());
                    }
                    let expected = bf16_bits_to_f32(f32_to_bf16_bits(acc));
                    let got = bf16_bits_to_f32(c_gpu[r * n + c].to_bits());
                    max_diff = max_diff.max((got - expected).abs());
                    max_ref = max_ref.max(expected.abs());
                }
            }
            let norm_err = max_diff / max_ref;
            if norm_err > 1e-3 {
                return Err(
                    format!("M={m} ({label}) verification failed: norm_err={norm_err}").into(),
                );
            }
            worst_err = worst_err.max(norm_err);
        }

        let graph_std = KernelGraph::capture(&stream, || unsafe {
            module
                .sm107_bf16_gemm(
                    &stream,
                    &prep_std,
                    a_tma,
                    b_tma,
                    d_c.cu_deviceptr() as *mut BFloat16,
                    m as u64,
                    n as u64,
                    k as u64,
                )
                .map_err(Into::into)
        })?;
        let graph_pdl = KernelGraph::capture(&stream, || unsafe {
            module
                .sm107_bf16_gemm_pdl(
                    &stream,
                    &prep_pdl,
                    a_tma,
                    b_tma,
                    d_c.cu_deviceptr() as *mut BFloat16,
                    m as u64,
                    n as u64,
                    k as u64,
                )
                .map_err(Into::into)
        })?;

        let std_us = graph_std.benchmark_us(25, 1000, 5)?;
        let pdl_us = graph_pdl.benchmark_us(25, 1000, 5)?;
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        let tflops = flops / (std_us * 1e6);

        println!(
            "{:>6} | {:>11.3} us | {:>11.3} us | {:>9.1} T | {:>12.6}",
            m, std_us, pdl_us, tflops, worst_err
        );
    }

    println!("\nSUCCESS: sm107_bf16_gemm_cute passed oracle verification and CUDA Graph replay.");
    Ok(())
}
