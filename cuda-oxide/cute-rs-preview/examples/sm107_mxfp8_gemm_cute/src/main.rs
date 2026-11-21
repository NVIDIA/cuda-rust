/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host input encoding (`Sm1xxBlockScaleKMajor<32>`), CPU `MXFP8` (`E4M3 * UE8M0 -> FP32 -> BF16`)
//! oracle verification, and 1,000x CUDA Graph replay benchmark on Vera Rubin (`sm_107a`), comparing
//! `128x128` (`CollectorDiscard`) vs `256x128` B-Collector Reuse (`CollectorFill` + `CollectorLastUse`) + PDL.

#![feature(f16)]
#![deny(missing_docs, clippy::missing_docs_in_private_items)]

mod kernel;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D, sys};
use cute_rs::nvgpu::cpasync::{TmaEncodeOptions, create_tensor_map_tiled_with_options};
use cute_rs::{BFloat16, Float8E4M3FN, Sm1xxBlockScaleKMajor};
use kernel::{
    ATileSmem, BTileSmem, DYNAMIC_SMEM_128, DYNAMIC_SMEM_256, ScaleAtomSmem, THREADS, kernels,
};
use std::{error::Error, sync::Arc};

/// Host result alias carrying CUDA driver and verification errors.
type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Default token/row counts swept on Rubin (`sm_107a`), all multiples of `256`.
const DEFAULT_M_LIST: [usize; 6] = [256, 512, 1024, 2048, 4096, 8192];

/// Convert a CUDA driver return status into a descriptive `Result`.
fn cuda_status(status: sys::CUresult, op: &str) -> Result<()> {
    if status == sys::cudaError_enum_CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{op}: {}", cuda_core::DriverError(status)).into())
    }
}

/// Exact table-free decode of one `Float8E4M3FN` byte to `f32`.
fn decode_e4m3_to_f32(bits: u8) -> f32 {
    let sign = if (bits & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((bits >> 3) & 0x0F) as i32;
    let mant = (bits & 0x07) as u32;
    if exp == 15 && mant == 7 {
        return f32::NAN;
    }
    if exp == 0 {
        sign * (mant as f32) * 2.0f32.powi(-9)
    } else {
        sign * (1.0 + (mant as f32) * 0.125) * 2.0f32.powi(exp - 7)
    }
}

/// Exact decode of one `Float8E8M0FNU` (`UE8M0`) power-of-two scale byte to `f32`.
fn decode_ue8m0_to_f32(bits: u8) -> f32 {
    if bits == 0xFF {
        f32::NAN
    } else {
        f32::from_bits(u32::from(bits) << 23)
    }
}

/// Round `f32` to IEEE `BFloat16` (`rne`).
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

/// Entry point running CPU `MXFP8` oracle verification and 1,000x CUDA Graph benchmarks on Rubin (`sm_107a`).
fn main() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let (cc_major, cc_minor) = ctx.compute_capability()?;
    if (cc_major, cc_minor) != (10, 7) {
        return Err(format!(
            "sm107_mxfp8_gemm_cute requires Vera Rubin SM107 (sm_107a); found sm_{cc_major}{cc_minor}"
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

    let n = 4096usize;
    let k = 4096usize;
    let max_m = *DEFAULT_M_LIST.iter().max().unwrap();

    // Populate deterministic finite E4M3 codes (exponents 5..9, mantissas 0..7) and UE8M0 scales (125..128).
    let a_host: Vec<Float8E4M3FN> = (0..max_m * k)
        .map(|i| {
            let sign = ((i * 7 + 3) & 1) as u8;
            let exp = (5 + ((i * 11 + 1) % 4)) as u8;
            let mant = ((i * 13 + 5) % 7) as u8;
            Float8E4M3FN::from_bits((sign << 7) | (exp << 3) | mant)
        })
        .collect();
    let b_host: Vec<Float8E4M3FN> = (0..n * k)
        .map(|i| {
            let sign = ((i * 5 + 1) & 1) as u8;
            let exp = (5 + ((i * 17 + 3) % 4)) as u8;
            let mant = ((i * 19 + 2) % 7) as u8;
            Float8E4M3FN::from_bits((sign << 7) | (exp << 3) | mant)
        })
        .collect();

    let sfa_layout = Sm1xxBlockScaleKMajor::<32>::new(max_m, k);
    let sfb_layout = Sm1xxBlockScaleKMajor::<32>::new(n, k);
    let k_groups = k / 32;
    let mut sfa_bytes = vec![127u8; sfa_layout.storage_len(1)];
    let mut sfb_bytes = vec![127u8; sfb_layout.storage_len(1)];
    for r in 0..max_m {
        for g in 0..k_groups {
            let exp_bias = (125 + ((r * 3 + g * 5) % 4)) as u8;
            sfa_bytes[sfa_layout.crd2idx(0, r, g)] = exp_bias;
        }
    }
    for c in 0..n {
        for g in 0..k_groups {
            let exp_bias = (125 + ((c * 7 + g * 3) % 4)) as u8;
            sfb_bytes[sfb_layout.crd2idx(0, c, g)] = exp_bias;
        }
    }

    let sfa_u16: Vec<u16> = sfa_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .copied()
        .map(u16::from_le_bytes)
        .collect();
    let sfb_u16: Vec<u16> = sfb_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .copied()
        .map(u16::from_le_bytes)
        .collect();

    let d_a = DeviceBuffer::from_host(&stream, &a_host)?;
    let d_b = DeviceBuffer::from_host(&stream, &b_host)?;
    let d_sfa = DeviceBuffer::from_host(&stream, &sfa_u16)?;
    let d_sfb = DeviceBuffer::from_host(&stream, &sfb_u16)?;
    let d_c = DeviceBuffer::from_host(&stream, &vec![BFloat16::from_bits(0); max_m * n])?;

    let k_atoms = k / 128;
    let b_tma = create_tensor_map_tiled_with_options::<Float8E4M3FN, BTileSmem>(
        d_b.cu_deviceptr() as *mut core::ffi::c_void,
        n as u64,
        k as u64,
        k as u64,
        TmaEncodeOptions::L2_128B,
    )?;
    let sfb_tma = create_tensor_map_tiled_with_options::<u16, ScaleAtomSmem>(
        d_sfb.cu_deviceptr() as *mut core::ffi::c_void,
        ((n / 128) * k_atoms * 4) as u64,
        64,
        64,
        TmaEncodeOptions::L2_128B,
    )?;

    println!(
        "\nRubin (sm_107a) Persistent MXFP8 (E4M3 + UE8M0 SF=32, K=64) GEMM on {} ({} SMs)",
        gpu_name, sm_count
    );
    println!(
        "Shape: N = {n}, K = {k} | Comparing 128x128 (Discard) vs 256x128 (B-Collector Reuse + PDL)"
    );
    println!("Timing: 1000x CUDA Graph Replay x 5 Samples (Warmup = 25)\n");
    println!(
        "{:>6} | {:>15} | {:>17} | {:>11} | {:>9} | {:>10}",
        "M", "128x128 (us)", "256x128 B-Reuse", "B-Reuse T/s", "Speedup", "Norm Error"
    );
    println!("---------------------------------------------------------------------------------");

    for &m in &DEFAULT_M_LIST {
        let a_tma = create_tensor_map_tiled_with_options::<Float8E4M3FN, ATileSmem>(
            d_a.cu_deviceptr() as *mut core::ffi::c_void,
            m as u64,
            k as u64,
            k as u64,
            TmaEncodeOptions::L2_128B,
        )?;
        let sfa_tma = create_tensor_map_tiled_with_options::<u16, ScaleAtomSmem>(
            d_sfa.cu_deviceptr() as *mut core::ffi::c_void,
            ((m / 128) * k_atoms * 4) as u64,
            64,
            64,
            TmaEncodeOptions::L2_128B,
        )?;

        let tiles_128 = ((m / 128) * (n / 128)) as u32;
        let tiles_256 = ((m / 256) * (n / 128)) as u32;
        let cfg_128 = LaunchConfig1D::new(tiles_128.min(sm_count), THREADS, DYNAMIC_SMEM_128);
        let cfg_256 = LaunchConfig1D::new(tiles_256.min(sm_count), THREADS, DYNAMIC_SMEM_256);

        let prep_128 = module.prepare_sm107_mxfp8_gemm(cfg_128)?;
        let prep_256 = module.prepare_sm107_mxfp8_gemm_breuse_pdl(cfg_256)?;

        let mut worst_err = 0.0f32;
        for (label, use_breuse) in [("128x128", false), ("256x128_breuse_pdl", true)] {
            if use_breuse {
                unsafe {
                    module.sm107_mxfp8_gemm_breuse_pdl(
                        &stream,
                        &prep_256,
                        a_tma,
                        b_tma,
                        sfa_tma,
                        sfb_tma,
                        d_c.cu_deviceptr() as *mut BFloat16,
                        m as u64,
                        n as u64,
                        k as u64,
                    )?;
                }
            } else {
                unsafe {
                    module.sm107_mxfp8_gemm(
                        &stream,
                        &prep_128,
                        a_tma,
                        b_tma,
                        sfa_tma,
                        sfb_tma,
                        d_c.cu_deviceptr() as *mut BFloat16,
                        m as u64,
                        n as u64,
                        k as u64,
                    )?;
                }
            }
            stream.synchronize()?;
            let c_gpu = d_c.to_host_vec(&stream)?;
            let sample_rows = [0usize, 127, 128, m / 2, m - 1];
            let sample_cols = [0usize, 31, 63, 127, 128, n / 2, n - 1];
            let mut max_diff = 0.0f32;
            let mut max_ref = 1e-6f32;
            for &r in &sample_rows {
                for &c in &sample_cols {
                    let mut acc = 0.0f32;
                    for g in 0..k_groups {
                        let sa = decode_ue8m0_to_f32(sfa_bytes[sfa_layout.crd2idx(0, r, g)]);
                        let sb = decode_ue8m0_to_f32(sfb_bytes[sfb_layout.crd2idx(0, c, g)]);
                        let mut block_dot = 0.0f32;
                        let base_k = g * 32;
                        for j in 0..32 {
                            let av = decode_e4m3_to_f32(a_host[r * k + base_k + j].to_bits());
                            let bv = decode_e4m3_to_f32(b_host[c * k + base_k + j].to_bits());
                            block_dot += av * bv;
                        }
                        acc += block_dot * (sa * sb);
                    }
                    let expected = bf16_bits_to_f32(f32_to_bf16_bits(acc));
                    let got = bf16_bits_to_f32(c_gpu[r * n + c].to_bits());
                    max_diff = max_diff.max((got - expected).abs());
                    max_ref = max_ref.max(expected.abs());
                }
            }
            let norm_err = max_diff / max_ref;
            if norm_err > 1e-3 {
                return Err(format!(
                    "M={m} ({label}) MXFP8 verification failed: norm_err={norm_err}"
                )
                .into());
            }
            worst_err = worst_err.max(norm_err);
        }

        let graph_128 = KernelGraph::capture(&stream, || unsafe {
            module
                .sm107_mxfp8_gemm(
                    &stream,
                    &prep_128,
                    a_tma,
                    b_tma,
                    sfa_tma,
                    sfb_tma,
                    d_c.cu_deviceptr() as *mut BFloat16,
                    m as u64,
                    n as u64,
                    k as u64,
                )
                .map_err(Into::into)
        })?;
        let graph_256 = KernelGraph::capture(&stream, || unsafe {
            module
                .sm107_mxfp8_gemm_breuse_pdl(
                    &stream,
                    &prep_256,
                    a_tma,
                    b_tma,
                    sfa_tma,
                    sfb_tma,
                    d_c.cu_deviceptr() as *mut BFloat16,
                    m as u64,
                    n as u64,
                    k as u64,
                )
                .map_err(Into::into)
        })?;

        let us_128 = graph_128.benchmark_us(25, 1000, 5)?;
        let us_256 = graph_256.benchmark_us(25, 1000, 5)?;
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        let tflops_256 = flops / (us_256 * 1e6);
        let speedup = us_128 / us_256;

        println!(
            "{:>6} | {:>12.3} us | {:>14.3} us | {:>9.1} T | {:>8.2}x | {:>10.6}",
            m, us_128, us_256, tflops_256, speedup, worst_err
        );
    }

    println!("\nSUCCESS: sm107_mxfp8_gemm_cute passed exact MXFP8 oracle verification on Rubin.");
    Ok(())
}
