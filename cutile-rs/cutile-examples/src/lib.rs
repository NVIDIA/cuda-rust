/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared utilities and reference implementations for cutile examples.

pub mod requirements;

#[cfg(feature = "reference-cpu")]
use candle_core::WithDType;
#[cfg(feature = "reference-cpu")]
use candle_nn::ops::softmax;
#[cfg(feature = "reference-cpu")]
use cuda_core::DType;

use cuda_core::{get_device_clock_rate, Device};

/// Convert a host slice to a candle CPU tensor.
#[cfg(feature = "reference-cpu")]
pub fn to_candle_tensor<T: DType + WithDType>(data: &[T], shape: &[usize]) -> candle_core::Tensor {
    candle_core::Tensor::from_slice(data, shape, &candle_core::Device::Cpu).unwrap()
}

/// Prints a 2D candle tensor in a formatted table for debugging.
#[cfg(feature = "reference-cpu")]
pub fn pretty_print_matrix<T: WithDType>(mat: &candle_core::Tensor) {
    let iter_dim = 0;
    let range = 0..mat.shape().dims()[iter_dim];
    println!(
        " {}",
        range
            .into_iter()
            .map(|x| format!("{:^9}", x))
            .collect::<Vec<String>>()
            .join("")
    );
    for j in 0..mat.shape().dims()[iter_dim] {
        let out_vec = mat.get_on_dim(iter_dim, j).unwrap().to_vec1::<T>().unwrap();
        let out_vec = out_vec
            .iter()
            .map(|x| format!("{:^8.1}|", x))
            .collect::<Vec<String>>()
            .join("");
        println!("|{}", out_vec);
    }
}

/// Formats a byte count into a human-readable size string (e.g. "1.5kb", "2.3mb").
pub fn size_label(size_bytes: usize) -> String {
    if size_bytes < 10usize.pow(3) {
        // bytes
        format!("{}b", size_bytes)
    } else if size_bytes < 10usize.pow(6) {
        // kb
        format!("{:.1}kb", size_bytes as f64 / 10usize.pow(3) as f64)
    } else if size_bytes < 10usize.pow(9) {
        // mb
        format!("{:.1}mb", size_bytes as f64 / 10usize.pow(6) as f64)
    } else {
        // gb
        format!("{:.1}gb", size_bytes as f64 / 10usize.pow(9) as f64)
    }
}

/// Computes a reference FMHA result on the host: `softmax(scale(Q @ K^T)) @ V`.
///
/// Takes host-side data as slices and shapes, constructs candle tensors internally.
#[cfg(feature = "reference-cpu")]
pub fn fmha_ref_exec<T: WithDType>(
    q_data: &[T],
    q_shape: &[usize],
    k_data: &[T],
    k_shape: &[usize],
    v_data: &[T],
    v_shape: &[usize],
    sm_scale: T,
) -> candle_core::Tensor {
    let q_host =
        candle_core::Tensor::from_slice(q_data, q_shape, &candle_core::Device::Cpu).unwrap();
    let k_host =
        candle_core::Tensor::from_slice(k_data, k_shape, &candle_core::Device::Cpu).unwrap();
    let v_host =
        candle_core::Tensor::from_slice(v_data, v_shape, &candle_core::Device::Cpu).unwrap();
    let k_trans = k_host.transpose(2, 3).expect("Failed to transpose k.");
    let qk = q_host
        .broadcast_matmul(&k_trans)
        .expect("Failed to execute q @ k^T."); // (m x d) @ (d x m)

    let sm_scale_tensor = candle_core::Tensor::full(sm_scale, qk.shape(), qk.device()).unwrap();
    let qk_scaled = qk.mul(&sm_scale_tensor).expect("Failed to scale qk.");
    let qk_softmax = softmax(&qk_scaled, 3).expect("Failed to softmax qk.");

    // (m x m) @ (m x d)
    qk_softmax
        .broadcast_matmul(&v_host)
        .expect("Failed to execute qk @ v.") // (b, h, m, d)
}

/// Computes the theoretical peak (speed-of-light) tensor core TFLOPS for a Blackwell GPU.
pub fn blackwell_tensorcore_sol_tflops(
    num_sms: f64,
    tensorcores_per_sm: f64,
    op_issue_freq: f64,
    flops_per_tensor_op: f64,
    clock_rate_mhz: f64,
) -> f64 {
    let clock_rate: f64 = clock_rate_mhz * 1e6;
    // How many ops can execute in parallel?
    let ops_per_device: f64 = num_sms * tensorcores_per_sm;
    // How many ops can be issued per second?
    let hmma_per_sec: f64 = clock_rate * op_issue_freq * ops_per_device;
    let sol_flops_per_sec: f64 = hmma_per_sec * flops_per_tensor_op;
    let sol_tflops_per_sec: f64 = sol_flops_per_sec * 1e-12;
    sol_tflops_per_sec
}

/// Returns RTX 5090 theoretical peak f16 tensor core TFLOPS at the given clock speed (MHz).
pub fn rtx_5090_tensorcore_f16_sol_tflops_at(clock_speed: f64) -> f64 {
    let num_sms = 170.0;
    let num_tensor_cores = 4.0;
    // Issue freq and max flops should be the fastest available mma op for f16.
    let hmma16816_f16_issue_freq = 1.0 / 16.0;
    // Number of flops per tensor core.
    let hmma_16816_f16_flops = 2.0 * 16.0 * 8.0 * 16.0;
    blackwell_tensorcore_sol_tflops(
        num_sms,
        num_tensor_cores,
        hmma16816_f16_issue_freq,
        hmma_16816_f16_flops,
        clock_speed,
    )
}

/// Returns RTX 5090 theoretical peak f16 tensor core TFLOPS using the device's clock rate.
pub fn rtx_5090_tensorcore_f16_sol_tflops(device_id: usize) -> f64 {
    let device = Device::new(device_id).unwrap();
    let clock_rate = unsafe { get_device_clock_rate(device.cu_device()).unwrap() } as f64;
    rtx_5090_tensorcore_f16_sol_tflops_at(clock_rate)
}

#[cfg(test)]
mod test_peak {
    use crate::rtx_5090_tensorcore_f16_sol_tflops_at;
    use cuda_core::{get_device_clock_rate, Device};

    #[test]
    fn test_5090() {
        let device = Device::new(0).unwrap();
        // This appears to be correct / matches architecture documentation.
        let clock_rate = unsafe { get_device_clock_rate(device.cu_device()).unwrap() } as f64;
        let clock_mhz = clock_rate * 1e-3;
        println!(
            "rtx_5090 @ {}Mhz max tflops = {}",
            clock_mhz,
            rtx_5090_tensorcore_f16_sol_tflops_at(clock_mhz)
        );
    }
}
