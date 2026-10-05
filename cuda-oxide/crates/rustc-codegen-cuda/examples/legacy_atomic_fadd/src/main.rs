// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{
    DisjointSlice,
    atomic::{AtomicOrdering, DeviceAtomicF32, DeviceAtomicF64},
    kernel, thread,
};
use cuda_host::cuda_module;

const N: usize = 256;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn atomic_add(
        counter_f32: &[DeviceAtomicF32],
        counter_f64: &[DeviceAtomicF64],
        mut old_values: DisjointSlice<(f32, f64)>,
    ) {
        let index = thread::index_1d();
        if index.get() >= N {
            return;
        }

        let old_f32 = counter_f32[0].fetch_add(1.0, AtomicOrdering::Relaxed);
        let old_f64 = counter_f64[0].fetch_add(1.0, AtomicOrdering::Relaxed);
        if let Some(slot) = old_values.get_mut(index) {
            *slot = (old_f32, old_f64);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = kernels::load(&context)?;
    let counter_f32 = DeviceBuffer::<f32>::zeroed(&stream, 1)?.cast_elem::<DeviceAtomicF32>();
    let counter_f64 = DeviceBuffer::<f64>::zeroed(&stream, 1)?.cast_elem::<DeviceAtomicF64>();
    let mut old_values = DeviceBuffer::<(f32, f64)>::zeroed(&stream, N)?;

    // SAFETY: the launch covers exactly N unique 1D indices, `old_values` has
    // N elements, and the one-element counters use atomic wrapper pointees so
    // every shared update is atomic rather than an aliased `&mut` access.
    unsafe {
        module.atomic_add(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &counter_f32,
            &counter_f64,
            &mut old_values,
        )?;
    }
    stream.synchronize()?;

    let got_f32 = counter_f32.cast_elem::<f32>().to_host_vec(&stream)?[0];
    let got_f64 = counter_f64.cast_elem::<f64>().to_host_vec(&stream)?[0];
    let old_values = old_values.to_host_vec(&stream)?;
    let mut old_f32 = old_values
        .iter()
        .map(|&(value, _)| value)
        .collect::<Vec<_>>();
    let mut old_f64 = old_values
        .iter()
        .map(|&(_, value)| value)
        .collect::<Vec<_>>();
    old_f32.sort_by(f32::total_cmp);
    old_f64.sort_by(f64::total_cmp);
    let old_f32_is_permutation = old_f32
        .iter()
        .enumerate()
        .all(|(index, &value)| value == index as f32);
    let old_f64_is_permutation = old_f64
        .iter()
        .enumerate()
        .all(|(index, &value)| value == index as f64);
    if got_f32 != N as f32
        || got_f64 != N as f64
        || !old_f32_is_permutation
        || !old_f64_is_permutation
    {
        return Err(format!(
            "legacy atomic add mismatch: f32={got_f32}, f64={got_f64}, old_f32_permutation={old_f32_is_permutation}, old_f64_permutation={old_f64_is_permutation}"
        )
        .into());
    }

    println!("legacy_atomic_fadd: PASS (f32={got_f32}, f64={got_f64})");
    Ok(())
}
