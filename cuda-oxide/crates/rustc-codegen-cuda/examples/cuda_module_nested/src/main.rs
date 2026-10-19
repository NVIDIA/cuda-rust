/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Nested-module `#[cuda_module]` example.
//!
//! Kernels live at three levels of inline nesting:
//!
//! - `init::fill_index`, `scale::scale_by`, and `offset::offset_by` one level
//!   down,
//! - `post::double::double_all` two levels down.
//!
//! Each namespace owns a `LoadedModule` launcher view. Child views borrow the
//! same loaded CUDA module through `LoadedModule::from_parent`.
//!
//! Build and run with:
//!   cargo oxide run cuda_module_nested

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::cuda_module;

#[cuda_module]
mod kernels {
    /// Inline nested module: out[i] = i.
    ///
    /// The root intentionally has no direct kernel. This checks that calling
    /// `kernels::load()` still pins an artifact owned entirely by descendants.
    pub mod init {
        use cuda_device::{DisjointSlice, kernel, thread};

        #[kernel]
        pub fn fill_index(mut out: DisjointSlice<f32>) {
            let idx = thread::index_1d();
            let idx_raw = idx.get();
            if let Some(elem) = out.get_mut(idx) {
                *elem = idx_raw as f32;
            }
        }
    }

    /// Inline nested module: out[i] = a[i] * 2
    pub mod scale {
        use cuda_device::{DisjointSlice, kernel, thread};

        #[kernel]
        pub fn scale_by(a: &[f32], mut out: DisjointSlice<f32>) {
            let idx = thread::index_1d();
            let idx_raw = idx.get();
            if let Some(elem) = out.get_mut(idx) {
                *elem = a[idx_raw] * 2.0;
            }
        }
    }

    /// Inline nested module: out[i] = a[i] + 10
    pub mod offset {
        use cuda_device::{DisjointSlice, kernel, thread};

        #[kernel]
        pub fn offset_by(a: &[f32], mut out: DisjointSlice<f32>) {
            let idx = thread::index_1d();
            let idx_raw = idx.get();
            if let Some(elem) = out.get_mut(idx) {
                *elem = a[idx_raw] + 10.0;
            }
        }
    }

    /// An empty bridge namespace containing a doubly nested kernel module.
    pub mod post {
        pub mod double {
            use cuda_device::{DisjointSlice, kernel, thread};

            #[kernel]
            pub fn double_all(a: &[f32], mut out: DisjointSlice<f32>) {
                let idx = thread::index_1d();
                let idx_raw = idx.get();
                if let Some(elem) = out.get_mut(idx) {
                    *elem = a[idx_raw] + a[idx_raw];
                }
            }
        }
    }
}

fn main() {
    println!("=== #[cuda_module] Nested Modules Test ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let mut idx_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut scaled_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut offset_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut doubled_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let init =
        kernels::init::LoadedModule::from_parent(&module).expect("Failed to bind init launchers");
    let scale =
        kernels::scale::LoadedModule::from_parent(&module).expect("Failed to bind scale launchers");
    let offset = kernels::offset::LoadedModule::from_parent(&module)
        .expect("Failed to bind offset launchers");
    let post =
        kernels::post::LoadedModule::from_parent(&module).expect("Failed to bind post launchers");
    let double = kernels::post::double::LoadedModule::from_parent(&post)
        .expect("Failed to bind double launchers");
    let config = LaunchConfig::for_num_elems(N as u32);

    // SAFETY: every launch is one-dimensional over N elements, and each
    // input/output buffer covers the full range accessed by its kernel.
    unsafe {
        // The init kernel feeds the three processing kernels.
        init.fill_index(&stream, config, &mut idx_dev)
            .expect("fill_index launch failed");
        scale
            .scale_by(&stream, config, &idx_dev, &mut scaled_dev)
            .expect("scale_by launch failed");
        offset
            .offset_by(&stream, config, &idx_dev, &mut offset_dev)
            .expect("offset_by launch failed");
        double
            .double_all(&stream, config, &idx_dev, &mut doubled_dev)
            .expect("double_all launch failed");
    }

    let scaled = scaled_dev.to_host_vec(&stream).unwrap();
    let offset = offset_dev.to_host_vec(&stream).unwrap();
    let doubled = doubled_dev.to_host_vec(&stream).unwrap();

    let mut errors = 0;
    for i in 0..N {
        let expected_scaled = i as f32 * 2.0;
        let expected_offset = i as f32 + 10.0;
        let expected_doubled = i as f32 + i as f32;
        if (scaled[i] - expected_scaled).abs() > 1e-5
            || (offset[i] - expected_offset).abs() > 1e-5
            || (doubled[i] - expected_doubled).abs() > 1e-5
        {
            if errors < 5 {
                eprintln!(
                    "  Error at [{i}]: scaled {} (want {expected_scaled}), offset {} (want {expected_offset}), doubled {} (want {expected_doubled})",
                    scaled[i], offset[i], doubled[i],
                );
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!("✓ SUCCESS: root-loaded nested inline kernels all ran");
    } else {
        println!("✗ FAILED: {errors} errors");
        std::process::exit(1);
    }
}
