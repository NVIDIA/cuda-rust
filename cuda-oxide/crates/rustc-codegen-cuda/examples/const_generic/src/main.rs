/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{cuda_module, device, kernel, launch_bounds, thread};

struct Eight;

impl Eight {
    const VALUE: usize = 8;
}

#[allow(non_upper_case_globals)]
#[cuda_module]
mod kernels {
    use super::*;

    #[device]
    unsafe fn write_element<const VALUE: usize>(output: *mut u32, index: u32) {
        // SAFETY: the kernel's launch contract guarantees one valid element
        // for every launched thread.
        unsafe {
            output.add(index as usize).write(index + VALUE as u32);
        }
    }

    #[kernel]
    #[launch_bounds(64)]
    pub unsafe fn write_value<const VALUE: usize>(output: *mut u32) {
        let index = thread::index_1d().get() as u32;
        // SAFETY: inherited from this kernel's caller contract.
        unsafe { write_element::<VALUE>(output, index) }
    }

    /// The const is deliberately unused. The compiler must still retain two
    /// separately named entry specializations for callers that choose 4 or 8.
    #[kernel]
    pub unsafe fn write_same<const UNUSED: usize>(output: *mut u32) {
        let index = thread::index_1d().get();
        // SAFETY: inherited from this kernel's caller contract.
        unsafe { output.add(index).write(index as u32) }
    }

    #[device]
    fn hygiene_device<const __cuda_oxide_arg_0: usize>(value: usize) -> usize {
        value + __cuda_oxide_arg_0
    }

    /// This specialization is retained only by asking for its PTX name. It
    /// also proves that generated locals cannot capture a user const generic.
    #[allow(non_upper_case_globals)]
    #[kernel]
    pub fn name_only<const cuda_oxide_kernel_scope_246e25db: usize>() {
        let _ = thread::index_1d();
        let _ = hygiene_device::<cuda_oxide_kernel_scope_246e25db>(0);
    }
}

fn specialization_names() -> [&'static str; 5] {
    [
        kernels::write_value_ptx_name::<4>(),
        kernels::write_value_ptx_name::<8>(),
        kernels::write_same_ptx_name::<4>(),
        kernels::write_same_ptx_name::<8>(),
        kernels::name_only_ptx_name::<4>(),
    ]
}

fn entry_body<'source>(document: &ptx_parse::Document<'source>, name: &str) -> &'source str {
    document
        .callables_named(name)
        .find(|callable| callable.kind() == ptx_parse::CallableKind::Entry)
        .and_then(|callable| callable.body_text().map(|_| callable.text()))
        .unwrap_or_else(|| panic!("missing or incomplete PTX entry `{name}`"))
}

fn verify_generated_ptx() {
    let ptx = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/const_generic.ptx"))
        .expect("read const_generic.ptx; run `cargo oxide build const_generic` first");
    let document = ptx_parse::Document::parse(&ptx).expect("parse generated PTX");
    let [value_4, value_8, unused_4, unused_8, name_only_4] = specialization_names();

    assert_ne!(value_4, value_8);
    assert_ne!(unused_4, unused_8);
    let body_4 = entry_body(&document, value_4);
    let body_8 = entry_body(&document, value_8);
    let _unused_body_4 = entry_body(&document, unused_4);
    let _unused_body_8 = entry_body(&document, unused_8);
    let _name_only_body_4 = entry_body(&document, name_only_4);
    assert!(body_4.contains(".maxntid 64, 1, 1"));
    assert!(body_8.contains(".maxntid 64, 1, 1"));
    assert!(
        body_4.contains(", 4;"),
        "VALUE=4 was not folded into its PTX entry"
    );
    assert!(
        body_8.contains(", 8;"),
        "VALUE=8 was not folded into its PTX entry"
    );
}

fn main() {
    const ELEMENTS: usize = 64;

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--print-specializations") {
        for name in specialization_names() {
            println!("{name}");
        }
        return;
    }
    if args.iter().any(|arg| arg == "--verify-ptx") {
        verify_generated_ptx();
        println!(
            "const_generic: PASS (host lookup names, retained PTX entries, launch bounds, and constants agree)"
        );
        return;
    }

    verify_generated_ptx();

    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let module = kernels::load(&context).expect("load const-generic module");
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (ELEMENTS as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let output_4 = DeviceBuffer::<u32>::zeroed(&stream, ELEMENTS).expect("allocate output");
    let output_8 = DeviceBuffer::<u32>::zeroed(&stream, ELEMENTS).expect("allocate output");
    let output_same_4 = DeviceBuffer::<u32>::zeroed(&stream, ELEMENTS).expect("allocate output");
    let output_same_8 = DeviceBuffer::<u32>::zeroed(&stream, ELEMENTS).expect("allocate output");

    // SAFETY: each launch has exactly ELEMENTS threads, and each pointer owns
    // ELEMENTS writable u32 values that remain live through synchronization.
    unsafe {
        module
            .write_value::<4>(&stream, config, output_4.cu_deviceptr() as *mut u32)
            .expect("launch write_value::<4>");
        module
            .write_value::<{ Eight::VALUE }>(&stream, config, output_8.cu_deviceptr() as *mut u32)
            .expect("launch write_value::<8>");
        module
            .write_same::<4>(&stream, config, output_same_4.cu_deviceptr() as *mut u32)
            .expect("launch write_same::<4>");
        module
            .write_same::<8>(&stream, config, output_same_8.cu_deviceptr() as *mut u32)
            .expect("launch write_same::<8>");
    }

    let values_4 = output_4.to_host_vec(&stream).expect("copy VALUE=4 output");
    let values_8 = output_8.to_host_vec(&stream).expect("copy VALUE=8 output");
    let same_4 = output_same_4
        .to_host_vec(&stream)
        .expect("copy unused VALUE=4 output");
    let same_8 = output_same_8
        .to_host_vec(&stream)
        .expect("copy unused VALUE=8 output");

    for index in 0..ELEMENTS {
        assert_eq!(values_4[index], index as u32 + 4);
        assert_eq!(values_8[index], index as u32 + 8);
        assert_eq!(same_4[index], index as u32);
        assert_eq!(same_8[index], index as u32);
    }

    println!("const_generic: PASS (distinct VALUE=4 and VALUE=8 kernel results)");
}

#[cfg(test)]
mod tests {
    #[test]
    fn host_lookup_names_distinguish_const_values() {
        let [value_4, value_8, unused_4, unused_8, name_only_4] = super::specialization_names();

        assert_ne!(value_4, value_8);
        assert_ne!(unused_4, unused_8);
        assert!(name_only_4.starts_with("name_only_TID_"));
        assert!(value_4.starts_with("write_value_TID_"));
        assert!(value_8.starts_with("write_value_TID_"));
        println!("VALUE=4 host lookup: {value_4}");
        println!("VALUE=8 host lookup: {value_8}");
    }
}
