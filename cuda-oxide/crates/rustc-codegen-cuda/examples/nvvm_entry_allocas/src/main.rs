/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression for entry-block placement of fixed scratch allocations across
//! both `branch_scratch` (#1681) and `core_zip` (#1687).
//!
//! `mir-lower` hoists eligible fixed-size `llvm.alloca` slots into each
//! function's `entry` block while keeping initialization at the use site.
//! The companion `verify-code-shape.sh` checks that every `alloca` in
//! `nvvm_entry_allocas.ll` sits in `entry:`, that extended split compilation
//! emits no dynamic-stack instructions while retaining `branch_scratch`'s
//! aggregate `consume` call and 128-bit parameter stores, and that `core_zip`
//! inlines `<ZipImpl as Iterator>::next` with zero `call` instructions and
//! zero `.local` stack frame.
//!
//! Run from `cuda-oxide/`:
//! ```text
//! cargo oxide run nvvm_entry_allocas --emit-nvvm-ir --arch sm_120
//! bash crates/rustc-codegen-cuda/examples/nvvm_entry_allocas/verify-code-shape.sh
//! ```

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig, LaunchConfig1D};
use cuda_device::view::MatrixView32;
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use libnvvm_sys::{LibNvvm, Program};
use nvjitlink_sys::{InputType, LibNvJitLink, Linker};

/// Compile this example's emitted NVVM IR with extended split compilation.
/// Writes `nvvm_entry_allocas.ptx` and `nvvm_entry_allocas.cubin` beside the
/// emitted IR for the code-shape check.
fn compile_split_cubin() -> Vec<u8> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let arch = std::fs::read_to_string(root.join("nvvm_entry_allocas.target"))
        .expect("read target file; run cargo oxide with --emit-nvvm-ir --arch sm_120");
    let arch = arch
        .lines()
        .next()
        .expect("target file has an architecture")
        .trim();
    let compute = arch.replacen("sm_", "compute_", 1);
    let ir = std::fs::read(root.join("nvvm_entry_allocas.ll")).expect("read emitted NVVM IR");
    let nvvm = LibNvvm::load().expect("load libNVVM");
    let mut program = Program::new(&nvvm).expect("create NVVM program");
    program
        .add_module(&ir, "nvvm_entry_allocas")
        .expect("add NVVM IR");
    let ltoir = program
        .compile(&[&format!("-arch={compute}"), "-gen-lto"])
        .expect("compile NVVM IR to LTOIR");
    std::fs::write(root.join("nvvm_entry_allocas.ltoir"), &ltoir).expect("write LTOIR");
    let library = LibNvJitLink::load().expect("load nvJitLink");
    let mut linker = Linker::new(
        &library,
        &[
            &format!("-arch={arch}"),
            "-lto",
            "-split-compile=0",
            "-split-compile-extended=0",
            "-fma=0",
            "-no-cache",
        ],
    )
    .expect("create extended split linker");
    linker
        .add(InputType::Ltoir, &ltoir, "nvvm_entry_allocas.ltoir")
        .expect("add LTOIR");
    let cubin = linker.finish().expect("link extended split cubin");
    std::fs::write(root.join("nvvm_entry_allocas.cubin"), &cubin).expect("write cubin");
    {
        let mut linker = Linker::new(
            &library,
            &[
                &format!("-arch={arch}"),
                "-lto",
                "-ptx",
                "-split-compile=0",
                "-split-compile-extended=0",
                "-fma=0",
                "-no-cache",
            ],
        )
        .unwrap();
        linker
            .add(InputType::Ltoir, &ltoir, "nvvm_entry_allocas.ltoir")
            .unwrap();
        std::fs::write(
            root.join("nvvm_entry_allocas.ptx"),
            linker.finish_ptx().unwrap(),
        )
        .unwrap();
    }
    cubin
}

/// Four lanes passed together as an outgoing device argument.
/// This keeps the aggregate call and its 128-bit parameter stores observable.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Lanes {
    /// Four lane values form the aggregate argument consumed after the scratch read.
    /// Distinct weights in `consume` make every lane contribute to the oracle.
    pub values: [f32; 4],
}

/// Keep an aggregate call boundary after the scratch slot's last read.
/// Each lane has a distinct weight so corruption cannot cancel in the sum.
#[inline(never)]
fn consume(prefix: u64, value: Lanes) -> f32 {
    prefix as f32
        + value.values[0]
        + 2.0 * value.values[1]
        + 4.0 * value.values[2]
        + 8.0 * value.values[3]
}

/// A tagged aggregate forces payload extraction after a control-flow branch.
/// Loading it by value lets enum lowering introduce a temporary payload slot.
#[derive(Clone, Copy)]
#[repr(C)]
pub enum Packet {
    /// Three lanes occupy 12 bytes and need four-byte alignment.
    /// Dynamic indexing keeps the payload read addressable.
    Data([f32; 3]),
    /// A scalar alternative keeps the match data dependent.
    /// It follows a different payload extraction path.
    Scalar(f32),
}

// SAFETY: both variants contain only plain f32 values, with no pointers,
// ownership, or drop glue. Zero bytes are Data([0.0; 3]); the host initializes
// every field before copying, and this program never copies enum bytes back.
unsafe impl cuda_core::DeviceCopy for Packet {}

/// Read private scratch through a separate device call at a runtime index.
/// The borrowed array keeps its storage addressable while this call executes.
#[inline(never)]
fn read_scratch(values: &[f32; 3], lane: usize) -> f32 {
    values[lane]
}

/// Extract a tagged payload into scratch in a conditional block.
/// Keeping this helper until LLVM inlining exercises entry normalization.
#[inline(always)]
fn read_packet(packet: Packet, lane: usize) -> f32 {
    match packet {
        Packet::Data(values) => read_scratch(&values, (lane >> 1) & 1),
        Packet::Scalar(value) => value,
    }
}

/// Kernels exercising entry-block scratch placement (#1681 and #1687).
#[cuda_module]
mod kernels {
    use super::*;

    /// Invoke the inlined scratch helper after a data-dependent branch (#1681).
    /// Each active thread owns one output slot through `DisjointSlice`.
    #[kernel]
    pub fn branch_scratch(input: &[Packet], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if let Some(slot) = output.get_mut(index) {
            let packet = input[i];
            let value = i as f32;
            let mut sum = 0.0;
            let mut round = 0;
            while round < (i & 3) + 1 {
                let value = value + round as f32;
                let selected = read_packet(packet, i + round);
                sum += consume(
                    i as u64,
                    Lanes {
                        values: [selected, value + 3.0, value + 4.0, value + 5.0],
                    },
                );
                round += 1;
            }
            *slot = sum;
        }
    }

    /// Column dot product `out[c] = sum_r w[r] * x[r, c]` using `core::iter::Zip`
    /// over a slice iterator and a `ColView32` iterator (#1687).
    #[kernel(launch_context = lc)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, coordinates = u32, block = (64, 1, 1))]
    pub fn core_zip(mut out: DisjointSlice<f32>, x: &[f32], w: &[f32]) {
        let t = thread::index_1d_u32(lc);
        let cols = out.len() as u32;
        let c = t.get();
        let Some(mut o) = out.element_thread32(t) else {
            return;
        };
        let Some(col) = MatrixView32::new(x, cols).col(c, w.len() as u32) else {
            return;
        };
        let mut acc = 0.0;
        for (a, b) in w.iter().zip(col.iter()) {
            acc += a * b;
        }
        o.write(acc);
    }
}

/// Launch both `branch_scratch` and `core_zip` and verify their outputs
/// against host reference calculations.
fn main() {
    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let input: Vec<Packet> = (0..1024)
        .map(|i| {
            let value = i as f32;
            if i & 1 == 0 {
                Packet::Data([value, value + 1.0, value + 2.0])
            } else {
                Packet::Scalar(value + 6.0)
            }
        })
        .collect();
    let device_input = DeviceBuffer::from_host(&stream, &input).expect("copy input");
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, input.len()).expect("allocate output");
    let cubin = compile_split_cubin();
    let module = context
        .load_module_from_image(&cubin)
        .expect("load extended split cubin");
    // SAFETY: `cubin` was compiled from this crate's own `nvvm_entry_allocas.ll`,
    // whose kernel entry signatures match `kernels`.
    let module =
        unsafe { kernels::from_module(module) }.expect("load typed nvvm_entry_allocas module");
    // SAFETY: both buffers cover the launch, and each active thread writes
    // its own output element and reads the corresponding input.
    unsafe {
        module.branch_scratch(
            &stream,
            LaunchConfig::for_num_elems(input.len() as u32),
            &device_input,
            &mut output,
        )
    }
    .expect("launch branch_scratch");
    let actual = output
        .to_host_vec(&stream)
        .expect("copy output and synchronize");
    for (i, (&packet, &got)) in input.iter().zip(&actual).enumerate() {
        let expected: f32 = (0..(i & 3) + 1)
            .map(|round| {
                let value = i as f32 + round as f32;
                let lane = i + round;
                i as f32
                    + (match packet {
                        Packet::Data(values) => values[(lane >> 1) & 1],
                        Packet::Scalar(value) => value,
                    })
                    + 2.0 * (value + 3.0)
                    + 4.0 * (value + 4.0)
                    + 8.0 * (value + 5.0)
            })
            .sum();
        assert_eq!(got, expected, "branch_scratch element {i}");
    }

    const ROWS: usize = 16;
    const COLS: usize = 64;
    let w_host: Vec<f32> = (0..ROWS).map(|r| (r + 1) as f32 * 0.25).collect();
    let x_host: Vec<f32> = (0..ROWS * COLS)
        .map(|idx| ((idx % 17) as f32) - 4.0)
        .collect();
    let w_dev = DeviceBuffer::from_host(&stream, &w_host).expect("copy w");
    let x_dev = DeviceBuffer::from_host(&stream, &x_host).expect("copy x");
    let mut zip_out = DeviceBuffer::<f32>::zeroed(&stream, COLS).expect("allocate zip_out");
    let zip_launch = module
        .prepare_core_zip(LaunchConfig1D::new(1, COLS as u32, 0))
        .expect("prepare core_zip");
    module
        .core_zip(&stream, &zip_launch, &mut zip_out, &x_dev, &w_dev)
        .expect("launch core_zip");
    let zip_actual = zip_out.to_host_vec(&stream).expect("copy zip_out");
    for (col, &got) in zip_actual.iter().enumerate() {
        let expected: f32 = (0..ROWS).map(|r| w_host[r] * x_host[r * COLS + col]).sum();
        assert_eq!(got, expected, "core_zip col {col}");
    }

    println!(
        "SUCCESS: nvvm_entry_allocas verified {} elements and {} zip columns",
        actual.len(),
        zip_actual.len()
    );
}
