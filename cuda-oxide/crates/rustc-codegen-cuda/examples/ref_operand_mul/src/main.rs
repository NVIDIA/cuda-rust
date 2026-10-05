/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Reference-operand `Mul` regression test (issue #133).
//!
//! A `#[kernel]` that multiplies two struct values through references:
//! `impl std::ops::Mul for &Foo` with `type Output = Foo`, called as
//! `&a * &b`. The key property is that the trait impl's `Output` associated
//! type (`Foo`) is NOT the same type as the impl's self type (`&Foo`).
//!
//! Before the fix, the device codegen backend rejected this with
//! `Alias type not yet supported: ... std::ops::Mul::Output`. The importer
//! typed call results from the callee's declared trait signature, whose
//! return type is the unresolved associated-type projection
//! `<&Foo as Mul>::Output`. A name-matching fallback then guessed
//! "Output = self type", which only covered non-reference self types (and
//! would have guessed the WRONG type here, a pointer instead of a struct).
//! The importer now types call results from the caller's destination place,
//! which rustc has already resolved to the concrete type `Foo`.
//!
//! Run: cargo oxide run ref_operand_mul

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_host::cuda_module;
use std::sync::Arc;

#[cuda_module]
mod kernels {
    use cuda_device::{DisjointSlice, kernel, thread};

    /// Mirrors the struct from the issue #133 report: a plain aggregate
    /// wrapping a small array of integer limbs.
    #[derive(Copy, Clone, Default)]
    pub struct Foo {
        pub pieces: [u32; 2],
    }

    /// The shape that regressed: the impl is on `&Foo` (a reference), but
    /// `Output` is the owned struct `Foo`. So `Output != Self`, and any
    /// "the output is the self type" guess types the result as a pointer
    /// where the caller expects a by-value struct.
    impl std::ops::Mul for &Foo {
        type Output = Foo;

        // `#[inline(never)]` keeps the trait-method call alive through MIR
        // inlining so the `<&Foo as Mul>::Output` projection actually
        // reaches the importer's call-typing path. The return type is
        // written as `Self::Output`, exactly as in the issue report.
        #[inline(never)]
        fn mul(self, rhs: Self) -> Self::Output {
            Foo {
                pieces: [
                    self.pieces[0] * rhs.pieces[0],
                    self.pieces[1] * rhs.pieces[1],
                ],
            }
        }
    }

    /// Computes `(a * b) * (a * b)` element-wise through `&Foo` operands
    /// and writes the sum of the result limbs.
    #[kernel]
    pub fn ref_pieces_mul(x0: u32, x1: u32, y0: u32, y1: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let a = Foo { pieces: [x0, x1] };
            let b = Foo { pieces: [y0, y1] };
            // Two-operand form: `<&Foo as Mul>::mul(&a, &b)`.
            let prod = &a * &b;
            // Same-operand form from the issue report: `&tmp * &tmp`.
            let sq = &prod * &prod;
            *out_elem = sq.pieces[0] + sq.pieces[1];
        }
    }
}

fn main() {
    println!("=== Ref-Operand Mul Test (issue #133) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    println!("Device ordinal: {}\n", ctx.ordinal());

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let stream = ctx.default_stream();
    let ok = run_ref_pieces_mul(&module, &stream);

    if !ok {
        println!("\nFAILURE: kernel returned a wrong value");
        std::process::exit(1);
    }
    println!("\n=== Test Complete ===");
}

fn run_ref_pieces_mul(module: &kernels::LoadedModule, stream: &Arc<CudaStream>) -> bool {
    let mut d_out = DeviceBuffer::<u32>::zeroed(stream, 1).unwrap();
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // a = (2, 3), b = (4, 5): prod = (8, 15), sq = (64, 225); sum = 289.
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.ref_pieces_mul(
            stream.as_ref(),
            config,
            2_u32,
            3_u32,
            4_u32,
            5_u32,
            &mut d_out,
        )
    }
    .expect("Kernel launch failed");

    let result = d_out.to_host_vec(stream).unwrap()[0];
    let expected = 289_u32;
    if result == expected {
        println!("ref_pieces_mul: PASS (result = {})", result);
        true
    } else {
        println!(
            "ref_pieces_mul: FAIL (expected {}, got {})",
            expected, result
        );
        false
    }
}
