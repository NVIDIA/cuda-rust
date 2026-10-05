/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed-API host-closure smoke test.
//!
//! Drives the typed `#[cuda_module]` launch path with a generic kernel that
//! takes a closure. This exercises the kernel-boundary ABI fix that lets the
//! typed path push the closure as a single byval `.param` and have the
//! backend emit a single matching `.param` declaration. Before the fix this
//! example only worked through the call-site `cuda_launch!` macro because
//! that path pushed each capture individually; the typed API kept the
//! closure intact and mis-matched the backend's flattened ABI.
//!
//! Build and run with:
//!   cargo oxide run host_closure
//!
//! ## What it covers
//!
//! 1. Generic kernel with `Fn` trait bound: `fn map<T, F: Fn(T) -> T + Copy>(...)`
//! 2. Closures with 0, 1, 2, 3, 4 captures.
//! 3. Type inference of `F` at the call site (`module.map::<f32, _>(...)`).
//! 4. The closure is passed as a single byval struct — no per-capture flattening
//!    on either side.
//! 5. Equivalent coverage through sync and async typed and untyped launch paths.
//! 6. A where-clause `Fn` bound and layout-sensitive captures.
//! 7. `FnMut` and `FnOnce` trait bounds, so the device-side dispatch path is
//!    exercised for all three closure traits.
//! 8. Both reordered (default Rust repr, mixed-size fields) and identity-order
//!    (`#[repr(C)]`) struct captures.

use cuda_async::simt::device_operation::DeviceOperation;
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::{cuda_launch, cuda_launch_async, cuda_module, load_kernel_module};

#[derive(Clone, Copy)]
struct MixedCapture {
    small: u8,
    wide: f64,
    scale: f32,
}

/// Same field set as [`MixedCapture`], but `#[repr(C)]` pins the layout to
/// declaration order (no rustc field reordering). The closure-layout fix has
/// to handle both the reordered and the identity-order cases, so we exercise
/// the identity path explicitly.
#[derive(Clone, Copy)]
#[repr(C)]
struct ReprCMixed {
    small: u8,
    wide: f64,
    scale: f32,
}

// =============================================================================
// CLOSURE-ACCEPTING GENERIC KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Generic map kernel — applies a function to each element.
    ///
    /// `F` is bound to the closure's anonymous type at the call site via
    /// the typed API's turbofish placeholder (`module.map::<f32, _>(...)`).
    /// The closure value itself is pushed as one byval kernel argument; the
    /// device reads it back as `F` and calls `f(input[idx])` on each thread.
    #[kernel]
    pub fn map<T: Copy, F: Fn(T) -> T + Copy>(f: F, input: &[T], mut out: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = f(input[idx_raw]);
        }
    }

    /// Closure-generic kernel with a const parameter that cannot be inferred
    /// from its runtime arguments. This exercises explicit generic forwarding
    /// in both low-level launch macros.
    #[kernel]
    pub fn map_with_const<F, const OFFSET: u32>(f: F, input: &[u32], mut out: DisjointSlice<u32>)
    where
        F: Fn(u32) -> u32 + Copy,
    {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = f(input[idx_raw]) + OFFSET;
        }
    }

    #[kernel]
    pub fn map_where<T, F>(f: F, input: &[T], mut out: DisjointSlice<T>)
    where
        T: Copy,
        F: Fn(T) -> T + Copy,
    {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = f(input[idx_raw]);
        }
    }

    /// Same as [`map`] but with a `FnMut` trait bound, so the device-side
    /// dispatch goes through `<F as FnMut>::call_mut` instead of `Fn::call`.
    /// The closure is `Copy`, so each thread mutates its own register-resident
    /// copy of the closure environment.
    #[kernel]
    pub fn map_mut<T, F>(mut f: F, input: &[T], mut out: DisjointSlice<T>)
    where
        T: Copy,
        F: FnMut(T) -> T + Copy,
    {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = f(input[idx_raw]);
        }
    }

    /// Same as [`map`] but with a `FnOnce` trait bound, so the device-side
    /// dispatch goes through `<F as FnOnce>::call_once` (which consumes the
    /// closure). The `Copy` bound lets the kernel re-invoke `call_once` per
    /// thread on a freshly-copied closure rather than running it exactly once.
    #[kernel]
    pub fn map_once<T, F>(f: F, input: &[T], mut out: DisjointSlice<T>)
    where
        T: Copy,
        F: FnOnce(T) -> T + Copy,
    {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = f(input[idx_raw]);
        }
    }
}

fn verify_output(
    label: &str,
    output: &[f32],
    input: &[f32],
    tolerance: f32,
    expected: impl Fn(f32) -> f32,
    failed: &mut bool,
) {
    let errors = output
        .iter()
        .zip(input)
        .filter(|(got, x)| (**got - expected(**x)).abs() > tolerance)
        .count();

    if errors == 0 {
        println!("  ✓ SUCCESS: {label}");
    } else {
        println!("  ✗ FAILED: {label}: {errors} errors");
        *failed = true;
        for (i, (&got, &x)) in output.iter().zip(input).take(5).enumerate() {
            let expected = expected(x);
            if (got - expected).abs() > tolerance {
                println!("    [{i}]: got {got}, expected {expected}");
            }
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Typed Closure Kernel Test ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let input_data: Vec<f32> = (0..N).map(|i| i as f32).collect();

    let input_dev = DeviceBuffer::from_host(&stream, &input_data)?;
    let mut output_dev: DeviceBuffer<f32>;

    let module = load_kernel_module(&ctx, "host_closure")
        .map_err(|err| format!("failed to load host_closure kernel module: {err}"))?;
    let typed_module = kernels::from_module(module.clone())
        .map_err(|err| format!("failed to initialize typed host_closure module: {err}"))?;
    use kernels::*;

    let mut failed = false;

    macro_rules! run_launch_matrix {
        (
            $label:literal,
            $kernel:path,
            $typed_method:ident,
            $typed_async_method:ident,
            $closure:expr,
            $expected:expr,
            $tolerance:expr
        ) => {{
            output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
            // SAFETY: closure-then-slices matches every kernel this matrix
            // launches (closure env, input slice, output slice); both buffers
            // are live DeviceBuffers and the closure's captures are
            // host-accessible via HMM.
            unsafe {
                cuda_launch! {
                    kernel: $kernel,
                    stream: stream,
                    module: module,
                    config: LaunchConfig::for_num_elems(N as u32),
                    args: [$closure, slice(input_dev), slice_mut(output_dev)]
                }
            }?;
            let output_host = output_dev.to_host_vec(&stream)?;
            verify_output(
                concat!("cuda_launch ", $label),
                &output_host,
                &input_data,
                $tolerance,
                $expected,
                &mut failed,
            );

            output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe {
                typed_module.$typed_method::<f32, _>(
                    stream.as_ref(),
                    LaunchConfig::for_num_elems(N as u32),
                    $closure,
                    &input_dev,
                    &mut output_dev,
                )
            }?;
            let output_host = output_dev.to_host_vec(&stream)?;
            verify_output(
                concat!("typed launch ", $label),
                &output_host,
                &input_data,
                $tolerance,
                $expected,
                &mut failed,
            );

            output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
            // SAFETY: this is a 1D launch and each kernel guards its 1D index
            // against the length carried by the output slice.
            unsafe {
                cuda_launch_async! {
                    kernel: $kernel,
                    module: module,
                    config: LaunchConfig::for_num_elems(N as u32),
                    args: [$closure, slice(input_dev), slice_mut(output_dev)]
                }
            }
            .sync_on(&stream)?;
            let output_host = output_dev.to_host_vec(&stream)?;
            verify_output(
                concat!("cuda_launch_async ", $label),
                &output_host,
                &input_data,
                $tolerance,
                $expected,
                &mut failed,
            );

            output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
            // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
            unsafe {
                typed_module.$typed_async_method::<f32, _>(
                    LaunchConfig::for_num_elems(N as u32),
                    $closure,
                    &input_dev,
                    &mut output_dev,
                )
            }?
            .sync_on(&stream)?;
            let output_host = output_dev.to_host_vec(&stream)?;
            verify_output(
                concat!("typed async launch ", $label),
                &output_host,
                &input_data,
                $tolerance,
                $expected,
                &mut failed,
            );
        }};
    }

    // =========================================================================
    // TEST 1: Closure with single capture
    // =========================================================================
    println!("Test 1: Single capture (scale by factor)");
    {
        let factor = 2.5f32;
        println!("  factor = {}", factor);
        println!("  N = {}", N);

        run_launch_matrix!(
            "single capture",
            map::<f32, _>,
            map,
            map_async,
            move |x: f32| x * factor,
            |x| x * factor,
            1e-5
        );
        println!();
    }

    // =========================================================================
    // TEST 2: Closure with multiple captures
    // =========================================================================
    println!("Test 2: Multiple captures (affine transform)");
    {
        let scale = 2.0f32;
        let offset = 10.0f32;
        println!("  scale = {}, offset = {}", scale, offset);

        run_launch_matrix!(
            "multiple captures",
            map::<f32, _>,
            map,
            map_async,
            move |x: f32| x * scale + offset,
            |x| x * scale + offset,
            1e-5
        );
        println!();
    }

    // =========================================================================
    // TEST 3: Zero-capture closure (inline constant)
    // =========================================================================
    println!("Test 3: Zero captures (double each element)");
    {
        run_launch_matrix!(
            "zero-capture closure",
            map::<f32, _>,
            map,
            map_async,
            |x: f32| x * 2.0,
            |x| x * 2.0,
            1e-5
        );
        println!();
    }

    // =========================================================================
    // TEST 4: Closure with 3 captures (polynomial transform)
    // =========================================================================
    println!("Test 4: Three captures (polynomial: a*x^2 + b*x + c)");
    {
        let a = 0.5f32;
        let b = 2.0f32;
        let c = 1.0f32;
        println!("  a = {}, b = {}, c = {}", a, b, c);

        run_launch_matrix!(
            "three captures",
            map::<f32, _>,
            map,
            map_async,
            move |x: f32| a * x * x + b * x + c,
            |x| a * x * x + b * x + c,
            1e-3
        );
        println!();
    }

    // =========================================================================
    // TEST 5: Closure with 4 captures (to ensure arbitrary count works)
    // =========================================================================
    println!("Test 5: Four captures (weighted sum: w1*x + w2 + w3*w4)");
    {
        let w1 = 3.0f32;
        let w2 = 5.0f32;
        let w3 = 2.0f32;
        let w4 = 7.0f32;
        println!("  w1 = {}, w2 = {}, w3 = {}, w4 = {}", w1, w2, w3, w4);

        run_launch_matrix!(
            "four captures",
            map::<f32, _>,
            map,
            map_async,
            move |x: f32| w1 * x + w2 + w3 * w4,
            |x| w1 * x + w2 + w3 * w4,
            1e-3
        );
        println!();
    }

    // =========================================================================
    // TEST 6: where-clause Fn bound
    // =========================================================================
    println!("Test 6: where-clause Fn bound");
    {
        let scale = 1.5f32;
        let bias = 4.0f32;
        println!("  scale = {}, bias = {}", scale, bias);

        run_launch_matrix!(
            "where-clause Fn bound",
            map_where::<f32, _>,
            map_where,
            map_where_async,
            move |x: f32| x * scale - bias,
            |x| x * scale - bias,
            1e-5
        );
        println!();
    }

    // =========================================================================
    // TEST 7: mixed-size captures
    // =========================================================================
    println!("Test 7: mixed-size captures");
    {
        let small = 7u8;
        let scale = 1.25f32;
        let wide = 3.5f64;
        println!("  small = {}, scale = {}, wide = {}", small, scale, wide);

        run_launch_matrix!(
            "mixed-size captures",
            map::<f32, _>,
            map,
            map_async,
            move |x: f32| x * scale + wide as f32 + small as f32,
            |x| x * scale + wide as f32 + small as f32,
            1e-5
        );
        println!();
    }

    // =========================================================================
    // TEST 8: struct capture with internal padding
    // =========================================================================
    println!("Test 8: struct capture with internal padding");
    {
        let mixed = MixedCapture {
            small: 9,
            wide: 2.25,
            scale: 0.75,
        };
        println!(
            "  small = {}, scale = {}, wide = {}",
            mixed.small, mixed.scale, mixed.wide
        );

        run_launch_matrix!(
            "struct capture with internal padding",
            map::<f32, _>,
            map,
            map_async,
            move |x: f32| x * mixed.scale + mixed.wide as f32 + mixed.small as f32,
            |x| x * mixed.scale + mixed.wide as f32 + mixed.small as f32,
            1e-5
        );
        println!();
    }

    // =========================================================================
    // TEST 9: non-move reference captures
    // =========================================================================
    //
    // NOTE: this case relies on Heterogeneous Memory Management. Without
    // `move`, the closure environment holds `&f32` pointers into the host
    // stack frame of `main`; the GPU dereferences those pointers at kernel
    // entry. Working configurations need an HMM-capable GPU (sm_75+ on
    // Linux) with HMM enabled in the driver. On systems without HMM this
    // surfaces as `CUDA_ERROR_ILLEGAL_ADDRESS` from the launch, which the
    // smoketest's `verdict_standard` correctly flags as FAIL.
    println!("Test 9: non-move reference captures");
    {
        let scale = 0.5f32;
        let bias = 8.0f32;
        println!("  scale = {}, bias = {}", scale, bias);
        println!("  captures are borrowed by the closure environment");

        run_launch_matrix!(
            "non-move reference captures",
            map::<f32, _>,
            map,
            map_async,
            |x: f32| x * scale + bias,
            |x| x * scale + bias,
            1e-5
        );
        println!();
    }

    // =========================================================================
    // TEST 10: FnMut bound — different device-side dispatch from Fn
    // =========================================================================
    println!("Test 10: FnMut bound (multi-capture)");
    {
        let scale = 1.5f32;
        let offset = 6.0f32;
        println!("  scale = {}, offset = {}", scale, offset);

        run_launch_matrix!(
            "FnMut bound, multi-capture",
            map_mut::<f32, _>,
            map_mut,
            map_mut_async,
            move |x: f32| x * scale + offset,
            |x| x * scale + offset,
            1e-5
        );
        println!();
    }

    // =========================================================================
    // TEST 11: FnOnce bound — call_once dispatch with Copy
    // =========================================================================
    println!("Test 11: FnOnce bound (multi-capture, Copy closure)");
    {
        let weight = 2.25f32;
        let intercept = 12.5f32;
        println!("  weight = {}, intercept = {}", weight, intercept);

        run_launch_matrix!(
            "FnOnce bound, multi-capture",
            map_once::<f32, _>,
            map_once,
            map_once_async,
            move |x: f32| x * weight + intercept,
            |x| x * weight + intercept,
            1e-5
        );
        println!();
    }

    // =========================================================================
    // TEST 12: #[repr(C)] struct capture (identity-order layout)
    // =========================================================================
    // Counterpart to test 8: the layout fix has to handle both the reordered
    // case (default Rust repr, fields packed by size) and the identity-order
    // case (`#[repr(C)]`, fields kept in declaration order). The closure
    // body is the same as test 8 so the expected output is comparable.
    println!("Test 12: #[repr(C)] struct capture (identity-order layout)");
    {
        let mixed = ReprCMixed {
            small: 9,
            wide: 2.25,
            scale: 0.75,
        };
        println!(
            "  small = {}, scale = {}, wide = {}",
            mixed.small, mixed.scale, mixed.wide
        );

        run_launch_matrix!(
            "repr(C) struct capture",
            map::<f32, _>,
            map,
            map_async,
            move |x: f32| x * mixed.scale + mixed.wide as f32 + mixed.small as f32,
            |x| x * mixed.scale + mixed.wide as f32 + mixed.small as f32,
            1e-5
        );
        println!();
    }

    // =========================================================================
    // TEST 13: Closure type inference plus an explicit const argument
    // =========================================================================
    println!("Test 13: closure plus const-generic specialization");
    {
        let input: Vec<u32> = (0..N as u32).collect();
        let input_dev_u32 = DeviceBuffer::from_host(&stream, &input)?;
        let factor = 3u32;

        let mut output_4 = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        unsafe {
            cuda_launch! {
                kernel: kernels::map_with_const::<_, 4>,
                stream: stream,
                module: module,
                config: LaunchConfig::for_num_elems(N as u32),
                args: [move |x: u32| x * factor, slice(input_dev_u32), slice_mut(output_4)]
            }
        }?;
        let result_4 = output_4.to_host_vec(&stream)?;
        assert!((0..N).all(|i| result_4[i] == input[i] * factor + 4));

        let mut output_8 = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        // SAFETY: `map_with_const` uses a guarded 1D index and this config has
        // no active Y or Z dimensions.
        unsafe {
            cuda_launch_async! {
                kernel: kernels::map_with_const::<_, 8>,
                module: module,
                config: LaunchConfig::for_num_elems(N as u32),
                args: [move |x: u32| x * factor, slice(input_dev_u32), slice_mut(output_8)]
            }
        }
        .sync_on(&stream)?;
        let result_8 = output_8.to_host_vec(&stream)?;
        assert!((0..N).all(|i| result_8[i] == input[i] * factor + 8));

        println!("  ✓ SUCCESS: sync and async launch macros forwarded OFFSET");
    }

    println!("=== All Tests Complete ===");
    if failed {
        Err("one or more host_closure tests failed".into())
    } else {
        Ok(())
    }
}
