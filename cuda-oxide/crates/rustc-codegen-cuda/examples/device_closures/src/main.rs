/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::redundant_closure_call)]

//! Unified Device Closures Example
//!
//! Demonstrates closure patterns in unified compilation:
//!
//! 1. **Inline closures**: Closures defined and used within the kernel (always work)
//! 2. **Scalarized captures**: Kernel parameters that represent closure captures
//! 3. **Closures passed to device functions**: Using FnOnce/Fn traits
//!
//! Build and run with:
//!   cargo oxide run device_closures
//!
//! ## The Closure Story
//!
//! In CUDA C++, you can write:
//! ```cpp
//! int factor = 5;
//! auto scale = [=](int x) { return x * factor; };
//! kernel<<<...>>>(scale, input, output);
//! ```
//!
//! The nvc++ compiler handles this by:
//! 1. Serializing the closure captures (factor = 5)
//! 2. Passing them as kernel arguments
//! 3. Reconstructing the closure on device
//!
//! In cuda-oxide unified compilation, we support two patterns:
//!
//! **Pattern A: Inline Closures** (works now!)
//! ```rust
//! #[kernel]
//! fn kernel(mut out: DisjointSlice<u32>) {
//!     let closure = |x: u32| x * 2;  // Defined inside kernel
//!     if let Some(slot) = out.get_mut(idx) {
//!         *slot = closure(val);  // Inlined by compiler
//!     }
//! }
//! ```
//!
//! **Pattern B: Scalarized Captures** (works now!)
//! ```rust
//! #[kernel]
//! fn scale_kernel(factor: u32, input: &[u32], mut out: DisjointSlice<u32>) {
//!     // 'factor' is the closure capture, passed as scalar argument
//!     let idx = thread::index_1d();
//!     if let Some(slot) = out.get_mut(idx) {
//!         *slot = input[idx.get()] * factor;
//!     }
//! }
//! ```
//!
//! **Pattern C: True Closures** (future work)
//! ```rust
//! let factor = 5;
//! let scale = move |x| x * factor;
//! module.map_kernel(&stream, config, scale, &input, &mut out)?;
//! // The typed launch method passes the capture values as kernel arguments
//! ```

use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Pattern A: Inline closure (defined and used within kernel)
    /// The closure has no captures from outside the kernel - it's fully self-contained.
    #[kernel]
    pub fn test_inline_closure(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            // Closure defined inline - will be optimized away
            let double = |x: u32| x * 2;
            let val = idx_raw as u32;
            *out_elem = double(val);
        }
    }

    /// Pattern B: Scalarized capture - single value
    /// The 'factor' parameter represents what would be a closure capture.
    /// Host passes factor=5, kernel uses it like a captured variable.
    #[kernel]
    pub fn scale_kernel(factor: u32, input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            // This is equivalent to: let scale = move |x| x * factor;
            *out_elem = input[idx_raw] * factor;
        }
    }

    /// Pattern B: Scalarized captures - multiple values
    /// Multiple captures (offset, scale) are passed as separate scalar arguments.
    #[kernel]
    pub fn transform_kernel(offset: i32, scale: i32, input: &[i32], mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            // Equivalent to: let transform = move |x| (x + offset) * scale;
            let x = input[idx_raw];
            *out_elem = (x + offset) * scale;
        }
    }

    /// Pattern A+B combined: Inline closure using kernel parameter
    /// The closure is defined inside the kernel but captures a kernel parameter.
    #[kernel]
    pub fn inline_with_param(factor: u32, input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            // Closure captures 'factor' which is a kernel parameter
            let scale = |x: u32| x * factor;
            *out_elem = scale(input[idx_raw]);
        }
    }

    // =============================================================================
    // ADVANCED: Closures Passed to Device Functions
    // =============================================================================

    /// Helper function that takes a closure (FnOnce)
    #[inline(always)]
    fn apply_closure<F: FnOnce(u32) -> u32>(f: F, x: u32) -> u32 {
        f(x)
    }

    #[inline(never)]
    fn apply_once_noinline<F: FnOnce(u32) -> u32>(f: F, x: u32) -> u32 {
        f(x)
    }

    struct ClosureWrapper<F> {
        f: F,
        offset: u32,
    }

    impl<F> ClosureWrapper<F>
    where
        F: Fn(u32) -> u32,
    {
        #[inline(never)]
        fn call(&self, args: (u32,)) -> u32 {
            (self.f)(args.0) + self.offset
        }
    }

    // Note: apply_twice with Fn trait is commented out due to control flow issues.
    // The `Fn` trait requires the closure to be called multiple times, which
    // can create complex control flow that the backend doesn't fully support yet.
    // #[inline(always)]
    // fn apply_twice<F: Fn(u32) -> u32>(f: F, x: u32) -> u32 {
    //     f(f(x))
    // }

    /// Test: Closure with no captures, no arguments - just returns constant
    #[kernel]
    pub fn test_closure_constant(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let f = || 42u32;
            *out_elem = f();
        }
    }

    /// Test: Closure with multiple arguments, no captures
    #[kernel]
    pub fn test_closure_multi_arg(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let add = |a: u32, b: u32| a + b;
            let val = idx_raw as u32;
            *out_elem = add(val, 10);
        }
    }

    /// Test: Passing closure to FnOnce generic function
    #[kernel]
    pub fn test_closure_fnonce(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let triple = |x: u32| x * 3;
            let val = idx_raw as u32;
            *out_elem = apply_closure(triple, val);
        }
    }

    // Note: test_closure_apply_twice is commented out due to control flow issues
    // with the Fn trait (closure called multiple times).
    // #[kernel]
    // pub fn test_closure_apply_twice(mut out: DisjointSlice<u32>) {
    //     let idx = thread::index_1d();
    //     if let Some(slot) = out.get_mut(idx) {
    //         let inc = |x: u32| x + 1;
    //         let val = idx.get() as u32;
    //         *slot = apply_twice(inc, val);
    //     }
    // }

    /// An immutable capture implements `Fn` but is consumed through `FnOnce`;
    /// the adapter's direct body call must synthesize an exact `&Closure`.
    #[kernel]
    pub fn test_closure_capture_fnonce(factor: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let scale = |x: u32| x * factor;
            let val = idx_raw as u32;
            *out_elem = apply_closure(scale, val);
        }
    }

    /// A closure that mutates its capture implements `FnMut`, but the generic
    /// helper consumes it through `FnOnce`. rustc inserts a ClosureOnce adapter
    /// whose direct body call must synthesize an exact `&mut Closure` receiver.
    #[kernel]
    pub fn test_closure_fnmut_adapter(seed: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut state = seed + idx_raw as u32;
            let bump = |delta: u32| {
                state += delta;
                state
            };
            *out_elem = apply_closure(bump, 3);
        }
    }

    /// Control: consuming a non-Copy capture makes this a genuine `FnOnce`.
    /// Its closure body receives the environment by value, so bypassing the
    /// callable-trait dispatch must not synthesize any receiver borrow.
    #[kernel]
    pub fn test_genuine_fnonce_receiver(seed: u32, mut out: DisjointSlice<u32>) {
        struct Token(u32);
        impl Token {
            fn into_value(self) -> u32 {
                self.0
            }
        }

        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let token = Token(seed + idx_raw as u32);
            let consume = move |delta: u32| token.into_value() + delta;
            *out_elem = apply_once_noinline(consume, 5);
        }
    }

    /// Wrapper method named `call` whose generic substitutions contain a
    /// closure. This is not a closure trait shim: the receiver is the wrapper,
    /// so the tuple argument must remain one ordinary method argument.
    #[kernel]
    pub fn test_wrapper_method_named_call(factor: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let scale = |x: u32| x * factor;
            let wrapper = ClosureWrapper {
                f: scale,
                offset: 1,
            };
            *out_elem = wrapper.call((idx_raw as u32,));
        }
    }

    /// Genuine closure call whose receiver is reached through a struct field
    /// (`(holder.f)(x)`). The tightened `receiver_is_closure` guard must still
    /// unpack the rust-call tuple here: rustc materializes the `&{closure}`
    /// receiver into its own temporary local, so the base-local type check
    /// sees a closure even though the closure lives in a field.
    #[kernel]
    pub fn test_closure_field_projection(factor: u32, mut out: DisjointSlice<u32>) {
        struct Holder<F> {
            f: F,
        }
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let holder = Holder {
                f: |x: u32| x * factor,
            };
            *out_elem = (holder.f)(idx_raw as u32);
        }
    }

    /// Genuine closure call through two reference levels (`(**rr)(x)`). The
    /// guard peels a single reference; this still classifies as a closure
    /// because rustc resolves the receiver to one `&{closure}` borrow at the
    /// call site regardless of the source-level double indirection.
    #[kernel]
    pub fn test_closure_double_ref(factor: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let closure = |x: u32| x * factor;
            let r = &closure;
            let rr = &r;
            *out_elem = (**rr)(idx_raw as u32);
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    println!("=== Unified Compilation: Closures ===\n");

    use cuda_core::simt::LaunchConfig;
    use cuda_core::{CudaContext, DeviceBuffer};

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // =========================================================================
    // Test 1: Inline closure (no external captures)
    // =========================================================================
    println!("Test 1: Inline closure (double = |x| x * 2)");
    {
        const N: usize = 8;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.test_inline_closure(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(N as u32),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        let out: Vec<u32> = out_dev.to_host_vec(&stream).unwrap();
        let expected: Vec<u32> = (0..N).map(|i| (i * 2) as u32).collect();

        println!("  Output:   {:?}", out);
        println!("  Expected: {:?}", expected);
        assert_eq!(out, expected);
        println!("  ✓ PASSED\n");
    }

    // =========================================================================
    // Test 2: Scalarized single capture (factor)
    // =========================================================================
    println!("Test 2: Scalarized capture (scale = |x| x * factor)");
    {
        const N: usize = 8;
        let input: Vec<u32> = (1..=N).map(|i| i as u32).collect();
        let factor: u32 = 5;

        println!("  factor = {}", factor);

        let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.scale_kernel(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(N as u32),
                factor,
                &input_dev,
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        let out: Vec<u32> = out_dev.to_host_vec(&stream).unwrap();
        let expected: Vec<u32> = input.iter().map(|&x| x * factor).collect();

        println!("  Input:    {:?}", input);
        println!("  Output:   {:?}", out);
        println!("  Expected: {:?}", expected);
        assert_eq!(out, expected);
        println!("  ✓ PASSED\n");
    }

    // =========================================================================
    // Test 3: Multiple scalarized captures (offset, scale)
    // =========================================================================
    println!("Test 3: Multiple captures (transform = |x| (x + offset) * scale)");
    {
        const N: usize = 8;
        let input: Vec<i32> = (0..N).map(|i| i as i32).collect();
        let offset: i32 = 10;
        let scale: i32 = 3;

        println!("  offset = {}, scale = {}", offset, scale);

        let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.transform_kernel(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(N as u32),
                offset,
                scale,
                &input_dev,
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        let out: Vec<i32> = out_dev.to_host_vec(&stream).unwrap();
        let expected: Vec<i32> = input.iter().map(|&x| (x + offset) * scale).collect();

        println!("  Input:    {:?}", input);
        println!("  Output:   {:?}", out);
        println!("  Expected: {:?}", expected);
        assert_eq!(out, expected);
        println!("  ✓ PASSED\n");
    }

    // =========================================================================
    // Test 4: Inline closure capturing kernel parameter
    // =========================================================================
    println!("Test 4: Inline closure with kernel parameter (|x| x * factor)");
    {
        const N: usize = 8;
        let input: Vec<u32> = (1..=N).map(|i| i as u32).collect();
        let factor: u32 = 7;

        println!("  factor = {}", factor);

        let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.inline_with_param(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(N as u32),
                factor,
                &input_dev,
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        let out: Vec<u32> = out_dev.to_host_vec(&stream).unwrap();
        let expected: Vec<u32> = input.iter().map(|&x| x * factor).collect();

        println!("  Input:    {:?}", input);
        println!("  Output:   {:?}", out);
        println!("  Expected: {:?}", expected);
        assert_eq!(out, expected);
        println!("  ✓ PASSED\n");
    }

    // =========================================================================
    // Test 5: Constant closure (no args, returns 42)
    // =========================================================================
    println!("Test 5: Constant closure (|| 42)");
    {
        const N: usize = 8;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.test_closure_constant(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(N as u32),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        let out: Vec<u32> = out_dev.to_host_vec(&stream).unwrap();
        let expected: Vec<u32> = vec![42; N];

        println!("  Output:   {:?}", out);
        println!("  Expected: {:?}", expected);
        assert_eq!(out, expected);
        println!("  ✓ PASSED\n");
    }

    // =========================================================================
    // Test 6: Multi-arg closure
    // =========================================================================
    println!("Test 6: Multi-arg closure (|a, b| a + b)");
    {
        const N: usize = 8;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.test_closure_multi_arg(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(N as u32),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        let out: Vec<u32> = out_dev.to_host_vec(&stream).unwrap();
        let expected: Vec<u32> = (0..N).map(|i| i as u32 + 10).collect();

        println!("  Output:   {:?}", out);
        println!("  Expected: {:?}", expected);
        assert_eq!(out, expected);
        println!("  ✓ PASSED\n");
    }

    // =========================================================================
    // Test 7: Closure passed to FnOnce
    // =========================================================================
    println!("Test 7: Closure passed to FnOnce (apply_closure(|x| x * 3, val))");
    {
        const N: usize = 8;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.test_closure_fnonce(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(N as u32),
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        let out: Vec<u32> = out_dev.to_host_vec(&stream).unwrap();
        let expected: Vec<u32> = (0..N).map(|i| (i * 3) as u32).collect();

        println!("  Output:   {:?}", out);
        println!("  Expected: {:?}", expected);
        assert_eq!(out, expected);
        println!("  ✓ PASSED\n");
    }

    // Note: Test 8 (apply_twice with Fn trait) is skipped due to control flow issues.
    // The Fn trait creates complex control flow when the closure is called multiple times.

    // =========================================================================
    // Test 8: Closure with capture passed to FnOnce
    // =========================================================================
    println!("Test 8: Closure with capture via FnOnce (|x| x * factor)");
    {
        const N: usize = 8;
        let factor: u32 = 4;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        println!("  factor = {}", factor);

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.test_closure_capture_fnonce(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(N as u32),
                factor,
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        let out: Vec<u32> = out_dev.to_host_vec(&stream).unwrap();
        let expected: Vec<u32> = (0..N).map(|i| (i * factor as usize) as u32).collect();

        println!("  Output:   {:?}", out);
        println!("  Expected: {:?}", expected);
        assert_eq!(out, expected);
        println!("  ✓ PASSED\n");
    }

    // =========================================================================
    // Test 9: FnMut closure consumed through a FnOnce adapter
    // =========================================================================
    println!("Test 9: FnMut closure through FnOnce adapter");
    {
        const N: usize = 8;
        let seed: u32 = 11;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.test_closure_fnmut_adapter(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(N as u32),
                seed,
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        let out: Vec<u32> = out_dev.to_host_vec(&stream).unwrap();
        let expected: Vec<u32> = (0..N).map(|i| seed + i as u32 + 3).collect();

        println!("  Output:   {:?}", out);
        println!("  Expected: {:?}", expected);
        assert_eq!(out, expected);
        println!("  ✓ PASSED\n");
    }

    // =========================================================================
    // Test 10: genuine by-value FnOnce receiver
    // =========================================================================
    println!("Test 10: genuine by-value FnOnce receiver");
    {
        const N: usize = 8;
        let seed: u32 = 13;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.test_genuine_fnonce_receiver(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(N as u32),
                seed,
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        let out: Vec<u32> = out_dev.to_host_vec(&stream).unwrap();
        let expected: Vec<u32> = (0..N).map(|i| seed + i as u32 + 5).collect();

        println!("  Output:   {:?}", out);
        println!("  Expected: {:?}", expected);
        assert_eq!(out, expected);
        println!("  ✓ PASSED\n");
    }

    // =========================================================================
    // Test 11: Wrapper method named call with closure generic
    // =========================================================================
    println!("Test 11: Wrapper::call with closure generic");
    {
        const N: usize = 8;
        let factor: u32 = 6;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        println!("  factor = {}", factor);

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.test_wrapper_method_named_call(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(N as u32),
                factor,
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        let out: Vec<u32> = out_dev.to_host_vec(&stream).unwrap();
        let expected: Vec<u32> = (0..N).map(|i| (i as u32) * factor + 1).collect();

        println!("  Output:   {:?}", out);
        println!("  Expected: {:?}", expected);
        assert_eq!(out, expected);
        println!("  ✓ PASSED\n");
    }

    // =========================================================================
    // Test 12: genuine closure called through a struct field projection
    // =========================================================================
    println!("Test 12: closure via field projection");
    {
        const N: usize = 8;
        let factor: u32 = 5;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.test_closure_field_projection(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(N as u32),
                factor,
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        let out: Vec<u32> = out_dev.to_host_vec(&stream).unwrap();
        let expected: Vec<u32> = (0..N).map(|i| (i as u32) * factor).collect();

        println!("  Output:   {:?}", out);
        println!("  Expected: {:?}", expected);
        assert_eq!(out, expected);
        println!("  ✓ PASSED\n");
    }

    // =========================================================================
    // Test 13: genuine closure called through a double reference
    // =========================================================================
    println!("Test 13: closure via double reference");
    {
        const N: usize = 8;
        let factor: u32 = 7;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.test_closure_double_ref(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(N as u32),
                factor,
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        let out: Vec<u32> = out_dev.to_host_vec(&stream).unwrap();
        let expected: Vec<u32> = (0..N).map(|i| (i as u32) * factor).collect();

        println!("  Output:   {:?}", out);
        println!("  Expected: {:?}", expected);
        assert_eq!(out, expected);
        println!("  ✓ PASSED\n");
    }

    println!("=== All device closure tests passed! ===");
}
