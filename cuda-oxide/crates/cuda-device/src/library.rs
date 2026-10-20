// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Internal, versioned boundary for native device library calls.
//!
//! This module is an unsafe compiler integration point, not a collective API.

/// Read a Cargo-provided registration and track its native build dependencies.
///
/// Pass the environment variable name as a string literal inside the registration
/// getter. Materialized builds query the provider for file, directory and
/// environment dependencies before rustc emits dep-info. Host checks and rustdoc
/// do not query CUDA inputs. Cargo's timestamp freshness rules still apply;
/// replacements with unchanged or older timestamps may not be detected.
///
/// This snippet requires a build script to set `MY_LIBRARY_REGISTRATION`; the
/// `native_library` example compiles and runs the same registration pattern.
///
/// ```ignore
/// fn registration() -> &'static str {
///     cuda_device::library::registration!("MY_LIBRARY_REGISTRATION")
/// }
/// ```
pub use cuda_macros::library_registration as registration;

/// Invoke a library recipe identified by a constant registration getter.
///
/// Call this marker directly inside a wrapper; the marker itself is not a
/// device function pointer or callable-trait value.
///
/// `Registration` must be a function item returning one constant UTF-8 JSON
/// string. The compiler reads its MIR without calling it. `Args` must be a
/// tuple of the exact scalar and raw-pointer types the provider declares.
/// `T` supplies the scalar type, and the const parameters supply the exact
/// one-dimensional block shape `[BLOCK_THREADS, 1, 1]` and per-thread extent.
///
/// # Safety
///
/// The caller must meet the provider's pointer, lifetime, initialization,
/// participation, and launch-geometry requirements. Library scratch views in
/// one kernel alias one dynamic shared base; the caller must synchronize before
/// reusing it. This version does not permit mixing native dynamic
/// shared views with library calls. It returns no allocation or ownership token.
#[inline(never)]
pub unsafe fn cuda_oxide_library_v1_246e25db_call<
    T: Copy,
    Registration: Fn() -> &'static str + Copy,
    Args: Copy,
    const BLOCK_THREADS: usize,
    const ITEMS_PER_THREAD: usize,
>(
    _registration: Registration,
    _args: Args,
) {
    panic!("library::call_v1 called outside device compilation");
}

pub use cuda_oxide_library_v1_246e25db_call as call_v1;

/// Invoke a full-block library operation with compiler-checked participation.
///
/// The exact block shape comes from the root kernel's launch contract.
/// `uniform` is `(memory_pointer, scalar_controls...)`; `payload` contains
/// lane-local const or mutable payload pointers matching the provider ABI.
/// The compiler checks all call contexts before reconstructing the provider's
/// ordinary scalar/pointer ABI.
///
/// # Safety
///
/// The caller supplies the correct provider recipe, live aligned memory and
/// payload pointers, sufficient capacity, and the provider's initialization and
/// aliasing requirements. Group checks do not prove bounds or exclusive memory
/// ownership. Scratch aliases the kernel's dynamic shared base; the caller must
/// synchronize before conflicting scratch or output reuse. Completion never
/// follows from ending a Rust borrow or running a destructor.
#[inline(never)]
pub unsafe fn cuda_oxide_library_v1_246e25db_checked_block_call<
    T: Copy,
    Registration: Fn() -> &'static str + Copy,
    UniformArgs: Copy,
    PayloadArgs: Copy,
    const ITEMS_PER_THREAD: usize,
>(
    _registration: Registration,
    _uniform: UniformArgs,
    _payload: PayloadArgs,
) {
    // A backend which does not recognize this version must fail to link rather
    // than silently compiling a panic/trap as the collective implementation.
    unsafe extern "C" {
        fn cuda_oxide_checked_block_call_requires_supported_backend();
    }
    unsafe { cuda_oxide_checked_block_call_requires_supported_backend() }
}

pub use cuda_oxide_library_v1_246e25db_checked_block_call as checked_block_call_v1;

/// Invoke a scratch-free full-block operation with a typed memory witness.
///
/// The compiler accepts `&[T]` for reads, a reference to
/// `DisjointSlice<T, CollectiveTiles<C>>` for reads, and a mutable reference to
/// that carrier for writes. It tracks the original kernel allocation through
/// helpers and requires a full-block barrier before conflicting reuse.
/// The witness is not passed to the native function. `uniform` and `payload`
/// have the same roles as [`checked_block_call_v1`].
///
/// # Safety
///
/// The native pointer and accessed range must describe `memory`. The wrapper
/// must validate capacity, canonical group ownership, arithmetic, and bounds
/// before forming that pointer. Native memory effects must stay within that
/// range and the lane-local payload; `WRITE = false` forbids writes to `memory`.
/// The recipe, payload initialization and native ABI must be correct. Compiler
/// provenance and participation checks do not verify these obligations of the
/// unsafe wrapper implementation.
#[inline(never)]
pub unsafe fn cuda_oxide_library_v1_246e25db_checked_block_resource_call<
    T: Copy,
    Registration: Fn() -> &'static str + Copy,
    Memory,
    UniformArgs: Copy,
    PayloadArgs: Copy,
    const ITEMS: usize,
    const WRITE: bool,
>(
    _registration: Registration,
    _memory: Memory,
    _uniform: UniformArgs,
    _payload: PayloadArgs,
) {
    unsafe extern "C" {
        fn cuda_oxide_checked_block_resource_call_requires_supported_backend();
    }
    unsafe { cuda_oxide_checked_block_resource_call_requires_supported_backend() }
}

pub use cuda_oxide_library_v1_246e25db_checked_block_resource_call as checked_block_resource_call_v1;
