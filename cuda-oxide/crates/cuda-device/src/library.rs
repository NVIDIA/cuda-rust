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
/// content changes preserving timestamps are not detected by this macro.
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
/// reusing it. This version does not permit mixing unplanned native dynamic
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
