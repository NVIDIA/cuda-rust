/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU Debug and Profiling Intrinsics
//!
//! This module provides intrinsics for debugging and profiling GPU kernels:
//!
//! | Function           | Description                   | CUDA C++ Equivalent |
//! |--------------------|-------------------------------|---------------------|
//! | [`clock()`]        | Read 32-bit GPU clock counter | `clock()`           |
//! | [`clock64()`]      | Read 64-bit GPU clock counter | `clock64()`         |
//! | [`globaltimer()`]  | Read GPU global timer         | `%globaltimer`      |
//! | [`trap()`]         | Abort kernel execution        | `__trap()`          |
//! | [`breakpoint()`]   | Insert cuda-gdb breakpoint    | `__brkpt()`         |
//! | [`prof_trigger()`] | Signal NVIDIA profiler        | `__prof_trigger(N)` |
//!
//! # Example: Micro-benchmarking
//!
//! ```rust,ignore
//! use cuda_device::debug;
//!
//! let start = debug::clock64();
//! // ... computation to measure ...
//! let end = debug::clock64();
//! let cycles = end - start;
//! ```
//!
//! # Example: Runtime Assertion
//!
//! ```rust,ignore
//! use cuda_device::debug;
//!
//! if value < 0 {
//!     debug::trap();  // Kernel aborts
//! }
//! ```

include!("generated/debug_sreg.rs");
include!("generated/debug_control.rs");

// =============================================================================
// Assertion Macro
// =============================================================================

/// GPU-side assertion macro.
///
/// Checks a condition at runtime and aborts the kernel if it fails.
/// This is equivalent to `assert!()` but works on GPU kernels.
///
/// # Usage
///
/// ```rust,ignore
/// use cuda_device::gpu_assert;
///
/// // Simple assertion
/// gpu_assert!(x >= 0);
///
/// // With a custom string-literal message
/// gpu_assert!(idx < len, "Index out of bounds");
/// ```
///
/// # Behavior
///
/// When the condition is false:
/// - The no-message form aborts kernel execution via [`trap()`]
/// - The message form reports the message and call-site metadata via CUDA's
///   device-side `__assertfail` system call
/// - The CUDA driver reports an error to the host
/// - Other threads may continue briefly before the error propagates
///
/// # Notes
///
/// - Use sparingly in performance-critical code
/// - Custom messages must be string literals
/// - For debugging, consider using [`breakpoint()`] instead
#[macro_export]
macro_rules! gpu_assert {
    ($cond:expr) => {
        if !$cond {
            $crate::debug::trap();
        }
    };
    ($cond:expr, $msg:literal) => {{
        if !$cond {
            const __GPU_ASSERT_MESSAGE_TEXT: &str = $msg;
            const __GPU_ASSERT_MESSAGE: [u8; __GPU_ASSERT_MESSAGE_TEXT.len() + 1] =
                $crate::debug::__gpu_assert_c_string::<{ __GPU_ASSERT_MESSAGE_TEXT.len() + 1 }>(
                    __GPU_ASSERT_MESSAGE_TEXT,
                );

            const __GPU_ASSERT_FILE_TEXT: &str = file!();
            const __GPU_ASSERT_FILE: [u8; __GPU_ASSERT_FILE_TEXT.len() + 1] =
                $crate::debug::__gpu_assert_c_string::<{ __GPU_ASSERT_FILE_TEXT.len() + 1 }>(
                    __GPU_ASSERT_FILE_TEXT,
                );

            const __GPU_ASSERT_FUNCTION_TEXT: &str = module_path!();
            const __GPU_ASSERT_FUNCTION: [u8; __GPU_ASSERT_FUNCTION_TEXT.len() + 1] =
                $crate::debug::__gpu_assert_c_string::<{ __GPU_ASSERT_FUNCTION_TEXT.len() + 1 }>(
                    __GPU_ASSERT_FUNCTION_TEXT,
                );

            $crate::debug::__gpu_assertfail(
                __GPU_ASSERT_MESSAGE.as_ptr(),
                __GPU_ASSERT_FILE.as_ptr(),
                line!(),
                __GPU_ASSERT_FUNCTION.as_ptr(),
                1,
            );
        }
    }};
    ($cond:expr, $msg:expr) => {
        compile_error!("gpu_assert! messages must be string literals")
    };
}

/// Builds a null-terminated byte array for CUDA assertion metadata.
///
/// This helper is public only because [`gpu_assert!`] expands in downstream
/// crates. It is evaluated at compile time by the message form of the macro.
#[doc(hidden)]
pub const fn __gpu_assert_c_string<const N: usize>(value: &str) -> [u8; N] {
    let bytes = value.as_bytes();
    assert!(
        N == bytes.len() + 1,
        "GPU assertion C string has an invalid length"
    );

    let mut output = [0; N];
    let mut index = 0;
    while index < bytes.len() {
        assert!(
            bytes[index] != 0,
            "gpu_assert! strings must not contain NUL bytes"
        );
        output[index] = bytes[index];
        index += 1;
    }
    output
}

/// Internal CUDA assertion-failure wrapper.
///
/// This function is recognized by the cuda-oxide compiler and lowered to
/// CUDA's device-side `__assertfail(message, file, line, function, char_size)`
/// system call. Do not call directly.
///
/// # Safety
///
/// This function only works within CUDA kernel context. Calling it from host
/// code will panic.
#[doc(hidden)]
#[inline(never)]
pub fn __gpu_assertfail(
    _message: *const u8,
    _file: *const u8,
    _line: u32,
    _function: *const u8,
    _char_size: usize,
) {
    unreachable!("__gpu_assertfail called outside CUDA kernel context")
}

// =============================================================================
// Printf Support
// =============================================================================

/// Internal vprintf wrapper for GPU printf support.
///
/// This function is recognized by the cuda-oxide compiler and replaced with
/// an actual `vprintf` call in the generated PTX. Do not call directly.
///
/// # Arguments
///
/// * `format` - Pointer to null-terminated C format string (in global memory)
/// * `args` - Pointer to packed argument buffer (following C vararg ABI)
///
/// # Returns
///
/// Number of arguments on success, negative value on error.
/// Note: Unlike standard C printf which returns character count, CUDA's vprintf
/// returns the argument count because the GPU only marshals args to a buffer -
/// the host does the actual formatting later.
///
/// # Safety
///
/// This function only works within CUDA kernel context. The compiler replaces
/// calls with actual vprintf instructions. Calling from host code will panic.
#[doc(hidden)]
#[inline(never)]
pub fn __gpu_vprintf(_format: *const u8, _args: *const u8) -> i32 {
    unreachable!("__gpu_vprintf called outside CUDA kernel context")
}

/// Trait for GPU printf argument promotion.
///
/// Implements C vararg promotion rules:
/// - `i8`, `i16` → `i32`
/// - `u8`, `u16` → `u32`
/// - `f32` → `f64`
/// - `bool` → `i32`
/// - 64-bit types stay as-is
pub trait GpuPrintfArg {
    /// The promoted type for C vararg ABI
    type Promoted: Copy;

    /// C format specifier character for this type
    const FORMAT_CHAR: char;

    /// Whether this is a 64-bit type (needs `ll` modifier)
    const IS_64BIT: bool;

    /// Whether this is a floating point type
    const IS_FLOAT: bool;

    /// Promote the value to the vararg type
    fn promote(self) -> Self::Promoted;
}

// Signed integers
impl GpuPrintfArg for i8 {
    type Promoted = i32;
    const FORMAT_CHAR: char = 'd';
    const IS_64BIT: bool = false;
    const IS_FLOAT: bool = false;
    fn promote(self) -> i32 {
        self as i32
    }
}

impl GpuPrintfArg for i16 {
    type Promoted = i32;
    const FORMAT_CHAR: char = 'd';
    const IS_64BIT: bool = false;
    const IS_FLOAT: bool = false;
    fn promote(self) -> i32 {
        self as i32
    }
}

impl GpuPrintfArg for i32 {
    type Promoted = i32;
    const FORMAT_CHAR: char = 'd';
    const IS_64BIT: bool = false;
    const IS_FLOAT: bool = false;
    fn promote(self) -> i32 {
        self
    }
}

impl GpuPrintfArg for i64 {
    type Promoted = i64;
    const FORMAT_CHAR: char = 'd';
    const IS_64BIT: bool = true;
    const IS_FLOAT: bool = false;
    fn promote(self) -> i64 {
        self
    }
}

impl GpuPrintfArg for isize {
    type Promoted = i64;
    const FORMAT_CHAR: char = 'd';
    const IS_64BIT: bool = true;
    const IS_FLOAT: bool = false;
    fn promote(self) -> i64 {
        self as i64
    }
}

// Unsigned integers
impl GpuPrintfArg for u8 {
    type Promoted = u32;
    const FORMAT_CHAR: char = 'u';
    const IS_64BIT: bool = false;
    const IS_FLOAT: bool = false;
    fn promote(self) -> u32 {
        self as u32
    }
}

impl GpuPrintfArg for u16 {
    type Promoted = u32;
    const FORMAT_CHAR: char = 'u';
    const IS_64BIT: bool = false;
    const IS_FLOAT: bool = false;
    fn promote(self) -> u32 {
        self as u32
    }
}

impl GpuPrintfArg for u32 {
    type Promoted = u32;
    const FORMAT_CHAR: char = 'u';
    const IS_64BIT: bool = false;
    const IS_FLOAT: bool = false;
    fn promote(self) -> u32 {
        self
    }
}

impl GpuPrintfArg for u64 {
    type Promoted = u64;
    const FORMAT_CHAR: char = 'u';
    const IS_64BIT: bool = true;
    const IS_FLOAT: bool = false;
    fn promote(self) -> u64 {
        self
    }
}

impl GpuPrintfArg for usize {
    type Promoted = u64;
    const FORMAT_CHAR: char = 'u';
    const IS_64BIT: bool = true;
    const IS_FLOAT: bool = false;
    fn promote(self) -> u64 {
        self as u64
    }
}

// Floating point
impl GpuPrintfArg for f32 {
    type Promoted = f64;
    const FORMAT_CHAR: char = 'f';
    const IS_64BIT: bool = false;
    const IS_FLOAT: bool = true;
    fn promote(self) -> f64 {
        self as f64
    }
}

impl GpuPrintfArg for f64 {
    type Promoted = f64;
    const FORMAT_CHAR: char = 'f';
    const IS_64BIT: bool = false;
    const IS_FLOAT: bool = true;
    fn promote(self) -> f64 {
        self
    }
}

// Boolean
impl GpuPrintfArg for bool {
    type Promoted = i32;
    const FORMAT_CHAR: char = 'd';
    const IS_64BIT: bool = false;
    const IS_FLOAT: bool = false;
    fn promote(self) -> i32 {
        self as i32
    }
}

// Pointers
impl<T> GpuPrintfArg for *const T {
    type Promoted = u64;
    const FORMAT_CHAR: char = 'p';
    const IS_64BIT: bool = true;
    const IS_FLOAT: bool = false;
    fn promote(self) -> u64 {
        self as u64
    }
}

impl<T> GpuPrintfArg for *mut T {
    type Promoted = u64;
    const FORMAT_CHAR: char = 'p';
    const IS_64BIT: bool = true;
    const IS_FLOAT: bool = false;
    fn promote(self) -> u64 {
        self as u64
    }
}

// Re-export the gpu_printf macro from cuda-macros
pub use cuda_macros::gpu_printf;
