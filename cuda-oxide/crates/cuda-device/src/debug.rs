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

/// Conversion table for `gpu_printf!` arguments.
///
/// A placeholder without a type character (`{}`, `{:8}`, `{:08}`, `{:.2}`)
/// takes its C conversion from this trait: [`FORMAT_CHAR`](Self::FORMAT_CHAR)
/// is `d`, `u`, `f` or `p`, [`IS_64BIT`](Self::IS_64BIT) adds the `ll` length
/// modifier to an integer conversion, and [`IS_FLOAT`](Self::IS_FLOAT) selects
/// `%f`. [`promote`](Self::promote) applies the C vararg promotion rules before
/// the value is packed:
/// - `i8`, `i16` → `i32`
/// - `u8`, `u16` → `u32`
/// - `f32` → `f64`
/// - `bool` → `i32`
/// - 64-bit integers and raw pointers stay as they are
///
/// `char`, fieldless enums, `i128` and `u128` have no impl. Print a `char`
/// with `{:c}` and cast an enum with `as i32`.
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

// =============================================================================
// Compile-time format string assembly
// =============================================================================

/// One conversion of a `gpu_printf!` format string.
///
/// The macro hands a list of these to [`__gpu_printf_format`] together with
/// the literal text around them. Public only because the macro expands in
/// downstream crates.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub enum __GpuPrintfSpec {
    /// A complete C conversion, `%` included, chosen by the macro from an
    /// explicit type character such as `{:x}` or `{:.2f}`.
    Fixed(&'static [u8]),
    /// A conversion whose type character comes from the argument's
    /// [`GpuPrintfArg`] constants. `prefix` holds the flags, width and
    /// precision that sit between the `%` and the type character.
    Typed {
        prefix: &'static [u8],
        format_char: char,
        is_64bit: bool,
        is_float: bool,
    },
}

/// Assembles the NUL-terminated C format string of one `gpu_printf!` call.
///
/// `parts` is the literal text around the conversions, with `%` already
/// doubled, and `specs` the conversions in order, so `parts.len()` is
/// `specs.len() + 1`. A typed conversion gets `ll` for a 64-bit integer that
/// is not a pointer, and `f` for a floating-point argument. `N` is an upper
/// bound the macro computes from the same data; the bytes after the NUL stay
/// zero.
///
/// The macro evaluates this at compile time, so each `assert!` and `panic!`
/// here is a compile error at the call site. Public only because the macro
/// expands in downstream crates.
#[doc(hidden)]
pub const fn __gpu_printf_format<const N: usize>(
    parts: &[&[u8]],
    specs: &[__GpuPrintfSpec],
) -> [u8; N] {
    assert!(
        parts.len() == specs.len() + 1,
        "gpu_printf: each conversion needs the literal text before and after it"
    );

    let mut out = [0u8; N];
    let mut at = 0;
    let mut index = 0;
    while index < parts.len() {
        at = gpu_printf_copy(&mut out, at, parts[index]);
        if index < specs.len() {
            at = match specs[index] {
                __GpuPrintfSpec::Fixed(bytes) => gpu_printf_copy(&mut out, at, bytes),
                __GpuPrintfSpec::Typed {
                    prefix,
                    format_char,
                    is_64bit,
                    is_float,
                } => gpu_printf_typed(&mut out, at, prefix, format_char, is_64bit, is_float),
            };
        }
        index += 1;
    }
    assert!(
        at < N,
        "gpu_printf: the format buffer is too small for its format string"
    );
    out[at] = 0;
    out
}

/// Writes one byte of the format string.
const fn gpu_printf_put(out: &mut [u8], at: usize, byte: u8) -> usize {
    assert!(
        at < out.len(),
        "gpu_printf: the format buffer is too small for its format string"
    );
    out[at] = byte;
    at + 1
}

/// Copies `bytes` into the format string.
const fn gpu_printf_copy(out: &mut [u8], mut at: usize, bytes: &[u8]) -> usize {
    let mut index = 0;
    while index < bytes.len() {
        at = gpu_printf_put(out, at, bytes[index]);
        index += 1;
    }
    at
}

/// Writes a conversion whose type character comes from [`GpuPrintfArg`].
///
/// C defines the `+` and space flags for signed conversions only; on `%u` or
/// `%p` the output is undefined, so the combination is rejected here.
const fn gpu_printf_typed(
    out: &mut [u8],
    mut at: usize,
    prefix: &[u8],
    format_char: char,
    is_64bit: bool,
    is_float: bool,
) -> usize {
    if !is_float && format_char != 'd' && gpu_printf_has_sign_flag(prefix) {
        panic!(
            "gpu_printf: the + and space flags need a signed integer or floating-point argument"
        );
    }
    at = gpu_printf_put(out, at, b'%');
    at = gpu_printf_copy(out, at, prefix);
    if !is_float && is_64bit && format_char != 'p' {
        at = gpu_printf_copy(out, at, b"ll");
    }
    let type_char = if is_float { b'f' } else { format_char as u8 };
    gpu_printf_put(out, at, type_char)
}

/// Whether a conversion prefix carries the `+` or the space flag.
const fn gpu_printf_has_sign_flag(prefix: &[u8]) -> bool {
    let mut index = 0;
    while index < prefix.len() {
        if prefix[index] == b'+' || prefix[index] == b' ' {
            return true;
        }
        index += 1;
    }
    false
}

// Re-export the gpu_printf macro from cuda-macros
pub use cuda_macros::gpu_printf;

#[cfg(test)]
mod tests {
    use super::{__GpuPrintfSpec, __gpu_printf_format, gpu_printf};

    fn typed(
        prefix: &'static [u8],
        format_char: char,
        is_64bit: bool,
        is_float: bool,
    ) -> __GpuPrintfSpec {
        __GpuPrintfSpec::Typed {
            prefix,
            format_char,
            is_64bit,
            is_float,
        }
    }

    /// The bytes before the NUL, as text.
    fn text(buffer: &[u8]) -> &str {
        let end = buffer
            .iter()
            .position(|&byte| byte == 0)
            .expect("format strings are NUL-terminated");
        core::str::from_utf8(&buffer[..end]).expect("format strings are ASCII")
    }

    #[test]
    fn typed_conversions_follow_the_trait_constants() {
        let cases: [(&[u8], char, bool, bool, &str); 8] = [
            (b"", 'd', false, false, "%d"),
            (b"", 'd', true, false, "%lld"),
            (b"", 'u', false, false, "%u"),
            (b"", 'u', true, false, "%llu"),
            (b"", 'f', false, true, "%f"),
            (b"", 'p', true, false, "%p"),
            (b".2", 'f', false, true, "%.2f"),
            (b".3", 'd', false, false, "%.3d"),
        ];
        for (prefix, format_char, is_64bit, is_float, expected) in cases {
            let buffer = __gpu_printf_format::<8>(
                &[b"", b""],
                &[typed(prefix, format_char, is_64bit, is_float)],
            );
            assert_eq!(text(&buffer), expected);
        }
    }

    #[test]
    fn prefixes_sit_between_the_percent_and_the_type_character() {
        let buffer = __gpu_printf_format::<32>(
            &[b"[", b"] [", b"] [", b"]\n"],
            &[
                typed(b"-8", 'd', false, false),
                typed(b"08", 'u', true, false),
                typed(b"*", 'd', true, false),
            ],
        );
        assert_eq!(text(&buffer), "[%-8d] [%08llu] [%*lld]\n");
    }

    #[test]
    fn fixed_conversions_and_literal_text_are_copied_verbatim() {
        let buffer = __gpu_printf_format::<32>(
            &[b"100%% done: ", b" ", b"\n"],
            &[
                __GpuPrintfSpec::Fixed(b"%#llx"),
                typed(b"", 'f', false, true),
            ],
        );
        assert_eq!(text(&buffer), "100%% done: %#llx %f\n");
    }

    #[test]
    fn the_buffer_may_be_filled_to_the_last_byte() {
        let buffer = __gpu_printf_format::<5>(&[b"", b""], &[typed(b"", 'd', true, false)]);
        assert_eq!(buffer, *b"%lld\0");
    }

    #[test]
    fn bytes_after_the_nul_stay_zero() {
        // The macro leaves room for `ll`; a 32-bit argument does not use it.
        let buffer = __gpu_printf_format::<5>(&[b"", b""], &[typed(b"", 'd', false, false)]);
        assert_eq!(buffer, *b"%d\0\0\0");
    }

    #[test]
    fn builds_in_a_const_context() {
        const FORMAT: [u8; 8] = __gpu_printf_format::<8>(
            &[b"v=", b"\n"],
            &[__GpuPrintfSpec::Typed {
                prefix: b"",
                format_char: 'u',
                is_64bit: true,
                is_float: false,
            }],
        );
        assert_eq!(text(&FORMAT), "v=%llu\n");
    }

    #[test]
    fn sign_flags_are_kept_on_signed_and_float_arguments() {
        let buffer = __gpu_printf_format::<16>(
            &[b"", b" ", b" ", b""],
            &[
                typed(b"+", 'd', true, false),
                typed(b" ", 'd', false, false),
                typed(b"+", 'f', false, true),
            ],
        );
        assert_eq!(text(&buffer), "%+lld % d %+f");
    }

    #[test]
    #[should_panic(expected = "the + and space flags need a signed integer or floating-point")]
    fn the_plus_flag_on_an_unsigned_argument_is_rejected() {
        let _ = __gpu_printf_format::<8>(&[b"", b""], &[typed(b"+", 'u', false, false)]);
    }

    #[test]
    #[should_panic(expected = "the + and space flags need a signed integer or floating-point")]
    fn the_space_flag_on_a_pointer_argument_is_rejected() {
        let _ = __gpu_printf_format::<8>(&[b"", b""], &[typed(b" ", 'p', true, false)]);
    }

    #[test]
    #[should_panic(expected = "the format buffer is too small")]
    fn a_buffer_too_small_for_the_text_is_rejected() {
        let _ = __gpu_printf_format::<4>(&[b"", b""], &[typed(b"", 'd', true, false)]);
    }

    #[test]
    #[should_panic(expected = "the format buffer is too small")]
    fn a_buffer_with_no_room_for_the_nul_is_rejected() {
        let _ = __gpu_printf_format::<2>(&[b"", b""], &[typed(b"", 'd', false, false)]);
    }

    #[test]
    #[should_panic(expected = "each conversion needs the literal text before and after it")]
    fn parts_and_conversions_must_interleave() {
        let _ = __gpu_printf_format::<8>(&[b""], &[typed(b"", 'd', false, false)]);
    }

    /// This call is reached, so the expansion is monomorphized and the inline
    /// const that builds its format string is evaluated while the test binary
    /// compiles. A size mismatch between the macro and the builder, or a
    /// missing `GpuPrintfArg` impl, fails this test at build time.
    /// `__gpu_vprintf` then panics because there is no kernel around it.
    #[test]
    #[should_panic(expected = "__gpu_vprintf called outside CUDA kernel context")]
    fn default_placeholders_expand_and_build_their_format_on_the_host() {
        let value: u64 = u64::MAX;
        let ratio: f32 = 1.25;
        let pointer: *const f32 = &ratio;
        gpu_printf!(
            "{} {:.2} {:08} {} {} {:x} 100% [{:*}]\n",
            value,
            ratio,
            42,
            true,
            pointer,
            255u32,
            6,
            -7i64
        );
    }
}
