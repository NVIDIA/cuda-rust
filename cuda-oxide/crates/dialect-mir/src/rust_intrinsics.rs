/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared names for internal Rust compiler intrinsic placeholder calls.
//!
//! The importer emits these names as ordinary `mir.call` callees when it sees a
//! rustc intrinsic that needs target-specific lowering. The MIR-to-LLVM pass
//! recognizes the same names and replaces them with LLVM or CUDA libdevice calls.
//! Keep the prefix centralized here so the planned magic-hash prefix change only
//! needs one edit.

/// Build an internal Rust intrinsic placeholder name from its stable suffix.
macro_rules! placeholder {
    ($suffix:literal) => {
        concat!("__cuda_oxide_rust_intrinsic_", $suffix)
    };
}

/// Prefix used for cuda-oxide internal Rust intrinsic placeholder calls.
pub const PLACEHOLDER_PREFIX: &str = placeholder!("");

/// Placeholder call used for `core::intrinsics::rotate_left`.
pub const CALLEE_ROTATE_LEFT: &str = placeholder!("rotate_left");
/// Placeholder call used for `core::intrinsics::rotate_right`.
pub const CALLEE_ROTATE_RIGHT: &str = placeholder!("rotate_right");
/// Placeholder call used for `core::intrinsics::ctpop`.
pub const CALLEE_CTPOP: &str = placeholder!("ctpop");
/// Placeholder call used for `core::intrinsics::ctlz`.
pub const CALLEE_CTLZ: &str = placeholder!("ctlz");
/// Placeholder call used for `core::intrinsics::ctlz_nonzero`.
pub const CALLEE_CTLZ_NONZERO: &str = placeholder!("ctlz_nonzero");
/// Placeholder call used for `core::intrinsics::cttz`.
pub const CALLEE_CTTZ: &str = placeholder!("cttz");
/// Placeholder call used for `core::intrinsics::cttz_nonzero`.
pub const CALLEE_CTTZ_NONZERO: &str = placeholder!("cttz_nonzero");
/// Placeholder call used for `core::intrinsics::bswap`.
pub const CALLEE_BSWAP: &str = placeholder!("bswap");
/// Placeholder call used for `core::intrinsics::bitreverse`.
pub const CALLEE_BITREVERSE: &str = placeholder!("bitreverse");

/// Placeholder call used for `core::intrinsics::saturating_add`.
pub const CALLEE_SATURATING_ADD: &str = placeholder!("saturating_add");
/// Placeholder call used for `core::intrinsics::saturating_sub`.
pub const CALLEE_SATURATING_SUB: &str = placeholder!("saturating_sub");

/// Placeholder call used for `core::intrinsics::exact_div`.
///
/// Division where the caller guarantees the divisor is non-zero and divides the
/// dividend exactly. Backs `slice::as_chunks` and friends, which compute their
/// chunk count with `exact_div(self.len(), N)`.
pub const CALLEE_EXACT_DIV: &str = placeholder!("exact_div");

/// Placeholder call used for `core::intrinsics::carrying_mul_add`.
/// Backs the bigint helper methods `carrying_mul_add`, `carrying_mul`,
/// and `widening_mul` on integer types.
pub const CALLEE_CARRYING_MUL_ADD: &str = placeholder!("carrying_mul_add");

/// Placeholder call used for `core::intrinsics::sqrtf32`.
pub const CALLEE_SQRT_F32: &str = placeholder!("sqrtf32");
/// Placeholder call used for `core::intrinsics::sqrtf64`.
pub const CALLEE_SQRT_F64: &str = placeholder!("sqrtf64");
/// Placeholder call used for `core::intrinsics::powif32`.
pub const CALLEE_POWI_F32: &str = placeholder!("powif32");
/// Placeholder call used for `core::intrinsics::powif64`.
pub const CALLEE_POWI_F64: &str = placeholder!("powif64");
/// Placeholder call used for `core::intrinsics::sinf32`.
pub const CALLEE_SIN_F32: &str = placeholder!("sinf32");
/// Placeholder call used for `core::intrinsics::sinf64`.
pub const CALLEE_SIN_F64: &str = placeholder!("sinf64");
/// Placeholder call used for `core::intrinsics::cosf32`.
pub const CALLEE_COS_F32: &str = placeholder!("cosf32");
/// Placeholder call used for `core::intrinsics::cosf64`.
pub const CALLEE_COS_F64: &str = placeholder!("cosf64");
/// Placeholder call used for `core::intrinsics::tanf32`.
pub const CALLEE_TAN_F32: &str = placeholder!("tanf32");
/// Placeholder call used for `core::intrinsics::tanf64`.
pub const CALLEE_TAN_F64: &str = placeholder!("tanf64");
/// Placeholder call used for `core::intrinsics::powf32`.
pub const CALLEE_POWF_F32: &str = placeholder!("powf32");
/// Placeholder call used for `core::intrinsics::powf64`.
pub const CALLEE_POWF_F64: &str = placeholder!("powf64");
/// Placeholder call used for `core::intrinsics::expf32`.
pub const CALLEE_EXP_F32: &str = placeholder!("expf32");
/// Placeholder call used for `core::intrinsics::expf64`.
pub const CALLEE_EXP_F64: &str = placeholder!("expf64");
/// Placeholder call used for `core::intrinsics::exp2f32`.
pub const CALLEE_EXP2_F32: &str = placeholder!("exp2f32");
/// Placeholder call used for `core::intrinsics::exp2f64`.
pub const CALLEE_EXP2_F64: &str = placeholder!("exp2f64");
/// Placeholder call used for `core::intrinsics::logf32`.
pub const CALLEE_LOG_F32: &str = placeholder!("logf32");
/// Placeholder call used for `core::intrinsics::logf64`.
pub const CALLEE_LOG_F64: &str = placeholder!("logf64");
/// Placeholder call used for `core::intrinsics::log2f32`.
pub const CALLEE_LOG2_F32: &str = placeholder!("log2f32");
/// Placeholder call used for `core::intrinsics::log2f64`.
pub const CALLEE_LOG2_F64: &str = placeholder!("log2f64");
/// Placeholder call used for `core::intrinsics::log10f32`.
pub const CALLEE_LOG10_F32: &str = placeholder!("log10f32");
/// Placeholder call used for `core::intrinsics::log10f64`.
pub const CALLEE_LOG10_F64: &str = placeholder!("log10f64");
/// Placeholder call used for `core::intrinsics::fmaf32`.
pub const CALLEE_FMA_F32: &str = placeholder!("fmaf32");
/// Placeholder call used for `core::intrinsics::fmaf64`.
pub const CALLEE_FMA_F64: &str = placeholder!("fmaf64");
/// Placeholder call used for `core::intrinsics::fmuladdf32`.
pub const CALLEE_FMULADD_F32: &str = placeholder!("fmuladdf32");
/// Placeholder call used for `core::intrinsics::fmuladdf64`.
pub const CALLEE_FMULADD_F64: &str = placeholder!("fmuladdf64");
/// Placeholder call used for `core::intrinsics::floorf32`.
pub const CALLEE_FLOOR_F32: &str = placeholder!("floorf32");
/// Placeholder call used for `core::intrinsics::floorf64`.
pub const CALLEE_FLOOR_F64: &str = placeholder!("floorf64");
/// Placeholder call used for `core::intrinsics::ceilf32`.
pub const CALLEE_CEIL_F32: &str = placeholder!("ceilf32");
/// Placeholder call used for `core::intrinsics::ceilf64`.
pub const CALLEE_CEIL_F64: &str = placeholder!("ceilf64");
/// Placeholder call used for `core::intrinsics::truncf32`.
pub const CALLEE_TRUNC_F32: &str = placeholder!("truncf32");
/// Placeholder call used for `core::intrinsics::truncf64`.
pub const CALLEE_TRUNC_F64: &str = placeholder!("truncf64");
/// Placeholder call used for `core::intrinsics::roundf32`.
pub const CALLEE_ROUND_F32: &str = placeholder!("roundf32");
/// Placeholder call used for `core::intrinsics::roundf64`.
pub const CALLEE_ROUND_F64: &str = placeholder!("roundf64");
/// Placeholder call used for `core::intrinsics::round_ties_even_f32`.
pub const CALLEE_ROUNDEVEN_F32: &str = placeholder!("round_ties_even_f32");
/// Placeholder call used for `core::intrinsics::round_ties_even_f64`.
pub const CALLEE_ROUNDEVEN_F64: &str = placeholder!("round_ties_even_f64");
/// Placeholder call used for generic `core::intrinsics::fabs`.
pub const CALLEE_FABS: &str = placeholder!("fabs");
/// Placeholder call used for `core::intrinsics::copysignf32`.
pub const CALLEE_COPYSIGN_F32: &str = placeholder!("copysignf32");
/// Placeholder call used for `core::intrinsics::copysignf64`.
pub const CALLEE_COPYSIGN_F64: &str = placeholder!("copysignf64");
/// Placeholder call used for `core::intrinsics::maximum_number_nsz_f32`
/// (the intrinsic backing `f32::max`).
pub const CALLEE_MAXNUM_NSZ_F32: &str = placeholder!("maximum_number_nsz_f32");
/// Placeholder call used for `core::intrinsics::maximum_number_nsz_f64`
/// (the intrinsic backing `f64::max`).
pub const CALLEE_MAXNUM_NSZ_F64: &str = placeholder!("maximum_number_nsz_f64");
/// Placeholder call used for `core::intrinsics::minimum_number_nsz_f32`
/// (the intrinsic backing `f32::min`).
pub const CALLEE_MINNUM_NSZ_F32: &str = placeholder!("minimum_number_nsz_f32");
/// Placeholder call used for `core::intrinsics::minimum_number_nsz_f64`
/// (the intrinsic backing `f64::min`).
pub const CALLEE_MINNUM_NSZ_F64: &str = placeholder!("minimum_number_nsz_f64");
/// Placeholder call used for `f32::asin` / `std::sys::cmath::asinf`.
pub const CALLEE_ASIN_F32: &str = placeholder!("asinf32");
/// Placeholder call used for `f64::asin` / `std::sys::cmath::asin`.
pub const CALLEE_ASIN_F64: &str = placeholder!("asinf64");
/// Placeholder call used for `f32::acos` / `std::sys::cmath::acosf`.
pub const CALLEE_ACOS_F32: &str = placeholder!("acosf32");
/// Placeholder call used for `f64::acos` / `std::sys::cmath::acos`.
pub const CALLEE_ACOS_F64: &str = placeholder!("acosf64");
/// Placeholder call used for `f32::atan2` / `std::sys::cmath::atan2f`.
pub const CALLEE_ATAN2_F32: &str = placeholder!("atan2f32");
/// Placeholder call used for `f64::atan2` / `std::sys::cmath::atan2`.
pub const CALLEE_ATAN2_F64: &str = placeholder!("atan2f64");
/// Placeholder call used for `f32::atan` / `std::sys::cmath::atanf`.
pub const CALLEE_ATAN_F32: &str = placeholder!("atanf32");
/// Placeholder call used for `f64::atan` / `std::sys::cmath::atan`.
pub const CALLEE_ATAN_F64: &str = placeholder!("atanf64");
/// Placeholder call used for `f32::cbrt` / `std::sys::cmath::cbrtf`.
pub const CALLEE_CBRT_F32: &str = placeholder!("cbrtf32");
/// Placeholder call used for `f64::cbrt` / `std::sys::cmath::cbrt`.
pub const CALLEE_CBRT_F64: &str = placeholder!("cbrtf64");

/// Placeholder call used for `f32::sinh` / `std::sys::cmath::sinhf`.
pub const CALLEE_SINH_F32: &str = placeholder!("sinhf32");
/// Placeholder call used for `f64::sinh` / `std::sys::cmath::sinh`.
pub const CALLEE_SINH_F64: &str = placeholder!("sinhf64");
/// Placeholder call used for `f32::cosh` / `std::sys::cmath::coshf`.
pub const CALLEE_COSH_F32: &str = placeholder!("coshf32");
/// Placeholder call used for `f64::cosh` / `std::sys::cmath::cosh`.
pub const CALLEE_COSH_F64: &str = placeholder!("coshf64");
/// Placeholder call used for `f32::tanh` / `std::sys::cmath::tanhf`.
pub const CALLEE_TANH_F32: &str = placeholder!("tanhf32");
/// Placeholder call used for `f64::tanh` / `std::sys::cmath::tanh`.
pub const CALLEE_TANH_F64: &str = placeholder!("tanhf64");
// The inverse hyperbolics were pure-Rust formulas in `std` (compositions of
// `ln`/`sqrt`/`ln_1p`). nightly-2026-08-28 (rustc e457a7b0d) reworked
// asinh/acosh into `std::sys::cmath::{asinh,acosh}{,f}` shims like the rest
// of this family, so they need placeholders of their own; atanh is still the
// pure-Rust `ln_1p` formula there and cmath declares no atanh, so its
// placeholders are defensive, for when std makes the same move.
/// Placeholder call used for `f32::asinh` / `std::sys::cmath::asinhf`.
pub const CALLEE_ASINH_F32: &str = placeholder!("asinhf32");
/// Placeholder call used for `f64::asinh` / `std::sys::cmath::asinh`.
pub const CALLEE_ASINH_F64: &str = placeholder!("asinhf64");
/// Placeholder call used for `f32::acosh` / `std::sys::cmath::acoshf`.
pub const CALLEE_ACOSH_F32: &str = placeholder!("acoshf32");
/// Placeholder call used for `f64::acosh` / `std::sys::cmath::acosh`.
pub const CALLEE_ACOSH_F64: &str = placeholder!("acoshf64");
/// Placeholder call used for `f32::atanh` / `std::sys::cmath::atanhf`.
pub const CALLEE_ATANH_F32: &str = placeholder!("atanhf32");
/// Placeholder call used for `f64::atanh` / `std::sys::cmath::atanh`.
pub const CALLEE_ATANH_F64: &str = placeholder!("atanhf64");
/// Placeholder call used for `f32::exp_m1` / `std::sys::cmath::expm1f`.
pub const CALLEE_EXPM1_F32: &str = placeholder!("expm1f32");
/// Placeholder call used for `f64::exp_m1` / `std::sys::cmath::expm1`.
pub const CALLEE_EXPM1_F64: &str = placeholder!("expm1f64");
/// Placeholder call used for `f32::ln_1p` / `std::sys::cmath::log1pf`.
pub const CALLEE_LOG1P_F32: &str = placeholder!("log1pf32");
/// Placeholder call used for `f64::ln_1p` / `std::sys::cmath::log1p`.
pub const CALLEE_LOG1P_F64: &str = placeholder!("log1pf64");
/// Placeholder call used for `f32::hypot` / `std::sys::cmath::hypotf` (binary).
pub const CALLEE_HYPOT_F32: &str = placeholder!("hypotf32");
/// Placeholder call used for `f64::hypot` / `std::sys::cmath::hypot` (binary).
pub const CALLEE_HYPOT_F64: &str = placeholder!("hypotf64");

/// Placeholder call used for `core::intrinsics::fadd_fast` (generic over float type).
///
/// Lowered to `llvm.fadd` with explicit `fast` fast-math flags. The `f*_fast` intrinsics
/// assume finite, non-NaN inputs; LLVM's fast-math flags express the same
/// preconditions, so the binop replaces the call directly.
pub const CALLEE_FADD_FAST: &str = placeholder!("fadd_fast");
/// Placeholder call used for `core::intrinsics::fsub_fast` (generic over float type).
pub const CALLEE_FSUB_FAST: &str = placeholder!("fsub_fast");
/// Placeholder call used for `core::intrinsics::fmul_fast` (generic over float type).
pub const CALLEE_FMUL_FAST: &str = placeholder!("fmul_fast");
/// Placeholder call used for `core::intrinsics::fdiv_fast` (generic over float type).
pub const CALLEE_FDIV_FAST: &str = placeholder!("fdiv_fast");
/// Placeholder call used for `core::intrinsics::frem_fast` (generic over float type).
pub const CALLEE_FREM_FAST: &str = placeholder!("frem_fast");

/// Placeholder call used for `core::intrinsics::select_unpredictable`.
///
/// Backs `core::hint::select_unpredictable`, which libcore reaches
/// pervasively from branchless helpers (slice sorting, `Ord` combinators).
/// Lowers to an LLVM `select`; the "unpredictable" branch-weight hint has
/// no device semantics and is dropped.
pub const CALLEE_SELECT_UNPREDICTABLE: &str = placeholder!("select_unpredictable");

/// Return whether `callee` is one of the exact internal Rust-intrinsic
/// placeholders understood by MIR lowering.
///
/// Do not replace this with a prefix check. The reserved prefix identifies
/// malformed or future placeholders too; accepting those as an external call
/// would let an unknown pseudo-call bypass its source-level signature rules.
pub fn is_known_placeholder(callee: &str) -> bool {
    is_libdevice_backed_placeholder(callee)
        || is_backend_dependent_libdevice_placeholder(callee)
        || matches!(
            callee,
            CALLEE_ROTATE_LEFT
                | CALLEE_ROTATE_RIGHT
                | CALLEE_CTPOP
                | CALLEE_CTLZ
                | CALLEE_CTLZ_NONZERO
                | CALLEE_CTTZ
                | CALLEE_CTTZ_NONZERO
                | CALLEE_BSWAP
                | CALLEE_BITREVERSE
                | CALLEE_SATURATING_ADD
                | CALLEE_SATURATING_SUB
                | CALLEE_EXACT_DIV
                | CALLEE_CARRYING_MUL_ADD
                | CALLEE_MAXNUM_NSZ_F32
                | CALLEE_MAXNUM_NSZ_F64
                | CALLEE_MINNUM_NSZ_F32
                | CALLEE_MINNUM_NSZ_F64
                | CALLEE_FADD_FAST
                | CALLEE_FSUB_FAST
                | CALLEE_FMUL_FAST
                | CALLEE_FDIV_FAST
                | CALLEE_FREM_FAST
                | CALLEE_SELECT_UNPREDICTABLE
        )
}

/// Return whether an internal placeholder lowers to a CUDA libdevice call
/// under every intrinsic backend.
///
/// This is deliberately an exact allow-list. Other placeholder families lower
/// to LLVM operations directly, including integer, saturating, bigint, and
/// `f*_fast` intrinsics, so matching the common placeholder prefix would select
/// the libNVVM backend for modules that do not need it.
///
/// The rounding placeholders (`floor`/`ceil`/`trunc`/`round`/`roundeven`)
/// and the sign placeholders (`fabs`/`copysign`) are intentionally NOT in
/// this list: on the LLVM NVPTX path they lower to the native
/// `llvm.floor.*`-family / `llvm.fabs.*` / `llvm.copysign.*` intrinsics and
/// need no libdevice at all. They fall back to libdevice only when the
/// pipeline emits NVVM IR; see
/// [`is_backend_dependent_libdevice_placeholder`].
///
/// The `max`/`min` placeholders (`maximum_number_nsz_*` /
/// `minimum_number_nsz_*`) are in NEITHER list: they lower to an ordered
/// compare/select expansion under every intrinsic backend, so they never
/// produce a `__nv_*` call. (`llvm.maxnum`/`llvm.minnum` are unusable
/// because under LLVM 21 they propagate signaling NaNs, contradicting
/// Rust's ignore-any-NaN contract; see #390.)
pub fn is_libdevice_backed_placeholder(callee: &str) -> bool {
    matches!(
        callee,
        CALLEE_SQRT_F32
            | CALLEE_SQRT_F64
            | CALLEE_POWI_F32
            | CALLEE_POWI_F64
            | CALLEE_SIN_F32
            | CALLEE_SIN_F64
            | CALLEE_COS_F32
            | CALLEE_COS_F64
            | CALLEE_TAN_F32
            | CALLEE_TAN_F64
            | CALLEE_POWF_F32
            | CALLEE_POWF_F64
            | CALLEE_EXP_F32
            | CALLEE_EXP_F64
            | CALLEE_EXP2_F32
            | CALLEE_EXP2_F64
            | CALLEE_LOG_F32
            | CALLEE_LOG_F64
            | CALLEE_LOG2_F32
            | CALLEE_LOG2_F64
            | CALLEE_LOG10_F32
            | CALLEE_LOG10_F64
            | CALLEE_FMA_F32
            | CALLEE_FMA_F64
            | CALLEE_FMULADD_F32
            | CALLEE_FMULADD_F64
            | CALLEE_ASIN_F32
            | CALLEE_ASIN_F64
            | CALLEE_ACOS_F32
            | CALLEE_ACOS_F64
            | CALLEE_ATAN2_F32
            | CALLEE_ATAN2_F64
            | CALLEE_ATAN_F32
            | CALLEE_ATAN_F64
            | CALLEE_CBRT_F32
            | CALLEE_CBRT_F64
            | CALLEE_SINH_F32
            | CALLEE_SINH_F64
            | CALLEE_COSH_F32
            | CALLEE_COSH_F64
            | CALLEE_TANH_F32
            | CALLEE_TANH_F64
            | CALLEE_ASINH_F32
            | CALLEE_ASINH_F64
            | CALLEE_ACOSH_F32
            | CALLEE_ACOSH_F64
            | CALLEE_ATANH_F32
            | CALLEE_ATANH_F64
            | CALLEE_EXPM1_F32
            | CALLEE_EXPM1_F64
            | CALLEE_LOG1P_F32
            | CALLEE_LOG1P_F64
            | CALLEE_HYPOT_F32
            | CALLEE_HYPOT_F64
    )
}

/// Return whether an internal placeholder lowers to a CUDA libdevice call
/// only under the libNVVM intrinsic backend.
///
/// The rounding placeholders lower to the native LLVM intrinsics
/// (`llvm.floor.*`, `llvm.ceil.*`, `llvm.trunc.*`, `llvm.round.*`,
/// `llvm.roundeven.*`) when the module is headed to LLVM's NVPTX backend, so
/// on that path they need no libdevice at all. The sign placeholders
/// (`fabs`/`copysign`) take the same route via `llvm.fabs.*` /
/// `llvm.copysign.*`, which the NVPTX backend selects to single PTX
/// `abs`/`copysign` instructions. When the pipeline emits NVVM IR instead,
/// the same placeholders fall back to `__nv_floorf`/`__nv_fabsf`/...
/// libdevice calls, because the legacy LLVM 7-based NVVM IR dialect predates
/// `llvm.roundeven.*` (added in LLVM 11) and admits only a small intrinsic
/// allow-list.
///
/// Like [`is_libdevice_backed_placeholder`], this is an exact allow-list, and
/// the two lists are mutually disjoint: a placeholder is either always
/// libdevice-backed, libdevice-backed only under libNVVM, or never
/// libdevice-backed.
pub fn is_backend_dependent_libdevice_placeholder(callee: &str) -> bool {
    matches!(
        callee,
        CALLEE_FLOOR_F32
            | CALLEE_FLOOR_F64
            | CALLEE_CEIL_F32
            | CALLEE_CEIL_F64
            | CALLEE_TRUNC_F32
            | CALLEE_TRUNC_F64
            | CALLEE_ROUND_F32
            | CALLEE_ROUND_F64
            | CALLEE_ROUNDEVEN_F32
            | CALLEE_ROUNDEVEN_F64
            | CALLEE_FABS
            | CALLEE_COPYSIGN_F32
            | CALLEE_COPYSIGN_F64
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every placeholder that must be classified as libdevice-backed under
    /// every intrinsic backend (transcendentals and fma).
    const ALWAYS_LIBDEVICE: &[&str] = &[
        CALLEE_SQRT_F32,
        CALLEE_SQRT_F64,
        CALLEE_POWI_F32,
        CALLEE_POWI_F64,
        CALLEE_SIN_F32,
        CALLEE_SIN_F64,
        CALLEE_COS_F32,
        CALLEE_COS_F64,
        CALLEE_TAN_F32,
        CALLEE_TAN_F64,
        CALLEE_POWF_F32,
        CALLEE_POWF_F64,
        CALLEE_EXP_F32,
        CALLEE_EXP_F64,
        CALLEE_EXP2_F32,
        CALLEE_EXP2_F64,
        CALLEE_LOG_F32,
        CALLEE_LOG_F64,
        CALLEE_LOG2_F32,
        CALLEE_LOG2_F64,
        CALLEE_LOG10_F32,
        CALLEE_LOG10_F64,
        CALLEE_FMA_F32,
        CALLEE_FMA_F64,
        CALLEE_FMULADD_F32,
        CALLEE_FMULADD_F64,
        CALLEE_ASIN_F32,
        CALLEE_ASIN_F64,
        CALLEE_ACOS_F32,
        CALLEE_ACOS_F64,
        CALLEE_ATAN2_F32,
        CALLEE_ATAN2_F64,
        CALLEE_ATAN_F32,
        CALLEE_ATAN_F64,
        CALLEE_CBRT_F32,
        CALLEE_CBRT_F64,
        CALLEE_SINH_F32,
        CALLEE_SINH_F64,
        CALLEE_COSH_F32,
        CALLEE_COSH_F64,
        CALLEE_TANH_F32,
        CALLEE_TANH_F64,
        CALLEE_ASINH_F32,
        CALLEE_ASINH_F64,
        CALLEE_ACOSH_F32,
        CALLEE_ACOSH_F64,
        CALLEE_ATANH_F32,
        CALLEE_ATANH_F64,
        CALLEE_EXPM1_F32,
        CALLEE_EXPM1_F64,
        CALLEE_LOG1P_F32,
        CALLEE_LOG1P_F64,
        CALLEE_HYPOT_F32,
        CALLEE_HYPOT_F64,
    ];

    /// Every placeholder that is libdevice-backed only under the libNVVM
    /// intrinsic backend (the ten rounding ops plus the three sign ops).
    const LIBNVVM_ONLY_LIBDEVICE: &[&str] = &[
        CALLEE_FLOOR_F32,
        CALLEE_FLOOR_F64,
        CALLEE_CEIL_F32,
        CALLEE_CEIL_F64,
        CALLEE_TRUNC_F32,
        CALLEE_TRUNC_F64,
        CALLEE_ROUND_F32,
        CALLEE_ROUND_F64,
        CALLEE_ROUNDEVEN_F32,
        CALLEE_ROUNDEVEN_F64,
        CALLEE_FABS,
        CALLEE_COPYSIGN_F32,
        CALLEE_COPYSIGN_F64,
    ];

    /// Callees that never lower to libdevice under any backend. `max`/`min`
    /// sit here because they expand to an ordered compare/select under every
    /// intrinsic backend (see #390).
    const NEVER_LIBDEVICE: &[&str] = &[
        CALLEE_MAXNUM_NSZ_F32,
        CALLEE_MAXNUM_NSZ_F64,
        CALLEE_MINNUM_NSZ_F32,
        CALLEE_MINNUM_NSZ_F64,
        CALLEE_ROTATE_LEFT,
        CALLEE_ROTATE_RIGHT,
        CALLEE_CTPOP,
        CALLEE_CTLZ,
        CALLEE_CTLZ_NONZERO,
        CALLEE_CTTZ,
        CALLEE_CTTZ_NONZERO,
        CALLEE_BSWAP,
        CALLEE_BITREVERSE,
        CALLEE_SATURATING_ADD,
        CALLEE_SATURATING_SUB,
        CALLEE_EXACT_DIV,
        CALLEE_CARRYING_MUL_ADD,
        CALLEE_FADD_FAST,
        CALLEE_FSUB_FAST,
        CALLEE_FMUL_FAST,
        CALLEE_FDIV_FAST,
        CALLEE_FREM_FAST,
        CALLEE_SELECT_UNPREDICTABLE,
        "__cuda_oxide_rust_intrinsic_unknown",
        "__nv_sinf",
    ];

    /// Both predicates are exact allow-lists and mutually disjoint: every
    /// placeholder in this module is asserted against BOTH predicates, so a
    /// callee added to one list without a classification decision here fails
    /// the test, and no callee can sit in both lists.
    #[test]
    fn libdevice_placeholder_classification_is_exact_and_disjoint() {
        for callee in ALWAYS_LIBDEVICE {
            assert!(is_known_placeholder(callee));
            assert!(
                is_libdevice_backed_placeholder(callee),
                "expected `{callee}` to require libdevice on every backend"
            );
            assert!(
                !is_backend_dependent_libdevice_placeholder(callee),
                "`{callee}` must not also be classified backend-dependent"
            );
        }

        for callee in LIBNVVM_ONLY_LIBDEVICE {
            assert!(is_known_placeholder(callee));
            assert!(
                is_backend_dependent_libdevice_placeholder(callee),
                "expected `{callee}` to require libdevice only under libNVVM"
            );
            assert!(
                !is_libdevice_backed_placeholder(callee),
                "`{callee}` must not also be classified always-libdevice"
            );
        }

        for callee in NEVER_LIBDEVICE {
            if callee.starts_with(PLACEHOLDER_PREFIX)
                && *callee != "__cuda_oxide_rust_intrinsic_unknown"
            {
                assert!(is_known_placeholder(callee));
            }
            assert!(
                !is_libdevice_backed_placeholder(callee),
                "expected `{callee}` not to require libdevice"
            );
            assert!(
                !is_backend_dependent_libdevice_placeholder(callee),
                "expected `{callee}` not to be backend-dependent libdevice"
            );
        }

        assert!(!is_known_placeholder("__cuda_oxide_rust_intrinsic_unknown"));
        assert!(!is_known_placeholder("__nv_sinf"));
    }
}
