/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rust compiler floating-point math intrinsics.

use super::super::helpers;
use crate::error::TranslationResult;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_mir::rust_intrinsics;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::location::Location;
use pliron::operation::Operation;
use rustc_public::mir;

/// Floating-point math intrinsic from libcore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RustFloatMathIntrinsic {
    /// `core::intrinsics::sqrtf32`.
    SqrtF32,
    /// `core::intrinsics::sqrtf64`.
    SqrtF64,
    /// `core::intrinsics::powif32`.
    PowiF32,
    /// `core::intrinsics::powif64`.
    PowiF64,
    /// `core::intrinsics::sinf32`.
    SinF32,
    /// `core::intrinsics::sinf64`.
    SinF64,
    /// `core::intrinsics::cosf32`.
    CosF32,
    /// `core::intrinsics::cosf64`.
    CosF64,
    /// `core::intrinsics::tanf32`.
    TanF32,
    /// `core::intrinsics::tanf64`.
    TanF64,
    /// `core::intrinsics::powf32`.
    PowfF32,
    /// `core::intrinsics::powf64`.
    PowfF64,
    /// `core::intrinsics::expf32`.
    ExpF32,
    /// `core::intrinsics::expf64`.
    ExpF64,
    /// `core::intrinsics::exp2f32`.
    Exp2F32,
    /// `core::intrinsics::exp2f64`.
    Exp2F64,
    /// `core::intrinsics::logf32`.
    LogF32,
    /// `core::intrinsics::logf64`.
    LogF64,
    /// `core::intrinsics::log2f32`.
    Log2F32,
    /// `core::intrinsics::log2f64`.
    Log2F64,
    /// `core::intrinsics::log10f32`.
    Log10F32,
    /// `core::intrinsics::log10f64`.
    Log10F64,
    /// `core::intrinsics::fmaf32`.
    FmaF32,
    /// `core::intrinsics::fmaf64`.
    FmaF64,
    /// `core::intrinsics::fmuladdf32`.
    FmuladdF32,
    /// `core::intrinsics::fmuladdf64`.
    FmuladdF64,
    /// `core::intrinsics::floorf32`.
    FloorF32,
    /// `core::intrinsics::floorf64`.
    FloorF64,
    /// `core::intrinsics::ceilf32`.
    CeilF32,
    /// `core::intrinsics::ceilf64`.
    CeilF64,
    /// `core::intrinsics::truncf32`.
    TruncF32,
    /// `core::intrinsics::truncf64`.
    TruncF64,
    /// `core::intrinsics::roundf32`.
    RoundF32,
    /// `core::intrinsics::roundf64`.
    RoundF64,
    /// `core::intrinsics::round_ties_even_f32`.
    RoundevenF32,
    /// `core::intrinsics::round_ties_even_f64`.
    RoundevenF64,
    /// Generic `core::intrinsics::fabs`.
    Fabs,
    /// `core::intrinsics::copysignf32`.
    CopysignF32,
    /// `core::intrinsics::copysignf64`.
    CopysignF64,
    /// `core::intrinsics::maximum_number_nsz_f32` (backs `f32::max`).
    MaxNumNszF32,
    /// `core::intrinsics::maximum_number_nsz_f64` (backs `f64::max`).
    MaxNumNszF64,
    /// `core::intrinsics::minimum_number_nsz_f32` (backs `f32::min`).
    MinNumNszF32,
    /// `core::intrinsics::minimum_number_nsz_f64` (backs `f64::min`).
    MinNumNszF64,
    /// `f32::asin` / `std::sys::cmath::asinf`.
    AsinF32,
    /// `f64::asin` / `std::sys::cmath::asin`.
    AsinF64,
    /// `f32::acos` / `std::sys::cmath::acosf`.
    AcosF32,
    /// `f64::acos` / `std::sys::cmath::acos`.
    AcosF64,
    /// `f32::atan2` / `std::sys::cmath::atan2f`.
    Atan2F32,
    /// `f64::atan2` / `std::sys::cmath::atan2`.
    Atan2F64,
    /// `f32::atan` / `std::sys::cmath::atanf`.
    AtanF32,
    /// `f64::atan` / `std::sys::cmath::atan`.
    AtanF64,
    /// `f32::cbrt` / `std::sys::cmath::cbrtf`.
    CbrtF32,
    /// `f64::cbrt` / `std::sys::cmath::cbrt`.
    CbrtF64,
    /// `f32::sinh` / `std::sys::cmath::sinhf`.
    SinhF32,
    /// `f64::sinh` / `std::sys::cmath::sinh`.
    SinhF64,
    /// `f32::cosh` / `std::sys::cmath::coshf`.
    CoshF32,
    /// `f64::cosh` / `std::sys::cmath::cosh`.
    CoshF64,
    /// `f32::tanh` / `std::sys::cmath::tanhf`.
    TanhF32,
    /// `f64::tanh` / `std::sys::cmath::tanh`.
    TanhF64,
    /// `f32::asinh` / `std::sys::cmath::asinhf`.
    AsinhF32,
    /// `f64::asinh` / `std::sys::cmath::asinh`.
    AsinhF64,
    /// `f32::acosh` / `std::sys::cmath::acoshf`.
    AcoshF32,
    /// `f64::acosh` / `std::sys::cmath::acosh`.
    AcoshF64,
    /// `f32::atanh` / `std::sys::cmath::atanhf`.
    AtanhF32,
    /// `f64::atanh` / `std::sys::cmath::atanh`.
    AtanhF64,
    /// `f32::exp_m1` / `std::sys::cmath::expm1f`.
    Expm1F32,
    /// `f64::exp_m1` / `std::sys::cmath::expm1`.
    Expm1F64,
    /// `f32::ln_1p` / `std::sys::cmath::log1pf`.
    Log1pF32,
    /// `f64::ln_1p` / `std::sys::cmath::log1p`.
    Log1pF64,
    /// `f32::hypot` / `std::sys::cmath::hypotf` (binary).
    HypotF32,
    /// `f64::hypot` / `std::sys::cmath::hypot` (binary).
    HypotF64,
    /// Generic `core::intrinsics::fadd_fast` (lowered to `llvm.fadd` + fast-math).
    FaddFast,
    /// Generic `core::intrinsics::fsub_fast` (lowered to `llvm.fsub` + fast-math).
    FsubFast,
    /// Generic `core::intrinsics::fmul_fast` (lowered to `llvm.fmul` + fast-math).
    FmulFast,
    /// Generic `core::intrinsics::fdiv_fast` (lowered to `llvm.fdiv` + fast-math).
    FdivFast,
    /// Generic `core::intrinsics::frem_fast` (lowered to `llvm.frem` + fast-math).
    FremFast,
}

impl RustFloatMathIntrinsic {
    /// Recognize the libcore intrinsic path that survived into MIR.
    pub fn from_core_path(name: &str) -> Option<Self> {
        match name {
            "core::intrinsics::sqrtf32" | "std::intrinsics::sqrtf32" => Some(Self::SqrtF32),
            "core::intrinsics::sqrtf64" | "std::intrinsics::sqrtf64" => Some(Self::SqrtF64),
            "core::intrinsics::powif32" | "std::intrinsics::powif32" => Some(Self::PowiF32),
            "core::intrinsics::powif64" | "std::intrinsics::powif64" => Some(Self::PowiF64),
            "core::intrinsics::sinf32" | "std::intrinsics::sinf32" => Some(Self::SinF32),
            "core::intrinsics::sinf64" | "std::intrinsics::sinf64" => Some(Self::SinF64),
            "core::intrinsics::cosf32" | "std::intrinsics::cosf32" => Some(Self::CosF32),
            "core::intrinsics::cosf64" | "std::intrinsics::cosf64" => Some(Self::CosF64),
            "core::intrinsics::tanf32" | "std::intrinsics::tanf32" => Some(Self::TanF32),
            "core::intrinsics::tanf64" | "std::intrinsics::tanf64" => Some(Self::TanF64),
            "core::intrinsics::powf32" | "std::intrinsics::powf32" => Some(Self::PowfF32),
            "core::intrinsics::powf64" | "std::intrinsics::powf64" => Some(Self::PowfF64),
            "core::intrinsics::expf32" | "std::intrinsics::expf32" => Some(Self::ExpF32),
            "core::intrinsics::expf64" | "std::intrinsics::expf64" => Some(Self::ExpF64),
            "core::intrinsics::exp2f32" | "std::intrinsics::exp2f32" => Some(Self::Exp2F32),
            "core::intrinsics::exp2f64" | "std::intrinsics::exp2f64" => Some(Self::Exp2F64),
            "core::intrinsics::logf32" | "std::intrinsics::logf32" => Some(Self::LogF32),
            "core::intrinsics::logf64" | "std::intrinsics::logf64" => Some(Self::LogF64),
            "core::intrinsics::log2f32" | "std::intrinsics::log2f32" => Some(Self::Log2F32),
            "core::intrinsics::log2f64" | "std::intrinsics::log2f64" => Some(Self::Log2F64),
            "core::intrinsics::log10f32" | "std::intrinsics::log10f32" => Some(Self::Log10F32),
            "core::intrinsics::log10f64" | "std::intrinsics::log10f64" => Some(Self::Log10F64),
            "core::intrinsics::fmaf32" | "std::intrinsics::fmaf32" => Some(Self::FmaF32),
            "core::intrinsics::fmaf64" | "std::intrinsics::fmaf64" => Some(Self::FmaF64),
            "core::intrinsics::fmuladdf32" | "std::intrinsics::fmuladdf32" => {
                Some(Self::FmuladdF32)
            }
            "core::intrinsics::fmuladdf64" | "std::intrinsics::fmuladdf64" => {
                Some(Self::FmuladdF64)
            }
            "core::intrinsics::floorf32" | "std::intrinsics::floorf32" => Some(Self::FloorF32),
            "core::intrinsics::floorf64" | "std::intrinsics::floorf64" => Some(Self::FloorF64),
            "core::intrinsics::ceilf32" | "std::intrinsics::ceilf32" => Some(Self::CeilF32),
            "core::intrinsics::ceilf64" | "std::intrinsics::ceilf64" => Some(Self::CeilF64),
            "core::intrinsics::truncf32" | "std::intrinsics::truncf32" => Some(Self::TruncF32),
            "core::intrinsics::truncf64" | "std::intrinsics::truncf64" => Some(Self::TruncF64),
            "core::intrinsics::roundf32" | "std::intrinsics::roundf32" => Some(Self::RoundF32),
            "core::intrinsics::roundf64" | "std::intrinsics::roundf64" => Some(Self::RoundF64),
            "core::intrinsics::round_ties_even_f32" | "std::intrinsics::round_ties_even_f32" => {
                Some(Self::RoundevenF32)
            }
            "core::intrinsics::round_ties_even_f64" | "std::intrinsics::round_ties_even_f64" => {
                Some(Self::RoundevenF64)
            }
            "core::intrinsics::fabs" | "std::intrinsics::fabs" => Some(Self::Fabs),
            "core::intrinsics::copysignf32" | "std::intrinsics::copysignf32" => {
                Some(Self::CopysignF32)
            }
            "core::intrinsics::copysignf64" | "std::intrinsics::copysignf64" => {
                Some(Self::CopysignF64)
            }
            "core::intrinsics::maximum_number_nsz_f32"
            | "std::intrinsics::maximum_number_nsz_f32" => Some(Self::MaxNumNszF32),
            "core::intrinsics::maximum_number_nsz_f64"
            | "std::intrinsics::maximum_number_nsz_f64" => Some(Self::MaxNumNszF64),
            "core::intrinsics::minimum_number_nsz_f32"
            | "std::intrinsics::minimum_number_nsz_f32" => Some(Self::MinNumNszF32),
            "core::intrinsics::minimum_number_nsz_f64"
            | "std::intrinsics::minimum_number_nsz_f64" => Some(Self::MinNumNszF64),
            // Trig that still routes through `std`'s libm shim. On current
            // nightlies `core_float_math` moves `sin`/`cos` into core
            // intrinsics (caught above), but `tan` is *not* in that feature,
            // so `f{32,64}::tan()` lowers to `std::sys::cmath::tan{,f}` and
            // would otherwise trip the forbidden-`std`-crate guard. Intercept
            // it (and sin/cos defensively, for toolchains/builds that take the
            // std path) and lower to `__nv_tan{,f}` like every other libdevice
            // math call.
            "std::sys::cmath::sinf" => Some(Self::SinF32),
            "std::sys::cmath::sin" => Some(Self::SinF64),
            "std::sys::cmath::cosf" => Some(Self::CosF32),
            "std::sys::cmath::cos" => Some(Self::CosF64),
            "std::sys::cmath::tanf" => Some(Self::TanF32),
            "std::sys::cmath::tan" => Some(Self::TanF64),
            "std::sys::cmath::asinf" => Some(Self::AsinF32),
            "std::sys::cmath::asin" => Some(Self::AsinF64),
            "std::sys::cmath::acosf" => Some(Self::AcosF32),
            "std::sys::cmath::acos" => Some(Self::AcosF64),
            "std::sys::cmath::atan2f" => Some(Self::Atan2F32),
            "std::sys::cmath::atan2" => Some(Self::Atan2F64),
            "std::sys::cmath::atanf" => Some(Self::AtanF32),
            "std::sys::cmath::atan" => Some(Self::AtanF64),
            "std::sys::cmath::cbrtf" => Some(Self::CbrtF32),
            "std::sys::cmath::cbrt" => Some(Self::CbrtF64),
            "core::num::imp::libm::cbrtf" => Some(Self::CbrtF32),
            "core::num::imp::libm::cbrt" => Some(Self::CbrtF64),
            // Hyperbolic + extended transcendentals. Like `tan`, none of
            // these are in `core_float_math`, so `f{32,64}::{sinh,..,hypot}()`
            // lower to their `std::sys::cmath::*` shims and would hit the
            // std-crate guard without interception.
            "std::sys::cmath::sinhf" => Some(Self::SinhF32),
            "std::sys::cmath::sinh" => Some(Self::SinhF64),
            "std::sys::cmath::coshf" => Some(Self::CoshF32),
            "std::sys::cmath::cosh" => Some(Self::CoshF64),
            "std::sys::cmath::tanhf" => Some(Self::TanhF32),
            "std::sys::cmath::tanh" => Some(Self::TanhF64),
            // Inverse hyperbolics: pure-Rust formulas in `std` before
            // nightly-2026-08-28. On this nightly (rustc e457a7b0d) only
            // asinh/acosh became `std::sys::cmath` shims; atanh keeps its
            // pure-Rust `ln_1p` formula and cmath declares no atanh, so the
            // atanh arms are defensive, for when std makes the same move.
            "std::sys::cmath::asinhf" => Some(Self::AsinhF32),
            "std::sys::cmath::asinh" => Some(Self::AsinhF64),
            "std::sys::cmath::acoshf" => Some(Self::AcoshF32),
            "std::sys::cmath::acosh" => Some(Self::AcoshF64),
            "std::sys::cmath::atanhf" => Some(Self::AtanhF32),
            "std::sys::cmath::atanh" => Some(Self::AtanhF64),
            "std::sys::cmath::expm1f" => Some(Self::Expm1F32),
            "std::sys::cmath::expm1" => Some(Self::Expm1F64),
            "std::sys::cmath::log1pf" => Some(Self::Log1pF32),
            "std::sys::cmath::log1p" => Some(Self::Log1pF64),
            "std::sys::cmath::hypotf" => Some(Self::HypotF32),
            "std::sys::cmath::hypot" => Some(Self::HypotF64),
            other => Self::from_libm_path(other).or_else(|| Self::from_fast_intrinsic_path(other)),
        }
    }

    /// Recognize `libm` crate float functions (the `nostd-libm` lowering used
    /// by glam on nvptx, e.g. `libm::math::sqrt::sqrtf`). These dependency-rlib
    /// functions frequently lack exportable MIR downstream, so we intercept them
    /// and lower to the same libdevice intrinsics as the core-intrinsic forms,
    /// giving hardware math instead of software libm. Match on the final path
    /// segment so both the canonical (`libm::math::sqrt::sqrtf`) and re-exported
    /// (`libm::sqrtf`) spellings are caught.
    fn from_libm_path(name: &str) -> Option<Self> {
        if !is_libm_path(name) {
            return None;
        }
        let seg = name.rsplit("::").next().unwrap_or(name);
        match seg {
            "sqrtf" => Some(Self::SqrtF32),
            "sqrt" => Some(Self::SqrtF64),
            "sinf" => Some(Self::SinF32),
            "sin" => Some(Self::SinF64),
            "cosf" => Some(Self::CosF32),
            "cos" => Some(Self::CosF64),
            "tanf" => Some(Self::TanF32),
            "tan" => Some(Self::TanF64),
            "expf" => Some(Self::ExpF32),
            "exp" => Some(Self::ExpF64),
            "exp2f" => Some(Self::Exp2F32),
            "exp2" => Some(Self::Exp2F64),
            "logf" => Some(Self::LogF32),
            "log" => Some(Self::LogF64),
            "log2f" => Some(Self::Log2F32),
            "log2" => Some(Self::Log2F64),
            "log10f" => Some(Self::Log10F32),
            "log10" => Some(Self::Log10F64),
            // libm: `powf` is the f32 power, `pow` is the f64 power.
            "powf" => Some(Self::PowfF32),
            "pow" => Some(Self::PowfF64),
            "floorf" => Some(Self::FloorF32),
            "floor" => Some(Self::FloorF64),
            "ceilf" => Some(Self::CeilF32),
            "ceil" => Some(Self::CeilF64),
            "truncf" => Some(Self::TruncF32),
            "trunc" => Some(Self::TruncF64),
            "roundf" => Some(Self::RoundF32),
            "round" => Some(Self::RoundF64),
            // libm's `rint` documents round-half-to-even; the device has no
            // dynamic rounding mode, so it shares the roundeven lowering.
            "rintf" | "roundevenf" => Some(Self::RoundevenF32),
            "rint" | "roundeven" => Some(Self::RoundevenF64),
            "fmaf" => Some(Self::FmaF32),
            "fma" => Some(Self::FmaF64),
            "fabsf" | "fabs" => Some(Self::Fabs),
            "copysignf" => Some(Self::CopysignF32),
            "copysign" => Some(Self::CopysignF64),
            "fmaxf" => Some(Self::MaxNumNszF32),
            "fmax" => Some(Self::MaxNumNszF64),
            "fminf" => Some(Self::MinNumNszF32),
            "fmin" => Some(Self::MinNumNszF64),
            "asinf" => Some(Self::AsinF32),
            "asin" => Some(Self::AsinF64),
            "acosf" => Some(Self::AcosF32),
            "acos" => Some(Self::AcosF64),
            "atan2f" => Some(Self::Atan2F32),
            "atan2" => Some(Self::Atan2F64),
            "atanf" => Some(Self::AtanF32),
            "atan" => Some(Self::AtanF64),
            "cbrtf" => Some(Self::CbrtF32),
            "cbrt" => Some(Self::CbrtF64),
            "sinhf" => Some(Self::SinhF32),
            "sinh" => Some(Self::SinhF64),
            "coshf" => Some(Self::CoshF32),
            "cosh" => Some(Self::CoshF64),
            "tanhf" => Some(Self::TanhF32),
            "tanh" => Some(Self::TanhF64),
            "asinhf" => Some(Self::AsinhF32),
            "asinh" => Some(Self::AsinhF64),
            "acoshf" => Some(Self::AcoshF32),
            "acosh" => Some(Self::AcoshF64),
            "atanhf" => Some(Self::AtanhF32),
            "atanh" => Some(Self::AtanhF64),
            "expm1f" => Some(Self::Expm1F32),
            "expm1" => Some(Self::Expm1F64),
            "log1pf" => Some(Self::Log1pF32),
            "log1p" => Some(Self::Log1pF64),
            "hypotf" => Some(Self::HypotF32),
            "hypot" => Some(Self::HypotF64),
            _ => None,
        }
    }

    /// Recognize `core::intrinsics` / `std::intrinsics` generic fast-float ops.
    fn from_fast_intrinsic_path(name: &str) -> Option<Self> {
        match name {
            // Generic finite-input arithmetic intrinsics. The FQDN carries no
            // float-type suffix because they're polymorphic over `T:
            // FloatPrimitive`; the float type is in the call's substs and
            // recovered from the destination's `body.locals()[…].ty` at
            // lowering time.
            "core::intrinsics::fadd_fast" | "std::intrinsics::fadd_fast" => Some(Self::FaddFast),
            "core::intrinsics::fsub_fast" | "std::intrinsics::fsub_fast" => Some(Self::FsubFast),
            "core::intrinsics::fmul_fast" | "std::intrinsics::fmul_fast" => Some(Self::FmulFast),
            "core::intrinsics::fdiv_fast" | "std::intrinsics::fdiv_fast" => Some(Self::FdivFast),
            "core::intrinsics::frem_fast" | "std::intrinsics::frem_fast" => Some(Self::FremFast),
            _ => None,
        }
    }

    /// Return the internal placeholder name used until MIR-to-LLVM lowering.
    pub fn placeholder_callee(self) -> &'static str {
        match self {
            Self::SqrtF32 => rust_intrinsics::CALLEE_SQRT_F32,
            Self::SqrtF64 => rust_intrinsics::CALLEE_SQRT_F64,
            Self::PowiF32 => rust_intrinsics::CALLEE_POWI_F32,
            Self::PowiF64 => rust_intrinsics::CALLEE_POWI_F64,
            Self::SinF32 => rust_intrinsics::CALLEE_SIN_F32,
            Self::SinF64 => rust_intrinsics::CALLEE_SIN_F64,
            Self::CosF32 => rust_intrinsics::CALLEE_COS_F32,
            Self::CosF64 => rust_intrinsics::CALLEE_COS_F64,
            Self::TanF32 => rust_intrinsics::CALLEE_TAN_F32,
            Self::TanF64 => rust_intrinsics::CALLEE_TAN_F64,
            Self::PowfF32 => rust_intrinsics::CALLEE_POWF_F32,
            Self::PowfF64 => rust_intrinsics::CALLEE_POWF_F64,
            Self::ExpF32 => rust_intrinsics::CALLEE_EXP_F32,
            Self::ExpF64 => rust_intrinsics::CALLEE_EXP_F64,
            Self::Exp2F32 => rust_intrinsics::CALLEE_EXP2_F32,
            Self::Exp2F64 => rust_intrinsics::CALLEE_EXP2_F64,
            Self::LogF32 => rust_intrinsics::CALLEE_LOG_F32,
            Self::LogF64 => rust_intrinsics::CALLEE_LOG_F64,
            Self::Log2F32 => rust_intrinsics::CALLEE_LOG2_F32,
            Self::Log2F64 => rust_intrinsics::CALLEE_LOG2_F64,
            Self::Log10F32 => rust_intrinsics::CALLEE_LOG10_F32,
            Self::Log10F64 => rust_intrinsics::CALLEE_LOG10_F64,
            Self::FmaF32 => rust_intrinsics::CALLEE_FMA_F32,
            Self::FmaF64 => rust_intrinsics::CALLEE_FMA_F64,
            Self::FmuladdF32 => rust_intrinsics::CALLEE_FMULADD_F32,
            Self::FmuladdF64 => rust_intrinsics::CALLEE_FMULADD_F64,
            Self::FloorF32 => rust_intrinsics::CALLEE_FLOOR_F32,
            Self::FloorF64 => rust_intrinsics::CALLEE_FLOOR_F64,
            Self::CeilF32 => rust_intrinsics::CALLEE_CEIL_F32,
            Self::CeilF64 => rust_intrinsics::CALLEE_CEIL_F64,
            Self::TruncF32 => rust_intrinsics::CALLEE_TRUNC_F32,
            Self::TruncF64 => rust_intrinsics::CALLEE_TRUNC_F64,
            Self::RoundF32 => rust_intrinsics::CALLEE_ROUND_F32,
            Self::RoundF64 => rust_intrinsics::CALLEE_ROUND_F64,
            Self::RoundevenF32 => rust_intrinsics::CALLEE_ROUNDEVEN_F32,
            Self::RoundevenF64 => rust_intrinsics::CALLEE_ROUNDEVEN_F64,
            Self::Fabs => rust_intrinsics::CALLEE_FABS,
            Self::CopysignF32 => rust_intrinsics::CALLEE_COPYSIGN_F32,
            Self::CopysignF64 => rust_intrinsics::CALLEE_COPYSIGN_F64,
            Self::MaxNumNszF32 => rust_intrinsics::CALLEE_MAXNUM_NSZ_F32,
            Self::MaxNumNszF64 => rust_intrinsics::CALLEE_MAXNUM_NSZ_F64,
            Self::MinNumNszF32 => rust_intrinsics::CALLEE_MINNUM_NSZ_F32,
            Self::MinNumNszF64 => rust_intrinsics::CALLEE_MINNUM_NSZ_F64,
            Self::AsinF32 => rust_intrinsics::CALLEE_ASIN_F32,
            Self::AsinF64 => rust_intrinsics::CALLEE_ASIN_F64,
            Self::AcosF32 => rust_intrinsics::CALLEE_ACOS_F32,
            Self::AcosF64 => rust_intrinsics::CALLEE_ACOS_F64,
            Self::Atan2F32 => rust_intrinsics::CALLEE_ATAN2_F32,
            Self::Atan2F64 => rust_intrinsics::CALLEE_ATAN2_F64,
            Self::AtanF32 => rust_intrinsics::CALLEE_ATAN_F32,
            Self::AtanF64 => rust_intrinsics::CALLEE_ATAN_F64,
            Self::CbrtF32 => rust_intrinsics::CALLEE_CBRT_F32,
            Self::CbrtF64 => rust_intrinsics::CALLEE_CBRT_F64,
            Self::SinhF32 => rust_intrinsics::CALLEE_SINH_F32,
            Self::SinhF64 => rust_intrinsics::CALLEE_SINH_F64,
            Self::CoshF32 => rust_intrinsics::CALLEE_COSH_F32,
            Self::CoshF64 => rust_intrinsics::CALLEE_COSH_F64,
            Self::TanhF32 => rust_intrinsics::CALLEE_TANH_F32,
            Self::TanhF64 => rust_intrinsics::CALLEE_TANH_F64,
            Self::AsinhF32 => rust_intrinsics::CALLEE_ASINH_F32,
            Self::AsinhF64 => rust_intrinsics::CALLEE_ASINH_F64,
            Self::AcoshF32 => rust_intrinsics::CALLEE_ACOSH_F32,
            Self::AcoshF64 => rust_intrinsics::CALLEE_ACOSH_F64,
            Self::AtanhF32 => rust_intrinsics::CALLEE_ATANH_F32,
            Self::AtanhF64 => rust_intrinsics::CALLEE_ATANH_F64,
            Self::Expm1F32 => rust_intrinsics::CALLEE_EXPM1_F32,
            Self::Expm1F64 => rust_intrinsics::CALLEE_EXPM1_F64,
            Self::Log1pF32 => rust_intrinsics::CALLEE_LOG1P_F32,
            Self::Log1pF64 => rust_intrinsics::CALLEE_LOG1P_F64,
            Self::HypotF32 => rust_intrinsics::CALLEE_HYPOT_F32,
            Self::HypotF64 => rust_intrinsics::CALLEE_HYPOT_F64,
            Self::FaddFast => rust_intrinsics::CALLEE_FADD_FAST,
            Self::FsubFast => rust_intrinsics::CALLEE_FSUB_FAST,
            Self::FmulFast => rust_intrinsics::CALLEE_FMUL_FAST,
            Self::FdivFast => rust_intrinsics::CALLEE_FDIV_FAST,
            Self::FremFast => rust_intrinsics::CALLEE_FREM_FAST,
        }
    }
}

/// Whether `name` is a path rooted in the `libm` crate: the first path
/// segment must be exactly `libm`. A bare substring test would also match
/// user functions whose path merely mentions libm (e.g.
/// `my_app::libm_compat::expf`), silently replacing the user's body with a
/// libdevice call.
pub fn is_libm_path(name: &str) -> bool {
    name.split("::").next() == Some("libm")
}

/// Recognize `libm::sincosf` / `libm::sincos` (glam's `nostd-libm` lowering of
/// `f32::sin_cos`). These return a `(sin, cos)` tuple, so they do not fit the
/// scalar `RustFloatMathIntrinsic` dispatch; [`emit_sincos`] handles them.
/// Returns `Some(is_f64)` when `name` is a libm sincos function.
pub fn libm_sincos_is_f64(name: &str) -> Option<bool> {
    if !is_libm_path(name) {
        return None;
    }
    match name.rsplit("::").next().unwrap_or(name) {
        "sincosf" => Some(false),
        "sincos" => Some(true),
        _ => None,
    }
}

/// Emit `libm::sincos{f}(x) -> (sin, cos)` as a `sinf`/`cosf` placeholder pair
/// (lowered to libdevice `__nv_sin/cosf`) packed into the destination tuple.
#[allow(clippy::too_many_arguments)]
pub fn emit_sincos(
    ctx: &mut Context,
    body: &mir::Body,
    is_f64: bool,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    use crate::error::TranslationErr;
    use crate::translator::rvalue;
    use dialect_mir::ops::{MirCallOp, MirConstructTupleOp};
    use pliron::builtin::attributes::StringAttr;
    use pliron::input_err;
    use pliron::location::Located;
    use pliron::op::Op;

    // Destination tuple type and its scalar element type. Typed from the
    // projected place, so a `RET.1 = sincos(..)`-style destination is the
    // tuple actually written rather than the whole local.
    let tuple_ty = types::translate_destination_type(ctx, body, destination, &loc)?;
    let scalar_ty = {
        let r = tuple_ty.deref(ctx);
        match r.downcast_ref::<dialect_mir::types::MirTupleType>() {
            Some(t) => t.get_types()[0],
            None => {
                return input_err!(
                    loc.clone(),
                    TranslationErr::unsupported(
                        "libm::sincos destination is not a tuple".to_string()
                    )
                );
            }
        }
    };

    // Translate the angle argument once and reuse it for both calls.
    let (arg_value, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let (prepared_destination, last_op) = helpers::prepare_destination_write(
        ctx,
        body,
        destination,
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let (sin_callee, cos_callee) = if is_f64 {
        (
            rust_intrinsics::CALLEE_SIN_F64,
            rust_intrinsics::CALLEE_COS_F64,
        )
    } else {
        (
            rust_intrinsics::CALLEE_SIN_F32,
            rust_intrinsics::CALLEE_COS_F32,
        )
    };

    let callee_id = pliron::identifier::Identifier::try_from("callee").unwrap();

    // sin(x)
    let sin_op = Operation::new(
        ctx,
        MirCallOp::get_concrete_op_info(),
        vec![scalar_ty],
        vec![arg_value],
        vec![],
        0,
    );
    sin_op.deref_mut(ctx).set_loc(loc.clone());
    sin_op
        .deref_mut(ctx)
        .attributes
        .set(callee_id.clone(), StringAttr::new(sin_callee.into()));
    if let Some(prev) = last_op {
        sin_op.insert_after(ctx, prev);
    } else {
        sin_op.insert_at_front(block_ptr, ctx);
    }
    let sin_val = sin_op.deref(ctx).get_result(0);

    // cos(x)
    let cos_op = Operation::new(
        ctx,
        MirCallOp::get_concrete_op_info(),
        vec![scalar_ty],
        vec![arg_value],
        vec![],
        0,
    );
    cos_op.deref_mut(ctx).set_loc(loc.clone());
    cos_op
        .deref_mut(ctx)
        .attributes
        .set(callee_id, StringAttr::new(cos_callee.into()));
    cos_op.insert_after(ctx, sin_op);
    let cos_val = cos_op.deref(ctx).get_result(0);

    // Pack (sin, cos) into the destination tuple.
    let tuple_op = Operation::new(
        ctx,
        MirConstructTupleOp::get_concrete_op_info(),
        vec![tuple_ty],
        vec![sin_val, cos_val],
        vec![],
        0,
    );
    tuple_op.deref_mut(ctx).set_loc(loc.clone());
    tuple_op.insert_after(ctx, cos_op);
    let tuple_val = tuple_op.deref(ctx).get_result(0);

    helpers::emit_prepared_result_and_goto(
        ctx,
        prepared_destination,
        tuple_val,
        target,
        block_ptr,
        tuple_op,
        value_map,
        block_map,
        loc,
        "libm::sincos call without target not supported",
    )
}

/// Emit a placeholder `mir.call` for a rustc float math intrinsic.
#[allow(clippy::too_many_arguments)]
pub fn emit_rust_float_math_intrinsic(
    ctx: &mut Context,
    body: &mir::Body,
    intrinsic: RustFloatMathIntrinsic,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let return_type = types::translate_destination_type(ctx, body, destination, &loc)?;
    helpers::emit_function_call(
        ctx,
        body,
        intrinsic.placeholder_callee(),
        args,
        destination,
        return_type,
        None,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::rust_intrinsics;

    /// `f32::max` / `f64::max` / `f32::min` / `f64::min` all lower to the
    /// `_nsz` flavor of the rustc maxNum/minNum intrinsics. Lock the four
    /// `core::intrinsics::*` paths and their `std::intrinsics::*` aliases to
    /// the dedicated enum variants so a rustc rename surfaces here as a
    /// compile-time failure rather than a runtime "intrinsic not lowered"
    /// error.
    #[test]
    fn from_core_path_recognizes_maxnum_minnum_nsz_intrinsics() {
        for (path, expected) in [
            (
                "core::intrinsics::maximum_number_nsz_f32",
                RustFloatMathIntrinsic::MaxNumNszF32,
            ),
            (
                "std::intrinsics::maximum_number_nsz_f32",
                RustFloatMathIntrinsic::MaxNumNszF32,
            ),
            (
                "core::intrinsics::maximum_number_nsz_f64",
                RustFloatMathIntrinsic::MaxNumNszF64,
            ),
            (
                "core::intrinsics::minimum_number_nsz_f32",
                RustFloatMathIntrinsic::MinNumNszF32,
            ),
            (
                "core::intrinsics::minimum_number_nsz_f64",
                RustFloatMathIntrinsic::MinNumNszF64,
            ),
        ] {
            assert_eq!(
                RustFloatMathIntrinsic::from_core_path(path),
                Some(expected),
                "`{path}` did not map to the expected intrinsic"
            );
        }

        // Negative case: the NaN-propagating `maximumf*` / `minimumf*`
        // family (backing `f32::maximum` / `f32::minimum`) is intentionally
        // not handled in this PR. Make sure it does not silently get
        // routed to the `_nsz` variants.
        assert_eq!(
            RustFloatMathIntrinsic::from_core_path("core::intrinsics::maximumf32"),
            None
        );
        assert_eq!(
            RustFloatMathIntrinsic::from_core_path("core::intrinsics::minimumf32"),
            None
        );
    }

    /// `f{32,64}::cbrt` reaches device codegen as either the `std::sys::cmath`
    /// C shim or the in-tree pure-Rust libm path, depending on toolchain.
    /// Both must map to the libdevice-backed `Cbrt*` variants; pin them so a
    /// rustc rename surfaces as a test failure rather than an undefined-symbol
    /// PTX verification error.
    #[test]
    fn from_core_path_recognizes_cbrt_via_cmath_and_libm() {
        for (path, expected) in [
            ("std::sys::cmath::cbrtf", RustFloatMathIntrinsic::CbrtF32),
            ("std::sys::cmath::cbrt", RustFloatMathIntrinsic::CbrtF64),
            (
                "core::num::imp::libm::cbrtf",
                RustFloatMathIntrinsic::CbrtF32,
            ),
            (
                "core::num::imp::libm::cbrt",
                RustFloatMathIntrinsic::CbrtF64,
            ),
            ("libm::cbrtf", RustFloatMathIntrinsic::CbrtF32),
            ("libm::math::cbrt::cbrt", RustFloatMathIntrinsic::CbrtF64),
        ] {
            assert_eq!(
                RustFloatMathIntrinsic::from_core_path(path),
                Some(expected),
                "`{path}` did not map to the expected cbrt intrinsic"
            );
        }
    }

    /// `asin`/`acos` reach libdevice from both the std `f{32,64}` inherent
    /// methods (`std::sys::cmath::*`) and the `libm` crate spellings glam's
    /// `nostd-libm` lowering emits on nvptx. Unlike `cbrt` these are std-only
    /// (no `core::num::imp::libm` form), so only the cmath and libm-crate
    /// spellings exist; both must map to the same `__nv_a{sin,cos}{,f}` path.
    #[test]
    fn from_core_path_recognizes_asin_acos_via_cmath_and_libm() {
        for (path, expected) in [
            ("std::sys::cmath::asinf", RustFloatMathIntrinsic::AsinF32),
            ("std::sys::cmath::asin", RustFloatMathIntrinsic::AsinF64),
            ("std::sys::cmath::acosf", RustFloatMathIntrinsic::AcosF32),
            ("std::sys::cmath::acos", RustFloatMathIntrinsic::AcosF64),
            ("libm::asinf", RustFloatMathIntrinsic::AsinF32),
            ("libm::math::asin::asin", RustFloatMathIntrinsic::AsinF64),
            ("libm::acosf", RustFloatMathIntrinsic::AcosF32),
            ("libm::math::acos::acos", RustFloatMathIntrinsic::AcosF64),
        ] {
            assert_eq!(
                RustFloatMathIntrinsic::from_core_path(path),
                Some(expected),
                "`{path}` did not map to the expected asin/acos intrinsic"
            );
        }
    }

    /// Libm interception must be anchored to the `libm` crate root. A user
    /// function that shares a libm function name, inside a path that merely
    /// mentions "libm", must stay a regular call: a bare `contains("libm")`
    /// test rerouted such calls to libdevice, silently replacing the user's
    /// body (miscompile caught in PR #142 review).
    #[test]
    fn libm_interception_is_anchored_to_the_libm_crate_root() {
        // Canonical and re-exported libm spellings are intercepted.
        for (path, expected) in [
            ("libm::math::expf::expf", RustFloatMathIntrinsic::ExpF32),
            ("libm::expf", RustFloatMathIntrinsic::ExpF32),
            ("libm::math::sqrt::sqrt", RustFloatMathIntrinsic::SqrtF64),
        ] {
            assert_eq!(
                RustFloatMathIntrinsic::from_core_path(path),
                Some(expected),
                "`{path}` should be intercepted as a libm function"
            );
        }

        // Adversarial: a user `expf` under a path containing "libm" is NOT
        // the libm crate and must not be rerouted.
        for path in [
            "my_app::libm_compat::expf",
            "my_app::libm::expf",
            "libmath::expf",
            "libm_math::lookalike::expf",
            "not_libm::expf",
        ] {
            assert_eq!(
                RustFloatMathIntrinsic::from_core_path(path),
                None,
                "user function `{path}` was wrongly rerouted to libdevice"
            );
        }
    }

    /// Pin the libm names that reuse existing enum variants: fmax/fmin map
    /// to the `_nsz` maxNum/minNum lowering (same as `f32::max`/`f32::min`),
    /// and rint/roundeven map to the round-ties-even lowering.
    #[test]
    fn libm_fmax_fmin_rint_roundeven_map_to_existing_variants() {
        for (path, expected) in [
            ("libm::fmaxf", RustFloatMathIntrinsic::MaxNumNszF32),
            ("libm::fmax", RustFloatMathIntrinsic::MaxNumNszF64),
            ("libm::fminf", RustFloatMathIntrinsic::MinNumNszF32),
            ("libm::fmin", RustFloatMathIntrinsic::MinNumNszF64),
            ("libm::rintf", RustFloatMathIntrinsic::RoundevenF32),
            ("libm::rint", RustFloatMathIntrinsic::RoundevenF64),
            ("libm::roundevenf", RustFloatMathIntrinsic::RoundevenF32),
            ("libm::roundeven", RustFloatMathIntrinsic::RoundevenF64),
        ] {
            assert_eq!(
                RustFloatMathIntrinsic::from_core_path(path),
                Some(expected),
                "`{path}` did not map to the expected intrinsic"
            );
        }
    }

    /// Same anchoring requirement for the tuple-returning sincos detector.
    #[test]
    fn libm_sincos_detection_is_anchored_to_the_libm_crate_root() {
        assert_eq!(libm_sincos_is_f64("libm::sincosf"), Some(false));
        assert_eq!(libm_sincos_is_f64("libm::math::sincos::sincos"), Some(true));
        assert_eq!(libm_sincos_is_f64("my_app::libm_compat::sincosf"), None);
        assert_eq!(libm_sincos_is_f64("libmath::sincos"), None);
    }

    #[test]
    fn maxnum_minnum_nsz_placeholders_round_trip_through_dialect_mir() {
        // The placeholder names must match between this importer crate and
        // `dialect-mir::rust_intrinsics`. A drift here would manifest as a
        // missed lowering in `mir-lower`, so spot-check both sides.
        assert_eq!(
            RustFloatMathIntrinsic::MaxNumNszF32.placeholder_callee(),
            rust_intrinsics::CALLEE_MAXNUM_NSZ_F32
        );
        assert_eq!(
            RustFloatMathIntrinsic::MaxNumNszF64.placeholder_callee(),
            rust_intrinsics::CALLEE_MAXNUM_NSZ_F64
        );
        assert_eq!(
            RustFloatMathIntrinsic::MinNumNszF32.placeholder_callee(),
            rust_intrinsics::CALLEE_MINNUM_NSZ_F32
        );
        assert_eq!(
            RustFloatMathIntrinsic::MinNumNszF64.placeholder_callee(),
            rust_intrinsics::CALLEE_MINNUM_NSZ_F64
        );
    }
}
