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

pub use dialect_mir::rust_intrinsics::{RustFloatMathIntrinsic, is_libm_path, libm_sincos_is_f64};

/// Resolve generic math intrinsics using rustc's monomorphized float type.
pub fn from_core_path_with_substs(
    name: &str,
    type_substs: &[rustc_public::ty::Ty],
) -> Result<Option<RustFloatMathIntrinsic>, String> {
    use rustc_public::ty::{RigidTy, TyKind};

    if let Some(intrinsic) = RustFloatMathIntrinsic::from_core_path(name) {
        return Ok(Some(intrinsic));
    }
    let float_ty = match type_substs {
        [ty] => match ty.kind() {
            TyKind::RigidTy(RigidTy::Float(float_ty)) => Some(float_ty),
            _ => None,
        },
        _ => None,
    };
    from_generic_core_path(name, float_ty)
}

/// Map a generic `core::intrinsics::{sin,cos,exp,exp2,log,log2,log10}` path and its concrete float type.
fn from_generic_core_path(
    name: &str,
    float_ty: Option<rustc_public::ty::FloatTy>,
) -> Result<Option<RustFloatMathIntrinsic>, String> {
    use rustc_public::ty::FloatTy;

    // These exact declarations are `fn<T: FloatPrimitive>(T) -> T`.
    // Reuse the existing f32/f64 placeholders and libdevice lowerings;
    // unsupported widths must not be silently narrowed.
    let (f32_intrinsic, f64_intrinsic) = match name {
        "core::intrinsics::sin" | "std::intrinsics::sin" => (
            RustFloatMathIntrinsic::SinF32,
            RustFloatMathIntrinsic::SinF64,
        ),
        "core::intrinsics::cos" | "std::intrinsics::cos" => (
            RustFloatMathIntrinsic::CosF32,
            RustFloatMathIntrinsic::CosF64,
        ),
        "core::intrinsics::exp" | "std::intrinsics::exp" => (
            RustFloatMathIntrinsic::ExpF32,
            RustFloatMathIntrinsic::ExpF64,
        ),
        "core::intrinsics::exp2" | "std::intrinsics::exp2" => (
            RustFloatMathIntrinsic::Exp2F32,
            RustFloatMathIntrinsic::Exp2F64,
        ),
        "core::intrinsics::log" | "std::intrinsics::log" => (
            RustFloatMathIntrinsic::LogF32,
            RustFloatMathIntrinsic::LogF64,
        ),
        "core::intrinsics::log2" | "std::intrinsics::log2" => (
            RustFloatMathIntrinsic::Log2F32,
            RustFloatMathIntrinsic::Log2F64,
        ),
        "core::intrinsics::log10" | "std::intrinsics::log10" => (
            RustFloatMathIntrinsic::Log10F32,
            RustFloatMathIntrinsic::Log10F64,
        ),
        _ => return Ok(None),
    };
    match float_ty {
        Some(FloatTy::F32) => Ok(Some(f32_intrinsic)),
        Some(FloatTy::F64) => Ok(Some(f64_intrinsic)),
        _ => Err(format!(
            "generic float intrinsic `{name}` requires one f32 or f64 type argument; found {float_ty:?}"
        )),
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

    /// Ensure `frexp`/`frexpf` resolve only `libm` paths and reject other namespaces.
    #[test]
    fn frexp_paths_resolve_only_libm_entry_points() {
        use RustFloatMathIntrinsic as I;
        for (name, variant, placeholder) in [
            ("frexpf", I::FrexpF32, rust_intrinsics::CALLEE_FREXP_F32),
            ("frexp", I::FrexpF64, rust_intrinsics::CALLEE_FREXP_F64),
        ] {
            for path in [
                format!("libm::{name}"),
                format!("libm::math::{name}::{name}"),
            ] {
                assert_eq!(I::from_core_path(&path), Some(variant), "{path}");
                assert_eq!(variant.placeholder_callee(), placeholder);
                assert!(rust_intrinsics::is_libdevice_backed_placeholder(
                    placeholder
                ));
            }
            for root in [
                "user::libm",
                "libm_lookalike",
                "std::sys::cmath",
                "core::intrinsics",
            ] {
                let path = format!("{root}::{name}");
                assert_eq!(I::from_core_path(&path), None, "{path}");
            }
        }
        assert_eq!(I::from_core_path("libm::sincos"), None);
        assert_eq!(libm_sincos_is_f64("libm::sincos"), Some(true));
        assert_eq!(I::from_core_path("libm::ldexp"), Some(I::LdexpF64));
    }

    /// Ensure `modf`/`remquo` resolve only `libm` paths and reject other namespaces.
    #[test]
    fn out_parameter_pairs_resolve_only_libm_entry_points() {
        use RustFloatMathIntrinsic as I;
        for (name, variant, placeholder, argc) in [
            ("modff", I::ModfF32, rust_intrinsics::CALLEE_MODF_F32, 1),
            ("modf", I::ModfF64, rust_intrinsics::CALLEE_MODF_F64, 1),
            (
                "remquof",
                I::RemquoF32,
                rust_intrinsics::CALLEE_REMQUO_F32,
                2,
            ),
            (
                "remquo",
                I::RemquoF64,
                rust_intrinsics::CALLEE_REMQUO_F64,
                2,
            ),
        ] {
            assert_eq!(I::from_core_path(&format!("libm::{name}")), Some(variant));
            assert_eq!(
                I::from_core_path(&format!("libm::math::{name}::{name}")),
                Some(variant)
            );
            assert_eq!(variant.placeholder_callee(), placeholder);
            assert_eq!(variant.arg_count(), argc);
            assert_eq!(variant.scalar_signature(false), None);
            assert!(rust_intrinsics::is_libdevice_backed_placeholder(
                placeholder
            ));
            for root in ["user::libm", "libm_lookalike", "core::intrinsics"] {
                assert_eq!(I::from_core_path(&format!("{root}::{name}")), None);
            }
        }
    }

    /// Check that special math, `lgamma_r`, C23 min/max, and `j1` paths resolve to their placeholders.
    #[test]
    fn special_math_paths_resolve_to_typed_placeholders() {
        use RustFloatMathIntrinsic as I;
        for (name, intrinsic, callee) in [
            ("tgammaf", I::TgammaF32, rust_intrinsics::CALLEE_TGAMMA_F32),
            ("tgamma", I::TgammaF64, rust_intrinsics::CALLEE_TGAMMA_F64),
            ("ilogbf", I::IlogbF32, rust_intrinsics::CALLEE_ILOGB_F32),
            ("ilogb", I::IlogbF64, rust_intrinsics::CALLEE_ILOGB_F64),
            ("ldexpf", I::LdexpF32, rust_intrinsics::CALLEE_LDEXP_F32),
            ("ldexp", I::LdexpF64, rust_intrinsics::CALLEE_LDEXP_F64),
            ("scalbnf", I::LdexpF32, rust_intrinsics::CALLEE_LDEXP_F32),
            ("scalbn", I::LdexpF64, rust_intrinsics::CALLEE_LDEXP_F64),
        ] {
            for path in [
                format!("libm::{name}"),
                format!("libm::math::{name}::{name}"),
            ] {
                assert_eq!(I::from_core_path(&path), Some(intrinsic), "{path}");
                assert_eq!(intrinsic.placeholder_callee(), callee);
            }
            for path in [
                format!("my_app::libm_compat::{name}"),
                format!("my_app::libm::{name}"),
                format!("libm_lookalike::{name}"),
                format!("std::sys::cmath::{name}_unsupported"),
            ] {
                assert_eq!(I::from_core_path(&path), None, "{path}");
            }
        }
        assert_eq!(
            I::from_core_path("std::sys::cmath::tgammaf"),
            Some(I::TgammaF32)
        );
        assert_eq!(
            I::from_core_path("std::sys::cmath::tgamma"),
            Some(I::TgammaF64)
        );
        // Existing scalar and tuple dispatch must keep their identities.
        assert_eq!(I::from_core_path("libm::hypot"), Some(I::HypotF64));
        assert_eq!(
            I::from_core_path("core::intrinsics::powif32"),
            Some(I::PowiF32)
        );
        assert_eq!(libm_sincos_is_f64("libm::sincosf"), Some(false));
        for (name, intrinsic, callee) in [
            (
                "lgammaf_r",
                I::LgammaRF32,
                rust_intrinsics::CALLEE_LGAMMA_R_F32,
            ),
            (
                "lgamma_r",
                I::LgammaRF64,
                rust_intrinsics::CALLEE_LGAMMA_R_F64,
            ),
        ] {
            assert_eq!(I::from_core_path(&format!("libm::{name}")), Some(intrinsic));
            assert_eq!(
                I::from_core_path(&format!("libm::math::{name}::{name}")),
                Some(intrinsic)
            );
            assert_eq!(intrinsic.placeholder_callee(), callee);
            assert_eq!(intrinsic.arg_count(), 1);
            assert_eq!(intrinsic.scalar_signature(false), None);
            assert_eq!(I::from_core_path(&format!("user::libm::{name}")), None);
        }
        for (name, intrinsic, callee) in [
            (
                "fmaximumf",
                I::MaximumF32,
                rust_intrinsics::CALLEE_FMAXIMUM_F32,
            ),
            (
                "fmaximum",
                I::MaximumF64,
                rust_intrinsics::CALLEE_FMAXIMUM_F64,
            ),
            (
                "fminimumf",
                I::MinimumF32,
                rust_intrinsics::CALLEE_FMINIMUM_F32,
            ),
            (
                "fminimum",
                I::MinimumF64,
                rust_intrinsics::CALLEE_FMINIMUM_F64,
            ),
            (
                "fmaximum_numf",
                I::MaximumNumberF32,
                rust_intrinsics::CALLEE_FMAXIMUM_NUM_F32,
            ),
            (
                "fmaximum_num",
                I::MaximumNumberF64,
                rust_intrinsics::CALLEE_FMAXIMUM_NUM_F64,
            ),
            (
                "fminimum_numf",
                I::MinimumNumberF32,
                rust_intrinsics::CALLEE_FMINIMUM_NUM_F32,
            ),
            (
                "fminimum_num",
                I::MinimumNumberF64,
                rust_intrinsics::CALLEE_FMINIMUM_NUM_F64,
            ),
        ] {
            assert_eq!(I::from_core_path(&format!("libm::{name}")), Some(intrinsic));
            assert_eq!(
                I::from_core_path(&format!("libm::math::{name}::{name}")),
                Some(intrinsic)
            );
            assert_eq!(intrinsic.placeholder_callee(), callee);
            assert_eq!(intrinsic.arg_count(), 2);
            assert!(rust_intrinsics::is_known_placeholder(callee));
            assert!(!rust_intrinsics::is_libdevice_backed_placeholder(callee));
            assert_eq!(I::from_core_path(&format!("user::libm::{name}")), None);
        }
        for (name, intrinsic, callee, symbol) in [
            ("j1f", I::J1F32, rust_intrinsics::CALLEE_J1_F32, "__nv_j1f"),
            ("j1", I::J1F64, rust_intrinsics::CALLEE_J1_F64, "__nv_j1"),
        ] {
            assert_eq!(I::from_core_path(&format!("libm::{name}")), Some(intrinsic));
            assert_eq!(
                I::from_core_path(&format!("libm::math::{name}::{name}")),
                Some(intrinsic)
            );
            assert_eq!(intrinsic.placeholder_callee(), callee);
            assert_eq!(intrinsic.libdevice_symbol(false), Some(symbol));
            assert_eq!(intrinsic.arg_count(), 1);
            assert!(rust_intrinsics::is_libdevice_backed_placeholder(callee));
            assert_eq!(I::from_core_path(&format!("user::libm::{name}")), None);
        }
        assert_eq!(I::from_core_path("libm::tgammaf128"), None);
    }

    /// Verify that generic float intrinsics resolve `f32`/`f64` and reject unsupported widths.
    #[test]
    fn generic_float_intrinsics_resolve_only_the_declared_float_type() {
        use RustFloatMathIntrinsic as I;
        use rustc_public::ty::FloatTy;
        for (name, f32_variant, f64_variant) in [
            ("sin", I::SinF32, I::SinF64),
            ("cos", I::CosF32, I::CosF64),
            ("exp", I::ExpF32, I::ExpF64),
            ("exp2", I::Exp2F32, I::Exp2F64),
            ("log", I::LogF32, I::LogF64),
            ("log2", I::Log2F32, I::Log2F64),
            ("log10", I::Log10F32, I::Log10F64),
        ] {
            for root in ["core", "std"] {
                let path = format!("{root}::intrinsics::{name}");
                assert_eq!(
                    from_generic_core_path(&path, Some(FloatTy::F32)).unwrap(),
                    Some(f32_variant)
                );
                assert_eq!(
                    from_generic_core_path(&path, Some(FloatTy::F64)).unwrap(),
                    Some(f64_variant)
                );
                for unsupported in [None, Some(FloatTy::F16), Some(FloatTy::F128)] {
                    let error = from_generic_core_path(&path, unsupported).unwrap_err();
                    assert!(
                        error.contains("requires one f32 or f64 type argument"),
                        "{error}"
                    );
                }
            }
        }
        for path in [
            "user::intrinsics::exp",
            "user::core::intrinsics::sin",
            "core::intrinsics::exp_m1",
            "core::intrinsics::sqrt",
            "core::intrinsics::expf32",
        ] {
            assert_eq!(
                from_generic_core_path(path, Some(FloatTy::F32)).unwrap(),
                None,
                "{path}"
            );
        }
        assert_eq!(
            I::from_core_path("core::intrinsics::expf32"),
            Some(I::ExpF32)
        );
        assert_eq!(
            I::from_core_path("core::intrinsics::expf64"),
            Some(I::ExpF64)
        );
    }

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

    /// Verify that `MaxNumNsz*`/`MinNumNsz*` placeholders match `dialect-mir::rust_intrinsics`.
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
