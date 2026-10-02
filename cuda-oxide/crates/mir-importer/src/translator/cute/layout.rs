/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Monomorphized substs -> layout extraction.
//!
//! `cute-rs` encodes static tile parameters as generics (`load_tile::<T, N>`);
//! by MIR-import time they are concrete in the callee's substs. This module
//! reads them out STRUCTURALLY (const values only exist as allocation bytes
//! in the Debug rendering, so substring matching cannot see them) and hands
//! them to the op emitters.

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::types::translate_type;
use pliron::context::Context;
use pliron::input_err;
use pliron::location::Location;
use pliron::r#type::TypeHandle;
use rustc_public::mir;

/// Read a monomorphized unsigned const without truncating it.
///
/// Rust's stable compiler API exposes some constants as raw allocation
/// bytes. Convert through `u128` and then check the destination width; `as`
/// would silently turn an invalid large layout constant into a different one.
pub(super) fn const_u64(
    c: &rustc_public::ty::TyConst,
    what: &str,
    loc: &Location,
) -> TranslationResult<u64> {
    use rustc_public::ty::TyConstKind;
    let raw = match c.kind() {
        TyConstKind::Value(_, alloc) => match alloc.read_uint() {
            Ok(value) => value,
            Err(error) => {
                return input_err!(
                    loc.clone(),
                    TranslationErr::unsupported(format!(
                        "{what} const generic could not be read: {error:?}"
                    ))
                );
            }
        },
        _ => match c.eval_target_usize() {
            Ok(value) => u128::from(value),
            Err(error) => {
                return input_err!(
                    loc.clone(),
                    TranslationErr::unsupported(format!(
                        "{what} const generic could not be evaluated: {error:?}"
                    ))
                );
            }
        },
    };
    match u64::try_from(raw) {
        Ok(value) => Ok(value),
        Err(_) => input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!("{what} const generic {raw} does not fit in u64"))
        ),
    }
}

/// Extract `(T, N)` from a call to a cute-rs tile function of shape
/// `fn f<T, const N: usize>(...)`: substs position 0 is the element type,
/// position 1 the tile length. Errors loudly on anything unexpected; a
/// silent default would hide a real compiler gap.
pub(crate) fn tile_elem_and_count(
    ctx: &mut Context,
    func: &mir::Operand,
    loc: &Location,
) -> TranslationResult<(TypeHandle, u64)> {
    let mir::Operand::Constant(const_op) = func else {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "cute-rs tile call through a non-constant callee (indirect call?)".to_string()
            )
        );
    };
    let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnDef(_, substs)) =
        const_op.const_.ty().kind()
    else {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported("cute-rs tile callee is not a FnDef".to_string())
        );
    };

    let Some(rustc_public::ty::GenericArgKind::Type(elem_rust_ty)) = substs.0.first() else {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "cute-rs tile call missing element type generic at substs[0]".to_string()
            )
        );
    };
    let elem = translate_type(ctx, elem_rust_ty)?;

    let Some(rustc_public::ty::GenericArgKind::Const(c)) = substs.0.get(1) else {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "cute-rs tile call missing tile-length const generic at substs[1]".to_string()
            )
        );
    };
    let n = const_u64(c, "cute-rs tile length", loc)?;
    if n == 0 || n > i64::MAX as u64 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "cute-rs tile length must be in 1..={}, got {n}",
                i64::MAX
            ))
        );
    }

    Ok((elem, n))
}

/// Extract `D` from `assume_div::<D>`, checking the promise is meaningful.
pub(crate) fn assume_divisor(func: &mir::Operand, loc: &Location) -> TranslationResult<u64> {
    let mir::Operand::Constant(const_op) = func else {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported("assume_div callee is not constant".to_string())
        );
    };
    let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnDef(_, substs)) =
        const_op.const_.ty().kind()
    else {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported("assume_div callee is not a FnDef".to_string())
        );
    };
    let Some(rustc_public::ty::GenericArgKind::Const(divisor)) = substs.0.first() else {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "assume_div is missing divisor const generic at substs[0]".to_string()
            )
        );
    };
    let divisor = const_u64(divisor, "assume_div divisor", loc)?;
    if divisor == 0 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported("assume_div divisor must be greater than zero".to_string())
        );
    }
    Ok(divisor)
}
