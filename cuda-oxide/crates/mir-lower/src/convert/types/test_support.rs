/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared fixtures for the `types` submodule unit tests.

use dialect_mir::types::{MirStructType, StructAbiKind};
use llvm_export::types as llvm_types;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::Context;
use pliron::r#type::TypeHandle;

use super::layout::make_padding_type;

pub(super) fn make_ctx() -> Context {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    crate::register(&mut ctx);
    ctx
}

/// A MIR-level unsigned integer type (what the importer produces).
pub(super) fn mir_uint(ctx: &mut Context, width: u32) -> TypeHandle {
    IntegerType::get(ctx, width, Signedness::Unsigned).into()
}

/// A converted (signless) LLVM integer type.
pub(super) fn llvm_int(ctx: &mut Context, width: u32) -> TypeHandle {
    IntegerType::get(ctx, width, Signedness::Signless).into()
}

/// `[n x i8]` padding type, as `make_padding_type` builds it.
pub(super) fn pad(ctx: &mut Context, n: u64) -> TypeHandle {
    make_padding_type(ctx, n)
}

/// A zero-sized MIR struct (PhantomData shape).
pub(super) fn mir_zst(ctx: &mut Context) -> TypeHandle {
    MirStructType::get(ctx, "Phantom".into(), vec![], vec![]).into()
}

pub(super) fn struct_fields(ctx: &Context, ty: TypeHandle) -> Vec<TypeHandle> {
    ty.deref(ctx)
        .downcast_ref::<llvm_types::StructType>()
        .expect("expected an LLVM struct type")
        .fields()
        .collect()
}

pub(super) fn transparent_u32(ctx: &mut Context, name: &str) -> TypeHandle {
    let u32_ty = mir_uint(ctx, 32);
    MirStructType::get_with_full_layout_and_abi(
        ctx,
        name.into(),
        vec!["value".into()],
        vec![u32_ty],
        vec![0],
        vec![0],
        4,
        4,
        StructAbiKind::TransparentScalar,
    )
    .into()
}
