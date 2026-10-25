/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed native contracts and source effects retained through MIR preparation.

use crate::library::{
    LibraryAccess, LibraryCallSite, LibraryNativeMetadata, LibraryRoleAttr, LibraryScalar,
    LibraryWitness,
};
use crate::types::{
    MirDisjointSliceType, MirFP16Type, MirPointerKind, MirPtrType, MirSliceType, MirTupleType,
};
use pliron::{
    builtin::{
        attributes::{StringAttr, TypeAttr},
        type_interfaces::FunctionTypeInterface,
        types::{FP32Type, FP64Type, FunctionType, IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    result::Error,
    r#type::{TypeHandle, Typed, TypedHandle},
    value::Value,
    verify_err,
};
use pliron_derive::pliron_op;

/// A genuine source-native hook with original runtime operands and declaration.
///
/// Tuple flattening, scratch insertion and external ABI normalization happen only
/// after the prepared-MIR proof. This operation remains conservatively effectful.
#[pliron_op(name = "mir.library", format, attributes = (
    role: LibraryRoleAttr, site: LibraryCallSite, library_callee: StringAttr,
    library_signature: TypeAttr))]
pub struct MirLibraryOp;
impl MirLibraryOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    pub fn build_call(
        ctx: &mut Context,
        role: LibraryRoleAttr,
        site: LibraryCallSite,
        callee: String,
        signature: TypedHandle<FunctionType>,
        operands: Vec<Value>,
    ) -> Ptr<Operation> {
        let results = signature.deref(ctx).res_types();
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            results,
            operands,
            vec![],
            0,
        );
        let library = Self::new(op);
        library.set_attr_role(ctx, role);
        library.set_attr_site(ctx, site);
        library.set_attr_library_callee(ctx, StringAttr::new(callee));
        library.set_attr_library_signature(ctx, TypeAttr::new(signature.into()));
        op
    }
}
impl Verify for MirLibraryOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let Some(role) = self.get_attr_role(ctx) else {
            return verify_err!(op.loc(), "mir.library requires a role");
        };
        let Some(site) = self.get_attr_site(ctx) else {
            return verify_err!(op.loc(), "mir.library requires a source call site");
        };
        if site.caller.is_empty() {
            return verify_err!(op.loc(), "mir.library requires a nonempty caller identity");
        }
        let Some(callee) = self.get_attr_library_callee(ctx) else {
            return verify_err!(op.loc(), "mir.library requires a callee identity");
        };
        if String::from((*callee).clone()).is_empty() {
            return verify_err!(op.loc(), "mir.library requires a nonempty callee identity");
        }
        if op.num_regions() != 0 || op.get_num_successors() != 0 {
            return verify_err!(op.loc(), "mir.library has no regions or successors");
        }
        verify_call(ctx, self, *role)
    }
}

fn valid_scalar(scalar: LibraryScalar) -> bool {
    match scalar {
        LibraryScalar::Signed(bits) | LibraryScalar::Unsigned(bits) => {
            matches!(bits, 8 | 16 | 32 | 64)
        }
        LibraryScalar::Float(bits) => matches!(bits, 16 | 32 | 64),
    }
}

fn scalar_type(ctx: &Context, ty: TypeHandle) -> Option<LibraryScalar> {
    let ty = ty.deref(ctx);
    let scalar = if let Some(integer) = ty.downcast_ref::<IntegerType>() {
        let width = u16::try_from(integer.width()).ok()?;
        match integer.signedness() {
            Signedness::Signed => LibraryScalar::Signed(width),
            Signedness::Unsigned => LibraryScalar::Unsigned(width),
            Signedness::Signless => return None,
        }
    } else {
        let bits = if ty.is::<MirFP16Type>() {
            16
        } else if ty.is::<FP32Type>() {
            32
        } else if ty.is::<FP64Type>() {
            64
        } else {
            return None;
        };
        LibraryScalar::Float(bits)
    };
    valid_scalar(scalar).then_some(scalar)
}

fn pointer_scalar(ctx: &Context, ty: TypeHandle) -> Option<LibraryScalar> {
    let ty = ty.deref(ctx);
    let ptr = ty.downcast_ref::<MirPtrType>()?;
    if !matches!(
        (ptr.kind, ptr.is_mutable),
        (MirPointerKind::RawConst, false) | (MirPointerKind::RawMut, true)
    ) {
        return None;
    }
    scalar_type(ctx, ptr.pointee)
}

fn tuple_fields(ctx: &Context, ty: TypeHandle) -> Option<Vec<TypeHandle>> {
    ty.deref(ctx)
        .downcast_ref::<MirTupleType>()
        .map(|tuple| tuple.types.clone())
}

/// Match the retained value without prematurely normalizing an ABI pointer.
/// Only address-space differences are permitted; tuple layout remains exact.
fn retained_type_matches(ctx: &Context, actual: TypeHandle, declared: TypeHandle) -> bool {
    if actual == declared {
        return true;
    }
    let actual = actual.deref(ctx);
    let declared = declared.deref(ctx);
    if let (Some(a), Some(d)) = (
        actual.downcast_ref::<MirPtrType>(),
        declared.downcast_ref::<MirPtrType>(),
    ) {
        return a.pointee == d.pointee && a.kind == d.kind && a.is_mutable == d.is_mutable;
    }
    if let (Some(a), Some(d)) = (
        actual.downcast_ref::<MirTupleType>(),
        declared.downcast_ref::<MirTupleType>(),
    ) {
        return a.types.len() == d.types.len()
            && a.mem_to_decl == d.mem_to_decl
            && a.field_offsets == d.field_offsets
            && a.total_size == d.total_size
            && a.abi_align == d.abi_align
            && a.types
                .iter()
                .zip(&d.types)
                .all(|(&a, &d)| retained_type_matches(ctx, a, d));
    }
    false
}

fn witness_valid(
    ctx: &Context,
    witness: LibraryWitness,
    access: LibraryAccess,
    ty: TypeHandle,
    scalar: LibraryScalar,
) -> bool {
    let ty = ty.deref(ctx);
    match witness {
        LibraryWitness::ReadOnlySlice => ty.downcast_ref::<MirSliceType>().is_some_and(|slice| {
            access == LibraryAccess::Read
                && slice.kind == MirPointerKind::SharedRef
                && !slice.is_mutable
                && scalar_type(ctx, slice.element_ty) == Some(scalar)
        }),
        LibraryWitness::CollectiveCarrier(grain) => {
            let Some(ptr) = ty.downcast_ref::<MirPtrType>() else {
                return false;
            };
            let reference = matches!(
                (ptr.kind, ptr.is_mutable),
                (MirPointerKind::SharedRef, false) | (MirPointerKind::UniqueRef, true)
            );
            grain > 0
                && reference
                && (access == LibraryAccess::Read || ptr.kind == MirPointerKind::UniqueRef)
                && ptr
                    .pointee
                    .deref(ctx)
                    .downcast_ref::<MirDisjointSliceType>()
                    .is_some_and(|carrier| scalar_type(ctx, carrier.element_ty) == Some(scalar))
        }
    }
}

fn native_valid(ctx: &Context, native: LibraryNativeMetadata, args: &[TypeHandle]) -> bool {
    use LibraryNativeMetadata as N;
    let spec = native.spec();
    if !valid_scalar(spec.scalar) || spec.items == 0 {
        return false;
    }
    let (uniform_index, payload_index) = match native {
        N::Raw { block_threads, .. } => {
            return block_threads > 0
                && args.len() == 1
                && tuple_fields(ctx, args[0]).is_some_and(|fields| {
                    fields.into_iter().all(|ty| {
                        scalar_type(ctx, ty).is_some() || pointer_scalar(ctx, ty).is_some()
                    })
                });
        }
        N::CheckedBlock { .. } => {
            if args.len() != 2 {
                return false;
            }
            (0, 1)
        }
        N::CheckedBlockResource {
            access, witness, ..
        } => {
            if args.len() != 3 || !witness_valid(ctx, witness, access, args[0], spec.scalar) {
                return false;
            }
            (1, 2)
        }
    };
    let Some(uniform) = tuple_fields(ctx, args[uniform_index]) else {
        return false;
    };
    let Some(payload) = tuple_fields(ctx, args[payload_index]) else {
        return false;
    };
    payload.first().and_then(|&ty| pointer_scalar(ctx, ty)) == Some(spec.scalar)
        && payload.iter().all(|&ty| pointer_scalar(ctx, ty).is_some())
        && uniform.first().and_then(|&ty| pointer_scalar(ctx, ty)) == Some(spec.scalar)
        && uniform[1..]
            .iter()
            .all(|&ty| scalar_type(ctx, ty).is_some())
}

fn verify_call(ctx: &Context, call: &MirLibraryOp, role: LibraryRoleAttr) -> Result<(), Error> {
    let op = call.get_operation().deref(ctx);
    let Some(signature) = call.get_attr_library_signature(ctx) else {
        return verify_err!(
            op.loc(),
            "native mir.library requires its runtime signature"
        );
    };
    let signature = signature.get_type(ctx);
    let signature = signature.deref(ctx);
    let Some(signature) = signature.downcast_ref::<FunctionType>() else {
        return verify_err!(op.loc(), "library_signature must be a FunctionType");
    };
    let args = signature.arg_types();
    let results = signature.res_types();
    if args.len() != op.get_num_operands()
        || results.len() != op.get_num_results()
        || !args
            .iter()
            .enumerate()
            .all(|(i, &ty)| retained_type_matches(ctx, op.get_operand(i).get_type(ctx), ty))
        || !results
            .iter()
            .enumerate()
            .all(|(i, &ty)| op.get_result(i).get_type(ctx) == ty)
    {
        return verify_err!(
            op.loc(),
            "mir.library operands/results do not match the retained runtime signature"
        );
    }
    let LibraryRoleAttr::Native(native) = role;
    if !results.is_empty() || !native_valid(ctx, native, &args) {
        return verify_err!(
            op.loc(),
            "mir.library runtime signature or metadata does not match its native role"
        );
    }
    Ok(())
}

/// Retain an expanded source call's effect and, when present, its actual result.
///
/// This proof event is effectful until validation removes it. A result operand
/// is observed before destination forwarding, including in Full slot mode.
#[pliron_op(name = "mir.library_source", format,
    attributes = (library_source_effect: crate::library::LibrarySourceEffectAttr))]
pub struct MirLibrarySourceOp;

impl MirLibrarySourceOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
    pub fn build(
        ctx: &mut Context,
        effect: crate::library::LibrarySourceEffectAttr,
        result: Option<Value>,
    ) -> Ptr<Operation> {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            result.into_iter().collect(),
            vec![],
            0,
        );
        Self::new(op).set_attr_library_source_effect(ctx, effect);
        op
    }
}
impl Verify for MirLibrarySourceOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if self.get_attr_library_source_effect(ctx).is_none()
            || op.get_num_results() != 0
            || op.get_num_operands() > 1
        {
            return verify_err!(
                op.loc(),
                "source event requires identity, zero results and at most its actual result operand"
            );
        }
        Ok(())
    }
}

/// Retain a source mutation check on actual evaluated destination addresses.
#[pliron_op(name = "mir.library_write_check", format,
    attributes = (library_write_source: crate::library::LibraryWriteSourceAttr))]
pub struct MirLibraryWriteCheckOp;

impl MirLibraryWriteCheckOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
    pub fn build(
        ctx: &mut Context,
        source: crate::library::LibraryWriteSourceAttr,
        addresses: Vec<Value>,
    ) -> Ptr<Operation> {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            addresses,
            vec![],
            0,
        );
        Self::new(op).set_attr_library_write_source(ctx, source);
        op
    }
}
impl Verify for MirLibraryWriteCheckOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if self.get_attr_library_write_source(ctx).is_none() || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "write check requires original source identity and zero results"
            );
        }
        Ok(())
    }
}

pub(crate) fn register(ctx: &mut Context) {
    MirLibrarySourceOp::register(ctx);
    MirLibraryWriteCheckOp::register(ctx);
    MirLibraryOp::register(ctx);
}
