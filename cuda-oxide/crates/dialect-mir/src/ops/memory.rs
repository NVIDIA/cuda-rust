/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR memory operations.
//!
//! This module defines memory access and allocation operations for the MIR dialect.

use pliron::{
    builtin::{
        attributes::{BoolAttr, IntegerAttr},
        op_interfaces::{NOpdsInterface, NResultsInterface, OneOpdInterface, OneResultInterface},
        types::IntegerType,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    derive::op_interface_impl,
    identifier::Identifier,
    irbuild::{inserter::Inserter, rewriter::Rewriter},
    location::Located,
    op::Op,
    operation::Operation,
    opts::mem2reg::{
        AllocInfo, PromotableAllocationInterface, PromotableOpInterface, PromotableOpKind,
    },
    result::Error,
    r#type::{TypeHandle, Typed},
    value::Value,
    verify_err,
};
use pliron_derive::pliron_op;

use crate::attributes::{MirPointerKindAuthorityAttr, MutabilityAttr};
use crate::ops::constants::MirUndefOp;
use crate::ops::debug::debug_value_for_promoted_slot;
use crate::types::{MirPointerKind, MirPtrType, is_opaque_fn_pointer_type};

type PlironResult<T> = pliron::result::Result<T>;

fn bool_integer_attr(ctx: &mut Context, value: bool) -> IntegerAttr {
    let i1_ty = IntegerType::get(ctx, 1, pliron::builtin::types::Signedness::Signless);
    IntegerAttr::new(
        i1_ty,
        pliron::utils::apint::APInt::from_u64(
            u64::from(value),
            std::num::NonZeroUsize::new(1).unwrap(),
        ),
    )
}

// ============================================================================
// MirAllocaOp
// ============================================================================

/// MIR stack allocation operation.
///
/// Reserves a stack slot for a single value of the result's pointee type and
/// yields a pointer to it. The alloca's pointee type is carried as the result
/// pointer's pointee. Compiler-only provenance may be attached for diagnostics;
/// it is not part of the operation's semantics.
///
/// This op is the foundation of the alloca + load/store translator model: every
/// Rust MIR local is backed by an `mir.alloca` emitted in the function's entry
/// block, and defs/uses of that local become `mir.store`/`mir.load` on the
/// returned pointer. After the `mem2reg` pass promotes the alloca back to SSA,
/// these ops are erased.
///
/// # Operands
///
/// (none)
///
/// # Results
///
/// ```text
/// | Name  | Type       | Description                           |
/// |-------|------------|---------------------------------------|
/// | `ptr` | MirPtrType | Pointer to the newly-allocated slot   |
/// ```
///
/// # Verification
///
/// - Result must be a mutable, generic-address-space `MirPtrType` with Erased
///   provenance.
#[pliron_op(
    name = "mir.alloca",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirAllocaOp;

impl MirAllocaOp {
    /// Create a new `MirAllocaOp` wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirAllocaOp { op }
    }

    /// Return the pointee (element) type carried by the result pointer.
    pub fn pointee_type(&self, ctx: &Context) -> TypeHandle {
        let res_ty = self.get_operation().deref(ctx).get_result(0).get_type(ctx);
        let ty_ref = res_ty.deref(ctx);
        ty_ref
            .downcast_ref::<MirPtrType>()
            .expect("MirAllocaOp result must be MirPtrType (enforced by verifier)")
            .pointee
    }
}

impl Verify for MirAllocaOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res_ty = op.get_result(0).get_type(ctx);
        let res_ty_obj = res_ty.deref(ctx);
        let Some(ptr_ty) = res_ty_obj.downcast_ref::<MirPtrType>() else {
            return verify_err!(op.loc(), "MirAllocaOp result must be a MirPtrType");
        };
        if ptr_ty.kind != MirPointerKind::Erased
            || !ptr_ty.is_mutable
            || ptr_ty.address_space != crate::types::address_space::GENERIC
        {
            return verify_err!(
                op.loc(),
                "MirAllocaOp must return a mutable Erased pointer in generic address space"
            );
        }
        Ok(())
    }
}

#[op_interface_impl]
impl PromotableAllocationInterface for MirAllocaOp {
    fn alloc_info(&self, ctx: &Context) -> Vec<AllocInfo> {
        vec![AllocInfo {
            ptr: self.get_operation().deref(ctx).get_result(0),
            ty: self.pointee_type(ctx),
        }]
    }

    fn default_value(
        &self,
        ctx: &mut Context,
        inserter: &mut dyn Inserter,
        alloc_info: &AllocInfo,
    ) -> PlironResult<Value> {
        assert!(
            alloc_info.ptr == self.get_operation().deref(ctx).get_result(0),
            "AllocInfo does not belong to this MirAllocaOp"
        );
        let undef = MirUndefOp::new(ctx, alloc_info.ty);
        let undef_val = undef.get_operation().deref(ctx).get_result(0);
        inserter.insert_op(ctx, &undef);
        Ok(undef_val)
    }

    fn promote(
        &self,
        ctx: &mut Context,
        rewriter: &mut dyn Rewriter,
        alloc_infos: &[AllocInfo],
    ) -> PlironResult<()> {
        assert!(
            alloc_infos.len() == 1
                && alloc_infos[0].ptr == self.get_operation().deref(ctx).get_result(0),
            "AllocInfo does not belong to this MirAllocaOp"
        );
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}

// ============================================================================
// MirAssignOp
// ============================================================================

/// MIR assign operation (local = value).
///
/// Represents a simple assignment or move.
///
/// # Operands
///
/// ```text
/// | Name    | Type      |
/// |---------|-----------|
/// | `value` | Any type  |
/// ```
///
/// # Results
///
/// ```text
/// | Name   | Type                    |
/// |--------|-------------------------|
/// | `res`  | Same type as operand    |
/// ```
///
/// # Verification
///
/// - Operand type must match result type.
#[pliron_op(
    name = "mir.assign",
    format,
    interfaces = [NOpdsInterface<1>, OneOpdInterface, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirAssignOp;

impl MirAssignOp {
    /// Create a new MirAssignOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirAssignOp { op }
    }
}

impl Verify for MirAssignOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let opd = op.get_operand(0);
        let res = op.get_result(0);

        if opd.get_type(ctx) != res.get_type(ctx) {
            return verify_err!(op.loc(), "MirAssignOp operand and result types must match");
        }
        Ok(())
    }
}

// ============================================================================
// MirStoreOp
// ============================================================================

/// MIR store operation (*ptr = value).
///
/// Stores a value to a memory location.
///
/// # Operands
///
/// ```text
/// | Name    | Type         | Description              |
/// |---------|--------------|--------------------------|
/// | `ptr`   | MirPtrType   | Destination pointer      |
/// | `value` | Any type     | Value to store           |
/// ```
///
/// # Verification
///
/// - First operand must be a `MirPtrType`.
/// - Second operand type must match the pointer's pointee type.
#[pliron_op(
    name = "mir.store",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
    attributes = (
        mir_store_volatile: IntegerAttr
    )
)]
pub struct MirStoreOp;

impl MirStoreOp {
    /// Create a new MirStoreOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirStoreOp { op }
    }

    /// Destination pointer operand (operand 0).
    pub fn address_opd(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Value being stored (operand 1).
    pub fn value_opd(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    /// Whether this store carries volatile semantics.
    pub fn is_volatile(&self, ctx: &Context) -> bool {
        self.get_attr_mir_store_volatile(ctx)
            .is_some_and(|attr| attr.value().to_u64() != 0)
    }

    /// Mark this store as volatile.
    pub fn set_volatile(&self, ctx: &mut Context, volatile: bool) {
        let attr = bool_integer_attr(ctx, volatile);
        self.set_attr_mir_store_volatile(ctx, attr);
    }
}

impl Verify for MirStoreOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let dest = op.get_operand(0);
        let src = op.get_operand(1);

        let dest_ty = dest.get_type(ctx);
        let src_ty = src.get_type(ctx);

        let dest_ty_obj = dest_ty.deref(ctx);
        let ptr_ty = match dest_ty_obj.downcast_ref::<MirPtrType>() {
            Some(ty) => ty,
            None => return verify_err!(op.loc(), "MirStoreOp destination must be a MirPtrType"),
        };

        if src_ty != ptr_ty.pointee {
            return verify_err!(
                op.loc(),
                "MirStoreOp value type must match pointer element type"
            );
        }
        Ok(())
    }
}

#[op_interface_impl]
impl PromotableOpInterface for MirStoreOp {
    fn promotion_kind(&self, ctx: &Context, alloc_info: &AllocInfo) -> PromotableOpKind {
        if self.address_opd(ctx) == alloc_info.ptr {
            if self.is_volatile(ctx) {
                PromotableOpKind::NonPromotableUse
            } else {
                PromotableOpKind::Store(self.value_opd(ctx))
            }
        } else {
            PromotableOpKind::NonPromotableUse
        }
    }

    fn promote(
        &self,
        ctx: &mut Context,
        alloc_info_reaching_defs: &[(AllocInfo, Value)],
        rewriter: &mut dyn Rewriter,
    ) -> PlironResult<()> {
        assert!(
            alloc_info_reaching_defs.len() == 1
                && self.address_opd(ctx) == alloc_info_reaching_defs[0].0.ptr,
            "AllocInfo does not belong to this MirStoreOp"
        );
        let (alloc_info, reaching_def) = &alloc_info_reaching_defs[0];
        let loc = self.get_operation().deref(ctx).loc().clone();
        // If `reaching_def` is a block argument, we still record the debug
        // update at this promoted source op. Eager block-entry placement needs
        // its own source-location policy.
        if let Some(dbg_value) =
            debug_value_for_promoted_slot(ctx, alloc_info.ptr, *reaching_def, loc)
        {
            rewriter.insert_op(ctx, &dbg_value);
        }
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}

// ============================================================================
// MirMemcpyOp
// ============================================================================

/// MIR memcpy operation.
///
/// Copies `count` elements from `src` to `dst`. The count is element-count, not
/// byte-count, matching MIR's `CopyNonOverlapping` intrinsic statement.
///
/// # Operands
///
/// ```text
/// | Name    | Type         | Description                         |
/// |---------|--------------|-------------------------------------|
/// | `dst`   | MirPtrType   | Destination pointer                 |
/// | `src`   | MirPtrType   | Source pointer                      |
/// | `count` | Integer      | Number of pointee elements to copy  |
/// ```
///
/// # Attributes
///
/// ```text
/// | Name        | Type     | Description                                  |
/// |-------------|----------|----------------------------------------------|
/// | `memcpy_elem_type` | TypeAttr | The pointee type, stamped by [`Self::build`] |
/// ```
///
/// Why an attribute when the operands are typed? Lowering runs after the
/// operands are converted to opaque `llvm.ptr`, and operand type HISTORY
/// breaks across value-forwarded kind-only casts. The attribute rides the
/// op itself, so the byte count always scales by the right element:
///
/// ```text
/// build time:  dst: mir.ptr<T> ──► memcpy_elem_type = T  (stamped once)
/// lowering:    bytes = count * size_of(elem_type)  (no history lookup)
/// ```
///
/// # Verification
///
/// - Destination and source operands must be `MirPtrType`.
/// - Destination and source pointee types must match.
/// - The elem-type attribute must be present and equal that shared pointee.
/// - Count operand must be an integer.
#[pliron_op(
    name = "mir.memcpy",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
    attributes = (memcpy_elem_type: pliron::builtin::attributes::TypeAttr)
)]
pub struct MirMemcpyOp;

impl MirMemcpyOp {
    /// Create a new MirMemcpyOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirMemcpyOp { op }
    }

    /// Build a `mir.memcpy` with its element-type fact stamped from `dst`.
    ///
    /// The one construction door: callers cannot forget the fact, and the
    /// verifier keeps it equal to the pointee. Fails if `dst` is not a MIR
    /// pointer at build time.
    pub fn build(
        ctx: &mut Context,
        dst: Value,
        src: Value,
        count: Value,
    ) -> Result<Ptr<Operation>, Error> {
        build_mem_transfer::<Self>(ctx, dst, src, count, "mir.memcpy", "memcpy_elem_type")
    }
}

impl Verify for MirMemcpyOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_mem_transfer(
            ctx,
            self.get_operation(),
            self.get_attr_memcpy_elem_type(ctx).map(|a| a.get_type(ctx)),
            "MirMemcpyOp",
        )
    }
}

/// Shared builder for `mir.memcpy` / `mir.memmove`: stamp `elem_type` from
/// `dst`'s build-time pointer type.
fn build_mem_transfer<OpT: Op>(
    ctx: &mut Context,
    dst: Value,
    src: Value,
    count: Value,
    op_name: &str,
    elem_type_key: &str,
) -> Result<Ptr<Operation>, Error> {
    let dst_ty = dst.get_type(ctx);
    let pointee = {
        let dst_ty_ref = dst_ty.deref(ctx);
        dst_ty_ref
            .downcast_ref::<MirPtrType>()
            .map(|ptr_ty| ptr_ty.pointee)
    };
    let Some(pointee) = pointee else {
        return pliron::input_err_noloc!(
            "{op_name} destination must be a MirPtrType at build time"
        );
    };
    let op = Operation::new(
        ctx,
        OpT::get_concrete_op_info(),
        vec![],
        vec![dst, src, count],
        vec![],
        0,
    );
    let elem_attr = pliron::builtin::attributes::TypeAttr::new(pointee);
    let key: pliron::identifier::Identifier = elem_type_key.try_into().expect("valid identifier");
    op.deref_mut(ctx).attributes.set(key, elem_attr);
    Ok(op)
}

/// Shared verifier body for `mir.memcpy` / `mir.memmove`.
fn verify_mem_transfer(
    ctx: &Context,
    op: Ptr<Operation>,
    elem_type: Option<pliron::r#type::TypeHandle>,
    op_name: &str,
) -> Result<(), Error> {
    let op = &*op.deref(ctx);
    let dst_ty = op.get_operand(0).get_type(ctx);
    let src_ty = op.get_operand(1).get_type(ctx);
    let count_ty = op.get_operand(2).get_type(ctx);

    let dst_ty_ref = dst_ty.deref(ctx);
    let Some(dst_ptr_ty) = dst_ty_ref.downcast_ref::<MirPtrType>() else {
        return verify_err!(op.loc(), "{op_name} destination must be a MirPtrType");
    };
    let src_ty_ref = src_ty.deref(ctx);
    let Some(src_ptr_ty) = src_ty_ref.downcast_ref::<MirPtrType>() else {
        return verify_err!(op.loc(), "{op_name} source must be a MirPtrType");
    };
    if dst_ptr_ty.pointee != src_ptr_ty.pointee {
        return verify_err!(
            op.loc(),
            "{op_name} source and destination pointee types must match"
        );
    }
    let Some(elem_type) = elem_type else {
        return verify_err!(op.loc(), "{op_name} missing elem_type attribute");
    };
    if elem_type != dst_ptr_ty.pointee {
        return verify_err!(
            op.loc(),
            "{op_name} elem_type must equal the dst/src pointee"
        );
    }
    if count_ty.deref(ctx).downcast_ref::<IntegerType>().is_none() {
        return verify_err!(op.loc(), "{op_name} count must be an integer");
    }

    Ok(())
}

// ============================================================================
// MirMemmoveOp
// ============================================================================

/// MIR memmove operation.
///
/// Identical to [`MirMemcpyOp`] but the source and destination ranges may
/// overlap. Backs `core::intrinsics::copy` (`ptr::copy`); the non-overlapping
/// `copy_nonoverlapping` reaches MIR as a `CopyNonOverlapping` statement and
/// lowers to `MirMemcpyOp`. The count is element-count, not byte-count.
///
/// # Operands
///
/// ```text
/// | Name    | Type         | Description                         |
/// |---------|--------------|-------------------------------------|
/// | `dst`   | MirPtrType   | Destination pointer                 |
/// | `src`   | MirPtrType   | Source pointer                      |
/// | `count` | Integer      | Number of pointee elements to move  |
/// ```
///
/// # Attributes
///
/// Same element-type fact as [`MirMemcpyOp`] (`memmove_elem_type`), stamped by [`Self::build`],
/// read by lowering for the byte count. See the memcpy doc for the why.
///
/// # Verification
///
/// - Destination and source operands must be `MirPtrType`.
/// - Destination and source pointee types must match.
/// - The elem-type attribute must be present and equal that shared pointee.
/// - Count operand must be an integer.
#[pliron_op(
    name = "mir.memmove",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
    attributes = (memmove_elem_type: pliron::builtin::attributes::TypeAttr)
)]
pub struct MirMemmoveOp;

impl MirMemmoveOp {
    /// Create a new MirMemmoveOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirMemmoveOp { op }
    }

    /// Build a `mir.memmove` with its element-type fact stamped from `dst`.
    /// See [`MirMemcpyOp::build`].
    pub fn build(
        ctx: &mut Context,
        dst: Value,
        src: Value,
        count: Value,
    ) -> Result<Ptr<Operation>, Error> {
        build_mem_transfer::<Self>(ctx, dst, src, count, "mir.memmove", "memmove_elem_type")
    }
}

impl Verify for MirMemmoveOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_mem_transfer(
            ctx,
            self.get_operation(),
            self.get_attr_memmove_elem_type(ctx)
                .map(|a| a.get_type(ctx)),
            "MirMemmoveOp",
        )
    }
}

// ============================================================================
// MirLoadOp
// ============================================================================

/// MIR load operation (value = *ptr).
///
/// Loads a value from a memory location.
///
/// # Operands
///
/// ```text
/// | Name  | Type       | Description     |
/// |-------|------------|-----------------|
/// | `ptr` | MirPtrType | Source pointer  |
/// ```
///
/// # Results
///
/// ```text
/// | Name    | Type                     |
/// |---------|--------------------------|
/// | `value` | Pointer's pointee type   |
/// ```
///
/// # Verification
///
/// - Operand must be a `MirPtrType`.
/// - Result type must match the pointer's pointee type.
#[pliron_op(
    name = "mir.load",
    format,
    interfaces = [NOpdsInterface<1>, OneOpdInterface, NResultsInterface<1>, OneResultInterface],
    attributes = (
        mir_load_volatile: IntegerAttr
    )
)]
pub struct MirLoadOp;

impl MirLoadOp {
    /// Create a new MirLoadOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirLoadOp { op }
    }

    /// Source pointer operand (operand 0).
    pub fn address_opd(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Whether this load carries volatile semantics.
    pub fn is_volatile(&self, ctx: &Context) -> bool {
        self.get_attr_mir_load_volatile(ctx)
            .is_some_and(|attr| attr.value().to_u64() != 0)
    }

    /// Mark this load as volatile.
    pub fn set_volatile(&self, ctx: &mut Context, volatile: bool) {
        let attr = bool_integer_attr(ctx, volatile);
        self.set_attr_mir_load_volatile(ctx, attr);
    }
}

impl Verify for MirLoadOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let operand = op.get_operand(0);
        let operand_ty = operand.get_type(ctx);

        let operand_ty_obj = operand_ty.deref(ctx);
        let ptr_ty = match operand_ty_obj.downcast_ref::<MirPtrType>() {
            Some(ty) => ty,
            None => return verify_err!(op.loc(), "MirLoadOp operand must be a MirPtrType"),
        };

        let result = op.get_result(0);
        let result_ty = result.get_type(ctx);

        if result_ty != ptr_ty.pointee {
            return verify_err!(
                op.loc(),
                "MirLoadOp result type must match pointer element type"
            );
        }
        Ok(())
    }
}

#[op_interface_impl]
impl PromotableOpInterface for MirLoadOp {
    fn promotion_kind(&self, ctx: &Context, alloc_info: &AllocInfo) -> PromotableOpKind {
        if self.address_opd(ctx) == alloc_info.ptr {
            if self.is_volatile(ctx) {
                PromotableOpKind::NonPromotableUse
            } else {
                PromotableOpKind::Load
            }
        } else {
            PromotableOpKind::NonPromotableUse
        }
    }

    fn promote(
        &self,
        ctx: &mut Context,
        alloc_info_reaching_defs: &[(AllocInfo, Value)],
        rewriter: &mut dyn Rewriter,
    ) -> PlironResult<()> {
        assert!(
            alloc_info_reaching_defs.len() == 1
                && self.address_opd(ctx) == alloc_info_reaching_defs[0].0.ptr,
            "AllocInfo does not belong to this MirLoadOp"
        );
        let (alloc_info, reaching_def) = &alloc_info_reaching_defs[0];
        let loc = self.get_operation().deref(ctx).loc().clone();
        // If `reaching_def` is a block argument, we still record the debug
        // update at this promoted source op. Eager block-entry placement needs
        // its own source-location policy.
        if let Some(dbg_value) =
            debug_value_for_promoted_slot(ctx, alloc_info.ptr, *reaching_def, loc)
        {
            rewriter.insert_op(ctx, &dbg_value);
        }
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![*reaching_def]);
        Ok(())
    }
}

// ============================================================================
// MirRefOp
// ============================================================================

/// MIR reference operation (`&value` or `&mut value`).
///
/// Produces a pointer to an SSA value that has no pre-existing address. The
/// common "address of a MIR local" case is handled by reading the local's
/// alloca slot directly; `mir.ref` is reserved for values that never get a
/// slot — e.g. intermediate SSA aggregates, ZST placeholders, and the
/// closure-captures tuple constructed for `Fn*`-trait calls. Lowered by
/// `mir-lower` to `llvm.alloca` + `llvm.store` + the slot pointer.
///
/// # Operands
///
/// ```text
/// | Name    | Type     | Description                    |
/// |---------|----------|--------------------------------|
/// | `value` | Any type | The value to get a reference to |
/// ```
///
/// # Attributes
///
/// ```text
/// | Name      | Type            | Description                              |
/// |-----------|-----------------|------------------------------------------|
/// | `mutable` | MutabilityAttr  | Boolean: true for &mut, false for &      |
/// | `ref_pointer_kind_authority` | MirPointerKindAuthorityAttr | `Reborrow` for a Rust reference, `RawAddress` for a raw address, or `StaticAddress` for a typed constant/promoted address |
/// ```
///
/// # Results
///
/// ```text
/// | Name  | Type       | Description              |
/// |-------|------------|--------------------------|
/// | `ptr` | MirPtrType | Pointer to the value     |
/// ```
///
/// # Verification
///
/// - Result must be a `MirPtrType`.
/// - Result pointee type must match operand type.
/// - Result must be in generic address space.
/// - `Reborrow` produces exactly `SharedRef` / `UniqueRef` according to
///   `mutable`; `RawAddress` produces exactly `RawConst` / `RawMut`;
///   `StaticAddress` may produce `SharedRef` or a raw kind, but never
///   `UniqueRef`.
#[pliron_op(
    name = "mir.ref",
    format,
    interfaces = [NOpdsInterface<1>, OneOpdInterface, NResultsInterface<1>, OneResultInterface],
    attributes = (
        mutable: MutabilityAttr,
        ref_pointer_kind_authority: MirPointerKindAuthorityAttr
    )
)]
pub struct MirRefOp;

impl MirRefOp {
    /// Create a new MirRefOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirRefOp { op }
    }

    /// Check if this is a mutable reference.
    pub fn is_mutable(&self, ctx: &Context) -> bool {
        self.get_attr_mutable(ctx)
            .map(|attr| attr.0)
            .unwrap_or(false)
    }

    /// Set the mutable attribute.
    pub fn set_mutable(&self, ctx: &mut Context, mutable: bool) {
        self.set_attr_mutable(ctx, MutabilityAttr(mutable));
    }

    /// Classify the Rust address-creation boundary represented by this op.
    pub fn set_pointer_kind_authority(
        &self,
        ctx: &mut Context,
        authority: MirPointerKindAuthorityAttr,
    ) {
        self.set_attr_ref_pointer_kind_authority(ctx, authority);
    }
}

impl Verify for MirRefOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        // Get operand and result types
        let operand = op.get_operand(0);
        let operand_ty = operand.get_type(ctx);

        let result = op.get_result(0);
        let result_ty = result.get_type(ctx);

        // Result must be a pointer type
        let result_ty_obj = result_ty.deref(ctx);
        let ptr_ty = match result_ty_obj.downcast_ref::<MirPtrType>() {
            Some(ty) => ty,
            None => return verify_err!(op.loc(), "MirRefOp result must be a MirPtrType"),
        };

        // Pointee type must match operand type
        if ptr_ty.pointee != operand_ty {
            return verify_err!(
                op.loc(),
                "MirRefOp result pointee type must match operand type"
            );
        }

        let Some(mutable) = self.get_attr_mutable(ctx) else {
            return verify_err!(op.loc(), "MirRefOp must have a mutable attribute");
        };
        if ptr_ty.is_mutable != mutable.0 {
            return verify_err!(
                op.loc(),
                "MirRefOp mutable attribute must match its result pointer mutability"
            );
        }

        let authority = self
            .get_attr_ref_pointer_kind_authority(ctx)
            .map(|authority| authority.clone());
        // Verifier read-back: recomputes the expected kind from the op's own
        // mutability evidence to CHECK a mint, not to create one.
        #[allow(clippy::disallowed_methods)]
        let kind_is_authorized = match authority.as_ref() {
            Some(MirPointerKindAuthorityAttr::Reborrow) => {
                ptr_ty.kind == MirPointerKind::from_reference_mutability(mutable.0)
            }
            Some(MirPointerKindAuthorityAttr::RawAddress) => {
                ptr_ty.kind == MirPointerKind::from_raw_mutability(mutable.0)
            }
            Some(MirPointerKindAuthorityAttr::StaticAddress) => match ptr_ty.kind {
                MirPointerKind::SharedRef | MirPointerKind::RawConst => !mutable.0,
                MirPointerKind::RawMut => mutable.0,
                MirPointerKind::UniqueRef | MirPointerKind::Erased => false,
            },
            Some(authority) => {
                return verify_err!(
                    op.loc(),
                    "MirRefOp cannot use pointer-kind authority {:?}",
                    authority
                );
            }
            None => {
                return verify_err!(
                    op.loc(),
                    "MirRefOp requires Reborrow, RawAddress, or StaticAddress pointer-kind authority"
                );
            }
        };
        if !kind_is_authorized {
            return verify_err!(
                op.loc(),
                "MirRefOp authority {:?} with mutable: {} cannot produce result kind {:?}",
                authority,
                mutable.0,
                ptr_ty.kind
            );
        }
        if ptr_ty.address_space != crate::types::address_space::GENERIC {
            return verify_err!(
                op.loc(),
                "MirRefOp materializes stack storage and requires generic address space"
            );
        }

        Ok(())
    }
}

// ============================================================================
// MirPtrOffsetOp
// ============================================================================

/// MIR pointer offset operation (ptr + offset).
///
/// Computes a pointer offset: `base_ptr + offset * size_of(pointee)`.
///
/// # Operands
///
/// ```text
/// | Name     | Type        | Description      |
/// |----------|-------------|------------------|
/// | `base`   | MirPtrType  | Base pointer     |
/// | `offset` | IntegerType | Integer offset   |
/// ```
///
/// # Results
///
/// ```text
/// | Name     | Type       | Description                      |
/// |----------|------------|----------------------------------|
/// | `result` | MirPtrType | Pointer with same pointee as base |
/// ```
///
/// # Verification
///
/// - Base must be `MirPtrType`.
/// - Offset must be `IntegerType`.
/// - Result must be `MirPtrType` with same pointee as base.
#[pliron_op(
    name = "mir.ptr_offset",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirPtrOffsetOp;

const PTR_OFFSET_INBOUNDS_KEY: &str = "mir_ptr_offset_inbounds";

impl MirPtrOffsetOp {
    /// Create a new MirPtrOffsetOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirPtrOffsetOp { op }
    }

    /// Set whether this offset carries Rust's in-bounds pointer-arithmetic
    /// promise. Wrapping offsets must set this to false.
    pub fn set_inbounds(&self, ctx: &mut Context, inbounds: bool) {
        let key = Identifier::try_new(PTR_OFFSET_INBOUNDS_KEY.to_string())
            .expect("valid ptr-offset attribute key");
        self.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(key, BoolAttr::new(inbounds));
    }

    /// Return whether this offset may lower to LLVM `getelementptr inbounds`.
    ///
    /// Pointer offsets without an explicit attribute carry Rust's ordinary
    /// in-bounds contract, so absence defaults to true.
    pub fn is_inbounds(&self, ctx: &Context) -> bool {
        let key = Identifier::try_new(PTR_OFFSET_INBOUNDS_KEY.to_string())
            .expect("valid ptr-offset attribute key");
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<BoolAttr>(&key)
            .map(|attr| bool::from(attr.clone()))
            .unwrap_or(true)
    }
}

impl Verify for MirPtrOffsetOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        let base = op.get_operand(0);
        let offset = op.get_operand(1);

        let base_ty = base.get_type(ctx);
        let offset_ty = offset.get_type(ctx);

        // Check base is MirPtrType
        let base_ty_obj = base_ty.deref(ctx);
        let ptr_ty = match base_ty_obj.downcast_ref::<MirPtrType>() {
            Some(ty) => ty,
            None => return verify_err!(op.loc(), "MirPtrOffsetOp base must be MirPtrType"),
        };

        // Check offset is IntegerType
        let offset_ty_obj = offset_ty.deref(ctx);
        if offset_ty_obj.downcast_ref::<IntegerType>().is_none() {
            return verify_err!(op.loc(), "MirPtrOffsetOp offset must be IntegerType");
        }

        // Check result is MirPtrType with same pointee
        let res = op.get_result(0);
        let res_ty = res.get_type(ctx);
        let res_ty_obj = res_ty.deref(ctx);
        let res_ptr_ty = match res_ty_obj.downcast_ref::<MirPtrType>() {
            Some(ty) => ty,
            None => return verify_err!(op.loc(), "MirPtrOffsetOp result must be MirPtrType"),
        };

        if is_opaque_fn_pointer_type(ctx, res_ty) {
            return verify_err!(
                op.loc(),
                "MirPtrOffsetOp cannot produce the canonical function-pointer value carrier"
            );
        }

        if ptr_ty.pointee != res_ptr_ty.pointee {
            return verify_err!(
                op.loc(),
                "MirPtrOffsetOp result pointee type must match base pointee type"
            );
        }

        if !ptr_ty.kind.can_retype_generically_to(res_ptr_ty.kind) {
            return verify_err!(
                op.loc(),
                "MirPtrOffsetOp cannot change pointer kind from {:?} to {:?}",
                ptr_ty.kind,
                res_ptr_ty.kind
            );
        }
        if ptr_ty.is_mutable != res_ptr_ty.is_mutable {
            return verify_err!(op.loc(), "MirPtrOffsetOp must preserve pointer mutability");
        }
        if ptr_ty.address_space != res_ptr_ty.address_space {
            return verify_err!(
                op.loc(),
                "MirPtrOffsetOp must preserve pointer address space"
            );
        }

        Ok(())
    }
}

// ============================================================================
// MirSharedAllocOp
// ============================================================================

/// MIR shared memory allocation operation.
///
/// Represents an allocation in shared memory (CUDA `__shared__`).
/// This is lowered to an LLVM global with addrspace(3).
///
/// # Attributes
///
/// ```text
/// | Name            | Type        | Description                        |
/// |-----------------|-------------|------------------------------------|
/// | `elem_type`     | TypeAttr    | Element type of the array          |
/// | `size`          | IntegerAttr | Number of elements                 |
/// | `alloc_key`     | StringAttr  | Unique key for deduplication       |
/// | `source_name`   | StringAttr  | Optional Rust path of the originating `static` |
/// | `source_key`    | StringAttr  | Optional unique mangled identity of that `static` |
/// | `mir_alignment` | IntegerAttr | Optional alignment (natural if not set) |
/// ```
///
/// `source_name` is diagnostic only. Lowering mints an anonymous
/// `__shared_mem_N` symbol for every shared allocation, which leaves a
/// consumer inspecting the generated PTX unable to tell which Rust
/// `SharedArray` or `Barrier` static accounts for which block of shared
/// memory. Carrying the Rust path alongside the deduplication key lets
/// lowering stamp it onto the emitted LLVM global without perturbing the
/// symbol name itself. Nothing in code generation reads it.
///
/// # Results
///
/// ```text
/// | Name  | Type       | Description                              |
/// |-------|------------|------------------------------------------|
/// | `ptr` | MirPtrType | Pointer to shared memory (address space 3) |
/// ```
///
/// # Verification
///
/// - Must have `elem_type` and `size` attributes.
/// - Result must be a pointer type with shared address space (3).
/// - Result pointee type must exactly match `elem_type`.
#[pliron_op(
    name = "mir.shared_alloc",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>, OneResultInterface],
    attributes = (
        elem_type: pliron::builtin::attributes::TypeAttr,
        size: IntegerAttr,
        alloc_key: pliron::builtin::attributes::StringAttr,
        source_name: pliron::builtin::attributes::StringAttr,
        source_key: pliron::builtin::attributes::StringAttr,
        mir_alignment: IntegerAttr
    )
)]
pub struct MirSharedAllocOp;

impl MirSharedAllocOp {
    /// Create a new MirSharedAllocOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirSharedAllocOp { op }
    }

    /// Get alignment as u64 (returns None if not set, meaning natural alignment).
    pub fn get_alignment_value(&self, ctx: &Context) -> Option<u64> {
        self.get_attr_mir_alignment(ctx)
            .map(|attr| attr.value().to_u64())
    }

    /// Set alignment as u64.
    pub fn set_alignment_value(&self, ctx: &mut Context, alignment: u64) {
        use pliron::builtin::types::Signedness;
        let i64_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
        let align_attr = IntegerAttr::new(
            i64_ty,
            pliron::utils::apint::APInt::from_u64(
                alignment,
                std::num::NonZeroUsize::new(64).unwrap(),
            ),
        );
        self.set_attr_mir_alignment(ctx, align_attr);
    }
}

impl Verify for MirSharedAllocOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        // Check required attributes
        let Some(elem_type_attr) = self.get_attr_elem_type(ctx) else {
            return verify_err!(op.loc(), "MirSharedAllocOp missing elem_type attribute");
        };
        let elem_type = elem_type_attr.get_type(ctx);
        if self.get_attr_size(ctx).is_none() {
            return verify_err!(op.loc(), "MirSharedAllocOp missing size attribute");
        }

        // Check result is a shared memory pointer
        let res = op.get_result(0);
        let res_ty = res.get_type(ctx);
        let res_ty_obj = res_ty.deref(ctx);

        if let Some(ptr_ty) = res_ty_obj.downcast_ref::<MirPtrType>() {
            if ptr_ty.address_space != crate::types::address_space::SHARED {
                return verify_err!(
                    op.loc(),
                    "MirSharedAllocOp result must be in shared address space (3)"
                );
            }
            if ptr_ty.kind != MirPointerKind::Erased {
                return verify_err!(
                    op.loc(),
                    "MirSharedAllocOp creates storage and must return an Erased pointer kind"
                );
            }
            if ptr_ty.pointee != elem_type {
                return verify_err!(
                    op.loc(),
                    "MirSharedAllocOp result pointee type must match elem_type"
                );
            }
        } else {
            return verify_err!(op.loc(), "MirSharedAllocOp result must be a pointer type");
        }

        Ok(())
    }
}

// ============================================================================
// MirGlobalAllocOp
// ============================================================================

const RUST_STATIC_GLOBAL_KEY_PREFIX: &str = "cuda-oxide:rust-static:v1:";
const PROMOTED_GLOBAL_KEY_PREFIX: &str = "cuda-oxide:promoted:v1:";

fn encode_device_global_key(prefix: &str, payload: &str) -> String {
    format!("{prefix}{}:{payload}", payload.len())
}

/// Encode rustc's opaque static symbol as a domain-tagged lowering key.
///
/// The tag keeps a user-controlled `#[export_name]` from aliasing a
/// compiler-made promoted allocation with the same string fingerprint. The
/// length prefix makes the symbol payload unambiguous even when it contains
/// punctuation.
pub fn encode_rust_static_global_key(symbol: &str) -> String {
    encode_device_global_key(RUST_STATIC_GLOBAL_KEY_PREFIX, symbol)
}

/// Encode one compiler-made promoted allocation fingerprint.
pub fn encode_promoted_global_key(fingerprint: &str) -> String {
    encode_device_global_key(PROMOTED_GLOBAL_KEY_PREFIX, fingerprint)
}

/// Recover the exact rustc symbol payload from a tagged static key.
///
/// Constant-memory globals use the payload as their externally visible
/// linkage name; the tagged key itself remains internal to lowering and
/// relocation lookup.
pub fn rust_static_symbol_from_global_key(key: &str) -> Option<&str> {
    let encoded = key.strip_prefix(RUST_STATIC_GLOBAL_KEY_PREFIX)?;
    let (length, symbol) = encoded.split_once(':')?;
    (length.parse::<usize>().ok()? == symbol.len()).then_some(symbol)
}

/// MIR device-global address operation.
///
/// Represents the address of an ordinary Rust `static` / `static mut` reachable
/// from device code. Lowered to an LLVM global in CUDA global memory
/// (`addrspace(1)`) by default, or constant memory (`addrspace(4)`) when the
/// static was tagged `#[constant]`. The choice is reflected in the result
/// pointer's address space; the verifier accepts both.
///
/// # Attributes
///
/// ```text
/// | Name            | Type        | Description                      |
/// |-----------------|-------------|----------------------------------|
/// | `global_type`   | TypeAttr    | Type stored in the global        |
/// | `global_key`    | StringAttr  | Domain-tagged deduplication key  |
/// | `global_alignment` | IntegerAttr | Optional alignment            |
/// | `global_immutable` | UnitAttr | Storage is never written         |
/// ```
///
/// `global_immutable` is set only for storage this compiler materialises from an
/// evaluated Rust constant, never for a user `static` / `static mut`. It travels
/// to the LLVM global so the exporter can write `constant` instead of `global`.
/// It describes the *storage*, not a pointer: a shared reference to a mutable
/// static is an immutable pointer to mutable storage, and #413 records that
/// `MirPtrType::is_mutable` must not be read as a promise about the pointee.
///
/// # Results
///
/// ```text
/// | Name  | Type       | Description                             |
/// |-------|------------|-----------------------------------------|
/// | `ptr` | MirPtrType | Pointer to global memory (addrspace 1) |
/// ```
#[pliron_op(
    name = "mir.global_alloc",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>, OneResultInterface],
    attributes = (
        global_type: pliron::builtin::attributes::TypeAttr,
        global_key: pliron::builtin::attributes::StringAttr,
        global_alignment: IntegerAttr,
        global_immutable: pliron::builtin::attributes::UnitAttr
    )
)]
pub struct MirGlobalAllocOp;

impl MirGlobalAllocOp {
    /// Create a new `MirGlobalAllocOp` wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirGlobalAllocOp { op }
    }

    /// Get alignment as u64 (returns None if not set, meaning natural alignment).
    pub fn get_alignment_value(&self, ctx: &Context) -> Option<u64> {
        self.get_attr_global_alignment(ctx)
            .map(|attr| attr.value().to_u64())
    }

    /// Set alignment as u64.
    pub fn set_alignment_value(&self, ctx: &mut Context, alignment: u64) {
        use pliron::builtin::types::Signedness;
        let i64_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
        let align_attr = IntegerAttr::new(
            i64_ty,
            pliron::utils::apint::APInt::from_u64(
                alignment,
                std::num::NonZeroUsize::new(64).unwrap(),
            ),
        );
        self.set_attr_global_alignment(ctx, align_attr);
    }

    /// Declare that nothing ever writes this global's storage.
    ///
    /// Only the compiler's own promoted constants may claim this; see the op's
    /// attribute table.
    pub fn mark_immutable(&self, ctx: &mut Context) {
        self.set_attr_global_immutable(ctx, pliron::builtin::attributes::UnitAttr);
    }

    /// Whether this global was declared never-written.
    pub fn is_immutable(&self, ctx: &Context) -> bool {
        self.get_attr_global_immutable(ctx).is_some()
    }
}

impl Verify for MirGlobalAllocOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        if self.get_attr_global_type(ctx).is_none() {
            return verify_err!(op.loc(), "MirGlobalAllocOp missing global_type attribute");
        }
        if self.get_attr_global_key(ctx).is_none() {
            return verify_err!(op.loc(), "MirGlobalAllocOp missing global_key attribute");
        }

        let res = op.get_result(0);
        let res_ty = res.get_type(ctx);
        let res_ty_obj = res_ty.deref(ctx);

        if let Some(ptr_ty) = res_ty_obj.downcast_ref::<MirPtrType>() {
            let as_ = ptr_ty.address_space;
            if as_ != crate::types::address_space::GLOBAL
                && as_ != crate::types::address_space::CONSTANT
            {
                return verify_err!(
                    op.loc(),
                    "MirGlobalAllocOp result must be in global (1) or constant (4) address space"
                );
            }
            if ptr_ty.kind != MirPointerKind::Erased {
                return verify_err!(
                    op.loc(),
                    "MirGlobalAllocOp creates storage and must return an Erased pointer kind"
                );
            }
        } else {
            return verify_err!(op.loc(), "MirGlobalAllocOp result must be a pointer type");
        }

        Ok(())
    }
}

#[cfg(test)]
mod device_global_key_tests {
    use super::{
        encode_promoted_global_key, encode_rust_static_global_key,
        rust_static_symbol_from_global_key,
    };

    #[test]
    fn static_and_promoted_key_domains_cannot_alias() {
        let adversarial = "__cuda_oxide_promoted_type_collision:with:punctuation";
        let static_key = encode_rust_static_global_key(adversarial);
        let promoted_key = encode_promoted_global_key(adversarial);

        assert_ne!(static_key, promoted_key);
        assert_eq!(
            rust_static_symbol_from_global_key(&static_key),
            Some(adversarial)
        );
        assert_eq!(rust_static_symbol_from_global_key(&promoted_key), None);
    }

    #[test]
    fn malformed_static_key_length_fails_closed() {
        assert_eq!(
            rust_static_symbol_from_global_key("cuda-oxide:rust-static:v1:999:short"),
            None
        );
    }
}

// ============================================================================
// MirExternSharedOp
// ============================================================================

/// MIR extern shared memory reference operation.
///
/// Represents a reference to dynamically-sized shared memory (CUDA `extern __shared__`).
/// Unlike [`MirSharedAllocOp`], the size is not known at compile time - it's
/// specified at kernel launch via `LaunchConfig::shared_mem_bytes`.
///
/// All `MirExternSharedOp` instances in a kernel refer to the **same** underlying
/// memory, with different byte offsets for partitioning.
///
/// This is lowered to an LLVM global with extern linkage and addrspace(3):
/// ```llvm
/// @__dynamic_smem = external addrspace(3) global [0 x i8], align 256
/// ```
///
/// # Attributes
///
/// ```text
/// | Name            | Type        | Description                              |
/// |-----------------|-------------|------------------------------------------|
/// | `byte_offset`   | IntegerAttr | Byte offset from start of dynamic smem   |
/// | `mir_alignment` | IntegerAttr | Alignment hint (global always uses 256)  |
/// ```
///
/// # Results
///
/// ```text
/// | Name  | Type       | Description                                |
/// |-------|------------|--------------------------------------------|
/// | `ptr` | MirPtrType | Pointer to shared memory (address space 3) |
/// ```
///
/// # Verification
///
/// - Result must be a pointer type with shared address space (3).
#[pliron_op(
    name = "mir.extern_shared",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>, OneResultInterface],
    attributes = (
        extern_byte_offset: IntegerAttr,
        extern_alignment: IntegerAttr
    )
)]
pub struct MirExternSharedOp;

impl MirExternSharedOp {
    /// Create a new MirExternSharedOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirExternSharedOp { op }
    }

    /// Get byte offset as u64 (returns 0 if not set).
    pub fn get_byte_offset_value(&self, ctx: &Context) -> u64 {
        self.get_attr_extern_byte_offset(ctx)
            .map(|attr| attr.value().to_u64())
            .unwrap_or(0)
    }

    /// Set byte offset as u64.
    pub fn set_byte_offset_value(&self, ctx: &mut Context, offset: u64) {
        use pliron::builtin::types::Signedness;
        let i64_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
        let offset_attr = IntegerAttr::new(
            i64_ty,
            pliron::utils::apint::APInt::from_u64(offset, std::num::NonZeroUsize::new(64).unwrap()),
        );
        self.set_attr_extern_byte_offset(ctx, offset_attr);
    }

    /// Get alignment as u64 (returns 128 if not set).
    ///
    /// Note: The actual global alignment is fixed at 256 bytes in mir-lower,
    /// regardless of this attribute value.
    pub fn get_alignment_value(&self, ctx: &Context) -> u64 {
        self.get_attr_extern_alignment(ctx)
            .map(|attr| attr.value().to_u64())
            .unwrap_or(128)
    }

    /// Set alignment as u64.
    pub fn set_alignment_value(&self, ctx: &mut Context, alignment: u64) {
        use pliron::builtin::types::Signedness;
        let i64_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
        let align_attr = IntegerAttr::new(
            i64_ty,
            pliron::utils::apint::APInt::from_u64(
                alignment,
                std::num::NonZeroUsize::new(64).unwrap(),
            ),
        );
        self.set_attr_extern_alignment(ctx, align_attr);
    }
}

impl Verify for MirExternSharedOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        // Check result is a shared memory pointer
        let res = op.get_result(0);
        let res_ty = res.get_type(ctx);
        let res_ty_obj = res_ty.deref(ctx);

        if let Some(ptr_ty) = res_ty_obj.downcast_ref::<MirPtrType>() {
            if ptr_ty.address_space != crate::types::address_space::SHARED {
                return verify_err!(
                    op.loc(),
                    "MirExternSharedOp result must be in shared address space (3)"
                );
            }
            if ptr_ty.kind != MirPointerKind::Erased {
                return verify_err!(
                    op.loc(),
                    "MirExternSharedOp creates storage and must return an Erased pointer kind"
                );
            }
        } else {
            return verify_err!(op.loc(), "MirExternSharedOp result must be a pointer type");
        }

        Ok(())
    }
}

/// Register memory operations into the given context.
pub fn register(ctx: &mut Context) {
    MirAllocaOp::register(ctx);
    MirAssignOp::register(ctx);
    MirStoreOp::register(ctx);
    MirMemcpyOp::register(ctx);
    MirMemmoveOp::register(ctx);
    MirLoadOp::register(ctx);
    MirRefOp::register(ctx);
    MirPtrOffsetOp::register(ctx);
    MirSharedAllocOp::register(ctx);
    MirGlobalAllocOp::register(ctx);
    MirExternSharedOp::register(ctx);
}
