/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Atomic operations for GPU memory.
//!
//! These operations represent atomic reads, writes, and read-modify-writes on
//! GPU memory. They carry **ordering** and **scope** attributes so that
//! downstream lowering can emit the correct LLVM IR (and ultimately PTX).
//!
//! # Operations
//!
//! ```text
//! ┌─────────────────────────┬─────────────────────────────────────────────────┐
//! │ Op                      │ LLVM IR it lowers to                            │
//! ├─────────────────────────┼─────────────────────────────────────────────────┤
//! │ NvvmAtomicLoadOp        │ load atomic <ty>, ptr %p syncscope(...) <ord>   │
//! │ NvvmAtomicStoreOp       │ store atomic <ty> %v, ptr %p syncscope(...) ... │
//! │ NvvmAtomicRmwOp         │ atomicrmw <op> ptr %p, <ty> %v syncscope(...)   │
//! │ NvvmAtomicCmpxchgOp     │ cmpxchg ptr %p, <ty> %cmp, <ty> %new ...       │
//! └─────────────────────────┴─────────────────────────────────────────────────┘
//! ```
//!
//! # Op Interface: `NvvmAtomicOpInterface`
//!
//! All four ops implement [`NvvmAtomicOpInterface`], which provides uniform
//! access to `ordering()`, `scope()`, and `ptr_operand()`. This lets
//! mir-lower handle ordering-to-fence and scope-to-syncscope mapping through
//! one code path.

use dialect_mir::types::{MirFP16Type, MirPtrType};
use pliron::{
    attribute::Attribute,
    builtin::op_interfaces::{
        NOpdsInterface, NResultsInterface, OneOpdInterface, OneResultInterface,
    },
    builtin::types::{FP32Type, FP64Type, IntegerType, Signedness},
    common_traits::Verify,
    context::{Context, Ptr},
    derive::{op_interface, op_interface_impl},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::{TypeHandle, Typed},
    value::Value,
    verify_err,
};
use pliron_derive::{pliron_attr, pliron_op};

// =============================================================================
// Attribute Enums
// =============================================================================

/// Memory ordering for atomic operations.
///
/// Maps to LLVM orderings and ultimately to PTX ordering qualifiers:
///
/// | Variant   | LLVM         | PTX          |
/// |-----------|--------------|--------------|
/// | Relaxed   | `monotonic`  | `.relaxed`   |
/// | Acquire   | `acquire`    | `.acquire`   |
/// | Release   | `release`    | `.release`   |
/// | AcqRel    | `acq_rel`    | `.acq_rel`   |
/// | SeqCst    | `seq_cst`    | `fence.sc +` |
#[pliron_attr(name = "nvvm.atomic_ordering", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum AtomicOrdering {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

/// Scope of the atomic operation -- which threads observe it.
///
/// | Variant | LLVM syncscope    | PTX    |
/// |---------|-------------------|--------|
/// | Device  | `"device"`        | `.gpu` |
/// | Block   | `"block"`         | `.cta` |
/// | System  | (default / `""`)  | `.sys` |
#[pliron_attr(name = "nvvm.atomic_scope", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum AtomicScope {
    Device,
    Block,
    System,
}

/// Kind of read-modify-write operation.
///
/// These map 1:1 to LLVM `atomicrmw` operation keywords.
#[pliron_attr(name = "nvvm.atomic_rmw_kind", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum AtomicRmwKind {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Xchg,
    Min,
    Max,
    UMin,
    UMax,
    FAdd,
}

// =============================================================================
// Op Interface
// =============================================================================

/// Shared interface for all NVVM atomic operations.
///
/// Provides uniform access to ordering and scope so that mir-lower can
/// handle the ordering->fence and scope->syncscope mapping through a
/// single code path, regardless of which atomic op it is.
#[op_interface]
pub trait NvvmAtomicOpInterface {
    /// The memory ordering for this atomic operation.
    fn ordering(&self, ctx: &Context) -> AtomicOrdering;

    /// The scope (which threads observe the atomic).
    fn scope(&self, ctx: &Context) -> AtomicScope;

    /// The pointer operand.
    ///
    /// This is operand 1 for stores and operand 0 for the other atomic ops.
    fn ptr_operand(&self, ctx: &Context) -> Value;

    fn verify(_op: &dyn Op, _ctx: &Context) -> pliron::result::Result<()>
    where
        Self: Sized,
    {
        // Structural verification is done by each op's own Verify impl.
        Ok(())
    }
}

fn is_integer_atomic_type(ctx: &Context, ty: TypeHandle) -> bool {
    ty.deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| matches!(integer.width(), 32 | 64))
}

fn is_float_atomic_type(ctx: &Context, ty: TypeHandle) -> bool {
    let ty = ty.deref(ctx);
    ty.downcast_ref::<MirFP16Type>().is_some()
        || ty.downcast_ref::<FP32Type>().is_some()
        || ty.downcast_ref::<FP64Type>().is_some()
}

fn is_pointer_atomic_type(ctx: &Context, ty: TypeHandle) -> bool {
    ty.deref(ctx)
        .downcast_ref::<MirPtrType>()
        .is_some_and(|pointer| pointer.address_space == dialect_mir::types::address_space::GENERIC)
}

fn is_atomic_value_type(ctx: &Context, ty: TypeHandle) -> bool {
    is_integer_atomic_type(ctx, ty)
        || is_float_atomic_type(ctx, ty)
        || is_pointer_atomic_type(ctx, ty)
}

fn verify_atomic_pointer(
    ctx: &Context,
    op: &Operation,
    operand: usize,
    name: &str,
) -> Result<(), Error> {
    let pointer_ty = op.get_operand(operand).get_type(ctx);
    let pointer_ty = pointer_ty.deref(ctx);
    if pointer_ty.downcast_ref::<MirPtrType>().is_none() {
        return verify_err!(op.loc(), "{name} address must be a MIR pointer");
    }
    Ok(())
}

fn verify_atomic_value_type(
    ctx: &Context,
    op: &Operation,
    ty: TypeHandle,
    name: &str,
) -> Result<(), Error> {
    if !is_atomic_value_type(ctx, ty) {
        return verify_err!(op.loc(), "{name} has an unsupported value type");
    }
    Ok(())
}

fn verify_rmw_kind(ctx: &Context, ty: TypeHandle, kind: &AtomicRmwKind) -> bool {
    let ty_obj = ty.deref(ctx);
    let integer = ty_obj.downcast_ref::<IntegerType>();
    match kind {
        AtomicRmwKind::FAdd => is_float_atomic_type(ctx, ty),
        AtomicRmwKind::Xchg => is_atomic_value_type(ctx, ty),
        AtomicRmwKind::Min | AtomicRmwKind::Max => integer.is_some_and(|integer| {
            matches!(integer.width(), 32 | 64) && integer.signedness() == Signedness::Signed
        }),
        AtomicRmwKind::UMin | AtomicRmwKind::UMax => integer.is_some_and(|integer| {
            matches!(integer.width(), 32 | 64) && integer.signedness() == Signedness::Unsigned
        }),
        AtomicRmwKind::Add
        | AtomicRmwKind::Sub
        | AtomicRmwKind::And
        | AtomicRmwKind::Or
        | AtomicRmwKind::Xor => is_integer_atomic_type(ctx, ty),
    }
}

// =============================================================================
// NvvmAtomicLoadOp
// =============================================================================

/// Atomic load from GPU memory.
///
/// # Operands
///
/// - `ptr`: pointer to the value to load
///
/// # Results
///
/// - loaded value (i32, i64, f16, f32, f64, or a generic pointer)
///
/// # Attributes
///
/// - `ordering`: `Relaxed`, `Acquire`, or `SeqCst`
/// - `scope`: `Device`, `Block`, or `System`
#[pliron_op(
    name = "nvvm.atomic_load",
    format,
    interfaces = [NResultsInterface<1>, OneResultInterface, NOpdsInterface<1>, OneOpdInterface],
    attributes = (nvvm_ld_ordering: AtomicOrdering, nvvm_ld_scope: AtomicScope)
)]
pub struct NvvmAtomicLoadOp;

impl NvvmAtomicLoadOp {
    /// Create a new atomic load op wrapping an existing operation.
    pub fn new(op: Ptr<Operation>) -> Self {
        NvvmAtomicLoadOp { op }
    }

    /// Create a new atomic load from scratch.
    pub fn build(
        ctx: &mut Context,
        ptr: Value,
        result_ty: pliron::r#type::TypeHandle,
        ordering: AtomicOrdering,
        scope: AtomicScope,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty],
            vec![ptr],
            vec![],
            0,
        );
        let this = NvvmAtomicLoadOp { op };
        this.set_attr_nvvm_ld_ordering(ctx, ordering);
        this.set_attr_nvvm_ld_scope(ctx, scope);
        this
    }
}

impl Verify for NvvmAtomicLoadOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "nvvm.atomic_load requires one operand and one result"
            );
        }
        verify_atomic_pointer(ctx, &op, 0, "nvvm.atomic_load")?;
        verify_atomic_value_type(ctx, &op, op.get_result(0).get_type(ctx), "nvvm.atomic_load")?;
        let Some(ordering) = self.get_attr_nvvm_ld_ordering(ctx) else {
            return verify_err!(op.loc(), "nvvm.atomic_load requires an ordering");
        };
        if !matches!(
            &*ordering,
            AtomicOrdering::Relaxed | AtomicOrdering::Acquire | AtomicOrdering::SeqCst
        ) {
            return verify_err!(op.loc(), "nvvm.atomic_load has an invalid ordering");
        }
        if self.get_attr_nvvm_ld_scope(ctx).is_none() {
            return verify_err!(op.loc(), "nvvm.atomic_load requires a scope");
        }
        Ok(())
    }
}

#[op_interface_impl]
impl NvvmAtomicOpInterface for NvvmAtomicLoadOp {
    fn ordering(&self, ctx: &Context) -> AtomicOrdering {
        self.get_attr_nvvm_ld_ordering(ctx)
            .expect("NvvmAtomicLoadOp missing ordering attribute")
            .clone()
    }

    fn scope(&self, ctx: &Context) -> AtomicScope {
        self.get_attr_nvvm_ld_scope(ctx)
            .expect("NvvmAtomicLoadOp missing scope attribute")
            .clone()
    }

    fn ptr_operand(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

// =============================================================================
// NvvmAtomicStoreOp
// =============================================================================

/// Atomic store to GPU memory.
///
/// # Operands
///
/// - `val`: value to store
/// - `ptr`: pointer to store to
///
/// # Results
///
/// None.
///
/// # Attributes
///
/// - `ordering`: `Relaxed`, `Release`, or `SeqCst`
/// - `scope`: `Device`, `Block`, or `System`
#[pliron_op(
    name = "nvvm.atomic_store",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
    attributes = (nvvm_st_ordering: AtomicOrdering, nvvm_st_scope: AtomicScope)
)]
pub struct NvvmAtomicStoreOp;

impl NvvmAtomicStoreOp {
    /// Create a new atomic store op wrapping an existing operation.
    pub fn new(op: Ptr<Operation>) -> Self {
        NvvmAtomicStoreOp { op }
    }

    /// Create a new atomic store from scratch.
    pub fn build(
        ctx: &mut Context,
        val: Value,
        ptr: Value,
        ordering: AtomicOrdering,
        scope: AtomicScope,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![val, ptr],
            vec![],
            0,
        );
        let this = NvvmAtomicStoreOp { op };
        this.set_attr_nvvm_st_ordering(ctx, ordering);
        this.set_attr_nvvm_st_scope(ctx, scope);
        this
    }

    /// Get the value operand.
    pub fn value_opd(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Get the pointer operand.
    pub fn address_opd(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

impl Verify for NvvmAtomicStoreOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "nvvm.atomic_store requires two operands and no results"
            );
        }
        verify_atomic_pointer(ctx, &op, 1, "nvvm.atomic_store")?;
        verify_atomic_value_type(
            ctx,
            &op,
            op.get_operand(0).get_type(ctx),
            "nvvm.atomic_store",
        )?;
        let Some(ordering) = self.get_attr_nvvm_st_ordering(ctx) else {
            return verify_err!(op.loc(), "nvvm.atomic_store requires an ordering");
        };
        if !matches!(
            &*ordering,
            AtomicOrdering::Relaxed | AtomicOrdering::Release | AtomicOrdering::SeqCst
        ) {
            return verify_err!(op.loc(), "nvvm.atomic_store has an invalid ordering");
        }
        if self.get_attr_nvvm_st_scope(ctx).is_none() {
            return verify_err!(op.loc(), "nvvm.atomic_store requires a scope");
        }
        Ok(())
    }
}

#[op_interface_impl]
impl NvvmAtomicOpInterface for NvvmAtomicStoreOp {
    fn ordering(&self, ctx: &Context) -> AtomicOrdering {
        self.get_attr_nvvm_st_ordering(ctx)
            .expect("NvvmAtomicStoreOp missing ordering attribute")
            .clone()
    }

    fn scope(&self, ctx: &Context) -> AtomicScope {
        self.get_attr_nvvm_st_scope(ctx)
            .expect("NvvmAtomicStoreOp missing scope attribute")
            .clone()
    }

    fn ptr_operand(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

// =============================================================================
// NvvmAtomicFenceOp
// =============================================================================

/// Memory fence with explicit ordering and visibility scope.
///
/// # Operands
///
/// None.
///
/// # Results
///
/// None.
///
/// # Attributes
///
/// - `ordering`: `Acquire`, `Release`, `AcqRel`, or `SeqCst`
/// - `scope`: `Device`, `Block`, or `System`
#[pliron_op(
    name = "nvvm.atomic_fence",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (nvvm_fence_ordering: AtomicOrdering, nvvm_fence_scope: AtomicScope)
)]
pub struct NvvmAtomicFenceOp;

impl NvvmAtomicFenceOp {
    /// Wrap an existing operation as an atomic fence op.
    pub fn new(op: Ptr<Operation>) -> Self {
        NvvmAtomicFenceOp { op }
    }

    /// Create a new atomic fence from scratch.
    pub fn build(ctx: &mut Context, ordering: AtomicOrdering, scope: AtomicScope) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let this = NvvmAtomicFenceOp { op };
        this.set_attr_nvvm_fence_ordering(ctx, ordering);
        this.set_attr_nvvm_fence_scope(ctx, scope);
        this
    }

    /// Get the fence ordering.
    pub fn ordering(&self, ctx: &Context) -> AtomicOrdering {
        self.get_attr_nvvm_fence_ordering(ctx)
            .expect("NvvmAtomicFenceOp missing ordering")
            .clone()
    }

    /// Get the fence scope.
    pub fn scope(&self, ctx: &Context) -> AtomicScope {
        self.get_attr_nvvm_fence_scope(ctx)
            .expect("NvvmAtomicFenceOp missing scope")
            .clone()
    }
}

impl Verify for NvvmAtomicFenceOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 0 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "nvvm.atomic_fence requires no operands and no results"
            );
        }

        let Some(ordering) = self.get_attr_nvvm_fence_ordering(ctx) else {
            return verify_err!(op.loc(), "nvvm.atomic_fence requires an ordering");
        };
        if matches!(&*ordering, AtomicOrdering::Relaxed) {
            return verify_err!(
                op.loc(),
                "nvvm.atomic_fence does not support Relaxed ordering"
            );
        }
        if self.get_attr_nvvm_fence_scope(ctx).is_none() {
            return verify_err!(op.loc(), "nvvm.atomic_fence requires a scope");
        }
        Ok(())
    }
}

// =============================================================================
// NvvmAtomicRmwOp
// =============================================================================

/// Atomic read-modify-write on GPU memory.
///
/// Returns the **previous** value at the memory location.
///
/// # Operands
///
/// - `ptr`: pointer to the target
/// - `val`: value to combine with the target
///
/// # Results
///
/// - the old value before the operation
///
/// # Attributes
///
/// - `rmw_kind`: `Add`, `Sub`, `And`, `Or`, `Xor`, `Xchg`, etc.
/// - `ordering`: `Relaxed`, `Acquire`, `Release`, `AcqRel`, or `SeqCst`
/// - `scope`: `Device`, `Block`, or `System`
#[pliron_op(
    name = "nvvm.atomic_rmw",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface],
    attributes = (
        nvvm_rmw_ordering: AtomicOrdering,
        nvvm_rmw_scope: AtomicScope,
        nvvm_rmw_kind: AtomicRmwKind
    )
)]
pub struct NvvmAtomicRmwOp;

impl NvvmAtomicRmwOp {
    /// Create a new atomic RMW op wrapping an existing operation.
    pub fn new(op: Ptr<Operation>) -> Self {
        NvvmAtomicRmwOp { op }
    }

    /// Create a new atomic RMW from scratch.
    pub fn build(
        ctx: &mut Context,
        ptr: Value,
        val: Value,
        result_ty: pliron::r#type::TypeHandle,
        rmw_kind: AtomicRmwKind,
        ordering: AtomicOrdering,
        scope: AtomicScope,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty],
            vec![ptr, val],
            vec![],
            0,
        );
        let this = NvvmAtomicRmwOp { op };
        this.set_attr_nvvm_rmw_kind(ctx, rmw_kind);
        this.set_attr_nvvm_rmw_ordering(ctx, ordering);
        this.set_attr_nvvm_rmw_scope(ctx, scope);
        this
    }

    /// Get the RMW operation kind.
    pub fn rmw_kind(&self, ctx: &Context) -> AtomicRmwKind {
        self.get_attr_nvvm_rmw_kind(ctx)
            .expect("NvvmAtomicRmwOp missing rmw_kind attribute")
            .clone()
    }

    /// Get the pointer operand.
    pub fn ptr_opd(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Get the value operand.
    pub fn val_opd(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

impl Verify for NvvmAtomicRmwOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "nvvm.atomic_rmw requires two operands and one result"
            );
        }
        verify_atomic_pointer(ctx, &op, 0, "nvvm.atomic_rmw")?;
        let value_ty = op.get_operand(1).get_type(ctx);
        verify_atomic_value_type(ctx, &op, value_ty, "nvvm.atomic_rmw")?;
        if op.get_result(0).get_type(ctx) != value_ty {
            return verify_err!(
                op.loc(),
                "nvvm.atomic_rmw value and result types must match"
            );
        }
        let Some(kind) = self.get_attr_nvvm_rmw_kind(ctx) else {
            return verify_err!(op.loc(), "nvvm.atomic_rmw requires a kind");
        };
        if !verify_rmw_kind(ctx, value_ty, &kind) {
            return verify_err!(
                op.loc(),
                "nvvm.atomic_rmw kind does not support its value type"
            );
        }
        if self.get_attr_nvvm_rmw_ordering(ctx).is_none() {
            return verify_err!(op.loc(), "nvvm.atomic_rmw requires an ordering");
        }
        if self.get_attr_nvvm_rmw_scope(ctx).is_none() {
            return verify_err!(op.loc(), "nvvm.atomic_rmw requires a scope");
        }
        Ok(())
    }
}

#[op_interface_impl]
impl NvvmAtomicOpInterface for NvvmAtomicRmwOp {
    fn ordering(&self, ctx: &Context) -> AtomicOrdering {
        self.get_attr_nvvm_rmw_ordering(ctx)
            .expect("NvvmAtomicRmwOp missing ordering attribute")
            .clone()
    }

    fn scope(&self, ctx: &Context) -> AtomicScope {
        self.get_attr_nvvm_rmw_scope(ctx)
            .expect("NvvmAtomicRmwOp missing scope attribute")
            .clone()
    }

    fn ptr_operand(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

// =============================================================================
// NvvmAtomicCmpxchgOp
// =============================================================================

/// Atomic compare-and-exchange on GPU memory.
///
/// If `*ptr == cmp`, stores `new` and returns the old value. Otherwise
/// returns the current value without modifying memory.
///
/// # Operands
///
/// - `ptr`: pointer to the target
/// - `cmp`: expected value
/// - `new`: value to store if comparison succeeds
///
/// # Results
///
/// - the old value at `*ptr`
///
/// # Attributes
///
/// - `success_ordering`: ordering if comparison succeeds
/// - `failure_ordering`: ordering if comparison fails
/// - `scope`: `Device`, `Block`, or `System`
#[pliron_op(
    name = "nvvm.atomic_cmpxchg",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>, OneResultInterface],
    attributes = (
        nvvm_cas_success_ordering: AtomicOrdering,
        nvvm_cas_failure_ordering: AtomicOrdering,
        nvvm_cas_scope: AtomicScope
    )
)]
pub struct NvvmAtomicCmpxchgOp;

impl NvvmAtomicCmpxchgOp {
    /// Create a new cmpxchg op wrapping an existing operation.
    pub fn new(op: Ptr<Operation>) -> Self {
        NvvmAtomicCmpxchgOp { op }
    }

    /// Create a new cmpxchg from scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        ctx: &mut Context,
        ptr: Value,
        cmp: Value,
        new: Value,
        result_ty: pliron::r#type::TypeHandle,
        success_ordering: AtomicOrdering,
        failure_ordering: AtomicOrdering,
        scope: AtomicScope,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty],
            vec![ptr, cmp, new],
            vec![],
            0,
        );
        let this = NvvmAtomicCmpxchgOp { op };
        this.set_attr_nvvm_cas_success_ordering(ctx, success_ordering);
        this.set_attr_nvvm_cas_failure_ordering(ctx, failure_ordering);
        this.set_attr_nvvm_cas_scope(ctx, scope);
        this
    }

    /// Get the success ordering.
    pub fn success_ordering(&self, ctx: &Context) -> AtomicOrdering {
        self.get_attr_nvvm_cas_success_ordering(ctx)
            .expect("NvvmAtomicCmpxchgOp missing success_ordering")
            .clone()
    }

    /// Get the failure ordering.
    pub fn failure_ordering(&self, ctx: &Context) -> AtomicOrdering {
        self.get_attr_nvvm_cas_failure_ordering(ctx)
            .expect("NvvmAtomicCmpxchgOp missing failure_ordering")
            .clone()
    }

    /// Get the pointer operand.
    pub fn ptr_opd(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Get the comparison operand.
    pub fn cmp_opd(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    /// Get the new value operand.
    pub fn new_opd(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }
}

impl Verify for NvvmAtomicCmpxchgOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 3 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "nvvm.atomic_cmpxchg requires three operands and one result"
            );
        }
        verify_atomic_pointer(ctx, &op, 0, "nvvm.atomic_cmpxchg")?;
        let value_ty = op.get_operand(1).get_type(ctx);
        verify_atomic_value_type(ctx, &op, value_ty, "nvvm.atomic_cmpxchg")?;
        if !is_integer_atomic_type(ctx, value_ty) && !is_pointer_atomic_type(ctx, value_ty) {
            return verify_err!(
                op.loc(),
                "nvvm.atomic_cmpxchg supports only 32-bit or 64-bit integers or generic pointers"
            );
        }
        if op.get_operand(2).get_type(ctx) != value_ty || op.get_result(0).get_type(ctx) != value_ty
        {
            return verify_err!(
                op.loc(),
                "nvvm.atomic_cmpxchg compare, new value, and result types must match"
            );
        }
        if self.get_attr_nvvm_cas_success_ordering(ctx).is_none() {
            return verify_err!(op.loc(), "nvvm.atomic_cmpxchg requires a success ordering");
        }
        let Some(failure) = self.get_attr_nvvm_cas_failure_ordering(ctx) else {
            return verify_err!(op.loc(), "nvvm.atomic_cmpxchg requires a failure ordering");
        };
        if matches!(&*failure, AtomicOrdering::Release | AtomicOrdering::AcqRel) {
            return verify_err!(
                op.loc(),
                "nvvm.atomic_cmpxchg failure ordering cannot be release or acq_rel"
            );
        }
        if self.get_attr_nvvm_cas_scope(ctx).is_none() {
            return verify_err!(op.loc(), "nvvm.atomic_cmpxchg requires a scope");
        }
        Ok(())
    }
}

#[op_interface_impl]
impl NvvmAtomicOpInterface for NvvmAtomicCmpxchgOp {
    /// For cmpxchg, `ordering()` returns the **success** ordering.
    fn ordering(&self, ctx: &Context) -> AtomicOrdering {
        self.success_ordering(ctx)
    }

    fn scope(&self, ctx: &Context) -> AtomicScope {
        self.get_attr_nvvm_cas_scope(ctx)
            .expect("NvvmAtomicCmpxchgOp missing scope attribute")
            .clone()
    }

    fn ptr_operand(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

// =============================================================================
// Registration
// =============================================================================

/// Register all atomic operations and attributes with the context.
pub fn register(ctx: &mut Context) {
    // Register attributes
    AtomicOrdering::register(ctx);
    AtomicScope::register(ctx);
    AtomicRmwKind::register(ctx);

    // Register ops
    NvvmAtomicLoadOp::register(ctx);
    NvvmAtomicStoreOp::register(ctx);
    NvvmAtomicFenceOp::register(ctx);
    NvvmAtomicRmwOp::register(ctx);
    NvvmAtomicCmpxchgOp::register(ctx);
}
